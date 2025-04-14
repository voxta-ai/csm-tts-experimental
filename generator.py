from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator
import logging
import time
from collections import OrderedDict

import torch
import numpy as np
from safetensors.torch import load_file
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    speaker: int
    text: str
    tokens: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None

def load_llama3_tokenizer(llama_model_path: str):
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path, local_files_only=True)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer

class Generator:
    def __init__(
        self,
        model: Model,
        llama_model_path: str,
        mimi_model_path: str,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer(llama_model_path)

        device = next(model.parameters()).device
        mimi = loaders.get_mimi(mimi_model_path, device=device)
        mimi.set_num_codebooks(model.config.audio_num_codebooks)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

        self._cache = OrderedDict()

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._audio_tokenizer.num_codebooks + 1).long()
        text_frame_mask = torch.zeros(len(text_tokens), self._audio_tokenizer.num_codebooks + 1).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(
            audio.unsqueeze(0).unsqueeze(0)
        )[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._audio_tokenizer.num_codebooks + 1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._audio_tokenizer.num_codebooks + 1).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, speaker: int, text: str, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        st = time.time()
        text_tokens, text_masks = self._tokenize_text_segment(text, speaker)
        audio_tokens, audio_masks = self._tokenize_audio(audio)

        out_tokens = torch.cat([text_tokens, audio_tokens], dim=0)
        out_masks = torch.cat([text_masks, audio_masks], dim=0)

        logger.debug(f"Tokenized segment ({time.time() - st:.2f}s)")

        return out_tokens, out_masks
    
    def get_segment(self, speaker: int, text: str) -> Optional[Segment]:
        key = (speaker, text)
        if key in self._cache:
            # Move to end for LRU behavior
            segment = self._cache.pop(key)
            self._cache[key] = segment
            logger.debug(f"Loaded segment from cache")
            return segment
        return None
    
    def create_segment(self, speaker: int, text: str, audio: Optional[torch.Tensor]) -> Segment:
        if audio is None:
            audio = torch.zeros(7, device=self.device)
        tokens, masks = self._tokenize_segment(speaker, text, audio)
        segment = Segment(speaker=speaker, text=text, tokens=tokens, masks=masks)
        if len(self._cache) >= 8:
            self._cache.popitem(last=False)
        self._cache[(speaker, text)] = segment
        return segment

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
    ) -> Iterator[torch.Tensor]:
        self._model.reset_caches()
        
        counter = 0
        batch_size = 10
        buffer_size = 20 # 1.6s
        skip_silence_duration = 1.6
        stop_silence_duration = 4.8
        silence_duration_counter = 0
        silence_threshold = 0.05
        sample_rate = 24000

        max_generation_len = int(max_audio_length_ms / 80)
        max_generation_len = int(np.ceil(max_generation_len / buffer_size) * buffer_size)
            
        tokens, tokens_mask = [], []
        for segment in context:
            tokens.append(segment.tokens)
            tokens_mask.append(segment.masks)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        zeros_1_1 = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample).bool()
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        st = time.monotonic()
        first_chunk = False
        frame_buffer = []
        out_tokens = []
        out_masks = []
        with self._audio_tokenizer.streaming(batch_size=1):
            i = 0
            while counter < max_generation_len:
                batch_end = min(i + batch_size, max_generation_len)
                batch_size_actual = batch_end - i

                batch_samples = []

                for _ in range(batch_size_actual):
                    if counter >= max_generation_len:
                        break

                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)

                    if torch.all(sample == 0):
                        break

                    counter += 1
                    batch_samples.append(sample)
                    update_tokens(sample)

                    out_tokens.append(
                        torch.cat([
                            sample, # shape (1, num_codebooks)
                            torch.zeros(1, self._audio_tokenizer.num_codebooks + 1 - sample.shape[1], device=sample.device, dtype=torch.long)
                        ], dim=1) # shape (1, 33)
                    )
                    out_masks.append(
                        torch.cat([
                            torch.ones(1, sample.shape[1], device=sample.device, dtype=torch.bool),
                            torch.zeros(1, self._audio_tokenizer.num_codebooks + 1 - sample.shape[1], device=sample.device, dtype=torch.bool)
                        ], dim=1) # shape (1, 33)
                    )

                if not batch_samples:
                    break

                frame_buffer.extend(batch_samples)
                i += len(batch_samples)

                if len(frame_buffer) >= buffer_size:
                    chunk = frame_buffer[:buffer_size]
                    frame_buffer = frame_buffer[buffer_size:]
                        
                    # DEBUG
                    frame_shapes = [frame.shape for frame in chunk]
                    
                    frames_stacked = torch.stack(chunk)
                    frames_stacked = frames_stacked.permute(1, 2, 0)
                    expected_tokens = self._audio_tokenizer.num_codebooks  # e.g. 32
                    if frames_stacked.shape[1] != expected_tokens:
                        logger.warning("Token mismatch: got %d tokens but expected %d", frames_stacked.shape[1], expected_tokens)
                    audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)

                    # Check for silence in the decoded chunk
                    audio_np = audio_chunk.cpu().numpy()
                    chunk_duration = audio_np.shape[0] / sample_rate
                    if np.max(np.abs(audio_np)) < silence_threshold:
                        silence_duration_counter += chunk_duration
                        if silence_duration_counter > stop_silence_duration:
                            logger.warning(f"Silence detected for {silence_duration_counter}s, skipping {chunk_duration}s and stopping generation")
                            break
                        elif silence_duration_counter > skip_silence_duration:
                            logger.info(f"Silence detected for {silence_duration_counter}s, skipping {chunk_duration}s chunk")
                            pass
                        else:
                            logger.info(f"Silence detected in {chunk_duration}s chunk")
                            yield audio_chunk.cpu()
                    else:
                        silence_duration_counter = 0.0
                        yield audio_chunk.cpu()

                    if not first_chunk:
                        first_chunk = True
                        logger.debug(f"First chunk: {time.monotonic() - st:.2f} seconds, {counter} samples", )

                if i >= 100 and (i % 100 == 0):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

        if frame_buffer:
            frames_stacked = torch.stack(frame_buffer).permute(1, 2, 0)
            audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
            audio_np = audio_chunk.cpu().numpy()
            chunk_duration = audio_np.shape[0] / sample_rate
            if np.max(np.abs(audio_np)) < silence_threshold:
                silence_duration_counter += chunk_duration
                if silence_duration_counter <= skip_silence_duration:
                    yield audio_chunk.cpu()
            else:
                yield audio_chunk.cpu()

        print(f"Total time: {time.monotonic() - st:.2f} seconds ({counter} samples)")
        
        gen_audio_tokens = torch.cat(out_tokens, dim=0)
        gen_audio_masks = torch.cat(out_masks, dim=0)
        all_tokens = torch.cat([gen_segment_tokens, gen_audio_tokens], dim=0)
        all_masks = torch.cat([gen_segment_tokens_mask, gen_audio_masks], dim=0)
        generated_segment = Segment(speaker=speaker, text=text, tokens=all_tokens, masks=all_masks)
        if len(self._cache) >= 8:
            self._cache.popitem(last=False)
        self._cache[(speaker, text)] = generated_segment

def load_csm_1b(
        csm_model_path: str,
        llama_model_path: str,
        mimi_model_path: str,
        num_codebooks: int,
        device: str = "cuda",
        ) -> Generator:
    logger.debug("Loading CSM 1B model from %s", csm_model_path)
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True

    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=num_codebooks, # 32, 24, 16, 12, 8
    )
    model = Model(config=model_args).to(device=device, dtype=torch.bfloat16)
    state_dict = load_file(csm_model_path)
    
    if(num_codebooks != 32):
        state_dict['audio_head'] = state_dict['audio_head'][:model_args.audio_num_codebooks - 1]
        vocab = model_args.audio_vocab_size
        state_dict['audio_embeddings.weight'] = state_dict['audio_embeddings.weight'][:vocab * model_args.audio_num_codebooks]

    model.load_state_dict(state_dict)
    model.decoder = torch.compile(model.decoder, fullgraph=True, backend='cudagraphs')
    model.forward = torch.compile(model.forward, mode="reduce-overhead")

    generator = Generator(model, llama_model_path, mimi_model_path)
    
    logger.debug(f"MIMI encoder uses {generator._audio_tokenizer.num_codebooks} / 32 codebooks")
    
    return generator
