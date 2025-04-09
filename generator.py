from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator
import logging
import time
from collections import OrderedDict

import torch
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor = None
    audio_path: str = ""

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
        self._segment_cache = OrderedDict()
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer(llama_model_path)

        device = next(model.parameters()).device
        mimi = loaders.get_mimi(mimi_model_path, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        key = (segment.audio_path, segment.speaker, segment.text)
        if key in self._segment_cache:
            # Move to end for LRU behavior
            tokens_masks = self._segment_cache.pop(key)
            self._segment_cache[key] = tokens_masks
            logger.debug(f"Loaded segment from cache")
            return tokens_masks

        st = time.time()
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        out_tokens = torch.cat([text_tokens, audio_tokens], dim=0)
        out_masks = torch.cat([text_masks, audio_masks], dim=0)

        # Cache and enforce a limit of 8
        if len(self._segment_cache) >= 8:
            self._segment_cache.popitem(last=False)
        self._segment_cache[key] = (out_tokens, out_masks)
        logger.debug(f"Tokenized segment ({time.time() - st:.2f}s)")

        return out_tokens, out_masks

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

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

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
        counter = 0
        batch_size = 10
        buffer_size = 30
        frame_buffer = []
        with self._audio_tokenizer.streaming(batch_size=1):
            i = 0
            while i < max_generation_len:
                batch_end = min(i + batch_size, max_generation_len)
                batch_size_actual = batch_end - i

                batch_samples = []

                for _ in range(batch_size_actual):
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)

                    if torch.all(sample == 0):
                        break

                    counter += 1
                    batch_samples.append(sample)
                    update_tokens(sample)

                if not batch_samples:
                    break

                frame_buffer.extend(batch_samples)
                i += len(batch_samples)

                if len(frame_buffer) >= buffer_size:
                    frames_stacked = torch.stack(frame_buffer).permute(1, 2, 0)
                    audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                    frame_buffer = []
                    yield audio_chunk.cpu()
                    if not first_chunk:
                        first_chunk = True
                        logger.info("First chunk: %.2f seconds", time.monotonic() - st)

                # Why?
                if i >= 100 and (i % 100 == 0):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

        if frame_buffer:
            frames_stacked = torch.stack(frame_buffer).permute(1, 2, 0)
            audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
            cpu_chunk = audio_chunk.cpu()
            yield cpu_chunk

        print(f"Total time: {time.monotonic() - st:.2f} seconds ({counter} samples)")

def load_csm_1b(
        csm_model_path: str,
        llama_model_path: str,
        mimi_model_path: str,
        device: str = "cuda",
        ) -> Generator:
    logger.info("Loading CSM 1B model from %s", csm_model_path)

    model = Model.from_pretrained(csm_model_path, local_files_only=True)

    # model = torch.compile(
    #     model,
    #     dynamic=True,
    #     fullgraph=True,
    #     backend='cudagraphs'
    # )

    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model, llama_model_path, mimi_model_path)
    return generator
