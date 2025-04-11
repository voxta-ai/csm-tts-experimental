import os
import io
import statistics
import argparse
import time
import logging
import wave
import winsound
import torch
import torchaudio
from generator import load_csm_1b, Segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CSMTest:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--csm-model-path", required=True)
        parser.add_argument("--mimi-model-path", required=True)
        parser.add_argument("--llama-model-path", required=True)
        parser.add_argument("--num-codebooks", default=32, type=int)
        parser.add_argument("--output-path", required=True)
        parser.add_argument("--warmup-runs", default=1, type=int)
        parser.add_argument("--runs", default=5, type=int)
        parser.add_argument("--play-test", default=False, action="store_true")
        parser.add_argument("--text", default="Hello, I am a conversational speech model made by Sesame. I'm saying something a bit long so we reach the total amount of tokens and we're sure to be cut off.")
        parser.add_argument("--text2", default=None, type=str)
        parser.add_argument("--max-duration", default=30, type=int)
        parser.add_argument("--voice-wav", default=None, type=str)
        parser.add_argument("--voice-transcript", default=None, type=str)
        args = parser.parse_args()

        self.generator = load_csm_1b(
            csm_model_path=args.csm_model_path,
            llama_model_path=args.llama_model_path,
            mimi_model_path=args.mimi_model_path,
            num_codebooks=args.num_codebooks,
        )

        self.output_path = args.output_path
        self.warmup_runs = args.warmup_runs
        self.runs = args.runs
        self.play_test = args.play_test

        self.text = args.text
        self.text2 = args.text2
        self.max_duration = args.max_duration

        voice_audio =self.load_audio(args.voice_wav)
        initial_segment = self.generator.create_segment(
            speaker=0,
            text=args.voice_transcript,
            audio=voice_audio,
        )

        self.context = [initial_segment]

        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        id = 1

        if self.warmup_runs > 0:
            for i in range(self.warmup_runs):
                bytes1 = self.run_one()
                bytes2 = self.run_two()
                if i == 0:
                    self.save_audio(bytes1, 0, 1)
                    self.save_audio(bytes2, 0, 2)

        if self.runs > 0:
            timings = []    
            for i in range(self.runs):
                id += 1
                st = time.monotonic()
                bytes1 = self.run_one()
                bytes2 = self.run_two()
                timings.append(time.monotonic() - st)
                self.save_audio(bytes1, i, 1)
                self.save_audio(bytes2, i, 2)
            
            # Calculate statistics
            mean_time = statistics.mean(timings)
            median_time = statistics.median(timings)
            min_time = min(timings)
            max_time = max(timings)
            logger.info(f"Mean time: {mean_time:.3f} seconds")
            logger.info(f"Median time: {median_time:.3f} seconds")
            logger.info(f"Min time: {min_time:.3f} seconds")
            logger.info(f"Max time: {max_time:.3f} seconds")
    
    def run_one(self) -> bytes:
        buffer = io.BytesIO()
        for chunk in self.generator.generate_stream(
            speaker=0,
            text=self.text,
            context=self.context,
            max_audio_length_ms=self.max_duration * 1000,
        ):
            buffer.write((chunk.numpy() * 32767).astype("int16").tobytes())
        return buffer.getvalue()
    
    def run_two(self) -> bytes:
        if self.text2 is None:
            return None
        context = [
            self.context[0],
            self.generator.create_segment(
                speaker=1,
                text="I'm good, thanks!",
                audio=None
            ),
            self.generator.get_segment(
                speaker=0,
                text=self.text,
            )
        ]
        buffer = io.BytesIO()
        for chunk in self.generator.generate_stream(
            speaker=0,
            text=self.text2,
            context=context,
            max_audio_length_ms=self.max_duration * 1000,
        ):
            buffer.write((chunk.numpy() * 32767).astype("int16").tobytes())
        return buffer.getvalue()
        
    def save_audio(self, bytes, i, turn=0):
        if bytes is None:
            return
        tmp_file = os.path.join(self.output_path, "csm-" + str(i) + "-" + str(turn) + ".wav")
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(bytes)

        with open(tmp_file, "wb") as f:
            f.write(wav_buffer.getvalue())
        if self.play_test:
            winsound.PlaySound(tmp_file, winsound.SND_FILENAME)  

    def load_audio(self, audio_path: str) -> torch.Tensor:
        if(audio_path == None or audio_path == ""):
            return None
        audio_tensor, sample_rate = torchaudio.load(audio_path)

        # Downmix to mono
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        # Resample to the model's sample rate
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=self.generator.sample_rate
        )

        return audio_tensor

if __name__ == "__main__":
    CSMTest().run()