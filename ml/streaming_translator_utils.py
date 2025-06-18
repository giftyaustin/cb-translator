# streaming_translator_utils.py

import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
from simuleval.data.segments import SpeechSegment, TextSegment
from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STJointVADAgent
from simuleval.utils.arguments import cli_argument_list
from simuleval import options
from pydub import AudioSegment
from pydub.playback import play

SAMPLE_RATE = 16000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PADDING_DURATION = 1  # Seconds


class StatelessBytesTranslator:
    def __init__(self, tgt_lang="hin", src_lang=None):
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        self.system = self._build_system()

    def _build_system(self):
        model_configs = {
            "source_segment_size": -1,
            "device": DEVICE,
            "dtype": "fp16",
            "min_unit_chunk_size": 320,
            "task": "s2st",
            "tgt_lang": self.tgt_lang,
            "detokenize_only": False,
            "force_finish": True,
        }
        parser = options.general_parser()
        parser.add_argument("-f", "--f", help="dummy argument", default="1")
        SeamlessStreamingS2STJointVADAgent.add_args(parser)
        args, _ = parser.parse_known_args(cli_argument_list(model_configs))
        return SeamlessStreamingS2STJointVADAgent.from_args(args)

    def _bytes_to_audio(self, audio_bytes, input_sample_rate, sample_width, channels):
        try:
            dtype = '<i2' if sample_width == 2 else '<i1'
            audio = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
            max_val = 32768.0 if sample_width == 2 else 128.0
            audio /= max_val

            if channels > 1:
                audio = audio.reshape(-1, channels)
                audio = np.mean(audio, axis=1)

            if input_sample_rate != SAMPLE_RATE:
                audio = torchaudio.functional.resample(torch.tensor(audio), input_sample_rate, SAMPLE_RATE).numpy()

            pad = int(SAMPLE_RATE * PADDING_DURATION)
            return np.concatenate([audio, np.zeros(pad)])
        except Exception as e:
            print(f"❌ Audio decode failed: {e}")
            return None

    def translate_chunk(self, audio_bytes, input_sample_rate=48000, sample_width=2, channels=2):
        audio_np = self._bytes_to_audio(audio_bytes, input_sample_rate, sample_width, channels)
        if audio_np is None or len(audio_np) < 320:
            return None, []

        segment = SpeechSegment(
            content=audio_np,
            sample_rate=SAMPLE_RATE,
            finished=True
        )
        segment.tgt_lang = self.tgt_lang
        if self.src_lang:
            segment.src_lang = self.src_lang

        system_states = self.system.build_states()
        segments = self.system.pushpop(segment, system_states)

        audio_out, text_out = [], []
        for seg in segments:
            if isinstance(seg, SpeechSegment):
                audio_out.extend(seg.content)
            elif isinstance(seg, TextSegment):
                text_out.append(seg.content)

        audio_out = np.array(audio_out, dtype=np.float32) if audio_out else None
        return audio_out, text_out

    @staticmethod
    def play_audio(audio_np):
        audio_segment = AudioSegment(
            (audio_np * 32767).astype(np.int16).tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        play(audio_segment)

    @staticmethod
    def save_audio(audio_np, path=None):
        if path is None:
            path = f"output_streaming.wav"
        sf.write(path, audio_np, SAMPLE_RATE)
