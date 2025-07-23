import io
import queue
import sounddevice as sd
import numpy as np
import soundfile
from pydub import AudioSegment
from pydub.playback import play

from simuleval.data.segments import SpeechSegment, EmptySegment, TextSegment
from simuleval.agents.pipeline import TreeAgentPipeline
from simuleval.agents.states import AgentStates

from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STJointVADAgent,
)
from simuleval.utils.arguments import cli_argument_list
from simuleval import options
import torchaudio
import torch
import time

SAMPLE_RATE = 16000
CHUNK_SIZE = 16000  # 1-second chunks

class OutputSegments:
    def __init__(self, segments):
        if isinstance(segments, (SpeechSegment, TextSegment, EmptySegment)):
            segments = [segments]
        self.segments = [s for s in segments]

    @property
    def is_empty(self):
        return all(getattr(segment, "is_empty", False) for segment in self.segments)

    @property
    def finished(self):
        return all(getattr(segment, "finished", False) for segment in self.segments)

def reset_states(system, states):
    if isinstance(system, TreeAgentPipeline):
        states_iter = states.values()
    else:
        states_iter = states
    for state in states_iter:
        state.reset()



def get_audio_bytes(samples, sr: int, target_sr: int = 96000, stereo: bool = False) -> bytes:
    import torchaudio
    import io

    # Convert to tensor if input is a list
    if isinstance(samples, list):
        samples = torch.tensor(samples, dtype=torch.float32)
    elif isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).float()

    if sr is None or sr <= 0 or target_sr is None or target_sr <= 0:
        raise ValueError(f"Invalid sample rate(s): sr={sr}, target_sr={target_sr}")

    # Convert to stereo if requested
    if stereo and samples.dim() == 1:
        samples = samples.unsqueeze(0).repeat(2, 1)  # [2, N]
    elif samples.dim() == 1:
        samples = samples.unsqueeze(0)  # [1, N]

    # Resample if needed
    if sr != target_sr:
        samples = torchaudio.functional.resample(samples, orig_freq=sr, new_freq=target_sr)

    # Save to WAV bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, samples.cpu(), sample_rate=target_sr, format="wav")
    buffer.seek(0)
    return buffer.read()




def play_audio1(audio_bytes):
    # Load WAV from bytes
    audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
    print(sr)

    # Convert to numpy (shape: [channels, time])
    audio_np = audio_tensor.cpu().numpy()

    # If mono: [1, N] → [N], else: keep stereo shape [2, N]
    if audio_np.shape[0] == 1:
        audio_np = audio_np.squeeze(0).T  # (N,)
    else:
        audio_np = audio_np.T  # Convert to shape (N, 2) for stereo

    # Play
    sd.play(audio_np, samplerate=sr)
    sd.wait()

def play_audio(audio_bytes, sample_rate=48000, num_channels=2, sample_width=2):
    # Determine correct dtype from sample_width
    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = dtype_map[sample_width]

    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=dtype)

    # Normalize to float32 range [-1, 1]
    if dtype == np.uint8:
        audio_np = (audio_np.astype(np.float32) - 128) / 128
    elif dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768
    elif dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648

    # Reshape if stereo
    if num_channels > 1:
        audio_np = audio_np.reshape(-1, num_channels)

    # Play
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()


def get_audiosegment(samples, sr):
    b = io.BytesIO()
    soundfile.write(b, samples, samplerate=sr, format="wav")
    b.seek(0)
    return AudioSegment.from_file(b)

def build_streaming_system(model_configs, agent_class):
    parser = options.general_parser()
    parser.add_argument("-f", "--f", help="dummy arg for IPython", default="1")

    agent_class.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(model_configs))
    system = agent_class.from_args(args)
    return system


def bytes_to_float32_mono_array(audio_bytes: bytes, input_sr=48000, target_sr=16000) -> np.ndarray:
    # 1. Decode stereo int16 bytes to numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_np = audio_np.reshape(-1, 2)  # 2 channels

    # 2. Convert to mono by averaging channels
    mono_np = audio_np.mean(axis=1)

    # 3. Convert to torch tensor and float32 [-1.0, 1.0]
    waveform = torch.tensor(mono_np, dtype=torch.float32) / 32768.0
    waveform = waveform.unsqueeze(0)  # (1, N)

    # 4. Resample to 16kHz
    resampled = torchaudio.functional.resample(waveform, orig_freq=input_sr, new_freq=target_sr)

    return resampled.squeeze(0).numpy()  # Return as 1D float32 array

def stream_translate_from_bytes(audio_bytes: bytes, system, system_states, input_sr=48000, target_sr=16000, tgt_lang="hin"):
    # Convert bytes → float32 mono array at 16kHz
    float_audio = bytes_to_float32_mono_array(audio_bytes, input_sr=input_sr, target_sr=target_sr)

    # Feed to SpeechSegment
    input_segment = SpeechSegment(content=float_audio, sample_rate=target_sr)
    input_segment.tgt_lang = tgt_lang

    output_segments = OutputSegments(system.pushpop(input_segment, system_states))

    translated_audio = b""
    translated_text = ""

    for seg in output_segments.segments:
        if isinstance(seg, SpeechSegment):
            translated_audio += get_audio_bytes(seg.content, seg.sample_rate)
        elif isinstance(seg, TextSegment):
            translated_text += seg.content + " "

    return translated_audio, translated_text.strip(), output_segments.finished


