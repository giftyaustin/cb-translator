import socketio
import asyncio
import os
import tempfile
import subprocess
import threading
import time
from pydub import AudioSegment
from pydub.playback import play
import time



#byte_resltime_stream_code_seamlessm4t

import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from seamless_communication.inference import Translator

# --- Global Initializations ---
# For efficiency, initialize the model only once.
MODEL_NAME = "seamlessM4T_v2_large"
VOCODER_NAME = "vocoder_v2" if MODEL_NAME == "seamlessM4T_v2_large" else "vocoder_36langs"

# Check for CUDA availability and set the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize the Translator object
try:
    translator = Translator(
        MODEL_NAME,
        VOCODER_NAME,
        device=DEVICE,
        dtype=DTYPE,
    )
    print(f"Translator initialized on device: {DEVICE}")
except Exception as e:
    print(f"Error initializing translator: {e}")
    translator = None


#raw_bytes_to_audio_translate

def translate_audio(raw_bytes: bytes, sample_width: int, frame_rate: int, channels: int, tgt_lang: str):
    """
    Translates raw audio bytes to a target language.

    Args:
        raw_bytes (bytes): The raw PCM audio data.
        sample_width (int): The sample width in bytes (e.g., 2 for 16-bit).
        frame_rate (int): The sample rate of the input audio (e.g., 44100).
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
        tgt_lang (str): The target language code (e.g., "spa" for Spanish, "fra" for French).

    Returns:
        tuple[torch.Tensor, int] | tuple[None, None]: A tuple containing the translated
        audio waveform as a torch.Tensor and its sample rate as an integer.
        Returns (None, None) if translation fails.
    """
    if not translator:
        print("Translator is not initialized. Cannot proceed with translation.")
        return None, None

    # --- 1. Pre-process Raw Bytes into a Torch Tensor ---
    audio_segment = AudioSegment(
        data=raw_bytes,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels
    )
    segment_filename = f"segment_{time.time()}.wav"
    #audio_segment.export(segment_filename, format="wav")
    samples = np.array(audio_segment.get_array_of_samples())
    if channels == 2:
        samples = samples.reshape((-1, 2))
    samples = samples.astype(np.float32) / 32768.0
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    waveform = torch.from_numpy(samples).unsqueeze(0).to(DEVICE)

    # --- 2. Resample Audio to the Model's Required Sample Rate (16kHz) ---
    model_sr = 16000
    if frame_rate != model_sr:
        print(f"Resampling audio from {frame_rate}Hz to {model_sr}Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=frame_rate, new_freq=model_sr).to(DEVICE)
        waveform = resampler(waveform)

    # --- 3. Translate the Audio ---
    print(f"Translating to {tgt_lang}...")
    try:
        text_output, speech_output = translator.predict(
            input=waveform,
            task_str="s2st",  # Speech-to-Speech Translation
            tgt_lang=tgt_lang,
        )
        translated_wav = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
        translated_sr = speech_output.sample_rate

        print(f"Translated text (this is a transcription of the input tone): {text_output[0]}")
        return translated_wav, translated_sr

    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None, None


#realtime_streaming_testing_seamless streamingimport io
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
PADDING_DURATION = 1  # Short padding for smoother output

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
        """Translate one raw PCM audio chunk — stateless"""
        audio_np = self._bytes_to_audio(audio_bytes, input_sample_rate, sample_width, channels)
        if audio_np is None or len(audio_np) < 320:
            return None, []

        segment = SpeechSegment(
            content=audio_np,
            sample_rate=SAMPLE_RATE,
            finished=True  # Important: force output immediately
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
    def save_audio(audio_np, path=f"output_{time.time()}.wav"):
        sf.write(path, audio_np, SAMPLE_RATE)


translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output

#existing code

def write_sdp_file(payload_type, codec_name, clock_rate, channels, rtp_port):
    sdp_content = f"""v=0
o=- 0 0 IN IP4 127.0.0.1
s=Mediasoup Audio
c=IN IP4 127.0.0.1
t=0 0
m=audio {rtp_port} RTP/AVP {payload_type}
a=rtpmap:{payload_type} {codec_name}/{clock_rate}/{channels}
a=recvonly
""".strip()

    tmp_dir = tempfile.gettempdir()
    sdp_path = os.path.join(tmp_dir, f"audio_{int(os.getpid())}.sdp")

    with open(sdp_path, "w") as f:
        f.write(sdp_content)

    return sdp_path

def run_ffmpeg(sdp_path):
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "info",
            "-protocol_whitelist", "file,udp,rtp",
            "-f", "sdp",
            "-i", sdp_path,
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "wav",
            "pipe:1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return ffmpeg_proc

def print_ffmpeg_logs(ffmpeg_proc):
    for line in iter(ffmpeg_proc.stderr.readline, b''):
        print("FFmpeg:", line.decode().strip())

sio = socketio.AsyncClient(
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1,
    reconnection_delay_max=5
)

@sio.event
async def connect():
    print('Connected to server')

@sio.event
async def disconnect():
    print('Disconnected from server')

@sio.on('translation:initiate')
async def message(data):
    print('Message received:', data)
    sdp_path = write_sdp_file(
        payload_type=data["payloadType"],
        codec_name=data["codec"],
        clock_rate=data["clockRate"],
        channels=data["channels"],
        rtp_port=data["rtpPort"]
    )
    ffmpeg_proc = run_ffmpeg(sdp_path)

    threading.Thread(target=print_ffmpeg_logs, args=(ffmpeg_proc,), daemon=True).start()

    chunk_size =   5 #seconds
    SEGMENT_SIZE = 48000 * 2 * 2 * chunk_size  # 5 sec * 48000 Hz * 2 bytes/sample * 2 channels
    buffer = b""
    segment_count = 0

    while True:
        chunk = ffmpeg_proc.stdout.read(4096)
        if not chunk:
            break

        buffer += chunk
        while len(buffer) >= SEGMENT_SIZE:
            segment = buffer[:SEGMENT_SIZE]
            #print(segment)
            buffer = buffer[SEGMENT_SIZE:]
            print("🔊 Received 5-second audio segment")


            #seamelessm4T
            sample_rate = 48000
            start_time = time.time()
            translated_wav, translated_sr = translate_audio(segment, 2,sample_rate,2, "hin")
            end_time = time.time()
            print(f"Inference time: {end_time-start_time: .4f} sec.")
            print(translated_sr)
            out_file = f"translated_raw_{time.time()}.wav"
            #torchaudio.save(out_file, translated_wav, 16000)
            #translated_segment = AudioSegment.from_wav(out_file)
            #play(translated_segment)
            translated_wav = translated_wav.squeeze().cpu().numpy()
            print(translated_wav)


            # #seamless_streaming
            # # Example usage
            
            # # Translate
            # #print(segment.shape)
            # #in_file = "/path/to/stereo_48k.wav"  # 48000 Hz, stereo file
            # #original_segment = AudioSegment.from_wav(in_file)

            # #audio_bytes = original_segment.raw_data
            # sample_width = 2
            # frame_rate = 48000
            # channels = 2

            # #print(f"Sample width: {sample_width}, Frame rate: {frame_rate}, Channels: {channels}")
            # start_time = time.time()
            # translated_wav, text = translator1.translate_chunk(
            #     segment,
            #     input_sample_rate=frame_rate,
            #     sample_width=sample_width,
            #     channels=channels
            #     )
            # end_time = time.time()
            # print(translated_wav, text)
            # print(f"Inference time: {end_time-start_time: .4f} sec.")
            # if translated_wav is not None:
            #     translator1.play_audio(translated_wav)
            #     translator1.save_audio(translated_wav)
            # if text:
            #     print("📝", text)

            # numpy array to bytes
            # 1. Assume this is your audio in float32 format (range -1.0 to 1.0)
            audio_np = np.array(translated_wav, dtype=np.float32)
            # 2. Clip to [-1, 1] just in case
            audio_np = np.clip(audio_np, -1.0, 1.0)
            # 3. Convert to int16 format (PCM 16-bit)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            # 4. Convert to raw PCM bytes
            translated_audio_bytes = audio_int16.tobytes()


            # sample_width = 2
            # frame_rate = 16000
            # channels = 1
            # # DO NOT hardcode the wrong sample rate
            # audio_segment = AudioSegment(
            #     data=translated_audio_bytes,
            #     sample_width=sample_width,
            #     frame_rate=frame_rate,
            #     channels=channels
            # )
            # segment_filename = f"segment_raw.wav"
            # audio_segment.export(segment_filename, format="wav")

            segment_count += 1
            
    ffmpeg_proc.stdout.close()

async def main():
    await sio.connect('http://localhost:3000')
    await sio.wait()

asyncio.run(main())
