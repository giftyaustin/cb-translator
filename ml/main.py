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
import numpy as np


from seamlessm4t_translator_utils import translate_audio
from streaming_translator_utils import StatelessBytesTranslator

translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output

# numpy array to bytes
def tensor_to_bytes(translated_wav):
    # 1. Assume this is your audio in float32 format (range -1.0 to 1.0)
    audio_np = np.array(translated_wav, dtype=np.float32)
    # 2. Clip to [-1, 1] just in case
    audio_np = np.clip(audio_np, -1.0, 1.0)
    # 3. Convert to int16 format (PCM 16-bit)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    # 4. Convert to raw PCM bytes
    translated_audio_bytes = audio_int16.tobytes()
    return translated_audio_bytes

# translated audio bytes to wav file for verification
def bytes_to_wav(translated_audio_bytes):
    sample_width = 2
    frame_rate = 16000
    channels = 1
    # DO NOT hardcode the wrong sample rate
    audio_segment = AudioSegment(
        data=translated_audio_bytes,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels
    )
    segment_filename = f"segment_raw.wav"
    audio_segment.export(segment_filename, format="wav")
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

            which_translator = 1

            if which_translator == 1:
                #seamelessm4T
                sample_rate = 48000
                start_time = time.time()
                translated_wav, translated_sr = translate_audio(segment, sample_width=2, frame_rate = sample_rate, channels = 2, tgt_lang = "hin")
                end_time = time.time()
                print(f"Inference time: {end_time-start_time: .4f} sec.")
                print(translated_sr)
                out_file = f"translated_raw_{time.time()}.wav"
                #torchaudio.save(out_file, translated_wav, 16000)
                #translated_segment = AudioSegment.from_wav(out_file)
                #play(translated_segment)
                #translated_wav = translated_wav.squeeze().cpu().numpy()
                print(translated_wav)

            if which_translator ==2:
                #seamless_streaming
                #audio_bytes = original_segment.raw_data
                sample_width = 2
                frame_rate = 48000
                channels = 2
                #print(f"Sample width: {sample_width}, Frame rate: {frame_rate}, Channels: {channels}")
                start_time = time.time()
                translated_wav, text = translator1.translate_chunk(
                    segment,
                    input_sample_rate=frame_rate,
                    sample_width=sample_width,
                    channels=channels
                    )
                end_time = time.time()
                print(translated_wav, text)
                print(f"Inference time: {end_time-start_time: .4f} sec.")
                #if translated_wav is not None:
                #    translator1.play_audio(translated_wav)
                    #translator1.save_audio(translated_wav)
                if text:
                    print("📝", text)

            translated_audio_bytes = tensor_to_bytes(translated_wav)
            bytes_to_wav(translated_audio_bytes)
            
            segment_count += 1
            
    ffmpeg_proc.stdout.close()

async def main():
    await sio.connect('http://localhost:3000')
    await sio.wait()

asyncio.run(main())
