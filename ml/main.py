import socketio
import asyncio
import os
import tempfile
import subprocess
import threading
import wave 
import time





def save_to_wav(audio_bytes: bytes, sample_rate=48000, num_channels=2, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/output_{int(time.time() * 1000)}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    print(f"💾 Saved audio segment to {filename}")

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

def run_ffmpeg_input(sdp_path):
    return subprocess.Popen(
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

def run_ffmpeg_output(target_ip, target_port):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-f", "s16le",
            "-ar", "48000",
            "-ac", "2",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-payload_type", "100",
            "-f", "rtp",
            f"rtp://{target_ip}:{target_port}"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

def print_ffmpeg_logs(proc, label):
    for line in iter(proc.stderr.readline, b''):
        print(f"{label}: {line.decode().strip()}")

# Async client
sio = socketio.AsyncClient(
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1,
    reconnection_delay_max=5
)

@sio.event
async def connect():
    print('✅ Connected to server')

@sio.event
async def disconnect():
    print('❌ Disconnected from server')

@sio.on('translation:initiate')
async def message(data):
    print('📥 Received translation initiation:', data)
    sdp_path = write_sdp_file(
        payload_type=data["payloadType"],
        codec_name=data["codec"],
        clock_rate=data["clockRate"],
        channels=data["channels"],
        rtp_port=data["rtpPort"]
    )

    ffmpeg_in = run_ffmpeg_input(sdp_path)
    ffmpeg_out = run_ffmpeg_output("127.0.0.1", 26000)

    threading.Thread(target=print_ffmpeg_logs, args=(ffmpeg_in, "FFmpeg-IN"), daemon=True).start()
    threading.Thread(target=print_ffmpeg_logs, args=(ffmpeg_out, "FFmpeg-OUT"), daemon=True).start()

    SEGMENT_SIZE = 48000 * 2 * 2 * 5  # 5 seconds of stereo s16le @ 48kHz
    buffer = b""

    while True:
        chunk = ffmpeg_in.stdout.read(4096)
        if not chunk:
            break

        buffer += chunk
        while len(buffer) >= SEGMENT_SIZE:
            segment = buffer[:SEGMENT_SIZE]
            buffer = buffer[SEGMENT_SIZE:]
            
            save_to_wav(segment)
            
            print(f"📦 Processed 5-second audio segment of size {len(segment)} bytes")

            ffmpeg_out.stdin.write(segment)
            ffmpeg_out.stdin.flush()
            print("🔊 Sent 5-second audio segment to Mediasoup")

    ffmpeg_in.stdout.close()
    ffmpeg_out.stdin.close()
    ffmpeg_in.wait()
    ffmpeg_out.wait()

async def main():
    await sio.connect('http://localhost:3000')
    await sio.wait()

if __name__ == "__main__":
    asyncio.run(main())
