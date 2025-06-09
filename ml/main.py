import socketio
import asyncio
import os
import tempfile
import subprocess
import threading

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


# def start_ffmpeg_output_process(target_ip, target_port):
#     return subprocess.Popen(
#         [
#             "ffmpeg",
#             "-f", "s16le",         # raw PCM
#             "-ar", "48000",        # sample rate
#             "-ac", "2",            # stereo
#             "-i", "pipe:0",        # read from stdin
#             "-c:a", "libopus",     # or pcm_mulaw if Mediasoup expects mulaw
#             "-f", "rtp",
#             f"rtp://{target_ip}:{target_port}"
#         ],
#         stdin=subprocess.PIPE,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.PIPE
#     )


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
            "pipe:1"  # or output.wav if you want to store
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return ffmpeg_proc


def print_ffmpeg_logs(ffmpeg_proc):
    for line in iter(ffmpeg_proc.stderr.readline, b''):
        print("FFmpeg:", line.decode().strip())

# Async client
sio = socketio.AsyncClient(
    reconnection=True,           # Enable auto-reconnect (default: True)
    reconnection_attempts=5,     # Number of attempts before giving up (default: Infinity)
    reconnection_delay=1,        # Seconds between attempts (default: 1)
    reconnection_delay_max=5     # Max delay cap (default: 5)
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
    
    # ffmpeg_send = start_ffmpeg_output_process(
    #     target_ip="127.0.0.1",
    #     target_port=data["listenRtpPort"]
    # )

    # ffmpeg_send.stdin.write(ffmpeg_proc.stdout.read())
    

    # Start logging FFmpeg output
    threading.Thread(target=print_ffmpeg_logs, args=(ffmpeg_proc,), daemon=True).start()

    SEGMENT_SIZE = 48000 * 2 * 2 * 5
    buffer = b""

    while True:
        chunk = ffmpeg_proc.stdout.read(4096)
        if not chunk:
            break

        buffer += chunk
        while len(buffer) >= SEGMENT_SIZE:
            segment = buffer[:SEGMENT_SIZE]
            buffer = buffer[SEGMENT_SIZE:]
            print(segment)
            print(buffer)
            print("🔊 Received 5-second audio segment")

    ffmpeg_proc.stdout.close()


async def main():
    await sio.connect('http://localhost:3000')
    await sio.wait()

asyncio.run(main())
