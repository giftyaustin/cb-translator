import asyncio
import os
import socketio
import subprocess
import tempfile
import threading
import time
import wave
import uuid
import socket

def get_free_port():
    """Finds a free UDP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def save_to_wav(audio_bytes: bytes,
                sample_rate=48000,
                num_channels=2,
                sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/output_{int(time.time()*1000)}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    print(f"💾 Saved audio segment to {filename}")


def write_sdp_file(payload_type,
                   codec_name,
                   clock_rate,
                   channels,
                   rtp_port):
    """
    Generates a one‐off SDP file that tells FFmpeg to listen on
    0.0.0.0:rtp_port for an RTP/AVP stream of the given codec.
    Returns the path to a unique tempfile.
    """
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=Mediasoup Audio\n"
        "c=IN IP4 0.0.0.0\n"
        "t=0 0\n"
        f"m=audio {rtp_port} RTP/AVP {payload_type}\n"
        f"a=rtpmap:{payload_type} {codec_name}/"
        f"{clock_rate}/{channels}\n"
        "a=recvonly\n"
        "a=rtcp-mux\n"
    )
    fn = f"audio_{uuid.uuid4().hex}.sdp"
    path = os.path.join(tempfile.gettempdir(), fn)
    with open(path, "w") as f:
        f.write(sdp)
    return path


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
            "-f", "s16le",   # raw PCM
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
            f"rtp://{target_ip}:{target_port}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

def print_ffmpeg_logs(proc, label):
    for line in iter(proc.stderr.readline, b""):
        text = line.decode().strip()
        # Only print lines that contain "error" (case-insensitive)
        if "error" in text.lower():
            print(f"{label}: {text}")

def pump_audio(ff_in, ff_out, segment_size, sdp_path):
    buf = b""
    try:
        while True:
            chunk = ff_in.stdout.read(4096)
            if not chunk:
                print('stopping now')
                break
            buf += chunk
            while len(buf) >= segment_size:
                seg, buf = buf[:segment_size], buf[segment_size:]
                save_to_wav(seg)
                print(f"📦 Processed segment: {len(seg)} bytes")
                try:
                    ff_out.stdin.write(seg)
                    ff_out.stdin.flush()
                    print("🔊 Sent segment to Mediasoup")
                except BrokenPipeError:
                    print("⚠️ FFmpeg-OUT pipe closed")
                    return
    finally:
        ff_in.stdout.close()
        ff_out.stdin.close()
        ff_in.wait()
        ff_out.wait()
        try:
            os.remove(sdp_path)
        except OSError:
            pass


sio = socketio.AsyncClient(
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1,
    reconnection_delay_max=5,
)


@sio.event
async def connect():
    print("✅ Connected to server")


@sio.event
async def disconnect():
    print("❌ Disconnected from server")


@sio.on("translation:initiate")
async def on_translation_initiate(data):
    print("📥 Received translation initiation:", data)

    # Use the rtpPort provided by the client
    sdp_path = write_sdp_file(
        payload_type=data["payloadType"],
        codec_name=data["codec"],
        clock_rate=data["clockRate"],
        channels=data["channels"],
        rtp_port=data["rtpPort"],  # Trusting the client's provided port (and the port next to it) is unique
    )

    ff_in = run_ffmpeg_input(sdp_path)
    ff_out = run_ffmpeg_output("127.0.0.1", data["rtpPort"] + 1)

    # Log FFmpeg stderr in the background
    threading.Thread(
        target=print_ffmpeg_logs,
        args=(ff_in, "FFmpeg-IN"),
        daemon=True,
    ).start()
    threading.Thread(
        target=print_ffmpeg_logs,
        args=(ff_out, "FFmpeg-OUT"),
        daemon=True,
    ).start()

    # 5s @48kHz stereo 16-bit = 48000 * 2 channels * 2 bytes * 5s
    segment_size = 48000 * 2 * 2 * 5

    threading.Thread(
        target=pump_audio,
        args=(ff_in, ff_out, segment_size, sdp_path),
        daemon=True,
    ).start()


async def main():
    await sio.connect("http://localhost:3000")
    await sio.wait()


if __name__ == "__main__":
    asyncio.run(main())