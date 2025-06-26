import asyncio
import os
import socketio
import subprocess
import tempfile
import threading
import time
import wave
import uuid
from subprocess import Popen
import numpy as np
from scipy import signal

from seamlessm4t_translator_utils import translate_audio
from streaming_translator_utils import SAMPLE_RATE, StatelessBytesTranslator
translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output

SAMPLE_READ_SIZE = 4096 # minimum number of bytes read from the audio buffers/arrays
OUTPUT_PERIOD = 0.02    # defines frequency at which output is written to the network

# class resposible for handling the queue used to output audio to the socket
# with special attention to thread safety
class OutputAudioQueue:
    def __init__(self):
        self.data = bytearray()                 # Array that stores the audio queue
        self.lock = threading.Lock()            # Lock used to controll access to the array between threads
        self.closed = False                     # Variable used to communicate when the process must be stopped to the threads
        self.timeout_seconds = 10*60            # Stops the threads after some time of innactivity (10 minutes here)
        self.last_write = time.perf_counter()   # Variable that saves the last time the queue was appended to

    # Appends new data to the queue.
    def enqueue(self, new_data: bytes):
        with self.lock:
            self.data.extend(new_data)
            self.last_write = time.perf_counter()

    # Reads the specified number of bytes from the queue (removing them)
    def dequeue(self, size):
        with self.lock:
            if(time.perf_counter() - self.last_write > self.timeout_seconds):   # Checks if timeout happened
                self.closed = True
            if len(self.data) == 0:             # If the data is empty, just return a copy of the entire queue (empty array)
                return self.data[:]
            if size > len(self.data):           # Adjusting size if requested size exceeds available data
                size = len(self.data)  
            dequeued_data = self.data[:size]    # Get the requested bytes
            self.data = self.data[size:]        # Remove the bytes from que array
            return dequeued_data
        
# Saves the bytes to a wav file in disk for debugging
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

# Initializes the file used to read input from the network
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

# creates pipe that reads the data from the sdp path provided
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

# creates pipe the writes to the destination rtp file
def run_ffmpeg_output(target_ip: str, target_port: InterruptedError,
                      payload_type: int, ssrc: int):
    cmd = [
      "ffmpeg",
      "-f","s16le", "-ar","48000","-ac","2",
      "-i","pipe:0",
      "-c:a","libopus",
      "-payload_type", str(payload_type),
      "-ssrc",str(ssrc),
      "-f","rtp",
      f"rtp://{target_ip}:{target_port}"
    ]
    return subprocess.Popen(cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)

# Logs ffmpeg errors
def print_ffmpeg_logs(proc, label):
    for line in iter(proc.stderr.readline, b""):
        text = line.decode().strip()
        # Only print lines that contain "error" (case-insensitive)
        if "error" in text.lower():
            print(f"{label}: {text}")

# resamples and converts from mono to stereo
def resample_audio(audio_bytes, original_sr=16000, target_sr=48000):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    new_length = int(len(audio_data) * target_sr / original_sr)
    resampled = signal.resample(audio_data, new_length)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    stereo_data = np.column_stack((resampled, resampled)).flatten()
    return stereo_data.tobytes()

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

# Function that runs the translation steps on the audio bytes,
# Returning the translated bytes in the provided sample rate and stereo
def translate(audio: bytes, sample_rate: int):
    which_translator = 2

    if which_translator == 1:
        #seamelessm4T
        start_time = time.time()
        translated_wav, translated_sr = translate_audio(audio, sample_width=2, frame_rate = sample_rate, channels = 2, tgt_lang = "hin")
        end_time = time.time()
        print(f"Inference time: {end_time-start_time: .4f} sec.")
        print(translated_sr)

    if which_translator == 2:
        #seamless_streaming
        sample_width = 2
        channels = 2
        start_time = time.time()
        translated_wav, text = translator1.translate_chunk(
            audio,
            input_sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels
        )
        end_time = time.time()
        print(translated_wav, text)
        print(f"Inference time: {end_time-start_time: .4f} sec.")
        if text:
            print("📝", text)
    translated_audio_bytes = tensor_to_bytes(translated_wav)
    return resample_audio(translated_audio_bytes, SAMPLE_RATE, sample_rate)

# Runs a loop that reads data from the input pipe, processes it,
# and adds the processed bytes to the output queue
def pump_audio(
        ff_in: Popen[bytes],
        ff_out: Popen[bytes],
        output_queue: OutputAudioQueue,
        segment_size: int,
        sample_rate: int,
        sdp_path: str):
    
    buf = b"" #Inner buffer used to chunk the data into bigger chunks for processing
    try:
        while True:
            chunk = ff_in.stdout.read(SAMPLE_READ_SIZE)

            # If returned chunk is null, there was an error in the input pipe
            if not chunk:
                print('empty chunk, stopping')
                break
            
            # Checks if the output queue is opperational
            if(output_queue.closed):
                print('output closed, stopping')
                break
            
            buf += chunk
            while len(buf) >= segment_size:
                seg, buf = buf[:segment_size], buf[segment_size:]   # reads from the start of the buffer and removes the data that was read
                save_to_wav(seg, sample_rate=sample_rate)           # saves audio for debugging
                translated_bytes = translate(seg, sample_rate)      # translates audio
                print(f"📦 Processed segment: {len(seg)} bytes")
                output_queue.enqueue(translated_bytes)              # writes to the output queue
    finally:
        output_queue.closed = True  # Makes sure the other threads are notified that the input was closed
        ff_in.stdout.close()        # Closes input pipe
        ff_out.stdin.close()        # Closes output pipe
        ff_in.wait()                # Waits for operations to complete
        ff_out.wait()
        try:
            os.remove(sdp_path)     # removes the sdp created for input
        except OSError:
            pass

# Runs the loop that reads data from the output queue and
# writes it to the output pipe (network) at the correct throughput
def write_to_output(output_queue: OutputAudioQueue, ff_out: Popen[bytes]):
    next_time = time.perf_counter()
    
    try:
        while not output_queue.closed:          # Runs the loop as long as the output queue is opperational
            # Send the frame if there is any available data
            seg = output_queue.dequeue(SAMPLE_READ_SIZE)
            if(len(seg)>=0):
                try:
                    ff_out.stdin.write(seg)
                    ff_out.stdin.flush()
                except BrokenPipeError:
                    print("⚠️ FFmpeg-OUT pipe closed")
                    return

            # Calculate when to send the next frame
            next_time += OUTPUT_PERIOD
            sleep_time = next_time - time.perf_counter()
            
            # Only sleep if theres time left
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Otherwise, set the time to send the next frame to now
            else:
                next_time = time.perf_counter()
                
    except Exception as e:
        print(f"Error in processing thread: {e}")
    finally:
        output_queue.closed = True  # Makes sure the other threads are notified that the output was closed


# Initializes socket client
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
    sample_rate = data["clockRate"]

    # Sets up the read file from the rtp port provided by the client
    sdp_path = write_sdp_file(
        payload_type=data["payloadType"],
        codec_name=data["codec"],
        clock_rate=sample_rate,
        channels=data["channels"],
        rtp_port=data["rtpPort"],   # Trusting the client's provided port is unique. This is handled by the typescript code
    )

    ff_in = run_ffmpeg_input(sdp_path)  # Sets up input pipe
    ff_out = run_ffmpeg_output(         # Sets up output pipe
        "127.0.0.1",
        data["outputPort"],         # Trusting the client's provided output port is unique. This is handled by the typescript code
        data["payloadType"],
        data["ssrc"])

    # Creating threads that Log errors encountered by FFmpeg
    threading.Thread(
        target=print_ffmpeg_logs,
        args=(ff_in, "FFmpeg-IN"),
        daemon=True
    ).start()
    threading.Thread(
        target=print_ffmpeg_logs,
        args=(ff_out, "FFmpeg-OUT"),
        daemon=True
    ).start()

    # 5s @48kHz stereo 16-bit = sample_rate * 2 channels * 2 bytes * 5s
    segment_size = sample_rate * 2 * 2 * 5
    
    # Initializes output audio queue
    output_queue = OutputAudioQueue()

    # Creating thread that processes audio
    threading.Thread(
        target=pump_audio,
        args=(ff_in, ff_out, output_queue, segment_size, sample_rate, sdp_path),
        daemon=True
    ).start()
    
    # Creating thread that outputs audio
    threading.Thread(
        target=write_to_output,
        args=(output_queue, ff_out),
        daemon=True
    ).start()


async def main():
    await sio.connect("http://localhost:3000")
    await sio.wait()


if __name__ == "__main__":
    asyncio.run(main())