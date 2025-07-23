import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from subprocess import Popen
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from scipy import signal
import torch
import torchaudio

# import noisereduce as nr


#####################################################
import numpy as np

import cv2
from io import BytesIO


##########################################

ENABLE_TRANSLATION = True
IS_PROD = False

seamlessm4t = 0
if ENABLE_TRANSLATION:
    seamless_streaming = 1
else:
    seamless_streaming = 0

if IS_PROD:
    MEDIASERVER_IP = "10.10.0.82"
else:
    MEDIASERVER_IP = "127.0.0.1"

video_frames_storage = {}
'''
user1 -> [frame1, frame2, frame3,....]
user2 -> [frame1, frame2, frame3,....]
user3 -> [frame1, frame2, frame3,....]
'''
frames_arrived = False


# from streaming_translator_utils import SAMPLE_RATE, StatelessBytesTranslator
# translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output
# start the voice clone
from multiprocessing.connection import Client


def request_voice_clone(audio):
    address = ("localhost", 6000)
    conn = Client(address, authkey=b"secret_vc")
    conn.send(audio)
    result = conn.recv()
    conn.close()
    return result


if seamless_streaming == 1:  # %%

    from seamless_streaming_utils import (
        OutputSegments,
        reset_states,
        get_audio_bytes,
        play_audio,
        play_audio1,
        get_audiosegment,
        build_streaming_system,
        bytes_to_float32_mono_array,
        stream_translate_from_bytes,
    )

    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
        SeamlessStreamingS2STJointVADAgent,
    )
    from simuleval.data.segments import SpeechSegment, EmptySegment, TextSegment

    agent_class = SeamlessStreamingS2STJointVADAgent
    tgt_lang = "hin"

    model_configs = dict(
        source_segment_size=320,
        device="cuda:0",
        dtype="fp16",
        min_starting_wait_w2vbert=192,
        decision_threshold=0.5,
        min_unit_chunk_size=50,
        no_early_stop=True,
        max_len_a=0,
        max_len_b=100,
        task="s2st",
        tgt_lang=tgt_lang,
        block_ngrams=True,
        detokenize_only=True,
    )
    system = build_streaming_system(model_configs, agent_class)
    print("✅ System ready.")

import pickle
def request_lipsync_in_worker(frame_buffer, audio_bytes, output_path):
    address = ('localhost', 6006)
    authkey = b'secret'

    with Client(address, authkey=authkey) as conn:
        data = pickle.dumps((frame_buffer, audio_bytes, output_path))
        conn.send_bytes(data)
        no_audio_path, final_path, error = conn.recv()
        if error:
            raise RuntimeError(f"Lip sync failed: {error}")
        return no_audio_path, final_path


import webrtcvad
import noisereduce as nr

vad = webrtcvad.Vad(1)  # Aggressiveness level

def float32_to_pcm16(audio_float):
    import numpy as np
    audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()

def is_voiced_float32(audio_float, sample_rate=16000, check_ms=300, frame_ms=30):
    pcm_bytes = float32_to_pcm16(audio_float)
    frame_size = int(sample_rate * frame_ms / 1000) * 2  # 2 bytes per sample
    max_bytes_to_check = int(sample_rate * check_ms / 1000) * 2

    for i in range(0, min(len(pcm_bytes), max_bytes_to_check), frame_size):
        frame = pcm_bytes[i:i + frame_size]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            return True
    return False

def timed_is_voiced(float_audio, sample_rate=16000):
    start_time = time.perf_counter()
    result = is_voiced_float32(float_audio, sample_rate)
    elapsed_ms = (time.perf_counter() - start_time) * 1000  # milliseconds
    return result, elapsed_ms

def process_translation_chunk(
    audio_chunk: bytes,
    target_lang: str,
    system,
    system_states,
    output_queue,
    voice_clone_enabled: bool = False,
    request_voice_clone=None,
    tensor_to_bytes=None,
    resample_audio=None,
    save_to_wav=None,
    input_sr: int = 48000,
    target_sr: int = 16000,
    video_frames_storage=None,
    session_id=None
):
    # Convert bytes to float32 mono
    float_audio = bytes_to_float32_mono_array(audio_chunk, input_sr=input_sr, target_sr=target_sr)

    # Optional: Noise reduction
    clean_chunk = nr.reduce_noise(y=float_audio, sr=target_sr)

    # Handle stereo manually if upstream bytes_to_float32_mono_array doesn't already do it
    if clean_chunk.ndim == 2:
        clean_chunk = clean_chunk.mean(axis=0)

    # Clip extremely large chunks (safety for Seamless model)
    MAX_SAMPLES = target_sr * 2  # e.g., 2 seconds max
    if clean_chunk.shape[-1] > MAX_SAMPLES:
        clean_chunk = clean_chunk[-MAX_SAMPLES:]

    # Ensure valid float32 values
    clean_chunk = np.nan_to_num(clean_chunk).astype(np.float32)

    # Run VAD
    is_speech, vad_latency_ms = timed_is_voiced(clean_chunk, sample_rate=target_sr)
    print(f"VAD latency: {vad_latency_ms:.4f} s")

    if is_speech:
        try:
            input_segment = SpeechSegment(content=clean_chunk, sample_rate=target_sr)
            input_segment.tgt_lang = target_lang

            output_segments = OutputSegments(system.pushpop(input_segment, system_states))

            for seg in output_segments.segments:
                if isinstance(seg, SpeechSegment) and seg.sample_rate > 1:
                    print("✅ audio_segment")

                    if voice_clone_enabled:
                        assert request_voice_clone and tensor_to_bytes and resample_audio, \
                            "Voice cloning functions must be provided."

                        clone_tensor = request_voice_clone(seg.content)
                        cloned_audio_bytes = tensor_to_bytes(clone_tensor)
                        translated_audio_bytes = resample_audio(cloned_audio_bytes, 22050, 48000)

                        if save_to_wav:
                            save_to_wav(translated_audio_bytes)
                    else:
                        translated_audio_bytes = get_audio_bytes(seg.content, seg.sample_rate)

                        # Optional: Lip sync if frames exist
                        video_frames = video_frames_storage.pop(session_id, None)
                        lip_syn_enabled = False
                        if lip_syn_enabled and video_frames:
                            print(f"Lip sync started on {len(video_frames)} frames")
                            start_time = time.time()
                            _, final_vid = request_lipsync_in_worker(
                                video_frames,
                                translated_audio_bytes,
                                f"output_{time.time()}.mp4"
                            )
                            print(f"🕒 Lip sync time: {(time.time() - start_time):.4f}s")

                    output_queue.enqueue(translated_audio_bytes)

                elif isinstance(seg, TextSegment):
                    print(f"📝 Translated text: {seg.content}")

            # Handle utterance end
            if output_segments.finished:
                time.sleep(0.3)
                print("⏹️ Utterance ended. Resetting...")
                reset_states(system, system_states)

        except Exception as e:
            print(f"❗ Error during pushpop: {str(e)}. Resetting system state.")
            reset_states(system, system_states)
    else:
        print("🔇 No speech detected. Skipping...")
        reset_states(system, system_states)


# %%

SAMPLE_READ_SIZE = 4096  # minimum number of bytes read from the audio buffers/arrays
OUTPUT_PERIOD = 0.02  # defines frequency at which output is written to the network

# ----------------- OutputAudioQueue ----------------- #
# class responsible for handling the queue used to output audio with thread safety
class OutputAudioQueue:
    def __init__(self):
        self.data = bytearray()  # Array that stores the audio queue
        self.lock = threading.Lock()  # Lock used to control access between threads
        self.closed = False  # Indicates when the process must be stopped
        self.timeout_seconds = 10 * 60  # Stops the threads after 10 min of inactivity
        self.last_write = time.perf_counter()  # Saves last enqueue time

    # Appends new data to the queue
    def enqueue(self, new_data: bytes):
        with self.lock:
            self.data.extend(new_data)
            self.last_write = time.perf_counter()

    # Reads and removes the specified number of bytes from the queue
    def dequeue(self, size):
        with self.lock:
            if time.perf_counter() - self.last_write > self.timeout_seconds:
                self.closed = True
            if len(self.data) == 0:
                # print("returning empty bytes to client")
                return b""
            if size > len(self.data):
                size = len(self.data)
            dequeued_data = self.data[:size]
            self.data = self.data[size:]
            return dequeued_data


# ----------------- Utilities ----------------- #


# Saves the bytes to a wav file on disk for debugging
def save_to_wav(audio_bytes: bytes, sample_rate=48000, num_channels=2, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/output_{int(time.time() * 1000)}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    print(f"💾 Saved audio segment to {filename}")


# Initializes the file used to read input from the network
def write_sdp_file(payload_type, codec_name, clock_rate, channels, rtp_port):
    """
    Generates a one-off SDP file that tells FFmpeg to listen on
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
        f"a=rtpmap:{payload_type} {codec_name}/{clock_rate}/{channels}\n"
        "a=recvonly\n"
        "a=rtcp-mux\n"
    )
    fn = f"audio_{uuid.uuid4().hex}.sdp"
    path = os.path.join(tempfile.gettempdir(), fn)
    with open(path, "w") as f:
        f.write(sdp)
    return path


# Creates pipe that reads the data from the provided SDP path
def run_ffmpeg_input(sdp_path):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "info",
            "-protocol_whitelist",
            "file,udp,rtp",
            "-f",
            "sdp",
            "-i",
            sdp_path,
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "s16le",  # raw PCM
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

# Creates pipe that writes to the destination RTP endpoint
def run_ffmpeg_output(target_ip: str, target_port: int, payload_type: int, ssrc: int):
    cmd = [
        "ffmpeg",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-payload_type",
        str(payload_type),
        "-ssrc",
        str(ssrc),
        "-f",
        "rtp",
        f"rtp://{target_ip}:{target_port}",
    ]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


# Logs FFmpeg errors
def print_ffmpeg_logs(proc, label):
    for line in iter(proc.stderr.readline, b""):
        text = line.decode(errors="ignore").strip()
        if "error" in text.lower():
            print(f"{label}: {text}")


# Resamples and converts mono to stereo
def resample_audio(audio_bytes, original_sr=16000, target_sr=48000):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    new_length = int(len(audio_data) * target_sr / original_sr)
    resampled = signal.resample(audio_data, new_length)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    stereo_data = np.column_stack((resampled, resampled)).flatten()
    return stereo_data.tobytes()


# Converts a numpy tensor to raw PCM bytes
def tensor_to_bytes(translated_wav):
    audio_np = np.clip(np.array(translated_wav, dtype=np.float32), -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


# Function that reads from the input pipe, processes audio, and enqueues to output
def pump_audio(
    ff_in: Popen,
    ff_out: Popen,
    output_queue: OutputAudioQueue,
    segment_size: int,
    sample_rate: int,
    sdp_path: str,
    system,
    system_states,
    target_lang,
    session_id: str,
):
    buf = b""
    chunk_count = 0
    try:
        while True:
            chunk = ff_in.stdout.read(SAMPLE_READ_SIZE)
            if not chunk:
                print("empty chunk, stopping")
                break
            if output_queue.closed:
                print("output closed, stopping")
                break
            buf += chunk
            while len(buf) >= segment_size:
                chunk_count += 1
                seg, buf = buf[:segment_size], buf[segment_size:]
                ##############################################################
                if seamless_streaming == 1:
                    process_translation_chunk(
                        seg,
                        target_lang=target_lang,
                        system=system,
                        system_states=system_states,
                        output_queue=output_queue,
                        voice_clone_enabled=False,
                        request_voice_clone=request_voice_clone,
                        tensor_to_bytes=tensor_to_bytes,
                        resample_audio=resample_audio,
                        save_to_wav=save_to_wav,
                        video_frames_storage=video_frames_storage,
                        session_id= session_id
                    )
                    #################################################
                else:
                    video_frames = video_frames_storage.pop(session_id, None)
                    print(
                        f"🟩 Processing video and audio segment for session {session_id}... {chunk_count}"
                        + (
                            f", video frames: {len(video_frames)}"
                            if video_frames is not None
                            else ", no video frames found."
                        )
                    )
                    
                    
                    output_queue.enqueue(seg)

    finally:
        output_queue.closed = True
        ff_in.stdout.close()
        ff_out.stdin.close()
        ff_in.wait()
        ff_out.wait()
        try:
            os.remove(sdp_path)
        except OSError:
            pass


# Function that writes translated audio from the output queue to the output pipe at correct throughput
def write_to_output(output_queue: OutputAudioQueue, ff_out: Popen):
    next_time = time.perf_counter()
    try:
        while not output_queue.closed:
            seg = output_queue.dequeue(SAMPLE_READ_SIZE)
            if seg:
                try:
                    # print("=======================", seg)
                    ff_out.stdin.write(seg)
                    ff_out.stdin.flush()
                except BrokenPipeError:
                    # stop the voice clone
                    print("⚠️ FFmpeg-OUT pipe closed")
                    return
            next_time += OUTPUT_PERIOD
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()
    except Exception as e:
        print(f"Error in processing thread: {e}")
    finally:
        output_queue.closed = True


# ----------------- FastAPI Server ----------------- #
app = FastAPI()


class TranslationRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    channels: int
    rtpPort: int
    outputPort: int
    ssrc: int
    targetLang: str
    sessionId: str


is_first_process = False


@app.post("/translation/initiate")
async def initiate_translation(data: TranslationRequest):
    global is_first_process  # Declare the variable as global
    if not ENABLE_TRANSLATION:
        is_first_process = False  # Set the global variable to False
    if is_first_process:
        is_first_process = False  # Set the global variable to False
        return
    print("📥 Received translation initiation:", data.dict())
    sample_rate = data.clockRate

    if ENABLE_TRANSLATION:
        system_states = system.build_states()
    else:
        system_states = None
        #system = None
    # Sets up the read file from the rtp port provided by the client
    sdp_path = write_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=sample_rate,
        channels=data.channels,
        rtp_port=data.rtpPort,
    )

    ff_in = run_ffmpeg_input(sdp_path)
    ff_out = run_ffmpeg_output(
        MEDIASERVER_IP, data.outputPort, data.payloadType, data.ssrc
    )

    # Create threads that log errors encountered by FFmpeg
    threading.Thread(
        target=print_ffmpeg_logs, args=(ff_in, "FFmpeg-IN"), daemon=True
    ).start()
    threading.Thread(
        target=print_ffmpeg_logs, args=(ff_out, "FFmpeg-OUT"), daemon=True
    ).start()

    if ENABLE_TRANSLATION:
        segment_size = int(sample_rate * 2 * 2 * 0.5)
    else:
        segment_size = int(sample_rate * 2 * 2 * 2)

    # Initializes output audio queue
    output_queue = OutputAudioQueue()
    # target_lang = data.targetLang
    target_lang = "eng"
    # manually enter the language code here
    # Create thread to process audio
    threading.Thread(
        target=pump_audio,
        args=(
            ff_in,
            ff_out,
            output_queue,
            segment_size,
            sample_rate,
            sdp_path,
            system,
            system_states,
            target_lang,
            data.sessionId,
        ),
        daemon=True,
    ).start()

    # Create thread to output audio
    threading.Thread(
        target=write_to_output, args=(output_queue, ff_out), daemon=True
    ).start()

    return {"status": "Translation pipeline started"}


def write_video_sdp_file(payload_type, codec_name, clock_rate, rtp_port):
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=Mediasoup Video\n"
        "c=IN IP4 0.0.0.0\n"
        "t=0 0\n"
        f"m=video {rtp_port} RTP/AVP {payload_type}\n"
        f"a=rtpmap:{payload_type} {codec_name}/{clock_rate}\n"
        "a=recvonly\n"
        "a=rtcp-mux\n"
    )
    fn = f"video_{uuid.uuid4().hex}.sdp"
    path = os.path.join(tempfile.gettempdir(), fn)
    with open(path, "w") as f:
        f.write(sdp)
    return path


def run_ffmpeg_video_pipe(sdp_path, width=640, height=480):
    print(f"Running FFmpeg with SDP path: {sdp_path}")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "debug",
        "-protocol_whitelist",
        "file,udp,rtp",
        "-f",
        "sdp",
        "-i",
        sdp_path,
        "-an",  # no audio
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
    )


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # for bgr24
FPS = 250
NUM_OF_SECONDS = 5


def save_video_from_frames(frames, output_path, fps=250, frame_size=None):
    if not frames:
        raise ValueError("No frames to write.")

    # Infer frame size from first frame if not provided
    if frame_size is None:
        height, width, _ = frames[0].shape
        frame_size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        # Ensure frame matches target size
        resized = cv2.resize(frame, frame_size)
        out.write(resized)

    out.release()
    print(f"✅ Video saved to {output_path}")


def save_video_async(frames, output_path, fps=30, frame_size=None):
    thread = threading.Thread(
        target=save_video_from_frames,
        args=(frames.copy(), output_path, fps, frame_size),
    )
    thread.daemon = True
    thread.start()


def capture_frames_forever(
    proc: Popen, frame_width: int, frame_height: int, fps: int, num_of_seconds: int = 5
):
    frame_size = frame_width * frame_height * 3  # BGR24
    max_frames = fps * num_of_seconds
    frame_buffer = []
    frame_count = 0
    start_time = time.time()
    try:
        while True:
            raw_frame = proc.stdout.read(frame_size)
            if not raw_frame:
                print("📤 FFmpeg pipe ended")
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (frame_height, frame_width, 3)
            )
            frame_buffer.append(frame)
            # print(len(frame_buffer))
            if len(frame_buffer) == max_frames:
                print(f"time: {time.time()-start_time}")
                # save_video_async(frame_buffer, f"out_{time.time()}.avi", fps=int(fps), frame_size=None)
                print(
                    f"📦 Collected {len(frame_buffer)} frames ({num_of_seconds}s chunk)"
                )

                # Clear buffer for next chunk
                frame_buffer.clear()

    except Exception as e:
        print(f"⚠️ Error in capture_frames_forever: {e}")
    finally:
        try:
            proc.stdout.close()
            proc.stderr.close()
            proc.terminate()
            proc.wait(timeout=5)
        except:
            pass
        cv2.destroyAllWindows()
        print("✅ Frame capture stopped")


def store_frames(
    proc: Popen,
    frame_width: int,
    frame_height: int,
    session_id: str = None,
    video_frames_storage=video_frames_storage,
):
    frame_size = frame_width * frame_height * 3  # BGR24
    count = 0
    try:
        while True:
            count += 1
            raw_frame = proc.stdout.read(frame_size)
            # print(f"📥 Received {len(raw_frame)} bytes for session {session_id} {count}")
            if not raw_frame:
                print("📤 FFmpeg pipe ended")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (frame_height, frame_width, 3)
            )
            # print(f"📦 Received frame for session {session_id} ({frame.shape})")
            video_frames_storage.setdefault(session_id, []).append(frame)


    except Exception as e:
        print(f"⚠️ Error in store_frames: {e}")
    finally:
        try:
            proc.stdout.close()
            proc.stderr.close()
            proc.terminate()
            proc.wait(timeout=5)
        except:
            pass
        print("✅ Frame storage stopped")

def store_frames_as_bytes(proc: Popen):
    """
    This function is a placeholder for storing frames as bytes.
    It can be implemented to convert frames to bytes and store them in a suitable format.
    """
    pass

class VideoCaptureRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    rtpPort: int
    sessionId: str


@app.post("/video/initiate")
async def initiate_video_capture(data: VideoCaptureRequest):
    print("📥 Received video capture initiation:", data.dict())

    sdp_path = write_video_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=data.clockRate,
        rtp_port=data.rtpPort,
    )

    ffmpeg_proc = run_ffmpeg_video_pipe(
        sdp_path, width=FRAME_WIDTH, height=FRAME_HEIGHT
    )
    
    print(f"🔄️ FFmpeg process started with PID {ffmpeg_proc.pid}")

    threading.Thread(
        target=print_ffmpeg_logs, args=(ffmpeg_proc, "FFmpeg-VIDEO"), daemon=True
    ).start()

    # threading.Thread(
    #     target=capture_frames_forever,
    #     args=(ffmpeg_proc, FRAME_WIDTH, FRAME_HEIGHT, FPS, NUM_OF_SECONDS),
    #     daemon=True,
    # ).start()
    threading.Thread(
        target=store_frames,
        args=(
            ffmpeg_proc,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            data.sessionId,
            video_frames_storage,
        ),
        daemon=True,
    ).start()

    return {"status": "Frame-based video capture started."}


# 640 x 480

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=2002, reload=False)
# %%