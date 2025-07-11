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
import noisereduce as nr


#####################################################
import av
import numpy as np
import cv2
from io import BytesIO

def video_bytes_to_frames(byte_data):
    # Use a memory buffer
    container = av.open(BytesIO(byte_data), format='mp4')  # or 'webm', 'h264', etc.

    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')  # Convert to OpenCV format
        frames.append(img)

    return frames

##########################################


seamlessm4t = 0
seamless_streaming = 1

# Import your translators
if seamlessm4t == 1:
    from seamlessm4t_translator_utils import translate_audio

#from streaming_translator_utils import SAMPLE_RATE, StatelessBytesTranslator
#translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output
#start the voice clone
from multiprocessing.connection import Client
def request_voice_clone(audio):
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'secret_vc')
    conn.send(audio)
    result = conn.recv()
    conn.close()
    return result


if seamless_streaming ==1:#%%
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



    def get_audio_bytes(samples, sr: int, target_sr: int = 48000, stereo: bool = False) -> bytes:
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
    # system_states = system.build_states()

    #stream_translate(system, tgt_lang)
#%%

SAMPLE_READ_SIZE = 4096  # minimum number of bytes read from the audio buffers/arrays
OUTPUT_PERIOD = 0.02     # defines frequency at which output is written to the network

# ----------------- OutputAudioQueue ----------------- #
# class responsible for handling the queue used to output audio with thread safety
class OutputAudioQueue:
    def __init__(self):
        self.data = bytearray()                # Array that stores the audio queue
        self.lock = threading.Lock()           # Lock used to control access between threads
        self.closed = False                    # Indicates when the process must be stopped
        self.timeout_seconds = 10 * 60         # Stops the threads after 10 min of inactivity
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
                return b''
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
            "-loglevel", "info",
            "-protocol_whitelist", "file,udp,rtp",
            "-f", "sdp",
            "-i", sdp_path,
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "s16le",  # raw PCM
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

# Creates pipe that writes to the destination RTP endpoint
def run_ffmpeg_output(target_ip: str, target_port: int, payload_type: int, ssrc: int):
    cmd = [
        "ffmpeg",
        "-f", "s16le", "-ar", "48000", "-ac", "2",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-payload_type", str(payload_type),
        "-ssrc", str(ssrc),
        "-f", "rtp",
        f"rtp://{target_ip}:{target_port}"
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

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

# # Function that runs the translation steps on the audio bytes
# def translate(audio: bytes, sample_rate: int):
#     #### Original translation code provided ####
#     which_translator = 1

#     if which_translator == 1:
#         #seamelessm4T
#         start_time = time.time()
#         translated_wav, translated_sr, translated_text= translate_audio(audio, sample_width=2, frame_rate = sample_rate, channels = 2, tgt_lang = "hin")
#         end_time = time.time()
#         print(f"Inference time: {end_time-start_time: .4f} sec.")
#         print(translated_text)

#     if which_translator == 2:
#         #seamless_streaming
#         sample_width = 2
#         channels = 2
#         #print(f"Sample width: {sample_width}, Frame rate: {frame_rate}, Channels: {channels}")
#         start_time = time.time()
#         translated_wav, translated_text = translator1.translate_chunk(
#             audio,
#             input_sample_rate=sample_rate,
#             sample_width=sample_width,
#             channels=channels
#         )
#         end_time = time.time()
#         #print(translated_wav, text)
#         print(f"Inference time: {end_time-start_time: .4f} sec.")
#         if translated_text:
#             print("📝", translated_text)
#     translated_audio_bytes = tensor_to_bytes(translated_wav)
#     return resample_audio(translated_audio_bytes, SAMPLE_RATE, sample_rate)

def translate(audio: bytes, sample_rate: int):
    #### Original translation code provided ####
    which_translator = 1

    if which_translator == 1:
        #seamelessm4T
        start_time = time.time()
        translated_wav, translated_sr, translated_text= translate_audio(audio, sample_width=2, frame_rate = sample_rate, channels = 2, tgt_lang = "hin")
        end_time = time.time()
        print(f"Inference time: {end_time-start_time: .4f} sec.")
        print(translated_text)

    # if which_translator == 2:
    #     #seamless_streaming
    #     sample_width = 2
    #     channels = 2
    #     #print(f"Sample width: {sample_width}, Frame rate: {frame_rate}, Channels: {channels}")
    #     start_time = time.time()
    #     translated_wav, translated_text = translator1.translate_chunk(
    #         audio,
    #         input_sample_rate=sample_rate,
    #         sample_width=sample_width,
    #         channels=channels
    #     )
    #     end_time = time.time()
    #     #print(translated_wav, text)
    #     print(f"Inference time: {end_time-start_time: .4f} sec.")
    #     if translated_text:
    #         print("📝", translated_text)
    #translated_audio_bytes = tensor_to_bytes(translated_wav)
    return translated_wav, translated_sr, translated_text

# Function that reads from the input pipe, processes audio, and enqueues to output
def pump_audio(ff_in: Popen, ff_out: Popen, output_queue: OutputAudioQueue, segment_size: int, sample_rate: int, sdp_path: str, system, system_states, target_lang):
    buf = b""
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
                seg, buf = buf[:segment_size], buf[segment_size:]
                ##############################################################
                if seamless_streaming == 1:
                    float_audio = bytes_to_float32_mono_array(seg, input_sr=48000, target_sr=16000)

                    # ⏱️ Inference time measurement
                    #clean_chunk = nr.reduce_noise(y=float_audio, sr=SAMPLE_RATE) 
                    input_segment = SpeechSegment(content=float_audio, sample_rate=16000)
                    input_segment.tgt_lang = target_lang

                    # ⏱️ Start timer before inference
                    start_time = time.time()

                    # Translation pipeline (likely includes STT → Translate → TTS)
                    output_segments = OutputSegments(system.pushpop(input_segment, system_states))

                    # ⏱️ End timer after inference
                    inference_time = time.time() - start_time
                    # print(f"🕒 Inference time: {inference_time:.3f} sec")
                    # output_queue.enqueue(get_audio_bytes(output_segments.segments))
                    for seg in output_segments.segments:
                        if isinstance(seg, SpeechSegment):
                            if seg.sample_rate > 1:
                                voice_clone = 0
                                #for cloning
                                if voice_clone == 1:
                                    clone_tensor = request_voice_clone(seg.content)
                                    #print(clone_tensor.shape)
                                    #torchaudio.save(f"received_clone_{time.time()}.wav", clone_tensor, sample_rate=24000)
                                    cloned_audio_bytes = tensor_to_bytes(clone_tensor)
                                    translated_audio_bytes = resample_audio(cloned_audio_bytes, 24000, 48000)
                                    save_to_wav(translated_audio_bytes)
                                    #play_audio(translated_audio_bytes)
                                else:
                                    translated_audio_bytes = get_audio_bytes(seg.content, seg.sample_rate)
                                    #play_audio1(translated_audio_bytes)
                                output_queue.enqueue(translated_audio_bytes)


                        elif isinstance(seg, TextSegment):
                            print(f"📝 Translated text: {seg.content}")

                    if output_segments.finished:
                        time.sleep(0.3)
                        print("⏹️ Utterance ended. Resetting...")
                        reset_states(system, system_states)

                if seamlessm4t == 1:
                    translated_wav, translated_sr, translated_text = translate(seg, sample_rate)
                    #print(translated_wav.shape)
                    voice_clone = 0
                    #for cloning
                    if voice_clone == 1:
                        print(str(translated_text))
                        clone_tensor = request_voice_clone(str(translated_text), language="hi")
                        #print(clone_tensor.squeeze().cpu().numpy().shape)
                        #torchaudio.save(f"received_clone_{time.time()}.wav", clone_tensor, sample_rate=24000)
                        cloned_audio_bytes = tensor_to_bytes(clone_tensor.squeeze().cpu().numpy())
                        translated_bytes = resample_audio(cloned_audio_bytes, 24000, sample_rate)
                    else: 
                        translated_audio_bytes = tensor_to_bytes(translated_wav)
                        translated_bytes = resample_audio(translated_audio_bytes, translated_sr, sample_rate)

                    save_to_wav(translated_bytes)
                    print(f"📦 Processed segment: {len(seg)} bytes")
                    output_queue.enqueue(translated_bytes)
                    ################################################
               
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
                    #stop the voice clone
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
    # targetLang: str

is_first_process = True

@app.post("/translation/initiate")
async def initiate_translation(data: TranslationRequest):
    global is_first_process  # Declare the variable as global

    if is_first_process:
        is_first_process = False  # Set the global variable to False
        return
    print("📥 Received translation initiation:", data.dict())
    sample_rate = data.clockRate

    print("======================== Initializing systems ==========================")
    start_time = time.time()
    # system = build_streaming_system(model_configs, agent_class)
    print("============== system initiated ================")
    system_states = system.build_states()
    end_time = time.time()
    print(end_time - start_time, "==============>>>> Inference time for model initiation")
    # Sets up the read file from the rtp port provided by the client
    sdp_path = write_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=sample_rate,
        channels=data.channels,
        rtp_port=data.rtpPort
    )

    ff_in = run_ffmpeg_input(sdp_path)
    ff_out = run_ffmpeg_output(
        "10.10.0.82", data.outputPort, data.payloadType, data.ssrc
    )

    # Create threads that log errors encountered by FFmpeg
    threading.Thread(target=print_ffmpeg_logs, args=(ff_in, "FFmpeg-IN"), daemon=True).start()
    threading.Thread(target=print_ffmpeg_logs, args=(ff_out, "FFmpeg-OUT"), daemon=True).start()

    # 5s @ 48kHz stereo 16-bit = sample_rate * 2 channels * 2 bytes * 5s
    if seamless_streaming == 1:
        segment_size = int(sample_rate * 2 * 2 * 0.5)
    if seamlessm4t ==1:
        segment_size = int(sample_rate * 2 * 2 * 5)

    # Initializes output audio queue
    output_queue = OutputAudioQueue()
    # target_lang = data.targetLang
    target_lang = "eng"
    #manually enter the language code here
    # Create thread to process audio
    threading.Thread(
        target=pump_audio,
        args=(ff_in, ff_out, output_queue, segment_size, sample_rate, sdp_path, system, system_states, target_lang),
        daemon=True
    ).start()

    # Create thread to output audio
    threading.Thread(
        target=write_to_output,
        args=(output_queue, ff_out),
        daemon=True
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
 
 
def run_ffmpeg_video_pipe(sdp_path):
    cmd = [
        "ffmpeg",
        "-loglevel", "info",
        "-protocol_whitelist", "file,udp,rtp",
        "-f", "sdp",
        "-i", sdp_path,
        "-c:v", "copy",
        "-f", "mpegts",
        "pipe:1"
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )

 
def capture_video_chunks_forever(proc: Popen, chunk_duration):
    buf = bytearray()
    start_time = time.perf_counter()
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                print("📤 FFmpeg pipe ended, restarting capture loop")
                break
            buf.extend(chunk)
            elapsed = time.perf_counter() - start_time
            if elapsed >= chunk_duration:
                print(type(buf))

                # # Example usage
                #video_bytes = bytearray(buf)  # your byte stream
                #frames = video_bytes_to_frames(video_bytes)
                #print(frames[0].shape)

                # # Display or save a frame
                #cv2.imshow("Frame 0", frames[0])
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()


                print(f"📦 Captured 5-second video chunk: {len(buf)} bytes")
                # buf for video
                buf.clear()
                start_time = time.perf_counter()
 
    except Exception as e:
        print(f"⚠️ Error in capture_video_chunks_forever: {e}")
    finally:
        try:
            proc.stdout.close()
            proc.stderr.close()
            proc.terminate()
            proc.wait(timeout=5)
        except:
            pass
        print("✅ Video chunk capture stopped cleanly")
 
 
 
class VideoCaptureRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    rtpPort: int
 
@app.post("/video/initiate")
async def initiate_video_capture(data: VideoCaptureRequest):
    print("📥 Received video capture initiation:", data.dict())
 
    # Write SDP file
    sdp_path = write_video_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=data.clockRate,
        rtp_port=data.rtpPort
    )
 
    # Start FFmpeg process
    ffmpeg_proc = run_ffmpeg_video_pipe(sdp_path)
 
    # FFmpeg logs thread
    threading.Thread(
        target=print_ffmpeg_logs, args=(ffmpeg_proc, "FFmpeg-VIDEO"), daemon=True
    ).start()
 
    # Video capture chunk loop thread
    capture_thread = threading.Thread(
        target=capture_video_chunks_forever,
        args=(ffmpeg_proc, 5),
        daemon=True
    )
    capture_thread.start()
 
    return {"status": "Video capture started, chunking 1-second slices infinitely."}
 




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=2002)
# %%
