import asyncio
import json
import logging
import numpy as np
import fractions
import wave
import os
import time
from collections import deque
import threading
import torch
import torchaudio
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from av.audio.fifo import AudioFifo

#initialization of seamless translation models 
from seamlessm4t_translator_utils import translate_audio
from streaming_translator_utils import StatelessBytesTranslator
translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output

# numpy array to bytes
# numpy array to bytes
def tensor_to_bytes_original(translated_wav):
    # 1. Assume this is your audio in float32 format (range -1.0 to 1.0)
    audio_np = np.array(translated_wav, dtype=np.float32)
    # 2. Clip to [-1, 1] just in case
    audio_np = np.clip(audio_np, -1.0, 1.0)
    # 3. Convert to int16 format (PCM 16-bit)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    # 4. Convert to raw PCM bytes
    translated_audio_bytes = audio_int16.tobytes()
    return translated_audio_bytes
def tensor_to_bytes(translated_wav, input_sr=16000, output_sr=48000, output_channels=2):
    # Step 1: Convert to torch.Tensor if not already
    if not isinstance(translated_wav, torch.Tensor):
        translated_wav = torch.tensor(translated_wav)

    # Step 2: Ensure shape = (1, N), i.e., mono
    if translated_wav.dim() == 1:
        translated_wav = translated_wav.unsqueeze(0)
    elif translated_wav.size(0) > 1:
        translated_wav = translated_wav.mean(dim=0, keepdim=True)  # force mono

    # Step 3: Resample to 48kHz
    resampler = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=output_sr)
    wav_48k = resampler(translated_wav)

    # Step 4: Mono to Stereo (duplicate channel)
    if output_channels == 2:
        wav_48k = wav_48k.repeat(2, 1)

    # Step 5: Clamp and convert to int16 PCM
    wav_48k = torch.clamp(wav_48k, -1.0, 1.0)
    wav_int16 = (wav_48k * 32767.0).to(torch.int16)

    # Step 6: Convert to bytes
    return wav_int16.transpose(0, 1).contiguous().numpy().tobytes()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pcs = set()
routes = web.RouteTableDef()

class ChunkedAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000, samples_per_frame=960):
        super().__init__()
        self.frame_queue = deque()
        self.timestamp = 0
        self._queue_event = asyncio.Event()
        self.update_info(sample_rate, samples_per_frame)
        
        # Thread control
        self._processing_thread = None
        self._stop_event = threading.Event()
        self._start_processing_thread()

    def update_info(self, sample_rate, samples_per_frame, format_name='s16', layout_name='stereo'):
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.frame_interval = samples_per_frame/sample_rate #used to guarantee proper timing when sending frames back

        # internal audio queue, used to send frames of correct size
        self.fifo = AudioFifo(  format=format_name,
                                layout=layout_name,
                                rate=sample_rate)
        
    def _start_processing_thread(self):
        if self._processing_thread is None or not self._processing_thread.is_alive():
            self._stop_event.clear()
            self._processing_thread = threading.Thread(
                target=self._process_frames_loop,
                daemon=True
            )
            self._processing_thread.start()

    def _process_frames_loop(self):
        next_time = time.perf_counter() # Variable that controls when to send the next frame
        
        while not self._stop_event.is_set():
            try:
                # Send the frame
                self.send_frame()

                # Calculate when to send the next frame
                next_time += self.frame_interval
                sleep_time = next_time - time.perf_counter()
                
                # Only sleep if theres time left
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Otherwise, set the time to send the next frame to now
                else:
                    next_time = time.perf_counter()

            except Exception as e:
                print(f"Error in processing thread: {e}")

    # Simply writes the frame to the FIFO - processing happens in separate thread
    def push_av_frame(self, frame: AudioFrame):
        self.fifo.write(frame)
    
    # Function that actually takes a frame from the fifo and sends it
    def send_frame(self):
        if(self.fifo.samples >= self.samples_per_frame):
            chunk_frame = self.fifo.read(samples=self.samples_per_frame)
            chunk_frame.time_base = fractions.Fraction(1, self.sample_rate)
            chunk_frame.pts = self.timestamp
            self.timestamp += chunk_frame.samples
            self.frame_queue.append(chunk_frame)
            logger.info(f"📦 Pushed frame {chunk_frame}")
            self._queue_event.set()

    # Function that receives a frame
    async def recv(self):
        while not self.frame_queue:
            self._queue_event.clear()
            await self._queue_event.wait()
        return self.frame_queue.popleft()
    
    # Cleanup functions
    def stop(self):
        if self._processing_thread and self._processing_thread.is_alive():
            self._stop_event.set()
            self._processing_thread.join(timeout=1.0)

    def __del__(self):
        self.stop()

# This function takes the audio frame, unwraps it to bytes and runs the
# operation_func, a callable (function, lambda, etc.) that takes bytes and 
# returns modified bytes in the same sample rate and datatype
def process_audio_frame_bytes(
    input_frame: AudioFrame,
    operation_func, 
) -> AudioFrame:
    input_frame_array = input_frame.to_ndarray()
    input_shape = input_frame_array.shape
    audio_bytes = input_frame_array.tobytes()
    expected_bytes_len = len(audio_bytes)

    processed_bytes, processed_bytes_original = operation_func(audio_bytes)
    #processed_bytes = processed_bytes + processed_bytes
    filename = f"processed_{time.time()}.wav"
    save_wav_from_bytes(filename, processed_bytes, 48000, 2, 2)
    save_wav_from_bytes("_processed.wav", processed_bytes_original, 16000, 1, 2)

    if not isinstance(processed_bytes, bytes):
        raise TypeError("operation_func must return bytes.")
    
    # Determine numpy dtype from input format
    if input_frame.format.name == "s16":
        np_dtype = np.int16
        bytes_per_sample = 2
    elif input_frame.format.name == "flt":
        np_dtype = np.float32
        bytes_per_sample = 4
    else:
        logger.warning(f"Unsupported input format {input_frame.format.name}. Defaulting to int16.")
        np_dtype = np.int16
        bytes_per_sample = 2

    # Calculate output dimensions
    num_channels = input_frame_array.shape[0]
    total_samples = len(processed_bytes) // bytes_per_sample
    
    if total_samples % num_channels != 0:
        raise ValueError(
            f"Processed bytes length ({len(processed_bytes)}) is not compatible "
            f"with {num_channels} channels and {bytes_per_sample} bytes per sample."
        )
    
    samples_per_channel = total_samples // num_channels
    output_shape = (num_channels, samples_per_channel)

    processed_ndarray = np.frombuffer(processed_bytes, dtype=np_dtype).reshape(
        output_shape
    )
    
    output_frame = AudioFrame.from_ndarray(
        processed_ndarray,
        format=input_frame.format.name,
        layout=input_frame.layout.name
    )
    output_frame.sample_rate = input_frame.sample_rate
    output_frame.time_base = input_frame.time_base
    
    # # Adjust PTS based on the length change
    # if input_frame.pts is not None:
    #     input_samples = input_frame_array.shape[1]
    #     pts_scale = samples_per_channel / input_samples if input_samples > 0 else 1
    #     output_frame.pts = int(input_frame.pts * pts_scale)
    # else:
    #     output_frame.pts = input_frame.pts
    output_frame.pts = None
    return output_frame

# Function that saves wav files for debugging
def save_wav_from_bytes(filename: str, audio_bytes: bytes, sample_rate=48000, num_channels=1, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filepath = os.path.join("recordings", filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    logger.info(f"💾 Saved WAV file: {filepath}")

# example function to pass to the process_audio_frame_bytes function. This one just calls save_wav_from_bytes
def audio_bytes_function(chunk_bytes, sample_rate):
    logger.info(f"💾 About to save chunk: samples={len(chunk_bytes)}")
    timestamp = int(time.time() * 1000)
    # translate here
    translated_wav, translated_sr = translate_audio(chunk_bytes, sample_width=2, frame_rate = sample_rate, channels = 2, tgt_lang = "hin")
    processed_bytes_original = tensor_to_bytes_original(translated_wav)
    processed_bytes = tensor_to_bytes(translated_wav)
    filename = f"chunk_{timestamp}.wav"
    #save_wav_from_bytes(filename, chunk_bytes, sample_rate=sample_rate, num_channels=2)
    return processed_bytes, processed_bytes_original

@routes.post("/offer")
async def offer(request):
    params = await request.json()
    logger.info("📥 Received offer")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Prepare outgoing audio track
    playback_track = ChunkedAudioStreamTrack()
    pc.addTrack(playback_track)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        logger.info(f"🎤 Track received: kind={track.kind}")
        if track.kind == "audio":
            fifo = None
            chunk_duration = 5.0
            try:
                while True:
                    frame: AudioFrame = await track.recv()
                    
                    if fifo is None:
                        frame_rate = frame.sample_rate
                        fifo = AudioFifo(format=frame.format.name,
                                         layout=frame.layout.name,
                                         rate=frame_rate)
                        samples_per_batch= int(chunk_duration * frame_rate)
                        playback_track.update_info(frame_rate, frame.samples, frame.format.name, frame.layout.name)
                        logger.info(f"Initialized AudioFifo: sample_rate={frame_rate}, samples_per_chunk={samples_per_batch}")

                    fifo.write(frame)
                    logging.info(f"received frame: {frame}")

                    while fifo.samples >= samples_per_batch:
                        # reads chunk from the input queue
                        chunk_frame = fifo.read(samples=samples_per_batch)
                        # processes the chunk
                        output_frame = process_audio_frame_bytes(chunk_frame, lambda audio_bytes:audio_bytes_function(audio_bytes, frame_rate))
                        # adds chunk to the output queue
                        playback_track.push_av_frame(output_frame)

            except Exception as e:
                logger.error(f"❌ Error while receiving audio: {e}", exc_info=True)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state changed: {pc.connectionState}")
        if pc.connectionState in ("closed", "failed"):
            pcs.discard(pc)
            await pc.close()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def on_shutdown(app):
    logger.info("💤 Closing peer connections")
    await asyncio.gather(*(pc.close() for pc in pcs))

app = web.Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)

cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    logger.info("Starting WebRTC server on port 8000")
    web.run_app(app, port=8000)