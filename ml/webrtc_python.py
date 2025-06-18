import asyncio
import json
import logging
import numpy as np
import fractions
import wave
import os
import time
from collections import deque

from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from av.audio.fifo import AudioFifo

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

    def update_info(self, sample_rate, samples_per_frame, format_name='s16', layout_name='stereo'):
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame

        # internal audio queue, used to send frames of correct size
        self.fifo = AudioFifo(  format=format_name,
                                layout=layout_name,
                                rate=sample_rate)

    def push_av_frame(self, frame: AudioFrame):
        self.fifo.write(frame)
        processed_any = False
        while self.fifo.samples >= self.samples_per_frame:
            processed_any = True
            chunk_frame = self.fifo.read(samples=self.samples_per_frame)
            chunk_frame.time_base = fractions.Fraction(1, self.sample_rate)
            chunk_frame.pts = self.timestamp
            self.timestamp += chunk_frame.samples
            self.frame_queue.append(chunk_frame)
            logging.info(f"📦 Pushed frame {frame}")
        if processed_any:
            self._queue_event.set()

    async def recv(self):
        while not self.frame_queue:
            self._queue_event.clear()
            await self._queue_event.wait()
        return self.frame_queue.popleft()

def process_audio_frame_bytes(
    input_frame: AudioFrame,
    operation_func, # A callable (function, lambda, etc.) that takes bytes and returns modified bytes (such as a language translation model that returns the same ammount of bytes it receives)
) -> AudioFrame:
    input_frame_array = input_frame.to_ndarray()
    input_shape = input_frame_array.shape
    audio_bytes = input_frame_array.tobytes()
    expected_bytes_len = len(audio_bytes)

    processed_bytes = operation_func(audio_bytes)

    if not isinstance(processed_bytes, bytes):
        raise TypeError("operation_func must return bytes.")
    if len(processed_bytes) != expected_bytes_len:
        raise ValueError(
            f"operation_func returned bytes of incorrect length. "
            f"Expected {expected_bytes_len}, got {len(processed_bytes)}."
        )
    
    if input_frame.format.name == "s16":
        np_dtype = np.int16
    elif input_frame.format.name == "flt":
        np_dtype = np.float32
    else:
        logger.warning(f"Unsupported input format {input_frame.format.name}. Defaulting to int16.")
        np_dtype = np.int16

    processed_ndarray = np.frombuffer(processed_bytes, dtype=np_dtype).reshape(
        input_shape[0], input_shape[1]
    )
    output_frame = AudioFrame.from_ndarray(
        processed_ndarray,
        format=input_frame.format.name,
        layout=input_frame.layout.name
    )
    output_frame.sample_rate = input_frame.sample_rate
    output_frame.time_base = input_frame.time_base
    output_frame.pts = input_frame.pts

    return output_frame

def save_wav_from_bytes(filename: str, audio_bytes: bytes, sample_rate=48000, num_channels=1, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filepath = os.path.join("recordings", filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    logger.info(f"💾 Saved WAV file: {filepath}")

def audio_bytes_function(chunk_bytes, sample_rate):
    logger.info(f"💾 About to save chunk: samples={len(chunk_bytes)}")
    timestamp = int(time.time() * 1000)
    filename = f"chunk_{timestamp}.wav"
    save_wav_from_bytes(filename, chunk_bytes, sample_rate=sample_rate, num_channels=2)
    return chunk_bytes

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
            chunk_duration = 1.0 # TODO in ChunkedAudioStreamTrack, send audio in a separated thread to ensure proper timing of the output, allowing bigger duration
            try:
                while True:
                    frame: AudioFrame = await track.recv()
                    
                    if fifo is None:
                        frame_rate = frame.sample_rate
                        fifo = AudioFifo(format=frame.format.name,
                                         layout=frame.layout.name,
                                         rate=frame_rate)
                        samples_per_chunk = int(chunk_duration * frame_rate)
                        playback_track.update_info(frame_rate, frame.samples, frame.format.name, frame.layout.name)
                        logger.info(f"Initialized AudioFifo: sample_rate={frame_rate}, samples_per_chunk={samples_per_chunk}")

                    fifo.write(frame)
                    logging.info(f"received frame: {frame}")

                    while fifo.samples >= samples_per_chunk:
                        chunk_frame = fifo.read(samples=samples_per_chunk)
                        output_frame = process_audio_frame_bytes(chunk_frame, lambda x:audio_bytes_function(x, frame_rate))

                        # Send chunk back over WebRTC
                        logging.info("Starting streaming back")
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