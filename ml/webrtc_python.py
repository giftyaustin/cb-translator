import asyncio
import json
import logging
import numpy as np
import fractions
import wave
import os
import av
import time
from collections import deque
import ssl

from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from av import VideoFrame
from av.audio.fifo import AudioFifo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pcs = set()
routes = web.RouteTableDef()


class ChunkedAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = 960
        self.frame_queue = deque()
        self.timestamp = 0
        self._queue_event = asyncio.Event()

    def push_chunk(self, chunk_bytes: bytes):
        samples = np.frombuffer(chunk_bytes, dtype=np.int16).reshape(1, -1)
        # samples = np.frombuffer(chunk_bytes, dtype=np.int16).reshape(2, -1)
        logging.info(f"🔊 Received {samples.shape[1]} samples with shape {samples.shape}")
        for i in range(0, samples.shape[1], self.samples_per_frame):
            frame_samples = samples[:, i: i+self.samples_per_frame]
            if frame_samples.shape[1] < self.samples_per_frame:
                pad_width = self.samples_per_frame - frame_samples.shape[1]
                # frame_samples = np.pad(frame_samples, ((0, 0), (0, pad_width)), mode='constant')
            # logging.info(f"==================== {frame_samples.shape}")
            frame = AudioFrame.from_ndarray(frame_samples, format="s16", layout="stereo")
            frame.sample_rate = self.sample_rate
            frame.time_base = fractions.Fraction(1, 48000)
            frame.pts = self.timestamp
            self.timestamp += self.samples_per_frame

            self.frame_queue.append(frame)
            self._queue_event.set()

    async def recv(self):
        while not self.frame_queue:
            self._queue_event.clear()
            await self._queue_event.wait()

        frame = self.frame_queue.popleft()
        frame_time = self.samples_per_frame / self.sample_rate
        # await asyncio.sleep(frame_time)
        return frame


def save_wav_from_bytes(filename: str, audio_bytes: bytes, sample_rate=48000, num_channels=1, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filepath = os.path.join("recordings", filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    logger.info(f"💾 Saved WAV file: {filepath}")
    
    
    
def save_video_chunk_mkv(frames, filename, frame_rate):
    os.makedirs("recordings", exist_ok=True)
    filename = filename.replace('.mp4', '.mkv')  # Ensure correct extension

    logger.info(f"💾 Saving video chunk to {filename} with {len(frames)} frames")

    container = av.open(filename, mode='w', format='matroska')
    stream = container.add_stream('libx264', rate=frame_rate)
    stream.width = frames[0].width
    stream.height = frames[0].height
    stream.pix_fmt = 'yuv420p'

    for frame in frames:
        if not isinstance(frame, av.VideoFrame):
            frame = av.VideoFrame.from_ndarray(frame.to_ndarray(), format='bgr24')

        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    logger.info(f"✅ Saved {filename}")


    
async def handle_video(track: MediaStreamTrack):
    logger.info("📹 Video track handler initialized")

    chunk_duration = 5.0
    frame_rate = 30  # Default fallback
    frames_per_chunk = int(chunk_duration * frame_rate)
    fifo = []
    logging.info(await track.recv())

    try:
        while True:
            frame:VideoFrame = await track.recv()
            frame_rate = frame_rate
            frames_per_chunk = int(chunk_duration * frame_rate)

            fifo.append(frame)

            if len(fifo) >= frames_per_chunk:
                timestamp = int(time.time() * 1000)
                filename = f"recordings/video_chunk_{timestamp}.mp4"
                save_video_chunk_mkv(fifo, filename, frame_rate)
                fifo.clear()

    except Exception as e:
        logger.error(f"❌ Error in video handler: {e}", exc_info=True)



@routes.get("/test")
async def test(request):
    # logger.info("✅ Test endpoint hit by client!")
    return web.json_response({"status": "ok", "message": "Translator WebRTC server is reachable"})



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
        if track.kind == "audio": # !! change this to "audio" to match the correct kind
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
                        sample_rate = frame_rate
                        samples_per_chunk = int(chunk_duration * sample_rate)
                        playback_track.sample_rate = sample_rate
                        logger.info(f"Initialized AudioFifo: sample_rate={sample_rate}, samples_per_chunk={samples_per_chunk}")

                    fifo.write(frame)

                    while fifo.samples >= samples_per_chunk:
                        chunk_frame = fifo.read(samples=samples_per_chunk)
                        logger.info(f"💾 About to save chunk: samples={chunk_frame.samples}")
                        samples = chunk_frame.to_ndarray()
                        logging.info(f"ℹ️ Accumulated {samples.shape[1]} samples with shape {samples.shape}")
                        chunk_bytes = samples.tobytes()    
                                            
                        timestamp = int(time.time() * 1000)
                        filename = f"chunk_{timestamp}.wav"
                        save_wav_from_bytes(filename, chunk_bytes, sample_rate=sample_rate, num_channels=2)

                        # Send chunk back over WebRTC
                        logging.info("Starting streaming back")
                        playback_track.push_chunk(chunk_bytes)

            except Exception as e:
                logger.error(f"❌ Error while receiving audio: {e}", exc_info=True)

        if track.kind == "video":
            await handle_video(track)

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
    # ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_ctx.load_cert_chain(certfile="cert.pem", keyfile="cert-key.pem")
    # web.run_app(app, port=8000, ssl_context=ssl_ctx) 
    web.run_app(app, port=8000)  
