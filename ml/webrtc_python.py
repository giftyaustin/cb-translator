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
        for i in range(0, samples.shape[1], self.samples_per_frame):
            frame_samples = samples[:, i:i+self.samples_per_frame]
            if frame_samples.shape[1] < self.samples_per_frame:
                pad_width = self.samples_per_frame - frame_samples.shape[1]
                frame_samples = np.pad(frame_samples, ((0, 0), (0, pad_width)), mode='constant')

            frame = AudioFrame.from_ndarray(frame_samples, format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            frame.pts = self.timestamp
            self.timestamp += self.samples_per_frame

            self.frame_queue.append(frame)
            self._queue_event.set()

    async def recv(self):
        while not self.frame_queue:
            self._queue_event.clear()
            await self._queue_event.wait()

        frame = self.frame_queue.popleft()
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)
        return frame


def save_wav_from_bytes(filename: str, audio_bytes: bytes, sample_rate=48000, num_channels=1, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filepath = os.path.join("recordings", filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    logging.info(f"💾 Saved WAV file: {filepath}")


@routes.post("/offer")
async def offer(request):
    params = await request.json()
    logging.info("📥 Received offer")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Prepare outgoing audio track
    playback_track = ChunkedAudioStreamTrack()
    pc.addTrack(playback_track)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        logging.info(f"🎤 Track received: {track.kind}")
        if track.kind == "audio":
            fifo = None
            chunk_duration = 5.0
            try:
                while True:
                    frame: AudioFrame = await track.recv()

                    if fifo is None:
                        fifo = AudioFifo(format=frame.format.name,
                                         layout=frame.layout.name,
                                         rate=frame.sample_rate)
                        sample_rate = frame.sample_rate
                        samples_per_chunk = int(chunk_duration * sample_rate)
                        playback_track.sample_rate = sample_rate
                        print(f"Receiving sample rate:{sample_rate} , samples per chunk: {samples_per_chunk}")

                    fifo.write(frame)

                    while fifo.samples >= samples_per_chunk:
                        chunk_frame = fifo.read(samples=samples_per_chunk)
                        logging.info(f"💾 About to save a chunk with {chunk_frame.samples} samples.")
                        samples = chunk_frame.to_ndarray()
                        samples = samples.astype(np.int16).flatten()
                        chunk_bytes = samples.tobytes()

                        timestamp = int(time.time() * 1000)
                        filename = f"chunk_{timestamp}.wav"
                        save_wav_from_bytes(filename, chunk_bytes, sample_rate=sample_rate)

                        # Send chunk back over WebRTC
                        playback_track.push_chunk(chunk_bytes)

            except Exception as e:
                logging.error(f"❌ Error while receiving audio: {e}")

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logging.info(f"Connection state: {pc.connectionState}")
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
    logging.info("💤 Closing peer connections")
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
    web.run_app(app, port=8000)
