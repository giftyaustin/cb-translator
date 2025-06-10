import asyncio
import json
import logging
import os
import uuid
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder
from av import AudioFrame, open as av_open

logging.basicConfig(level=logging.INFO)
pcs = set()
routes = web.RouteTableDef()

RECORDINGS_DIR = os.path.join(os.getcwd(), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# A custom MediaStreamTrack that streams back recorded audio frames
class WavChunkStreamer(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()


    async def recv(self):
        frame = await self.queue.get()
        return frame

    def push_frame(self, frame):
        self.queue.put_nowait(frame)


@routes.post("/offer")
async def offer(request):
    params = await request.json()
    logging.info("📥 Received offer")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)
    logging.info("✅ Created RTCPeerConnection")

    # Create the return track and add it to the connection early
    streamer = WavChunkStreamer()
    pc.addTrack(streamer)
    logging.info("📤 Added streamer return track")

    track_received = asyncio.Event()

    @pc.on("track")
    async def on_track(track):
        logging.info(f"🎤 Track received: {track.kind}")

        if track.kind == "audio":
            track_received.set()

            async def record_and_stream_chunks():
                logging.info("🔄 Starting chunk recording and streaming...")
                try:
                    while True:
                        chunk_id = uuid.uuid4().hex
                        filename = os.path.join(RECORDINGS_DIR, f"audio_{chunk_id}.wav")
                        recorder = MediaRecorder(filename)
                        recorder.addTrack(track)
                        await recorder.start()
                        logging.info(f"⏺ Recording chunk: {filename}")
                        await asyncio.sleep(5)
                        await recorder.stop()
                        logging.info(f"✅ Chunk saved: {filename}")

                        # Stream recorded frames to the client
                        container = av_open(filename, mode="r")
                        audio_stream = container.streams.audio[0]
                        decoder = audio_stream.codec_context
                        for packet in container.demux(audio_stream):
                            for frame in decoder.decode(packet):
                                streamer.push_frame(frame)
                        container.close()
                        logging.info(f"📤 Streamed chunk: {filename}")

                except Exception as e:
                    logging.error(f"🛑 Error during chunk streaming: {e}")

            asyncio.create_task(record_and_stream_chunks())

    await pc.setRemoteDescription(offer)
    await track_received.wait()  # Ensure incoming track is processed

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logging.info("📤 Sending answer")

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def on_shutdown(app):
    logging.info("💤 Closing peer connections")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

app = web.Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)

# CORS setup
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
