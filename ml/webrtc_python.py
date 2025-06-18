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

from seamlessm4t_translator_utils import translate_audio
from streaming_translator_utils import StatelessBytesTranslator
translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output

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
            frame_samples = samples[:, i:i+self.samples_per_frame]
            if frame_samples.shape[1] < self.samples_per_frame:
                pad_width = self.samples_per_frame - frame_samples.shape[1]
                frame_samples = np.pad(frame_samples, ((0, 0), (0, pad_width)), mode='constant')
            # logging.info(f"==================== {frame_samples.shape}")
            frame = AudioFrame.from_ndarray(frame_samples, format="s16", layout="stereo")
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
        frame_time = self.samples_per_frame / self.sample_rate
        await asyncio.sleep(frame_time)
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
                        #print(chunk_bytes)

                        which_translator = 2

                        if which_translator == 1:
                            #seamelessm4T
                            sample_rate = 48000
                            start_time = time.time()
                            translated_wav, translated_sr = translate_audio(chunk_bytes, sample_width=2, frame_rate = sample_rate, channels = 2, tgt_lang = "hin")
                            end_time = time.time()
                            print(f"Inference time: {end_time-start_time: .4f} sec.")
                            print(translated_sr)
                            out_file = f"translated_raw_{time.time()}.wav"
                            #torchaudio.save(out_file, translated_wav, 16000)
                            #translated_segment = AudioSegment.from_wav(out_file)
                            #play(translated_segment)
                            #translated_wav = translated_wav.squeeze().cpu().numpy()
                            print(translated_wav)

                        if which_translator ==2:
                            #seamless_streaming
                            #audio_bytes = original_segment.raw_data
                            sample_width = 2
                            frame_rate = 48000
                            channels = 2
                            #print(f"Sample width: {sample_width}, Frame rate: {frame_rate}, Channels: {channels}")
                            start_time = time.time()
                            translated_wav, text = translator1.translate_chunk(
                                chunk_bytes,
                                input_sample_rate=frame_rate,
                                sample_width=sample_width,
                                channels=channels
                                )
                            end_time = time.time()
                            print(translated_wav, text)
                            print(f"Inference time: {end_time-start_time: .4f} sec.")
                            if translated_wav is not None:
                                translator1.play_audio(translated_wav)
                                #translator1.save_audio(translated_wav)
                            if text:
                                print("📝", text)
                        
                        translated_audio_bytes = tensor_to_bytes(translated_wav)

                        timestamp = int(time.time() * 1000)
                        filename = f"chunk_{timestamp}.wav"
                        save_wav_from_bytes(filename, chunk_bytes, sample_rate=sample_rate, num_channels=2)

                        # Send chunk back over WebRTC
                        logging.info("Starting streaming back")
                        #playback_track.push_chunk(chunk_bytes)
                        playback_track.push_chunk(translated_audio_bytes)


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
