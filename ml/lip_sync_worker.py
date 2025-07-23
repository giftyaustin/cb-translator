# lip_sync_worker.py

import sys
import pickle
import torch
from multiprocessing.connection import Listener

#for lip sync
import numpy as np
import os
import subprocess
from tqdm import tqdm
from models import Wav2Lip
import audio
import argparse
import time
from ultralytics import YOLO
import ffmpeg
import cv2
import threading
import queue
import torchaudio
import io

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Inference code for lip-syncing videos using Wav2Lip models')
parser.add_argument('--checkpoint_path', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/checkpoints/wav2lip_Chinese.pth')
parser.add_argument('--face', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/demo.mp4')
parser.add_argument('--audio', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/demo.wav')
parser.add_argument('--outfile', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/result20.mp4')
parser.add_argument('--static', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--fps', type=float, default=25)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
parser.add_argument('--face_det_batch_size', type=int, default=16)
parser.add_argument('--wav2lip_batch_size', type=int, default=64)
parser.add_argument('--resize_factor', type=int, default=1)
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
parser.add_argument('--rotate', default=False, action='store_true')
parser.add_argument('--nosmooth', default=False, action='store_true')
args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
    args.static = True

from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")

#detector = MTCNN(device=device)
detector = YOLO('yolov8n-face.pt').to(device)
dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
_ = detector.predict(dummy_img, verbose=False)# def face_detect1(images):

def face_detect(images):
    start_time = time.time()
    results = []
    last_box = None

    for i, img in enumerate(images):
        if i % 10 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yolo_result = detector.predict(img_rgb, verbose=False)
            boxes = yolo_result[0].boxes.xyxy

            if boxes is None or len(boxes) == 0:
                last_box = None
                results.append(None)
                continue

            x1, y1, x2, y2 = boxes[0].int().tolist()
            pady1, pady2, padx1, padx2 = args.pads
            y1 = max(0, y1 - pady1)
            y2 = min(img.shape[0], y2 + pady2)
            x1 = max(0, x1 - padx1)
            x2 = min(img.shape[1], x2 + padx2)
            last_box = [x1, y1, x2, y2]

        results.append(last_box)

    return results

def load_model(path):
    start = time.time()
    model = Wav2Lip()
    print("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device)
    model.eval()
    print(f"[load_model] Loaded in {time.time() - start:.2f}s")
    return model

model = load_model("wav2lip_Chinese.pth")

# ✅ MODIFIED datagen (preserve original frame if no face)
def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    #print(len(frames), len(mels))
    if args.box[0] == -1:
        print("face detection in data gen")
        face_det_results = face_detect(frames if not args.static else [frames[0]])
    else:
        print('Using the specified bounding box...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[x1, y1, x2, y2] for _ in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame = frames[idx]
        face_coords = face_det_results[idx]

        if face_coords is None:
            # No face detected: keep frame, insert None for coords
            frame_batch.append(frame)
            coords_batch.append(None)
            continue

        x1, y1, x2, y2 = face_coords
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (args.img_size, args.img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame)
        coords_batch.append((x1, y1, x2, y2))

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch = np.asarray(img_batch, dtype=np.float32) / 255.0
        mel_batch = np.asarray(mel_batch, dtype=np.float32)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:, :, :] = 0
        img_combined = np.empty((img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], img_batch.shape[3] * 2), dtype=np.float32)
        img_combined[:, :, :, :img_batch.shape[3]] = img_masked
        img_combined[:, :, :, img_batch.shape[3]:] = img_batch
        mel_batch = mel_batch[..., np.newaxis]
        yield img_combined, mel_batch, frame_batch, coords_batch

def video_writer_worker(write_queue, output_path, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        frame = write_queue.get()
        if frame is None:
            break
        writer.write(frame)

    writer.release()


def run_lipsync_from_frames(frame_buffer: list, audio_bytes: bytes, output_path: str, with_audio=True, fps=250):

    args.outfile = output_path
    full_frames = frame_buffer
    fps = int(fps)
    print(f"[stream] Received {len(full_frames)} frames")

    # === 1. Audio Preprocessing ===
    audio_start = time.time()
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    wav = waveform.squeeze().cpu().numpy()
    mel = audio.melspectrogram(wav)
    print(f"[stream] Audio processed in {time.time() - audio_start:.2f}s")

    # === 2. Mel Chunks Creation ===
    mel_chunks, mel_step_size = [], 16
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, -mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1
    print(f"[stream] Mel chunks: {len(mel_chunks)}")

    # === 3. Match mel_chunks and frames ===
    print(f"mel chunks: {len(mel_chunks)}, fullframes: {len(full_frames)}")
    if len(mel_chunks) > len(full_frames):
        full_frames += [full_frames[-1]] * (len(mel_chunks) - len(full_frames))
    else:
        full_frames = full_frames[:len(mel_chunks)]

    # === 4. Prepare datagen ===
    gen = datagen(full_frames.copy(), mel_chunks)

    os.makedirs("temp", exist_ok=True)
    no_audio_video_path = f"temp/result_{int(time.time())}.avi"
    frame_h, frame_w = full_frames[0].shape[:2]

    # === 5. Threaded Video Writer ===
    write_queue = queue.Queue()
    writer_thread = threading.Thread(target=video_writer_worker, args=(write_queue, no_audio_video_path, (frame_w, frame_h), fps))
    writer_thread.start()

    # === 6. Run Inference ===
    frames_written = 0
    args.wav2lip_batch_size = min(len(mel_chunks), args.wav2lip_batch_size or 32)
    if args.wav2lip_batch_size == 0:
        raise ValueError("No mel chunks available.")

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(gen, total=int(np.ceil(len(mel_chunks) / args.wav2lip_batch_size)))
    ):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            if c is None:
                write_queue.put(f)
                continue
            x1, y1, x2, y2 = c
            if x2 <= x1 or y2 <= y1:
                write_queue.put(f)
                continue
            try:
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = cv2.addWeighted(f[y1:y2, x1:x2], 0.2, p, 0.8, 0)
            except Exception as e:
                print(f"[warning] Error applying lipsync: {e}")
            write_queue.put(f)
            frames_written += 1

    # === 7. Fallback if no frames written ===
    if frames_written == 0:
        print("⚠️ No processed frames. Writing fallback video with original frames.")
        for f in full_frames:
            write_queue.put(f)

    write_queue.put(None)
    writer_thread.join()
    print(f"[stream] Lip-synced video saved to {no_audio_video_path}")

    # === 8. Mux Audio (if required) ===
    final_output_path = args.outfile
    if with_audio:
        mux_start = time.time()
        audio_wav_path = "temp/input_audio.wav"
        with open(audio_wav_path, "wb") as f:
            f.write(audio_bytes)

        command = [
            'ffmpeg', '-y',
            '-i', no_audio_video_path,
            '-i', audio_wav_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-c:a', 'aac',
            '-shortest',
            final_output_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ FFmpeg failed:")
            print(result.stderr)
        else:
            print(f"[muxing] Audio+Video muxed in {time.time() - mux_start:.2f}s")

        if not os.path.exists(final_output_path):
            print("❌ Output file missing after muxing")
            final_output_path = None
        else:
            print(f"✅ Output muxed video: {final_output_path}")

    return no_audio_video_path, final_output_path

address = ('localhost', 6006)  # or a UNIX socket path
authkey = b'secret'

print("[Worker] Starting worker...")
with Listener(address, authkey=authkey) as listener:
    while True:
        print("[Worker] Waiting for a job...")
        with listener.accept() as conn:
            print("[Worker] Job accepted")

            try:
                data = conn.recv_bytes()
                frame_buffer, audio_path, output_path = pickle.loads(data)
                print(f"frame_size: {len(frame_buffer)}")

                print("[Worker] Running lip sync...")
                no_audio_path, final_path = run_lipsync_from_frames(
                    frame_buffer=frame_buffer,
                    audio_bytes=audio_path,
                    output_path=output_path,
                    with_audio=True
                )
                conn.send((no_audio_path, final_path, None))
                print("[Worker] Job completed")

            except Exception as e:
                print(f"[Worker] Error: {e}")
                conn.send((None, None, str(e)))
