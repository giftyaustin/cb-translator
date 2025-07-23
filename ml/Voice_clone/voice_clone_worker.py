# voice_clone_worker.py
import torchaudio
from multiprocessing.connection import Listener
#from TTS.api import TTS
import time
import torch
from pathlib import Path
import scipy
import numpy as np
from TTS.api import TTS
import os
from custom_openvoice import OpenVoice

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_to_wav(wav):
    path = "testOutputs/"
    os.makedirs(path, exist_ok=True)
    path +=f"{time.time}.wav"

    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    scipy.io.wavfile.write(path, vc_model.config.audio.output_sample_rate, wav_norm)

def preprocess_live_audio_for_clone(audio_data, orig_sr, target_sr):

    # Convert list/NumPy to torch.Tensor if needed
    if isinstance(audio_data, list):
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
    elif isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data).float()

    # Ensure audio is float32 tensor
    if not isinstance(audio_data, torch.Tensor):
        raise TypeError("audio_data must be a list, np.ndarray, or torch.Tensor")

    # Convert stereo to mono if needed
    if audio_data.dim() > 1:
        audio_data = torch.mean(audio_data, dim=0)

    # Normalize to [-1, 1]
    audio_data = audio_data / audio_data.abs().max().clamp(min=1e-5)

    # Resample
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    audio_resampled = resampler(audio_data.unsqueeze(0)).squeeze(0)

    return audio_resampled


print("Downloading model...")
tts = TTS(model_name="voice_conversion_models/multilingual/multi-dataset/openvoice_v2").cuda()
vc_model: OpenVoice = tts.voice_converter.vc_model
vc_model.__class__ = OpenVoice
vc_model = vc_model.to("cuda")
speaker_id = "Vijay_ENG"


# Voice clone server loop
def voice_clone_server():
    address = ('0.0.0.0', 6008)
    listener = Listener(address, authkey=b'secret_vc')
    print("[VC Worker] Voice clone server is listening...")

    while True:
        conn = listener.accept()
        try:
            audio_data = conn.recv()
            print(f"[VC Worker] Received audio")
            #print(type(text))
            
            #audio = vc_model.new_load_audio(audio_data)
            #type(audio_data)
            start_time = time.time()
            preprocessed = preprocess_live_audio_for_clone(audio_data, 16000, vc_model.config.audio.input_sample_rate)
            print(f"preprocessing time: {(time.time() - start_time):04f}")
            converted_wav = vc_model.voice_conversion(preprocessed.to(vc_model.device), speaker_id=speaker_id, voice_dir="./voice_embeddings")
            print(f"conversion time: {(time.time() - start_time):04f}")

            #save_to_wav(converted_wav)
            #audio_tensor = torch.tensor(np.array(converted_wav, dtype=np.float32)).unsqueeze(0)  # Shape: (1, N)
            #print(audio_tensor.shape)
            #torchaudio.save(f"clone_{time.time()}.wav", audio_tensor, sample_rate=24000)
            conn.send(converted_wav)
        except Exception as e:
            print(f"[VC Worker] Error: {e}")
            conn.send(None)
        finally:
            conn.close()

if __name__ == "__main__":
    voice_clone_server()
