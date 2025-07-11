# voice_clone_worker.py
import torch
import torchaudio
import numpy as np
from multiprocessing.connection import Listener
from TTS.api import TTS
import time

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts_model = tts.synthesizer.tts_model

# Load precomputed speaker latent and embedding
latent = torch.load("/home/test/xtts_voice_clone/xtts2-ui-main/gpt_cond_latents_aryan_hindi.pt", weights_only=True).to(device)
embedding = torch.load("/home/test/xtts_voice_clone/xtts2-ui-main/speaker_embedding_aryan_hindi.pt", weights_only=True).to(device)

# Voice clone server loop
def voice_clone_server():
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'secret_vc')
    print("[VC Worker] Voice clone server is listening...")

    while True:
        conn = listener.accept()
        try:
            text, language = conn.recv()
            print(f"[VC Worker] Received text: {text[:50]}... | Language: {language}")
            #print(type(text))
            with torch.inference_mode():
                wav = tts.tts(
                    text= text,
                    language=language,
                    precomputed_embedding=embedding,
                    precomputed_latent=latent,
                    split_sentences=False,
                    speed=3.0
                )
            audio_tensor = torch.tensor(np.array(wav, dtype=np.float32)).unsqueeze(0)  # Shape: (1, N)
            #print(audio_tensor.shape)
            #torchaudio.save(f"clone_{time.time()}.wav", audio_tensor, sample_rate=24000)
            conn.send(audio_tensor)
        except Exception as e:
            print(f"[VC Worker] Error: {e}")
            conn.send(None)
        finally:
            conn.close()

if __name__ == "__main__":
    voice_clone_server()
