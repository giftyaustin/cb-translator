#byte_resltime_stream_code_seamlessm4t

import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from seamless_communication.inference import Translator
import time

# --- Global Initializations ---
# For efficiency, initialize the model only once.
MODEL_NAME = "seamlessM4T_v2_large"
VOCODER_NAME = "vocoder_v2" if MODEL_NAME == "seamlessM4T_v2_large" else "vocoder_36langs"

# Check for CUDA availability and set the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize the Translator object
try:
    translator = Translator(
        MODEL_NAME,
        VOCODER_NAME,
        device=DEVICE,
        dtype=DTYPE,
    )
    print(f"Translator initialized on device: {DEVICE}")
except Exception as e:
    print(f"Error initializing translator: {e}")
    translator = None


#raw_bytes_to_audio_translate

def translate_audio(raw_bytes: bytes, sample_width: int, frame_rate: int, channels: int, tgt_lang: str):
    """
    Translates raw audio bytes to a target language.

    Args:
        raw_bytes (bytes): The raw PCM audio data.
        sample_width (int): The sample width in bytes (e.g., 2 for 16-bit).
        frame_rate (int): The sample rate of the input audio (e.g., 44100).
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
        tgt_lang (str): The target language code (e.g., "spa" for Spanish, "fra" for French).

    Returns:
        tuple[torch.Tensor, int] | tuple[None, None]: A tuple containing the translated
        audio waveform as a torch.Tensor and its sample rate as an integer.
        Returns (None, None) if translation fails.
    """
    if not translator:
        print("Translator is not initialized. Cannot proceed with translation.")
        return None, None

    # --- 1. Pre-process Raw Bytes into a Torch Tensor ---
    audio_segment = AudioSegment(
        data=raw_bytes,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels
    )
    segment_filename = f"segment_{time.time()}.wav"
    #audio_segment.export(segment_filename, format="wav")
    samples = np.array(audio_segment.get_array_of_samples())
    if channels == 2:
        samples = samples.reshape((-1, 2))
    samples = samples.astype(np.float32) / 32768.0
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    waveform = torch.from_numpy(samples).unsqueeze(0).to(DEVICE)

    # --- 2. Resample Audio to the Model's Required Sample Rate (16kHz) ---
    model_sr = 16000
    if frame_rate != model_sr:
        print(f"Resampling audio from {frame_rate}Hz to {model_sr}Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=frame_rate, new_freq=model_sr).to(DEVICE)
        waveform = resampler(waveform)

    # --- 3. Translate the Audio ---
    print(f"Translating to {tgt_lang}...")
    try:
        text_output, speech_output = translator.predict(
            input=waveform,
            task_str="s2st",  # Speech-to-Speech Translation
            tgt_lang=tgt_lang,
        )
        translated_wav = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
        translated_sr = speech_output.sample_rate

        print(f"Translated text (this is a transcription of the input tone): {text_output[0]}")
        translated_wav = translated_wav.squeeze().cpu().numpy()
        return translated_wav, translated_sr

    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None, None
