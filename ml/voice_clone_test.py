# main.py
import torch
from multiprocessing.connection import Client

def request_voice_clone(text, language='en'):
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'secret_vc')
    conn.send((text, language))
    result = conn.recv()
    conn.close()
    return result

if __name__ == "__main__":
    texts = [
        "Hello, this is live cloning!",
        "We are testing multiple calls.",
        "This should run continuously.",
        "How are you today?",
        "This is the fifth message.",
        "Almost done!"
    ]

    for idx, text in enumerate(texts):
        output_tensor = request_voice_clone(text, language="en")
        if isinstance(output_tensor, torch.Tensor):
            print(f"[Main] Received tensor {idx+1}: {output_tensor.shape}")
            import torchaudio
            torchaudio.save(f"received_clone_{idx+1}.wav", output_tensor, sample_rate=24000)
        else:
            print(f"[Main] Voice clone failed for message {idx+1}")

