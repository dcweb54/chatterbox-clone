import time
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load


model = ChatterboxTTS.from_pretrained(device="cpu")

torch.load = original_torch_load

text = "Hello, This is Mahimai"

start_time = time.perf_counter_ns()
wav = model.generate(text)
end_time = time.perf_counter_ns()

ta.save("test-1.wav", wav, model.sr)

print(f"Time taken: {end_time - start_time} seconds")