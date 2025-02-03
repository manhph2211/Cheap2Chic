from diffusers import AudioLDMPipeline
import torch

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "low quality" # "high quality" and "normal quality"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]