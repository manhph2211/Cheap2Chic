import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaTextModel
from concurrent.futures import ThreadPoolExecutor

import random
from tqdm import tqdm
import json

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

converter = read_json("assets/converter.json")
print(converter)

device_nums = 10
emb_nums_each_device = 100

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
text_model = MllamaTextModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

def process_device(i):
    device = converter[f"y{i}"]
    # input_prompt = f"Write a very short paragraph about the frequency responses of device {i}, e.g. report peak or trough values and trends in low (0-250Hz), mid (250-4000Hz), high (4000-12000Hz) and ultra-High (12000-20000Hz) frequency range (no recommendations, no suggestions, no steps, no thinking, no applications)."
    input_prompt = f"Write a single short paragraph about unique key observations in the frequency responses of {device} device."
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": input_prompt}
        ]}
    ]
    image = Image.open(f"assets/embeddings/y{i}/y{i}.jpg").convert("RGB")
    input_text = processor.apply_chat_template(messages) #add_generation_prompt=True
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)
    print(f"\n####################### PROCESSING DEVICE {i} ########################")
    for j in tqdm(range(1, 1 + emb_nums_each_device)):
        output = model.generate(**inputs, max_new_tokens=300, temperature=0.3, output_hidden_states=True)
        out_text = processor.decode(output[0], skip_special_tokens=True)
        print("***************************")
        print(out_text.replace(input_prompt, "").replace("assistant", "").replace("user", "").strip())
        text_inputs = processor(text=out_text.replace(input_prompt, "").replace("assistant", "").replace("user", "").strip(), return_tensors="pt")
        text_output = text_model(**text_inputs) # print(output.last_hidden_state.shape)
        torch.save(text_output.last_hidden_state.mean(dim=1), f"assets/embeddings/y{i}/y{i}_{j}.pt")

for i in range(9, device_nums + 1):
    process_device(i)

# with ThreadPoolExecutor(max_workers=device_nums) as executor:
#     executor.map(process_device, range(1, device_nums + 1))
