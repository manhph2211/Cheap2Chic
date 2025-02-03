import os
import torch
import librosa
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers import AudioLDMPipeline, AudioLDM2Pipeline
from torch.optim import AdamW
import soundfile as sf
import numpy as np
from utils import *


def wav_feature_extraction(self, waveform):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft, energy = get_mel_from_wav(waveform, self.STFT)

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
    return log_mel_spec, stft


class AudioLDMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform = torch.tensor(item["audio"], dtype=torch.float32)
        prompt = item["prompt"]
        return {"audio": waveform, "prompt": prompt}

def preprocess_audioldm(data_path, segment_length, stride_length, target_sampling_rate, total_files=60):
    qualities = {
        "EN_y2_headphone": "1. high quality",
        "EN_x": "2. normal quality",
        "EN_y1": "3. low quality",
    }

    train_files = int(0.9 * total_files)
    dataset = {"train": [], "val": []}

    for quality_folder, prompt in qualities.items():
        folder_path = os.path.join(data_path, quality_folder)
        for i in range(1, total_files + 1):
            file_path = os.path.join(folder_path, f"{i}.wav")
            waveform, _ = librosa.load(file_path, sr=target_sampling_rate)

            segment_samples = int(segment_length * target_sampling_rate)
            stride_samples = int(stride_length * target_sampling_rate)

            for start in range(0, len(waveform) - segment_samples + 1, stride_samples):
                segment = waveform[start:start + segment_samples]
                subset = "train" if i <= train_files else "val"
                dataset[subset].append({"audio": segment, "prompt": prompt})

    return dataset

def train_one_epoch(pipe, dataloader, optimizer):
    pipe.unet.train()
    total_loss = 0.0

    for batch in dataloader:
        waveforms = batch["audio"].to("cuda")
        prompts = batch["prompt"]

        noise = torch.randn_like(waveforms)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (waveforms.size(0),), device="cuda")
        # noisy_waveforms = pipe.scheduler.add_noise(waveforms, noise, timesteps)


        prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
            prompt="low quality",
            device="cuda",
            do_classifier_free_guidance=False,
            num_waveforms_per_prompt=1
        )


        vocoder_upsample_factor = np.prod(pipe.vocoder.config.upsample_rates) / pipe.vocoder.config.sampling_rate
        audio_length_in_s = segment_length

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * pipe.vocoder.config.sampling_rate)
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        if height % vae_scale_factor != 0:
            height = int(np.ceil(height / vae_scale_factor)) * vae_scale_factor


        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator=None,
            latents=None,
        )

        noise_pred = pipe.unet(
            latents,
            timesteps,
            encoder_hidden_states=generated_prompt_embeds,
            encoder_hidden_states_1=prompt_embeds,
            encoder_attention_mask_1=attention_mask,
            return_dict=False,
        )[0]
        
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader):
    model.unet.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["audio"].to("cuda")
            prompts = batch["prompt"]

            tokenized_prompts = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
            noise = torch.randn_like(waveforms)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (waveforms.size(0),), device="cuda")
            noisy_waveforms = model.scheduler.add_noise(waveforms, noise, timesteps)

            noise_pred = model.unet(noisy_waveforms, timesteps, encoder_hidden_states=tokenized_prompts.input_ids)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    data_path = "data/v2"
    segment_length = 5.0 
    stride_length = 0.5 
    target_sampling_rate = 16000
    total_files = 60
    num_epochs = 20
    batch_size = 16
    device = 'cuda'
    best_model_path = "assets/best_audioldm_checkpoint"

    print("Preprocessing data...")
    dataset = preprocess_audioldm(data_path, segment_length, stride_length, target_sampling_rate, total_files)

    train_dataset = AudioLDMDataset(dataset["train"])
    val_dataset = AudioLDMDataset(dataset["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Loading model...")
    pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2").to("cuda")

    optimizer = AdamW(pipe.unet.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):

        train_loss = train_one_epoch(pipe, train_loader, optimizer)

        val_loss = validate_one_epoch(pipe, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pipe.unet.save_pretrained(best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
