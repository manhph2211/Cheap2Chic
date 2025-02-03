import os
import torch
import librosa
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import soundfile as sf
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler


class AudioLDMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform = torch.tensor(item["audio"], dtype=torch.float32).reshape(1,-1)
        prompt = item["prompt"]
        return {"audio": waveform, "prompt": prompt}


def preprocess_audioldm(data_path, segment_length, stride_length, target_sampling_rate, total_files=60):
    qualities = {
        "EN_y2_headphone": "1",
        "EN_x": "2",
        "EN_y1": "3",
    }

    train_files = int(0.95 * total_files)
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


# Training function
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    device = "cuda"

    for batch in tqdm(dataloader):
        waveforms = batch["audio"].to(device)
        prompts = batch["prompt"]

        loss = model(
            waveforms,
            text=prompts,
            embedding_mask_proba=0.0
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Validation function
def validate_one_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    device = "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveforms = batch["audio"].to(device)
            prompts = batch["prompt"]

            # Calculate loss
            loss = model(
                waveforms,
                text=prompts,
                embedding_mask_proba=0.0
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    data_path = "data/v2"
    segment_length = 2**16/16000
    stride_length = 0.4
    target_sampling_rate = 16000
    total_files = 60
    num_epochs = 20
    batch_size = 48
    device = 'cuda'
    best_model_path = "assets/best_model_checkpoint.pth"

    print("Preprocessing data...")
    dataset = preprocess_audioldm(data_path, segment_length, stride_length, target_sampling_rate, total_files)

    train_dataset = AudioLDMDataset(dataset["train"])
    val_dataset = AudioLDMDataset(dataset["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing model...")
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=1,
        channels=[8, 32, 64, 128, 256, 512], # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2], # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2], # U-Net: number of repeating items at each layer
        attentions=[0, 0, 0, 0, 0, 1], # U-Net: attention enabled/disabled at each layer
        attention_heads=8,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        use_text_conditioning=True,
        use_embedding_cfg=True,
        embedding_max_length=64,
        embedding_features=768,
        cross_attentions=[0, 0, 0, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
    ).to(device)
    # model.load_state_dict(torch.load("assets/best_model_checkpoint.pth", map_location='cuda'))

    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate_one_epoch(model, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
