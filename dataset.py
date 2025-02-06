import torch
from torch.utils.data import Dataset, DataLoader
import random

class EqualizerDataset(Dataset):
    def __init__(self, digital_waveforms, record_target_waveforms, speakers=None, return_dict=False, embedding_pool=True):
        assert len(digital_waveforms) == len(record_target_waveforms), \
            "Input and output waveforms lists must be of the same length"

        self.digital_waveforms = digital_waveforms
        self.record_target_waveforms = record_target_waveforms
        self.speaker_list = speakers
        self.return_dict = return_dict
        self.embedding_pool = embedding_pool

    def __len__(self):
        return len(self.digital_waveforms)

    def __getitem__(self, idx):
        digital_sample = torch.tensor(self.digital_waveforms[idx], dtype=torch.float32)
        record_target_sample = torch.tensor(self.record_target_waveforms[idx], dtype=torch.float32)
        if self.speaker_list is not None and len(self.speaker_list) > 0:
            speaker = self.speaker_list[idx]
            if self.embedding_pool:
                em_idx = random.randint(1, 80)
                text_emb = torch.load(f"assets/embeddings/{speaker}/{speaker}_{em_idx}.pt", weights_only=True)[0].cpu()
            else:
                text_emb = torch.load(f"assets/embeddings/{speaker}/{speaker}_1.pt", weights_only=True)[0].cpu()

        else:
            text_emb = None

        if self.return_dict:
            return {'input_values': digital_sample, 'labels': record_target_sample, "text_emb": text_emb}

        return digital_sample, record_target_sample, text_emb
    