import torch
from torch.utils.data import Dataset, DataLoader


class EqualizerDataset(Dataset):
    def __init__(self, digital_waveforms, record_low_waveforms, return_dict=False):
        assert len(digital_waveforms) == len(record_low_waveforms), \
            "Input and output waveforms lists must be of the same length"

        self.digital_waveforms = digital_waveforms
        self.record_low_waveforms = record_low_waveforms
        self.return_dict = return_dict

    def __len__(self):
        return len(self.digital_waveforms)

    def __getitem__(self, idx):
        digital_sample = torch.tensor(self.digital_waveforms[idx], dtype=torch.float32)
        record_low_sample = torch.tensor(self.record_low_waveforms[idx], dtype=torch.float32)
        if self.return_dict:
            return {'input_values': digital_sample, 'labels': record_low_sample}

        return digital_sample, record_low_sample