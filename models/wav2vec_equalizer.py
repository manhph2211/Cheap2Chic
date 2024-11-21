
import torch
import torch.nn as nn 
from transformers import Wav2Vec2Model


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=layers, bidirectional=bi, batch_first=True)
        self.linear = nn.Linear(dim * 2, dim) if bi else None

    def forward(self, x):
        x, _ = self.lstm(x)  
        if self.linear:
            x = self.linear(x)
        return x  


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.down_conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.down_conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.down_conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        
        self.up_conv1 = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up_conv2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.ConvTranspose1d(hidden_dim, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x1 = self.down_conv1(x) 
        x2 = self.down_conv2(x1)  
        x3 = self.down_conv3(x2)
        
        x = self.up_conv1(x3) 
        x = self.up_conv2(x) 
        x = self.final_conv(x)     
        
        return x


class Wav2VecEqualizer(nn.Module):
    def __init__(self, freeze_encoder=True, target_length=160000):
        super().__init__()
        
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "pos_conv_embed" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # self.lstm = BLSTM(dim=768)
        
        self.decoder = Decoder(input_dim=499, hidden_dim=256)
        self.target_length = target_length
        self.fc = nn.Linear(768, target_length)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state
        x = self.decoder(x).squeeze(1)        
        x = self.fc(x)
        return x