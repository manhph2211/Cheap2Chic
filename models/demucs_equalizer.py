import torch
import torchaudio
import torch.nn as nn 
from demucs import pretrained


class DemucsEqualizer(nn.Module):
    def __init__(self, freeze=False, model_name="htdemucs", device="cuda"):
        super().__init__()
        self.model = pretrained.get_model(model_name).models[0].to(device)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = torch.cat([x, x], dim=1)
        x = self.model(x) 
        x = x.mean(dim=1)[:,0,:]  
        return x.reshape(x.shape[0], -1)


class DoubleDemucsEqualizer(nn.Module):
    def __init__(self, checkpoint_path, model_name="htdemucs", device="cuda"):
        super().__init__()
        self.model1 = DemucsEqualizer(model_name=model_name, device=device)
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            self.model1.load_state_dict(state_dict)
        except:
            raise "STH WRONG: CHECKPOINT FOR MODEL 1"
        
        self.model1.eval()
            
        for name, param in self.model1.named_parameters():
            param.requires_grad = False

        self.model2 = pretrained.get_model(model_name).models[0].to(device)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = torch.cat([x, x], dim=1)        
        x = self.model2(x)
        x = x.mean(dim=1)[:,0,:]  
        x = x.reshape(x.shape[0], -1)
        x = self.model1(x)
        return x
    
    
''' V2

from denoiser import pretrained

class DemucsEqualizer(nn.Module):
    def __init__(self, freeze=False, model_name="dns64", device="cuda"):
        super().__init__()
        self.model = pretrained.__dict__[model_name]().to(device)
        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        self.sample_rate = self.model.sample_rate

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.model(x) 
        return x.reshape(x.shape[0], -1)


class DoubleDemucsEqualizer(nn.Module):
    def __init__(self, checkpoint_path, model_name="dns64", device="cuda"):
        super().__init__()
        self.model1 = DemucsEqualizer(model_name=model_name, device=device)
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            self.model1.load_state_dict(state_dict)
        except:
            raise "STH WRONG: CHECKPOINT FOR MODEL 1"
        
        self.model1.eval()
            
        for name, param in self.model1.named_parameters():
            param.requires_grad = False

        self.model2 = pretrained.__dict__[model_name]().to(device)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1) 
        x = self.model2(x).reshape(x.shape[0], 1, -1) 
        x = self.model1(x)
        return x.reshape(x.shape[0], -1)
'''

