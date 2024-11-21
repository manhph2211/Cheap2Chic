import torch
import torchaudio
from denoiser import pretrained
import torch.nn as nn 
# from soundstream import from_pretrained
from speechbrain.pretrained import SepformerSeparation as separator


class STEqualizer(nn.Module):
    def __init__(self, freeze=False, model_name="speechbrain/sepformer-wham16k-enhancement", device="cuda"):
        super().__init__()
        self.device = device
        self.model = separator.from_hparams(source=model_name, savedir='pretrained_models/sepformer-wham16k-enhancement', run_opts={"device":device}).to(device)
        #from_pretrained().to(device)  #torch.hub.load(model_name, "soundstream_16khz").to(device)
        self.model.train()  
        for param in self.model.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        torch.cuda.empty_cache()  

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.reshape(x.shape[0], -1)
        x = self.model(x)

        return x.reshape(x.shape[0], -1)


class DoubleSTEqualizer(nn.Module):
    def __init__(self, checkpoint_path, model_name="speechbrain/sepformer-wham16k-enhancement", device="cuda"):
        super().__init__()
        self.model1 = STEqualizer(device=device)
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            self.model1.load_state_dict(state_dict)
        except:
            raise "STH WRONG: CHECKPOINT FOR MODEL 1"
        
        self.model1.eval()
        for name, param in self.model1.named_parameters():
            param.requires_grad = False

        self.model2 = separator.from_hparams(source=model_name, savedir='pretrained_models/sepformer-wham16k-enhancement', run_opts={"device":device}).to(device)
        self.model2.train()  
        for param in self.model2.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        if len(x.shape) != 2:
            x = x.reshape(x.shape[0], -1)        
        
        x = self.model2(x).reshape(x.shape[0], -1) 
        x = self.model1(x)
        return x.reshape(x.shape[0], -1)
