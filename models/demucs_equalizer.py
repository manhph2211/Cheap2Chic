import torch
import torchaudio
import torch.nn as nn 
from models.core.htdemucs import HTDemucs

class Pooler(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.dense = nn.Linear(4096, hidden_size)
        self.activation = nn.Sigmoid()#nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        
        pooled_output = pooled_output.unsqueeze(-1).unsqueeze(-1)          
        return pooled_output


class FiLM_Modulation(nn.Module):
    def __init__(self, feature_dim=4, condition_dim=4096):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim) 
        self.beta_fc = nn.Linear(condition_dim, feature_dim)   

    def forward(self, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # Shape: (B, C, 1, 1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)    # Shape: (B, C, 1, 1)
        return beta, gamma


class StyleTransform1(nn.Module):
    ''' Pooling and Weighting
    '''
    def __init__(self, freeze=False, model_name="htdemucs", device="cuda"):
        super().__init__()
        pkg = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th", map_location='cpu', check_hash=True) 
        self.model1 = HTDemucs(**pkg["kwargs"])
        self.model1.load_state_dict(pkg["state"])
        self.model1 = self.model1.to(device)
        self.pool = Pooler().to(device)

    def forward(self, x, text_emd=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            x = torch.cat([x, x], dim=1)
        x = self.model1(x) 
        x = self.pool(text_emd).expand_as(x) * x
        x = x.mean(dim=1)[:,0,:]  
        return x.reshape(x.shape[0], -1)
    
class StyleTransform2(nn.Module):
    ''' FILM
    '''
    def __init__(self, freeze=False, model_name="htdemucs", device="cuda"):
        super().__init__()
        pkg = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th", map_location='cpu', check_hash=True) 
        self.model1 = HTDemucs(**pkg["kwargs"])
        self.model1.load_state_dict(pkg["state"])
        self.model1 = self.model1.to(device)
        self.s_film = FiLM_Modulation().to(device)
        self.e_film = FiLM_Modulation().to(device)

    def forward(self, x, text_emd=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            x = torch.cat([x, x], dim=1)
        x = self.model1(x, self.s_film(text_emd), self.e_film(text_emd)) 
        x = x.mean(dim=1)#[:,0,:]  
        return x.reshape(x.shape[0], 2, -1) # stereo
    
    
class DemucsEqualizer(nn.Module):
    def __init__(self, freeze=False, model_name="htdemucs", device="cuda"):
        super().__init__()
        pkg = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th", map_location='cpu', check_hash=True) 
        self.model1 = HTDemucs(**pkg["kwargs"])
        self.model1.load_state_dict(pkg["state"])
        self.model1 = self.model1.to(device)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            x = torch.cat([x, x], dim=1)
        x = self.model1(x) 
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
        
        pkg = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th", map_location='cpu', check_hash=True) 
        self.model2 = HTDemucs(**pkg["kwargs"])
        self.model2.load_state_dict(pkg["state"])
        self.model2 = self.model2.to(device)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = torch.cat([x, x], dim=1)        
        x = self.model2(x)
        x = x.mean(dim=1)[:,0,:]  
        x = x.reshape(x.shape[0], -1)
        x = self.model1(x)
        return x
    
    
