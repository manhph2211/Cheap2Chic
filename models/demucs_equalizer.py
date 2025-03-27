import torch
import torch.nn as nn 
from models.core.htdemucs import HTDemucs
import torch
from tqdm import tqdm
import random


def gen(input_waveforms, target_waveforms, checkpoint_path='', gpu='cuda', batch_size=32, speaker='y1'):
    print("############ START GENERATING DATA ###########")

    model = StyleTransform2()
    state_dict = torch.load(checkpoint_path, map_location=gpu)
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(gpu)

    gen_waveforms = []
    num_samples = len(input_waveforms)

    gen_waveforms.extend(target_waveforms[:2125])    
    for i in tqdm(range(2125, num_samples, batch_size)):
        em_idxes =  [random.randint(1, 30) for _ in range(batch_size)]
        text_emb = [torch.load(f"assets/embeddings/{speaker}/{speaker}_{em_idx}.pt", weights_only=True)[0].cpu() for em_idx in em_idxes]
        batch = torch.tensor(input_waveforms[i:i + batch_size]).to(gpu)
        em_batch = torch.tensor(text_emb).to(gpu)
        with torch.no_grad():
            output = model(batch, em_batch).squeeze(1).cpu().numpy()
        gen_waveforms.extend(output)

    return gen_waveforms

# def gen(input_waveforms, target_waveforms, checkpoint_path='', gpu='cuda', batch_size=32):
#     print("############ START GENERATING DATA ###########")

#     model = DemucsEqualizer(device=gpu)
#     state_dict = torch.load(checkpoint_path, map_location=gpu)
#     state_dict = {k[6:]: v for k, v in state_dict.items()}
#     model.load_state_dict(state_dict)
#     model.eval()
#     model.to(gpu)

#     gen_waveforms = []
#     num_samples = len(input_waveforms)

#     # First 2127 samples: Directly append target waveforms
#     gen_waveforms.extend(target_waveforms[:2125])

#     # Process remaining samples in batches
#     for i in tqdm(range(2125, num_samples, batch_size)):
#         batch = torch.tensor(input_waveforms[i:i + batch_size]).to(gpu)
#         with torch.no_grad():
#             output = model(batch).squeeze(1).cpu().numpy()
#         gen_waveforms.extend(output)

#     return gen_waveforms


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
        x = x.mean(dim=1)#[:,0,:]  
        return x.reshape(x.shape[0], 2, -1)
    
class StyleTransform2(nn.Module):
    ''' FILM
    '''
    def __init__(self, device="cuda"):
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
        if text_emd is None or not isinstance(text_emd, torch.Tensor) or text_emd.nelement() == 0:
            x = self.model1(x)
        else:
            x = self.model1(x, self.s_film(text_emd), self.e_film(text_emd)) 
        x = x.mean(dim=1)#[:,0,:]  
        return x.reshape(x.shape[0], 2, -1) # stereo
    
    
class DemucsEqualizer(nn.Module):
    def __init__(self, device="cuda"):
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
        x = x.mean(dim=1)#[:,0,:]  
        return x.reshape(x.shape[0], 2, -1)


class DoubleDemucsEqualizer(nn.Module):
    def __init__(self, checkpoint_path, model_name="v1", device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.model1 = DemucsEqualizer(device=device) if model_name=='v1' else StyleTransform2(device=device)
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

    def forward(self, x, text_emd=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            x = torch.cat([x, x], dim=1)        
        x = self.model2(x)
        x = x.mean(dim=1)[:,0,:]  
        x = x.reshape(x.shape[0], -1)
        x = self.model1(x) if self.model_name=="v1" else self.model1(x, text_emd)
        return x
    