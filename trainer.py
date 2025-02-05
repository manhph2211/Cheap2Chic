import torch
import torch.nn as nn 
from loss import STFTLoss
import torch.nn.functional as F


STFT_loss = STFTLoss()


def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids) if not isinstance(pred.label_ids, torch.Tensor) else pred.label_ids
    preds = torch.tensor(pred.predictions) if not isinstance(pred.predictions, torch.Tensor) else pred.predictions
    
    # labels = labels.to('cpu')
    # preds = preds.to('cpu')
    # stft = STFT_loss(labels, preds)
    mse = ((preds - labels) ** 2).mean() 
    return {"mse": mse.item()}


# class TrainerModelWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.mse_loss_fn = nn.MSELoss()  

#     def forward(self, input_values, labels=None):
#         outputs = self.model(input_values)
#         loss = self.mse_loss_fn(outputs, labels) if labels is not None else None
#         return (loss, outputs) if loss is not None else outputs

class TrainerModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mse_loss_fn = nn.MSELoss()

    def fconv(self, x):
        x = self.model.model1._spec(x)
        x = self.model.model1._magnitude(x)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        return x

    def forward(self, input_values, labels=None, text_emb=None):
        outputs = self.model(input_values, text_emb)
        if labels is not None:
            mse_loss = self.mse_loss_fn(outputs, labels)
            # stft_loss = STFT_loss(labels, outputs)

            # target_mag = torch.abs(labels)
            # pred_mag = torch.abs(outputs)
            # stft_loss = F.l1_loss(self.fconv(pred_mag), self.fconv(target_mag))
            # loss = mse_loss 
            return (mse_loss, outputs)
        return outputs
