import torch
import torch.nn as nn 
from loss import STFTLoss


STFT_loss = STFTLoss()


def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids) if not isinstance(pred.label_ids, torch.Tensor) else pred.label_ids
    preds = torch.tensor(pred.predictions) if not isinstance(pred.predictions, torch.Tensor) else pred.predictions
    
    labels = labels.to('cpu')
    preds = preds.to('cpu')
    
    # stft = STFT_loss(labels, preds)
    mse = ((preds - labels) ** 2).mean() #+ stft.item()
    return {"mse": mse.item()}


class TrainerModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mse_loss_fn = nn.MSELoss()  

    def forward(self, input_values, labels=None):
        outputs = self.model(input_values)
        loss = self.mse_loss_fn(outputs, labels) if labels is not None else None
        return (loss, outputs) if loss is not None else outputs

