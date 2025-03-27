import torch
import torch.nn as nn 
from loss import STFTLoss
import torch.nn.functional as F


def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids) if not isinstance(pred.label_ids, torch.Tensor) else pred.label_ids
    preds = torch.tensor(pred.predictions) if not isinstance(pred.predictions, torch.Tensor) else pred.predictions
    mse = ((preds - labels) ** 2).mean() 
    return {"mse": mse.item()}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


class TrainerModelWrapper(nn.Module):
    def __init__(self, model, version='v2'):
        super().__init__()
        self.model = model
        self.mse_loss_fn = nn.MSELoss()
        self.version = version

    def forward(self, input_values, labels=None, text_emb=None):
        outputs = self.model(input_values, text_emb) if self.version == 'v2' else self.model(input_values)
        if labels is not None:
            mse_loss = self.mse_loss_fn(outputs, labels)
            return (mse_loss, outputs)
        return outputs
