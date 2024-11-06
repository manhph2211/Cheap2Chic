import torch
import torch.nn as nn 


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mse = ((preds - labels) ** 2).mean()    
    return {"mse": mse}


class TrainerModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mse_loss_fn = nn.MSELoss()  

    def forward(self, input_values, labels=None):
        outputs = self.model(input_values)
        loss = self.mse_loss_fn(outputs, labels) if labels is not None else None
        return (loss, outputs) if loss is not None else outputs

