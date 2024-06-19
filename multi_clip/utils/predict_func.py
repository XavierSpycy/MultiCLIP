import torch

def default_compute_predict_proba(model_outputs: torch.Tensor):
    return torch.sigmoid(model_outputs)