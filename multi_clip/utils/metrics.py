import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def f1_score_micro(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='micro')

def f1_score_macro(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='macro')

def f1_score_weighted(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='weighted')

def precision_score_binary(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    y_true = y_true.detach().cpu().numpy()
    return precision_score(y_true, y_pred)

def recall_score_binary(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    y_true = y_true.detach().cpu().numpy()
    return recall_score(y_true, y_pred)