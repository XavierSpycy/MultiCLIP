import torch
import torch.nn.functional as F

def compute_mean_squared_error_loss(outputs, labels, reduction='mean'):
    loss = F.mse_loss(outputs, labels, reduction=reduction)
    return loss

def compute_binary_cross_entropy_loss_with_logits(outputs, labels, reduction='mean'):
    loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction=reduction)
    return loss

def compute_binary_cross_entropy_loss(outputs, labels, reduction='mean'):
    loss = F.binary_cross_entropy(outputs, labels, reduction=reduction)
    return loss

def compute_smooth_loss(outputs, labels, smoothing=0.0, reduction='mean'):
    assert 0 <= smoothing < 1
    labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction=reduction)
    return loss

def compute_binary_focal_loss_with_logits(outputs, labels, alpha=0.25, gamma=2.0, reduction='mean'):
    BCE_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
    labels = labels.type(torch.float32)
    at = alpha * labels + (1 - alpha) * (1 - labels)
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1-pt) ** gamma * BCE_loss

    if reduction == 'mean':
        return torch.mean(F_loss)
    elif reduction == 'sum':
        return torch.sum(F_loss)
    else:
        return F_loss

def compute_angular_additive_margin_loss_with_logits(outputs, labels, s=30.0, m=0.50, reduction='mean'):
    phi = outputs - m
    outputs = labels * phi + (1 - labels) * outputs
    outputs *= s
    loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction=reduction)
    return loss

def compute_zlpr_loss_with_logits(outputs, labels, reduction='mean'):
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_term = torch.exp(-outputs[pos_mask])
    neg_term = torch.exp(outputs[neg_mask])
    
    loss_pos = torch.log(1 + pos_term.sum())
    loss_neg = torch.log(1 + neg_term.sum())

    loss = loss_pos + loss_neg
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss