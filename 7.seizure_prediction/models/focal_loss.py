import torch.nn.functional as F
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None  # Convert to tensor
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)[targets]  # Now works because self.alpha is a tensor
            loss = alpha * loss
        return loss.mean()