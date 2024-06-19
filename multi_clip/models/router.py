import torch
import torch.nn as nn
    
class Router(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int) -> None:
        super(Router, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)