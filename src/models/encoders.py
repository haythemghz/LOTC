import torch
import torch.nn as nn
import torchvision.models as models

class IdentityEncoder(nn.Module):
    """Passthrough encoder for datasets already in a suitable feature space."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron for tabular data or flattened images.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (e.g. for MNIST images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

class ResNetEncoder(nn.Module):
    """
    ResNet-18 backbone with a custom projection head for images.
    """
    def __init__(self, output_dim: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection head (from 512 to output_dim)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
