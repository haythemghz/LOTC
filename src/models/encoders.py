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


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for small images (e.g. MNIST 28x28)."""
    def __init__(self, in_channels: int = 1, output_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class ResNetEncoder(nn.Module):
    """
    ResNet backbone with a custom projection head for images.
    Supports resnet18 (512-dim) and resnet50 (2048-dim).
    """
    def __init__(self, output_dim: int = 128, pretrained: bool = True,
                 backbone: str = 'resnet18'):
        super().__init__()
        self.backbone_name = backbone

        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head: feat_dim → 512 → output_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class DINOViTEncoder(nn.Module):
    """
    DINO ViT-S/16 backbone with a projection head.
    Uses torch.hub to load the pretrained model from facebookresearch/dino.
    Produces 384-dim CLS token features → projected to output_dim.
    """
    def __init__(self, output_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dino:main', 'dino_vits16',
            pretrained=pretrained
        )
        # Freeze backbone by default (fine-tune head only)
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_dim = 384  # ViT-S/16 CLS token dimension

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)  # (B, 384)
        return self.head(features)


# ---- Encoder registry (for convenience) ----

ENCODERS = {
    'identity': IdentityEncoder,
    'mlp': MLPEncoder,
    'cnn': CNNEncoder,
    'resnet18': lambda output_dim, pretrained=True: ResNetEncoder(output_dim, pretrained, 'resnet18'),
    'resnet50': lambda output_dim, pretrained=True: ResNetEncoder(output_dim, pretrained, 'resnet50'),
    'resnet': lambda output_dim, pretrained=True: ResNetEncoder(output_dim, pretrained, 'resnet18'),
    'dino': lambda output_dim, pretrained=True: DINOViTEncoder(output_dim, pretrained),
}
