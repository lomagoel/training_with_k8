import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Instantiates a vision model adapted for CIFAR-10."""
    if architecture.lower() == "resnet18":
        # Load ResNet-18 without pre-trained weights for cold training
        model = resnet18(weights=None)

        # Adapt initial convolution for CIFAR-10 32x32 image input size
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()  # Remove maxpool to retain spatial dimensions

        # Replace classification head
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")