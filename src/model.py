import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DeepfakeDetector(nn.Module):
    """ResNet-18 based binary classifier for audio deepfake detection."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet-18 with optional ImageNet weights
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)

        # Modify first conv to accept 1 channel (mono spectrogram) instead of 3 (RGB)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # If pretrained, copy the mean of RGB weights to the single channel
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Replace the final FC layer for binary classification
        # ResNet-18 has 512 features at the final FC layer
        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, x):
        """
        x: (batch, 1, n_mels, time_frames)
        returns: (batch,) raw logits for binary classification
        """
        return self.backbone(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Creating model...")
    model = DeepfakeDetector(pretrained=True)
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Test forward pass with dummy input matching our dataset
    dummy_input = torch.randn(4, 1, 128, 126)  # batch=4, 1 channel, 128 mels, 126 frames
    print(f"\nInput shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values (raw logits): {output.tolist()}")
    print(f"Output as probabilities: {torch.sigmoid(output).tolist()}")

    print("\nModel ready for training.")