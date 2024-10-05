import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class CNNCrossConv(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, out_channels=6, num_layers=2):
        super(CNNCrossConv, self).__init__()
        layers = []
        # Input layer
        layers.append(CrossConv2d(in_channels, hidden_channels))
        layers.append(nn.Tanh())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(CrossConv2d(hidden_channels, hidden_channels))
            layers.append(nn.LeakyReLU())
        # Output layer
        layers.append(CrossConv2d(hidden_channels, out_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, in_channels, height, width]
        """
        return self.network(x)

    def to(self, device):
        """
        Moves the model to the specified device and returns self.
        """
        return super(CNNCrossConv, self).to(device)


class CrossConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CrossConv2d, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv = weight_norm(conv, name="weight")

    def forward(self, x):
        return self.conv(x)


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a sample input tensor
    x = torch.randn((1, 6, 32, 32))

    # Initialize the model
    model = CNNCrossConv(
        in_channels=6, hidden_channels=32, out_channels=6, num_layers=3
    )

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
