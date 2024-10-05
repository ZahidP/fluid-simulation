import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm


import torch
from torch_geometric.data import Data


def angle_to_sin_cos(angle_attr):
    """
    Converts angles in degrees to sine and cosine components.

    Args:
        angle_attr (Tensor): Angles in degrees.

    Returns:
        Tuple[Tensor, Tensor]: Sine and cosine of the angles.
    """
    radians = torch.deg2rad(angle_attr)
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    return sin, cos


def create_grid_graph_with_angles(width, height):
    """
    Creates a 2D grid graph with neighbors up to 2 hops away and angular edge attributes.

    Args:
        width (int): Number of columns in the grid.
        height (int): Number of rows in the grid.

    Returns:
        Data: PyTorch Geometric Data object containing edge_index, edge_attr, and edge_distance.
    """
    edge_index = []
    edge_attr = []
    edge_distance = []

    # Direction vectors and their corresponding angles in degrees for distance 1
    directions = {
        "E": (1, 0, 0),  # East
        "NE": (1, 1, 45),  # Northeast
        "N": (0, 1, 90),  # North
        "NW": (-1, 1, 135),  # Northwest
        "W": (-1, 0, 180),  # West
        "SW": (-1, -1, 225),  # Southwest
        "S": (0, -1, 270),  # South
        "SE": (1, -1, 315),  # Southeast
    }

    # Directions for distance 2 (we'll include additional positions)
    directions_distance_2 = {
        # 'E2': (2, 0, 0),
        # 'N2': (0, 2, 90),
        # 'W2': (-2, 0, 180),
        # 'S2': (0, -2, 270),
        # 'NE2': (2, 2, 45),
        # 'NW2': (-2, 2, 135),
        # 'SW2': (-2, -2, 225),
        # 'SE2': (2, -2, 315),
        # 'E_N': (1, 2, 63.4349),
        # 'N_E': (2, 1, 26.5651),
        # 'N_W': (-2, 1, 153.4349),
        # 'W_N': (-1, 2, 116.5651),
        # 'W_S': (-1, -2, 243.4349),
        # 'S_W': (-2, -1, 206.5651),
        # 'S_E': (2, -1, 333.4349),
        # 'E_S': (1, -2, 296.5651)
    }

    for y in range(height):
        for x in range(width):
            node = y * width + x

            # Distance 1 edges
            for dir_name, (dx, dy, angle) in directions.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    edge_index.append([node, neighbor])
                    edge_attr.append(angle)
                    edge_distance.append(1)

            # Distance 2 edges
            for dir_name, (dx, dy, angle) in directions_distance_2.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    edge_index.append([node, neighbor])
                    edge_attr.append(angle)
                    edge_distance.append(2)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Use float for angles
    edge_distance = torch.tensor(edge_distance, dtype=torch.long)  # Distance attribute
    return edge_index, edge_attr, edge_distance
    # data = Data(edge_index=edge_index, edge_attr=edge_attr)
    # data.num_nodes = width * height
    # data.edge_distance = edge_distance
    # return data


class ComplexGNNLayerWithAngles(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_directions=8,
        use_angle=True,
        use_target_node_feat=False,
        final_layer=False,
    ):
        super(ComplexGNNLayerWithAngles, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_directions = num_directions
        self.use_angle = use_angle
        self.use_target_node_feat = use_target_node_feat
        self.final_layer = final_layer

        edge_attr_dim = 2 if self.use_angle else 0
        edge_distance_dim = 1

        if self.use_target_node_feat:
            input_features = 2 * in_channels + edge_attr_dim + edge_distance_dim
        else:
            input_features = in_channels + edge_attr_dim + edge_distance_dim

        num_layers = 4
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Linear(input_features, input_features))
                for _ in range(num_layers - 1)
            ]
        )

        self.layers.append(weight_norm(nn.Linear(input_features, out_channels)))
        self.agg = weight_norm(nn.Linear(out_channels, out_channels))
        self.activation = nn.GELU()

        # Attention mechanism
        self.attention = weight_norm(nn.Linear(out_channels, 1))

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr, edge_distance):
        return (
            self.propagate(
                edge_index, x=x, edge_attr=edge_attr, edge_distance=edge_distance
            )
            + self.bias
        )

    def message(self, x_j, edge_attr, x_i, edge_distance):
        features = [x_j]
        edge_features = []

        if self.use_target_node_feat:
            features.append(x_i)

        if self.use_angle:
            sin, cos = angle_to_sin_cos(edge_attr)
            angle_features = torch.stack([sin, cos], dim=1)
            edge_features.append(angle_features)

        edge_distance = edge_distance.unsqueeze(1).float()
        edge_features.append(edge_distance)

        features = features + edge_features

        combined = torch.cat(features, dim=1)

        result = combined
        for layer in self.layers:
            result = layer(result)
        if not self.final_layer:
            result = self.activation(result)
        return result

    # def aggregate(self, inputs, index, dim_size=None):
    #     # Compute attention weights for all inputs at once
    #     attention_weights = self.attention(inputs).squeeze(-1)

    #     # Create a mask for each target node
    #     unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
    #     mask = F.one_hot(inverse_indices, num_classes=unique_indices.size(0)).to(inputs.dtype)

    #     # Apply softmax to attention weights for each target node
    #     masked_attention = attention_weights.unsqueeze(0) * mask.unsqueeze(-1)
    #     softmax_attention = F.softmax(masked_attention, dim=1)

    #     # Apply attention weights to inputs
    #     weighted_inputs = inputs * softmax_attention.transpose(0, 1)

    #     # Aggregate using scatter_add
    #     output = torch.zeros(dim_size, inputs.size(-1), device=inputs.device)
    #     output.scatter_add_(0, index.unsqueeze(-1).expand_as(weighted_inputs), weighted_inputs)

    #     return output

    def update(self, aggr_out):
        # print('update shape')
        # print(aggr_out.shape)
        return self.agg(aggr_out)


class DirectionalGNNLayerWithAngles(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_directions=8,
        use_angle=True,
        use_target_node_feat=False,
        final_layer=False,
    ):
        super(DirectionalGNNLayerWithAngles, self).__init__(
            aggr="mean"
        )  # Aggregation can be 'add', 'mean', etc.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_directions = num_directions
        self.use_angle = use_angle
        self.use_target_node_feat = use_target_node_feat

        # Edge attribute dimensions
        edge_attr_dim = 2 if self.use_angle else 0  # For sin and cos components
        edge_distance_dim = 1  # For edge distance

        # Calculate input features based on the options
        if self.use_target_node_feat:
            input_features = 2 * in_channels + edge_attr_dim + edge_distance_dim
        else:
            input_features = in_channels + edge_attr_dim + edge_distance_dim

        # Define a linear transformation for combined node features and edge attributes
        num_layers = 5
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Linear(input_features, input_features), name="weight")
                for _ in range(num_layers - 1)
            ]
        )

        # Add the final layer that outputs to out_channels
        self.layers.append(nn.Linear(input_features, out_channels))
        self.agg = nn.Linear(out_channels, out_channels)
        self.activation = nn.GELU()
        self.final_layer = final_layer

        # Optional: Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr, edge_distance):
        """
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges]
            edge_attr (Tensor): Edge angle attributes in degrees with shape [num_edges]
            edge_distance (Tensor): Edge distance attributes with shape [num_edges]
        Returns:
            Tensor: Updated node feature matrix
        """
        return (
            self.propagate(
                edge_index, x=x, edge_attr=edge_attr, edge_distance=edge_distance
            )
            + self.bias
        )

    def message(self, x_j, edge_attr, x_i, edge_distance):
        """
        Args:
            x_j (Tensor): Source node features for each edge
            x_i (Tensor): Target node features for each edge
            edge_attr (Tensor): Edge angle attributes in degrees
            edge_distance (Tensor): Edge distance attributes
        Returns:
            Tensor: Messages for each edge
        """
        features = [x_j]  # Start with source node features

        edge_features = []

        if self.use_target_node_feat:
            features.append(x_i)  # Include target node features

        if self.use_angle:
            sin, cos = angle_to_sin_cos(edge_attr)
            angle_features = torch.stack([sin, cos], dim=1)
            edge_features.append(angle_features)  # Include angle features

        # Include edge_distance
        edge_distance = edge_distance.unsqueeze(1).float()  # Shape: [num_edges, 1]
        edge_features.append(edge_distance)

        features = features + edge_features

        # Concatenate all features
        combined = torch.cat(features, dim=1)

        # Apply linear transformation
        result = combined
        for layer in self.layers:
            result = layer(result)
        if not self.final_layer:
            result = self.activation(result)
        return result

    def update(self, aggr_out):
        """
        Args:
            aggr_out (Tensor): Aggregated messages for each node
        Returns:
            Tensor: Updated node features
        """
        aggr_out = self.agg(aggr_out)
        return aggr_out


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6
    ):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Updated GNN model with multiple layers and optional target node features
class GridGNNWithAngles(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_directions=8,
        num_layers=2,
        use_angle=True,
        use_target_node_feat=False,
    ):
        super(GridGNNWithAngles, self).__init__()
        self.num_layers = num_layers
        self.use_angle = use_angle
        self.use_target_node_feat = use_target_node_feat
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            DirectionalGNNLayerWithAngles(
                in_channels,
                hidden_channels,
                num_directions,
                use_angle=use_angle,
                use_target_node_feat=use_target_node_feat,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                DirectionalGNNLayerWithAngles(
                    hidden_channels,
                    hidden_channels,
                    num_directions,
                    use_angle=use_angle,
                    use_target_node_feat=use_target_node_feat,
                )
            )

        # Output layer
        self.layers.append(
            DirectionalGNNLayerWithAngles(
                hidden_channels,
                out_channels,
                num_directions,
                use_angle=use_angle,
                use_target_node_feat=use_target_node_feat,
                final_layer=True,
            )
        )

    def forward(self, x, edge_index, edge_attr, edge_distance):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr, edge_distance)
            if i < self.num_layers - 1:
                x = F.tanh(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=6, num_layers=2):
        super(CNN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.GELU())
        # Output layer
        layers.append(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, in_channels, height, width]
        """
        return self.network(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossConv2d, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 5) / (in_channels * 5) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch, in_channels, height, width = x.shape
        out_channels = self.weight.shape[0]

        # Create the cross-shaped kernel
        kernel = torch.zeros(out_channels, in_channels, 3, 3, device=x.device)
        kernel[:, :, 1, 1] = self.weight[:, :, 0]  # Center
        kernel[:, :, 0, 1] = self.weight[:, :, 1]  # Top
        kernel[:, :, 2, 1] = self.weight[:, :, 2]  # Bottom
        kernel[:, :, 1, 0] = self.weight[:, :, 3]  # Left
        kernel[:, :, 1, 2] = self.weight[:, :, 4]  # Right

        return F.conv2d(x, kernel, self.bias, padding=1)


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
            layers.append(nn.Tanh())
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


class LightweightUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, hidden_channels=64):
        super(LightweightUNet, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, hidden_channels)
        self.enc2 = self.conv_block(hidden_channels, hidden_channels * 2)
        self.enc3 = self.conv_block(hidden_channels * 2, hidden_channels * 4)

        # Decoder (Upsampling)
        self.dec3 = self.conv_block(hidden_channels * 4, hidden_channels * 2)
        self.dec2 = self.conv_block(hidden_channels * 4, hidden_channels)
        self.dec1 = self.conv_block(hidden_channels * 2, hidden_channels)

        self.final = CrossConv2d(hidden_channels, out_channels)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            CrossConv2d(in_channels, out_channels),
            nn.ReLU(inplace=True),
            CrossConv2d(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Decoder
        dec3 = self.dec3(self.up(enc3))
        dec2 = self.dec2(torch.cat([self.up(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2), enc1], dim=1))

        return self.final(dec1)

    def to(self, device):
        return super(LightweightUNet, self).to(device)


# class CNN(nn.Module):
#     def __init__(self, in_channels=6, hidden_channels=64, out_channels=6, num_layers=4):
#         super(CNN, self).__init__()
#         layers = []
#         # Input layer
#         layers.append(CrossConv2d(in_channels, hidden_channels))
#         layers.append(nn.ReLU(inplace=True))
#         # Hidden layers
#         for _ in range(num_layers - 2):
#             layers.append(CrossConv2d(hidden_channels, hidden_channels))
#             layers.append(nn.ReLU(inplace=True))
#         # Output layer
#         layers.append(CrossConv2d(hidden_channels, out_channels))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         x: Input tensor of shape [batch_size, in_channels, height, width]
#         """
#         return self.network(x)

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import math

# Utility function to convert angles to sine and cosine
def angle_to_sin_cos(angle_deg):
    """
    Converts angle in degrees to sine and cosine components.

    Args:
        angle_deg (Tensor): Tensor of angles in degrees.

    Returns:
        Tuple[Tensor, Tensor]: Sine and cosine of the angles.
    """
    angle_rad = angle_deg * math.pi / 180.0  # Convert to radians
    sin = torch.sin(angle_rad)
    cos = torch.cos(angle_rad)
    return sin, cos


import torch
import torch.nn as nn


class CrossConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CrossConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class LightweightUNetCrossConv(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=6, hidden_channels=64, *args, **kwargs
    ):
        super(LightweightUNetCrossConv, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, hidden_channels)
        self.enc2 = self.conv_block(hidden_channels, hidden_channels * 2)
        self.enc3 = self.conv_block(hidden_channels * 2, hidden_channels * 4)

        # Decoder (Upsampling)
        self.dec3 = self.conv_block(hidden_channels * 4, hidden_channels * 2)
        self.dec2 = self.conv_block(hidden_channels * 4, hidden_channels)
        self.dec1 = self.conv_block(hidden_channels * 2, hidden_channels)

        self.final = CrossConv2d(hidden_channels, out_channels)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            CrossConv2d(in_channels, out_channels),
            nn.ReLU(inplace=True),
            CrossConv2d(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Decoder
        dec3 = self.dec3(self.up(enc3))
        dec2 = self.dec2(torch.cat([self.up(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2), enc1], dim=1))

        return self.final(dec1)

    def to(self, device):
        return super(LightweightUNet, self).to(device)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class LightweightUNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, features=[12, 24, 48], *args, **kwargs
    ):
        super(LightweightUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
