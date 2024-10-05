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


# Updated grid graph construction with angles
def create_grid_graph_with_angles(width, height):
    """
    Creates a 2D grid graph with 8-connected neighbors and angular edge attributes.

    Args:
        width (int): Number of columns in the grid.
        height (int): Number of rows in the grid.

    Returns:
        Data: PyTorch Geometric Data object containing edge_index and edge_attr.
    """
    edge_index = []
    edge_attr = []

    # Direction vectors and their corresponding angles in degrees
    directions = {
        "E": 0,  # East
        "NE": 45,  # Northeast
        "N": 90,  # North
        "NW": 135,  # Northwest
        "W": 180,  # West
        "SW": 225,  # Southwest
        "S": 270,  # South
        "SE": 315,  # Southeast
    }

    # Define movement offsets
    direction_offsets = {
        "E": (1, 0),
        "NE": (1, 1),
        "N": (0, 1),
        "NW": (-1, 1),
        "W": (-1, 0),
        "SW": (-1, -1),
        "S": (0, -1),
        "SE": (1, -1),
    }

    for y in range(height):
        for x in range(width):
            node = y * width + x
            for dir_name, angle in directions.items():
                dx, dy = direction_offsets[dir_name]
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    edge_index.append([node, neighbor])
                    edge_attr.append(angle)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Use float for angles

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = width * height
    return data


# Updated GNN layer with angle embedding
class DirectionalGNNLayerWithAngles(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_directions=8,
        use_angle=True,
        final_layer=False,
        use_tgt_feat=False,
    ):
        super(DirectionalGNNLayerWithAngles, self).__init__(
            aggr="add"
        )  # Aggregation can be 'add', 'mean', etc.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_directions = num_directions
        self.use_angle = use_angle
        self.use_tgt_feat = use_tgt_feat

        # If using angles, we'll add sine and cosine components (2 additional features)
        if use_tgt_feat:
            input_features = 2 * in_channels + 2 if use_angle else 2 * in_channels
        else:
            input_features = in_channels + 2 if use_angle else in_channels

        # Define a linear transformation for combined node features and angles
        self.lin = nn.Linear(input_features, out_channels)
        self.activation = nn.GELU()
        self.final_layer = final_layer

        # Optional: Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges]
            edge_attr (Tensor): Edge angle attributes in degrees with shape [num_edges]
        Returns:
            Tensor: Updated node feature matrix
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.bias

    def message(self, x_j, edge_attr):
        """
        Args:
            x_j (Tensor): Source node features for each edge
            edge_attr (Tensor): Edge angle attributes in degrees
        Returns:
            Tensor: Messages for each edge
        """
        if self.use_angle:
            sin, cos = angle_to_sin_cos(edge_attr)
            # Stack sin and cos to form [num_edges, 2]
            angle_features = torch.stack([sin, cos], dim=1)
            # Concatenate node features with angle features
            if self.use_tgt_feat:
                combined = torch.cat([x_j, x_i, angle_features], dim=1)
            else:
                combined = torch.cat([x_j, angle_features], dim=1)
        else:
            combined = x_j

        # Apply linear transformation
        result = self.lin(combined)
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
        return aggr_out


# Updated GNN model with multiple layers
class GridGNNWithAngles(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_directions=8,
        num_layers=2,
        use_angle=True,
        use_tgt_features=False,
    ):
        super(GridGNNWithAngles, self).__init__()
        self.num_layers = num_layers
        self.use_angle = use_angle
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            DirectionalGNNLayerWithAngles(
                in_channels, hidden_channels, num_directions, use_angle=use_angle
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
                )
            )

        # Output layer
        self.layers.append(
            DirectionalGNNLayerWithAngles(
                hidden_channels,
                out_channels,
                num_directions,
                use_angle=use_angle,
                final_layer=True,
            )
        )

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=6, num_layers=4):
        super(CNN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.LeakyRELU(inplace=True))
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


class CNN(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=6, num_layers=4):
        super(CNN, self).__init__()
        layers = []
        # Input layer
        layers.append(CrossConv2d(in_channels, hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(CrossConv2d(hidden_channels, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
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
        return super(CNN, self).to(device)


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


# Updated grid graph construction with angles
def create_grid_graph_with_angles(width, height):
    """
    Creates a 2D grid graph with 8-connected neighbors and angular edge attributes.

    Args:
        width (int): Number of columns in the grid.
        height (int): Number of rows in the grid.

    Returns:
        Data: PyTorch Geometric Data object containing edge_index and edge_attr.
    """
    edge_index = []
    edge_attr = []

    # Direction vectors and their corresponding angles in degrees
    directions = {
        "E": 0,  # East
        "NE": 45,  # Northeast
        "N": 90,  # North
        "NW": 135,  # Northwest
        "W": 180,  # West
        "SW": 225,  # Southwest
        "S": 270,  # South
        "SE": 315,  # Southeast
    }

    # Define movement offsets
    direction_offsets = {
        "E": (1, 0),
        "NE": (1, 1),
        "N": (0, 1),
        "NW": (-1, 1),
        "W": (-1, 0),
        "SW": (-1, -1),
        "S": (0, -1),
        "SE": (1, -1),
    }

    for y in range(height):
        for x in range(width):
            node = y * width + x
            for dir_name, angle in directions.items():
                dx, dy = direction_offsets[dir_name]
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    edge_index.append([node, neighbor])
                    edge_attr.append(angle)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Use float for angles

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = width * height
    return data


# Updated GNN layer with angle embedding
class DirectionalGNNLayerWithAngles(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_directions=8,
        use_angle=True,
        final_layer=False,
    ):
        super(DirectionalGNNLayerWithAngles, self).__init__(
            aggr="add"
        )  # Aggregation can be 'add', 'mean', etc.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_directions = num_directions
        self.use_angle = use_angle

        # If using angles, we'll add sine and cosine components (2 additional features)
        input_features = in_channels + 2 if use_angle else in_channels

        # Define a linear transformation for combined node features and angles
        self.lin = nn.Linear(input_features, out_channels)
        self.tanh = nn.Tanh()
        self.final_layer = final_layer

        # Optional: Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges]
            edge_attr (Tensor): Edge angle attributes in degrees with shape [num_edges]
        Returns:
            Tensor: Updated node feature matrix
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.bias

    def message(self, x_j, edge_attr):
        """
        Args:
            x_j (Tensor): Source node features for each edge
            edge_attr (Tensor): Edge angle attributes in degrees
        Returns:
            Tensor: Messages for each edge
        """
        if self.use_angle:
            sin, cos = angle_to_sin_cos(edge_attr)
            # Stack sin and cos to form [num_edges, 2]
            angle_features = torch.stack([sin, cos], dim=1)
            # Concatenate node features with angle features
            combined = torch.cat([x_j, angle_features], dim=1)
        else:
            combined = x_j

        # Apply linear transformation
        result = self.lin(combined)
        if not self.final_layer:
            result = self.tanh(result)
        return result

    def update(self, aggr_out):
        """
        Args:
            aggr_out (Tensor): Aggregated messages for each node
        Returns:
            Tensor: Updated node features
        """
        return aggr_out


# Updated GNN model with multiple layers
class GridGNNWithAngles(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_directions=8,
        num_layers=2,
        use_angle=True,
    ):
        super(GridGNNWithAngles, self).__init__()
        self.num_layers = num_layers
        self.use_angle = use_angle
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            DirectionalGNNLayerWithAngles(
                in_channels, hidden_channels, num_directions, use_angle=use_angle
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
                )
            )

        # Output layer
        self.layers.append(
            DirectionalGNNLayerWithAngles(
                hidden_channels,
                out_channels,
                num_directions,
                use_angle=use_angle,
                final_layer=True,
            )
        )

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x
