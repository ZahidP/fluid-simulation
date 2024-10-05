import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMessagePassing(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super(AttentionMessagePassing, self).__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.attention = nn.Linear(3 * hidden_dim, 1)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        row, col = edge_index

        # Project node and edge features
        x_proj = self.node_proj(x)
        # print('edge_attr')
        # print(edge_attr.shape)
        edge_proj = self.edge_proj(edge_attr)

        # Compute attention weights
        alpha = self.attention(torch.cat([x_proj[row], x_proj[col], edge_proj], dim=-1))
        alpha = F.softmax(alpha, dim=0)

        print("alpha")
        print(alpha)

        # Compute messages
        messages = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.message_mlp(messages)

        # Apply attention and aggregate messages
        weighted_messages = alpha * messages
        output = torch.zeros_like(x)
        # The index_add_ operation then aggregates messages only
        # to the target nodes (col) of these existing edges.
        output.index_add_(0, col, weighted_messages)
        print("forward output")
        print(output)
        return output


class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, use_gru=True):
        super(GNNLayer, self).__init__()
        self.message_passing = AttentionMessagePassing(
            node_dim, edge_dim, hidden_dim, out_dim
        )
        self.update = nn.GRUCell(node_dim, node_dim)
        self.use_gru = use_gru
        if use_gru:
            self.update = nn.GRUCell(node_dim, node_dim)
        else:
            self.update = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, node_dim),
            )
        self.reshape = nn.Linear(node_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        messages = self.message_passing(x, edge_index, edge_attr)
        if self.use_gru:
            output = self.update(messages, x)
        else:
            output = self.update(messages)
        output = self.reshape(output)
        return output


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, out_dim):
        super(GNN, self).__init__()
        # this is not a great way to do these layers, it makes a lot of assumptions
        self.layers = nn.ModuleList(
            [
                GNNLayer(node_dim, edge_dim, hidden_dim, out_dim=node_dim)
                for _ in range(num_layers)
            ]
        )
        self.layers[-1] = GNNLayer(node_dim, edge_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, edge_distance=None):
        if len(edge_attr.shape) < 2:
            edge_attr = edge_attr.unsqueeze(1)
        # we are throwing away edge_distance for now
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


if __name__ == "__main__":
    """

    Note the 0s in the forward output, 0, 3, and 9 are not target nodes
    so they have 0 messages.

    tensor([[4, 5, 6, 7, 9, 7, 9, 3, 1, 9],
            [5, 4, 6, 8, 6, 5, 8, 7, 1, 2]])
    alpha
    tensor([[0.0851],
            [0.1065],
            [0.0949],
            [0.0881],
            [0.1054],
            [0.0652],
            [0.1159],
            [0.1502],
            [0.0610],
            [0.1276]], grad_fn=<SoftmaxBackward0>)
    forward output
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0194,  0.0095, -0.0047, -0.0033],
            [-0.0042, -0.0093, -0.0098, -0.0133],
            [ 0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0344, -0.0069,  0.0106, -0.0130],
            [ 0.0059, -0.0011, -0.0337,  0.0334],
            [-0.0494,  0.0011, -0.0043, -0.0423],
            [-0.0422, -0.0435,  0.0079, -0.0762],
            [-0.0359, -0.0356, -0.0186, -0.0031],
            [ 0.0000,  0.0000,  0.0000,  0.0000]], grad_fn=<IndexAddBackward0>)
    """
    # Example usage
    node_dim = 4
    edge_dim = 1
    hidden_dim = 12
    num_layers = 2
    out_dim = 4

    model = GNN(node_dim, edge_dim, hidden_dim, num_layers, out_dim)

    # Dummy data
    num_nodes = 10
    num_edges = 8
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    print(edge_index)

    # Forward pass
    output = model(x, edge_index, edge_attr)
    print(output.shape)  # Should be (num_nodes, node_dim)
