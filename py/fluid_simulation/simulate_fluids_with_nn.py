import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.nn import NNConv
from fluid_simulation.datasets_v2 import GridDatasetGNN, GridDatasetCNN
from fluid_simulation.models_v2 import CNN, GridGNNWithAngles
from fluid_simulation.gat.gat_torch_scatter import GNN
from fluid_simulation.utils import (
    create_grid_graph_with_angles,
    denormalize_z_score,
    denormalize_specific_channels,
)
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

torch.manual_seed(8)
np.random.seed(8)

# Set device
device = "cpu"  # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
gnn_device = "cpu"
print(f"Using device: {device}")

# -------------------------------
# Define Prediction Functions
# -------------------------------


def create_border_mask(H, W, device):
    mask = torch.zeros(1, H, W, device=device)
    mask[:, 0, :] = 1  # Top border
    mask[:, -1, :] = 1  # Bottom border
    mask[:, :, 0] = 1  # Left border
    mask[:, :, -1] = 1  # Right border
    return mask


def predict_with_cnn(cnn, features, H, W, use_deltas=False):
    with torch.no_grad():
        cnn_output = cnn(features)  # [1, C', H, W]

        # Create correct border mask
        border_mask = create_border_mask(H, W, device)

        # Append border information to CNN output
        is_fluid = torch.ones((1, 1, H, W)).to(device)
        if use_deltas:
            is_fluid = is_fluid * 0
        cnn_output_with_border = torch.cat(
            [cnn_output, is_fluid, border_mask.unsqueeze(1)], dim=1
        )  # [1, C'+1, H, W]
    return cnn_output_with_border


def predict_with_gnn(
    gnn, features, edge_index, edge_attr, edge_distance, device, H, W, use_deltas=False
):
    with torch.no_grad():
        # Flatten features for GNN
        x = (
            features[0].permute(1, 2, 0).reshape(-1, features.shape[1]).to(device)
        )  # [num_nodes, C]

        # Create GeoData object
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        data = GeoData(
            x=x, edge_index=edge_index, edge_attr=edge_attr, edge_distance=edge_distance
        )
        batch = Batch.from_data_list([data]).to(device)  # Batch size of 1
        # GNN Prediction
        gnn_output = gnn(
            batch.x, batch.edge_index, batch.edge_attr, batch.edge_distance
        )  # [num_nodes, C']

        # Reshape to grid format
        gnn_output_grid = (
            gnn_output.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)
        )  # [1, C', H, W]
        border_mask = create_border_mask(H, W, device)

        # Append border information to CNN output
        is_fluid = torch.ones((1, 1, H, W)).to(device)
        if use_deltas:
            is_fluid = is_fluid * 0
        gnn_output_with_border = torch.cat(
            [gnn_output_grid, is_fluid, border_mask.unsqueeze(1)], dim=1
        )  # [1, C'+1, H, W]
        gnn_output_with_border = torch.clip(gnn_output_with_border, -20, 20)
        # gnn_output_with_border[:, 2, :, :] = torch.clip(gnn_output_with_border[:, 2, :, :], -1, 3)
    return gnn_output_with_border


def main():
    # -------------------------------
    # Initialize Models
    # -------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gnn_path", type=str, default="./models/gnn-chan_32-nl_2-epoch_20__WD.pt"
    )
    parser.add_argument(
        "--gat_path", type=str, default="./models/gat-chan_80-nl_3-epoch_75__WD.pt"
    )
    parser.add_argument(
        "--cnn_path", type=str, default="./models/cnn-chan_32-epoch_15__WD.pt"
    )
    parser.add_argument("--use_deltas", action="store_true", default=False)
    parser.add_argument("--gat", action="store_true", default=True)

    args = parser.parse_args()

    use_deltas = args.use_deltas

    print(f"Using device: {device}")

    # TODO: Hard coding this because we are missing the weights for GNN
    args.gat = True

    # Define feature channels
    input_cols = ["u", "v", "density", "is_fluid", "border"]
    target_cols = ["u_next", "v_next", "density_next"]

    gnn_hidden_channels = 32
    gat_hidden_channels = 80
    gnn_n_layers = 3
    cnn_hidden_channels = 32

    if args.gat:
        gnn_model = GNN(
            node_dim=len(input_cols),
            edge_dim=1,
            hidden_dim=gat_hidden_channels,
            num_layers=gnn_n_layers,
            out_dim=len(target_cols),
            use_gru=False,
        )
    else:
        gnn_model = GridGNNWithAngles(
            in_channels=len(input_cols),
            hidden_channels=gnn_hidden_channels,
            out_channels=len(target_cols),
            num_layers=gnn_n_layers,
            use_angle=True,
            use_target_node_feat=False,
        )

    gnn_model = gnn_model.to(gnn_device)

    cnn_model = CNN(
        in_channels=len(input_cols),
        hidden_channels=cnn_hidden_channels,
        out_channels=len(target_cols),
        num_layers=3,
    )
    cnn_model = cnn_model.to(device)

    # # Load pre-trained weights if available
    cnn_model.load_state_dict(
        torch.load(args.cnn_path, weights_only=True, map_location=torch.device("cpu"))
    )
    if args.gat:
        gnn_model.load_state_dict(
            torch.load(
                args.gat_path, weights_only=True, map_location=torch.device("cpu")
            )
        )
    else:
        gnn_model.load_state_dict(
            torch.load(
                args.gnn_path, weights_only=True, map_location=torch.device("cpu")
            )
        )

    # cnn_model = torch.load(args.cnn_path)
    # gnn_model = torch.load(args.gnn_path)

    cnn_model.eval()
    gnn_model.eval()

    # -------------------------------
    # Prepare Dummy DataFrame (Replace with Actual Data)
    # -------------------------------

    H, W = 120, 120  # Grid size
    num_nodes = H * W

    # Example DataFrame structure
    data_dict = {
        "simulation_id": np.repeat([0], num_nodes),
        "timestep": np.repeat([0], num_nodes),
        "row": np.tile(np.arange(H), W),
        "col": np.repeat(np.arange(W), H),
        "u": np.zeros(num_nodes),
        "v": np.zeros(num_nodes),
        "density": np.ones(num_nodes) * 0.05,
        #'pressure': np.zeros(num_nodes),
        "is_fluid": np.zeros(num_nodes),
        "border": np.zeros(num_nodes),
        "u_next": np.zeros(num_nodes),
        "v_next": np.zeros(num_nodes),
        "density_next": np.ones(num_nodes) * 0.05,
        #'pressure_next': np.zeros(num_nodes),
        "is_fluid_next": np.ones(num_nodes),
    }

    data_dict["border"] = (
        (data_dict["row"] == 0)
        | (data_dict["col"] == 0)
        | (data_dict["row"] == H - 1)
        | (data_dict["col"] == W - 1)
    ).astype(int)

    # data_dict['is_fluid'] = ((data_dict['row'] != 0) | (data_dict['col'] != 0) |
    #                     (data_dict['row'] != H-1) | (data_dict['col'] != W-1)).astype(int)

    data_dict["is_fluid"] = 1

    print(f"Using device: {device}")

    # -------------------------------
    # Prepare Dummy DataFrame (Replace with Actual Data)
    # -------------------------------

    df_simulation = pd.DataFrame(data_dict)

    # Get all float columns
    float_columns = df_simulation.select_dtypes(include=["float64", "float32"]).columns

    # Round only the float columns to 10 decimal places
    df_simulation[float_columns] = df_simulation[float_columns].round(decimals=10)

    print(f"Input Columns: {input_cols}")
    print(f"Target Columns: {target_cols}")

    means = df_simulation[float_columns].mean()
    stds = df_simulation[float_columns].std()

    print(means)

    with open("norm_mean.csv", "r") as f:
        means = pd.read_csv(f, index_col=0).squeeze()

    print(means)

    with open("norm_std.csv", "r") as f:
        stds = pd.read_csv(f, index_col=0).squeeze()

    print(stds)

    df_simulation[float_columns] = (means[float_columns] - means[float_columns]) / stds[
        float_columns
    ]

    # -------------------------------
    # Initialize Data (Without Dataset Class)
    # -------------------------------

    # Create edge_index and edge_attr
    data = create_grid_graph_with_angles(H, W)
    num_nodes = data["num_nodes"]
    edge_index = data["edge_index"]
    edge_attr = data["edge_attr"]
    edge_distance = data["edge_distance"]

    # Extract node features and targets
    features = df_simulation[input_cols].values.astype(np.float32).flatten()
    features = (
        torch.tensor(features, dtype=torch.float)
        .view(1, len(input_cols), H, W)
        .to(device)
    )  # [1, C, H, W]

    # Modify 'u' velocity in specific region
    center_row = H // 2
    row_start = max(center_row - 4, 0)
    row_end = min(center_row + 4, H)
    cols_to_modify = [0, 1, 2, 3, 4]

    features_np = features.cpu().numpy().copy()
    features_np[0, input_cols.index("u"), row_start:row_end, cols_to_modify] = 7.0
    features_np[0, input_cols.index("v"), row_start:row_end, cols_to_modify] = 2.0
    features_np[0, input_cols.index("density"), row_start:row_end, cols_to_modify] = 7.0
    features = torch.tensor(features_np).to(device)

    # Initialize time
    current_time = torch.tensor([[0.0]]).float().to(device)  # [1, 1]

    density_index = target_cols.index("density_next")
    u_index = target_cols.index("u_next")

    # -------------------------------
    # Prediction Loop
    # -------------------------------

    # Simulation loop
    num_steps = 50
    cnn_predictions = []  # features.detach().cpu().numpy()]
    gnn_predictions = []  # features.detach().cpu().numpy()]
    debug = False

    start = time.time()

    cnn_features = features
    gnn_features = features

    print("Beginning prediction loop")

    print(f"Features shape: {features.shape}")

    # Specific channels to denormalize
    specific_channels = ["u", "v", "density"]
    # Identify their indices in input_cols
    specific_indices = [input_cols.index(col) for col in specific_channels]
    print("Specific Channel Indices:", specific_indices)

    # Convert means and stds to tensors for specific channels
    means_specific = torch.tensor(
        means[specific_channels].values, dtype=torch.float32, device=device
    )
    stds_specific = torch.tensor(
        stds[specific_channels].values, dtype=torch.float32, device=device
    )

    # Reshape for broadcasting: (1, C, 1, 1)
    means_specific = means_specific.view(1, -1, 1, 1)  # Shape: (1, 3, 1, 1)
    stds_specific = stds_specific.view(1, -1, 1, 1)  # Shape: (1, 3, 1, 1)

    # this is basically intended to provide stability to prevent small values from
    # allowing the simulation to start producing density or velocity in random parts of the simulation
    use_correction_mask = False

    for step in range(num_steps):
        if step % 5 == 0:
            print(f"Step {step+1}/{num_steps}")

        # CNN Prediction
        cnn_output = predict_with_cnn(cnn_model, cnn_features, H, W)

        # GNN Prediction
        gnn_output = predict_with_gnn(
            gnn_model, gnn_features, edge_index, edge_attr, edge_distance, device, H, W
        )

        # Update Time
        new_time = current_time + 1.0

        # Debugging: Print shapes and time
        if debug:
            print(f"  Current Features Shape: {features.shape}")
            print(f"  CNN Output Shape: {cnn_output.shape}")
            print(f"  GNN Output Shape: {gnn_output.shape}")
            print(
                f"  Current Time: {current_time.item()}, Next Time: {new_time.item()}"
            )
            print(features.mean())

        # Append to predictions
        if not use_deltas:
            cnn_predictions.append(cnn_output.cpu().numpy())
            gnn_predictions.append(gnn_output.cpu().numpy())

        # this is basically intended to provide stability to prevent small values from
        # allowing the simulation to start producing density or velocity in random parts of the simulation
        if use_correction_mask:
            correction_mask_cnn = (
                torch.abs(
                    cnn_output[:, density_index, :, :]
                    - cnn_features[:, density_index, :, :]
                )
                < 0.01
            )
            correction_mask_cnn = correction_mask_cnn.unsqueeze(1).expand_as(cnn_output)
            cnn_output[correction_mask_cnn] = cnn_features[correction_mask_cnn]

            correction_mask_gnn = (
                torch.abs(
                    gnn_output[:, density_index, :, :]
                    - gnn_features[:, density_index, :, :]
                )
                < 0.01
            )
            correction_mask_gnn = correction_mask_gnn.unsqueeze(1).expand_as(gnn_output)
            gnn_output[correction_mask_gnn] = gnn_features[correction_mask_gnn]

        # Update current features (using CNN output for this example, but you can choose CNN or GNN)
        cnn_features = cnn_output
        gnn_features = gnn_output
        current_time = new_time

        # Modify 'u' velocity in specific region to maintain U velocity at 5
        cnn_features_np = cnn_features.cpu().numpy().copy()  # [1, C', H, W]
        cnn_u_index = 0  # Assuming 'u' is the first channel in the output
        cnn_features_np[0, u_index, row_start:row_end, cols_to_modify] = 5.0
        cnn_features_np[0, density_index, row_start:row_end, cols_to_modify] = 6.0
        # Convert back to tensor
        if use_deltas:
            cnn_features = torch.tensor(cnn_features_np) + cnn_features.detach().cpu()
            cnn_predictions.append(cnn_features.cpu().numpy())
            cnn_features = cnn_features.to(device)
        else:
            cnn_features = torch.tensor(cnn_features_np).to(device)

        # Modify 'u' velocity in specific region to maintain U velocity at 5
        gnn_features_np = gnn_features.cpu().numpy().copy()  # [1, C', H, W]
        gnn_u_index = 0  # Assuming 'u' is the first channel in the output
        gnn_features_np[0, u_index, row_start:row_end, cols_to_modify] = 7.0
        gnn_features_np[
            0, input_cols.index("v"), row_start:row_end, cols_to_modify
        ] = 2.0
        gnn_features_np[0, density_index, row_start:row_end, cols_to_modify] = 7.0
        # Convert back to tensor

        if use_deltas:
            gnn_features = torch.tensor(gnn_features_np) + gnn_features.detach().cpu()
            gnn_predictions.append(gnn_features.cpu().numpy())
            gnn_features = gnn_features.to(device)
        else:
            gnn_features = torch.tensor(gnn_features_np).to(device)

    # Convert predictions to numpy arrays
    density_index = target_cols.index("density_next")

    # we can just operate in normalized space for the visuals for now because
    # this introduces some unnecessary complexity
    # TODO: put this back in eventually

    # cnn_predictions = denormalize_specific_channels(
    #     torch.tensor(cnn_predictions), specific_indices, means_specific.cpu(), stds_specific.cpu()
    # ).cpu().numpy()
    # gnn_predictions = denormalize_specific_channels(
    #         torch.tensor(gnn_predictions), specific_indices, means_specific.cpu(), stds_specific.cpu()
    # ).cpu().numpy()

    cnn_predictions = np.array(cnn_predictions)
    gnn_predictions = np.array(gnn_predictions)

    print(f"Simulation complete. {time.time() - start} s")
    print(f"CNN Predictions Shape: {cnn_predictions.shape}")
    print(f"GNN Predictions Shape: {gnn_predictions.shape}")

    # Example of accessing predictions
    print("\nExample of accessing predictions:")
    print("CNN prediction for step 1, channel 1:")
    print(cnn_predictions[0, 0, 2, :5, :5])  # Show 5x5 grid of channel 2 at step 5
    print("\nGNN prediction for step 1, channel 1:")
    print(gnn_predictions[0, 0, 2, :5, :5])  # Show 5x5 grid of channel 2 at step 5

    # -------------------------------
    # Visualization Functions
    # -------------------------------

    def create_animation(predictions, model_name, index, index_label):
        fig, ax = plt.subplots()
        ims = []
        for step in range(len(predictions)):
            im = ax.imshow(
                predictions[step][0, index, :, :], animated=True, cmap="viridis"
            )

            # Create a text annotation for the current step
            txt = ax.text(
                10,
                10,
                f"Step: {step}",
                color="white",
                fontsize=12,
                backgroundcolor="black",
                verticalalignment="top",
            )

            # Append both the image and text to the frames list
            ims.append([im, txt])

        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=1000
        )
        plt.colorbar(im)
        plt.title(f"{model_name} {index_label} Evolution")

        # Save the animation
        ani.save(f"./gifs/{model_name}_{index_label}_evolution.gif", writer="pillow")

        plt.show()

    # -------------------------------
    # Visualization
    # -------------------------------
    density_index = input_cols.index("density")
    u_index = input_cols.index("u")

    # Animations
    create_animation(cnn_predictions, "CNN", density_index, "density")
    create_animation(
        gnn_predictions, "GAT" if args.gat else "GNN", density_index, "density"
    )
    create_animation(cnn_predictions, "CNN", u_index, "u")
    create_animation(gnn_predictions, "GAT" if args.gat else "GNN", u_index, "u")


if __name__ == "__main__":
    main()
