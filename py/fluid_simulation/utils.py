import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

import numpy as np
import torch
import pandas as pd
import glob


def load_normalization_params(mean_path, std_path):
    """
    Loads normalization means and standard deviations from CSV files.

    Args:
        mean_path (str): Path to the mean CSV file.
        std_path (str): Path to the standard deviation CSV file.

    Returns:
        pd.Series, pd.Series: Means and standard deviations.
    """
    means = pd.read_csv(mean_path, index_col=0).squeeze()
    stds = pd.read_csv(std_path, index_col=0).squeeze()
    return means, stds


def denormalize_specific_channels(output, specific_indices, means, stds):
    """
    Denormalizes specific channels in the output tensor.

    Args:
        output (torch.Tensor): Normalized output tensor of shape (batch_size, C, *spatial_dims).
        specific_indices (list of int): Indices of channels to denormalize.
        means (torch.Tensor): Means for the specific channels, shape (1, C_selected, 1, 1, ..., 1).
        stds (torch.Tensor): Standard deviations for the specific channels, shape (1, C_selected, 1, 1, ..., 1).

    Returns:
        torch.Tensor: Tensor with specific channels denormalized.
    """
    # Clone the output to avoid in-place modifications
    denorm_output = output.clone()

    # Determine the number of spatial dimensions
    spatial_dims = output.dim() - 2  # Exclude batch and channel dimensions

    # Reshape means and stds to match (1, C_selected, 1, 1, ..., 1)
    # Number of 1's equals spatial_dims
    view_shape = [1, 1, -1, 1, 1]
    means = means.view(*view_shape)
    stds = stds.view(*view_shape)

    # Select the specific channels
    specific_channels = denorm_output[:, 0, specific_indices, ...]

    # Apply denormalization: denorm = norm * std + mean
    denorm_specific = specific_channels * stds + means

    # Assign back the denormalized channels
    print(denorm_output.shape)
    denorm_output[:, 0, specific_indices, ...] = denorm_specific

    return denorm_output


def denormalize_output(output, input_cols, specific_channels, means, stds):
    """
    Denormalizes specified channels in the output tensor.

    Args:
        output (torch.Tensor): Normalized output tensor of shape (batch_size, C, *spatial_dims).
        input_cols (list of str): List of channel names corresponding to the output channels.
        specific_channels (list of str): List of channel names to denormalize.
        means (pd.Series): Means for each channel.
        stds (pd.Series): Standard deviations for each channel.

    Returns:
        torch.Tensor: Tensor with specified channels denormalized.
    """
    # Identify channel indices
    specific_indices = [input_cols.index(col) for col in specific_channels]

    # Extract means and stds for specific channels
    means_specific = torch.tensor(
        means[specific_channels].values, dtype=output.dtype, device=output.device
    )
    stds_specific = torch.tensor(
        stds[specific_channels].values, dtype=output.dtype, device=output.device
    )

    # Denormalize
    denorm_output = denormalize_specific_channels(
        output, specific_indices, means_specific, stds_specific
    )

    return denorm_output


def constraint_error(
    output,
    specific_channel,
    n_channels=4,
    tgt_value=0,
    reshape=None,
    constraint_weight=0.25,
):
    """
    Combines the standard MSE loss with a custom constraint loss.

    Args:
        output (torch.Tensor): The output tensor from the model (batch_size, N, H, W).
        specific_channel (int): The index of the channel to apply the constraint.
        constraint_weight (float): Weight for the constraint loss.

    Returns:
        torch.Tensor: The combined loss.
    """
    # Extract the specific channel (shape: batch_size, 1, H, W)
    # output = output.reshape(n_channels, )

    if reshape:
        output = (
            output.reshape(reshape[0], reshape[1], -1).permute(2, 0, 1).unsqueeze(0)
        )

    channel = output[:, specific_channel : specific_channel + 1, :, :]

    # Define the convolution kernel to sum the node and its four neighbors
    # Shape: (out_channels, in_channels/groups, kH, kW)
    # TODO: should the center kernel be 0??
    if specific_channel == 0:
        kernel = torch.tensor(
            [[[[0, 0, 0], [1, 1, 1], [0, 0, 0]]]],
            dtype=output.dtype,
            device=output.device,
        )
    elif specific_channel == 1:
        kernel = torch.tensor(
            [[[[0, 1, 0], [0, 1, 0], [0, 1, 0]]]],
            dtype=output.dtype,
            device=output.device,
        )

    # Apply convolution
    # Since we're applying the same kernel to each sample independently, set groups=1
    # Ensure padding=1 to maintain spatial dimensions
    sum_neighbors = F.conv2d(channel, kernel, padding=1)

    # The desired sum is zero, so the target for the constraint is zero
    constraint_target = torch.ones_like(sum_neighbors) * tgt_value

    # Compute the constraint loss (MSE between sum_neighbors and zero)
    constraint_loss = F.mse_loss(sum_neighbors, constraint_target)

    # Combine the losses
    total_loss = constraint_weight * constraint_loss

    return total_loss


def denormalize_z_score(normalized_df, original_mean, original_std):
    """
    Denormalizes a DataFrame that was Z-score normalized (standardized).

    Parameters:
    - normalized_df: The DataFrame with standardized values.
    - original_mean: A Pandas Series or array of the original mean values for each feature.
    - original_std: A Pandas Series or array of the original standard deviation values for each feature.

    Returns:
    - denormalized_df: The DataFrame with denormalized values.
    """
    denormalized_df = normalized_df * original_std + original_mean
    return denormalized_df


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


def create_grid_graph_with_angles(
    width, height, sample_indices=None, extra_neighbors=False
):
    """
    Creates a 2D grid graph with neighbors up to 2 hops away and angular edge attributes.

    Args:
        width (int): Number of columns in the grid.
        height (int): Number of rows in the grid.

    Returns:
        dict: A dictionary containing 'edge_index', 'edge_attr', 'edge_distance', and 'num_nodes'.
    """
    edge_index = []
    edge_attr = []
    edge_distance = []

    # Direction vectors and their corresponding angles in degrees for distance 1
    directions = {
        "E": (1, 0, 0),
        "NE": (1, 1, 45),
        "N": (0, 1, 90),
        "NW": (-1, 1, 135),
        "W": (-1, 0, 180),
        "SW": (-1, -1, 225),
        "S": (0, -1, 270),
        "SE": (1, -1, 315),
    }

    # Directions for distance 2
    if extra_neighbors:
        directions_distance_2 = {
            "E2": (2, 0, 0),
            "N2": (0, 2, 90),
            "W2": (-2, 0, 180),
            "S2": (0, -2, 270),
            "NE2": (2, 2, 45),
            "NW2": (-2, 2, 135),
            "SW2": (-2, -2, 225),
            "SE2": (2, -2, 315),
            "E_N": (1, 2, 63.4349),
            "N_E": (2, 1, 26.5651),
            "N_W": (-2, 1, 153.4349),
            "W_N": (-1, 2, 116.5651),
            "W_S": (-1, -2, 243.4349),
            "S_W": (-2, -1, 206.5651),
            "S_E": (2, -1, 333.4349),
            "E_S": (1, -2, 296.5651),
        }  # this is so slow maybe try without these

    for y in range(height):
        for x in range(width):

            if sample_indices is not None and (y, x) not in sample_indices:
                continue

            node = y * width + x

            # self edge
            edge_index.append([node, node])
            edge_attr.append(0)
            edge_distance.append(0)

            # Distance 1 edges
            for dir_name, (dx, dy, angle) in directions.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    edge_index.append([node, neighbor])
                    edge_attr.append(angle)
                    edge_distance.append(1)

            # Distance 2 edges
            if extra_neighbors:
                for dir_name, (dx, dy, angle) in directions_distance_2.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = ny * width + nx
                        edge_index.append([node, neighbor])
                        edge_attr.append(angle)
                        edge_distance.append(2)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Use float for angles
    edge_distance = torch.tensor(
        edge_distance, dtype=torch.float
    )  # Use float for distances

    # Optionally normalize edge_distance
    edge_distance = edge_distance / edge_distance.max()  # Normalize to [0, 1]

    data = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_distance": edge_distance,
        "num_nodes": width * height,
    }

    return data


def process_csv_files(data_folder, output_file):
    # Get all CSV files in the data folder
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    # Function to read a CSV and add a simulation ID
    def read_csv_with_sim_id(file, sim_id):
        print(file)
        df = pd.read_csv(file)
        df["simulation_id"] = sim_id
        return df

    # Combine all CSV files into a single DataFrame, adding simulation IDs
    combined_df = pd.concat(
        [read_csv_with_sim_id(f, i) for i, f in enumerate(csv_files)], ignore_index=True
    )

    # Sort the DataFrame by simulation_id, timestep, row, and column
    combined_df = combined_df.sort_values(["simulation_id", "timestep", "row", "col"])

    # Group by simulation_id, row, and column
    grouped = combined_df.groupby(["simulation_id", "row", "col"])

    # Function to calculate deltas for a group
    def calculate_deltas(group):
        result = group.copy()
        result = result.sort_values(["timestep"])
        for column in group.columns:
            if column not in ["row", "col", "simulation_id"]:
                result[f"{column}_next"] = group[column].shift(-1)
        return result

    # Apply the delta calculation to each group
    result_df = grouped.apply(calculate_deltas).reset_index(drop=True)

    # Sort the result by simulation_id, timestep, row, and column
    result_df = result_df.sort_values(["simulation_id", "timestep", "row", "col"])

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")

    return result_df


def prepare_data(df):
    metadata_cols = [
        "simulation_id",
        "timestep",
        "row",
        "col",
        "iter",
        "time",
        "pressure",
        "pressure_next",
    ]
    input_cols = [
        col
        for col in df.columns
        if col not in metadata_cols and not col.endswith("_next")
    ] + ["border"]
    target_cols = [
        col
        for col in df.columns
        if col.endswith("_next") and col.replace("_next", "") not in metadata_cols
    ]

    print(f"Input columns: {input_cols}")
    print(f"Target columns: {target_cols}")

    row_max = df.row.max()
    col_max = df.col.max()
    df.loc[:, "border"] = 0.0
    df.loc[df["row"].isin([0, row_max]) | df["col"].isin([0, col_max]), "border"] = 1.0

    return df, input_cols, target_cols


import pandas as pd


def calculate_deltas(df):
    # Get all column names
    columns = df.columns

    # Find columns with '_next' suffix
    next_columns = [col for col in columns if col.endswith("_next")]

    # For each '_next' column, find its counterpart and calculate delta
    for next_col in next_columns:
        base_col = next_col.replace("_next", "")

        # Check if the base column exists
        if base_col in columns:
            delta_col = f"delta_{base_col}"
            df[delta_col] = df[next_col] - df[base_col]

    return df


def prepare_data_v2(
    df,
    target_pattern="_next",
    input_pattern_filter="_next",
    input_pattern_filter_2=None,
):
    metadata_cols = [
        "simulation_id",
        "timestep",
        "row",
        "col",
        "iter",
        "time",
        "pressure",
        "pressure_next",
    ]
    input_cols = [
        col
        for col in df.columns
        if col not in metadata_cols and not input_pattern_filter in col
    ] + ["border"]
    if input_pattern_filter_2:
        input_cols = [col for col in input_cols if not input_pattern_filter_2 in col]
    target_cols = [
        col
        for col in df.columns
        if target_pattern in col
        and col.replace(target_pattern, "") not in metadata_cols
    ]

    print(f"Input columns: {input_cols}")
    print(f"Target columns: {target_cols}")

    row_max = df.row.max()
    col_max = df.col.max()
    df.loc[:, "border"] = 0.0
    df.loc[df["row"].isin([0, row_max]) | df["col"].isin([0, col_max]), "border"] = 1.0

    return df, input_cols, target_cols
