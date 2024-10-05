import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

from fluid_simulation.utils import create_grid_graph_with_angles


class GridDatasetGNN(Dataset):
    def __init__(self, df, feature_cols, target_cols, height, width, n_samples=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing simulation data with columns ['simulation_id', 'timestep', 'row', 'col', ...].
            input_cols (list): List of column names to be used as input features.
            target_cols (list): List of column names to be used as target features.
            height (int): Number of rows in the grid.
            width (int): Number of columns in the grid.
        """
        super(GridDatasetGNN, self).__init__()
        self.df = df
        self.input_cols = feature_cols
        self.target_cols = target_cols
        self.height = height
        self.width = width
        self.sample_set = None
        if n_samples:
            sample_indices = np.random.randint(0, 160, [n_samples, 2])
            sample_set = set()
            for item in sample_indices:
                sample_set.add(tuple(item))
            self.sample_set = sample_set
            self.edge_index = None
            self.edge_attr = None
            self.edge_distance = None
            self.data_list = None
            self.data_list = self.process_data()
        else:
            sample_indices = None
            graph = create_grid_graph_with_angles(
                height, width, None
            )  # [2, 4 * num_nodes], [4 * num_nodes, 4]
            self.edge_index = graph["edge_index"]
            self.edge_attr = graph["edge_attr"]
            self.edge_distance = graph["edge_distance"]
            self.data_list = self.process_data()

    def process_data(self):
        data_list = []
        grouped = self.df.groupby(["simulation_id", "timestep"])
        for (sim_id, timestep), group in grouped:
            # Ensure unique (row, col) per simulation_id and timestep
            assert (
                group[["row", "col"]].duplicated().sum() == 0
            ), "Duplicate (row, col) found."

            if self.sample_set:
                graph = create_grid_graph_with_angles(
                    self.height, self.width, sample_indices=self.sample_set
                )
                self.edge_index = graph["edge_index"]
                self.edge_attr = graph["edge_attr"]
                self.edge_distance = graph["edge_distance"]
                group = group[
                    group[["row", "col"]].apply(tuple, axis=1).isin(self.sample_set)
                ]

            # Sort the group by row and then by column to maintain consistency
            group = group.sort_values(["row", "col"]).reset_index(drop=True)

            # Extract node features
            features = group[self.input_cols].values.astype(
                np.float32
            )  # Ensure float type

            # Remove positional encodings
            # No concatenation of x_pos and y_pos

            x = torch.tensor(features, dtype=torch.float)  # [num_nodes, in_channels]
            y = torch.tensor(
                group[self.target_cols].values.astype(np.float32), dtype=torch.float
            )  # [num_nodes, out_channels]

            # Create Data object with edge attributes

            # this will kind of blow up memory unless the edge objects are shared -- not sure if they are
            data = Data(
                x=x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                edge_distance=self.edge_distance,
                y=y,
            )
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class GridDatasetCNN(Dataset):
    def __init__(self, df, input_cols, target_cols, row_max=None, col_max=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing simulation data with columns ['simulation_id', 'timestep', 'row', 'col', ...].
            input_cols (list): List of column names to be used as input features.
            target_cols (list): List of column names to be used as target features.
            row_max (int, optional): Maximum row index. If None, inferred from df.
            col_max (int, optional): Maximum column index. If None, inferred from df.
        """
        super(GridDatasetCNN, self).__init__()
        self.df = df
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.simulations = df["simulation_id"].unique()
        self.timesteps = df["timestep"].unique()
        self.row_max = row_max if row_max is not None else df["row"].max()
        self.col_max = col_max if col_max is not None else df["col"].max()

    def __len__(self):
        return len(self.simulations) * len(self.timesteps)

    def __getitem__(self, idx):
        sim_idx = idx // len(self.timesteps)
        time_idx = idx % len(self.timesteps)

        sim_id = self.simulations[sim_idx]
        timestep = self.timesteps[time_idx]

        current_df = self.df[
            (self.df["simulation_id"] == sim_id) & (self.df["timestep"] == timestep)
        ]

        # Initialize grids
        input_grid = np.zeros(
            (len(self.input_cols), self.row_max + 1, self.col_max + 1), dtype=np.float32
        )
        target_grid = np.zeros(
            (len(self.target_cols), self.row_max + 1, self.col_max + 1),
            dtype=np.float32,
        )

        # Extract row and column indices
        rows = current_df["row"].astype(int).values
        cols = current_df["col"].astype(int).values

        # Extract input and target values
        input_values = current_df[self.input_cols].values
        target_values = current_df[self.target_cols].values

        # Assign values to the grids directly since each (row, col) is unique
        input_grid[:, rows, cols] = input_values.T
        target_grid[:, rows, cols] = target_values.T

        # Convert to torch tensors
        input_tensor = torch.FloatTensor(input_grid)  # [channels, height, width]
        target_tensor = torch.FloatTensor(target_grid)  # [channels, height, width]

        return input_tensor, target_tensor
