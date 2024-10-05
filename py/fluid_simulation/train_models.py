import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader

from fluid_simulation.datasets_v2 import GridDatasetGNN, GridDatasetCNN
from fluid_simulation.models_v2 import CNN, GridGNNWithAngles
from fluid_simulation.gat.gat_torch_scatter import GNN
from fluid_simulation.utils import constraint_error
from fluid_simulation.utils import prepare_data, calculate_deltas, prepare_data_v2


def normalize_tensor(tensor, mean, std):
    # Broadcasting the mean and std tensors to match the tensor shape
    return (tensor - mean) / (std + 1e-7)


torch.manual_seed(8)
np.random.seed(8)


def train_model(
    model,
    dataloader,
    num_epochs=10,
    learning_rate=0.001,
    device=None,
    model_type="gnn",
    tgt_constraint_value=0,
    gat=False,
):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    # Add LR scheduler
    scheduler = StepLR(
        optimizer, step_size=2, gamma=0.9
    )  # Reduce LR by factor of 0.1 every 5 epochs
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Starting training: {next(model.parameters()).device}")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        i = 0
        for batch in dataloader:
            if model_type == "cnn":
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss_u = constraint_error(outputs, 0, n_channels=3)
                loss_v = constraint_error(outputs, 1, n_channels=3)
                mse_loss = criterion(targets, outputs)
                loss = mse_loss + loss_u + loss_v
            elif model_type == "gnn":
                batch = batch.to(device)
                out = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.edge_distance
                )  # .squeeze()
                if gat:
                    # print(f'out shape: {out.shape}')
                    # print(batch.y.shape)
                    pass
                loss_u = constraint_error(
                    out,
                    0,
                    n_channels=3,
                    reshape=(120, 120),
                    tgt_value=tgt_constraint_value,
                    constraint_weight=0.75,
                )
                loss_v = constraint_error(
                    out,
                    1,
                    n_channels=3,
                    reshape=(120, 120),
                    tgt_value=tgt_constraint_value,
                    constraint_weight=0.75,
                )
                mse_loss = criterion(batch.y, out)
                loss = mse_loss + loss_u + loss_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 20 == 0:
            #     print(f"Batch {i} loss: {loss.item()}")
            #     #print(batch)
            # i += 1

        # Step the scheduler
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        # if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    print("Finished training")
    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_deltas", action="store_true", default=False)
    parser.add_argument("--n_epochs_gnn", type=int, default=10)
    parser.add_argument("--n_epochs_cnn", type=int, default=10)
    parser.add_argument("--hidden_channels_cnn", type=int, default=32)
    parser.add_argument("--hidden_channels_gnn", type=int, default=32)
    parser.add_argument("--train_cnn", action="store_true", default=False)
    parser.add_argument("--train_gnn", action="store_true", default=False)
    parser.add_argument("--use_3rd_simulation", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--gat", action="store_true", default=False)

    args = parser.parse_args()

    if not args.data_path:
        csv_file = "./combined_pressure_free_data_with_deltas.csv"
    else:
        csv_file = args.data_path
    # Load your data into a DataFrame
    timestep_n_rows = 14400
    n_steps = 200
    df = pd.read_csv(csv_file, nrows=timestep_n_rows * n_steps)
    df2 = pd.read_csv(
        csv_file, nrows=timestep_n_rows * n_steps, skiprows=timestep_n_rows * 200
    )
    df2.columns = df.columns
    print(df.shape)
    print(df.shape[0] // 51_200)
    # df2.simulation_id.value_counts()
    df = pd.concat([df, df2])

    df = df.dropna()

    print(df.isna().sum())

    device = "cpu"  # torch.device("mps" if torch.backends.mps.is_available() else "cpu"), unfortunately MPS does not accommodate torch_scatter
    cnn_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    use_deltas = args.use_deltas

    if use_deltas:
        df = calculate_deltas(df)
        df, input_cols, target_cols = prepare_data_v2(
            df, target_pattern="delta_", input_pattern_filter_2="delta"
        )
    else:
        df, input_cols, target_cols = prepare_data_v2(
            df, target_pattern="_next", input_pattern_filter_2="delta"
        )

    target_cols = [c for c in target_cols if "fluid" not in c]

    model_gnn = None
    model_cnn = None
    loader_gnn = None
    loader_cnn = None
    gnn_hidden_channels = 32
    gnn_n_layers = 2
    cnn_hidden_channels = 40

    # Get all float columns
    float_columns = df.select_dtypes(include=["float64", "float32"]).columns

    print(float_columns)

    # Round only the float columns to 10 decimal places
    df[float_columns] = df[float_columns].round(decimals=10)

    print(f"Input Columns: {input_cols}")
    print(f"Target Columns: {target_cols}")

    means = df[float_columns].mean()
    stds = df[float_columns].std()

    with open("norm_mean.csv", "w") as f:
        means.to_csv(f)

    with open("norm_std.csv", "w") as f:
        stds.to_csv(f)

    df[float_columns] = (df[float_columns] - means) / stds

    normed_0 = means["u"] / stds["u"]

    if args.train_gnn:
        # Instantiate the dataset
        dataset_gnn = GridDatasetGNN(
            df=df,
            feature_cols=input_cols,
            target_cols=target_cols,
            height=120,
            width=120,
        )

        """
        First Layer:
        Node B aggregates information from nodes A and C.
        Second Layer:
        Node B now indirectly aggregates information from nodes A, C, and D (since node C aggregates from B and D in the first layer).
        """

        # HIDDEN CHANNELS ACTUALLY GETS USED FOR BOTH X_i and X_j so it's times 2
        if args.gat:
            model_gnn = GNN(
                node_dim=len(input_cols),
                edge_dim=1,
                hidden_dim=gnn_hidden_channels,
                num_layers=gnn_n_layers,
                out_dim=len(target_cols),
                use_gru=False,
            )
        else:
            model_gnn = GridGNNWithAngles(
                in_channels=len(input_cols),
                hidden_channels=gnn_hidden_channels,
                out_channels=len(target_cols),
                num_layers=gnn_n_layers,
                use_angle=True,
                use_target_node_feat=False,
            )

        model_gnn = model_gnn.to(device)

        batch_size = 8
        loader_gnn = DataLoader(dataset_gnn, batch_size=batch_size, shuffle=True)
        # Example: Iterate through the DataLoader
        for batch in loader_gnn:
            batch = batch.to(device)
            # batch.x: [batch_size * num_nodes, in_features]
            # batch.edge_index: [2, batch_size * 4 * num_nodes]
            # batch.y: [batch_size * num_nodes, target_features]
            print(batch)
            output = model_gnn(
                batch.x, batch.edge_index, batch.edge_attr, batch.edge_distance
            )
            # Compute loss, backpropagate, etc.
            print(output.shape)
            break  # Remove this to iterate through the entire dataset
        print("---- GNN is Valid ----")

    if args.train_cnn:
        cnn_dataset = GridDatasetCNN(
            df, input_cols, target_cols, row_max=None, col_max=None
        )
        # Instantiate the dataset
        model_cnn = CNN(
            in_channels=len(input_cols),
            hidden_channels=cnn_hidden_channels,
            out_channels=len(target_cols),
            num_layers=2,
        )
        model_cnn = model_cnn.to(cnn_device)

        print(next(model_cnn.parameters()).device)
        print(sum(p.numel() for p in model_cnn.parameters() if p.requires_grad))

        batch_size = 4
        loader_cnn = DataLoader(cnn_dataset, batch_size=batch_size, shuffle=True)

        # Example: Iterate through the DataLoader
        for inputs, targets in loader_cnn:
            inputs = inputs.to(cnn_device)
            targets = targets.to(cnn_device)
            print(inputs.shape)
            output = model_cnn(inputs)
            # Compute loss, backpropagate, etc.
            print(output.shape)
            break  # Remove this to iterate through the entire dataset
        print("---- CNN is Valid ----")

    if args.train_gnn:
        print("--- Training GNN ---")
        n_epochs = args.n_epochs_gnn
        print(sum(p.numel() for p in model_gnn.parameters() if p.requires_grad))
        print(device)
        print(
            sum(p.numel() for p in model_gnn.layers[0].parameters() if p.requires_grad)
        )
        gnn_trained = train_model(
            model_gnn,
            loader_gnn,
            num_epochs=n_epochs,
            learning_rate=0.001,
            device=device,
            model_type="gnn",
            tgt_constraint_value=normed_0,
            gat=args.gat,
        )
        gnn_model_name = "gat" if args.gat else "gnn"
        gnn_path = f"models/{gnn_model_name}-chan_{gnn_hidden_channels}-nl_{gnn_n_layers}-epoch_{n_epochs}__WD.pt"
        torch.save(gnn_trained.state_dict(), gnn_path)
        print(f"GNN saved to {gnn_path}")

    if args.train_cnn:
        print("--- Training CNN ---")
        n_epochs = args.n_epochs_cnn
        print(sum(p.numel() for p in model_cnn.parameters() if p.requires_grad))
        print(cnn_device)
        cnn_trained = train_model(
            model_cnn,
            loader_cnn,
            num_epochs=n_epochs,
            learning_rate=0.0005,
            device=cnn_device,
            model_type="cnn",
            tgt_constraint_value=normed_0,
        )
        torch.save(
            cnn_trained.state_dict(),
            f"models/cnn-chan_{cnn_hidden_channels}-epoch_{n_epochs}__WD.pt",
        )


if __name__ == "__main__":
    main()
