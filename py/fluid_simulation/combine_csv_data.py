import pandas as pd
import numpy as np
import torch
import os
import glob


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


if __name__ == "__main__":
    # Usage
    data_folder = "../../../data/pressure_free/"
    output_file = "combined_pressure_free_data_with_deltas.csv"
    result_df = process_csv_files(data_folder, output_file)
    print(result_df.head())
