import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_folder="../data/MC"):
    """
    Loads the latest run folder and metadata, and returns the data for plotting.

    Returns:
        data: list of hashmaps
    """
    # Get list of all subfolders in data_folder, sorted by timestamp (latest first)
    run_folders = sorted([d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))], reverse=True)

    if not run_folders:
        print("No simulation runs found.")
        return []

    # Get the latest run folder (latest timestamp)
    latest_run_folder = os.path.join(data_folder, run_folders[0])

    # Get all metadata files in the latest folder
    metadata_files = [f for f in os.listdir(latest_run_folder) if f.endswith("_metadata.json")]

    if len(metadata_files) < 4:
        print("Not enough saved simulations to plot (need at least 4).")
        return []

    data = []
    for metadata_file in sorted(metadata_files)[:4]:
        # Load metadata
        with open(os.path.join(latest_run_folder, metadata_file), "r") as metafile:
            metadata = json.load(metafile)

        # Get the filenames for the agent and cluster data from metadata
        agent_csv_filename = metadata["agent_csv"]
        cluster_csv_filename = metadata["cluster_csv"]

        # Load the agent and cluster data from CSV
        grid = np.loadtxt(agent_csv_filename, delimiter=",")
        cluster = np.loadtxt(cluster_csv_filename, delimiter=",")

        # Append the loaded data and metadata
        data.append({
            "grid": grid,
            "cluster": cluster,
            "sticking_probability": metadata["sticking_probability"]
        })

    return data


def plot_data(data):
    """
    Plots the 4 simulations in a 2x2 subplot.
    """
    if not data:
        print("No data to plot.")
        return

    # Create the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, item in enumerate(data):
        grid = item["grid"]
        cluster = item["cluster"]
        ps = item["sticking_probability"]

        # Plot the grid
        ax = axes[i]
        ax.imshow(grid, extent=[0, 1, 0, 1], origin='lower', cmap='Spectral', aspect='equal', vmin=0, vmax=1)

        # Overlay cluster data
        if cluster.any():
            cluster_array = np.ma.masked_where(cluster == 0, cluster)
            ax.imshow(cluster_array, extent=[0, 1, 0, 1], origin='lower', cmap='gray', alpha=0.5)

        # Set title and labels
        ax.set_title(f"MC with ps={ps}", fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("../figures/MC_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    data = load_data()
    plot_data(data)
