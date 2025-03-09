"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: plots the most recent run done by run_MC.py. It plots the final
states of the last run which (for now atleast) contains 4 different settings
for the ps variable.

Typical usage example:

python3 plot_MC.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Get the Set2 color palette from Seaborn
COLORS = sns.color_palette("Set3", 8)

# Create a colormap of the cherry picked colours that we want
CMAP = LinearSegmentedColormap.from_list("Set3_custom", COLORS[1:5], N=256)

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# SIZE PARAMEERS
TITLESIZE = 16
LABELSIZE = 20
TICKSIZE = 18


def load_data(run_index=-1):
    """
    Loads the latest run folder and metadata, and returns the data for plotting.

    Returns:
        data: list of hashmaps
    """
    data_folder = "../data/MC"

    # Get list of all subfolders in data_folder, sorted by timestamp (latest first)
    run_folders = sorted([d for d in os.listdir(data_folder) if
                          os.path.isdir(os.path.join(data_folder, d))],)

    if not run_folders:
        print("No simulation runs found.")
        return []

    # Get the latest run folder (latest timestamp)
    latest_run_folder = os.path.join(data_folder, run_folders[run_index])

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


def mc_plot_data():
    """
    Plots the 4 simulations in a 2x2 subplot.
    """
    data = load_data()
    if not data:
        print("No data to plot.")
        return

    # Sort the data by sticking probability (ps) in descending order
    data_sorted = sorted(data, key=lambda x: x["sticking_probability"], reverse=True)

    # Create the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, item in enumerate(data_sorted):
        grid = item["grid"]
        cluster = item["cluster"]
        ps = item["sticking_probability"]

        # Plot the grid
        ax = axes[i]
        ax.imshow(grid, extent=[0, 1, 0, 1], origin='lower', cmap=CMAP,
                  aspect='equal', vmin=0, vmax=1)

        # Overlay cluster data
        if cluster.any():
            cluster_array = np.ma.masked_where(cluster == 0, cluster)
            ax.imshow(cluster_array, extent=[0, 1, 0, 1], origin='lower',
                      cmap='gray', alpha=0.5)

        # Set title and labels
        ax.set_title(f"MC with ps={ps}", fontsize=TITLESIZE)
        ax.set_xlabel("x", fontsize=LABELSIZE)
        ax.set_ylabel("y", fontsize=LABELSIZE)
        ax.tick_params(axis='both', labelsize=TICKSIZE)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.show()


def mc_report_plot():
    """
    Plots the desired ps values for the Monte Carlo DLA implementation.
    The values for Ps that are plotted are: [1, 0.5, 0.25, 0.15, 0.1, 0.02
    """
    # loads the date of the different runs
    data0 = load_data(0)  # contains 1, 0.75, 0.5, and 0.25
    data1 = load_data(1)  # contains 0.2, 0.15, 0.1, 0.05
    data2 = load_data(2)  # contains 0.04, 0.03, 0.02, 0.01

    # Merge all data into one list
    all_data = data0 + data1 + data2

    # List of desired sticking probabilities to plot
    desired_ps = [1, 0.5, 0.25, 0.15, 0.1, 0.02]

    # Filter data to only include the desired sticking probabilities
    filtered_data = [item for item in all_data if item["sticking_probability"] in desired_ps]

    # Sort the filtered data by sticking probability (ps), descending order
    filtered_data_sorted = sorted(filtered_data, key=lambda x: x["sticking_probability"],
                                  reverse=True)

    # Create a 3x2 grid of subplots, with shared x and y axes
    fig, axes = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each of the filtered data points
    for i, item in enumerate(filtered_data_sorted):
        grid = item["grid"]
        cluster = item["cluster"]
        ps = item["sticking_probability"]

        # Plot the grid
        ax = axes[i]
        ax.imshow(grid, extent=[0, 1, 0, 1], origin='lower', cmap=CMAP,
                  aspect='equal', vmin=0, vmax=1)

        # Overlay cluster data
        if cluster.any():
            cluster_array = np.ma.masked_where(cluster == 0, cluster)
            ax.imshow(cluster_array, extent=[0, 1, 0, 1], origin='lower',
                      cmap='gray', alpha=0.7)

        # Set figure sizes
        ax.set_title(f"Sticking probability={ps}", fontsize=TITLESIZE)
        ax.tick_params(axis='both', labelsize=TICKSIZE)

    # Set x and y labels only for the outermost subplots
    for ax in axes[3:]:  # Last row, bottom row subplots
        ax.set_xlabel("x", fontsize=LABELSIZE)

    for ax in axes[::2]:  # First column, leftmost column subplots
        ax.set_ylabel("y", fontsize=LABELSIZE)

    # Adjust layout to reduce horizontal space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    plt.savefig("../figures/MC/MC_report.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    mc_plot_data()
    mc_report_plot()
