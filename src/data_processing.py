"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Data Processing Module

Functions for loading and saving DLA simulation data.
"""
import os
import csv
import pickle
import pandas as pd


def save_step_data_to_csv(step_data, eta):
    """Save step data to CSV file for later analysis"""
    os.makedirs("../eta_analysis/data", exist_ok=True)

    filename = f"../eta_analysis/data/steps_eta{eta}.csv"

    with open(filename, 'w', newline='') as csvfile:
        if not step_data:
            print(f"  No step data to save for eta={eta}")
            return

        fieldnames = step_data[0].keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step_info in step_data:
            writer.writerow(step_info)

    print(f"  Saved {len(step_data)} steps to {filename}")

    os.makedirs("../eta_analysis/clusters", exist_ok=True)
    print("step_data", step_data[-1])
    if step_data and 'cluster' in step_data[-1]:
        print("saving cluster", step_data[-1]['cluster'])
        with open(f"../eta_analysis/clusters/cluster_eta{eta}.pkl", 'wb') as f:
            pickle.dump(step_data[-1]['cluster'], f)


def save_summary_results(summary_results):
    """Save summary results to CSV file"""
    os.makedirs("../eta_analysis", exist_ok=True)

    with open('../eta_analysis/summary_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['eta', 'final_height', 'final_width', 'final_size',
                      'steps_completed', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in summary_results:
            writer.writerow(result)

    print(f"Saved summary results for {len(summary_results)} eta values")


def load_all_data():
    """Load all saved CSV files with step data"""
    data_dir = "../eta_analysis/data"

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return None

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found!")
        return None

    all_data = {}
    for file in csv_files:
        # Extract eta value from filename
        eta = float(file.split('eta')[1].split('.csv')[0])

        # Load data
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        # Store in dictionary
        all_data[eta] = df

    print(f"Loaded data for {len(all_data)} eta values: {sorted(all_data.keys())}")
    return all_data


def load_clusters():
    """Load the saved cluster states"""
    clusters_dir = "../eta_analysis/clusters"

    if not os.path.exists(clusters_dir):
        print(f"Directory {clusters_dir} not found!")
        return None
    print(clusters_dir)
    pkl_files = [f for f in os.listdir(clusters_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print("No pickle files found!")
        return None

    clusters = {}
    for file in pkl_files:
        # Extract eta value from filename
        eta = float(file.split('eta')[1].split('.pkl')[0])

        # Load cluster
        with open(os.path.join(clusters_dir, file), 'rb') as f:
            cluster = pickle.load(f)

        # Store in dictionary
        clusters[eta] = cluster

    print(f"Loaded {len(clusters)} clusters")
    return clusters
