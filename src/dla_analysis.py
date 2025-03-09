"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: DLA Analysis Module

Contains functions for running DLA simulations with different eta values
and analyzing the results.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from DLA import DLA
from data_processing import save_step_data_to_csv, load_all_data, load_clusters

# Global settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)
LABELSIZE = 18
TICKSIZE = 16

def run_simulation_with_data_saving(eta, N=100, max_steps=1000):
    """Run a single DLA simulation and save data"""
    print(f"\nSimulating with eta = {eta}")
    dla = DLA(N=N, eta=eta)
    step_data = []
    start_time = time.time()
    
    for step in range(max_steps):
        try:
            # metrics before growth
            cluster_points = np.argwhere(dla.cluster == 1)
            size = len(cluster_points)
            height = np.max(cluster_points[:, 0]) if size > 0 else 0
            
            # Calculate width
            min_x = np.min(cluster_points[:, 1]) if size > 0 else 0
            max_x = np.max(cluster_points[:, 1]) if size > 0 else 0
            width = max_x - min_x + 1 if size > 0 else 0
            
            # Calculate aspect ratio
            aspect = height / max(1, width)
            
            # one step using the step method
            start_iter_time = time.time()
            
            try:
                dla.step()
                step_successful = True
            except StopIteration:
                print(f"  Cluster reached top at step {step}")
                step_successful = True
                break
            except Exception as e:
                print(f"  Error in step() method: {e}")
                step_successful = False
            
            diffusion_time = time.time() - start_iter_time
            
            if step_successful:
                step_info = {
                    'step': step,
                    'eta': eta,
                    'height': height,
                    'width': width,
                    'size': size,
                    'aspect_ratio': aspect,
                    'diffusion_time': diffusion_time,
                    'elapsed_time': time.time() - start_time
                }
                
                step_data.append(step_info)
                
                if step % 10 == 0:
                    print(f"  Step {step}, Height: {height}, Size: {size}")
                    
        except Exception as e:
            print(f"  Unexpected error at step {step}: {e}")
    dla.save_as_csv()
    save_step_data_to_csv(step_data, eta)
    
    final_height = height if 'height' in locals() else 0
    final_width = width if 'width' in locals() else 0
    final_size = size if 'size' in locals() else 0
    
    return {
        'eta': eta,
        'final_height': final_height,
        'final_width': final_width,
        'final_size': final_size,
        'steps_completed': len(step_data),
        'time': time.time() - start_time
    }

def create_separate_plots(all_data, clusters):
    """Create and save each plot separately using the same color palette as GrayScott"""
    if not all_data or not clusters:
        return
    
    eta_values = sorted(all_data.keys())
    
    # 1. Height Growth Plot
    plt.figure(figsize=(6, 4))
    for i, eta in enumerate(eta_values):
        df = all_data[eta]
        plt.plot(df['step'], df['height'], label=rf'$\eta$={eta}', color=colors[i % len(colors)])
    
    plt.xlabel('Step', fontsize=LABELSIZE)
    plt.ylabel('Height', fontsize=LABELSIZE)
    # plt.title('Cluster Height Growth', fontsize=LABELSIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=TICKSIZE)
    plt.xticks(fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig('../figures/eta_figures/height_growth.pdf')
    plt.close()
    
    # 2. Growth Rate Comparison
    plt.figure(figsize=(10, 8))
    
    growth_rates = []
    
    for eta in eta_values:
        df = all_data[eta]
        if len(df) > 10:  # Ensure enough data points
            # linear regression to find slope (growth rate)
            steps = df['step'].values
            heights = df['height'].values
            
            slope = np.polyfit(steps, heights, 1)[0]
            growth_rates.append(slope)
        else:
            growth_rates.append(0)
    
    bars = plt.bar(range(len(eta_values)), growth_rates, color=[colors[i % len(colors)] for i in range(len(eta_values))])
    
    for bar, rate in zip(bars, growth_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=TICKSIZE)
    
    plt.xlabel('Eta Value', fontsize=LABELSIZE)
    plt.ylabel('Height Growth Rate', fontsize=LABELSIZE)
    # plt.title('Growth Rate Comparison', fontsize=LABELSIZE)
    plt.xticks(range(len(eta_values)), [str(eta) for eta in eta_values], fontsize=TICKSIZE-5)
    plt.yticks(fontsize=TICKSIZE)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../figures/eta_figures/growth_rates.pdf')
    plt.close()
    
    # 3. Side-by-side comparison of all clusters
    if eta_values:
        cols = 2
        rows = (len(eta_values) + cols -1 )// cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharex=True, sharey=True)
        
        # Flatten axes for easier indexing if multiple rows
        if rows > 1:
            axes = axes.flatten()
        elif rows == 1 and cols == 1:
            axes = [axes]  
            
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list('cluster_cmap', ['#FFF7CC', '#6BAED6'])
        
        # Plot each cluster
        for i, eta in enumerate(eta_values):
            if i < len(axes):
                cluster = np.loadtxt(f"../results/DLA_cluster_eta{eta}.csv", delimiter=",")
                cluster_array = np.ma.masked_where(cluster == 0, cluster)
                concentration = np.loadtxt(f"../results/DLA_concentration_eta{eta}.csv", delimiter=",")
                
                axes[i].imshow(concentration, cmap='Spectral', origin='lower', vmin=0, vmax=1)
                axes[i].imshow(cluster, cmap='gray', origin='lower', alpha=0.5)
                axes[i].set_title(rf'$\eta$ = {eta}', fontsize=LABELSIZE+5)
                axes[i].tick_params(axis='both', which='major', labelsize=TICKSIZE+5)
                axes[i].set_xlabel('x', fontsize=LABELSIZE+5)
                axes[i].set_ylabel('y', fontsize=LABELSIZE+5)

        for i in range(len(eta_values), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('../figures/eta_figures/all_clusters.pdf', dpi=300)
        plt.close()

def analyze_eta_influence():
    """Run simulations and analysis for different eta values"""
    os.makedirs("../eta_analysis", exist_ok=True)
    os.makedirs("../eta_analysis/data", exist_ok=True)
    os.makedirs("../eta_analysis/clusters", exist_ok=True)
    
    eta_values = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    max_steps = 1000
    summary_results = []
    
    # simulations for each eta and save data
    for eta in eta_values:
        result = run_simulation_with_data_saving(eta, max_steps=max_steps)
        summary_results.append(result)
    
    print("\nAll step data has been saved to CSV files in eta_analysis/data/")
    print("These files can be used for further statistical analysis.")
    
    return summary_results

def visualize_results():
    """Load saved data and create visualizations"""
    os.makedirs("eta_analysis", exist_ok=True)
    
    # Load data
    print("Loading data...")
    all_data = load_all_data()
    clusters = load_clusters()
    
    if not all_data or not clusters:
        print("Error loading data. Make sure the simulations have been run.")
        return
    
    # Create plots
    print("Creating plots...")
    create_separate_plots(all_data, clusters)
    
    print("Analysis complete! Check the figures folder for results.")