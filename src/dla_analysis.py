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

def run_simulation_with_data_saving(eta, N=100, max_steps=100):
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
    """Create and save each plot separately"""
    if not all_data or not clusters:
        return
    
    eta_values = sorted(all_data.keys())
    colors = sns.color_palette("viridis", len(eta_values))
    
    os.makedirs("eta_analysis", exist_ok=True)
    
    # 1. Height Growth Plot
    plt.figure(figsize=(10, 6))
    for i, eta in enumerate(eta_values):
        df = all_data[eta]
        plt.plot(df['step'], df['height'], label=f'eta = {eta}', color=colors[i])
    
    plt.xlabel('Step')
    plt.ylabel('Height')
    plt.title('Cluster Height Growth', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eta_analysis/height_growth.png', dpi=300)
    plt.close()
    
    # 2. Size Growth Plot
    plt.figure(figsize=(10, 6))
    for i, eta in enumerate(eta_values):
        df = all_data[eta]
        plt.plot(df['step'], df['size'], label=f'eta = {eta}', color=colors[i])
    
    plt.xlabel('Step')
    plt.ylabel('Cluster Size (sites)')
    plt.title('Cluster Size Growth', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eta_analysis/size_growth.png', dpi=300)
    plt.close()
    
    # 3. Height vs Size Plot
    plt.figure(figsize=(10, 6))
    for i, eta in enumerate(eta_values):
        df = all_data[eta]
        plt.plot(df['size'], df['height'], label=f'eta = {eta}', color=colors[i])
    
    plt.xlabel('Cluster Size (sites)')
    plt.ylabel('Height')
    plt.title('Height vs Size Relationship', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eta_analysis/height_vs_size.png', dpi=300)
    plt.close()
    
    # 4. Growth Rate Comparison
    plt.figure(figsize=(10, 6))
    
    # growth rates (height per step)
    growth_rates = []
    
    for eta in eta_values:
        df = all_data[eta]
        if len(df) > 10: 
            # linear regression to find slope (growth rate)
            steps = df['step'].values
            heights = df['height'].values
            
            slope = np.polyfit(steps, heights, 1)[0]
            growth_rates.append(slope)
        else:
            growth_rates.append(0)
    
    bars = plt.bar(range(len(eta_values)), growth_rates, color=colors)
    
    for bar, rate in zip(bars, growth_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Eta Value')
    plt.ylabel('Height Growth Rate')
    plt.title('Growth Rate Comparison', fontsize=14)
    plt.xticks(range(len(eta_values)), [str(eta) for eta in eta_values])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('eta_analysis/growth_rates.png', dpi=300)
    plt.close()
        
    # 5. Side-by-side comparison of all clusters
    if eta_values:
        if len(eta_values) <= 4:
            rows, cols = 1, len(eta_values)
        else:
            cols = 4 
            rows = (len(eta_values) + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Flatten axes
        if rows > 1:
            axes = axes.flatten()
        elif rows == 1 and cols == 1:
            axes = [axes]  
            
        cmap = LinearSegmentedColormap.from_list('cluster_cmap', ['white', 'darkblue'])
        
        for i, eta in enumerate(eta_values):
            if i < len(axes):
                cluster = clusters[eta]
                axes[i].imshow(cluster, cmap=cmap, origin='lower')
                axes[i].set_title(f'eta = {eta}')
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('y')
        
        # hide unused subplots
        for i in range(len(eta_values), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('eta_analysis/all_clusters.png', dpi=300)
        plt.close()

def analyze_eta_influence():
    """Run simulations and analysis for different eta values"""
    os.makedirs("eta_analysis", exist_ok=True)
    os.makedirs("eta_analysis/data", exist_ok=True)
    os.makedirs("eta_analysis/clusters", exist_ok=True)
    
    eta_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    max_steps = 100
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
    
    print("Analysis complete! Check the eta_analysis directory for results.")