#!/usr/bin/env python3
"""
Convergence Plotter for FJSP-ACO Results

This script reads convergence CSV files and generates publication-quality plots
showing convergence over iterations and time.

Usage:
    python plot_convergence.py                    # Plot all convergence_*.csv files
    python plot_convergence.py Mk01 Mk02 Mk03    # Plot specific instances
    python plot_convergence.py --combined         # Create combined plot for all instances
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150


def load_convergence_data(filename):
    """Load convergence data from CSV file."""
    data = {'iterations': [], 'time': [], 'runs': [], 'mean': []}
    
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
        num_runs = len([h for h in header if h.startswith('Run') and not h.endswith('Time')])
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            
            data['iterations'].append(int(parts[0]))
            data['time'].append(float(parts[1]))
            
            run_values = []
            for i in range(2, 2 + num_runs):
                if i < len(parts) and parts[i]:
                    run_values.append(float(parts[i]))
            data['runs'].append(run_values)
            
            # Mean is the last column
            if parts[-1]:
                data['mean'].append(float(parts[-1]))
    
    return data


def plot_single_instance(filename, output_dir='plots'):
    """Plot convergence for a single instance."""
    instance_name = os.path.basename(filename).replace('convergence_', '').replace('.csv', '')
    data = load_convergence_data(filename)
    
    if not data['iterations']:
        print(f"  Warning: No data in {filename}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots (iterations and time)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    iterations = np.array(data['iterations'])
    times = np.array(data['time'])
    mean_values = np.array(data['mean'])
    
    # Calculate min/max for runs
    runs_array = np.array(data['runs'])
    run_min = np.min(runs_array, axis=1)
    run_max = np.max(runs_array, axis=1)
    
    # Plot 1: Convergence vs Iterations
    ax1 = axes[0]
    ax1.fill_between(iterations, run_min, run_max, alpha=0.3, color='blue', label='Min-Max range')
    ax1.plot(iterations, mean_values, 'b-', linewidth=2, label='Mean')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Makespan')
    ax1.set_title(f'{instance_name} - Convergence vs Iteration')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot 2: Convergence vs Time
    ax2 = axes[1]
    ax2.fill_between(times, run_min, run_max, alpha=0.3, color='green', label='Min-Max range')
    ax2.plot(times, mean_values, 'g-', linewidth=2, label='Mean')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Makespan')
    ax2.set_title(f'{instance_name} - Convergence vs Time')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'convergence_{instance_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def plot_combined(files, output_dir='plots', group_name='all'):
    """Create combined convergence plot for multiple instances."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine grid size
    n = len(files)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, filename in enumerate(files):
        instance_name = os.path.basename(filename).replace('convergence_', '').replace('.csv', '')
        data = load_convergence_data(filename)
        
        if not data['iterations']:
            continue
        
        ax = axes[idx]
        iterations = np.array(data['iterations'])
        mean_values = np.array(data['mean'])
        runs_array = np.array(data['runs'])
        run_min = np.min(runs_array, axis=1)
        run_max = np.max(runs_array, axis=1)
        
        ax.fill_between(iterations, run_min, run_max, alpha=0.3, color='blue')
        ax.plot(iterations, mean_values, 'b-', linewidth=1.5)
        ax.set_title(instance_name, fontsize=10)
        ax.set_xlabel('Iteration', fontsize=8)
        ax.set_ylabel('Makespan', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide empty subplots
    for idx in range(len(files), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Convergence Curves - {group_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'convergence_combined_{group_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved combined plot: {output_file}")


def plot_comparison_time(files, output_dir='plots'):
    """Plot convergence vs time for all instances on one plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(files)))
    
    for idx, filename in enumerate(files):
        instance_name = os.path.basename(filename).replace('convergence_', '').replace('.csv', '')
        data = load_convergence_data(filename)
        
        if not data['time']:
            continue
        
        times = np.array(data['time'])
        mean_values = np.array(data['mean'])
        
        # Normalize to percentage of initial value
        if mean_values[0] > 0:
            normalized = (mean_values / mean_values[0]) * 100
            ax.plot(times, normalized, '-', linewidth=1.5, color=colors[idx], label=instance_name)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Makespan (% of initial)')
    ax.set_title('Convergence Comparison - All Instances')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'convergence_comparison_time.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot: {output_file}")


def main():
    print("=" * 60)
    print("        CONVERGENCE PLOT GENERATOR")
    print("=" * 60)
    
    # Find all convergence files
    all_files = sorted(glob.glob('convergence_*.csv'))
    
    if not all_files:
        print("\nNo convergence_*.csv files found in current directory.")
        print("Run the FJSP-ACO program first to generate convergence data.")
        return
    
    print(f"\nFound {len(all_files)} convergence files.")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--combined':
            # Separate Mk and LA files
            mk_files = [f for f in all_files if 'Mk' in f]
            la_files = [f for f in all_files if 'LA' in f or 'la' in f]
            
            if mk_files:
                print("\nPlotting combined Brandimarte instances...")
                plot_combined(mk_files, group_name='Brandimarte')
            
            if la_files:
                print("\nPlotting combined Lawrence instances...")
                plot_combined(la_files, group_name='Lawrence')
            
            if all_files:
                print("\nPlotting time comparison...")
                plot_comparison_time(all_files)
        else:
            # Plot specific instances
            for name in sys.argv[1:]:
                matching = [f for f in all_files if name in f]
                for f in matching:
                    print(f"\nPlotting {f}...")
                    plot_single_instance(f)
    else:
        # Plot all individual instances
        print("\nGenerating individual plots...")
        for f in all_files:
            print(f"\nProcessing {f}...")
            plot_single_instance(f)
        
        # Also create combined plots
        mk_files = [f for f in all_files if 'Mk' in f]
        la_files = [f for f in all_files if 'LA' in f or 'la' in f]
        
        if mk_files:
            print("\nCreating combined Brandimarte plot...")
            plot_combined(mk_files, group_name='Brandimarte')
        
        if la_files:
            print("\nCreating combined Lawrence plot...")
            plot_combined(la_files, group_name='Lawrence')
        
        if len(all_files) > 1:
            print("\nCreating time comparison plot...")
            plot_comparison_time(all_files)
    
    print("\n" + "=" * 60)
    print("Plotting complete! Check the 'plots' directory.")
    print("=" * 60)


if __name__ == '__main__':
    main()
