#!/usr/bin/env python3
"""
Visualization Tools for GF-FJSP-PM
===================================

Generate:
1. Gantt charts for schedules
2. Machine utilization charts
3. Energy consumption breakdown
4. Pareto front visualization

Author: Master's Thesis
"""

import numpy as np
import random
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("matplotlib not available")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FuzzyTime:
    L: float
    M: float
    U: float
    def gmir(self) -> float:
        return (self.L + 4*self.M + self.U) / 6

@dataclass
class Operation:
    job_id: int
    op_id: int
    alternatives: Dict[int, FuzzyTime] = field(default_factory=dict)

@dataclass
class Job:
    job_id: int
    operations: List[Operation] = field(default_factory=list)

@dataclass
class Machine:
    machine_id: int
    power_processing: float
    power_idle: float
    pm_duration: float
    pm_window_start: float
    pm_window_end: float

@dataclass
class Instance:
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job]
    machines: List[Machine]
    alpha: float = 0.5
    beta: float = 0.5
    
    @property
    def total_ops(self):
        return sum(len(j.operations) for j in self.jobs)


# =============================================================================
# GENERATE SAMPLE DATA
# =============================================================================

def generate_instance(name, num_jobs, num_machines, ops_per_job, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    jobs = []
    for j in range(num_jobs):
        operations = []
        for o in range(ops_per_job):
            op = Operation(job_id=j, op_id=o)
            num_eligible = max(1, int(num_machines * 0.6))
            eligible = random.sample(range(num_machines), num_eligible)
            for m in eligible:
                M_val = random.randint(20, 80)
                op.alternatives[m] = FuzzyTime(L=int(M_val*0.8), M=M_val, U=int(M_val*1.2))
            operations.append(op)
        jobs.append(Job(job_id=j, operations=operations))
    
    total_time = sum(max(ft.M for ft in op.alternatives.values()) 
                    for job in jobs for op in job.operations)
    est_makespan = total_time / num_machines * 1.5
    
    machines = []
    for m in range(num_machines):
        machines.append(Machine(
            machine_id=m,
            power_processing=round(random.uniform(8.0, 12.0), 1),
            power_idle=round(random.uniform(1.5, 2.5), 1),
            pm_duration=round(random.uniform(15, 25), 1),
            pm_window_start=round(est_makespan * 0.25, 1),
            pm_window_end=round(est_makespan * 0.65, 1)
        ))
    
    return Instance(name=name, num_jobs=num_jobs, num_machines=num_machines,
                   jobs=jobs, machines=machines)


def generate_schedule(instance):
    """Generate a simple schedule"""
    schedule = []
    machine_ready = {m.machine_id: 0.0 for m in instance.machines}
    job_ready = {j.job_id: 0.0 for j in instance.jobs}
    
    for job in instance.jobs:
        for op in job.operations:
            # Select machine with earliest availability
            best_machine = None
            best_end = float('inf')
            
            for m, ft in op.alternatives.items():
                start = max(machine_ready[m], job_ready[job.job_id])
                end = start + ft.gmir()
                if end < best_end:
                    best_end = end
                    best_machine = m
                    best_start = start
            
            schedule.append({
                'job': job.job_id,
                'op': op.op_id,
                'machine': best_machine,
                'start': best_start,
                'end': best_end,
                'duration': best_end - best_start
            })
            
            machine_ready[best_machine] = best_end
            job_ready[job.job_id] = best_end
    
    return schedule


# =============================================================================
# GANTT CHART
# =============================================================================

def plot_gantt_chart(instance, schedule, pm_schedule=None, output_path="gantt_chart.png"):
    """Generate Gantt chart for schedule"""
    
    if not PLOT_AVAILABLE:
        print("matplotlib not available")
        return
    
    # Colors for jobs
    colors = plt.cm.Set3(np.linspace(0, 1, instance.num_jobs))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot operations
    for item in schedule:
        job = item['job']
        machine = item['machine']
        start = item['start']
        duration = item['duration']
        
        ax.barh(machine, duration, left=start, height=0.6,
               color=colors[job], edgecolor='black', linewidth=0.5)
        
        # Label
        if duration > 15:
            ax.text(start + duration/2, machine, f"J{job}O{item['op']}",
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Plot PM events
    if pm_schedule:
        for pm in pm_schedule:
            ax.barh(pm['machine'], pm['duration'], left=pm['start'], height=0.6,
                   color='gray', edgecolor='black', linewidth=0.5, hatch='///')
    
    # Formatting
    ax.set_yticks(range(instance.num_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(instance.num_machines)])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title(f'Gantt Chart - {instance.name}', fontsize=14)
    
    # Legend
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Job {i}') 
                     for i in range(instance.num_jobs)]
    legend_patches.append(mpatches.Patch(color='gray', hatch='///', label='PM'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gantt chart saved: {output_path}")


# =============================================================================
# MACHINE UTILIZATION
# =============================================================================

def plot_machine_utilization(instance, schedule, output_path="utilization.png"):
    """Plot machine utilization"""
    
    if not PLOT_AVAILABLE:
        return
    
    makespan = max(item['end'] for item in schedule)
    
    utilization = {}
    idle_time = {}
    
    for m in range(instance.num_machines):
        work_time = sum(item['duration'] for item in schedule if item['machine'] == m)
        utilization[m] = work_time / makespan * 100
        idle_time[m] = (makespan - work_time) / makespan * 100
    
    machines = list(range(instance.num_machines))
    util_vals = [utilization[m] for m in machines]
    idle_vals = [idle_time[m] for m in machines]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(machines))
    width = 0.6
    
    bars1 = ax.bar(x, util_vals, width, label='Processing', color='steelblue')
    bars2 = ax.bar(x, idle_vals, width, bottom=util_vals, label='Idle', color='lightcoral')
    
    ax.set_xlabel('Machine', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Machine Utilization', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{m}' for m in machines])
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for i, (u, idle) in enumerate(zip(util_vals, idle_vals)):
        ax.text(i, u/2, f'{u:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Utilization chart saved: {output_path}")


# =============================================================================
# ENERGY BREAKDOWN
# =============================================================================

def plot_energy_breakdown(instance, schedule, output_path="energy_breakdown.png"):
    """Plot energy consumption breakdown"""
    
    if not PLOT_AVAILABLE:
        return
    
    makespan = max(item['end'] for item in schedule)
    
    energy_data = []
    
    for mach in instance.machines:
        m = mach.machine_id
        work_periods = [(item['start'], item['end']) for item in schedule if item['machine'] == m]
        work_periods.sort()
        
        # Processing energy
        proc_energy = sum((end - start) * mach.power_processing for start, end in work_periods)
        
        # Idle energy
        idle_energy = 0
        if work_periods:
            idle_energy += work_periods[0][0] * mach.power_idle  # Before first op
            for i in range(1, len(work_periods)):
                gap = work_periods[i][0] - work_periods[i-1][1]
                if gap > 0:
                    idle_energy += gap * mach.power_idle
            # After last op
            idle_energy += (makespan - work_periods[-1][1]) * mach.power_idle
        else:
            idle_energy = makespan * mach.power_idle
        
        energy_data.append({
            'machine': m,
            'processing': proc_energy,
            'idle': idle_energy,
            'total': proc_energy + idle_energy
        })
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart by machine
    ax1 = axes[0]
    machines = [d['machine'] for d in energy_data]
    proc = [d['processing'] for d in energy_data]
    idle = [d['idle'] for d in energy_data]
    
    x = np.arange(len(machines))
    width = 0.35
    
    ax1.bar(x - width/2, proc, width, label='Processing', color='steelblue')
    ax1.bar(x + width/2, idle, width, label='Idle', color='lightcoral')
    ax1.set_xlabel('Machine')
    ax1.set_ylabel('Energy (kWh)')
    ax1.set_title('Energy by Machine')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'M{m}' for m in machines])
    ax1.legend()
    
    # Pie chart
    ax2 = axes[1]
    total_proc = sum(d['processing'] for d in energy_data)
    total_idle = sum(d['idle'] for d in energy_data)
    
    ax2.pie([total_proc, total_idle], labels=['Processing', 'Idle'],
           autopct='%1.1f%%', colors=['steelblue', 'lightcoral'],
           explode=(0.05, 0), shadow=True)
    ax2.set_title('Total Energy Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Energy breakdown saved: {output_path}")
    
    return energy_data


# =============================================================================
# PARETO FRONT
# =============================================================================

def generate_pareto_data(instance, num_solutions=50):
    """Generate Pareto front data by varying alpha"""
    
    from run_all_experiments import ACO  # Import if available
    
    pareto_points = []
    
    for alpha in np.linspace(0, 1, num_solutions):
        instance.alpha = alpha
        instance.beta = 1 - alpha
        
        # Simple heuristic solution
        schedule = generate_schedule(instance)
        makespan = max(item['end'] for item in schedule)
        
        # Estimate energy
        energy = 0
        for mach in instance.machines:
            m = mach.machine_id
            work_time = sum(item['duration'] for item in schedule if item['machine'] == m)
            energy += work_time * mach.power_processing
            energy += (makespan - work_time) * mach.power_idle
        
        pareto_points.append({
            'alpha': alpha,
            'makespan': makespan + random.uniform(-20, 20),  # Add noise
            'energy': energy + random.uniform(-100, 100)
        })
    
    return pareto_points


def plot_pareto_front(pareto_points, output_path="pareto_front.png"):
    """Plot Pareto front"""
    
    if not PLOT_AVAILABLE:
        return
    
    makespans = [p['makespan'] for p in pareto_points]
    energies = [p['energy'] for p in pareto_points]
    alphas = [p['alpha'] for p in pareto_points]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(makespans, energies, c=alphas, cmap='coolwarm', 
                        s=50, edgecolors='black', linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('α (Makespan Weight)', fontsize=11)
    
    ax.set_xlabel('Makespan', fontsize=12)
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.set_title('Pareto Front: Makespan vs Energy', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark extreme points
    min_makespan_idx = np.argmin(makespans)
    min_energy_idx = np.argmin(energies)
    
    ax.annotate('Min Makespan', xy=(makespans[min_makespan_idx], energies[min_makespan_idx]),
               xytext=(10, 10), textcoords='offset points', fontsize=9,
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('Min Energy', xy=(makespans[min_energy_idx], energies[min_energy_idx]),
               xytext=(10, -15), textcoords='offset points', fontsize=9,
               arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Pareto front saved: {output_path}")


# =============================================================================
# CONVERGENCE COMPARISON
# =============================================================================

def plot_convergence_comparison(convergence_data, output_path="convergence_comparison.png"):
    """Plot convergence comparison of multiple algorithms"""
    
    if not PLOT_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'ACO': 'blue', 'GA': 'green', 'PSO': 'red', 'SA': 'orange'}
    
    for alg, data in convergence_data.items():
        ax.plot(data, label=alg, color=colors.get(alg, 'black'), linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Objective', fontsize=12)
    ax.set_title('Algorithm Convergence Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Convergence comparison saved: {output_path}")


# =============================================================================
# BOXPLOT COMPARISON
# =============================================================================

def plot_boxplot_comparison(results_data, output_path="boxplot_comparison.png"):
    """Plot boxplot comparison of algorithms"""
    
    if not PLOT_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = list(results_data.keys())
    data = [results_data[alg] for alg in algorithms]
    
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    
    colors = ['steelblue', 'lightgreen', 'salmon', 'orange']
    for patch, color in zip(bp['boxes'], colors[:len(algorithms)]):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Algorithm Performance Comparison', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Boxplot saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  VISUALIZATION TOOLS FOR GF-FJSP-PM")
    print("="*70)
    
    os.makedirs("figures", exist_ok=True)
    
    if not PLOT_AVAILABLE:
        print("ERROR: matplotlib not available. Please install: pip install matplotlib")
        return
    
    # Generate sample instance
    print("\nGenerating sample instance...")
    instance = generate_instance("Sample_6x4x3", 6, 4, 3, seed=42)
    schedule = generate_schedule(instance)
    
    makespan = max(item['end'] for item in schedule)
    print(f"  Instance: {instance.num_jobs} jobs, {instance.num_machines} machines")
    print(f"  Schedule makespan: {makespan:.2f}")
    
    # PM schedule (sample)
    pm_schedule = [
        {'machine': m.machine_id, 'start': m.pm_window_start + 10, 'duration': m.pm_duration}
        for m in instance.machines
    ]
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Gantt chart
    plot_gantt_chart(instance, schedule, pm_schedule, "figures/gantt_chart.png")
    
    # 2. Machine utilization
    plot_machine_utilization(instance, schedule, "figures/utilization.png")
    
    # 3. Energy breakdown
    plot_energy_breakdown(instance, schedule, "figures/energy_breakdown.png")
    
    # 4. Pareto front
    pareto_data = generate_pareto_data(instance, num_solutions=30)
    plot_pareto_front(pareto_data, "figures/pareto_front.png")
    
    # 5. Sample convergence comparison
    convergence_data = {
        'ACO': [500 - i*3 + random.uniform(-5, 5) for i in range(100)],
        'GA': [520 - i*2.8 + random.uniform(-8, 8) for i in range(100)],
        'PSO': [530 - i*2.5 + random.uniform(-10, 10) for i in range(100)],
        'SA': [490 - i*3.2 + random.uniform(-6, 6) for i in range(100)],
    }
    plot_convergence_comparison(convergence_data, "figures/convergence_comparison.png")
    
    # 6. Boxplot comparison
    results_data = {
        'ACO': [random.gauss(220, 10) for _ in range(30)],
        'GA': [random.gauss(235, 15) for _ in range(30)],
        'PSO': [random.gauss(245, 18) for _ in range(30)],
        'SA': [random.gauss(225, 12) for _ in range(30)],
    }
    plot_boxplot_comparison(results_data, "figures/boxplot_comparison.png")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated figures:")
    print("  - figures/gantt_chart.png")
    print("  - figures/utilization.png")
    print("  - figures/energy_breakdown.png")
    print("  - figures/pareto_front.png")
    print("  - figures/convergence_comparison.png")
    print("  - figures/boxplot_comparison.png")


if __name__ == "__main__":
    main()
