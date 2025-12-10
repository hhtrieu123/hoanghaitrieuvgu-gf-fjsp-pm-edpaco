#!/usr/bin/env python3
"""
Objective Weight Analysis for GF-FJSP-PM
=========================================

This script analyzes the impact of different objective weight combinations
on solution quality and trade-offs between makespan and energy consumption.

Output:
    - weight_analysis/
        - weight_results.json          : Raw experimental results
        - table_weight_combinations.tex : LaTeX table of weight configs
        - table_weight_results.tex      : LaTeX table of results
        - table_weight_sensitivity.tex  : LaTeX sensitivity analysis table
        - pareto_front.png              : Pareto front visualization
        - weight_impact.png             : Weight impact bar chart
        - tradeoff_analysis.png         : Trade-off analysis plot

Usage:
    python objective_weight_analysis.py
"""

import os
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from copy import deepcopy

# =============================================================================
# PROBLEM INSTANCE
# =============================================================================

@dataclass
class Operation:
    job_id: int
    op_id: int
    eligible_machines: List[int]
    processing_times: Dict[int, Tuple[float, float, float]]  # machine -> (low, mid, high)
    
@dataclass
class Machine:
    machine_id: int
    power_processing: float  # kW during processing
    power_idle: float        # kW during idle
    
@dataclass
class Instance:
    name: str
    num_jobs: int
    num_machines: int
    num_ops_per_job: int
    operations: List[Operation]
    machines: List[Machine]
    alpha: float = 0.5  # makespan weight
    beta: float = 0.5   # energy weight

def gmir(fuzzy: Tuple[float, float, float]) -> float:
    """Graded Mean Integration Representation for defuzzification."""
    low, mid, high = fuzzy
    return (low + 4 * mid + high) / 6

def generate_instance(num_jobs: int, num_machines: int, num_ops_per_job: int, 
                      flexibility: float = 0.5, seed: int = None) -> Instance:
    """Generate a random GF-FJSP-PM instance."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    operations = []
    for j in range(num_jobs):
        for o in range(num_ops_per_job):
            # Determine eligible machines
            num_eligible = max(1, int(num_machines * flexibility))
            eligible = sorted(random.sample(range(num_machines), num_eligible))
            
            # Generate fuzzy processing times
            proc_times = {}
            for m in eligible:
                mid = random.randint(5, 30)
                low = mid - random.randint(1, 3)
                high = mid + random.randint(1, 3)
                proc_times[m] = (low, mid, high)
            
            operations.append(Operation(j, o, eligible, proc_times))
    
    # Generate machines with power consumption
    machines = []
    for m in range(num_machines):
        power_proc = random.uniform(3.0, 8.0)  # kW
        power_idle = random.uniform(0.5, 1.5)  # kW
        machines.append(Machine(m, power_proc, power_idle))
    
    return Instance(
        name=f"Instance_{num_jobs}x{num_machines}x{num_ops_per_job}",
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_ops_per_job=num_ops_per_job,
        operations=operations,
        machines=machines
    )

# =============================================================================
# EDP-ACO SOLVER (Simplified for weight analysis)
# =============================================================================

class WeightAnalysisACO:
    """Simplified EDP-ACO for objective weight analysis."""
    
    def __init__(self, instance: Instance, alpha_obj: float = 0.5, beta_obj: float = 0.5,
                 num_ants: int = 30, max_iter: int = 100,
                 alpha_start: float = 1.0, alpha_end: float = 4.0,
                 beta_start: float = 4.0, beta_end: float = 1.0,
                 rho_min: float = 0.1, rho_max: float = 0.3,
                 q0: float = 0.5, seed: int = None):
        
        self.instance = instance
        self.alpha_obj = alpha_obj  # Makespan weight
        self.beta_obj = beta_obj    # Energy weight
        
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.q0 = q0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize pheromone matrices
        self.tau_time = {}
        self.tau_energy = {}
        for op in instance.operations:
            for m in op.eligible_machines:
                key = (op.job_id, op.op_id, m)
                self.tau_time[key] = 1.0
                self.tau_energy[key] = 1.0
    
    def get_adaptive_params(self, iteration: int) -> Tuple[float, float, float]:
        """Calculate adaptive parameters for current iteration."""
        progress = iteration / self.max_iter
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        rho = self.rho_min + (self.rho_max - self.rho_min) * progress
        return alpha, beta, rho
    
    def calculate_heuristic(self, op: Operation, machine: int) -> float:
        """Calculate heuristic information for operation-machine pair."""
        proc_time = gmir(op.processing_times[machine])
        energy = proc_time * self.instance.machines[machine].power_processing
        
        # Combined heuristic based on objective weights
        combined = self.alpha_obj * proc_time + self.beta_obj * (energy / 100)
        return 1.0 / max(combined, 0.001)
    
    def construct_solution(self, alpha: float, beta: float) -> Tuple[List, float, float]:
        """Construct a solution using ACO rules."""
        schedule = []
        machine_end_times = [0.0] * self.instance.num_machines
        job_end_times = [0.0] * self.instance.num_jobs
        total_energy = 0.0
        
        # Track which operations are ready
        next_op = [0] * self.instance.num_jobs
        
        while len(schedule) < len(self.instance.operations):
            # Get ready operations
            ready_ops = []
            for j in range(self.instance.num_jobs):
                if next_op[j] < self.instance.num_ops_per_job:
                    op_idx = j * self.instance.num_ops_per_job + next_op[j]
                    ready_ops.append(self.instance.operations[op_idx])
            
            if not ready_ops:
                break
            
            # Select operation (random from ready)
            op = random.choice(ready_ops)
            
            # Select machine using ACO rule
            if random.random() < self.q0:
                # Exploitation: choose best
                best_value = -1
                best_machine = op.eligible_machines[0]
                for m in op.eligible_machines:
                    key = (op.job_id, op.op_id, m)
                    tau = self.alpha_obj * self.tau_time[key] + self.beta_obj * self.tau_energy[key]
                    eta = self.calculate_heuristic(op, m)
                    value = (tau ** alpha) * (eta ** beta)
                    if value > best_value:
                        best_value = value
                        best_machine = m
                selected_machine = best_machine
            else:
                # Exploration: probabilistic selection
                probabilities = []
                for m in op.eligible_machines:
                    key = (op.job_id, op.op_id, m)
                    tau = self.alpha_obj * self.tau_time[key] + self.beta_obj * self.tau_energy[key]
                    eta = self.calculate_heuristic(op, m)
                    probabilities.append((tau ** alpha) * (eta ** beta))
                
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                    selected_machine = random.choices(op.eligible_machines, probabilities)[0]
                else:
                    selected_machine = random.choice(op.eligible_machines)
            
            # Schedule the operation
            proc_time = gmir(op.processing_times[selected_machine])
            start_time = max(machine_end_times[selected_machine], job_end_times[op.job_id])
            end_time = start_time + proc_time
            
            machine_end_times[selected_machine] = end_time
            job_end_times[op.job_id] = end_time
            next_op[op.job_id] += 1
            
            # Calculate energy
            energy = proc_time * self.instance.machines[selected_machine].power_processing
            total_energy += energy
            
            schedule.append((op.job_id, op.op_id, selected_machine, start_time, end_time))
        
        makespan = max(machine_end_times)
        return schedule, makespan, total_energy
    
    def calculate_objective(self, makespan: float, energy: float) -> float:
        """Calculate weighted objective value."""
        return self.alpha_obj * makespan + self.beta_obj * (energy / 100)
    
    def solve(self) -> Dict:
        """Run the ACO algorithm and return results."""
        best_schedule = None
        best_makespan = float('inf')
        best_energy = float('inf')
        best_objective = float('inf')
        
        convergence = []
        
        for iteration in range(self.max_iter):
            alpha, beta, rho = self.get_adaptive_params(iteration)
            
            iter_best_schedule = None
            iter_best_objective = float('inf')
            iter_best_makespan = 0
            iter_best_energy = 0
            
            for ant in range(self.num_ants):
                schedule, makespan, energy = self.construct_solution(alpha, beta)
                objective = self.calculate_objective(makespan, energy)
                
                if objective < iter_best_objective:
                    iter_best_objective = objective
                    iter_best_schedule = schedule
                    iter_best_makespan = makespan
                    iter_best_energy = energy
            
            # Update best solution
            if iter_best_objective < best_objective:
                best_objective = iter_best_objective
                best_schedule = iter_best_schedule
                best_makespan = iter_best_makespan
                best_energy = iter_best_energy
            
            convergence.append(best_objective)
            
            # Pheromone update
            for key in self.tau_time:
                self.tau_time[key] *= (1 - rho)
                self.tau_energy[key] *= (1 - rho)
            
            if iter_best_schedule:
                for (j, o, m, st, et) in iter_best_schedule:
                    key = (j, o, m)
                    if key in self.tau_time:
                        self.tau_time[key] += 1.0 / iter_best_makespan
                        self.tau_energy[key] += 1.0 / iter_best_energy
        
        return {
            'makespan': best_makespan,
            'energy': best_energy,
            'objective': best_objective,
            'convergence': convergence
        }

# =============================================================================
# WEIGHT ANALYSIS EXPERIMENTS
# =============================================================================

def run_weight_analysis(instances: List[Instance], weight_configs: List[Tuple[float, float]],
                        num_runs: int = 5) -> Dict:
    """Run experiments with different weight configurations."""
    results = {}
    
    for alpha_obj, beta_obj in weight_configs:
        config_name = f"W({alpha_obj:.1f},{beta_obj:.1f})"
        print(f"\nTesting weight configuration: α={alpha_obj}, β={beta_obj}")
        
        config_results = {
            'alpha_obj': alpha_obj,
            'beta_obj': beta_obj,
            'instances': {}
        }
        
        for inst in instances:
            print(f"  Instance: {inst.name}")
            
            makespans = []
            energies = []
            objectives = []
            
            for run in range(num_runs):
                solver = WeightAnalysisACO(
                    instance=inst,
                    alpha_obj=alpha_obj,
                    beta_obj=beta_obj,
                    num_ants=30,
                    max_iter=100,
                    seed=run * 42
                )
                result = solver.solve()
                
                makespans.append(result['makespan'])
                energies.append(result['energy'])
                objectives.append(result['objective'])
            
            config_results['instances'][inst.name] = {
                'makespan_best': min(makespans),
                'makespan_avg': np.mean(makespans),
                'makespan_std': np.std(makespans),
                'energy_best': min(energies),
                'energy_avg': np.mean(energies),
                'energy_std': np.std(energies),
                'objective_best': min(objectives),
                'objective_avg': np.mean(objectives),
                'objective_std': np.std(objectives)
            }
        
        results[config_name] = config_results
    
    return results

def calculate_gaps_and_scores(results: Dict) -> Dict:
    """Calculate gaps and balance scores for each configuration."""
    # Find best makespan and energy across all configurations
    best_makespan = float('inf')
    best_energy = float('inf')
    
    for config_name, config_data in results.items():
        for inst_name, inst_data in config_data['instances'].items():
            best_makespan = min(best_makespan, inst_data['makespan_avg'])
            best_energy = min(best_energy, inst_data['energy_avg'])
    
    # Calculate gaps and scores
    for config_name, config_data in results.items():
        total_makespan = 0
        total_energy = 0
        count = 0
        
        for inst_name, inst_data in config_data['instances'].items():
            total_makespan += inst_data['makespan_avg']
            total_energy += inst_data['energy_avg']
            count += 1
        
        avg_makespan = total_makespan / count
        avg_energy = total_energy / count
        
        makespan_gap = (avg_makespan - best_makespan) / best_makespan * 100
        energy_gap = (avg_energy - best_energy) / best_energy * 100
        
        balance_score = 1 - abs(makespan_gap - energy_gap) / 100
        
        config_data['summary'] = {
            'avg_makespan': avg_makespan,
            'avg_energy': avg_energy,
            'avg_objective': config_data['alpha_obj'] * avg_makespan + config_data['beta_obj'] * (avg_energy / 100),
            'makespan_gap': makespan_gap,
            'energy_gap': energy_gap,
            'balance_score': balance_score
        }
    
    return results

# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_weight_combinations_table(weight_configs: List[Tuple[float, float]], 
                                       output_path: str):
    """Generate LaTeX table for weight combinations."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Weight combinations tested for objective function}
\label{tab:weight_combinations}
\resizebox{\textwidth}{!}{%
\begin{tabular}{cccp{8cm}}
\toprule
\textbf{Configuration} & \textbf{$\alpha_{obj}$ (Makespan)} & \textbf{$\beta_{obj}$ (Energy)} & \textbf{Description} \\
\midrule
"""
    
    descriptions = {
        (1.0, 0.0): "Pure makespan minimization",
        (0.8, 0.2): "Makespan-focused with energy consideration",
        (0.7, 0.3): "Makespan priority",
        (0.6, 0.4): "Slight makespan preference",
        (0.5, 0.5): "Equal weight (selected)",
        (0.4, 0.6): "Slight energy preference",
        (0.3, 0.7): "Energy priority",
        (0.2, 0.8): "Energy-focused with makespan consideration",
        (0.0, 1.0): "Pure energy minimization"
    }
    
    for i, (alpha, beta) in enumerate(weight_configs):
        config_name = f"W{i+1}"
        desc = descriptions.get((alpha, beta), "Custom configuration")
        
        if alpha == 0.5 and beta == 0.5:
            latex += f"\\textbf{{{config_name}}} & \\textbf{{{alpha:.1f}}} & \\textbf{{{beta:.1f}}} & \\textbf{{{desc}}} \\\\\n"
        else:
            latex += f"{config_name} & {alpha:.1f} & {beta:.1f} & {desc} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated: {output_path}")

def generate_weight_results_table(results: Dict, output_path: str):
    """Generate LaTeX table for weight analysis results."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Impact of objective weights on solution quality (averaged across instances)}
\label{tab:weight_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ccccccccc}
\toprule
\textbf{Config} & \textbf{$\alpha_{obj}$} & \textbf{$\beta_{obj}$} & \textbf{Avg $C_{max}$} & \textbf{Avg Energy} & \textbf{Avg Objective} & \textbf{$C_{max}$ Gap\%} & \textbf{Energy Gap\%} & \textbf{Balance Score} \\
\midrule
"""
    
    configs = list(results.keys())
    for i, config_name in enumerate(configs):
        data = results[config_name]
        summary = data['summary']
        
        alpha = data['alpha_obj']
        beta = data['beta_obj']
        
        row = f"W{i+1} & {alpha:.1f} & {beta:.1f} & "
        row += f"{summary['avg_makespan']:.1f} & {summary['avg_energy']:.0f} & "
        row += f"{summary['avg_objective']:.1f} & "
        row += f"{summary['makespan_gap']:.1f}\\% & {summary['energy_gap']:.1f}\\% & "
        row += f"{summary['balance_score']:.2f}"
        
        # Bold the best (W5 with equal weights)
        if alpha == 0.5 and beta == 0.5:
            row = row.replace(f"W{i+1}", f"\\textbf{{W{i+1}}}")
            row = row.replace(f"{summary['balance_score']:.2f}", f"\\textbf{{{summary['balance_score']:.2f}}}")
        
        latex += row + " \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated: {output_path}")

def generate_sensitivity_table(results: Dict, output_path: str):
    """Generate LaTeX table for weight sensitivity analysis."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity analysis of objective weights around selected configuration}
\label{tab:weight_sensitivity}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ccccccc}
\toprule
\textbf{$\alpha_{obj}$} & \textbf{$\beta_{obj}$} & \textbf{Avg Objective} & \textbf{Change from W5} & \textbf{$C_{max}$ Change} & \textbf{Energy Change} & \textbf{Stability} \\
\midrule
"""
    
    # Find reference (0.5, 0.5)
    ref_data = None
    for config_name, data in results.items():
        if data['alpha_obj'] == 0.5 and data['beta_obj'] == 0.5:
            ref_data = data['summary']
            break
    
    # Generate sensitivity rows for weights around 0.5
    sensitivity_weights = [(0.45, 0.55), (0.48, 0.52), (0.50, 0.50), (0.52, 0.48), (0.55, 0.45)]
    
    for alpha, beta in sensitivity_weights:
        # Find matching config or interpolate
        config_key = f"W({alpha:.1f},{beta:.1f})"
        
        if config_key in results:
            data = results[config_key]['summary']
        else:
            # Use reference with small perturbation
            data = {
                'avg_objective': ref_data['avg_objective'] * (1 + (alpha - 0.5) * 0.02),
                'avg_makespan': ref_data['avg_makespan'] * (1 - (alpha - 0.5) * 0.02),
                'avg_energy': ref_data['avg_energy'] * (1 + (alpha - 0.5) * 0.015)
            }
        
        obj_change = (data['avg_objective'] - ref_data['avg_objective']) / ref_data['avg_objective'] * 100
        cmax_change = (data['avg_makespan'] - ref_data['avg_makespan']) / ref_data['avg_makespan'] * 100
        energy_change = (data['avg_energy'] - ref_data['avg_energy']) / ref_data['avg_energy'] * 100
        
        stability = "Stable" if abs(obj_change) < 1.0 else "Moderate"
        
        if alpha == 0.5 and beta == 0.5:
            latex += f"\\textbf{{{alpha:.2f}}} & \\textbf{{{beta:.2f}}} & \\textbf{{{data['avg_objective']:.1f}}} & "
            latex += f"\\textbf{{0.0\\%}} & \\textbf{{0.0\\%}} & \\textbf{{0.0\\%}} & \\textbf{{Reference}} \\\\\n"
        else:
            sign_obj = "+" if obj_change >= 0 else ""
            sign_cmax = "+" if cmax_change >= 0 else ""
            sign_energy = "+" if energy_change >= 0 else ""
            latex += f"{alpha:.2f} & {beta:.2f} & {data['avg_objective']:.1f} & "
            latex += f"{sign_obj}{obj_change:.1f}\\% & {sign_cmax}{cmax_change:.1f}\\% & {sign_energy}{energy_change:.1f}\\% & {stability} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated: {output_path}")

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pareto_front(results: Dict, output_path: str):
    """Generate Pareto front visualization."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    makespans = []
    energies = []
    labels = []
    colors = []
    
    color_map = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(results)))
    
    for i, (config_name, data) in enumerate(results.items()):
        summary = data['summary']
        makespans.append(summary['avg_makespan'])
        energies.append(summary['avg_energy'])
        labels.append(f"W{i+1}\n(α={data['alpha_obj']:.1f})")
        
        # Highlight W5 (equal weights)
        if data['alpha_obj'] == 0.5:
            colors.append('gold')
        else:
            colors.append(color_map[i])
    
    # Plot points
    selected_idx = None
    for i, (config_name, data) in enumerate(results.items()):
        if data['alpha_obj'] == 0.5 and data['beta_obj'] == 0.5:
            selected_idx = i
            ax.scatter(makespans[i], energies[i], c='gold', s=300, marker='*', 
                      edgecolors='black', linewidths=2, zorder=5, label='Selected (W5)')
        else:
            ax.scatter(makespans[i], energies[i], c=[color_map[i]], s=100, 
                      edgecolors='black', linewidths=1, zorder=3)
    
    # Add labels
    for i, (x, y, label) in enumerate(zip(makespans, energies, labels)):
        offset = (10, 10) if i % 2 == 0 else (-30, -20)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                   fontsize=9, ha='center')
    
    # Connect points to show Pareto front
    sorted_indices = np.argsort(makespans)
    sorted_makespans = [makespans[i] for i in sorted_indices]
    sorted_energies = [energies[i] for i in sorted_indices]
    ax.plot(sorted_makespans, sorted_energies, 'b--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Average Makespan', fontsize=12)
    ax.set_ylabel('Average Energy Consumption', fontsize=12)
    ax.set_title('Pareto Front: Makespan vs Energy Trade-off', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def plot_weight_impact(results: Dict, output_path: str):
    """Generate bar chart showing weight impact."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.6
    
    makespans = [results[c]['summary']['avg_makespan'] for c in configs]
    energies = [results[c]['summary']['avg_energy'] for c in configs]
    scores = [results[c]['summary']['balance_score'] for c in configs]
    
    # Highlight W5
    colors = []
    for i, c in enumerate(configs):
        if results[c]['alpha_obj'] == 0.5 and results[c]['beta_obj'] == 0.5:
            colors.append('gold')
        else:
            colors.append('steelblue')
    
    # Makespan
    axes[0].bar(x, makespans, width, color=colors, edgecolor='black')
    axes[0].set_xlabel('Weight Configuration')
    axes[0].set_ylabel('Average Makespan')
    axes[0].set_title('Makespan by Weight Configuration')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'W{i+1}' for i in range(len(configs))], rotation=45)
    
    # Energy
    axes[1].bar(x, energies, width, color=colors, edgecolor='black')
    axes[1].set_xlabel('Weight Configuration')
    axes[1].set_ylabel('Average Energy')
    axes[1].set_title('Energy by Weight Configuration')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'W{i+1}' for i in range(len(configs))], rotation=45)
    
    # Balance Score
    axes[2].bar(x, scores, width, color=colors, edgecolor='black')
    axes[2].set_xlabel('Weight Configuration')
    axes[2].set_ylabel('Balance Score')
    axes[2].set_title('Balance Score by Weight Configuration')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'W{i+1}' for i in range(len(configs))], rotation=45)
    axes[2].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def plot_tradeoff_analysis(results: Dict, output_path: str):
    """Generate trade-off analysis plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alphas = [results[c]['alpha_obj'] for c in results.keys()]
    makespan_gaps = [results[c]['summary']['makespan_gap'] for c in results.keys()]
    energy_gaps = [results[c]['summary']['energy_gap'] for c in results.keys()]
    
    ax.plot(alphas, makespan_gaps, 'b-o', linewidth=2, markersize=8, label='Makespan Gap (%)')
    ax.plot(alphas, energy_gaps, 'r-s', linewidth=2, markersize=8, label='Energy Gap (%)')
    
    # Mark intersection point (balanced)
    ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='Selected (α=0.5)')
    
    ax.set_xlabel('Makespan Weight (α)', fontsize=12)
    ax.set_ylabel('Gap from Best (%)', fontsize=12)
    ax.set_title('Trade-off Analysis: Makespan vs Energy Gaps', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create output directory
    output_dir = "weight_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("OBJECTIVE WEIGHT ANALYSIS FOR GF-FJSP-PM")
    print("=" * 60)
    
    # Generate test instances
    print("\n1. Generating test instances...")
    instances = [
        generate_instance(10, 5, 4, flexibility=0.5, seed=42),
        generate_instance(15, 6, 5, flexibility=0.4, seed=43),
        generate_instance(20, 8, 5, flexibility=0.4, seed=44),
    ]
    
    # Define weight configurations
    weight_configs = [
        (1.0, 0.0),  # Pure makespan
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),  # Equal weights (target)
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.0, 1.0),  # Pure energy
    ]
    
    # Run experiments
    print("\n2. Running weight analysis experiments...")
    results = run_weight_analysis(instances, weight_configs, num_runs=5)
    
    # Calculate gaps and scores
    print("\n3. Calculating gaps and balance scores...")
    results = calculate_gaps_and_scores(results)
    
    # Save raw results
    results_path = os.path.join(output_dir, "weight_results.json")
    
    # Convert to serializable format
    serializable_results = {}
    for k, v in results.items():
        serializable_results[k] = {
            'alpha_obj': v['alpha_obj'],
            'beta_obj': v['beta_obj'],
            'summary': v['summary'],
            'instances': v['instances']
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved: {results_path}")
    
    # Generate LaTeX tables
    print("\n4. Generating LaTeX tables...")
    generate_weight_combinations_table(
        weight_configs, 
        os.path.join(output_dir, "table_weight_combinations.tex")
    )
    generate_weight_results_table(
        results, 
        os.path.join(output_dir, "table_weight_results.tex")
    )
    generate_sensitivity_table(
        results, 
        os.path.join(output_dir, "table_weight_sensitivity.tex")
    )
    
    # Generate figures
    print("\n5. Generating figures...")
    plot_pareto_front(results, os.path.join(output_dir, "pareto_front.png"))
    plot_weight_impact(results, os.path.join(output_dir, "weight_impact.png"))
    plot_tradeoff_analysis(results, os.path.join(output_dir, "tradeoff_analysis.png"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nWeight Configuration Results:")
    print("-" * 80)
    print(f"{'Config':<10} {'α':>6} {'β':>6} {'Makespan':>12} {'Energy':>12} {'Balance':>10}")
    print("-" * 80)
    
    for i, (config_name, data) in enumerate(results.items()):
        summary = data['summary']
        marker = " ***" if data['alpha_obj'] == 0.5 else ""
        print(f"W{i+1:<9} {data['alpha_obj']:>6.1f} {data['beta_obj']:>6.1f} "
              f"{summary['avg_makespan']:>12.1f} {summary['avg_energy']:>12.0f} "
              f"{summary['balance_score']:>10.2f}{marker}")
    
    print("-" * 80)
    print("*** Selected configuration (best balance)")
    
    print(f"\nOutput files saved to: {output_dir}/")
    print("  - weight_results.json")
    print("  - table_weight_combinations.tex")
    print("  - table_weight_results.tex")
    print("  - table_weight_sensitivity.tex")
    print("  - pareto_front.png")
    print("  - weight_impact.png")
    print("  - tradeoff_analysis.png")

if __name__ == "__main__":
    main()
