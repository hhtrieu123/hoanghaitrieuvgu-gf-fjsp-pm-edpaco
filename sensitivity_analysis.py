#!/usr/bin/env python3
"""
Sensitivity Analysis and Parameter Tuning
==========================================

Analyze the effect of:
1. Alpha/Beta weights on objective
2. Fuzziness levels on solution quality
3. Algorithm parameters (ACO, GA, PSO, SA)
4. Problem size scaling

Author: Master's Thesis
"""

import numpy as np
import random
import time
import json
import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from copy import deepcopy

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES (same as before)
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

@dataclass
class Solution:
    schedule: List[Tuple[int, int, int]]
    makespan: float = 0.0
    energy: float = 0.0
    objective: float = 0.0


# =============================================================================
# GENERATOR AND DECODER
# =============================================================================

def generate_instance(name, num_jobs, num_machines, ops_per_job, 
                      flexibility=0.5, fuzziness=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    jobs = []
    for j in range(num_jobs):
        operations = []
        for o in range(ops_per_job):
            op = Operation(job_id=j, op_id=o)
            num_eligible = max(1, int(num_machines * flexibility))
            eligible = random.sample(range(num_machines), num_eligible)
            for m in eligible:
                M_val = random.randint(10, 100)
                L_val = max(1, int(M_val * (1 - fuzziness)))
                U_val = int(M_val * (1 + fuzziness))
                op.alternatives[m] = FuzzyTime(L=L_val, M=M_val, U=U_val)
            operations.append(op)
        jobs.append(Job(job_id=j, operations=operations))
    
    total_time = sum(max(ft.M for ft in op.alternatives.values()) 
                    for job in jobs for op in job.operations)
    est_makespan = total_time / num_machines * 1.5
    
    machines = []
    for m in range(num_machines):
        machines.append(Machine(
            machine_id=m,
            power_processing=round(random.uniform(5.0, 15.0), 1),
            power_idle=round(random.uniform(1.0, 3.0), 1),
            pm_duration=round(random.uniform(10, 30), 1),
            pm_window_start=round(est_makespan * 0.2, 1),
            pm_window_end=round(est_makespan * 0.7, 1)
        ))
    
    return Instance(name=name, num_jobs=num_jobs, num_machines=num_machines,
                   jobs=jobs, machines=machines)


def decode(instance, assignment):
    machine_ready = {m.machine_id: 0.0 for m in instance.machines}
    job_ready = {j.job_id: 0.0 for j in instance.jobs}
    start_times, end_times = {}, {}
    
    for job_id, op_id, mach_id in assignment:
        op = instance.jobs[job_id].operations[op_id]
        proc_time = op.alternatives[mach_id].gmir()
        start = max(machine_ready[mach_id], job_ready[job_id])
        end = start + proc_time
        start_times[(job_id, op_id)] = start
        end_times[(job_id, op_id)] = end
        machine_ready[mach_id] = end
        job_ready[job_id] = end
    
    makespan = max(end_times.values()) if end_times else 0
    
    # Energy
    energy_proc, energy_idle = 0.0, 0.0
    machine_work = {m.machine_id: [] for m in instance.machines}
    for job_id, op_id, mach_id in assignment:
        machine_work[mach_id].append((start_times[(job_id, op_id)], end_times[(job_id, op_id)]))
    
    for mach in instance.machines:
        mid = mach.machine_id
        work_periods = sorted(machine_work[mid])
        for start, end in work_periods:
            energy_proc += mach.power_processing * (end - start)
        if work_periods:
            energy_idle += mach.power_idle * work_periods[0][0]
            for i in range(1, len(work_periods)):
                gap = work_periods[i][0] - work_periods[i-1][1]
                if gap > 0:
                    energy_idle += mach.power_idle * gap
    
    energy = energy_proc + energy_idle
    objective = instance.alpha * makespan + instance.beta * energy / 100
    
    return Solution(schedule=assignment, makespan=makespan, energy=energy, objective=objective)


def repair_precedence(instance, assignment):
    repaired = []
    job_next_op = {j.job_id: 0 for j in instance.jobs}
    remaining = list(assignment)
    max_iter = len(remaining) * 2
    it = 0
    while remaining and it < max_iter:
        it += 1
        for i, (job_id, op_id, machine) in enumerate(remaining):
            if op_id == job_next_op[job_id]:
                repaired.append((job_id, op_id, machine))
                job_next_op[job_id] += 1
                remaining.pop(i)
                break
    return repaired


# =============================================================================
# ACO WITH CONFIGURABLE PARAMETERS
# =============================================================================

class ACO:
    def __init__(self, instance, num_ants=20, max_iter=100, alpha=1.0, beta=2.0, rho=0.1):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.pheromone = {}
        for job in instance.jobs:
            for op in job.operations:
                for m in op.alternatives.keys():
                    self.pheromone[(job.job_id, op.op_id, m)] = 1.0
    
    def construct_solution(self):
        assignment = []
        job_next_op = {j.job_id: 0 for j in self.instance.jobs}
        total_ops = self.instance.total_ops
        
        while len(assignment) < total_ops:
            ready = [(job.job_id, job_next_op[job.job_id]) 
                    for job in self.instance.jobs 
                    if job_next_op[job.job_id] < len(job.operations)]
            if not ready:
                break
            
            job_id, op_id = random.choice(ready)
            op = self.instance.jobs[job_id].operations[op_id]
            
            probs = []
            machines = list(op.alternatives.keys())
            for m in machines:
                tau = self.pheromone.get((job_id, op_id, m), 1.0)
                eta = 1.0 / op.alternatives[m].gmir()
                probs.append((tau ** self.alpha) * (eta ** self.beta))
            
            total = sum(probs)
            probs = [p / total for p in probs]
            selected = np.random.choice(machines, p=probs)
            
            assignment.append((job_id, op_id, selected))
            job_next_op[job_id] += 1
        
        return decode(self.instance, assignment)
    
    def solve(self):
        best_solution = None
        convergence = []
        
        for _ in range(self.max_iter):
            solutions = [self.construct_solution() for _ in range(self.num_ants)]
            iter_best = min(solutions, key=lambda s: s.objective)
            
            if best_solution is None or iter_best.objective < best_solution.objective:
                best_solution = iter_best
            
            # Pheromone update
            for key in self.pheromone:
                self.pheromone[key] *= (1 - self.rho)
            deposit = 1.0 / best_solution.objective if best_solution.objective > 0 else 1.0
            for job_id, op_id, mach_id in best_solution.schedule:
                self.pheromone[(job_id, op_id, mach_id)] += deposit
            
            convergence.append(best_solution.objective)
        
        return best_solution, convergence


# =============================================================================
# SENSITIVITY ANALYSIS 1: ALPHA/BETA WEIGHTS
# =============================================================================

def analyze_alpha_beta(instance, runs=5):
    """Analyze effect of alpha/beta weights"""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: Alpha/Beta Weights")
    print("="*70)
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for alpha in alpha_values:
        beta = 1.0 - alpha
        instance.alpha = alpha
        instance.beta = beta
        
        makespans = []
        energies = []
        objectives = []
        
        for _ in range(runs):
            solver = ACO(instance, max_iter=100)
            solution, _ = solver.solve()
            makespans.append(solution.makespan)
            energies.append(solution.energy)
            objectives.append(solution.objective)
        
        results.append({
            'alpha': alpha,
            'beta': beta,
            'avg_makespan': np.mean(makespans),
            'avg_energy': np.mean(energies),
            'avg_objective': np.mean(objectives),
            'std_objective': np.std(objectives)
        })
        
        print(f"α={alpha:.2f}, β={beta:.2f}: Makespan={np.mean(makespans):.2f}, "
              f"Energy={np.mean(energies):.2f}, Obj={np.mean(objectives):.2f}")
    
    return results


def plot_alpha_beta(results, output_path):
    """Plot alpha/beta sensitivity"""
    if not PLOT_AVAILABLE:
        return
    
    alphas = [r['alpha'] for r in results]
    makespans = [r['avg_makespan'] for r in results]
    energies = [r['avg_energy'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Alpha (Makespan Weight)', fontsize=12)
    ax1.set_ylabel('Makespan', color='blue', fontsize=12)
    ax1.plot(alphas, makespans, 'b-o', linewidth=2, markersize=8, label='Makespan')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy (kWh)', color='red', fontsize=12)
    ax2.plot(alphas, energies, 'r-s', linewidth=2, markersize=8, label='Energy')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Effect of Alpha/Beta Weights on Objectives', fontsize=14)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# SENSITIVITY ANALYSIS 2: FUZZINESS LEVELS
# =============================================================================

def analyze_fuzziness(base_seed=42, runs=5):
    """Analyze effect of fuzziness levels"""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: Fuzziness Levels")
    print("="*70)
    
    fuzziness_values = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    results = []
    
    for fuzz in fuzziness_values:
        instance = generate_instance(
            f"Fuzz_{fuzz}", 10, 5, 4, 
            flexibility=0.5, fuzziness=fuzz, seed=base_seed
        )
        
        objectives = []
        makespans = []
        
        for run in range(runs):
            solver = ACO(instance, max_iter=100)
            solution, _ = solver.solve()
            objectives.append(solution.objective)
            makespans.append(solution.makespan)
        
        results.append({
            'fuzziness': fuzz,
            'avg_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'avg_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans)
        })
        
        print(f"Fuzziness={fuzz:.2f}: Obj={np.mean(objectives):.2f}±{np.std(objectives):.2f}, "
              f"Makespan={np.mean(makespans):.2f}±{np.std(makespans):.2f}")
    
    return results


def plot_fuzziness(results, output_path):
    """Plot fuzziness sensitivity"""
    if not PLOT_AVAILABLE:
        return
    
    fuzz = [r['fuzziness'] for r in results]
    obj = [r['avg_objective'] for r in results]
    std = [r['std_objective'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(fuzz, obj, yerr=std, fmt='o-', linewidth=2, markersize=8, capsize=5)
    plt.xlabel('Fuzziness Level (δ)', fontsize=12)
    plt.ylabel('Average Objective', fontsize=12)
    plt.title('Effect of Fuzziness Level on Solution Quality', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# SENSITIVITY ANALYSIS 3: ACO PARAMETERS
# =============================================================================

def analyze_aco_parameters(instance, runs=5):
    """Analyze ACO parameter sensitivity"""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: ACO Parameters")
    print("="*70)
    
    results = []
    
    # Test different alpha (pheromone importance)
    print("\n--- Pheromone Importance (α) ---")
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
        objectives = []
        for _ in range(runs):
            solver = ACO(instance, max_iter=100, alpha=alpha, beta=2.0, rho=0.1)
            solution, _ = solver.solve()
            objectives.append(solution.objective)
        
        results.append({
            'parameter': 'alpha',
            'value': alpha,
            'avg_obj': np.mean(objectives),
            'std_obj': np.std(objectives)
        })
        print(f"  α={alpha}: Obj={np.mean(objectives):.2f}±{np.std(objectives):.2f}")
    
    # Test different beta (heuristic importance)
    print("\n--- Heuristic Importance (β) ---")
    for beta in [1.0, 2.0, 3.0, 4.0, 5.0]:
        objectives = []
        for _ in range(runs):
            solver = ACO(instance, max_iter=100, alpha=1.0, beta=beta, rho=0.1)
            solution, _ = solver.solve()
            objectives.append(solution.objective)
        
        results.append({
            'parameter': 'beta',
            'value': beta,
            'avg_obj': np.mean(objectives),
            'std_obj': np.std(objectives)
        })
        print(f"  β={beta}: Obj={np.mean(objectives):.2f}±{np.std(objectives):.2f}")
    
    # Test different rho (evaporation rate)
    print("\n--- Evaporation Rate (ρ) ---")
    for rho in [0.05, 0.1, 0.15, 0.2, 0.3]:
        objectives = []
        for _ in range(runs):
            solver = ACO(instance, max_iter=100, alpha=1.0, beta=2.0, rho=rho)
            solution, _ = solver.solve()
            objectives.append(solution.objective)
        
        results.append({
            'parameter': 'rho',
            'value': rho,
            'avg_obj': np.mean(objectives),
            'std_obj': np.std(objectives)
        })
        print(f"  ρ={rho}: Obj={np.mean(objectives):.2f}±{np.std(objectives):.2f}")
    
    # Test different number of ants
    print("\n--- Number of Ants ---")
    for num_ants in [10, 20, 30, 50, 100]:
        objectives = []
        times = []
        for _ in range(runs):
            start = time.time()
            solver = ACO(instance, max_iter=100, num_ants=num_ants)
            solution, _ = solver.solve()
            objectives.append(solution.objective)
            times.append(time.time() - start)
        
        results.append({
            'parameter': 'num_ants',
            'value': num_ants,
            'avg_obj': np.mean(objectives),
            'std_obj': np.std(objectives),
            'avg_time': np.mean(times)
        })
        print(f"  Ants={num_ants}: Obj={np.mean(objectives):.2f}±{np.std(objectives):.2f}, Time={np.mean(times):.2f}s")
    
    return results


def plot_aco_parameters(results, output_dir):
    """Plot ACO parameter sensitivity"""
    if not PLOT_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for param in ['alpha', 'beta', 'rho', 'num_ants']:
        param_results = [r for r in results if r['parameter'] == param]
        if not param_results:
            continue
        
        values = [r['value'] for r in param_results]
        objs = [r['avg_obj'] for r in param_results]
        stds = [r['std_obj'] for r in param_results]
        
        plt.figure(figsize=(8, 5))
        plt.errorbar(values, objs, yerr=stds, fmt='o-', linewidth=2, markersize=8, capsize=5)
        plt.xlabel(f'Parameter: {param}', fontsize=12)
        plt.ylabel('Average Objective', fontsize=12)
        plt.title(f'ACO Sensitivity: {param}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'aco_sensitivity_{param}.png'), dpi=150)
        plt.close()
    
    print(f"ACO parameter plots saved to {output_dir}/")


# =============================================================================
# SENSITIVITY ANALYSIS 4: PROBLEM SIZE SCALING
# =============================================================================

def analyze_scaling(runs=5):
    """Analyze algorithm scaling with problem size"""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: Problem Size Scaling")
    print("="*70)
    
    configurations = [
        (5, 3, 3),    # 15 ops
        (10, 5, 4),   # 40 ops
        (15, 6, 4),   # 60 ops
        (20, 8, 5),   # 100 ops
        (30, 10, 5),  # 150 ops
        (40, 10, 5),  # 200 ops
        (50, 10, 6),  # 300 ops
    ]
    
    results = []
    
    for n, m, o in configurations:
        instance = generate_instance(f"Scale_{n}x{m}x{o}", n, m, o, seed=42)
        total_ops = instance.total_ops
        
        objectives = []
        times = []
        
        for _ in range(runs):
            start = time.time()
            solver = ACO(instance, max_iter=100)
            solution, _ = solver.solve()
            objectives.append(solution.objective)
            times.append(time.time() - start)
        
        results.append({
            'jobs': n,
            'machines': m,
            'ops_per_job': o,
            'total_ops': total_ops,
            'avg_obj': np.mean(objectives),
            'std_obj': np.std(objectives),
            'avg_time': np.mean(times),
            'std_time': np.std(times)
        })
        
        print(f"{n}x{m}x{o} ({total_ops} ops): Obj={np.mean(objectives):.2f}, Time={np.mean(times):.2f}s")
    
    return results


def plot_scaling(results, output_path):
    """Plot scaling analysis"""
    if not PLOT_AVAILABLE:
        return
    
    ops = [r['total_ops'] for r in results]
    times = [r['avg_time'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(ops, times, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Total Operations', fontsize=12)
    ax1.set_ylabel('Computation Time (s)', fontsize=12)
    ax1.set_title('Algorithm Scaling with Problem Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Fit polynomial
    coeffs = np.polyfit(ops, times, 2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(ops), max(ops), 100)
    ax1.plot(x_fit, poly(x_fit), 'r--', label=f'Quadratic fit')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_sensitivity_results(all_results, output_dir):
    """Export all sensitivity analysis results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    with open(os.path.join(output_dir, 'sensitivity_analysis.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # LaTeX tables
    latex_path = os.path.join(output_dir, 'sensitivity_tables.tex')
    with open(latex_path, 'w') as f:
        # Alpha/Beta table
        if 'alpha_beta' in all_results:
            f.write("% Alpha/Beta Sensitivity\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Effect of weight parameters on objectives}\n")
            f.write("\\label{tab:alpha_beta}\n")
            f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccccc@{}}\n")
            f.write("\\toprule\n")
            f.write("$\\alpha$ & $\\beta$ & Makespan & Energy & Objective \\\\\n")
            f.write("\\midrule\n")
            for r in all_results['alpha_beta']:
                f.write(f"{r['alpha']:.2f} & {r['beta']:.2f} & {r['avg_makespan']:.2f} & "
                       f"{r['avg_energy']:.2f} & {r['avg_objective']:.2f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular*}\n")
            f.write("\\end{table}\n\n")
        
        # Fuzziness table
        if 'fuzziness' in all_results:
            f.write("% Fuzziness Sensitivity\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Effect of fuzziness level on solution quality}\n")
            f.write("\\label{tab:fuzziness}\n")
            f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccccc@{}}\n")
            f.write("\\toprule\n")
            f.write("$\\delta$ & Avg Obj & Std & Avg Makespan & Std \\\\\n")
            f.write("\\midrule\n")
            for r in all_results['fuzziness']:
                f.write(f"{r['fuzziness']:.2f} & {r['avg_objective']:.2f} & {r['std_objective']:.2f} & "
                       f"{r['avg_makespan']:.2f} & {r['std_makespan']:.2f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular*}\n")
            f.write("\\end{table}\n\n")
        
        # Scaling table
        if 'scaling' in all_results:
            f.write("% Scaling Analysis\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Algorithm scaling with problem size}\n")
            f.write("\\label{tab:scaling}\n")
            f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccccc@{}}\n")
            f.write("\\toprule\n")
            f.write("Size & Ops & Avg Obj & Std & Time (s) \\\\\n")
            f.write("\\midrule\n")
            for r in all_results['scaling']:
                size = f"{r['jobs']}×{r['machines']}×{r['ops_per_job']}"
                f.write(f"{size} & {r['total_ops']} & {r['avg_obj']:.2f} & "
                       f"{r['std_obj']:.2f} & {r['avg_time']:.2f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular*}\n")
            f.write("\\end{table}\n")
    
    print(f"Results exported to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  SENSITIVITY ANALYSIS FOR GF-FJSP-PM")
    print("="*70)
    
    os.makedirs("sensitivity_results", exist_ok=True)
    os.makedirs("sensitivity_figures", exist_ok=True)
    
    all_results = {}
    
    # Base instance for parameter analysis
    base_instance = generate_instance("Base_10x5x4", 10, 5, 4, seed=42)
    
    # 1. Alpha/Beta analysis
    print("\n[1/4] Analyzing Alpha/Beta weights...")
    alpha_beta_results = analyze_alpha_beta(base_instance, runs=5)
    all_results['alpha_beta'] = alpha_beta_results
    plot_alpha_beta(alpha_beta_results, "sensitivity_figures/alpha_beta.png")
    
    # 2. Fuzziness analysis
    print("\n[2/4] Analyzing Fuzziness levels...")
    fuzziness_results = analyze_fuzziness(runs=5)
    all_results['fuzziness'] = fuzziness_results
    plot_fuzziness(fuzziness_results, "sensitivity_figures/fuzziness.png")
    
    # 3. ACO parameter analysis
    print("\n[3/4] Analyzing ACO parameters...")
    aco_results = analyze_aco_parameters(base_instance, runs=5)
    all_results['aco_parameters'] = aco_results
    plot_aco_parameters(aco_results, "sensitivity_figures")
    
    # 4. Scaling analysis
    print("\n[4/4] Analyzing problem size scaling...")
    scaling_results = analyze_scaling(runs=5)
    all_results['scaling'] = scaling_results
    plot_scaling(scaling_results, "sensitivity_figures/scaling.png")
    
    # Export
    export_sensitivity_results(all_results, "sensitivity_results")
    
    # Summary
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    
    # Best alpha
    best_ab = min(all_results['alpha_beta'], key=lambda x: x['avg_objective'])
    print(f"  - Best α/β: α={best_ab['alpha']:.2f}, β={best_ab['beta']:.2f}")
    
    # ACO parameters
    for param in ['alpha', 'beta', 'rho', 'num_ants']:
        param_results = [r for r in all_results['aco_parameters'] if r['parameter'] == param]
        if param_results:
            best = min(param_results, key=lambda x: x['avg_obj'])
            print(f"  - Best ACO {param}: {best['value']}")
    
    print("\nFiles generated:")
    print("  - sensitivity_results/sensitivity_analysis.json")
    print("  - sensitivity_results/sensitivity_tables.tex")
    print("  - sensitivity_figures/*.png")
    
    return all_results


if __name__ == "__main__":
    results = main()
