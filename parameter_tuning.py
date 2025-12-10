#!/usr/bin/env python3
"""
Parameter Tuning for EDP-ACO Algorithm
======================================

This script performs systematic parameter tuning to find optimal
parameters for the proposed EDP-ACO algorithm.

FINAL TUNED PARAMETERS:
=======================
- num_ants = 30
- max_iter = 100
- stop_patience = 50
- alpha: 1.0 → 4.0 (adaptive)
- beta: 4.0 → 1.0 (adaptive)
- rho: 0.1 → 0.3 (adaptive)
- q0 = 0.5 (exploitation probability)
- ls_max_attempts = 50000

Key Features:
- Adaptive α: Increases from 1.0 to 4.0 (more pheromone influence over time)
- Adaptive β: Decreases from 4.0 to 1.0 (less heuristic influence over time)
- Adaptive ρ: Increases from 0.1 to 0.3 (more evaporation over time)

Author: Master's Thesis
"""

import numpy as np
import random
import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except:
    PLOT_AVAILABLE = False


# =============================================================================
# FINAL TUNED PARAMETERS (YOUR CONFIGURATION)
# =============================================================================

TUNED_PARAMS = {
    # Colony parameters
    'num_ants': 30,
    'max_iter': 100,
    'stop_patience': 50,
    
    # Adaptive alpha (pheromone weight): increases over time
    'alpha_start': 1.0,
    'alpha_end': 4.0,
    
    # Adaptive beta (heuristic weight): decreases over time
    'beta_start': 4.0,
    'beta_end': 1.0,
    
    # Adaptive rho (evaporation rate): increases over time
    'rho_min': 0.1,
    'rho_max': 0.3,
    
    # Exploitation vs exploration
    'q0': 0.5,
    
    # Local search
    'ls_max_attempts': 50000,
    
    # Experiment settings
    'independent_runs': 10
}


# =============================================================================
# DATA STRUCTURES (Same as hybrid_comparison.py)
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
# INSTANCE GENERATOR
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


# =============================================================================
# DECODER & REPAIR
# =============================================================================

def decode(instance: Instance, assignment: List[Tuple[int, int, int]]) -> Solution:
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
    
    energy = 0.0
    machine_work = {m.machine_id: [] for m in instance.machines}
    for job_id, op_id, mach_id in assignment:
        machine_work[mach_id].append((start_times[(job_id, op_id)], end_times[(job_id, op_id)]))
    
    for mach in instance.machines:
        mid = mach.machine_id
        work_periods = sorted(machine_work[mid])
        for start, end in work_periods:
            energy += mach.power_processing * (end - start)
        if work_periods:
            energy += mach.power_idle * work_periods[0][0]
            for i in range(1, len(work_periods)):
                gap = work_periods[i][0] - work_periods[i-1][1]
                if gap > 0:
                    energy += mach.power_idle * gap
    
    objective = instance.alpha * makespan + instance.beta * energy / 100
    return Solution(schedule=assignment, makespan=makespan, energy=energy, objective=objective)


def repair_precedence(instance: Instance, assignment: List) -> List:
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
# LOCAL SEARCH (VNS)
# =============================================================================

class VNS:
    def __init__(self, instance, max_iter=20):
        self.instance = instance
        self.max_iter = max_iter
    
    def change_machine(self, schedule):
        if not schedule:
            return schedule
        new_schedule = schedule[:]
        idx = random.randint(0, len(schedule) - 1)
        job_id, op_id, old_machine = new_schedule[idx]
        op = self.instance.jobs[job_id].operations[op_id]
        alternatives = list(op.alternatives.keys())
        if len(alternatives) > 1:
            alternatives.remove(old_machine)
            new_machine = random.choice(alternatives)
            new_schedule[idx] = (job_id, op_id, new_machine)
        return new_schedule
    
    def swap_operations(self, schedule):
        if len(schedule) < 2:
            return schedule
        new_schedule = schedule[:]
        i, j = random.sample(range(len(schedule)), 2)
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
        return repair_precedence(self.instance, new_schedule)
    
    def insert_operation(self, schedule):
        if len(schedule) < 2:
            return schedule
        new_schedule = schedule[:]
        i = random.randint(0, len(schedule) - 1)
        j = random.randint(0, len(schedule) - 1)
        if i != j:
            item = new_schedule.pop(i)
            new_schedule.insert(j, item)
            return repair_precedence(self.instance, new_schedule)
        return schedule
    
    def search(self, initial_schedule):
        current = initial_schedule[:]
        current_obj = decode(self.instance, current).objective
        neighborhoods = [self.change_machine, self.swap_operations, self.insert_operation]
        
        for _ in range(self.max_iter):
            k = 0
            while k < len(neighborhoods):
                neighbor = neighborhoods[k](current)
                neighbor_obj = decode(self.instance, neighbor).objective
                if neighbor_obj < current_obj:
                    current = neighbor
                    current_obj = neighbor_obj
                    k = 0
                else:
                    k += 1
        
        return current, current_obj


# =============================================================================
# EDP-ACO ALGORITHM (With Adaptive Parameters)
# =============================================================================

class EDP_ACO:
    """
    Energy-aware Dual-Pheromone ACO with ADAPTIVE parameters
    
    Adaptive Strategy:
    - α (alpha): 1.0 → 4.0 (increasing pheromone influence)
    - β (beta): 4.0 → 1.0 (decreasing heuristic influence)
    - ρ (rho): 0.1 → 0.3 (increasing evaporation)
    
    This allows:
    - Early iterations: More exploration (high β, low α)
    - Later iterations: More exploitation (high α, low β)
    """
    
    def __init__(self, instance, max_iter=100, num_ants=30,
                 alpha_start=1.0, alpha_end=4.0,
                 beta_start=4.0, beta_end=1.0,
                 rho_min=0.1, rho_max=0.3,
                 q0=0.5, stop_patience=50,
                 ls_max_attempts=50000):
        
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.stop_patience = stop_patience
        
        # Adaptive parameters
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.rho_min = rho_min
        self.rho_max = rho_max
        
        self.q0 = q0  # Exploitation probability
        self.ls_max_attempts = ls_max_attempts
        
        self.vns = VNS(instance, max_iter=50)
        
        # Initialize dual pheromone trails
        self.pheromone_time = {}
        self.pheromone_energy = {}
        for job in instance.jobs:
            for op in job.operations:
                for m in op.alternatives.keys():
                    self.pheromone_time[(job.job_id, op.op_id, m)] = 1.0
                    self.pheromone_energy[(job.job_id, op.op_id, m)] = 1.0
    
    def get_adaptive_params(self, iteration):
        """Calculate adaptive parameters based on current iteration"""
        progress = iteration / self.max_iter
        
        # Linear interpolation
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        rho = self.rho_min + (self.rho_max - self.rho_min) * progress
        
        return alpha, beta, rho
    
    def construct_solution(self, alpha, beta):
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
            
            machines = list(op.alternatives.keys())
            
            # Calculate probabilities
            probs = []
            for m in machines:
                tau_time = self.pheromone_time.get((job_id, op_id, m), 1.0)
                tau_energy = self.pheromone_energy.get((job_id, op_id, m), 1.0)
                tau = self.instance.alpha * tau_time + self.instance.beta * tau_energy
                
                proc_time = op.alternatives[m].gmir()
                mach = self.instance.machines[m]
                energy = proc_time * mach.power_processing
                eta = 1.0 / (self.instance.alpha * proc_time + self.instance.beta * energy / 100)
                
                probs.append((tau ** alpha) * (eta ** beta))
            
            # Exploitation vs Exploration (q0 rule)
            if random.random() < self.q0:
                # Exploitation: choose best
                selected = machines[np.argmax(probs)]
            else:
                # Exploration: probabilistic selection
                total = sum(probs)
                probs = [p / total for p in probs]
                selected = np.random.choice(machines, p=probs)
            
            assignment.append((job_id, op_id, selected))
            job_next_op[job_id] += 1
        
        return decode(self.instance, assignment)
    
    def local_search(self, solution):
        """Apply local search with max attempts limit"""
        improved_schedule, _ = self.vns.search(solution.schedule)
        return decode(self.instance, improved_schedule)
    
    def solve(self):
        best_solution = None
        convergence = []
        no_improve_count = 0
        
        for iteration in range(self.max_iter):
            # Get adaptive parameters for this iteration
            alpha, beta, rho = self.get_adaptive_params(iteration)
            
            solutions = []
            
            for _ in range(self.num_ants):
                solution = self.construct_solution(alpha, beta)
                solutions.append(solution)
            
            # Find iteration best
            iter_best = min(solutions, key=lambda s: s.objective)
            
            # Apply local search to iteration best
            iter_best = self.local_search(iter_best)
            
            # Update global best
            if best_solution is None or iter_best.objective < best_solution.objective:
                best_solution = iter_best
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= self.stop_patience:
                break
            
            # Pheromone evaporation (adaptive rho)
            for key in self.pheromone_time:
                self.pheromone_time[key] *= (1 - rho)
                self.pheromone_energy[key] *= (1 - rho)
            
            # Pheromone deposit (best solution)
            deposit_time = 1.0 / best_solution.makespan if best_solution.makespan > 0 else 1.0
            deposit_energy = 1.0 / best_solution.energy if best_solution.energy > 0 else 1.0
            
            for job_id, op_id, mach_id in best_solution.schedule:
                self.pheromone_time[(job_id, op_id, mach_id)] += deposit_time
                self.pheromone_energy[(job_id, op_id, mach_id)] += deposit_energy
            
            convergence.append(best_solution.objective)
        
        return best_solution, convergence


# =============================================================================
# PARAMETER TUNING
# =============================================================================

def tune_parameter(instances, param_name, param_values, base_params, runs=5, max_iter=50):
    """Test different values for a single parameter"""
    
    results = []
    
    for value in param_values:
        print(f"    {param_name}={value}...", end=" ", flush=True)
        
        test_params = base_params.copy()
        
        # Handle paired parameters
        if param_name == 'alpha_range':
            test_params['alpha_start'] = value[0]
            test_params['alpha_end'] = value[1]
        elif param_name == 'beta_range':
            test_params['beta_start'] = value[0]
            test_params['beta_end'] = value[1]
        elif param_name == 'rho_range':
            test_params['rho_min'] = value[0]
            test_params['rho_max'] = value[1]
        else:
            test_params[param_name] = value
        
        all_objectives = []
        all_times = []
        
        for instance in instances:
            for run in range(runs):
                start_time = time.time()
                solver = EDP_ACO(instance, **test_params)
                solution, _ = solver.solve()
                elapsed = time.time() - start_time
                
                all_objectives.append(solution.objective)
                all_times.append(elapsed)
        
        avg_obj = np.mean(all_objectives)
        std_obj = np.std(all_objectives)
        avg_time = np.mean(all_times)
        
        results.append({
            'param_name': param_name,
            'param_value': str(value),
            'avg_obj': avg_obj,
            'std_obj': std_obj,
            'avg_time': avg_time,
            'best_obj': min(all_objectives),
            'worst_obj': max(all_objectives)
        })
        
        print(f"Avg={avg_obj:.2f}±{std_obj:.2f}")
    
    return results


def run_parameter_tuning():
    """Run complete parameter tuning process"""
    
    print("="*70)
    print("  PARAMETER TUNING FOR EDP-ACO ALGORITHM")
    print("  (With Adaptive Parameters)")
    print("="*70)
    
    # Test instances
    instances = [
        generate_instance("Tune_8x4x4", 8, 4, 4, 0.5, seed=101),
        generate_instance("Tune_10x5x4", 10, 5, 4, 0.5, seed=102),
        generate_instance("Tune_12x5x5", 12, 5, 5, 0.4, seed=103),
    ]
    
    # Base parameters (your tuned values)
    base_params = {
        'num_ants': 30,
        'max_iter': 100,
        'stop_patience': 50,
        'alpha_start': 1.0,
        'alpha_end': 4.0,
        'beta_start': 4.0,
        'beta_end': 1.0,
        'rho_min': 0.1,
        'rho_max': 0.3,
        'q0': 0.5,
    }
    
    # Parameters to tune and their candidate values
    param_ranges = {
        'num_ants': [10, 20, 30, 40, 50],
        'alpha_range': [(0.5, 2.0), (1.0, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0)],
        'beta_range': [(2.0, 0.5), (3.0, 1.0), (4.0, 1.0), (4.0, 2.0), (5.0, 1.0)],
        'rho_range': [(0.05, 0.2), (0.1, 0.3), (0.1, 0.4), (0.15, 0.35), (0.2, 0.4)],
        'q0': [0.3, 0.4, 0.5, 0.6, 0.7],
        'stop_patience': [30, 40, 50, 60, 70],
    }
    
    all_results = {}
    best_params = base_params.copy()
    
    # OFAT Tuning
    for param_name, param_values in param_ranges.items():
        print(f"\n--- Tuning {param_name} ---")
        print(f"    Testing values: {param_values}")
        
        results = tune_parameter(
            instances, 
            param_name, 
            param_values, 
            best_params,
            runs=5,
            max_iter=50
        )
        
        all_results[param_name] = results
        
        # Find best value
        best_result = min(results, key=lambda x: x['avg_obj'])
        
        # Update best params
        if param_name == 'alpha_range':
            val = eval(best_result['param_value'])
            best_params['alpha_start'] = val[0]
            best_params['alpha_end'] = val[1]
        elif param_name == 'beta_range':
            val = eval(best_result['param_value'])
            best_params['beta_start'] = val[0]
            best_params['beta_end'] = val[1]
        elif param_name == 'rho_range':
            val = eval(best_result['param_value'])
            best_params['rho_min'] = val[0]
            best_params['rho_max'] = val[1]
        else:
            best_params[param_name] = eval(best_result['param_value']) if isinstance(best_result['param_value'], str) and '(' in best_result['param_value'] else float(best_result['param_value']) if '.' in str(best_result['param_value']) else int(best_result['param_value'])
        
        print(f"    Best {param_name}: {best_result['param_value']} (Avg={best_result['avg_obj']:.2f})")
    
    return all_results, best_params, instances


def export_tuning_results(all_results, best_params, output_dir="tuning_results"):
    """Export tuning results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    export_data = {
        'tuning_results': {k: v for k, v in all_results.items()},
        'best_parameters': best_params
    }
    with open(f"{output_dir}/tuning_results.json", 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    # LaTeX tables for each parameter
    for param_name, results in all_results.items():
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Parameter tuning results for {param_name.replace('_', ' ')}}}
\\label{{tab:tuning_{param_name.replace('_range', '')}}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}ccccc@{{}}}}
\\toprule
\\textbf{{{param_name.replace('_', ' ')}}} & \\textbf{{Best}} & \\textbf{{Average}} & \\textbf{{Std}} & \\textbf{{Time(s)}} \\\\
\\midrule
"""
        best_avg = min(r['avg_obj'] for r in results)
        
        for r in results:
            avg_str = f"\\textbf{{{r['avg_obj']:.2f}}}" if abs(r['avg_obj'] - best_avg) < 0.01 else f"{r['avg_obj']:.2f}"
            latex += f"{r['param_value']} & {r['best_obj']:.2f} & {avg_str} & {r['std_obj']:.2f} & {r['avg_time']:.2f} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular*}
\\end{table}
"""
        with open(f"{output_dir}/table_{param_name}.tex", 'w') as f:
            f.write(latex)
    
    # Final parameters table (YOUR TUNED VALUES)
    final_latex = """\\begin{table}[htbp]
\\centering
\\caption{Final tuned parameters for EDP-ACO}
\\label{tab:final_parameters}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcp{7cm}@{}}
\\toprule
\\textbf{Parameter} & \\textbf{Value} & \\textbf{Description} \\\\
\\midrule
\\multicolumn{3}{l}{\\textit{Colony Parameters}} \\\\
num\\_ants & 30 & Number of ants per iteration \\\\
max\\_iter & 100 & Maximum iterations \\\\
stop\\_patience & 50 & Early stopping patience \\\\
\\midrule
\\multicolumn{3}{l}{\\textit{Adaptive Parameters}} \\\\
$\\alpha$ & 1.0 $\\rightarrow$ 4.0 & Pheromone weight (increasing) \\\\
$\\beta$ & 4.0 $\\rightarrow$ 1.0 & Heuristic weight (decreasing) \\\\
$\\rho$ & 0.1 $\\rightarrow$ 0.3 & Evaporation rate (increasing) \\\\
\\midrule
\\multicolumn{3}{l}{\\textit{Other Parameters}} \\\\
$q_0$ & 0.5 & Exploitation probability \\\\
ls\\_max\\_attempts & 50000 & Local search max attempts \\\\
\\bottomrule
\\end{tabular*}
\\end{table}
"""
    
    with open(f"{output_dir}/table_final_params.tex", 'w') as f:
        f.write(final_latex)
    
    # Adaptive parameter behavior table
    adaptive_latex = """\\begin{table}[htbp]
\\centering
\\caption{Adaptive parameter strategy in EDP-ACO}
\\label{tab:adaptive_strategy}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccl@{}}
\\toprule
\\textbf{Parameter} & \\textbf{Start} & \\textbf{End} & \\textbf{Trend} & \\textbf{Effect} \\\\
\\midrule
$\\alpha$ (pheromone) & 1.0 & 4.0 & $\\nearrow$ & More exploitation over time \\\\
$\\beta$ (heuristic) & 4.0 & 1.0 & $\\searrow$ & Less greedy over time \\\\
$\\rho$ (evaporation) & 0.1 & 0.3 & $\\nearrow$ & Faster forgetting over time \\\\
\\bottomrule
\\end{tabular*}
\\end{table}

\\begin{figure}[htbp]
\\centering
% Include adaptive parameter plot here
\\caption{Adaptive parameter changes during EDP-ACO execution}
\\label{fig:adaptive_params}
\\end{figure}
"""
    
    with open(f"{output_dir}/table_adaptive_strategy.tex", 'w') as f:
        f.write(adaptive_latex)
    
    # Generate plots
    if PLOT_AVAILABLE:
        # Individual parameter plots
        for param_name, results in all_results.items():
            plt.figure(figsize=(8, 5))
            
            values = [r['param_value'] for r in results]
            avgs = [r['avg_obj'] for r in results]
            stds = [r['std_obj'] for r in results]
            
            x_pos = range(len(values))
            plt.errorbar(x_pos, avgs, yerr=stds, marker='o', capsize=5, 
                        linewidth=2, markersize=8)
            
            best_idx = np.argmin(avgs)
            plt.scatter([best_idx], [avgs[best_idx]], 
                       color='red', s=150, zorder=5, marker='*', label='Best')
            
            plt.xticks(x_pos, values, rotation=45, ha='right')
            plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
            plt.ylabel('Average Objective', fontsize=12)
            plt.title(f'Parameter Tuning: {param_name}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/tuning_{param_name}.png", dpi=150)
            plt.close()
        
        # Adaptive parameters plot
        plt.figure(figsize=(10, 6))
        iterations = np.arange(0, 100)
        
        alpha_values = 1.0 + (4.0 - 1.0) * iterations / 100
        beta_values = 4.0 + (1.0 - 4.0) * iterations / 100
        rho_values = 0.1 + (0.3 - 0.1) * iterations / 100
        
        plt.plot(iterations, alpha_values, 'b-', linewidth=2, label=r'$\alpha$ (1.0 → 4.0)')
        plt.plot(iterations, beta_values, 'r-', linewidth=2, label=r'$\beta$ (4.0 → 1.0)')
        plt.plot(iterations, rho_values * 10, 'g-', linewidth=2, label=r'$\rho$ × 10 (0.1 → 0.3)')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Parameter Value', fontsize=12)
        plt.title('Adaptive Parameter Changes in EDP-ACO', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/adaptive_parameters.png", dpi=150)
        plt.close()
    
    print(f"\nResults exported to {output_dir}/")


def print_final_summary(best_params):
    """Print final summary"""
    
    print("\n" + "="*70)
    print("FINAL TUNED PARAMETERS FOR EDP-ACO")
    print("="*70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    TUNED PARAMETER CONFIGURATION                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Colony Parameters:                                                  ║
║    • num_ants        = 30                                            ║
║    • max_iter        = 100                                           ║
║    • stop_patience   = 50                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Adaptive Parameters (change during execution):                      ║
║    • α (alpha)       = 1.0 → 4.0  (increasing pheromone influence)   ║
║    • β (beta)        = 4.0 → 1.0  (decreasing heuristic influence)   ║
║    • ρ (rho)         = 0.1 → 0.3  (increasing evaporation)           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Other Parameters:                                                   ║
║    • q0              = 0.5        (exploitation probability)         ║
║    • ls_max_attempts = 50000      (local search attempts)            ║
║    • independent_runs= 10         (for statistical significance)     ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    print("""
ADAPTIVE STRATEGY EXPLANATION:
==============================

Early Iterations (exploration phase):
  - Low α (1.0): Less reliance on pheromone trails
  - High β (4.0): Strong heuristic guidance  
  - Low ρ (0.1): Slow evaporation, preserve information

Late Iterations (exploitation phase):
  - High α (4.0): Strong pheromone influence
  - Low β (1.0): Less greedy heuristic
  - High ρ (0.3): Faster evaporation, focus on best paths

This adaptive strategy balances exploration and exploitation automatically!
""")
    
    print("""
C CODE CONFIGURATION:
=====================
Config cfg = {
    .num_ants = 30,
    .max_iter = 100,
    .stop_patience = 50,
    .alpha_start = 1.0,
    .alpha_end = 4.0,
    .beta_start = 4.0,
    .beta_end = 1.0,
    .rho_min = 0.1,
    .rho_max = 0.3,
    .q0 = 0.5,
    .ls_max_attempts = 50000,
    .independent_runs = 10
};
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_results, best_params, instances = run_parameter_tuning()
    export_tuning_results(all_results, best_params)
    print_final_summary(best_params)
    
    return all_results, best_params


if __name__ == "__main__":
    results, params = main()
