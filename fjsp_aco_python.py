"""
==============================================================================
FJSP-ACO: Ant Colony Optimization for Flexible Job Shop Scheduling
Pure Python Implementation - Run directly in PyCharm
==============================================================================

This script implements:
- ACO algorithm for FJSP
- Support for Brandimarte (Mk01-Mk10) and Lawrence (LA01-LA40) benchmarks
- Convergence tracking with time
- CSV and text report output
- Convergence plotting

Usage:
    python fjsp_aco_python.py                    # Run all benchmarks
    python fjsp_aco_python.py --mk               # Run Brandimarte only
    python fjsp_aco_python.py --la               # Run Lawrence only
    python fjsp_aco_python.py --mk --start 1 --end 5   # Run Mk01-Mk05

Author: [Your Name]
==============================================================================
"""

import numpy as np
import time
import os
import csv
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    num_ants: int = 30
    max_iter: int = 100
    stop_patience: int = 50
    alpha_start: float = 1.0
    alpha_end: float = 4.0
    beta_start: float = 4.0
    beta_end: float = 1.0
    rho_min: float = 0.1
    rho_max: float = 0.3
    q0: float = 0.5
    ls_max_attempts: int = 50000
    independent_runs: int = 10


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Operation:
    job_id: int
    op_idx: int
    alternatives: Dict[int, int]  # machine_id -> processing_time


@dataclass
class Job:
    job_id: int
    operations: List[Operation]


@dataclass
class FJSPInstance:
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job]
    bks: int = 0  # Best Known Solution
    
    @property
    def total_ops(self) -> int:
        return sum(len(j.operations) for j in self.jobs)


@dataclass
class ScheduleOp:
    job_id: int
    op_idx: int
    machine_id: int
    proc_time: int
    start_time: float = 0
    finish_time: float = 0


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_brandimarte(filepath: str) -> Optional[FJSPInstance]:
    """Load Brandimarte FJSP format"""
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().split()
            num_jobs = int(first_line[0])
            num_machines = int(first_line[1])
            
            jobs = []
            for i in range(num_jobs):
                line = f.readline().split()
                num_ops = int(line[0])
                idx = 1
                
                operations = []
                for j in range(num_ops):
                    num_alts = int(line[idx])
                    idx += 1
                    
                    alternatives = {}
                    for _ in range(num_alts):
                        m_id = int(line[idx]) - 1  # 0-based
                        p_time = int(line[idx + 1])
                        idx += 2
                        alternatives[m_id] = p_time
                    
                    operations.append(Operation(job_id=i, op_idx=j, alternatives=alternatives))
                
                jobs.append(Job(job_id=i, operations=operations))
        
        name = os.path.basename(filepath).replace('.txt', '')
        return FJSPInstance(name=name, num_jobs=num_jobs, num_machines=num_machines, jobs=jobs)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_lawrence(filepath: str) -> Optional[FJSPInstance]:
    """Load Lawrence JSP format (standard JSP)"""
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().split()
            num_jobs = int(first_line[0])
            num_machines = int(first_line[1])
            
            jobs = []
            for i in range(num_jobs):
                line = f.readline().split()
                operations = []
                
                for j in range(num_machines):
                    m_id = int(line[j * 2])
                    p_time = int(line[j * 2 + 1])
                    # JSP: each operation has exactly 1 machine
                    alternatives = {m_id: p_time}
                    operations.append(Operation(job_id=i, op_idx=j, alternatives=alternatives))
                
                jobs.append(Job(job_id=i, operations=operations))
        
        name = os.path.basename(filepath).replace('.txt', '').upper()
        return FJSPInstance(name=name, num_jobs=num_jobs, num_machines=num_machines, jobs=jobs)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# ==============================================================================
# MAKESPAN CALCULATION
# ==============================================================================

def calculate_makespan(schedule: List[ScheduleOp], num_jobs: int, num_machines: int) -> float:
    """Calculate makespan and update start/finish times"""
    machine_times = [0.0] * num_machines
    job_times = [0.0] * num_jobs
    makespan = 0.0
    
    for op in schedule:
        start = max(job_times[op.job_id], machine_times[op.machine_id])
        finish = start + op.proc_time
        
        op.start_time = start
        op.finish_time = finish
        
        machine_times[op.machine_id] = finish
        job_times[op.job_id] = finish
        
        if finish > makespan:
            makespan = finish
    
    return makespan


# ==============================================================================
# ANT CLASS
# ==============================================================================

class Ant:
    def __init__(self, instance: FJSPInstance, pheromones: Dict, alpha: float, beta: float, q0: float):
        self.instance = instance
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.schedule: List[ScheduleOp] = []
        self.makespan = float('inf')
    
    def build_solution(self):
        """Build a complete solution using ACO rules"""
        self.schedule = []
        machine_times = [0.0] * self.instance.num_machines
        job_times = [0.0] * self.instance.num_jobs
        op_counters = [0] * self.instance.num_jobs
        
        total_ops = self.instance.total_ops
        
        while len(self.schedule) < total_ops:
            # Gather all possible moves
            moves = []
            probs = []
            
            for job in self.instance.jobs:
                j = job.job_id
                if op_counters[j] >= len(job.operations):
                    continue
                
                op = job.operations[op_counters[j]]
                
                for m_id, p_time in op.alternatives.items():
                    start = max(job_times[j], machine_times[m_id])
                    finish = start + p_time
                    
                    # Get pheromone
                    tau = self.pheromones.get((j, op.op_idx, m_id), 1.0)
                    eta = 1.0 / p_time if p_time > 0 else 1.0
                    
                    prob = (tau ** self.alpha) * (eta ** self.beta)
                    
                    moves.append(ScheduleOp(
                        job_id=j, op_idx=op.op_idx, machine_id=m_id,
                        proc_time=p_time, start_time=start, finish_time=finish
                    ))
                    probs.append(prob)
            
            if not moves:
                break
            
            # Select move
            probs = np.array(probs)
            
            if np.random.random() < self.q0:
                # Exploitation
                selected_idx = np.argmax(probs)
            else:
                # Exploration (roulette wheel)
                probs = probs / probs.sum()
                selected_idx = np.random.choice(len(moves), p=probs)
            
            selected = moves[selected_idx]
            
            # Apply move
            machine_times[selected.machine_id] = selected.finish_time
            job_times[selected.job_id] = selected.finish_time
            op_counters[selected.job_id] += 1
            
            self.schedule.append(selected)
        
        # Calculate final makespan
        self.makespan = max(op.finish_time for op in self.schedule) if self.schedule else float('inf')


# ==============================================================================
# LOCAL SEARCH
# ==============================================================================

def local_search(schedule: List[ScheduleOp], instance: FJSPInstance, max_attempts: int) -> Tuple[List[ScheduleOp], float]:
    """Stochastic hill climbing local search"""
    best_schedule = [ScheduleOp(op.job_id, op.op_idx, op.machine_id, op.proc_time) for op in schedule]
    best_mk = calculate_makespan(best_schedule, instance.num_jobs, instance.num_machines)
    
    current_schedule = [ScheduleOp(op.job_id, op.op_idx, op.machine_id, op.proc_time) for op in best_schedule]
    
    for _ in range(max_attempts):
        # Choose move type
        if np.random.random() < 0.5:
            # Swap adjacent operations (different jobs)
            if len(current_schedule) < 2:
                continue
            idx = np.random.randint(0, len(current_schedule) - 1)
            if current_schedule[idx].job_id != current_schedule[idx + 1].job_id:
                current_schedule[idx], current_schedule[idx + 1] = current_schedule[idx + 1], current_schedule[idx]
        else:
            # Change machine assignment
            idx = np.random.randint(0, len(current_schedule))
            op = current_schedule[idx]
            job = instance.jobs[op.job_id]
            operation = job.operations[op.op_idx]
            
            if len(operation.alternatives) > 1:
                machines = list(operation.alternatives.keys())
                new_m = np.random.choice(machines)
                current_schedule[idx].machine_id = new_m
                current_schedule[idx].proc_time = operation.alternatives[new_m]
        
        # Evaluate
        new_mk = calculate_makespan(current_schedule, instance.num_jobs, instance.num_machines)
        
        if new_mk <= best_mk:
            best_mk = new_mk
            best_schedule = [ScheduleOp(op.job_id, op.op_idx, op.machine_id, op.proc_time) for op in current_schedule]
        else:
            # Revert
            current_schedule = [ScheduleOp(op.job_id, op.op_idx, op.machine_id, op.proc_time) for op in best_schedule]
    
    return best_schedule, best_mk


# ==============================================================================
# ACO OPTIMIZER
# ==============================================================================

class ACOOptimizer:
    def __init__(self, instance: FJSPInstance, config: Config):
        self.instance = instance
        self.config = config
        self.pheromones: Dict[Tuple[int, int, int], float] = {}
        self.min_tau = 0.1
        self.max_tau = 10.0
        
        # Initialize pheromones
        for job in instance.jobs:
            for op in job.operations:
                for m_id in op.alternatives.keys():
                    self.pheromones[(job.job_id, op.op_idx, m_id)] = 1.0
    
    def optimize(self) -> Tuple[float, List[float], List[float]]:
        """Run ACO optimization, return (best_makespan, convergence_curve, time_curve)"""
        
        cfg = self.config
        best_global_mk = float('inf')
        best_global_schedule = None
        stagnation = 0
        
        convergence = []
        time_curve = []
        start_time = time.time()
        
        for iteration in range(cfg.max_iter):
            # Dynamic parameters
            ratio = iteration / cfg.max_iter
            alpha = cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * ratio
            beta = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * ratio
            rho = cfg.rho_max if stagnation > 10 else cfg.rho_min
            
            iter_best_mk = float('inf')
            iter_best_ant = None
            iter_worst_mk = 0
            iter_worst_ant = None
            
            # Build solutions
            ants = []
            for _ in range(cfg.num_ants):
                ant = Ant(self.instance, self.pheromones, alpha, beta, cfg.q0)
                ant.build_solution()
                ants.append(ant)
                
                if ant.makespan < iter_best_mk:
                    iter_best_mk = ant.makespan
                    iter_best_ant = ant
                if ant.makespan > iter_worst_mk:
                    iter_worst_mk = ant.makespan
                    iter_worst_ant = ant
            
            # Local search on best ant
            if iter_best_ant:
                improved_schedule, improved_mk = local_search(
                    iter_best_ant.schedule, self.instance, cfg.ls_max_attempts
                )
                if improved_mk < iter_best_mk:
                    iter_best_mk = improved_mk
                    iter_best_ant.schedule = improved_schedule
                    iter_best_ant.makespan = improved_mk
            
            # Global update
            if iter_best_mk < best_global_mk:
                best_global_mk = iter_best_mk
                best_global_schedule = iter_best_ant.schedule if iter_best_ant else None
                stagnation = 0
            else:
                stagnation += 1
            
            # Record convergence
            convergence.append(best_global_mk)
            time_curve.append(time.time() - start_time)
            
            # Pheromone update
            deposit = 100.0 / best_global_mk if best_global_mk > 0 else 0
            punishment = 0.5 * (100.0 / iter_worst_mk) if iter_worst_mk > 0 else 0
            
            # Evaporation
            for key in self.pheromones:
                self.pheromones[key] *= (1 - rho)
                self.pheromones[key] = max(self.min_tau, self.pheromones[key])
            
            # Reward best
            if best_global_schedule:
                for op in best_global_schedule:
                    key = (op.job_id, op.op_idx, op.machine_id)
                    self.pheromones[key] = min(self.max_tau, self.pheromones[key] + deposit)
            
            # Punish worst
            if iter_worst_ant:
                for op in iter_worst_ant.schedule:
                    key = (op.job_id, op.op_idx, op.machine_id)
                    self.pheromones[key] = max(self.min_tau, self.pheromones[key] - punishment)
            
            # Early stopping
            if stagnation >= cfg.stop_patience:
                # Fill remaining iterations
                while len(convergence) < cfg.max_iter:
                    convergence.append(best_global_mk)
                    time_curve.append(time.time() - start_time)
                break
        
        return best_global_mk, convergence, time_curve


# ==============================================================================
# BENCHMARK DATA
# ==============================================================================

BRANDIMARTE_BKS = {
    'Mk01': 40, 'Mk02': 26, 'Mk03': 204, 'Mk04': 60, 'Mk05': 172,
    'Mk06': 57, 'Mk07': 139, 'Mk08': 523, 'Mk09': 307, 'Mk10': 197
}

LAWRENCE_BKS = {
    'LA01': 666, 'LA02': 655, 'LA03': 597, 'LA04': 590, 'LA05': 593,
    'LA06': 926, 'LA07': 890, 'LA08': 863, 'LA09': 951, 'LA10': 958,
    'LA11': 1222, 'LA12': 1039, 'LA13': 1150, 'LA14': 1292, 'LA15': 1207,
    'LA16': 945, 'LA17': 784, 'LA18': 848, 'LA19': 842, 'LA20': 902,
    'LA21': 1046, 'LA22': 927, 'LA23': 1032, 'LA24': 935, 'LA25': 977,
    'LA26': 1218, 'LA27': 1235, 'LA28': 1216, 'LA29': 1152, 'LA30': 1355,
    'LA31': 1784, 'LA32': 1850, 'LA33': 1719, 'LA34': 1721, 'LA35': 1888,
    'LA36': 1268, 'LA37': 1397, 'LA38': 1196, 'LA39': 1233, 'LA40': 1222
}


# ==============================================================================
# RESULTS MANAGEMENT
# ==============================================================================

@dataclass
class RunResult:
    instance_name: str
    bks: int
    run_scores: List[float]
    run_times: List[float]
    convergence: List[List[float]]
    time_curves: List[List[float]]
    
    @property
    def best_score(self) -> float:
        return min(self.run_scores)
    
    @property
    def worst_score(self) -> float:
        return max(self.run_scores)
    
    @property
    def mean_score(self) -> float:
        return np.mean(self.run_scores)
    
    @property
    def std_score(self) -> float:
        return np.std(self.run_scores)
    
    @property
    def gap_best(self) -> float:
        return ((self.best_score - self.bks) / self.bks) * 100 if self.bks > 0 else 0
    
    @property
    def gap_mean(self) -> float:
        return ((self.mean_score - self.bks) / self.bks) * 100 if self.bks > 0 else 0
    
    @property
    def mean_time(self) -> float:
        return np.mean(self.run_times)


def write_csv_results(filename: str, results: List[RunResult]):
    """Write results to CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Instance', 'BKS', 'Best', 'Worst', 'Mean', 'Std', 'Gap_Best(%)', 'Gap_Mean(%)', 'Mean_Time(s)']
        max_runs = max(len(r.run_scores) for r in results)
        for i in range(max_runs):
            header.extend([f'Run{i+1}_Score', f'Run{i+1}_Time'])
        writer.writerow(header)
        
        # Data
        for r in results:
            row = [r.instance_name, r.bks, r.best_score, r.worst_score, 
                   f'{r.mean_score:.2f}', f'{r.std_score:.2f}',
                   f'{r.gap_best:.2f}', f'{r.gap_mean:.2f}', f'{r.mean_time:.2f}']
            for i in range(max_runs):
                if i < len(r.run_scores):
                    row.extend([f'{r.run_scores[i]:.0f}', f'{r.run_times[i]:.2f}'])
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    
    print(f"Results saved to: {filename}")


def write_text_report(filename: str, results: List[RunResult], config: Config):
    """Write detailed text report"""
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("         FJSP-ACO EXPERIMENTAL RESULTS - STATISTICAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write("ALGORITHM CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Number of ants:        {config.num_ants}\n")
        f.write(f"  Max iterations:        {config.max_iter}\n")
        f.write(f"  Stopping patience:     {config.stop_patience}\n")
        f.write(f"  Alpha (start-end):     {config.alpha_start} - {config.alpha_end}\n")
        f.write(f"  Beta (start-end):      {config.beta_start} - {config.beta_end}\n")
        f.write(f"  Rho (min-max):         {config.rho_min} - {config.rho_max}\n")
        f.write(f"  q0:                    {config.q0}\n")
        f.write(f"  Local search attempts: {config.ls_max_attempts}\n")
        f.write(f"  Independent runs:      {config.independent_runs}\n\n")
        
        # Summary table
        f.write("SUMMARY RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Instance':<12} {'BKS':>6} {'Best':>8} {'Worst':>8} {'Mean':>8} {'Std':>8} {'Gap_B%':>10} {'Gap_M%':>10} {'Time':>8}\n")
        f.write("-" * 80 + "\n")
        
        total_gap_best = 0
        total_gap_mean = 0
        optimal_count = 0
        
        for r in results:
            f.write(f"{r.instance_name:<12} {r.bks:>6} {r.best_score:>8.0f} {r.worst_score:>8.0f} ")
            f.write(f"{r.mean_score:>8.2f} {r.std_score:>8.2f} {r.gap_best:>10.2f} {r.gap_mean:>10.2f} {r.mean_time:>8.2f}\n")
            
            total_gap_best += r.gap_best
            total_gap_mean += r.gap_mean
            if r.gap_best <= 0.01:
                optimal_count += 1
        
        f.write("-" * 80 + "\n")
        avg_gap_best = total_gap_best / len(results)
        avg_gap_mean = total_gap_mean / len(results)
        f.write(f"{'AVERAGE':<12} {'':<6} {'':<8} {'':<8} {'':<8} {'':<8} {avg_gap_best:>10.2f} {avg_gap_mean:>10.2f}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total instances tested:    {len(results)}\n")
        f.write(f"  Optimal solutions found:   {optimal_count} ({100*optimal_count/len(results):.1f}%)\n")
        f.write(f"  Average gap (best):        {avg_gap_best:.2f}%\n")
        f.write(f"  Average gap (mean):        {avg_gap_mean:.2f}%\n")
    
    print(f"Report saved to: {filename}")


def plot_convergence(results: List[RunResult], output_dir: str = 'plots'):
    """Generate convergence plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    for r in results:
        if not r.convergence:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # Calculate mean and range
        conv_array = np.array(r.convergence)
        mean_conv = np.mean(conv_array, axis=0)
        min_conv = np.min(conv_array, axis=0)
        max_conv = np.max(conv_array, axis=0)
        iterations = np.arange(len(mean_conv))
        
        # Time curve (use first run)
        time_curve = r.time_curves[0] if r.time_curves else iterations
        
        # Plot vs iterations
        axes[0].fill_between(iterations, min_conv, max_conv, alpha=0.3, color='blue')
        axes[0].plot(iterations, mean_conv, 'b-', linewidth=2, label='Mean')
        axes[0].axhline(y=r.bks, color='r', linestyle='--', label=f'BKS={r.bks}')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Makespan')
        axes[0].set_title(f'{r.instance_name} - Convergence vs Iteration')
        axes[0].legend()
        
        # Plot vs time
        axes[1].fill_between(time_curve, min_conv, max_conv, alpha=0.3, color='green')
        axes[1].plot(time_curve, mean_conv, 'g-', linewidth=2, label='Mean')
        axes[1].axhline(y=r.bks, color='r', linestyle='--', label=f'BKS={r.bks}')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Makespan')
        axes[1].set_title(f'{r.instance_name} - Convergence vs Time')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'convergence_{r.instance_name}.png'), dpi=150)
        plt.close()
    
    print(f"Plots saved to: {output_dir}/")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_experiments(instances: List[Tuple[FJSPInstance, int]], config: Config) -> List[RunResult]:
    """Run experiments on all instances"""
    results = []
    
    for instance, bks in instances:
        print(f"\nTesting {instance.name} (BKS={bks})...")
        
        run_scores = []
        run_times = []
        convergence = []
        time_curves = []
        
        for run in range(config.independent_runs):
            start_time = time.time()
            
            optimizer = ACOOptimizer(instance, config)
            score, conv, time_curve = optimizer.optimize()
            
            elapsed = time.time() - start_time
            
            run_scores.append(score)
            run_times.append(elapsed)
            convergence.append(conv)
            time_curves.append(time_curve)
            
            gap = ((score - bks) / bks) * 100 if bks > 0 else 0
            print(f"  Run {run+1:2d}: Score={score:6.0f}, Gap={gap:6.2f}%, Time={elapsed:.2f}s")
        
        result = RunResult(
            instance_name=instance.name,
            bks=bks,
            run_scores=run_scores,
            run_times=run_times,
            convergence=convergence,
            time_curves=time_curves
        )
        
        print(f"  Summary: Best={result.best_score:.0f}, Mean={result.mean_score:.2f}, Gap_Best={result.gap_best:.2f}%")
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FJSP-ACO Solver')
    parser.add_argument('--mk', action='store_true', help='Run Brandimarte instances')
    parser.add_argument('--la', action='store_true', help='Run Lawrence instances')
    parser.add_argument('--start', type=int, default=1, help='Start instance number')
    parser.add_argument('--end', type=int, default=None, help='End instance number')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs')
    parser.add_argument('--iter', type=int, default=100, help='Max iterations')
    args = parser.parse_args()
    
    # If neither specified, run both
    if not args.mk and not args.la:
        args.mk = True
        args.la = True
    
    # Configuration
    config = Config(
        independent_runs=args.runs,
        max_iter=args.iter
    )
    
    print("=" * 70)
    print("         FJSP-ACO: Ant Colony Optimization Solver")
    print("=" * 70)
    
    instances = []
    
    # Load Brandimarte instances
    if args.mk:
        end = args.end or 10
        print(f"\nLoading Brandimarte instances Mk{args.start:02d}-Mk{end:02d}...")
        for i in range(args.start, end + 1):
            filepath = f"benchmarks/Mk{i:02d}.txt"
            if os.path.exists(filepath):
                inst = load_brandimarte(filepath)
                if inst:
                    bks = BRANDIMARTE_BKS.get(f'Mk{i:02d}', 0)
                    instances.append((inst, bks))
                    print(f"  Loaded {inst.name}: {inst.num_jobs}x{inst.num_machines}, {inst.total_ops} ops")
            else:
                print(f"  Warning: {filepath} not found")
    
    # Load Lawrence instances
    if args.la:
        end = args.end or 40
        print(f"\nLoading Lawrence instances LA{args.start:02d}-LA{end:02d}...")
        for i in range(args.start, end + 1):
            filepath = f"benchmarks/la{i:02d}.txt"
            if os.path.exists(filepath):
                inst = load_lawrence(filepath)
                if inst:
                    bks = LAWRENCE_BKS.get(f'LA{i:02d}', 0)
                    instances.append((inst, bks))
                    print(f"  Loaded {inst.name}: {inst.num_jobs}x{inst.num_machines}, {inst.total_ops} ops")
            else:
                print(f"  Warning: {filepath} not found")
    
    if not instances:
        print("\nNo instances loaded! Make sure benchmark files are in 'benchmarks/' folder.")
        print("Expected files: benchmarks/Mk01.txt, benchmarks/la01.txt, etc.")
        return
    
    # Run experiments
    print(f"\n{'=' * 70}")
    print(f"Running experiments on {len(instances)} instances...")
    print(f"{'=' * 70}")
    
    results = run_experiments(instances, config)
    
    # Save results
    print(f"\n{'=' * 70}")
    print("Saving results...")
    print(f"{'=' * 70}")
    
    write_csv_results('results.csv', results)
    write_text_report('report.txt', results, config)
    
    # Generate plots
    try:
        plot_convergence(results)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("                      FINAL SUMMARY")
    print(f"{'=' * 70}")
    
    total_gap_best = sum(r.gap_best for r in results)
    total_gap_mean = sum(r.gap_mean for r in results)
    optimal_count = sum(1 for r in results if r.gap_best <= 0.01)
    
    print(f"\n{'Instance':<12} {'BKS':>6} {'Best':>8} {'Mean':>8} {'Gap_B%':>10} {'Gap_M%':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.instance_name:<12} {r.bks:>6} {r.best_score:>8.0f} {r.mean_score:>8.2f} {r.gap_best:>10.2f} {r.gap_mean:>10.2f}")
    print("-" * 60)
    print(f"{'AVERAGE':<12} {'':<6} {'':<8} {'':<8} {total_gap_best/len(results):>10.2f} {total_gap_mean/len(results):>10.2f}")
    print(f"\nOptimal solutions: {optimal_count}/{len(results)} ({100*optimal_count/len(results):.1f}%)")
    
    print("\nOutput files: results.csv, report.txt, plots/")
    print("Done!")


if __name__ == '__main__':
    main()
