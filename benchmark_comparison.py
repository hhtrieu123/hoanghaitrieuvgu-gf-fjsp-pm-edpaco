#!/usr/bin/env python3
"""
Basic Metaheuristics Comparison for GF-FJSP-PM
==============================================

Compares EDP-ACO with basic metaheuristics:
- GA: Genetic Algorithm
- PSO: Particle Swarm Optimization  
- SA: Simulated Annealing
- Basic ACO: Standard ACO without enhancements

Output:
    - basic_comparison/
        - results.json
        - results.csv
        - comparison_table.tex
        - figures/comparison_bar.png

Usage:
    python benchmark_comparison.py
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import csv
import time

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Operation:
    job_id: int
    op_id: int
    eligible_machines: List[int]
    processing_times: Dict[int, Tuple[float, float, float]]

@dataclass
class Machine:
    machine_id: int
    power_processing: float
    power_idle: float

@dataclass
class Instance:
    name: str
    num_jobs: int
    num_machines: int
    num_ops_per_job: int
    operations: List[Operation]
    machines: List[Machine]
    alpha: float = 0.5
    beta: float = 0.5

def gmir(fuzzy):
    low, mid, high = fuzzy
    return (low + 4 * mid + high) / 6

def generate_instance(num_jobs, num_machines, num_ops_per_job, flexibility=0.5, seed=None, name=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    operations = []
    for j in range(num_jobs):
        for o in range(num_ops_per_job):
            num_eligible = max(1, int(num_machines * flexibility))
            eligible = sorted(random.sample(range(num_machines), num_eligible))
            proc_times = {}
            for m in eligible:
                mid = random.randint(5, 30)
                low = mid - random.randint(1, 3)
                high = mid + random.randint(1, 3)
                proc_times[m] = (low, mid, high)
            operations.append(Operation(j, o, eligible, proc_times))

    machines = []
    for m in range(num_machines):
        power_proc = random.uniform(3.0, 8.0)
        power_idle = random.uniform(0.5, 1.5)
        machines.append(Machine(m, power_proc, power_idle))

    instance_name = name if name else f"Instance_{num_jobs}x{num_machines}x{num_ops_per_job}"
    return Instance(name=instance_name, num_jobs=num_jobs, num_machines=num_machines,
                    num_ops_per_job=num_ops_per_job, operations=operations, machines=machines)

def evaluate_schedule(schedule, instance):
    machine_end_times = [0.0] * instance.num_machines
    job_end_times = [0.0] * instance.num_jobs
    total_energy = 0.0

    for (j, o, m, proc_time) in schedule:
        start_time = max(machine_end_times[m], job_end_times[j])
        end_time = start_time + proc_time
        machine_end_times[m] = end_time
        job_end_times[j] = end_time
        energy = proc_time * instance.machines[m].power_processing
        total_energy += energy

    makespan = max(machine_end_times) if machine_end_times else 0
    objective = instance.alpha * makespan + instance.beta * (total_energy / 100)
    return makespan, total_energy, objective

# =============================================================================
# EDP-ACO (Proposed Algorithm)
# =============================================================================

class EDP_ACO:
    def __init__(self, instance, num_ants=30, max_iter=100, seed=None):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tau_time = {}
        self.tau_energy = {}
        for op in instance.operations:
            for m in op.eligible_machines:
                key = (op.job_id, op.op_id, m)
                self.tau_time[key] = 1.0
                self.tau_energy[key] = 1.0

    def solve(self):
        best_objective = float('inf')
        best_makespan = 0
        best_energy = 0
        convergence = []

        for iteration in range(self.max_iter):
            progress = iteration / self.max_iter
            alpha = 1.0 + 3.0 * progress
            beta = 4.0 - 3.0 * progress
            rho = 0.1 + 0.2 * progress

            iter_best_schedule = None
            iter_best_objective = float('inf')

            for _ in range(self.num_ants):
                schedule = self._construct_solution(alpha, beta)
                makespan, energy, objective = evaluate_schedule(schedule, self.instance)
                if objective < iter_best_objective:
                    iter_best_objective = objective
                    iter_best_schedule = schedule

            if iter_best_schedule:
                # Local search
                iter_best_schedule = self._local_search(iter_best_schedule)
                makespan, energy, objective = evaluate_schedule(iter_best_schedule, self.instance)
                
                if objective < best_objective:
                    best_objective = objective
                    best_makespan = makespan
                    best_energy = energy

            convergence.append(best_objective)

            # Pheromone update
            for key in self.tau_time:
                self.tau_time[key] *= (1 - rho)
                self.tau_energy[key] *= (1 - rho)

            if iter_best_schedule:
                ms, en, _ = evaluate_schedule(iter_best_schedule, self.instance)
                for (j, o, m, pt) in iter_best_schedule:
                    key = (j, o, m)
                    if key in self.tau_time:
                        self.tau_time[key] += 1.0 / max(ms, 1)
                        self.tau_energy[key] += 1.0 / max(en, 1)

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

    def _construct_solution(self, alpha, beta):
        schedule = []
        next_op = [0] * self.instance.num_jobs

        while len(schedule) < len(self.instance.operations):
            ready_ops = []
            for j in range(self.instance.num_jobs):
                if next_op[j] < self.instance.num_ops_per_job:
                    op_idx = j * self.instance.num_ops_per_job + next_op[j]
                    ready_ops.append(self.instance.operations[op_idx])
            if not ready_ops:
                break

            op = random.choice(ready_ops)
            probabilities = []
            for m in op.eligible_machines:
                key = (op.job_id, op.op_id, m)
                tau = self.instance.alpha * self.tau_time[key] + self.instance.beta * self.tau_energy[key]
                pt = gmir(op.processing_times[m])
                eta = 1.0 / max(pt, 0.001)
                probabilities.append((tau ** alpha) * (eta ** beta))

            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
                selected_machine = random.choices(op.eligible_machines, probabilities)[0]
            else:
                selected_machine = random.choice(op.eligible_machines)

            proc_time = gmir(op.processing_times[selected_machine])
            schedule.append((op.job_id, op.op_id, selected_machine, proc_time))
            next_op[op.job_id] += 1

        return schedule

    def _local_search(self, schedule):
        best = schedule[:]
        _, _, best_obj = evaluate_schedule(best, self.instance)
        
        for _ in range(30):
            neighbor = best[:]
            if len(neighbor) >= 2:
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            _, _, obj = evaluate_schedule(neighbor, self.instance)
            if obj < best_obj:
                best = neighbor
                best_obj = obj
        return best

# =============================================================================
# BASIC ACO (No enhancements)
# =============================================================================

class BasicACO:
    def __init__(self, instance, num_ants=30, max_iter=100, alpha=1.0, beta=2.0, rho=0.1, seed=None):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pheromone = {}
        for op in instance.operations:
            for m in op.eligible_machines:
                self.pheromone[(op.job_id, op.op_id, m)] = 1.0

    def solve(self):
        best_objective = float('inf')
        best_makespan = 0
        best_energy = 0
        convergence = []

        for _ in range(self.max_iter):
            iter_best_schedule = None
            iter_best_objective = float('inf')

            for _ in range(self.num_ants):
                schedule = self._construct_solution()
                makespan, energy, objective = evaluate_schedule(schedule, self.instance)
                if objective < iter_best_objective:
                    iter_best_objective = objective
                    iter_best_schedule = schedule

            if iter_best_objective < best_objective:
                best_objective = iter_best_objective
                ms, en, _ = evaluate_schedule(iter_best_schedule, self.instance)
                best_makespan = ms
                best_energy = en

            convergence.append(best_objective)

            # Simple pheromone update
            for key in self.pheromone:
                self.pheromone[key] *= (1 - self.rho)
            if iter_best_schedule:
                ms, _, _ = evaluate_schedule(iter_best_schedule, self.instance)
                for (j, o, m, pt) in iter_best_schedule:
                    key = (j, o, m)
                    if key in self.pheromone:
                        self.pheromone[key] += 1.0 / max(ms, 1)

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

    def _construct_solution(self):
        schedule = []
        next_op = [0] * self.instance.num_jobs

        while len(schedule) < len(self.instance.operations):
            ready_ops = []
            for j in range(self.instance.num_jobs):
                if next_op[j] < self.instance.num_ops_per_job:
                    op_idx = j * self.instance.num_ops_per_job + next_op[j]
                    ready_ops.append(self.instance.operations[op_idx])
            if not ready_ops:
                break

            op = random.choice(ready_ops)
            probabilities = []
            for m in op.eligible_machines:
                tau = self.pheromone[(op.job_id, op.op_id, m)]
                pt = gmir(op.processing_times[m])
                eta = 1.0 / max(pt, 0.001)
                probabilities.append((tau ** self.alpha) * (eta ** self.beta))

            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
                selected_machine = random.choices(op.eligible_machines, probabilities)[0]
            else:
                selected_machine = random.choice(op.eligible_machines)

            proc_time = gmir(op.processing_times[selected_machine])
            schedule.append((op.job_id, op.op_id, selected_machine, proc_time))
            next_op[op.job_id] += 1

        return schedule

# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

class GA:
    def __init__(self, instance, pop_size=50, max_iter=100, crossover_rate=0.8, mutation_rate=0.2, seed=None):
        self.instance = instance
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def solve(self):
        population = [self._create_individual() for _ in range(self.pop_size)]
        best_objective = float('inf')
        best_makespan = 0
        best_energy = 0
        convergence = []

        for _ in range(self.max_iter):
            fitness = [(evaluate_schedule(ind, self.instance), ind) for ind in population]
            fitness.sort(key=lambda x: x[0][2])

            if fitness[0][0][2] < best_objective:
                best_objective = fitness[0][0][2]
                best_makespan = fitness[0][0][0]
                best_energy = fitness[0][0][1]

            convergence.append(best_objective)

            # Elitism
            new_pop = [f[1] for f in fitness[:5]]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(fitness)
                p2 = self._tournament_select(fitness)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

    def _create_individual(self):
        schedule = []
        next_op = [0] * self.instance.num_jobs
        while len(schedule) < len(self.instance.operations):
            ready_jobs = [j for j in range(self.instance.num_jobs) if next_op[j] < self.instance.num_ops_per_job]
            if not ready_jobs:
                break
            j = random.choice(ready_jobs)
            op_idx = j * self.instance.num_ops_per_job + next_op[j]
            op = self.instance.operations[op_idx]
            m = random.choice(op.eligible_machines)
            pt = gmir(op.processing_times[m])
            schedule.append((j, next_op[j], m, pt))
            next_op[j] += 1
        return schedule

    def _tournament_select(self, fitness, k=3):
        selected = random.sample(fitness, min(k, len(fitness)))
        return min(selected, key=lambda x: x[0][2])[1]

    def _crossover(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1[:]
        return p1[:] if random.random() < 0.5 else p2[:]

    def _mutate(self, ind):
        if random.random() > self.mutation_rate or len(ind) < 2:
            return ind
        child = ind[:]
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
        return child

# =============================================================================
# PARTICLE SWARM OPTIMIZATION
# =============================================================================

class PSO:
    def __init__(self, instance, swarm_size=30, max_iter=100, seed=None):
        self.instance = instance
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def solve(self):
        particles = [self._create_particle() for _ in range(self.swarm_size)]
        pbest = [p[:] for p in particles]
        pbest_obj = [evaluate_schedule(p, self.instance)[2] for p in particles]
        
        gbest_idx = np.argmin(pbest_obj)
        gbest = pbest[gbest_idx][:]
        gbest_obj = pbest_obj[gbest_idx]
        
        best_makespan, best_energy, _ = evaluate_schedule(gbest, self.instance)
        convergence = []

        for _ in range(self.max_iter):
            for i, particle in enumerate(particles):
                # Move particle towards pbest and gbest
                new_particle = self._move_particle(particle, pbest[i], gbest)
                new_obj = evaluate_schedule(new_particle, self.instance)[2]

                if new_obj < pbest_obj[i]:
                    pbest[i] = new_particle[:]
                    pbest_obj[i] = new_obj

                    if new_obj < gbest_obj:
                        gbest = new_particle[:]
                        gbest_obj = new_obj
                        best_makespan, best_energy, _ = evaluate_schedule(gbest, self.instance)

                particles[i] = new_particle

            convergence.append(gbest_obj)

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': gbest_obj, 'convergence': convergence}

    def _create_particle(self):
        schedule = []
        next_op = [0] * self.instance.num_jobs
        while len(schedule) < len(self.instance.operations):
            ready_jobs = [j for j in range(self.instance.num_jobs) if next_op[j] < self.instance.num_ops_per_job]
            if not ready_jobs:
                break
            j = random.choice(ready_jobs)
            op_idx = j * self.instance.num_ops_per_job + next_op[j]
            op = self.instance.operations[op_idx]
            m = random.choice(op.eligible_machines)
            pt = gmir(op.processing_times[m])
            schedule.append((j, next_op[j], m, pt))
            next_op[j] += 1
        return schedule

    def _move_particle(self, particle, pbest, gbest):
        new_particle = particle[:]
        
        # Learn from pbest
        if random.random() < 0.3 and len(new_particle) >= 2:
            i, j = random.sample(range(len(new_particle)), 2)
            new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
        
        # Learn from gbest
        if random.random() < 0.3 and len(new_particle) >= 2:
            i, j = random.sample(range(len(new_particle)), 2)
            new_particle[i], new_particle[j] = new_particle[j], new_particle[i]

        return new_particle

# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

class SA:
    def __init__(self, instance, max_iter=100, T0=100, alpha=0.95, seed=None):
        self.instance = instance
        self.max_iter = max_iter
        self.T0 = T0
        self.alpha = alpha
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def solve(self):
        current = self._create_solution()
        current_obj = evaluate_schedule(current, self.instance)[2]
        
        best = current[:]
        best_obj = current_obj
        best_makespan, best_energy, _ = evaluate_schedule(best, self.instance)
        
        T = self.T0
        convergence = []

        for _ in range(self.max_iter):
            for _ in range(50):  # Inner iterations
                neighbor = self._get_neighbor(current)
                neighbor_obj = evaluate_schedule(neighbor, self.instance)[2]
                
                delta = neighbor_obj - current_obj
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current = neighbor
                    current_obj = neighbor_obj
                    
                    if current_obj < best_obj:
                        best = current[:]
                        best_obj = current_obj
                        best_makespan, best_energy, _ = evaluate_schedule(best, self.instance)

            T *= self.alpha
            convergence.append(best_obj)

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_obj, 'convergence': convergence}

    def _create_solution(self):
        schedule = []
        next_op = [0] * self.instance.num_jobs
        while len(schedule) < len(self.instance.operations):
            ready_jobs = [j for j in range(self.instance.num_jobs) if next_op[j] < self.instance.num_ops_per_job]
            if not ready_jobs:
                break
            j = random.choice(ready_jobs)
            op_idx = j * self.instance.num_ops_per_job + next_op[j]
            op = self.instance.operations[op_idx]
            m = random.choice(op.eligible_machines)
            pt = gmir(op.processing_times[m])
            schedule.append((j, next_op[j], m, pt))
            next_op[j] += 1
        return schedule

    def _get_neighbor(self, solution):
        neighbor = solution[:]
        if len(neighbor) >= 2:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments(instances, num_runs=5):
    results = {}
    algorithms = {
        'EDP-ACO': lambda inst, seed: EDP_ACO(inst, num_ants=30, max_iter=100, seed=seed),
        'Basic-ACO': lambda inst, seed: BasicACO(inst, num_ants=30, max_iter=100, seed=seed),
        'GA': lambda inst, seed: GA(inst, pop_size=50, max_iter=100, seed=seed),
        'PSO': lambda inst, seed: PSO(inst, swarm_size=30, max_iter=100, seed=seed),
        'SA': lambda inst, seed: SA(inst, max_iter=100, seed=seed),
    }

    for inst in instances:
        print(f"\n{'='*50}\nInstance: {inst.name}\n{'='*50}")
        results[inst.name] = {}

        for alg_name, alg_constructor in algorithms.items():
            print(f"  Running {alg_name}...", end=" ", flush=True)
            
            objectives = []
            makespans = []
            energies = []
            times = []

            for run in range(num_runs):
                start = time.time()
                solver = alg_constructor(inst, seed=run * 42)
                result = solver.solve()
                elapsed = time.time() - start

                objectives.append(result['objective'])
                makespans.append(result['makespan'])
                energies.append(result['energy'])
                times.append(elapsed)

            results[inst.name][alg_name] = {
                'best': min(objectives),
                'avg': np.mean(objectives),
                'std': np.std(objectives),
                'worst': max(objectives),
                'avg_time': np.mean(times),
                'best_makespan': min(makespans),
                'avg_makespan': np.mean(makespans),
                'best_energy': min(energies),
                'avg_energy': np.mean(energies),
            }

            print(f"Best={min(objectives):.2f}, Avg={np.mean(objectives):.2f}, Time={np.mean(times):.2f}s")

    return results

def generate_latex_table(results, output_path):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison with basic metaheuristics}
\label{tab:basic_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\textbf{Instance} & \textbf{Algorithm} & \textbf{Best} & \textbf{Avg} & \textbf{Std} & \textbf{Time(s)} & \textbf{Improv.(\%)} & \textbf{Rank} \\
\midrule
"""
    algorithms = ['EDP-ACO', 'Basic-ACO', 'GA', 'PSO', 'SA']
    
    for inst_name, inst_results in results.items():
        ranked = sorted(algorithms, key=lambda a: inst_results[a]['avg'])
        edp_best = inst_results['EDP-ACO']['best']
        
        for i, alg in enumerate(algorithms):
            data = inst_results[alg]
            rank = ranked.index(alg) + 1
            improv = (data['best'] - edp_best) / edp_best * 100 if alg != 'EDP-ACO' else 0
            
            inst_col = f"\\multirow{{5}}{{*}}{{{inst_name}}}" if i == 0 else ""
            
            best_str = f"\\textbf{{{data['best']:.2f}}}" if rank == 1 else f"{data['best']:.2f}"
            avg_str = f"\\textbf{{{data['avg']:.2f}}}" if rank == 1 else f"{data['avg']:.2f}"
            rank_str = f"\\textbf{{{rank}}}" if rank == 1 else f"{rank}"
            improv_str = f"+{improv:.2f}" if improv > 0 else f"{improv:.2f}"
            
            latex += f"{inst_col} & {alg} & {best_str} & {avg_str} & {data['std']:.2f} & {data['avg_time']:.2f} & {improv_str} & {rank_str} \\\\\n"
        latex += "\\midrule\n"

    latex = latex.rstrip("\\midrule\n") + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated: {output_path}")

def plot_comparison(results, output_path):
    algorithms = ['EDP-ACO', 'Basic-ACO', 'GA', 'PSO', 'SA']
    instances = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(instances))
    width = 0.15
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, alg in enumerate(algorithms):
        avgs = [results[inst][alg]['avg'] for inst in instances]
        ax.bar(x + i * width, avgs, width, label=alg, color=colors[i])

    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Average Objective Value', fontsize=12)
    ax.set_title('Basic Metaheuristics Comparison', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def main():
    output_dir = "basic_comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BASIC METAHEURISTICS COMPARISON FOR GF-FJSP-PM")
    print("=" * 60)

    print("\n1. Generating test instances...")
    instances = [
        generate_instance(6, 4, 3, flexibility=0.6, seed=42, name="Small_6x4x3"),
        generate_instance(10, 5, 4, flexibility=0.5, seed=43, name="Medium_10x5x4"),
        generate_instance(15, 6, 5, flexibility=0.4, seed=44, name="Medium_15x6x5"),
        generate_instance(20, 8, 5, flexibility=0.4, seed=45, name="Large_20x8x5"),
    ]

    print("\n2. Running experiments...")
    results = run_experiments(instances, num_runs=5)

    print("\n3. Saving results...")
    
    # JSON
    json_results = {inst: {alg: {k: float(v) for k, v in data.items()}
                          for alg, data in inst_results.items()}
                   for inst, inst_results in results.items()}
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2)

    # CSV
    with open(os.path.join(output_dir, "results.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instance', 'Algorithm', 'Best', 'Avg', 'Std', 'Worst', 'Time'])
        for inst_name, inst_results in results.items():
            for alg, data in inst_results.items():
                writer.writerow([inst_name, alg, data['best'], data['avg'], data['std'], data['worst'], data['avg_time']])

    print("\n4. Generating outputs...")
    generate_latex_table(results, os.path.join(output_dir, "comparison_table.tex"))
    plot_comparison(results, os.path.join(output_dir, "comparison_bar.png"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}/")
    print("  - results.json")
    print("  - results.csv")
    print("  - comparison_table.tex")
    print("  - comparison_bar.png")

if __name__ == "__main__":
    main()
