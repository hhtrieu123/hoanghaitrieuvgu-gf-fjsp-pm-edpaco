#!/usr/bin/env python3
"""
Complete Thesis Experiment Runner
==================================

Run all experiments for thesis:
1. MILP verification on small instances
2. Metaheuristic comparison (ACO, GA, PSO, SA)
3. Convergence analysis
4. Statistical tests
5. Generate all tables and figures

Usage:
    python run_all_experiments.py

Author: Master's Thesis
"""

import numpy as np
import random
import time
import json
import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Try imports
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("matplotlib not available - plots will be skipped")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available - statistical tests will be skipped")

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("pulp not available - MILP will be skipped")


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


@dataclass
class Solution:
    schedule: List[Tuple[int, int, int]]
    makespan: float = 0.0
    energy: float = 0.0
    objective: float = 0.0
    start_times: Dict = field(default_factory=dict)
    end_times: Dict = field(default_factory=dict)


# =============================================================================
# INSTANCE GENERATOR
# =============================================================================

def generate_instance(name: str, num_jobs: int, num_machines: int, ops_per_job: int,
                      flexibility: float = 0.5, fuzziness: float = 0.2,
                      time_range: Tuple[int, int] = (10, 100), seed: int = None) -> Instance:
    
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
                M_val = random.randint(time_range[0], time_range[1])
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
# DECODER
# =============================================================================

def decode(instance: Instance, assignment: List[Tuple[int, int, int]]) -> Solution:
    machine_ready = {m.machine_id: 0.0 for m in instance.machines}
    job_ready = {j.job_id: 0.0 for j in instance.jobs}
    
    start_times = {}
    end_times = {}
    
    for job_id, op_id, mach_id in assignment:
        job = instance.jobs[job_id]
        op = job.operations[op_id]
        proc_time = op.alternatives[mach_id].gmir()
        
        start = max(machine_ready[mach_id], job_ready[job_id])
        end = start + proc_time
        
        start_times[(job_id, op_id)] = start
        end_times[(job_id, op_id)] = end
        
        machine_ready[mach_id] = end
        job_ready[job_id] = end
    
    makespan = max(end_times.values()) if end_times else 0
    
    # Energy calculation
    energy_proc = 0.0
    energy_idle = 0.0
    
    machine_work = {m.machine_id: [] for m in instance.machines}
    for job_id, op_id, mach_id in assignment:
        machine_work[mach_id].append((start_times[(job_id, op_id)], 
                                      end_times[(job_id, op_id)]))
    
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
    
    return Solution(schedule=assignment, makespan=makespan, energy=energy,
                   objective=objective, start_times=start_times, end_times=end_times)


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
# ACO ALGORITHM
# =============================================================================

class ACO:
    def __init__(self, instance: Instance, num_ants: int = 20, max_iter: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1):
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
    
    def construct_solution(self) -> Solution:
        assignment = []
        job_next_op = {j.job_id: 0 for j in self.instance.jobs}
        total_ops = self.instance.total_ops
        
        while len(assignment) < total_ops:
            ready = []
            for job in self.instance.jobs:
                next_op = job_next_op[job.job_id]
                if next_op < len(job.operations):
                    ready.append((job.job_id, next_op))
            
            if not ready:
                break
            
            job_id, op_id = random.choice(ready)
            op = self.instance.jobs[job_id].operations[op_id]
            
            probs = []
            machines = list(op.alternatives.keys())
            
            for m in machines:
                tau = self.pheromone.get((job_id, op_id, m), 1.0)
                eta = 1.0 / op.alternatives[m].gmir()
                prob = (tau ** self.alpha) * (eta ** self.beta)
                probs.append(prob)
            
            total = sum(probs)
            probs = [p / total for p in probs]
            selected_machine = np.random.choice(machines, p=probs)
            
            assignment.append((job_id, op_id, selected_machine))
            job_next_op[job_id] += 1
        
        return decode(self.instance, assignment)
    
    def update_pheromone(self, best_solution: Solution):
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)
        
        deposit = 1.0 / best_solution.objective if best_solution.objective > 0 else 1.0
        for job_id, op_id, mach_id in best_solution.schedule:
            self.pheromone[(job_id, op_id, mach_id)] += deposit
    
    def solve(self) -> Tuple[Solution, List[float]]:
        best_solution = None
        convergence = []
        
        for _ in range(self.max_iter):
            solutions = [self.construct_solution() for _ in range(self.num_ants)]
            iter_best = min(solutions, key=lambda s: s.objective)
            
            if best_solution is None or iter_best.objective < best_solution.objective:
                best_solution = iter_best
            
            self.update_pheromone(best_solution)
            convergence.append(best_solution.objective)
        
        return best_solution, convergence


# =============================================================================
# GA ALGORITHM
# =============================================================================

class GA:
    def __init__(self, instance: Instance, pop_size: int = 50, max_iter: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.instance = instance
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def random_chromosome(self) -> List[Tuple[int, int, int]]:
        chromosome = []
        for job in self.instance.jobs:
            for op in job.operations:
                machine = random.choice(list(op.alternatives.keys()))
                chromosome.append((job.job_id, op.op_id, machine))
        random.shuffle(chromosome)
        return repair_precedence(self.instance, chromosome)
    
    def crossover(self, p1: List, p2: List) -> Tuple[List, List]:
        if random.random() > self.crossover_rate:
            return p1[:], p2[:]
        
        size = len(p1)
        pt1, pt2 = sorted(random.sample(range(size), 2))
        
        c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        
        return repair_precedence(self.instance, c1), repair_precedence(self.instance, c2)
    
    def mutate(self, chromosome: List) -> List:
        if random.random() > self.mutation_rate:
            return chromosome
        
        mutated = chromosome[:]
        idx = random.randint(0, len(mutated) - 1)
        job_id, op_id, _ = mutated[idx]
        op = self.instance.jobs[job_id].operations[op_id]
        new_machine = random.choice(list(op.alternatives.keys()))
        mutated[idx] = (job_id, op_id, new_machine)
        return mutated
    
    def solve(self) -> Tuple[Solution, List[float]]:
        population = [self.random_chromosome() for _ in range(self.pop_size)]
        best_solution = None
        convergence = []
        
        for _ in range(self.max_iter):
            solutions = [decode(self.instance, chrom) for chrom in population]
            iter_best = min(solutions, key=lambda s: s.objective)
            
            if best_solution is None or iter_best.objective < best_solution.objective:
                best_solution = iter_best
            
            convergence.append(best_solution.objective)
            
            fitness = [1.0 / (s.objective + 1) for s in solutions]
            new_population = []
            
            while len(new_population) < self.pop_size:
                candidates = random.sample(range(self.pop_size), 3)
                winner = max(candidates, key=lambda i: fitness[i])
                new_population.append(population[winner][:])
            
            for i in range(0, self.pop_size - 1, 2):
                new_population[i], new_population[i+1] = self.crossover(
                    new_population[i], new_population[i+1])
            
            new_population = [self.mutate(chrom) for chrom in new_population]
            
            best_idx = min(range(len(solutions)), key=lambda i: solutions[i].objective)
            new_population[0] = population[best_idx][:]
            
            population = new_population
        
        return best_solution, convergence


# =============================================================================
# PSO ALGORITHM
# =============================================================================

class PSO:
    def __init__(self, instance: Instance, num_particles: int = 30, max_iter: int = 100,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.instance = instance
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = instance.total_ops
    
    def decode_position(self, position: np.ndarray) -> List[Tuple[int, int, int]]:
        assignment = []
        idx = 0
        for job in self.instance.jobs:
            for op in job.operations:
                machines = list(op.alternatives.keys())
                machine_idx = int(abs(position[idx]) * 1000) % len(machines)
                assignment.append((job.job_id, op.op_id, machines[machine_idx]))
                idx += 1
        
        indices = np.argsort(position)
        sorted_assignment = [assignment[i] for i in indices]
        return repair_precedence(self.instance, sorted_assignment)
    
    def solve(self) -> Tuple[Solution, List[float]]:
        positions = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        
        pbest_pos = positions.copy()
        pbest_obj = np.full(self.num_particles, np.inf)
        
        gbest_pos = None
        gbest_obj = np.inf
        best_solution = None
        convergence = []
        
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                assignment = self.decode_position(positions[i])
                solution = decode(self.instance, assignment)
                
                if solution.objective < pbest_obj[i]:
                    pbest_obj[i] = solution.objective
                    pbest_pos[i] = positions[i].copy()
                
                if solution.objective < gbest_obj:
                    gbest_obj = solution.objective
                    gbest_pos = positions[i].copy()
                    best_solution = solution
            
            convergence.append(gbest_obj)
            
            r1, r2 = np.random.random(2)
            for i in range(self.num_particles):
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (pbest_pos[i] - positions[i]) +
                               self.c2 * r2 * (gbest_pos - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], -2, 2)
        
        return best_solution, convergence


# =============================================================================
# SA ALGORITHM
# =============================================================================

class SA:
    def __init__(self, instance: Instance, max_iter: int = 1000,
                 temp_init: float = 1000, temp_min: float = 1, cooling: float = 0.995):
        self.instance = instance
        self.max_iter = max_iter
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.cooling = cooling
    
    def random_solution(self) -> List[Tuple[int, int, int]]:
        assignment = []
        for job in self.instance.jobs:
            for op in job.operations:
                machine = random.choice(list(op.alternatives.keys()))
                assignment.append((job.job_id, op.op_id, machine))
        random.shuffle(assignment)
        return repair_precedence(self.instance, assignment)
    
    def neighbor(self, current: List) -> List:
        neighbor = current[:]
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            idx = random.randint(0, len(neighbor) - 1)
            job_id, op_id, _ = neighbor[idx]
            op = self.instance.jobs[job_id].operations[op_id]
            new_machine = random.choice(list(op.alternatives.keys()))
            neighbor[idx] = (job_id, op_id, new_machine)
        elif move_type == 1 and len(neighbor) >= 2:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor = repair_precedence(self.instance, neighbor)
        elif len(neighbor) >= 2:
            i = random.randint(0, len(neighbor) - 1)
            j = random.randint(0, len(neighbor) - 1)
            item = neighbor.pop(i)
            neighbor.insert(j, item)
            neighbor = repair_precedence(self.instance, neighbor)
        
        return neighbor
    
    def solve(self) -> Tuple[Solution, List[float]]:
        current = self.random_solution()
        current_sol = decode(self.instance, current)
        
        best = current[:]
        best_sol = current_sol
        
        temp = self.temp_init
        convergence = []
        
        for _ in range(self.max_iter):
            neighbor = self.neighbor(current)
            neighbor_sol = decode(self.instance, neighbor)
            
            delta = neighbor_sol.objective - current_sol.objective
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                current = neighbor
                current_sol = neighbor_sol
                
                if current_sol.objective < best_sol.objective:
                    best = current[:]
                    best_sol = current_sol
            
            temp = max(self.temp_min, temp * self.cooling)
            convergence.append(best_sol.objective)
        
        return best_sol, convergence


# =============================================================================
# MILP SOLVER
# =============================================================================

def solve_milp(instance: Instance, time_limit: int = 300) -> Optional[Tuple[Solution, float]]:
    if not PULP_AVAILABLE:
        return None
    
    if instance.total_ops > 25:
        return None
    
    M_big = 100000
    model = pulp.LpProblem("GF_FJSP_PM", pulp.LpMinimize)
    
    x, S, C = {}, {}, {}
    
    for job in instance.jobs:
        for op in job.operations:
            for m in op.alternatives.keys():
                x[job.job_id, op.op_id, m] = pulp.LpVariable(
                    f"x_{job.job_id}_{op.op_id}_{m}", cat='Binary')
            S[job.job_id, op.op_id] = pulp.LpVariable(f"S_{job.job_id}_{op.op_id}", lowBound=0)
            C[job.job_id, op.op_id] = pulp.LpVariable(f"C_{job.job_id}_{op.op_id}", lowBound=0)
    
    C_max = pulp.LpVariable("C_max", lowBound=0)
    
    y = {}
    all_ops = [(j.job_id, o.op_id) for j in instance.jobs for o in j.operations]
    for i1, j1 in all_ops:
        for i2, j2 in all_ops:
            if (i1, j1) < (i2, j2):
                op1 = instance.jobs[i1].operations[j1]
                op2 = instance.jobs[i2].operations[j2]
                common = set(op1.alternatives.keys()) & set(op2.alternatives.keys())
                for k in common:
                    y[i1, j1, i2, j2, k] = pulp.LpVariable(f"y_{i1}_{j1}_{i2}_{j2}_{k}", cat='Binary')
    
    model += C_max
    
    for job in instance.jobs:
        for op in job.operations:
            model += pulp.lpSum(x[job.job_id, op.op_id, m] for m in op.alternatives.keys()) == 1
    
    for job in instance.jobs:
        for op in job.operations:
            for m, ft in op.alternatives.items():
                model += C[job.job_id, op.op_id] >= S[job.job_id, op.op_id] + ft.gmir() - M_big * (1 - x[job.job_id, op.op_id, m])
    
    for job in instance.jobs:
        for idx in range(1, len(job.operations)):
            prev_op = job.operations[idx - 1]
            curr_op = job.operations[idx]
            model += S[job.job_id, curr_op.op_id] >= C[job.job_id, prev_op.op_id]
    
    for i1, j1 in all_ops:
        for i2, j2 in all_ops:
            if (i1, j1) < (i2, j2):
                op1 = instance.jobs[i1].operations[j1]
                op2 = instance.jobs[i2].operations[j2]
                common = set(op1.alternatives.keys()) & set(op2.alternatives.keys())
                for k in common:
                    model += S[i2, j2] >= C[i1, j1] - M_big * (3 - x[i1, j1, k] - x[i2, j2, k] - y[i1, j1, i2, j2, k])
                    model += S[i1, j1] >= C[i2, j2] - M_big * (2 - x[i1, j1, k] - x[i2, j2, k] + y[i1, j1, i2, j2, k])
    
    for job in instance.jobs:
        last_op = job.operations[-1]
        model += C_max >= C[job.job_id, last_op.op_id]
    
    start_time = time.time()
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
    status = model.solve(solver)
    solve_time = time.time() - start_time
    
    if pulp.LpStatus[status] in ['Optimal', 'Feasible']:
        assignment = []
        for job in instance.jobs:
            for op in job.operations:
                for m in op.alternatives.keys():
                    if pulp.value(x[job.job_id, op.op_id, m]) > 0.5:
                        assignment.append((job.job_id, op.op_id, m))
                        break
        
        assignment.sort(key=lambda t: pulp.value(S[t[0], t[1]]))
        solution = decode(instance, assignment)
        solution.makespan = pulp.value(C_max)
        return solution, solve_time
    
    return None


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_algorithm(instance: Instance, algorithm: str, max_iter: int = 100) -> Tuple[Solution, List[float], float]:
    start_time = time.time()
    
    if algorithm == 'ACO':
        solver = ACO(instance, max_iter=max_iter)
    elif algorithm == 'GA':
        solver = GA(instance, max_iter=max_iter)
    elif algorithm == 'PSO':
        solver = PSO(instance, max_iter=max_iter)
    elif algorithm == 'SA':
        solver = SA(instance, max_iter=max_iter * 10)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    solution, convergence = solver.solve()
    elapsed = time.time() - start_time
    
    return solution, convergence, elapsed


def run_experiments(instances: List[Instance], algorithms: List[str],
                    max_iter: int = 100, runs: int = 10,
                    include_milp: bool = True) -> Dict:
    """Run all experiments"""
    
    results = {
        'summary': [],
        'convergence': {},
        'all_runs': {}
    }
    
    for instance in instances:
        print(f"\n{'='*70}")
        print(f"Instance: {instance.name} ({instance.num_jobs}x{instance.num_machines}, {instance.total_ops} ops)")
        print(f"{'='*70}")
        
        # MILP (if applicable)
        milp_obj = None
        if include_milp and instance.total_ops <= 25:
            print("  MILP...", end=" ", flush=True)
            milp_result = solve_milp(instance, time_limit=300)
            if milp_result:
                milp_sol, milp_time = milp_result
                milp_obj = milp_sol.objective
                print(f"Optimal={milp_obj:.2f}, Time={milp_time:.2f}s")
                results['summary'].append({
                    'instance': instance.name,
                    'algorithm': 'MILP',
                    'best_obj': milp_obj,
                    'best_makespan': milp_sol.makespan,
                    'avg_obj': milp_obj,
                    'std_obj': 0,
                    'avg_time': milp_time,
                    'gap': 0
                })
            else:
                print("Failed/Timeout")
        
        # Metaheuristics
        for alg in algorithms:
            print(f"  {alg}...", end=" ", flush=True)
            
            all_objectives = []
            all_makespans = []
            all_times = []
            all_convergence = []
            best_solution = None
            
            for run in range(runs):
                solution, convergence, elapsed = run_algorithm(instance, alg, max_iter)
                
                all_objectives.append(solution.objective)
                all_makespans.append(solution.makespan)
                all_times.append(elapsed)
                all_convergence.append(convergence)
                
                if best_solution is None or solution.objective < best_solution.objective:
                    best_solution = solution
            
            best_obj = min(all_objectives)
            avg_obj = np.mean(all_objectives)
            std_obj = np.std(all_objectives)
            avg_time = np.mean(all_times)
            
            gap = ((best_obj - milp_obj) / milp_obj * 100) if milp_obj else 0
            
            print(f"Best={best_obj:.2f}, Avg={avg_obj:.2f}±{std_obj:.2f}, Time={avg_time:.2f}s", end="")
            if milp_obj:
                print(f", Gap={gap:.2f}%")
            else:
                print()
            
            results['summary'].append({
                'instance': instance.name,
                'algorithm': alg,
                'best_obj': best_obj,
                'best_makespan': min(all_makespans),
                'avg_obj': avg_obj,
                'std_obj': std_obj,
                'avg_time': avg_time,
                'gap': gap
            })
            
            # Store convergence (average across runs)
            avg_convergence = np.mean(all_convergence, axis=0).tolist()
            results['convergence'][(instance.name, alg)] = avg_convergence
            
            # Store all runs
            results['all_runs'][(instance.name, alg)] = all_objectives
    
    return results


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def statistical_analysis(results: Dict, algorithms: List[str]) -> Dict:
    """Perform statistical tests"""
    
    if not SCIPY_AVAILABLE:
        print("scipy not available - skipping statistical tests")
        return {}
    
    stats_results = {}
    
    # Group by instance
    instances = list(set(r['instance'] for r in results['summary'] if r['algorithm'] in algorithms))
    
    for instance in instances:
        instance_data = {}
        for alg in algorithms:
            key = (instance, alg)
            if key in results['all_runs']:
                instance_data[alg] = results['all_runs'][key]
        
        if len(instance_data) >= 2:
            # Friedman test (if more than 2 algorithms)
            if len(instance_data) >= 3:
                data_arrays = [instance_data[alg] for alg in algorithms if alg in instance_data]
                try:
                    stat, p_value = stats.friedmanchisquare(*data_arrays)
                    stats_results[f'{instance}_friedman'] = {'statistic': stat, 'p_value': p_value}
                except:
                    pass
            
            # Pairwise Wilcoxon tests
            alg_list = [alg for alg in algorithms if alg in instance_data]
            for i, alg1 in enumerate(alg_list):
                for alg2 in alg_list[i+1:]:
                    try:
                        stat, p_value = stats.wilcoxon(instance_data[alg1], instance_data[alg2])
                        stats_results[f'{instance}_{alg1}_vs_{alg2}'] = {
                            'statistic': stat, 'p_value': p_value
                        }
                    except:
                        pass
    
    return stats_results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_convergence(results: Dict, instances: List[str], algorithms: List[str], output_dir: str):
    """Generate convergence plots"""
    
    if not PLOT_AVAILABLE:
        print("matplotlib not available - skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {'ACO': 'blue', 'GA': 'green', 'PSO': 'red', 'SA': 'orange'}
    
    for instance in instances:
        plt.figure(figsize=(10, 6))
        
        for alg in algorithms:
            key = (instance, alg)
            if key in results['convergence']:
                conv = results['convergence'][key]
                plt.plot(conv, label=alg, color=colors.get(alg, 'black'), linewidth=2)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Objective Value', fontsize=12)
        plt.title(f'Convergence - {instance}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'convergence_{instance}.png'), dpi=150)
        plt.close()
    
    # Combined plot for all instances
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances[:4]):
        ax = axes[idx]
        for alg in algorithms:
            key = (instance, alg)
            if key in results['convergence']:
                conv = results['convergence'][key]
                ax.plot(conv, label=alg, color=colors.get(alg, 'black'), linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.set_title(instance)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_combined.png'), dpi=150)
    plt.close()
    
    print(f"Convergence plots saved to {output_dir}/")


def plot_comparison_bar(results: Dict, output_dir: str):
    """Generate bar chart comparison"""
    
    if not PLOT_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    summary = results['summary']
    instances = list(dict.fromkeys(r['instance'] for r in summary))
    algorithms = list(dict.fromkeys(r['algorithm'] for r in summary if r['algorithm'] != 'MILP'))
    
    # Prepare data
    data = {alg: [] for alg in algorithms}
    for instance in instances:
        for alg in algorithms:
            for r in summary:
                if r['instance'] == instance and r['algorithm'] == alg:
                    data[alg].append(r['best_obj'])
                    break
    
    # Plot
    x = np.arange(len(instances))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'ACO': 'blue', 'GA': 'green', 'PSO': 'red', 'SA': 'orange'}
    
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms)/2 + 0.5) * width
        ax.bar(x + offset, data[alg], width, label=alg, color=colors.get(alg, 'gray'))
    
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Best Objective', fontsize=12)
    ax.set_title('Algorithm Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar.png'), dpi=150)
    plt.close()


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(results: Dict, output_dir: str):
    """Export results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV summary
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['instance', 'algorithm', 'best_obj', 'best_makespan', 
                     'avg_obj', 'std_obj', 'avg_time', 'gap']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results['summary']:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v 
                           for k, v in r.items() if k in fieldnames})
    
    # JSON full results
    json_path = os.path.join(output_dir, 'results_full.json')
    export_data = {
        'summary': results['summary'],
        'convergence': {f"{k[0]}_{k[1]}": v for k, v in results['convergence'].items()},
        'all_runs': {f"{k[0]}_{k[1]}": v for k, v in results['all_runs'].items()}
    }
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    # LaTeX table
    latex_path = os.path.join(output_dir, 'results_table.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of metaheuristic algorithms}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Instance} & \\textbf{Algorithm} & \\textbf{Best} & \\textbf{Avg} & "
               "\\textbf{Std} & \\textbf{Time(s)} & \\textbf{Gap(\\%)} \\\\\n")
        f.write("\\midrule\n")
        
        current_instance = None
        for r in results['summary']:
            instance_str = r['instance'] if r['instance'] != current_instance else ""
            current_instance = r['instance']
            
            f.write(f"{instance_str} & {r['algorithm']} & {r['best_obj']:.2f} & "
                   f"{r['avg_obj']:.2f} & {r['std_obj']:.2f} & {r['avg_time']:.2f} & "
                   f"{r['gap']:.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{table}\n")
    
    print(f"\nResults exported to {output_dir}/")
    print(f"  - results_summary.csv")
    print(f"  - results_full.json")
    print(f"  - results_table.tex")


def export_instances(instances: List[Instance], output_dir: str):
    """Export instances to JSON and TXT"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for instance in instances:
        # JSON
        json_path = os.path.join(output_dir, f"{instance.name}.json")
        data = {
            'name': instance.name,
            'num_jobs': instance.num_jobs,
            'num_machines': instance.num_machines,
            'total_operations': instance.total_ops,
            'jobs': [
                {
                    'job_id': job.job_id,
                    'operations': [
                        {
                            'op_id': op.op_id,
                            'alternatives': {
                                str(m): {'L': ft.L, 'M': ft.M, 'U': ft.U, 'GMIR': round(ft.gmir(), 2)}
                                for m, ft in op.alternatives.items()
                            }
                        }
                        for op in job.operations
                    ]
                }
                for job in instance.jobs
            ],
            'machines': [
                {
                    'machine_id': m.machine_id,
                    'power_processing': m.power_processing,
                    'power_idle': m.power_idle,
                    'pm_duration': m.pm_duration,
                    'pm_window_start': m.pm_window_start,
                    'pm_window_end': m.pm_window_end
                }
                for m in instance.machines
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # TXT
        txt_path = os.path.join(output_dir, f"{instance.name}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"# {instance.name}\n")
            f.write(f"{instance.num_jobs} {instance.num_machines}\n")
            for job in instance.jobs:
                line = f"{len(job.operations)}"
                for op in job.operations:
                    line += f" {len(op.alternatives)}"
                    for m, ft in op.alternatives.items():
                        line += f" {m+1} {int(ft.M)}"
                f.write(line + "\n")
            f.write("# Machine: P_proc P_idle PM_dur PM_start PM_end\n")
            for m in instance.machines:
                f.write(f"{m.power_processing} {m.power_idle} {m.pm_duration} "
                       f"{m.pm_window_start} {m.pm_window_end}\n")
    
    print(f"Instances exported to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  GF-FJSP-PM COMPLETE EXPERIMENT RUNNER")
    print("="*70)
    
    # Configuration
    RUNS = 10
    MAX_ITER = 100
    algorithms = ['ACO', 'GA', 'PSO', 'SA']
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("instances", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # =========================================================================
    # PART 1: Small instances (MILP comparison)
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: SMALL INSTANCES (with MILP optimal)")
    print("="*70)
    
    small_instances = [
        generate_instance("Small_3x2x2", 3, 2, 2, 0.8, 0.2, seed=42),
        generate_instance("Small_4x3x2", 4, 3, 2, 0.7, 0.2, seed=123),
        generate_instance("Small_5x3x3", 5, 3, 3, 0.6, 0.2, seed=456),
        generate_instance("Small_6x4x3", 6, 4, 3, 0.5, 0.2, seed=789),
    ]
    
    export_instances(small_instances, "instances/small")
    
    small_results = run_experiments(small_instances, algorithms, 
                                    max_iter=MAX_ITER, runs=RUNS, include_milp=True)
    
    export_results(small_results, "results/small")
    plot_convergence(small_results, [i.name for i in small_instances], algorithms, "figures/small")
    
    # =========================================================================
    # PART 2: Hard instances (metaheuristic comparison only)
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: HARD INSTANCES (metaheuristic comparison)")
    print("="*70)
    
    hard_instances = [
        generate_instance("Hard_10x5x4", 10, 5, 4, 0.5, 0.2, seed=1001),
        generate_instance("Hard_10x6x5", 10, 6, 5, 0.5, 0.2, seed=1002),
        generate_instance("Hard_15x6x4", 15, 6, 4, 0.4, 0.2, seed=1003),
        generate_instance("Hard_15x8x5", 15, 8, 5, 0.4, 0.2, seed=1004),
        generate_instance("Hard_20x8x5", 20, 8, 5, 0.4, 0.2, seed=1005),
        generate_instance("Hard_20x10x5", 20, 10, 5, 0.3, 0.2, seed=1006),
        generate_instance("Hard_30x10x5", 30, 10, 5, 0.3, 0.2, seed=1007),
        generate_instance("Hard_30x10x6", 30, 10, 6, 0.3, 0.2, seed=1008),
    ]
    
    export_instances(hard_instances, "instances/hard")
    
    hard_results = run_experiments(hard_instances, algorithms,
                                   max_iter=MAX_ITER*2, runs=RUNS, include_milp=False)
    
    export_results(hard_results, "results/hard")
    plot_convergence(hard_results, [i.name for i in hard_instances[:4]], algorithms, "figures/hard")
    plot_comparison_bar(hard_results, "figures")
    
    # =========================================================================
    # PART 3: Statistical analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PART 3: STATISTICAL ANALYSIS")
    print("="*70)
    
    stats_small = statistical_analysis(small_results, algorithms)
    stats_hard = statistical_analysis(hard_results, algorithms)
    
    if stats_small or stats_hard:
        with open("results/statistical_tests.json", 'w') as f:
            json.dump({'small': stats_small, 'hard': stats_hard}, f, indent=2)
        print("Statistical tests saved to results/statistical_tests.json")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    print("\nSmall Instances Summary:")
    print("-" * 80)
    print(f"{'Instance':<15} {'MILP':<12} {'ACO':<12} {'GA':<12} {'PSO':<12} {'SA':<12}")
    print("-" * 80)
    
    for inst in small_instances:
        row = f"{inst.name:<15}"
        for alg in ['MILP'] + algorithms:
            for r in small_results['summary']:
                if r['instance'] == inst.name and r['algorithm'] == alg:
                    row += f"{r['best_obj']:<12.2f}"
                    break
        print(row)
    
    print("\nHard Instances Summary:")
    print("-" * 70)
    print(f"{'Instance':<20} {'ACO':<12} {'GA':<12} {'PSO':<12} {'SA':<12}")
    print("-" * 70)
    
    for inst in hard_instances:
        row = f"{inst.name:<20}"
        for alg in algorithms:
            for r in hard_results['summary']:
                if r['instance'] == inst.name and r['algorithm'] == alg:
                    row += f"{r['best_obj']:<12.2f}"
                    break
        print(row)
    
    print("\n" + "="*70)
    print("Files generated:")
    print("  instances/small/*.json, *.txt")
    print("  instances/hard/*.json, *.txt")
    print("  results/small/results_summary.csv, results_table.tex")
    print("  results/hard/results_summary.csv, results_table.tex")
    print("  figures/small/convergence_*.png")
    print("  figures/hard/convergence_*.png")
    print("  figures/comparison_bar.png")
    print("="*70)
    
    return small_results, hard_results


if __name__ == "__main__":
    small_results, hard_results = main()
