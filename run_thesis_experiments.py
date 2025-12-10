#!/usr/bin/env python3
"""
Thesis Experiment Runner
========================

Runs experiments for thesis:
1. MILP exact solutions for small instances
2. ACO, GA, PSO, SA comparison for medium instances
3. Results collection and table generation

Author: Master's Thesis
"""

import json
import os
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import copy

# Try to import MILP solver
try:
    import pulp
    MILP_AVAILABLE = True
except ImportError:
    MILP_AVAILABLE = False
    print("Warning: PuLP not installed. Install with: pip install pulp")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FuzzyTime:
    L: int
    M: int  
    U: int
    
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
    fuzziness: float
    alpha: float
    beta: float


def load_instance(filepath: str) -> Instance:
    """Load instance from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    jobs = []
    for job_data in data['jobs']:
        operations = []
        for op_data in job_data['operations']:
            op = Operation(job_id=job_data['job_id'], op_id=op_data['op_id'])
            for m_str, ft_data in op_data['alternatives'].items():
                op.alternatives[int(m_str)] = FuzzyTime(
                    L=ft_data['L'], M=ft_data['M'], U=ft_data['U']
                )
            operations.append(op)
        jobs.append(Job(job_id=job_data['job_id'], operations=operations))
    
    machines = []
    for m_data in data['machines']:
        machines.append(Machine(
            machine_id=m_data['machine_id'],
            power_processing=m_data['power_processing'],
            power_idle=m_data['power_idle'],
            pm_duration=m_data['pm_duration'],
            pm_window_start=m_data['pm_window_start'],
            pm_window_end=m_data['pm_window_end']
        ))
    
    return Instance(
        name=data['name'],
        num_jobs=data['num_jobs'],
        num_machines=data['num_machines'],
        jobs=jobs,
        machines=machines,
        fuzziness=data['fuzziness'],
        alpha=data['alpha'],
        beta=data['beta']
    )


# =============================================================================
# MILP EXACT SOLVER
# =============================================================================

def solve_milp(instance: Instance, time_limit: int = 300, verbose: bool = False) -> Dict:
    """Solve instance with MILP exact algorithm."""
    
    if not MILP_AVAILABLE:
        return {'status': 'MILP not available', 'makespan': None, 'objective': None}
    
    M_big = 100000
    
    model = pulp.LpProblem("GF_FJSP_PM", pulp.LpMinimize)
    
    # Index sets
    all_ops = [(j.job_id, op.op_id) for j in instance.jobs for op in j.operations]
    
    # Decision variables
    x = {}  # x[i,j,k] = 1 if op j of job i assigned to machine k
    for j in instance.jobs:
        for op in j.operations:
            for k in op.alternatives.keys():
                x[j.job_id, op.op_id, k] = pulp.LpVariable(
                    f"x_{j.job_id}_{op.op_id}_{k}", cat='Binary')
    
    S = {}  # Start times
    C = {}  # Completion times
    for j in instance.jobs:
        for op in j.operations:
            S[j.job_id, op.op_id] = pulp.LpVariable(f"S_{j.job_id}_{op.op_id}", lowBound=0)
            C[j.job_id, op.op_id] = pulp.LpVariable(f"C_{j.job_id}_{op.op_id}", lowBound=0)
    
    C_max = pulp.LpVariable("C_max", lowBound=0)
    
    # Sequencing variables
    y = {}
    for i1, j1 in all_ops:
        for i2, j2 in all_ops:
            if (i1, j1) < (i2, j2):
                op1 = instance.jobs[i1].operations[j1]
                op2 = instance.jobs[i2].operations[j2]
                common = set(op1.alternatives.keys()) & set(op2.alternatives.keys())
                for k in common:
                    y[i1, j1, i2, j2, k] = pulp.LpVariable(f"y_{i1}_{j1}_{i2}_{j2}_{k}", cat='Binary')
    
    # PM start times
    SM = {m.machine_id: pulp.LpVariable(f"SM_{m.machine_id}", lowBound=0) 
          for m in instance.machines}
    
    # Energy variables
    E_proc = pulp.LpVariable("E_proc", lowBound=0)
    E_idle = pulp.LpVariable("E_idle", lowBound=0)
    
    # Objective: minimize α*C_max + β*E_total (normalized)
    # We'll use a simple weighted sum
    model += instance.alpha * C_max + instance.beta * (E_proc + E_idle) * 0.01
    
    # Constraints
    
    # 1. Assignment constraint
    for j in instance.jobs:
        for op in j.operations:
            model += pulp.lpSum(x[j.job_id, op.op_id, k] for k in op.alternatives.keys()) == 1
    
    # 2. Completion time
    for j in instance.jobs:
        for op in j.operations:
            for k, ft in op.alternatives.items():
                p = ft.gmir()
                model += C[j.job_id, op.op_id] >= S[j.job_id, op.op_id] + p - M_big*(1-x[j.job_id, op.op_id, k])
    
    # 3. Precedence
    for j in instance.jobs:
        for idx in range(1, len(j.operations)):
            prev_op = j.operations[idx-1]
            curr_op = j.operations[idx]
            model += S[j.job_id, curr_op.op_id] >= C[j.job_id, prev_op.op_id]
    
    # 4. Machine capacity
    for i1, j1 in all_ops:
        for i2, j2 in all_ops:
            if (i1, j1) < (i2, j2):
                op1 = instance.jobs[i1].operations[j1]
                op2 = instance.jobs[i2].operations[j2]
                common = set(op1.alternatives.keys()) & set(op2.alternatives.keys())
                for k in common:
                    p1 = op1.alternatives[k].gmir()
                    model += S[i2, j2] >= C[i1, j1] - M_big*(3 - x[i1,j1,k] - x[i2,j2,k] - y[i1,j1,i2,j2,k])
                    model += S[i1, j1] >= C[i2, j2] - M_big*(2 - x[i1,j1,k] - x[i2,j2,k] + y[i1,j1,i2,j2,k])
    
    # 5. Makespan
    for j in instance.jobs:
        last_op = j.operations[-1]
        model += C_max >= C[j.job_id, last_op.op_id]
    
    # 6. PM window
    for m in instance.machines:
        model += SM[m.machine_id] >= m.pm_window_start
        model += SM[m.machine_id] <= m.pm_window_end
    
    # 7. Energy calculation (simplified)
    model += E_proc == pulp.lpSum(
        x[j.job_id, op.op_id, k] * op.alternatives[k].gmir() * instance.machines[k].power_processing
        for j in instance.jobs for op in j.operations for k in op.alternatives.keys()
    )
    
    # Solve
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    start_time = time.time()
    status = model.solve(solver)
    solve_time = time.time() - start_time
    
    if pulp.LpStatus[status] in ['Optimal', 'Feasible']:
        makespan = pulp.value(C_max)
        objective = pulp.value(model.objective)
        energy = pulp.value(E_proc) + pulp.value(E_idle) if pulp.value(E_idle) else pulp.value(E_proc)
        
        # Extract schedule
        schedule = []
        for j in instance.jobs:
            for op in j.operations:
                for k in op.alternatives.keys():
                    if pulp.value(x[j.job_id, op.op_id, k]) > 0.5:
                        schedule.append({
                            'job': j.job_id,
                            'op': op.op_id,
                            'machine': k,
                            'start': pulp.value(S[j.job_id, op.op_id]),
                            'end': pulp.value(C[j.job_id, op.op_id])
                        })
        
        return {
            'status': pulp.LpStatus[status],
            'makespan': makespan,
            'objective': objective,
            'energy': energy,
            'solve_time': solve_time,
            'schedule': schedule
        }
    else:
        return {
            'status': pulp.LpStatus[status],
            'makespan': None,
            'objective': None,
            'solve_time': solve_time
        }


# =============================================================================
# METAHEURISTIC: GENETIC ALGORITHM (GA)
# =============================================================================

class GeneticAlgorithm:
    """Genetic Algorithm for GF-FJSP-PM."""
    
    def __init__(self, instance: Instance, pop_size: int = 50, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.instance = instance
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def create_individual(self) -> List[Tuple[int, int, int]]:
        """Create random individual: list of (job, op, machine)."""
        individual = []
        for job in self.instance.jobs:
            for op in job.operations:
                machine = random.choice(list(op.alternatives.keys()))
                individual.append((job.job_id, op.op_id, machine))
        random.shuffle(individual)
        return individual
    
    def decode(self, individual: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        """Decode individual to schedule and compute fitness."""
        machine_time = [0.0] * self.instance.num_machines
        job_time = [0.0] * self.instance.num_jobs
        total_proc_energy = 0.0
        
        for job_id, op_id, machine in individual:
            op = self.instance.jobs[job_id].operations[op_id]
            proc_time = op.alternatives[machine].gmir()
            
            start = max(machine_time[machine], job_time[job_id])
            end = start + proc_time
            
            machine_time[machine] = end
            job_time[job_id] = end
            
            total_proc_energy += proc_time * self.instance.machines[machine].power_processing
        
        makespan = max(machine_time)
        energy = total_proc_energy
        
        objective = self.instance.alpha * makespan + self.instance.beta * energy * 0.01
        return makespan, objective
    
    def crossover(self, p1: List, p2: List) -> Tuple[List, List]:
        """Two-point crossover."""
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        
        size = len(p1)
        pt1, pt2 = sorted(random.sample(range(size), 2))
        
        c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        
        return c1, c2
    
    def mutate(self, individual: List) -> List:
        """Mutation: change machine assignment or swap operations."""
        ind = individual.copy()
        
        for i in range(len(ind)):
            if random.random() < self.mutation_rate:
                job_id, op_id, _ = ind[i]
                op = self.instance.jobs[job_id].operations[op_id]
                new_machine = random.choice(list(op.alternatives.keys()))
                ind[i] = (job_id, op_id, new_machine)
        
        # Swap mutation
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        
        return ind
    
    def repair(self, individual: List) -> List:
        """Repair individual to ensure valid precedence."""
        # Sort by (job, op) to maintain precedence
        ind = individual.copy()
        job_next_op = [0] * self.instance.num_jobs
        result = []
        remaining = set(range(len(ind)))
        
        while remaining:
            for idx in list(remaining):
                job_id, op_id, machine = ind[idx]
                if op_id == job_next_op[job_id]:
                    result.append((job_id, op_id, machine))
                    job_next_op[job_id] += 1
                    remaining.remove(idx)
                    break
        
        return result
    
    def solve(self, max_iterations: int = 100, verbose: bool = False) -> Dict:
        """Run GA."""
        start_time = time.time()
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.pop_size)]
        population = [self.repair(ind) for ind in population]
        
        best_makespan = float('inf')
        best_objective = float('inf')
        best_individual = None
        
        for iteration in range(max_iterations):
            # Evaluate
            fitness = []
            for ind in population:
                makespan, obj = self.decode(ind)
                fitness.append((makespan, obj, ind))
                
                if obj < best_objective:
                    best_objective = obj
                    best_makespan = makespan
                    best_individual = ind.copy()
            
            # Selection (tournament)
            new_population = []
            for _ in range(self.pop_size):
                t1, t2 = random.sample(fitness, 2)
                winner = t1 if t1[1] < t2[1] else t2
                new_population.append(winner[2])
            
            # Crossover and mutation
            offspring = []
            for i in range(0, self.pop_size - 1, 2):
                c1, c2 = self.crossover(new_population[i], new_population[i+1])
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                offspring.extend([self.repair(c1), self.repair(c2)])
            
            # Elitism
            population = offspring[:self.pop_size-1] + [best_individual]
            
            if verbose and iteration % 20 == 0:
                print(f"  GA Iter {iteration}: Best Obj = {best_objective:.2f}")
        
        solve_time = time.time() - start_time
        
        return {
            'status': 'Completed',
            'makespan': best_makespan,
            'objective': best_objective,
            'solve_time': solve_time,
            'algorithm': 'GA'
        }


# =============================================================================
# METAHEURISTIC: PARTICLE SWARM OPTIMIZATION (PSO)
# =============================================================================

class ParticleSwarmOptimization:
    """PSO for GF-FJSP-PM."""
    
    def __init__(self, instance: Instance, swarm_size: int = 30,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.instance = instance
        self.swarm_size = swarm_size
        self.w = w    # inertia
        self.c1 = c1  # cognitive
        self.c2 = c2  # social
    
    def create_particle(self) -> np.ndarray:
        """Create particle: continuous position mapped to discrete solution."""
        total_ops = sum(len(j.operations) for j in self.instance.jobs)
        return np.random.rand(total_ops * 2)  # position + machine selection
    
    def decode(self, position: np.ndarray) -> Tuple[float, float]:
        """Decode particle position to schedule."""
        total_ops = sum(len(j.operations) for j in self.instance.jobs)
        
        # Build operation list with priorities
        ops = []
        idx = 0
        for job in self.instance.jobs:
            for op in job.operations:
                priority = position[idx]
                machine_prob = position[total_ops + idx]
                machines = list(op.alternatives.keys())
                machine = machines[int(machine_prob * len(machines)) % len(machines)]
                ops.append((priority, job.job_id, op.op_id, machine))
                idx += 1
        
        # Sort by priority while respecting precedence
        job_next_op = [0] * self.instance.num_jobs
        schedule = []
        ops_sorted = sorted(ops, key=lambda x: x[0])
        remaining = list(ops_sorted)
        
        while remaining:
            for item in remaining:
                _, job_id, op_id, machine = item
                if op_id == job_next_op[job_id]:
                    schedule.append((job_id, op_id, machine))
                    job_next_op[job_id] += 1
                    remaining.remove(item)
                    break
        
        # Calculate fitness
        machine_time = [0.0] * self.instance.num_machines
        job_time = [0.0] * self.instance.num_jobs
        total_energy = 0.0
        
        for job_id, op_id, machine in schedule:
            op = self.instance.jobs[job_id].operations[op_id]
            proc_time = op.alternatives[machine].gmir()
            
            start = max(machine_time[machine], job_time[job_id])
            end = start + proc_time
            
            machine_time[machine] = end
            job_time[job_id] = end
            total_energy += proc_time * self.instance.machines[machine].power_processing
        
        makespan = max(machine_time)
        objective = self.instance.alpha * makespan + self.instance.beta * total_energy * 0.01
        
        return makespan, objective
    
    def solve(self, max_iterations: int = 100, verbose: bool = False) -> Dict:
        """Run PSO."""
        start_time = time.time()
        
        dim = sum(len(j.operations) for j in self.instance.jobs) * 2
        
        # Initialize swarm
        positions = [self.create_particle() for _ in range(self.swarm_size)]
        velocities = [np.random.rand(dim) * 0.1 - 0.05 for _ in range(self.swarm_size)]
        
        # Personal best
        p_best_pos = [p.copy() for p in positions]
        p_best_fit = [self.decode(p)[1] for p in positions]
        
        # Global best
        g_best_idx = np.argmin(p_best_fit)
        g_best_pos = p_best_pos[g_best_idx].copy()
        g_best_fit = p_best_fit[g_best_idx]
        g_best_makespan = self.decode(g_best_pos)[0]
        
        for iteration in range(max_iterations):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (p_best_pos[i] - positions[i]) +
                                self.c2 * r2 * (g_best_pos - positions[i]))
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                
                # Evaluate
                makespan, fitness = self.decode(positions[i])
                
                # Update personal best
                if fitness < p_best_fit[i]:
                    p_best_fit[i] = fitness
                    p_best_pos[i] = positions[i].copy()
                    
                    # Update global best
                    if fitness < g_best_fit:
                        g_best_fit = fitness
                        g_best_pos = positions[i].copy()
                        g_best_makespan = makespan
            
            if verbose and iteration % 20 == 0:
                print(f"  PSO Iter {iteration}: Best Obj = {g_best_fit:.2f}")
        
        solve_time = time.time() - start_time
        
        return {
            'status': 'Completed',
            'makespan': g_best_makespan,
            'objective': g_best_fit,
            'solve_time': solve_time,
            'algorithm': 'PSO'
        }


# =============================================================================
# METAHEURISTIC: SIMULATED ANNEALING (SA)
# =============================================================================

class SimulatedAnnealing:
    """Simulated Annealing for GF-FJSP-PM."""
    
    def __init__(self, instance: Instance, 
                 initial_temp: float = 1000, cooling_rate: float = 0.995):
        self.instance = instance
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def create_solution(self) -> List[Tuple[int, int, int]]:
        """Create initial solution."""
        solution = []
        for job in self.instance.jobs:
            for op in job.operations:
                machine = random.choice(list(op.alternatives.keys()))
                solution.append((job.job_id, op.op_id, machine))
        return solution
    
    def evaluate(self, solution: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        """Evaluate solution."""
        # Ensure precedence
        job_next_op = [0] * self.instance.num_jobs
        ordered = []
        remaining = list(solution)
        
        while remaining:
            for item in remaining:
                job_id, op_id, machine = item
                if op_id == job_next_op[job_id]:
                    ordered.append(item)
                    job_next_op[job_id] += 1
                    remaining.remove(item)
                    break
        
        machine_time = [0.0] * self.instance.num_machines
        job_time = [0.0] * self.instance.num_jobs
        total_energy = 0.0
        
        for job_id, op_id, machine in ordered:
            op = self.instance.jobs[job_id].operations[op_id]
            proc_time = op.alternatives[machine].gmir()
            
            start = max(machine_time[machine], job_time[job_id])
            end = start + proc_time
            
            machine_time[machine] = end
            job_time[job_id] = end
            total_energy += proc_time * self.instance.machines[machine].power_processing
        
        makespan = max(machine_time)
        objective = self.instance.alpha * makespan + self.instance.beta * total_energy * 0.01
        
        return makespan, objective
    
    def neighbor(self, solution: List) -> List:
        """Generate neighbor solution."""
        new_sol = solution.copy()
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Change machine
            idx = random.randint(0, len(new_sol) - 1)
            job_id, op_id, _ = new_sol[idx]
            op = self.instance.jobs[job_id].operations[op_id]
            new_machine = random.choice(list(op.alternatives.keys()))
            new_sol[idx] = (job_id, op_id, new_machine)
        else:
            # Swap operations
            i, j = random.sample(range(len(new_sol)), 2)
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
        
        return new_sol
    
    def solve(self, max_iterations: int = 1000, verbose: bool = False) -> Dict:
        """Run SA."""
        start_time = time.time()
        
        current = self.create_solution()
        current_makespan, current_obj = self.evaluate(current)
        
        best = current.copy()
        best_makespan = current_makespan
        best_obj = current_obj
        
        temp = self.initial_temp
        
        for iteration in range(max_iterations):
            neighbor = self.neighbor(current)
            neighbor_makespan, neighbor_obj = self.evaluate(neighbor)
            
            delta = neighbor_obj - current_obj
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current = neighbor
                current_obj = neighbor_obj
                current_makespan = neighbor_makespan
                
                if current_obj < best_obj:
                    best = current.copy()
                    best_obj = current_obj
                    best_makespan = current_makespan
            
            temp *= self.cooling_rate
            
            if verbose and iteration % 200 == 0:
                print(f"  SA Iter {iteration}: Best Obj = {best_obj:.2f}, Temp = {temp:.2f}")
        
        solve_time = time.time() - start_time
        
        return {
            'status': 'Completed',
            'makespan': best_makespan,
            'objective': best_obj,
            'solve_time': solve_time,
            'algorithm': 'SA'
        }


# =============================================================================
# ACO (Simplified version for comparison)
# =============================================================================

class AntColonyOptimization:
    """Simplified ACO for GF-FJSP-PM."""
    
    def __init__(self, instance: Instance, num_ants: int = 20,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1):
        self.instance = instance
        self.num_ants = num_ants
        self.alpha_aco = alpha
        self.beta_aco = beta
        self.rho = rho
        
        # Initialize pheromone
        self.pheromone = {}
        for job in instance.jobs:
            for op in job.operations:
                for m in op.alternatives.keys():
                    self.pheromone[(job.job_id, op.op_id, m)] = 1.0
    
    def construct_solution(self) -> List[Tuple[int, int, int]]:
        """Construct solution using pheromone."""
        solution = []
        
        for job in self.instance.jobs:
            for op in job.operations:
                # Calculate probabilities
                probs = []
                machines = list(op.alternatives.keys())
                
                for m in machines:
                    tau = self.pheromone[(job.job_id, op.op_id, m)]
                    eta = 1.0 / op.alternatives[m].gmir()  # Heuristic
                    probs.append((tau ** self.alpha_aco) * (eta ** self.beta_aco))
                
                total = sum(probs)
                probs = [p / total for p in probs]
                
                # Roulette wheel selection
                r = random.random()
                cumsum = 0
                selected = machines[0]
                for m, p in zip(machines, probs):
                    cumsum += p
                    if r <= cumsum:
                        selected = m
                        break
                
                solution.append((job.job_id, op.op_id, selected))
        
        return solution
    
    def evaluate(self, solution: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        """Evaluate solution."""
        machine_time = [0.0] * self.instance.num_machines
        job_time = [0.0] * self.instance.num_jobs
        total_energy = 0.0
        
        for job_id, op_id, machine in solution:
            op = self.instance.jobs[job_id].operations[op_id]
            proc_time = op.alternatives[machine].gmir()
            
            start = max(machine_time[machine], job_time[job_id])
            end = start + proc_time
            
            machine_time[machine] = end
            job_time[job_id] = end
            total_energy += proc_time * self.instance.machines[machine].power_processing
        
        makespan = max(machine_time)
        objective = self.instance.alpha * makespan + self.instance.beta * total_energy * 0.01
        
        return makespan, objective
    
    def update_pheromone(self, solutions: List, objectives: List):
        """Update pheromone trails."""
        # Evaporation
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)
        
        # Deposit
        best_idx = np.argmin(objectives)
        best_sol = solutions[best_idx]
        best_obj = objectives[best_idx]
        
        for job_id, op_id, machine in best_sol:
            self.pheromone[(job_id, op_id, machine)] += 1.0 / best_obj
    
    def solve(self, max_iterations: int = 100, verbose: bool = False) -> Dict:
        """Run ACO."""
        start_time = time.time()
        
        best_makespan = float('inf')
        best_obj = float('inf')
        
        for iteration in range(max_iterations):
            solutions = []
            objectives = []
            
            for _ in range(self.num_ants):
                sol = self.construct_solution()
                makespan, obj = self.evaluate(sol)
                solutions.append(sol)
                objectives.append(obj)
                
                if obj < best_obj:
                    best_obj = obj
                    best_makespan = makespan
            
            self.update_pheromone(solutions, objectives)
            
            if verbose and iteration % 20 == 0:
                print(f"  ACO Iter {iteration}: Best Obj = {best_obj:.2f}")
        
        solve_time = time.time() - start_time
        
        return {
            'status': 'Completed',
            'makespan': best_makespan,
            'objective': best_obj,
            'solve_time': solve_time,
            'algorithm': 'ACO'
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment_milp_comparison(instances_dir: str = "thesis_benchmarks/small"):
    """Run MILP vs Metaheuristics comparison on small instances."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: MILP vs ACO Comparison (Small Instances)")
    print("="*70)
    
    results = []
    
    for filename in sorted(os.listdir(instances_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(instances_dir, filename)
            instance = load_instance(filepath)
            
            print(f"\n--- {instance.name} ({instance.num_jobs}×{instance.num_machines}) ---")
            
            # MILP
            print("  Running MILP...")
            milp_result = solve_milp(instance, time_limit=300)
            print(f"    Status: {milp_result['status']}, Makespan: {milp_result['makespan']:.2f}, "
                  f"Time: {milp_result['solve_time']:.2f}s")
            
            # ACO
            print("  Running ACO...")
            aco = AntColonyOptimization(instance)
            aco_result = aco.solve(max_iterations=100)
            print(f"    Makespan: {aco_result['makespan']:.2f}, Time: {aco_result['solve_time']:.2f}s")
            
            # Calculate gap
            if milp_result['makespan']:
                gap = (aco_result['makespan'] - milp_result['makespan']) / milp_result['makespan'] * 100
            else:
                gap = None
            
            results.append({
                'instance': instance.name,
                'size': f"{instance.num_jobs}×{instance.num_machines}",
                'milp_makespan': milp_result['makespan'],
                'milp_time': milp_result['solve_time'],
                'aco_makespan': aco_result['makespan'],
                'aco_time': aco_result['solve_time'],
                'gap': gap
            })
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY: MILP vs ACO")
    print("="*70)
    print(f"{'Instance':<12} {'Size':<8} {'MILP C_max':<12} {'MILP Time':<10} "
          f"{'ACO C_max':<12} {'ACO Time':<10} {'Gap %':<8}")
    print("-"*70)
    
    for r in results:
        gap_str = f"{r['gap']:.2f}%" if r['gap'] is not None else "N/A"
        print(f"{r['instance']:<12} {r['size']:<8} {r['milp_makespan']:<12.2f} "
              f"{r['milp_time']:<10.2f} {r['aco_makespan']:<12.2f} "
              f"{r['aco_time']:<10.2f} {gap_str:<8}")
    
    return results


def run_experiment_metaheuristic_comparison(instances_dir: str = "thesis_benchmarks/medium",
                                            num_runs: int = 10):
    """Run metaheuristic comparison on medium instances."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Metaheuristic Comparison (Medium Instances)")
    print(f"Number of runs per algorithm: {num_runs}")
    print("="*70)
    
    results = []
    
    for filename in sorted(os.listdir(instances_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(instances_dir, filename)
            instance = load_instance(filepath)
            
            print(f"\n--- {instance.name} ({instance.num_jobs}×{instance.num_machines}) ---")
            
            instance_results = {'instance': instance.name, 
                               'size': f"{instance.num_jobs}×{instance.num_machines}"}
            
            for algo_name, algo_class in [('ACO', AntColonyOptimization),
                                          ('GA', GeneticAlgorithm),
                                          ('PSO', ParticleSwarmOptimization),
                                          ('SA', SimulatedAnnealing)]:
                
                makespans = []
                objectives = []
                times = []
                
                print(f"  Running {algo_name}...", end=" ")
                
                for run in range(num_runs):
                    random.seed(run * 100)
                    np.random.seed(run * 100)
                    
                    algo = algo_class(instance)
                    result = algo.solve(max_iterations=100)
                    
                    makespans.append(result['makespan'])
                    objectives.append(result['objective'])
                    times.append(result['solve_time'])
                
                instance_results[f'{algo_name}_best'] = min(makespans)
                instance_results[f'{algo_name}_avg'] = np.mean(makespans)
                instance_results[f'{algo_name}_std'] = np.std(makespans)
                instance_results[f'{algo_name}_time'] = np.mean(times)
                
                print(f"Best={min(makespans):.2f}, Avg={np.mean(makespans):.2f}, "
                      f"Std={np.std(makespans):.2f}")
            
            results.append(instance_results)
    
    # Print summary table
    print("\n" + "="*90)
    print("RESULTS SUMMARY: Metaheuristic Comparison")
    print("="*90)
    print(f"{'Instance':<12} {'Size':<8} | {'ACO Best':<10} {'GA Best':<10} "
          f"{'PSO Best':<10} {'SA Best':<10}")
    print("-"*90)
    
    for r in results:
        print(f"{r['instance']:<12} {r['size']:<8} | {r['ACO_best']:<10.2f} "
              f"{r['GA_best']:<10.2f} {r['PSO_best']:<10.2f} {r['SA_best']:<10.2f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check if instances exist
    if not os.path.exists("thesis_benchmarks"):
        print("Generating benchmark instances first...")
        os.system("python generate_thesis_benchmarks.py")
    
    # Run experiments
    print("\n" + "="*70)
    print("THESIS EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: MILP comparison
    milp_results = run_experiment_milp_comparison()
    
    # Experiment 2: Metaheuristic comparison  
    meta_results = run_experiment_metaheuristic_comparison(num_runs=5)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nResults can be used to create tables for thesis Chapter 5.")
