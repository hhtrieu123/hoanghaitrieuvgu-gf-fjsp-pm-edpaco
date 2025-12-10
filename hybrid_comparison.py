#!/usr/bin/env python3
"""
Hybrid Algorithm Comparison for GF-FJSP-PM
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats
import csv
import time

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

class EDP_ACO:
    def __init__(self, instance, num_ants=30, max_iter=100, alpha_start=1.0, alpha_end=4.0,
                 beta_start=4.0, beta_end=1.0, rho_min=0.1, rho_max=0.3, q0=0.5, 
                 stop_patience=50, seed=None):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha_start, self.alpha_end = alpha_start, alpha_end
        self.beta_start, self.beta_end = beta_start, beta_end
        self.rho_min, self.rho_max = rho_min, rho_max
        self.q0 = q0
        self.stop_patience = stop_patience

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

    def get_adaptive_params(self, iteration):
        progress = iteration / self.max_iter
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        rho = self.rho_min + (self.rho_max - self.rho_min) * progress
        return alpha, beta, rho

    def calculate_heuristic(self, op, machine):
        proc_time = gmir(op.processing_times[machine])
        energy = proc_time * self.instance.machines[machine].power_processing
        combined = self.instance.alpha * proc_time + self.instance.beta * (energy / 100)
        return 1.0 / max(combined, 0.001)

    def construct_solution(self, alpha, beta):
        schedule = []
        next_op = [0] * self.instance.num_jobs
        total_ops = len(self.instance.operations)

        while len(schedule) < total_ops:
            ready_ops = []
            for j in range(self.instance.num_jobs):
                if next_op[j] < self.instance.num_ops_per_job:
                    op_idx = j * self.instance.num_ops_per_job + next_op[j]
                    ready_ops.append(self.instance.operations[op_idx])
            if not ready_ops:
                break

            op = random.choice(ready_ops)

            if random.random() < self.q0:
                best_value = -1
                best_machine = op.eligible_machines[0]
                for m in op.eligible_machines:
                    key = (op.job_id, op.op_id, m)
                    tau = self.instance.alpha * self.tau_time[key] + self.instance.beta * self.tau_energy[key]
                    eta = self.calculate_heuristic(op, m)
                    value = (tau ** alpha) * (eta ** beta)
                    if value > best_value:
                        best_value = value
                        best_machine = m
                selected_machine = best_machine
            else:
                probabilities = []
                for m in op.eligible_machines:
                    key = (op.job_id, op.op_id, m)
                    tau = self.instance.alpha * self.tau_time[key] + self.instance.beta * self.tau_energy[key]
                    eta = self.calculate_heuristic(op, m)
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

    def local_search(self, schedule):
        best_schedule = schedule[:]
        _, _, best_obj = evaluate_schedule(best_schedule, self.instance)

        for _ in range(50):
            neighbor = best_schedule[:]
            move = random.choice(['swap', 'change_machine', 'insert'])

            if move == 'swap' and len(neighbor) >= 2:
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif move == 'change_machine':
                idx = random.randint(0, len(neighbor) - 1)
                j, o, m, pt = neighbor[idx]
                op = self.instance.operations[j * self.instance.num_ops_per_job + o]
                if len(op.eligible_machines) > 1:
                    new_m = random.choice([x for x in op.eligible_machines if x != m])
                    new_pt = gmir(op.processing_times[new_m])
                    neighbor[idx] = (j, o, new_m, new_pt)
            elif move == 'insert' and len(neighbor) >= 2:
                i = random.randint(0, len(neighbor) - 1)
                item = neighbor.pop(i)
                j = random.randint(0, len(neighbor))
                neighbor.insert(j, item)

            _, _, obj = evaluate_schedule(neighbor, self.instance)
            if obj < best_obj:
                best_schedule = neighbor
                best_obj = obj

        return best_schedule

    def solve(self):
        best_schedule = None
        best_makespan = float('inf')
        best_energy = float('inf')
        best_objective = float('inf')
        convergence = []
        no_improve = 0

        for iteration in range(self.max_iter):
            alpha, beta, rho = self.get_adaptive_params(iteration)
            iter_best_schedule = None
            iter_best_objective = float('inf')

            for _ in range(self.num_ants):
                schedule = self.construct_solution(alpha, beta)
                makespan, energy, objective = evaluate_schedule(schedule, self.instance)
                if objective < iter_best_objective:
                    iter_best_objective = objective
                    iter_best_schedule = schedule

            if iter_best_schedule:
                iter_best_schedule = self.local_search(iter_best_schedule)
                makespan, energy, objective = evaluate_schedule(iter_best_schedule, self.instance)
                if objective < best_objective:
                    best_objective = objective
                    best_schedule = iter_best_schedule
                    best_makespan = makespan
                    best_energy = energy
                    no_improve = 0
                else:
                    no_improve += 1

            convergence.append(best_objective)

            if no_improve >= self.stop_patience:
                convergence.extend([best_objective] * (self.max_iter - iteration - 1))
                break

            for key in self.tau_time:
                self.tau_time[key] *= (1 - rho)
                self.tau_energy[key] *= (1 - rho)

            if iter_best_schedule:
                iter_makespan, iter_energy, _ = evaluate_schedule(iter_best_schedule, self.instance)
                for (j, o, m, pt) in iter_best_schedule:
                    key = (j, o, m)
                    if key in self.tau_time:
                        self.tau_time[key] += 1.0 / max(iter_makespan, 1)
                        self.tau_energy[key] += 1.0 / max(iter_energy, 1)

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

class HACO_TS:
    def __init__(self, instance, num_ants=30, max_iter=100, alpha=1.0, beta=2.0, rho=0.1,
                 tabu_tenure=7, tabu_iter=20, seed=None):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tabu_tenure = tabu_tenure
        self.tabu_iter = tabu_iter

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pheromone = {}
        for op in instance.operations:
            for m in op.eligible_machines:
                self.pheromone[(op.job_id, op.op_id, m)] = 1.0

    def construct_solution(self):
        schedule = []
        next_op = [0] * self.instance.num_jobs
        total_ops = len(self.instance.operations)

        while len(schedule) < total_ops:
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
                eta = 1.0 / max(gmir(op.processing_times[m]), 0.001)
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

    def tabu_search(self, schedule):
        best_schedule = schedule[:]
        _, _, best_obj = evaluate_schedule(best_schedule, self.instance)
        current_schedule = schedule[:]
        tabu_list = []

        for _ in range(self.tabu_iter):
            best_neighbor = None
            best_neighbor_obj = float('inf')
            best_move = None

            for _ in range(10):
                idx = random.randint(0, len(current_schedule) - 1)
                j, o, m, pt = current_schedule[idx]
                op = self.instance.operations[j * self.instance.num_ops_per_job + o]

                if len(op.eligible_machines) > 1:
                    for new_m in op.eligible_machines:
                        if new_m != m:
                            move = (j, o, m, new_m)
                            if move not in tabu_list:
                                neighbor = current_schedule[:]
                                new_pt = gmir(op.processing_times[new_m])
                                neighbor[idx] = (j, o, new_m, new_pt)
                                _, _, obj = evaluate_schedule(neighbor, self.instance)
                                if obj < best_neighbor_obj:
                                    best_neighbor_obj = obj
                                    best_neighbor = neighbor
                                    best_move = move

            if best_neighbor:
                current_schedule = best_neighbor
                tabu_list.append(best_move)
                if len(tabu_list) > self.tabu_tenure:
                    tabu_list.pop(0)
                if best_neighbor_obj < best_obj:
                    best_obj = best_neighbor_obj
                    best_schedule = best_neighbor[:]

        return best_schedule

    def solve(self):
        best_schedule = None
        best_makespan = float('inf')
        best_energy = float('inf')
        best_objective = float('inf')
        convergence = []

        for iteration in range(self.max_iter):
            iter_best_schedule = None
            iter_best_objective = float('inf')

            for _ in range(self.num_ants):
                schedule = self.construct_solution()
                makespan, energy, objective = evaluate_schedule(schedule, self.instance)
                if objective < iter_best_objective:
                    iter_best_objective = objective
                    iter_best_schedule = schedule

            if iter_best_schedule:
                iter_best_schedule = self.tabu_search(iter_best_schedule)
                makespan, energy, objective = evaluate_schedule(iter_best_schedule, self.instance)
                if objective < best_objective:
                    best_objective = objective
                    best_schedule = iter_best_schedule
                    best_makespan = makespan
                    best_energy = energy

            convergence.append(best_objective)

            for key in self.pheromone:
                self.pheromone[key] *= (1 - self.rho)

            if iter_best_schedule:
                iter_makespan, _, _ = evaluate_schedule(iter_best_schedule, self.instance)
                for (j, o, m, pt) in iter_best_schedule:
                    key = (j, o, m)
                    if key in self.pheromone:
                        self.pheromone[key] += 1.0 / max(iter_makespan, 1)

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

class HIGA:
    def __init__(self, instance, pop_size=100, max_iter=200, crossover_rate=0.8,
                 mutation_rate=0.2, elitism=10, vns_iter=30, seed=None):
        self.instance = instance
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.vns_iter = vns_iter
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def create_individual(self):
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

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:]
        jobs = list(range(self.instance.num_jobs))
        random.shuffle(jobs)
        job_set = set(jobs[:len(jobs)//2])
        child = []
        p2_ops = [op for op in parent2 if op[0] not in job_set]
        p2_idx = 0
        for op in parent1:
            if op[0] in job_set:
                child.append(op)
            else:
                if p2_idx < len(p2_ops):
                    child.append(p2_ops[p2_idx])
                    p2_idx += 1
        return child if len(child) == len(parent1) else parent1[:]

    def mutate(self, individual):
        if random.random() > self.mutation_rate or len(individual) < 2:
            return individual
        child = individual[:]
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
        return child

    def vns(self, individual):
        best = individual[:]
        _, _, best_obj = evaluate_schedule(best, self.instance)
        for _ in range(self.vns_iter):
            neighbor = best[:]
            move = random.choice(['swap', 'change_machine'])
            if move == 'swap' and len(neighbor) >= 2:
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif move == 'change_machine':
                idx = random.randint(0, len(neighbor) - 1)
                j, o, m, pt = neighbor[idx]
                op = self.instance.operations[j * self.instance.num_ops_per_job + o]
                if len(op.eligible_machines) > 1:
                    new_m = random.choice([x for x in op.eligible_machines if x != m])
                    new_pt = gmir(op.processing_times[new_m])
                    neighbor[idx] = (j, o, new_m, new_pt)
            _, _, obj = evaluate_schedule(neighbor, self.instance)
            if obj < best_obj:
                best = neighbor
                best_obj = obj
        return best

    def solve(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_schedule = None
        best_makespan = float('inf')
        best_energy = float('inf')
        best_objective = float('inf')
        convergence = []

        for iteration in range(self.max_iter):
            fitness = []
            for ind in population:
                makespan, energy, objective = evaluate_schedule(ind, self.instance)
                fitness.append((objective, makespan, energy, ind))
            fitness.sort(key=lambda x: x[0])

            if fitness[0][0] < best_objective:
                best_objective = fitness[0][0]
                best_makespan = fitness[0][1]
                best_energy = fitness[0][2]
                best_schedule = fitness[0][3]

            convergence.append(best_objective)

            new_population = [f[3] for f in fitness[:self.elitism]]
            while len(new_population) < self.pop_size:
                tournament = random.sample(fitness, min(5, len(fitness)))
                parent1 = min(tournament, key=lambda x: x[0])[3]
                tournament = random.sample(fitness, min(5, len(fitness)))
                parent2 = min(tournament, key=lambda x: x[0])[3]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            new_population[0] = self.vns(new_population[0])
            population = new_population

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

class HMA:
    def __init__(self, instance, pop_size=60, max_iter=150, crossover_rate=0.9,
                 mutation_rate=0.1, elitism=5, ls_freq=5, ls_iter=30, seed=None):
        self.instance = instance
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.ls_freq = ls_freq
        self.ls_iter = ls_iter
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def create_individual(self):
        schedule = []
        next_op = [0] * self.instance.num_jobs
        machine_load = [0.0] * self.instance.num_machines

        while len(schedule) < len(self.instance.operations):
            ready_ops = []
            for j in range(self.instance.num_jobs):
                if next_op[j] < self.instance.num_ops_per_job:
                    op_idx = j * self.instance.num_ops_per_job + next_op[j]
                    ready_ops.append(self.instance.operations[op_idx])
            if not ready_ops:
                break

            best_op = None
            best_machine = None
            best_end = float('inf')
            for op in ready_ops:
                for m in op.eligible_machines:
                    pt = gmir(op.processing_times[m])
                    end_time = machine_load[m] + pt
                    if end_time < best_end:
                        best_end = end_time
                        best_op = op
                        best_machine = m

            if best_op:
                pt = gmir(best_op.processing_times[best_machine])
                schedule.append((best_op.job_id, best_op.op_id, best_machine, pt))
                machine_load[best_machine] += pt
                next_op[best_op.job_id] += 1

        return schedule

    def local_search(self, individual):
        best = individual[:]
        _, _, best_obj = evaluate_schedule(best, self.instance)
        for _ in range(self.ls_iter):
            neighbor = best[:]
            idx = random.randint(0, len(neighbor) - 1)
            j, o, m, pt = neighbor[idx]
            op = self.instance.operations[j * self.instance.num_ops_per_job + o]
            if len(op.eligible_machines) > 1:
                new_m = random.choice([x for x in op.eligible_machines if x != m])
                new_pt = gmir(op.processing_times[new_m])
                neighbor[idx] = (j, o, new_m, new_pt)
                _, _, obj = evaluate_schedule(neighbor, self.instance)
                if obj < best_obj:
                    best = neighbor
                    best_obj = obj
        return best

    def solve(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_schedule = None
        best_makespan = float('inf')
        best_energy = float('inf')
        best_objective = float('inf')
        convergence = []

        for iteration in range(self.max_iter):
            fitness = []
            for ind in population:
                makespan, energy, objective = evaluate_schedule(ind, self.instance)
                fitness.append((objective, makespan, energy, ind))
            fitness.sort(key=lambda x: x[0])

            if fitness[0][0] < best_objective:
                best_objective = fitness[0][0]
                best_makespan = fitness[0][1]
                best_energy = fitness[0][2]
                best_schedule = fitness[0][3]

            convergence.append(best_objective)

            new_population = [f[3] for f in fitness[:self.elitism]]
            while len(new_population) < self.pop_size:
                idx1 = random.randint(0, len(fitness) - 1)
                idx2 = random.randint(0, len(fitness) - 1)
                parent1 = fitness[idx1][3]
                parent2 = fitness[idx2][3]
                child = parent1[:] if random.random() < 0.5 else parent2[:]
                if random.random() < self.mutation_rate and len(child) >= 2:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                new_population.append(child)

            if iteration % self.ls_freq == 0:
                for i in range(min(5, len(new_population))):
                    new_population[i] = self.local_search(new_population[i])

            population = new_population

        return {'makespan': best_makespan, 'energy': best_energy, 'objective': best_objective, 'convergence': convergence}

def run_experiments(instances, num_runs=10):
    results = {}
    algorithms = {
        'EDP-ACO': lambda inst, seed: EDP_ACO(inst, seed=seed),
        'HACO-TS': lambda inst, seed: HACO_TS(inst, seed=seed),
        'HIGA': lambda inst, seed: HIGA(inst, seed=seed),
        'HMA': lambda inst, seed: HMA(inst, seed=seed)
    }

    for inst in instances:
        print(f"\n{'='*60}\nInstance: {inst.name}\n{'='*60}")
        results[inst.name] = {}

        for alg_name, alg_constructor in algorithms.items():
            print(f"\n  Running {alg_name}...")
            run_results = {'objectives': [], 'makespans': [], 'energies': [], 'times': [], 'convergence': []}

            for run in range(num_runs):
                start_time = time.time()
                solver = alg_constructor(inst, seed=run * 42 + 7)
                result = solver.solve()
                elapsed = time.time() - start_time

                run_results['makespans'].append(result['makespan'])
                run_results['energies'].append(result['energy'])
                run_results['objectives'].append(result['objective'])
                run_results['times'].append(elapsed)
                run_results['convergence'].append(result['convergence'])
                print(f"    Run {run+1}/{num_runs}: Obj={result['objective']:.2f}, Time={elapsed:.2f}s")

            results[inst.name][alg_name] = {
                'best': min(run_results['objectives']),
                'avg': np.mean(run_results['objectives']),
                'std': np.std(run_results['objectives']),
                'worst': max(run_results['objectives']),
                'avg_time': np.mean(run_results['times']),
                'best_makespan': min(run_results['makespans']),
                'avg_makespan': np.mean(run_results['makespans']),
                'best_energy': min(run_results['energies']),
                'avg_energy': np.mean(run_results['energies']),
                'all_objectives': run_results['objectives'],
                'convergence': run_results['convergence'][0]
            }

    return results

def statistical_tests(results):
    stat_results = {}
    algorithms = ['EDP-ACO', 'HACO-TS', 'HIGA', 'HMA']

    for inst_name, inst_results in results.items():
        stat_results[inst_name] = {}
        edp_aco_objs = inst_results['EDP-ACO']['all_objectives']

        for alg in algorithms[1:]:
            alg_objs = inst_results[alg]['all_objectives']
            try:
                w_stat, w_pvalue = stats.wilcoxon(edp_aco_objs, alg_objs)
            except:
                w_stat, w_pvalue = 0, 1.0

            stat_results[inst_name][f'EDP-ACO_vs_{alg}'] = {
                'wilcoxon_stat': float(w_stat),
                'wilcoxon_pvalue': float(w_pvalue),
                'significant': w_pvalue < 0.05,
                'edp_aco_wins': sum(1 for e, a in zip(edp_aco_objs, alg_objs) if e < a),
                'ties': sum(1 for e, a in zip(edp_aco_objs, alg_objs) if abs(e - a) < 0.01),
                'edp_aco_loses': sum(1 for e, a in zip(edp_aco_objs, alg_objs) if e > a)
            }

    all_ranks = {alg: [] for alg in algorithms}
    for inst_name, inst_results in results.items():
        objs = [(alg, inst_results[alg]['avg']) for alg in algorithms]
        objs.sort(key=lambda x: x[1])
        for rank, (alg, _) in enumerate(objs, 1):
            all_ranks[alg].append(rank)

    avg_ranks = {alg: np.mean(ranks) for alg, ranks in all_ranks.items()}
    n = len(list(results.keys()))
    k = len(algorithms)
    friedman_stat = 12 * n / (k * (k + 1)) * sum((r - (k + 1) / 2) ** 2 for r in avg_ranks.values()) if n > 0 else 0
    friedman_pvalue = 1 - stats.chi2.cdf(friedman_stat, k - 1) if n > 0 else 1.0

    stat_results['overall'] = {
        'friedman_stat': float(friedman_stat),
        'friedman_pvalue': float(friedman_pvalue),
        'avg_ranks': {k: float(v) for k, v in avg_ranks.items()}
    }

    return stat_results

def generate_latex_comparison_table(results, output_path):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of hybrid metaheuristics for GF-FJSP-PM}
\label{tab:hybrid_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Instance} & \textbf{Algorithm} & \textbf{Best} & \textbf{Avg} & \textbf{Std} & \textbf{Worst} & \textbf{Time(s)} & \textbf{RPD(\%)} & \textbf{Rank} \\
\midrule
"""
    algorithms = ['EDP-ACO', 'HACO-TS', 'HIGA', 'HMA']
    for inst_name, inst_results in results.items():
        best_overall = min(inst_results[alg]['best'] for alg in algorithms)
        ranked = sorted(algorithms, key=lambda a: inst_results[a]['avg'])
        for i, alg in enumerate(algorithms):
            data = inst_results[alg]
            rpd = (data['best'] - best_overall) / best_overall * 100 if best_overall > 0 else 0
            rank = ranked.index(alg) + 1
            inst_col = f"\\multirow{{4}}{{*}}{{{inst_name}}}" if i == 0 else ""
            best_str = f"\\textbf{{{data['best']:.2f}}}" if rank == 1 else f"{data['best']:.2f}"
            avg_str = f"\\textbf{{{data['avg']:.2f}}}" if rank == 1 else f"{data['avg']:.2f}"
            rpd_str = f"\\textbf{{{rpd:.2f}}}" if rank == 1 else f"{rpd:.2f}"
            rank_str = f"\\textbf{{{rank}}}" if rank == 1 else f"{rank}"
            latex += f"{inst_col} & {alg} & {best_str} & {avg_str} & {data['std']:.2f} & {data['worst']:.2f} & {data['avg_time']:.2f} & {rpd_str} & {rank_str} \\\\\n"
        latex += "\\midrule\n"
    latex = latex.rstrip("\\midrule\n") + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated: {output_path}")

def generate_latex_statistical_table(stat_results, output_path):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Statistical test results (EDP-ACO vs. other algorithms)}
\label{tab:statistical_tests}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Comparison} & \textbf{W-stat} & \textbf{p-value} & \textbf{Significant?} & \textbf{Wins} & \textbf{Ties} & \textbf{Losses} & \textbf{Winner} \\
\midrule
"""
    for comp in ['EDP-ACO_vs_HACO-TS', 'EDP-ACO_vs_HIGA', 'EDP-ACO_vs_HMA']:
        total_wins, total_ties, total_losses, pvalues = 0, 0, 0, []
        for inst_name, inst_stats in stat_results.items():
            if inst_name == 'overall':
                continue
            if comp in inst_stats:
                total_wins += inst_stats[comp]['edp_aco_wins']
                total_ties += inst_stats[comp]['ties']
                total_losses += inst_stats[comp]['edp_aco_loses']
                pvalues.append(inst_stats[comp]['wilcoxon_pvalue'])
        avg_pvalue = np.mean(pvalues) if pvalues else 1.0
        significant = "Yes" if avg_pvalue < 0.05 else "No"
        winner = "EDP-ACO" if total_wins > total_losses else comp.split('_vs_')[1]
        comp_name = comp.replace('_', ' ').replace('vs', 'vs.')
        latex += f"{comp_name} & -- & {avg_pvalue:.4f} & {significant} & {total_wins} & {total_ties} & {total_losses} & {winner} \\\\\n"

    latex += f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\textit{{Friedman Test:}} $\\chi^2 = {stat_results['overall']['friedman_stat']:.2f}$, $p = {stat_results['overall']['friedman_pvalue']:.4f}$}} \\\\\n\\bottomrule\n\\end{{tabular}}%\n}}\n\\end{{table}}\n"
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated: {output_path}")

def plot_convergence(results, output_dir):
    algorithms = ['EDP-ACO', 'HACO-TS', 'HIGA', 'HMA']
    colors = {'EDP-ACO': 'blue', 'HACO-TS': 'green', 'HIGA': 'red', 'HMA': 'orange'}
    styles = {'EDP-ACO': '-', 'HACO-TS': '--', 'HIGA': '-.', 'HMA': ':'}
    for inst_name, inst_results in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for alg in algorithms:
            conv = inst_results[alg]['convergence']
            ax.plot(conv, label=alg, color=colors[alg], linestyle=styles[alg], linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Objective Value', fontsize=12)
        ax.set_title(f'Convergence Comparison - {inst_name}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'convergence_{inst_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Generated: convergence plots")

def plot_boxplot(results, output_path):
    algorithms = ['EDP-ACO', 'HACO-TS', 'HIGA', 'HMA']
    fig, ax = plt.subplots(figsize=(12, 6))
    data, labels = [], []
    for inst_name, inst_results in results.items():
        for alg in algorithms:
            data.append(inst_results[alg]['all_objectives'])
            labels.append(f"{alg}\n({inst_name[:10]})")
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'] * len(results)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Algorithm Comparison (Boxplot)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def plot_bar_comparison(results, output_path):
    algorithms = ['EDP-ACO', 'HACO-TS', 'HIGA', 'HMA']
    instances = list(results.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(instances))
    width = 0.2
    for i, alg in enumerate(algorithms):
        avgs = [results[inst][alg]['avg'] for inst in instances]
        stds = [results[inst][alg]['std'] for inst in instances]
        ax.bar(x + i * width, avgs, width, label=alg, yerr=stds, capsize=3)
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Average Objective Value', fontsize=12)
    ax.set_title('Algorithm Comparison (Bar Chart)', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def main():
    output_dir = "hybrid_comparison"
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("HYBRID ALGORITHM COMPARISON FOR GF-FJSP-PM")
    print("=" * 60)

    print("\n1. Generating test instances...")
    instances = [
        generate_instance(6, 4, 3, flexibility=0.6, seed=42, name="Small_6x4x3"),
        generate_instance(10, 5, 4, flexibility=0.5, seed=43, name="Medium_10x5x4"),
        generate_instance(15, 6, 5, flexibility=0.4, seed=44, name="Medium_15x6x5"),
        generate_instance(20, 8, 5, flexibility=0.4, seed=45, name="Large_20x8x5"),
        generate_instance(30, 10, 5, flexibility=0.3, seed=46, name="Large_30x10x5"),
    ]

    print("\n2. Running experiments...")
    results = run_experiments(instances, num_runs=10)

    print("\n3. Performing statistical tests...")
    stat_results = statistical_tests(results)

    print("\n4. Saving results...")
    json_results = {inst: {alg: {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                                  for k, v in data.items() if k not in ['all_objectives', 'convergence']}
                           for alg, data in inst_results.items()}
                    for inst, inst_results in results.items()}
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2)

    with open(os.path.join(output_dir, "results.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instance', 'Algorithm', 'Best', 'Avg', 'Std', 'Worst', 'Time'])
        for inst_name, inst_results in results.items():
            for alg, data in inst_results.items():
                writer.writerow([inst_name, alg, data['best'], data['avg'], data['std'], data['worst'], data['avg_time']])

    with open(os.path.join(output_dir, "statistical_tests.json"), 'w') as f:
        json.dump(stat_results, f, indent=2)

    print("\n5. Generating LaTeX tables...")
    generate_latex_comparison_table(results, os.path.join(output_dir, "comparison_table.tex"))
    generate_latex_statistical_table(stat_results, os.path.join(output_dir, "statistical_table.tex"))

    print("\n6. Generating figures...")
    plot_convergence(results, figures_dir)
    plot_boxplot(results, os.path.join(figures_dir, "boxplot.png"))
    plot_bar_comparison(results, os.path.join(figures_dir, "comparison_bar.png"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nAverage Rankings:")
    for alg, rank in stat_results['overall']['avg_ranks'].items():
        marker = " ***" if alg == 'EDP-ACO' else ""
        print(f"  {alg}: {rank:.2f}{marker}")
    print(f"\nFriedman Test: chi2 = {stat_results['overall']['friedman_stat']:.2f}, p = {stat_results['overall']['friedman_pvalue']:.4f}")
    print(f"\nOutput files saved to: {output_dir}/")

if __name__ == "__main__":
    main()
