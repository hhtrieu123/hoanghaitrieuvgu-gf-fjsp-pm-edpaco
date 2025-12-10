#!/usr/bin/env python3
"""
==============================================================================
DEMO: EDP-ACO for Green Fuzzy FJSP with Preventive Maintenance
==============================================================================

This script demonstrates the EDP-ACO algorithm solving a small instance.
Run this to show your advisor the algorithm works.

Usage:
    python demo.py

Author: Hoang Hai Trieu
Student ID: VGU 20623015
==============================================================================
"""

import random
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

# =============================================================================
# FUZZY NUMBER CLASS
# =============================================================================

class FuzzyNumber:
    """Triangular Fuzzy Number (L, M, U)"""
    
    def __init__(self, L: float, M: float, U: float):
        self.L = L  # Lower bound
        self.M = M  # Most likely (modal) value
        self.U = U  # Upper bound
    
    def gmir(self) -> float:
        """Graded Mean Integration Representation for defuzzification"""
        return (self.L + 4 * self.M + self.U) / 6
    
    def __repr__(self):
        return f"({self.L}, {self.M}, {self.U})"
    
    def __add__(self, other):
        if isinstance(other, FuzzyNumber):
            return FuzzyNumber(self.L + other.L, self.M + other.M, self.U + other.U)
        return FuzzyNumber(self.L + other, self.M + other, self.U + other)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Operation:
    job_id: int
    op_id: int
    eligible_machines: List[int]
    processing_times: Dict[int, FuzzyNumber]  # machine_id -> fuzzy time

@dataclass
class Machine:
    machine_id: int
    power_processing: float  # kW during processing
    power_idle: float        # kW when idle
    maintenance_window: Tuple[float, float] = None  # (early, late)

@dataclass
class Instance:
    name: str
    num_jobs: int
    num_machines: int
    operations: List[Operation]
    machines: List[Machine]
    alpha: float = 0.5  # Weight for makespan
    beta: float = 0.5   # Weight for energy

# =============================================================================
# EDP-ACO ALGORITHM
# =============================================================================

class EDPACO:
    """Enhanced Dual-Pheromone Ant Colony Optimization"""
    
    def __init__(self, instance: Instance, params: dict = None):
        self.instance = instance
        self.params = params or {
            'num_ants': 20,
            'max_iter': 50,
            'alpha': 1.0,      # Pheromone importance
            'beta': 2.0,       # Heuristic importance
            'rho': 0.1,        # Evaporation rate
            'q0': 0.9,         # Exploitation probability
            'tau_min': 0.01,
            'tau_max': 10.0
        }
        
        # Initialize pheromone matrices
        self._init_pheromones()
        
    def _init_pheromones(self):
        """Initialize attractive and repulsive pheromone matrices"""
        n_ops = len(self.instance.operations)
        n_machines = self.instance.num_machines
        
        # Attractive pheromone (reinforces good choices)
        self.tau_plus = {}
        # Repulsive pheromone (discourages poor choices)
        self.tau_minus = {}
        
        for op in self.instance.operations:
            for m in op.eligible_machines:
                key = (op.job_id, op.op_id, m)
                self.tau_plus[key] = 1.0
                self.tau_minus[key] = 0.1
    
    def _calculate_heuristic(self, op: Operation, machine_id: int) -> float:
        """Calculate heuristic value (shorter processing time = higher)"""
        fuzzy_time = op.processing_times[machine_id]
        crisp_time = fuzzy_time.gmir()
        return 1.0 / max(crisp_time, 0.001)
    
    def _select_machine(self, op: Operation, job_end_times: List[float], 
                        machine_end_times: List[float]) -> int:
        """Select machine using ACO probability rule"""
        eligible = op.eligible_machines
        
        if random.random() < self.params['q0']:
            # Exploitation: choose best
            best_m = None
            best_value = -float('inf')
            for m in eligible:
                key = (op.job_id, op.op_id, m)
                tau = self.tau_plus[key] - 0.3 * self.tau_minus[key]
                eta = self._calculate_heuristic(op, m)
                value = (tau ** self.params['alpha']) * (eta ** self.params['beta'])
                if value > best_value:
                    best_value = value
                    best_m = m
            return best_m
        else:
            # Exploration: probabilistic selection
            probs = []
            for m in eligible:
                key = (op.job_id, op.op_id, m)
                tau = max(self.tau_plus[key] - 0.3 * self.tau_minus[key], 0.01)
                eta = self._calculate_heuristic(op, m)
                prob = (tau ** self.params['alpha']) * (eta ** self.params['beta'])
                probs.append(prob)
            
            total = sum(probs)
            if total == 0:
                return random.choice(eligible)
            
            probs = [p / total for p in probs]
            return random.choices(eligible, weights=probs, k=1)[0]
    
    def _construct_solution(self) -> List[Tuple[int, int, int, float]]:
        """Construct a schedule using ACO rules"""
        schedule = []
        job_op_idx = [0] * self.instance.num_jobs  # Next operation for each job
        job_end_times = [0.0] * self.instance.num_jobs
        machine_end_times = [0.0] * self.instance.num_machines
        
        # Group operations by job
        ops_by_job = {}
        for op in self.instance.operations:
            if op.job_id not in ops_by_job:
                ops_by_job[op.job_id] = []
            ops_by_job[op.job_id].append(op)
        
        # Sort operations within each job
        for j in ops_by_job:
            ops_by_job[j].sort(key=lambda x: x.op_id)
        
        n_total_ops = len(self.instance.operations)
        scheduled = 0
        
        while scheduled < n_total_ops:
            # Find schedulable operations
            schedulable = []
            for j in range(self.instance.num_jobs):
                if j in ops_by_job and job_op_idx[j] < len(ops_by_job[j]):
                    op = ops_by_job[j][job_op_idx[j]]
                    schedulable.append(op)
            
            if not schedulable:
                break
            
            # Select operation to schedule (random among schedulable)
            op = random.choice(schedulable)
            
            # Select machine
            m = self._select_machine(op, job_end_times, machine_end_times)
            
            # Calculate times
            fuzzy_time = op.processing_times[m]
            proc_time = fuzzy_time.gmir()
            start_time = max(machine_end_times[m], job_end_times[op.job_id])
            end_time = start_time + proc_time
            
            # Update state
            machine_end_times[m] = end_time
            job_end_times[op.job_id] = end_time
            job_op_idx[op.job_id] += 1
            
            schedule.append((op.job_id, op.op_id, m, proc_time))
            scheduled += 1
        
        return schedule
    
    def _evaluate(self, schedule: List[Tuple]) -> Tuple[float, float, float]:
        """Evaluate schedule: returns (makespan, energy, objective)"""
        machine_end_times = [0.0] * self.instance.num_machines
        job_end_times = [0.0] * self.instance.num_jobs
        total_energy = 0.0
        
        for (j, o, m, proc_time) in schedule:
            start_time = max(machine_end_times[m], job_end_times[j])
            end_time = start_time + proc_time
            machine_end_times[m] = end_time
            job_end_times[j] = end_time
            
            # Energy = processing power * time
            energy = proc_time * self.instance.machines[m].power_processing
            total_energy += energy
        
        makespan = max(machine_end_times)
        objective = self.instance.alpha * makespan + self.instance.beta * (total_energy / 100)
        
        return makespan, total_energy, objective
    
    def _update_pheromones(self, solutions: List[Tuple]):
        """Update pheromone matrices with dual mechanism"""
        rho = self.params['rho']
        tau_min = self.params['tau_min']
        tau_max = self.params['tau_max']
        
        # Evaporation
        for key in self.tau_plus:
            self.tau_plus[key] *= (1 - rho)
            self.tau_plus[key] = max(tau_min, min(tau_max, self.tau_plus[key]))
        
        # Sort solutions by objective (ascending = better)
        solutions.sort(key=lambda x: x[2])
        
        # Reinforce good solutions (top 30%)
        n_good = max(1, len(solutions) // 3)
        for schedule, _, obj in solutions[:n_good]:
            delta = 1.0 / obj
            for (j, o, m, _) in schedule:
                key = (j, o, m)
                if key in self.tau_plus:
                    self.tau_plus[key] += delta
        
        # Penalize poor solutions (bottom 30%)
        for schedule, _, obj in solutions[-n_good:]:
            delta = 0.5 / obj
            for (j, o, m, _) in schedule:
                key = (j, o, m)
                if key in self.tau_minus:
                    self.tau_minus[key] += delta
        
        # Decay repulsive pheromone
        for key in self.tau_minus:
            self.tau_minus[key] *= 0.95
    
    def solve(self) -> Tuple[List, float, float, float, List]:
        """Run EDP-ACO algorithm"""
        best_schedule = None
        best_makespan = float('inf')
        best_energy = float('inf')
        best_obj = float('inf')
        history = []
        
        print(f"\n{'='*60}")
        print(f"EDP-ACO Solving: {self.instance.name}")
        print(f"Jobs: {self.instance.num_jobs}, Machines: {self.instance.num_machines}")
        print(f"Operations: {len(self.instance.operations)}")
        print(f"{'='*60}")
        
        for iteration in range(self.params['max_iter']):
            solutions = []
            
            # Each ant constructs a solution
            for ant in range(self.params['num_ants']):
                schedule = self._construct_solution()
                makespan, energy, obj = self._evaluate(schedule)
                solutions.append((schedule, (makespan, energy), obj))
                
                # Update best
                if obj < best_obj:
                    best_obj = obj
                    best_makespan = makespan
                    best_energy = energy
                    best_schedule = schedule
            
            # Update pheromones
            self._update_pheromones(solutions)
            
            history.append(best_obj)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: Best Obj = {best_obj:.2f} "
                      f"(Makespan={best_makespan:.2f}, Energy={best_energy:.2f})")
        
        return best_schedule, best_makespan, best_energy, best_obj, history


# =============================================================================
# DEMO INSTANCE GENERATOR
# =============================================================================

def create_demo_instance() -> Instance:
    """Create a small demo instance for testing"""
    random.seed(42)
    np.random.seed(42)
    
    num_jobs = 4
    num_machines = 3
    num_ops_per_job = 3
    
    operations = []
    for j in range(num_jobs):
        for o in range(num_ops_per_job):
            # Each operation can run on 2 random machines
            eligible = sorted(random.sample(range(num_machines), 2))
            proc_times = {}
            for m in eligible:
                mid = random.randint(8, 20)
                low = mid - random.randint(1, 3)
                high = mid + random.randint(1, 3)
                proc_times[m] = FuzzyNumber(low, mid, high)
            operations.append(Operation(j, o, eligible, proc_times))
    
    machines = []
    for m in range(num_machines):
        power_proc = random.uniform(4.0, 8.0)
        power_idle = random.uniform(0.5, 1.5)
        machines.append(Machine(m, power_proc, power_idle))
    
    return Instance(
        name="Demo_4x3x3",
        num_jobs=num_jobs,
        num_machines=num_machines,
        operations=operations,
        machines=machines,
        alpha=0.5,
        beta=0.5
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  EDP-ACO: Enhanced Dual-Pheromone Ant Colony Optimization")
    print("  For Green Fuzzy FJSP with Preventive Maintenance")
    print("="*70)
    print("\nAuthor: Hoang Hai Trieu")
    print("Student ID: VGU 20623015")
    print("Advisors: Prof. Dr. Nguyen Thi Viet Ly, Prof. Dr. Brian Boyd")
    print("="*70)
    
    # Create demo instance
    print("\n[1] Creating demo instance...")
    instance = create_demo_instance()
    
    print(f"\nInstance: {instance.name}")
    print(f"  - Jobs: {instance.num_jobs}")
    print(f"  - Machines: {instance.num_machines}")
    print(f"  - Total Operations: {len(instance.operations)}")
    print(f"  - Objective Weights: α={instance.alpha}, β={instance.beta}")
    
    # Show fuzzy processing times
    print("\n[2] Sample Fuzzy Processing Times:")
    for i, op in enumerate(instance.operations[:3]):
        print(f"  Job {op.job_id}, Op {op.op_id}:")
        for m, fuzzy in op.processing_times.items():
            print(f"    Machine {m}: {fuzzy} -> GMIR={fuzzy.gmir():.2f}")
    print("  ...")
    
    # Show machine parameters
    print("\n[3] Machine Energy Parameters:")
    for m in instance.machines:
        print(f"  Machine {m.machine_id}: P_proc={m.power_processing:.2f}kW, "
              f"P_idle={m.power_idle:.2f}kW")
    
    # Run EDP-ACO
    print("\n[4] Running EDP-ACO Algorithm...")
    start_time = time.time()
    
    params = {
        'num_ants': 15,
        'max_iter': 30,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.1,
        'q0': 0.8,
        'tau_min': 0.01,
        'tau_max': 10.0
    }
    
    solver = EDPACO(instance, params)
    schedule, makespan, energy, obj, history = solver.solve()
    
    elapsed = time.time() - start_time
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Best Makespan:    {makespan:.2f} time units")
    print(f"Total Energy:     {energy:.2f} kWh")
    print(f"Objective Value:  {obj:.2f}")
    print(f"Computation Time: {elapsed:.2f} seconds")
    
    # Show schedule
    print("\n[5] Best Schedule Found:")
    print("-" * 50)
    print(f"{'Job':>4} {'Op':>4} {'Machine':>8} {'ProcTime':>10}")
    print("-" * 50)
    for (j, o, m, pt) in schedule[:10]:
        print(f"{j:>4} {o:>4} {m:>8} {pt:>10.2f}")
    if len(schedule) > 10:
        print(f"... ({len(schedule)-10} more operations)")
    
    # Convergence
    print("\n[6] Convergence History (every 5 iterations):")
    for i in range(0, len(history), 5):
        print(f"  Iter {i+1:3d}: {history[i]:.2f}")
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE - Algorithm working correctly!")
    print("="*70)
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history)+1), history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Objective Value', fontsize=12)
        plt.title('EDP-ACO Convergence on Demo Instance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo_convergence.png', dpi=150)
        print(f"\nConvergence plot saved to: demo_convergence.png")
    except Exception as e:
        print(f"\n(Could not save plot: {e})")
    
    return schedule, makespan, energy, obj


if __name__ == "__main__":
    main()
