#!/usr/bin/env python3
"""
Fuzzy Benchmark Generator for GF-FJSP-PM
=========================================

This script:
1. Extends standard FJSP benchmarks (Brandimarte, Kacem) with fuzzy processing times,
   energy parameters, and preventive maintenance
2. Generates harder instances for metaheuristic comparison
3. Solves small instances with MILP for optimal reference values
4. Exports instances in multiple formats (JSON, TXT)

Author: Master's Thesis Research
"""

import json
import os
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import time

# Try to import MILP solver
try:
    import pulp
    MILP_AVAILABLE = True
except ImportError:
    MILP_AVAILABLE = False
    print("Warning: PuLP not installed. MILP solving disabled.")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FuzzyTime:
    """Triangular Fuzzy Number (L, M, U)"""
    L: float  # Lower bound
    M: float  # Modal value
    U: float  # Upper bound
    
    def gmir(self) -> float:
        """Graded Mean Integration Representation"""
        return (self.L + 4*self.M + self.U) / 6
    
    def to_dict(self) -> dict:
        return {'L': self.L, 'M': self.M, 'U': self.U, 'GMIR': round(self.gmir(), 2)}


@dataclass
class Operation:
    """Operation with fuzzy processing times on eligible machines"""
    job_id: int
    op_id: int
    alternatives: Dict[int, FuzzyTime] = field(default_factory=dict)  # machine_id -> FuzzyTime


@dataclass
class Job:
    """Job consisting of sequential operations"""
    job_id: int
    operations: List[Operation] = field(default_factory=list)


@dataclass 
class Machine:
    """Machine with energy and PM parameters"""
    machine_id: int
    power_processing: float  # kW
    power_idle: float        # kW
    pm_duration: float       # time units
    pm_window_start: float   # earliest PM start
    pm_window_end: float     # latest PM start


@dataclass
class FuzzyInstance:
    """Complete GF-FJSP-PM instance"""
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job]
    machines: List[Machine]
    alpha: float = 0.5  # makespan weight
    beta: float = 0.5   # energy weight
    fuzziness: float = 0.2
    source: str = "generated"
    optimal_makespan: Optional[float] = None
    optimal_objective: Optional[float] = None
    
    @property
    def total_operations(self) -> int:
        return sum(len(j.operations) for j in self.jobs)
    
    @property
    def avg_flexibility(self) -> float:
        total_alts = sum(len(op.alternatives) for j in self.jobs for op in j.operations)
        return total_alts / self.total_operations if self.total_operations > 0 else 0


# =============================================================================
# BRANDIMARTE BENCHMARK PARSER
# =============================================================================

def parse_brandimarte_file(filepath: str) -> Tuple[int, int, List[List[List[Tuple[int, int]]]]]:
    """
    Parse Brandimarte format file.
    Returns: (num_jobs, num_machines, job_data)
    job_data[job][op] = [(machine, time), ...]
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # First line: num_jobs num_machines
    first_line = lines[0].split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])
    
    job_data = []
    for i in range(1, num_jobs + 1):
        if i >= len(lines):
            break
        parts = list(map(int, lines[i].split()))
        idx = 0
        num_ops = parts[idx]
        idx += 1
        
        ops = []
        for _ in range(num_ops):
            num_machines_for_op = parts[idx]
            idx += 1
            alternatives = []
            for _ in range(num_machines_for_op):
                machine = parts[idx] - 1  # Convert to 0-indexed
                time = parts[idx + 1]
                idx += 2
                alternatives.append((machine, time))
            ops.append(alternatives)
        job_data.append(ops)
    
    return num_jobs, num_machines, job_data


def extend_to_fuzzy(name: str, num_jobs: int, num_machines: int, 
                    job_data: List[List[List[Tuple[int, int]]]],
                    fuzziness: float = 0.2,
                    seed: int = 42) -> FuzzyInstance:
    """
    Extend crisp benchmark to fuzzy GF-FJSP-PM instance.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create jobs with fuzzy processing times
    jobs = []
    for j_idx, ops_data in enumerate(job_data):
        operations = []
        for o_idx, alternatives in enumerate(ops_data):
            op = Operation(job_id=j_idx, op_id=o_idx)
            for machine, crisp_time in alternatives:
                # Create fuzzy time around crisp value
                L = max(1, int(crisp_time * (1 - fuzziness)))
                M = crisp_time
                U = int(crisp_time * (1 + fuzziness))
                op.alternatives[machine] = FuzzyTime(L=L, M=M, U=U)
            operations.append(op)
        jobs.append(Job(job_id=j_idx, operations=operations))
    
    # Estimate makespan for PM window calculation
    total_time = sum(
        max(ft.M for ft in op.alternatives.values())
        for job in jobs for op in job.operations
    )
    est_makespan = total_time / num_machines * 1.5
    
    # Create machines with energy and PM parameters
    machines = []
    for m_idx in range(num_machines):
        power_proc = round(random.uniform(5.0, 15.0), 1)
        power_idle = round(power_proc * random.uniform(0.1, 0.25), 1)
        pm_duration = round(random.uniform(10, 30), 1)
        pm_start = round(est_makespan * 0.2, 1)
        pm_end = round(est_makespan * 0.7, 1)
        
        machines.append(Machine(
            machine_id=m_idx,
            power_processing=power_proc,
            power_idle=power_idle,
            pm_duration=pm_duration,
            pm_window_start=pm_start,
            pm_window_end=pm_end
        ))
    
    return FuzzyInstance(
        name=f"GF-{name}",
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs,
        machines=machines,
        fuzziness=fuzziness,
        source=f"Extended from {name}"
    )


# =============================================================================
# HARDER INSTANCE GENERATOR
# =============================================================================

def generate_hard_instance(
    name: str,
    num_jobs: int,
    num_machines: int,
    ops_per_job: int,
    flexibility: float = 0.5,
    fuzziness: float = 0.2,
    time_range: Tuple[int, int] = (10, 100),
    seed: int = None
) -> FuzzyInstance:
    """
    Generate harder instances for metaheuristic comparison.
    
    Parameters:
    -----------
    name: Instance name
    num_jobs: Number of jobs
    num_machines: Number of machines
    ops_per_job: Operations per job
    flexibility: Fraction of machines eligible per operation (0.0-1.0)
    fuzziness: Fuzzy spread (e.g., 0.2 = ±20%)
    time_range: (min_time, max_time) for processing times
    seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    jobs = []
    for j in range(num_jobs):
        operations = []
        for o in range(ops_per_job):
            op = Operation(job_id=j, op_id=o)
            
            # Determine eligible machines
            num_eligible = max(1, int(num_machines * flexibility))
            eligible_machines = random.sample(range(num_machines), num_eligible)
            
            for m in eligible_machines:
                # Generate modal processing time
                M = random.randint(time_range[0], time_range[1])
                L = max(1, int(M * (1 - fuzziness)))
                U = int(M * (1 + fuzziness))
                op.alternatives[m] = FuzzyTime(L=L, M=M, U=U)
            
            operations.append(op)
        jobs.append(Job(job_id=j, operations=operations))
    
    # Estimate makespan
    total_time = sum(
        max(ft.M for ft in op.alternatives.values())
        for job in jobs for op in job.operations
    )
    est_makespan = total_time / num_machines * 1.5
    
    # Create machines
    machines = []
    for m in range(num_machines):
        power_proc = round(random.uniform(5.0, 15.0), 1)
        power_idle = round(power_proc * random.uniform(0.1, 0.25), 1)
        pm_duration = round(random.uniform(10, 30), 1)
        pm_start = round(est_makespan * 0.2, 1)
        pm_end = round(est_makespan * 0.7, 1)
        
        machines.append(Machine(
            machine_id=m,
            power_processing=power_proc,
            power_idle=power_idle,
            pm_duration=pm_duration,
            pm_window_start=pm_start,
            pm_window_end=pm_end
        ))
    
    return FuzzyInstance(
        name=name,
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs,
        machines=machines,
        fuzziness=fuzziness,
        source="Generated hard instance"
    )


# =============================================================================
# MILP EXACT SOLVER
# =============================================================================

def solve_with_milp(instance: FuzzyInstance, time_limit: int = 300, verbose: bool = False) -> Dict:
    """
    Solve instance with MILP (exact algorithm) for optimal reference.
    Only feasible for small instances (≤8 jobs, ≤5 machines).
    """
    if not MILP_AVAILABLE:
        return {'status': 'MILP not available', 'makespan': None, 'objective': None}
    
    n = instance.num_jobs
    m = instance.num_machines
    
    # Check size
    if instance.total_operations > 30:
        return {'status': 'Too large for MILP', 'makespan': None, 'objective': None}
    
    # Big-M constant
    M_big = 10000
    
    # Create model
    model = pulp.LpProblem("GF_FJSP_PM", pulp.LpMinimize)
    
    # Index sets
    all_ops = [(j.job_id, op.op_id) for j in instance.jobs for op in j.operations]
    
    # Decision variables
    # x[i,j,k] = 1 if operation j of job i assigned to machine k
    x = {}
    for j in instance.jobs:
        for op in j.operations:
            for k in op.alternatives.keys():
                x[j.job_id, op.op_id, k] = pulp.LpVariable(
                    f"x_{j.job_id}_{op.op_id}_{k}", cat='Binary')
    
    # S[i,j] = start time of operation j of job i
    S = {}
    for j in instance.jobs:
        for op in j.operations:
            S[j.job_id, op.op_id] = pulp.LpVariable(
                f"S_{j.job_id}_{op.op_id}", lowBound=0)
    
    # C[i,j] = completion time of operation j of job i
    C = {}
    for j in instance.jobs:
        for op in j.operations:
            C[j.job_id, op.op_id] = pulp.LpVariable(
                f"C_{j.job_id}_{op.op_id}", lowBound=0)
    
    # C_max = makespan
    C_max = pulp.LpVariable("C_max", lowBound=0)
    
    # y[i,j,i',j',k] = 1 if (i,j) precedes (i',j') on machine k
    y = {}
    for i1, j1 in all_ops:
        for i2, j2 in all_ops:
            if (i1, j1) != (i2, j2):
                job1 = instance.jobs[i1]
                job2 = instance.jobs[i2]
                op1 = job1.operations[j1]
                op2 = job2.operations[j2]
                common_machines = set(op1.alternatives.keys()) & set(op2.alternatives.keys())
                for k in common_machines:
                    y[i1, j1, i2, j2, k] = pulp.LpVariable(
                        f"y_{i1}_{j1}_{i2}_{j2}_{k}", cat='Binary')
    
    # SM[k] = PM start time on machine k
    SM = {}
    for mach in instance.machines:
        SM[mach.machine_id] = pulp.LpVariable(
            f"SM_{mach.machine_id}", lowBound=0)
    
    # Idle time on machine k
    I = {}
    for mach in instance.machines:
        I[mach.machine_id] = pulp.LpVariable(
            f"I_{mach.machine_id}", lowBound=0)
    
    # --- Constraints ---
    
    # 1. Assignment: each operation assigned to exactly one machine
    for j in instance.jobs:
        for op in j.operations:
            model += pulp.lpSum(x[j.job_id, op.op_id, k] 
                               for k in op.alternatives.keys()) == 1
    
    # 2. Completion time definition
    for j in instance.jobs:
        for op in j.operations:
            for k, ft in op.alternatives.items():
                p = ft.gmir()
                model += C[j.job_id, op.op_id] >= S[j.job_id, op.op_id] + p - M_big * (1 - x[j.job_id, op.op_id, k])
    
    # 3. Precedence within job
    for j in instance.jobs:
        for idx in range(1, len(j.operations)):
            op_prev = j.operations[idx - 1]
            op_curr = j.operations[idx]
            model += S[j.job_id, op_curr.op_id] >= C[j.job_id, op_prev.op_id]
    
    # 4. Machine capacity (disjunctive)
    for i1, j1 in all_ops:
        for i2, j2 in all_ops:
            if (i1, j1) < (i2, j2):
                job1 = instance.jobs[i1]
                job2 = instance.jobs[i2]
                op1 = job1.operations[j1]
                op2 = job2.operations[j2]
                common = set(op1.alternatives.keys()) & set(op2.alternatives.keys())
                for k in common:
                    p1 = op1.alternatives[k].gmir()
                    p2 = op2.alternatives[k].gmir()
                    # Either (i1,j1) before (i2,j2) or vice versa
                    model += S[i2, j2] >= C[i1, j1] - M_big * (3 - x[i1, j1, k] - x[i2, j2, k] - y[i1, j1, i2, j2, k])
                    model += S[i1, j1] >= C[i2, j2] - M_big * (2 - x[i1, j1, k] - x[i2, j2, k] + y[i1, j1, i2, j2, k])
    
    # 5. PM window constraints
    for mach in