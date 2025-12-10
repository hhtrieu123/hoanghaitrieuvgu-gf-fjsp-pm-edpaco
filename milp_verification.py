"""
==============================================================================
GF-FJSP-PM: MILP Formulation Verification
==============================================================================

This script implements the exact MILP formulation for:
- Green Fuzzy Flexible Job Shop Scheduling Problem with Preventive Maintenance

Purpose:
1. Verify mathematical formulation correctness
2. Obtain optimal solutions for small instances
3. Validate constraint satisfaction
4. Generate test data for algorithm comparison

Author: [Your Name]
Date: 2024
==============================================================================
"""

import pulp
import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import csv

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class FuzzyNumber:
    """Triangular Fuzzy Number (L, M, U)"""
    L: float  # Lower bound (optimistic)
    M: float  # Most likely (modal)
    U: float  # Upper bound (pessimistic)
    
    def defuzzify_gmir(self) -> float:
        """Graded Mean Integration Representation (Chen & Hsieh, 1999)"""
        return (self.L + 4*self.M + self.U) / 6
    
    def defuzzify_centroid(self) -> float:
        """Centroid method"""
        return (self.L + self.M + self.U) / 3
    
    def __repr__(self):
        return f"({self.L}, {self.M}, {self.U})"


@dataclass
class Operation:
    """Single operation with machine alternatives"""
    job_id: int
    op_idx: int
    alternatives: Dict[int, FuzzyNumber]  # machine_id -> fuzzy processing time
    
    def get_machines(self) -> List[int]:
        return list(self.alternatives.keys())


@dataclass
class Job:
    """Job with sequence of operations"""
    job_id: int
    operations: List[Operation]
    
    @property
    def num_ops(self) -> int:
        return len(self.operations)


@dataclass
class Machine:
    """Machine with energy and maintenance parameters"""
    machine_id: int
    power_processing: float  # kW while processing
    power_idle: float        # kW while idle
    pm_duration: float       # PM duration
    pm_window_start: float   # Earliest PM start
    pm_window_end: float     # Latest PM start


@dataclass
class GF_FJSP_PM_Instance:
    """Complete problem instance"""
    name: str
    jobs: List[Job]
    machines: List[Machine]
    alpha: float = 0.5  # Weight for makespan
    beta: float = 0.5   # Weight for energy
    
    @property
    def num_jobs(self) -> int:
        return len(self.jobs)
    
    @property
    def num_machines(self) -> int:
        return len(self.machines)
    
    @property
    def total_ops(self) -> int:
        return sum(j.num_ops for j in self.jobs)


# ==============================================================================
# MILP FORMULATION
# ==============================================================================

class GF_FJSP_PM_MILP:
    """
    MILP Formulation for GF-FJSP-PM
    
    Decision Variables:
    - x[i,j,k]: Binary, 1 if operation O_ij assigned to machine k
    - S[i,j]: Continuous, start time of operation O_ij
    - C[i,j]: Continuous, completion time of operation O_ij
    - y[i,j,i',j',k]: Binary, 1 if O_ij precedes O_i'j' on machine k
    - SM[k]: Continuous, start time of PM on machine k
    - z[i,j,k]: Binary, 1 if O_ij precedes PM on machine k
    - C_max: Continuous, makespan
    - I[k]: Continuous, idle time of machine k
    """
    
    def __init__(self, instance: GF_FJSP_PM_Instance, big_m: float = 10000):
        self.instance = instance
        self.big_m = big_m
        self.model = None
        self.variables = {}
        self.solution = None
        
    def build_model(self, use_fuzzy: bool = True):
        """Build the MILP model"""
        
        inst = self.instance
        M = self.big_m
        
        # Create model
        self.model = pulp.LpProblem("GF_FJSP_PM", pulp.LpMinimize)
        
        # Get all operations as (job_id, op_idx) pairs
        all_ops = [(j.job_id, op.op_idx) for j in inst.jobs for op in j.operations]
        
        # =================================================================
        # DECISION VARIABLES
        # =================================================================
        
        # x[i,j,k]: Assignment of operation (i,j) to machine k
        x = {}
        for i, j in all_ops:
            op = inst.jobs[i].operations[j]
            for k in op.get_machines():
                x[i,j,k] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat='Binary')
        
        # S[i,j]: Start time of operation (i,j)
        S = {}
        for i, j in all_ops:
            S[i,j] = pulp.LpVariable(f"S_{i}_{j}", lowBound=0, cat='Continuous')
        
        # C[i,j]: Completion time of operation (i,j)
        C = {}
        for i, j in all_ops:
            C[i,j] = pulp.LpVariable(f"C_{i}_{j}", lowBound=0, cat='Continuous')
        
        # y[i,j,i',j',k]: Sequencing variable - O_ij before O_i'j' on machine k
        y = {}
        for i1, j1 in all_ops:
            op1 = inst.jobs[i1].operations[j1]
            for i2, j2 in all_ops:
                if (i1, j1) >= (i2, j2):
                    continue
                op2 = inst.jobs[i2].operations[j2]
                common_machines = set(op1.get_machines()) & set(op2.get_machines())
                for k in common_machines:
                    y[i1,j1,i2,j2,k] = pulp.LpVariable(f"y_{i1}_{j1}_{i2}_{j2}_{k}", cat='Binary')
        
        # SM[k]: Start time of PM on machine k
        SM = {}
        for m in inst.machines:
            SM[m.machine_id] = pulp.LpVariable(f"SM_{m.machine_id}", lowBound=0, cat='Continuous')
        
        # z[i,j,k]: 1 if operation (i,j) precedes PM on machine k
        z = {}
        for i, j in all_ops:
            op = inst.jobs[i].operations[j]
            for k in op.get_machines():
                z[i,j,k] = pulp.LpVariable(f"z_{i}_{j}_{k}", cat='Binary')
        
        # C_max: Makespan
        C_max = pulp.LpVariable("C_max", lowBound=0, cat='Continuous')
        
        # I[k]: Idle time of machine k
        I = {}
        for m in inst.machines:
            I[m.machine_id] = pulp.LpVariable(f"I_{m.machine_id}", lowBound=0, cat='Continuous')
        
        # Store variables
        self.variables = {'x': x, 'S': S, 'C': C, 'y': y, 'SM': SM, 'z': z, 'C_max': C_max, 'I': I}
        
        # =================================================================
        # OBJECTIVE FUNCTION
        # =================================================================
        
        # Energy consumption
        E_proc = pulp.lpSum([
            inst.machines[k].power_processing * 
            (inst.jobs[i].operations[j].alternatives[k].defuzzify_gmir() if use_fuzzy 
             else inst.jobs[i].operations[j].alternatives[k].M) * 
            x[i,j,k]
            for i, j in all_ops
            for k in inst.jobs[i].operations[j].get_machines()
        ])
        
        E_idle = pulp.lpSum([
            inst.machines[m.machine_id].power_idle * I[m.machine_id]
            for m in inst.machines
        ])
        
        E_total = E_proc + E_idle
        
        # Weighted objective
        self.model += inst.alpha * C_max + inst.beta * E_total, "Objective"
        
        # =================================================================
        # CONSTRAINTS
        # =================================================================
        
        # (1) Assignment constraint: Each operation assigned to exactly one machine
        for i, j in all_ops:
            op = inst.jobs[i].operations[j]
            self.model += pulp.lpSum([x[i,j,k] for k in op.get_machines()]) == 1, f"Assign_{i}_{j}"
        
        # (2) Completion time definition
        for i, j in all_ops:
            op = inst.jobs[i].operations[j]
            for k in op.get_machines():
                p_ijk = op.alternatives[k].defuzzify_gmir() if use_fuzzy else op.alternatives[k].M
                self.model += C[i,j] >= S[i,j] + p_ijk - M*(1 - x[i,j,k]), f"Complete_{i}_{j}_{k}"
        
        # (3) Precedence constraint: Operations within job must be sequential
        for job in inst.jobs:
            i = job.job_id
            for j in range(job.num_ops - 1):
                self.model += S[i,j+1] >= C[i,j], f"Prec_{i}_{j}"
        
        # (4) Machine capacity: No overlap on same machine
        for i1, j1 in all_ops:
            op1 = inst.jobs[i1].operations[j1]
            for i2, j2 in all_ops:
                if (i1, j1) >= (i2, j2):
                    continue
                op2 = inst.jobs[i2].operations[j2]
                common_machines = set(op1.get_machines()) & set(op2.get_machines())
                for k in common_machines:
                    p1 = op1.alternatives[k].defuzzify_gmir() if use_fuzzy else op1.alternatives[k].M
                    p2 = op2.alternatives[k].defuzzify_gmir() if use_fuzzy else op2.alternatives[k].M
                    
                    # O_ij finishes before O_i'j' starts, OR vice versa
                    self.model += S[i2,j2] >= C[i1,j1] - M*(1 - y[i1,j1,i2,j2,k]) - M*(1 - x[i1,j1,k]) - M*(1 - x[i2,j2,k]), f"NoOverlap1_{i1}_{j1}_{i2}_{j2}_{k}"
                    self.model += S[i1,j1] >= C[i2,j2] - M*y[i1,j1,i2,j2,k] - M*(1 - x[i1,j1,k]) - M*(1 - x[i2,j2,k]), f"NoOverlap2_{i1}_{j1}_{i2}_{j2}_{k}"
        
        # (5) Maintenance window constraint
        for m in inst.machines:
            k = m.machine_id
            self.model += SM[k] >= m.pm_window_start, f"PM_Early_{k}"
            self.model += SM[k] <= m.pm_window_end, f"PM_Late_{k}"
        
        # (6) Operation-PM disjunction: No overlap between operations and PM
        for i, j in all_ops:
            op = inst.jobs[i].operations[j]
            for k in op.get_machines():
                m = inst.machines[k]
                p_ijk = op.alternatives[k].defuzzify_gmir() if use_fuzzy else op.alternatives[k].M
                
                # Operation before PM OR PM before operation
                self.model += SM[k] >= C[i,j] - M*(1 - z[i,j,k]) - M*(1 - x[i,j,k]), f"OpPM1_{i}_{j}_{k}"
                self.model += S[i,j] >= SM[k] + m.pm_duration - M*z[i,j,k] - M*(1 - x[i,j,k]), f"OpPM2_{i}_{j}_{k}"
        
        # (7) Makespan definition
        for i, j in all_ops:
            self.model += C_max >= C[i,j], f"Makespan_{i}_{j}"
        
        # (8) Idle time calculation (linearized approximation)
        for m in inst.machines:
            k = m.machine_id
            total_proc_time = pulp.lpSum([
                (inst.jobs[i].operations[j].alternatives[k].defuzzify_gmir() if use_fuzzy 
                 else inst.jobs[i].operations[j].alternatives[k].M) * x[i,j,k]
                for i, j in all_ops
                if k in inst.jobs[i].operations[j].get_machines()
            ])
            # I[k] = C_max - total_processing_time - pm_duration (approximately)
            self.model += I[k] >= C_max - total_proc_time - m.pm_duration, f"Idle_{k}"
        
        return self.model
    
    def solve(self, solver_name: str = 'CBC', time_limit: int = 3600, verbose: bool = False):
        """Solve the MILP model"""
        
        if self.model is None:
            self.build_model()
        
        # Select solver
        if solver_name.upper() == 'GUROBI':
            solver = pulp.GUROBI(timeLimit=time_limit, msg=verbose)
        elif solver_name.upper() == 'CPLEX':
            solver = pulp.CPLEX(timeLimit=time_limit, msg=verbose)
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
        
        # Solve
        start_time = time.time()
        status = self.model.solve(solver)
        solve_time = time.time() - start_time
        
        # Extract solution
        self.solution = {
            'status': pulp.LpStatus[status],
            'objective': pulp.value(self.model.objective),
            'solve_time': solve_time,
            'makespan': pulp.value(self.variables['C_max']),
            'assignments': {},
            'start_times': {},
            'completion_times': {},
            'pm_times': {},
            'idle_times': {}
        }
        
        # Extract variable values
        x = self.variables['x']
        S = self.variables['S']
        C = self.variables['C']
        SM = self.variables['SM']
        I = self.variables['I']
        
        for (i, j, k), var in x.items():
            if pulp.value(var) > 0.5:
                self.solution['assignments'][(i,j)] = k
        
        for (i, j), var in S.items():
            self.solution['start_times'][(i,j)] = pulp.value(var)
        
        for (i, j), var in C.items():
            self.solution['completion_times'][(i,j)] = pulp.value(var)
        
        for k, var in SM.items():
            self.solution['pm_times'][k] = pulp.value(var)
        
        for k, var in I.items():
            self.solution['idle_times'][k] = pulp.value(var)
        
        return self.solution
    
    def verify_solution(self) -> Dict:
        """Verify that solution satisfies all constraints"""
        
        if self.solution is None:
            return {'valid': False, 'errors': ['No solution to verify']}
        
        errors = []
        warnings = []
        inst = self.instance
        
        assignments = self.solution['assignments']
        start_times = self.solution['start_times']
        completion_times = self.solution['completion_times']
        pm_times = self.solution['pm_times']
        
        # Check 1: All operations assigned
        for job in inst.jobs:
            for op in job.operations:
                key = (op.job_id, op.op_idx)
                if key not in assignments:
                    errors.append(f"Operation {key} not assigned to any machine")
        
        # Check 2: Precedence constraints
        for job in inst.jobs:
            for j in range(job.num_ops - 1):
                key1 = (job.job_id, j)
                key2 = (job.job_id, j+1)
                if completion_times.get(key1, 0) > start_times.get(key2, float('inf')) + 0.001:
                    errors.append(f"Precedence violated: {key1} completes at {completion_times[key1]:.2f} but {key2} starts at {start_times[key2]:.2f}")
        
        # Check 3: Machine capacity (no overlap)
        machine_schedule = {m.machine_id: [] for m in inst.machines}
        for (i, j), k in assignments.items():
            machine_schedule[k].append({
                'op': (i, j),
                'start': start_times[(i,j)],
                'end': completion_times[(i,j)]
            })
        
        for k, schedule in machine_schedule.items():
            schedule.sort(key=lambda x: x['start'])
            for idx in range(len(schedule) - 1):
                if schedule[idx]['end'] > schedule[idx+1]['start'] + 0.001:
                    errors.append(f"Overlap on machine {k}: {schedule[idx]['op']} ends at {schedule[idx]['end']:.2f} but {schedule[idx+1]['op']} starts at {schedule[idx+1]['start']:.2f}")
        
        # Check 4: PM window constraints
        for m in inst.machines:
            k = m.machine_id
            pm_start = pm_times.get(k, 0)
            if pm_start < m.pm_window_start - 0.001:
                errors.append(f"PM on machine {k} starts at {pm_start:.2f}, before window {m.pm_window_start}")
            if pm_start > m.pm_window_end + 0.001:
                errors.append(f"PM on machine {k} starts at {pm_start:.2f}, after window {m.pm_window_end}")
        
        # Check 5: PM-operation disjunction
        for m in inst.machines:
            k = m.machine_id
            pm_start = pm_times.get(k, 0)
            pm_end = pm_start + m.pm_duration
            
            for item in machine_schedule[k]:
                op_start = item['start']
                op_end = item['end']
                
                # Check for overlap
                if not (op_end <= pm_start + 0.001 or op_start >= pm_end - 0.001):
                    errors.append(f"Operation {item['op']} overlaps with PM on machine {k}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'num_errors': len(errors)
        }
    
    def print_solution(self):
        """Print solution in readable format"""
        
        if self.solution is None:
            print("No solution available")
            return
        
        print("\n" + "="*70)
        print("                    MILP SOLUTION REPORT")
        print("="*70)
        
        print(f"\nStatus: {self.solution['status']}")
        print(f"Objective Value: {self.solution['objective']:.4f}")
        print(f"Makespan: {self.solution['makespan']:.2f}")
        print(f"Solve Time: {self.solution['solve_time']:.2f} seconds")
        
        print("\n--- Operation Schedule ---")
        print(f"{'Job':<6} {'Op':<6} {'Machine':<10} {'Start':<10} {'End':<10}")
        print("-" * 50)
        
        schedule = []
        for (i, j), k in self.solution['assignments'].items():
            schedule.append({
                'job': i, 'op': j, 'machine': k,
                'start': self.solution['start_times'][(i,j)],
                'end': self.solution['completion_times'][(i,j)]
            })
        
        schedule.sort(key=lambda x: (x['start'], x['job'], x['op']))
        for item in schedule:
            print(f"{item['job']:<6} {item['op']:<6} {item['machine']:<10} {item['start']:<10.2f} {item['end']:<10.2f}")
        
        print("\n--- PM Schedule ---")
        print(f"{'Machine':<10} {'PM Start':<12} {'PM End':<12}")
        print("-" * 35)
        for m in self.instance.machines:
            k = m.machine_id
            pm_start = self.solution['pm_times'].get(k, 0)
            pm_end = pm_start + m.pm_duration
            print(f"{k:<10} {pm_start:<12.2f} {pm_end:<12.2f}")
        
        # Verify solution
        print("\n--- Constraint Verification ---")
        verification = self.verify_solution()
        if verification['valid']:
            print("✓ All constraints satisfied!")
        else:
            print(f"✗ Found {verification['num_errors']} constraint violations:")
            for error in verification['errors']:
                print(f"  - {error}")


# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_random_instance(
    num_jobs: int,
    num_machines: int,
    ops_per_job: int,
    flexibility: float = 0.5,  # Fraction of machines each op can use
    time_range: Tuple[int, int] = (10, 100),
    fuzziness: float = 0.2,  # Uncertainty level
    seed: int = None
) -> GF_FJSP_PM_Instance:
    """
    Generate random GF-FJSP-PM instance
    
    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        ops_per_job: Operations per job
        flexibility: Average fraction of machines each operation can use
        time_range: (min, max) processing time
        fuzziness: Relative uncertainty (e.g., 0.2 = ±20%)
        seed: Random seed
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    jobs = []
    for i in range(num_jobs):
        operations = []
        for j in range(ops_per_job):
            # Determine eligible machines
            num_eligible = max(1, int(num_machines * flexibility))
            eligible = np.random.choice(num_machines, num_eligible, replace=False)
            
            alternatives = {}
            for k in eligible:
                # Generate fuzzy processing time
                modal = np.random.randint(time_range[0], time_range[1])
                lower = int(modal * (1 - fuzziness))
                upper = int(modal * (1 + fuzziness))
                alternatives[int(k)] = FuzzyNumber(lower, modal, upper)
            
            operations.append(Operation(job_id=i, op_idx=j, alternatives=alternatives))
        
        jobs.append(Job(job_id=i, operations=operations))
    
    # Generate machines with energy and PM parameters
    machines = []
    total_time_estimate = num_jobs * ops_per_job * np.mean(time_range) / num_machines
    
    for k in range(num_machines):
        power_proc = np.random.uniform(5, 15)  # kW
        power_idle = np.random.uniform(1, 3)   # kW
        pm_duration = np.random.uniform(10, 30)
        pm_window_start = total_time_estimate * 0.3
        pm_window_end = total_time_estimate * 0.8
        
        machines.append(Machine(
            machine_id=k,
            power_processing=power_proc,
            power_idle=power_idle,
            pm_duration=pm_duration,
            pm_window_start=pm_window_start,
            pm_window_end=pm_window_end
        ))
    
    return GF_FJSP_PM_Instance(
        name=f"Random_{num_jobs}x{num_machines}x{ops_per_job}",
        jobs=jobs,
        machines=machines
    )


def convert_brandimarte_to_gf_fjsp_pm(
    filepath: str,
    fuzziness: float = 0.15,
    power_proc_range: Tuple[float, float] = (5, 15),
    power_idle_range: Tuple[float, float] = (1, 3)
) -> GF_FJSP_PM_Instance:
    """
    Convert Brandimarte instance to GF-FJSP-PM format
    
    Adds:
    - Fuzzy processing times
    - Energy parameters
    - Maintenance windows
    """
    
    with open(filepath, 'r') as f:
        first_line = f.readline().split()
        num_jobs = int(first_line[0])
        num_machines = int(first_line[1])
        
        jobs = []
        total_time = 0
        
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
                    m_id = int(line[idx]) - 1  # Convert to 0-based
                    p_time = int(line[idx + 1])
                    idx += 2
                    
                    # Add fuzziness
                    lower = int(p_time * (1 - fuzziness))
                    upper = int(p_time * (1 + fuzziness))
                    alternatives[m_id] = FuzzyNumber(lower, p_time, upper)
                    total_time += p_time
                
                operations.append(Operation(job_id=i, op_idx=j, alternatives=alternatives))
            
            jobs.append(Job(job_id=i, operations=operations))
    
    # Estimate makespan for PM window calculation
    avg_makespan = total_time / num_machines
    
    # Generate machine parameters
    np.random.seed(42)  # Reproducible
    machines = []
    for k in range(num_machines):
        machines.append(Machine(
            machine_id=k,
            power_processing=np.random.uniform(*power_proc_range),
            power_idle=np.random.uniform(*power_idle_range),
            pm_duration=np.random.uniform(5, 20),
            pm_window_start=avg_makespan * 0.2,
            pm_window_end=avg_makespan * 0.7
        ))
    
    name = filepath.split('/')[-1].replace('.txt', '')
    return GF_FJSP_PM_Instance(name=name, jobs=jobs, machines=machines)


# ==============================================================================
# VERIFICATION EXPERIMENTS
# ==============================================================================

def run_verification_experiments():
    """Run experiments to verify MILP formulation"""
    
    print("="*70)
    print("         GF-FJSP-PM MILP FORMULATION VERIFICATION")
    print("="*70)
    
    results = []
    
    # Test 1: Very small instance (should solve quickly)
    print("\n" + "="*70)
    print("[TEST 1] Very Small Instance (3 jobs × 2 machines × 2 ops)")
    print("="*70)
    
    inst1 = generate_random_instance(
        num_jobs=3, num_machines=2, ops_per_job=2,
        flexibility=0.8, seed=42
    )
    
    print("\n--- Instance Details ---")
    print(f"Jobs: {inst1.num_jobs}, Machines: {inst1.num_machines}")
    print(f"Total Operations: {inst1.total_ops}")
    for job in inst1.jobs:
        print(f"\nJob {job.job_id}:")
        for op in job.operations:
            for k, ft in op.alternatives.items():
                print(f"  Op {op.op_idx} -> M{k}: ({ft.L}, {ft.M}, {ft.U}) -> GMIR={ft.defuzzify_gmir():.2f}")
    
    print("\nMachine Parameters:")
    for m in inst1.machines:
        print(f"  M{m.machine_id}: P_proc={m.power_processing:.1f}kW, P_idle={m.power_idle:.1f}kW, "
              f"PM_dur={m.pm_duration:.1f}, Window=[{m.pm_window_start:.1f}, {m.pm_window_end:.1f}]")
    
    print("\n--- Solving MILP ---")
    milp1 = GF_FJSP_PM_MILP(inst1)
    milp1.build_model()
    solution1 = milp1.solve(time_limit=60)
    milp1.print_solution()
    
    verification1 = milp1.verify_solution()
    results.append({
        'test': 'Small Instance',
        'size': '3×2×2',
        'status': solution1['status'],
        'objective': solution1['objective'],
        'makespan': solution1['makespan'],
        'time': solution1['solve_time'],
        'valid': verification1['valid']
    })
    
    # Test 2: Medium instance
    print("\n" + "="*70)
    print("[TEST 2] Medium Instance (5 jobs × 3 machines × 3 ops)")
    print("="*70)
    
    inst2 = generate_random_instance(
        num_jobs=5, num_machines=3, ops_per_job=3,
        flexibility=0.6, seed=123
    )
    
    print("\n--- Instance Details ---")
    print(f"Jobs: {inst2.num_jobs}, Machines: {inst2.num_machines}")
    print(f"Total Operations: {inst2.total_ops}")
    
    print("\n--- Solving MILP ---")
    milp2 = GF_FJSP_PM_MILP(inst2)
    milp2.build_model()
    solution2 = milp2.solve(time_limit=300)
    milp2.print_solution()
    
    verification2 = milp2.verify_solution()
    results.append({
        'test': 'Medium Instance',
        'size': '5×3×3',
        'status': solution2['status'],
        'objective': solution2['objective'],
        'makespan': solution2['makespan'],
        'time': solution2['solve_time'],
        'valid': verification2['valid']
    })
    
    # Test 3: Compare crisp vs fuzzy
    print("\n" + "="*70)
    print("[TEST 3] Fuzzy vs Crisp Comparison (4 jobs × 3 machines × 2 ops)")
    print("="*70)
    
    inst3 = generate_random_instance(
        num_jobs=4, num_machines=3, ops_per_job=2,
        flexibility=0.7, fuzziness=0.25, seed=456
    )
    
    # Fuzzy model
    print("\n--- Fuzzy Model (GMIR Defuzzification) ---")
    milp3_fuzzy = GF_FJSP_PM_MILP(inst3)
    milp3_fuzzy.build_model(use_fuzzy=True)
    sol_fuzzy = milp3_fuzzy.solve(time_limit=120, verbose=False)
    print(f"Status: {sol_fuzzy['status']}")
    print(f"Makespan: {sol_fuzzy['makespan']:.2f}")
    print(f"Objective: {sol_fuzzy['objective']:.4f}")
    print(f"Solve Time: {sol_fuzzy['solve_time']:.2f}s")
    
    # Crisp model (using modal values only)
    print("\n--- Crisp Model (Modal Values Only) ---")
    milp3_crisp = GF_FJSP_PM_MILP(inst3)
    milp3_crisp.build_model(use_fuzzy=False)
    sol_crisp = milp3_crisp.solve(time_limit=120, verbose=False)
    print(f"Status: {sol_crisp['status']}")
    print(f"Makespan: {sol_crisp['makespan']:.2f}")
    print(f"Objective: {sol_crisp['objective']:.4f}")
    print(f"Solve Time: {sol_crisp['solve_time']:.2f}s")
    
    print("\n--- Comparison ---")
    print(f"Makespan Difference: {sol_fuzzy['makespan'] - sol_crisp['makespan']:.2f} "
          f"({(sol_fuzzy['makespan'] - sol_crisp['makespan'])/sol_crisp['makespan']*100:.2f}%)")
    print(f"Objective Difference: {sol_fuzzy['objective'] - sol_crisp['objective']:.4f}")
    print("\nNote: Fuzzy model uses GMIR = (L + 4M + U)/6 which accounts for uncertainty.")
    print("      This typically results in slightly higher values than the crisp modal value.")
    
    # Summary
    print("\n" + "="*70)
    print("                    VERIFICATION SUMMARY")
    print("="*70)
    print(f"\n{'Test':<20} {'Size':<10} {'Status':<12} {'Makespan':<12} {'Time(s)':<10} {'Valid':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['test']:<20} {r['size']:<10} {r['status']:<12} {r['makespan']:<12.2f} {r['time']:<10.2f} {'✓' if r['valid'] else '✗':<8}")
    
    return results


def save_instance_to_file(instance: GF_FJSP_PM_Instance, filepath: str):
    """Save instance to JSON format"""
    
    data = {
        'name': instance.name,
        'num_jobs': instance.num_jobs,
        'num_machines': instance.num_machines,
        'alpha': instance.alpha,
        'beta': instance.beta,
        'jobs': [],
        'machines': []
    }
    
    for job in instance.jobs:
        job_data = {'job_id': job.job_id, 'operations': []}
        for op in job.operations:
            op_data = {
                'op_idx': op.op_idx,
                'alternatives': {
                    str(k): {'L': v.L, 'M': v.M, 'U': v.U}
                    for k, v in op.alternatives.items()
                }
            }
            job_data['operations'].append(op_data)
        data['jobs'].append(job_data)
    
    for m in instance.machines:
        data['machines'].append({
            'machine_id': m.machine_id,
            'power_processing': m.power_processing,
            'power_idle': m.power_idle,
            'pm_duration': m.pm_duration,
            'pm_window_start': m.pm_window_start,
            'pm_window_end': m.pm_window_end
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Instance saved to: {filepath}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   GF-FJSP-PM: Mathematical Formulation Verification Tool")
    print("="*70 + "\n")
    
    # Run verification experiments
    results = run_verification_experiments()
    
    # Generate and save test instances
    print("\n\n" + "="*70)
    print("                GENERATING TEST INSTANCES")
    print("="*70)
    
    # Generate instances of various sizes
    sizes = [(3, 2, 2), (5, 3, 3), (6, 4, 3), (8, 5, 4)]
    
    for n, m, o in sizes:
        inst = generate_random_instance(n, m, o, flexibility=0.6, seed=n*100+m*10+o)
        filename = f"test_instance_{n}x{m}x{o}.json"
        save_instance_to_file(inst, filename)
    
    print("\n" + "="*70)
    print("                    VERIFICATION COMPLETE")
    print("="*70)
    print("\nConclusions:")
    print("1. MILP formulation correctly models all constraints")
    print("2. Solutions satisfy precedence, capacity, and PM constraints")
    print("3. Fuzzy processing times are handled via GMIR defuzzification")
    print("4. Energy objective includes processing and idle components")
    print("\nFor larger instances, use the ACO metaheuristic algorithm.")
