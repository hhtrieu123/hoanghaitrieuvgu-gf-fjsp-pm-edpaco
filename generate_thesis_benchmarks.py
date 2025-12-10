#!/usr/bin/env python3
"""
Thesis Benchmark Instance Generator
====================================

Generates benchmark instances for GF-FJSP-PM thesis experiments:
- Category A: Small instances (for MILP comparison)
- Category B: Medium instances (for metaheuristic comparison)
- Category C: Large instances (for scalability test)

Output formats: TXT (for solver) and JSON (for analysis)
"""

import json
import os
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FuzzyTime:
    """Triangular Fuzzy Number (L, M, U)"""
    L: int
    M: int
    U: int
    
    def gmir(self) -> float:
        """Graded Mean Integration Representation"""
        return (self.L + 4*self.M + self.U) / 6


@dataclass
class Operation:
    """Operation with machine alternatives"""
    job_id: int
    op_id: int
    alternatives: Dict[int, FuzzyTime] = field(default_factory=dict)


@dataclass
class Job:
    """Job with sequential operations"""
    job_id: int
    operations: List[Operation] = field(default_factory=list)


@dataclass
class Machine:
    """Machine with energy and PM parameters"""
    machine_id: int
    power_processing: float
    power_idle: float
    pm_duration: float
    pm_window_start: float
    pm_window_end: float


@dataclass
class Instance:
    """Complete GF-FJSP-PM instance"""
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job]
    machines: List[Machine]
    fuzziness: float
    alpha: float
    beta: float
    seed: int
    category: str
    optimal_makespan: Optional[float] = None
    optimal_objective: Optional[float] = None


# =============================================================================
# INSTANCE GENERATOR
# =============================================================================

def generate_instance(
    name: str,
    num_jobs: int,
    num_machines: int,
    ops_per_job: int,
    flexibility: float = 0.5,
    fuzziness: float = 0.2,
    time_range: Tuple[int, int] = (10, 100),
    alpha: float = 0.5,
    beta: float = 0.5,
    seed: int = 42,
    category: str = "medium"
) -> Instance:
    """Generate a GF-FJSP-PM instance."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    jobs = []
    
    for j in range(num_jobs):
        operations = []
        
        for o in range(ops_per_job):
            op = Operation(job_id=j, op_id=o)
            
            # Determine number of eligible machines
            num_eligible = max(1, int(num_machines * flexibility))
            # Ensure some randomness but at least 1 machine
            num_eligible = random.randint(max(1, num_eligible - 1), 
                                          min(num_machines, num_eligible + 1))
            
            eligible_machines = sorted(random.sample(range(num_machines), num_eligible))
            
            for m in eligible_machines:
                # Generate modal processing time
                M = random.randint(time_range[0], time_range[1])
                L = max(1, int(M * (1 - fuzziness)))
                U = int(M * (1 + fuzziness))
                op.alternatives[m] = FuzzyTime(L=L, M=M, U=U)
            
            operations.append(op)
        
        jobs.append(Job(job_id=j, operations=operations))
    
    # Estimate makespan for PM window
    total_time = sum(
        max(ft.M for ft in op.alternatives.values())
        for job in jobs for op in job.operations
    )
    est_makespan = total_time / num_machines * 1.5
    
    # Generate machine parameters
    machines = []
    for m in range(num_machines):
        power_proc = round(random.uniform(5.0, 15.0), 1)
        power_idle = round(power_proc * random.uniform(0.10, 0.25), 1)
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
    
    return Instance(
        name=name,
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs,
        machines=machines,
        fuzziness=fuzziness,
        alpha=alpha,
        beta=beta,
        seed=seed,
        category=category
    )


# =============================================================================
# FILE EXPORT FUNCTIONS
# =============================================================================

def save_txt(instance: Instance, filepath: str):
    """Save instance in TXT format for solver."""
    
    with open(filepath, 'w') as f:
        # Header line
        f.write(f"# GF-FJSP-PM Instance: {instance.name}\n")
        f.write(f"# Category: {instance.category}\n")
        f.write(f"# Seed: {instance.seed}\n")
        f.write(f"# Format: [jobs] [machines] [fuzziness] [alpha] [beta]\n")
        f.write(f"# Job format: [num_ops] [num_machines] [m L M U] ...\n")
        f.write(f"# Machine format: [id] [P_proc] [P_idle] [PM_dur] [PM_start] [PM_end]\n")
        f.write("\n")
        
        # Problem size line
        f.write(f"{instance.num_jobs} {instance.num_machines} {instance.fuzziness} {instance.alpha} {instance.beta}\n")
        
        # Job data
        for job in instance.jobs:
            line_parts = [str(len(job.operations))]
            
            for op in job.operations:
                line_parts.append(str(len(op.alternatives)))
                for m, ft in sorted(op.alternatives.items()):
                    line_parts.extend([str(m), str(ft.L), str(ft.M), str(ft.U)])
            
            f.write(" ".join(line_parts) + "\n")
        
        # Machine data
        f.write("# Machine parameters\n")
        for mach in instance.machines:
            f.write(f"{mach.machine_id} {mach.power_processing} {mach.power_idle} "
                   f"{mach.pm_duration} {mach.pm_window_start} {mach.pm_window_end}\n")


def save_json(instance: Instance, filepath: str):
    """Save instance in JSON format for analysis."""
    
    data = {
        "name": instance.name,
        "category": instance.category,
        "num_jobs": instance.num_jobs,
        "num_machines": instance.num_machines,
        "total_operations": sum(len(j.operations) for j in instance.jobs),
        "fuzziness": instance.fuzziness,
        "alpha": instance.alpha,
        "beta": instance.beta,
        "seed": instance.seed,
        "jobs": [],
        "machines": [],
        "optimal_makespan": instance.optimal_makespan,
        "optimal_objective": instance.optimal_objective
    }
    
    for job in instance.jobs:
        job_data = {
            "job_id": job.job_id,
            "operations": []
        }
        for op in job.operations:
            op_data = {
                "op_id": op.op_id,
                "alternatives": {
                    str(m): {"L": ft.L, "M": ft.M, "U": ft.U, "GMIR": round(ft.gmir(), 2)}
                    for m, ft in op.alternatives.items()
                }
            }
            job_data["operations"].append(op_data)
        data["jobs"].append(job_data)
    
    for mach in instance.machines:
        data["machines"].append({
            "machine_id": mach.machine_id,
            "power_processing": mach.power_processing,
            "power_idle": mach.power_idle,
            "pm_duration": mach.pm_duration,
            "pm_window_start": mach.pm_window_start,
            "pm_window_end": mach.pm_window_end
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# INSTANCE CONFIGURATIONS
# =============================================================================

# Category A: Small instances (MILP comparison)
SMALL_INSTANCES = [
    {"name": "Small_01", "jobs": 3, "machines": 2, "ops": 2, "flex": 0.8, "seed": 42},
    {"name": "Small_02", "jobs": 4, "machines": 3, "ops": 2, "flex": 0.7, "seed": 123},
    {"name": "Small_03", "jobs": 5, "machines": 3, "ops": 3, "flex": 0.6, "seed": 456},
]

# Category B: Medium instances (Metaheuristic comparison)
MEDIUM_INSTANCES = [
    {"name": "Medium_01", "jobs": 10, "machines": 5, "ops": 3, "flex": 0.6, "seed": 1001},
    {"name": "Medium_02", "jobs": 10, "machines": 6, "ops": 4, "flex": 0.5, "seed": 1002},
    {"name": "Medium_03", "jobs": 15, "machines": 5, "ops": 3, "flex": 0.6, "seed": 1003},
    {"name": "Medium_04", "jobs": 15, "machines": 8, "ops": 4, "flex": 0.5, "seed": 1004},
    {"name": "Medium_05", "jobs": 20, "machines": 5, "ops": 3, "flex": 0.6, "seed": 1005},
]

# Category C: Large instances (Scalability test)
LARGE_INSTANCES = [
    {"name": "Large_01", "jobs": 20, "machines": 10, "ops": 5, "flex": 0.5, "seed": 2001},
    {"name": "Large_02", "jobs": 30, "machines": 10, "ops": 5, "flex": 0.5, "seed": 2002},
    {"name": "Large_03", "jobs": 50, "machines": 10, "ops": 5, "flex": 0.4, "seed": 2003},
    {"name": "Large_04", "jobs": 50, "machines": 15, "ops": 6, "flex": 0.4, "seed": 2004},
    {"name": "Large_05", "jobs": 100, "machines": 20, "ops": 5, "flex": 0.3, "seed": 2005},
]


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_all_instances(output_dir: str = "thesis_benchmarks"):
    """Generate all benchmark instances."""
    
    # Create output directories
    os.makedirs(f"{output_dir}/small", exist_ok=True)
    os.makedirs(f"{output_dir}/medium", exist_ok=True)
    os.makedirs(f"{output_dir}/large", exist_ok=True)
    
    all_instances = []
    
    # Generate small instances
    print("\n" + "="*60)
    print("CATEGORY A: Small Instances (MILP Comparison)")
    print("="*60)
    
    for config in SMALL_INSTANCES:
        inst = generate_instance(
            name=config["name"],
            num_jobs=config["jobs"],
            num_machines=config["machines"],
            ops_per_job=config["ops"],
            flexibility=config["flex"],
            seed=config["seed"],
            category="small"
        )
        
        save_txt(inst, f"{output_dir}/small/{config['name']}.txt")
        save_json(inst, f"{output_dir}/small/{config['name']}.json")
        
        total_ops = sum(len(j.operations) for j in inst.jobs)
        avg_flex = sum(len(op.alternatives) for j in inst.jobs for op in j.operations) / total_ops
        
        print(f"  {config['name']}: {config['jobs']}×{config['machines']}×{config['ops']} "
              f"({total_ops} ops, avg flex={avg_flex:.1f})")
        
        all_instances.append(inst)
    
    # Generate medium instances
    print("\n" + "="*60)
    print("CATEGORY B: Medium Instances (Metaheuristic Comparison)")
    print("="*60)
    
    for config in MEDIUM_INSTANCES:
        inst = generate_instance(
            name=config["name"],
            num_jobs=config["jobs"],
            num_machines=config["machines"],
            ops_per_job=config["ops"],
            flexibility=config["flex"],
            seed=config["seed"],
            category="medium"
        )
        
        save_txt(inst, f"{output_dir}/medium/{config['name']}.txt")
        save_json(inst, f"{output_dir}/medium/{config['name']}.json")
        
        total_ops = sum(len(j.operations) for j in inst.jobs)
        avg_flex = sum(len(op.alternatives) for j in inst.jobs for op in j.operations) / total_ops
        
        print(f"  {config['name']}: {config['jobs']}×{config['machines']}×{config['ops']} "
              f"({total_ops} ops, avg flex={avg_flex:.1f})")
        
        all_instances.append(inst)
    
    # Generate large instances
    print("\n" + "="*60)
    print("CATEGORY C: Large Instances (Scalability Test)")
    print("="*60)
    
    for config in LARGE_INSTANCES:
        inst = generate_instance(
            name=config["name"],
            num_jobs=config["jobs"],
            num_machines=config["machines"],
            ops_per_job=config["ops"],
            flexibility=config["flex"],
            seed=config["seed"],
            category="large"
        )
        
        save_txt(inst, f"{output_dir}/large/{config['name']}.txt")
        save_json(inst, f"{output_dir}/large/{config['name']}.json")
        
        total_ops = sum(len(j.operations) for j in inst.jobs)
        avg_flex = sum(len(op.alternatives) for j in inst.jobs for op in j.operations) / total_ops
        
        print(f"  {config['name']}: {config['jobs']}×{config['machines']}×{config['ops']} "
              f"({total_ops} ops, avg flex={avg_flex:.1f})")
        
        all_instances.append(inst)
    
    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total instances generated: {len(all_instances)}")
    print(f"  - Small (MILP):    {len(SMALL_INSTANCES)}")
    print(f"  - Medium (Meta):   {len(MEDIUM_INSTANCES)}")
    print(f"  - Large (Scale):   {len(LARGE_INSTANCES)}")
    print(f"\nOutput directory: {output_dir}/")
    
    return all_instances


def print_instance_details(instance: Instance):
    """Print detailed instance information."""
    
    print(f"\n{'='*60}")
    print(f"Instance: {instance.name}")
    print(f"{'='*60}")
    print(f"Jobs: {instance.num_jobs}, Machines: {instance.num_machines}")
    print(f"Total Operations: {sum(len(j.operations) for j in instance.jobs)}")
    print(f"Fuzziness: {instance.fuzziness*100:.0f}%")
    print(f"Weights: α={instance.alpha}, β={instance.beta}")
    
    print(f"\n--- Processing Times ---")
    for job in instance.jobs:
        print(f"Job {job.job_id}:")
        for op in job.operations:
            alts = ", ".join([f"M{m}:({ft.L},{ft.M},{ft.U})" 
                             for m, ft in sorted(op.alternatives.items())])
            print(f"  Op {op.op_id}: {alts}")
    
    print(f"\n--- Machine Parameters ---")
    print(f"{'ID':<4} {'P_proc':<8} {'P_idle':<8} {'PM_dur':<8} {'PM_window':<15}")
    print("-" * 45)
    for m in instance.machines:
        print(f"{m.machine_id:<4} {m.power_processing:<8.1f} {m.power_idle:<8.1f} "
              f"{m.pm_duration:<8.1f} [{m.pm_window_start:.1f}, {m.pm_window_end:.1f}]")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Generate all instances
    instances = generate_all_instances("thesis_benchmarks")
    
    # Print details for small instances
    print("\n" + "="*60)
    print("DETAILED SMALL INSTANCE DATA")
    print("="*60)
    
    for inst in instances[:3]:  # Small instances
        print_instance_details(inst)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run MILP on small instances to get optimal solutions")
    print("2. Run ACO on all instances")
    print("3. Run GA, PSO, SA on medium instances for comparison")
    print("4. Collect results and create comparison tables")
