"""
==============================================================================
SYNTHETIC DATA GENERATION FOR GF-FJSP-PM
==============================================================================

This module provides:
1. Systematic synthetic data generation methods
2. Validation procedures to ensure data quality
3. Statistical analysis of generated instances
4. Comparison with real benchmark characteristics

For Master's Thesis: Chapter 4 - Experimental Design
==============================================================================
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import csv

# ==============================================================================
# SECTION 1: DATA STRUCTURES
# ==============================================================================

@dataclass
class FuzzyNumber:
    """
    Triangular Fuzzy Number (TFN) representation
    
    A TFN is defined by three parameters:
    - L (Lower): Optimistic estimate (minimum possible value)
    - M (Modal): Most likely value
    - U (Upper): Pessimistic estimate (maximum possible value)
    
    Properties: L ≤ M ≤ U
    
    Reference: Zadeh, L.A. (1965). Fuzzy Sets. Information and Control.
    """
    L: float  # Lower bound (optimistic)
    M: float  # Most likely (modal)
    U: float  # Upper bound (pessimistic)
    
    def __post_init__(self):
        """Validate TFN properties"""
        assert self.L <= self.M <= self.U, f"Invalid TFN: L={self.L}, M={self.M}, U={self.U}"
    
    def defuzzify_gmir(self) -> float:
        """
        Graded Mean Integration Representation (GMIR)
        
        Formula: p̂ = (L + 4M + U) / 6
        
        Reference: Chen, S.H. & Hsieh, C.H. (1999). Graded mean integration 
        representation of generalized fuzzy number.
        """
        return (self.L + 4*self.M + self.U) / 6
    
    def defuzzify_centroid(self) -> float:
        """Centroid method: (L + M + U) / 3"""
        return (self.L + self.M + self.U) / 3
    
    def spread(self) -> float:
        """Measure of uncertainty: U - L"""
        return self.U - self.L
    
    def relative_spread(self) -> float:
        """Relative uncertainty: (U - L) / M"""
        return (self.U - self.L) / self.M if self.M > 0 else 0


@dataclass
class Operation:
    """Operation with multiple machine alternatives"""
    job_id: int
    op_idx: int
    alternatives: Dict[int, FuzzyNumber]  # machine_id -> fuzzy processing time


@dataclass
class Job:
    """Job consisting of sequential operations"""
    job_id: int
    operations: List[Operation]


@dataclass
class Machine:
    """Machine with energy and maintenance parameters"""
    machine_id: int
    power_processing: float  # kW during processing
    power_idle: float        # kW during idle
    pm_duration: float       # Preventive maintenance duration
    pm_window_start: float   # Earliest PM start time
    pm_window_end: float     # Latest PM start time


@dataclass
class SyntheticInstance:
    """Complete synthetic GF-FJSP-PM instance"""
    name: str
    jobs: List[Job]
    machines: List[Machine]
    
    # Generation parameters (for reproducibility)
    seed: int = 0
    generation_method: str = ""
    parameters: Dict = field(default_factory=dict)
    
    @property
    def num_jobs(self) -> int:
        return len(self.jobs)
    
    @property
    def num_machines(self) -> int:
        return len(self.machines)
    
    @property
    def total_ops(self) -> int:
        return sum(len(j.operations) for j in self.jobs)
    
    @property
    def avg_ops_per_job(self) -> float:
        return self.total_ops / self.num_jobs if self.num_jobs > 0 else 0
    
    @property
    def avg_flexibility(self) -> float:
        """Average number of machine alternatives per operation"""
        total_alts = sum(len(op.alternatives) for j in self.jobs for op in j.operations)
        return total_alts / self.total_ops if self.total_ops > 0 else 0


# ==============================================================================
# SECTION 2: SYNTHETIC DATA GENERATION METHODS
# ==============================================================================

class SyntheticDataGenerator:
    """
    Synthetic Data Generator for GF-FJSP-PM
    
    Generates instances with controllable characteristics:
    - Problem size (jobs, machines, operations)
    - Flexibility level (machine alternatives)
    - Processing time distribution
    - Uncertainty level (fuzziness)
    - Energy parameters
    - Maintenance windows
    
    Methods follow established practices from scheduling literature:
    - Taillard (1993) for JSP generation
    - Brandimarte (1993) for FJSP generation
    - Dauzère-Pérès et al. (2024) for FJSP benchmarks
    """
    
    def __init__(self, seed: int = None):
        """Initialize generator with optional seed for reproducibility"""
        self.seed = seed if seed is not None else np.random.randint(0, 100000)
        np.random.seed(self.seed)
        self.generation_log = []
    
    def generate_instance(
        self,
        num_jobs: int,
        num_machines: int,
        ops_per_job: int | Tuple[int, int],  # Fixed or (min, max)
        flexibility: float = 0.5,             # Fraction of machines per operation
        time_distribution: str = 'uniform',   # 'uniform', 'normal', 'exponential'
        time_params: Dict = None,             # Distribution parameters
        fuzziness: float = 0.2,               # Relative uncertainty level
        fuzziness_type: str = 'symmetric',    # 'symmetric', 'right_skewed', 'left_skewed'
        energy_params: Dict = None,           # Energy consumption parameters
        pm_params: Dict = None,               # Maintenance parameters
        name: str = None
    ) -> SyntheticInstance:
        """
        Generate a complete GF-FJSP-PM instance
        
        Parameters:
        -----------
        num_jobs : int
            Number of jobs (n)
        num_machines : int
            Number of machines (m)
        ops_per_job : int or (int, int)
            Operations per job. If tuple, randomly sampled from [min, max]
        flexibility : float (0, 1]
            Average fraction of machines each operation can use
            - 1.0 = fully flexible (can use any machine)
            - 1/m = no flexibility (JSP case)
        time_distribution : str
            Distribution for modal processing times
        time_params : dict
            Parameters for time distribution (default: {'min': 10, 'max': 100})
        fuzziness : float
            Relative uncertainty level (e.g., 0.2 = ±20%)
        fuzziness_type : str
            How uncertainty is distributed around modal value
        energy_params : dict
            Machine energy parameters
        pm_params : dict
            Preventive maintenance parameters
        name : str
            Instance name (auto-generated if None)
        
        Returns:
        --------
        SyntheticInstance : Complete problem instance
        """
        
        # Default parameters
        if time_params is None:
            time_params = {'min': 10, 'max': 100}
        if energy_params is None:
            energy_params = {
                'power_proc_range': (5, 15),    # kW
                'power_idle_range': (1, 3),      # kW
            }
        if pm_params is None:
            pm_params = {
                'duration_range': (10, 30),      # Time units
                'window_start_ratio': 0.2,       # Start at 20% of estimated makespan
                'window_end_ratio': 0.7,         # End at 70% of estimated makespan
            }
        
        # Generate name
        if name is None:
            name = f"Synth_{num_jobs}x{num_machines}_s{self.seed}"
        
        # Step 1: Generate jobs and operations
        jobs = self._generate_jobs(
            num_jobs, num_machines, ops_per_job, flexibility,
            time_distribution, time_params, fuzziness, fuzziness_type
        )
        
        # Step 2: Estimate makespan for PM window calculation
        estimated_makespan = self._estimate_makespan(jobs, num_machines)
        
        # Step 3: Generate machines with energy and PM parameters
        machines = self._generate_machines(
            num_machines, energy_params, pm_params, estimated_makespan
        )
        
        # Create instance
        instance = SyntheticInstance(
            name=name,
            jobs=jobs,
            machines=machines,
            seed=self.seed,
            generation_method='controlled_random',
            parameters={
                'num_jobs': num_jobs,
                'num_machines': num_machines,
                'ops_per_job': ops_per_job,
                'flexibility': flexibility,
                'time_distribution': time_distribution,
                'time_params': time_params,
                'fuzziness': fuzziness,
                'fuzziness_type': fuzziness_type,
                'energy_params': energy_params,
                'pm_params': pm_params
            }
        )
        
        # Log generation
        self.generation_log.append({
            'name': name,
            'seed': self.seed,
            'size': f"{num_jobs}x{num_machines}x{instance.avg_ops_per_job:.1f}",
            'total_ops': instance.total_ops,
            'avg_flexibility': instance.avg_flexibility
        })
        
        return instance
    
    def _generate_jobs(
        self, num_jobs, num_machines, ops_per_job, flexibility,
        time_distribution, time_params, fuzziness, fuzziness_type
    ) -> List[Job]:
        """Generate all jobs with operations"""
        
        jobs = []
        
        for i in range(num_jobs):
            # Determine number of operations for this job
            if isinstance(ops_per_job, tuple):
                n_ops = np.random.randint(ops_per_job[0], ops_per_job[1] + 1)
            else:
                n_ops = ops_per_job
            
            operations = []
            for j in range(n_ops):
                # Determine eligible machines
                num_eligible = max(1, int(np.ceil(num_machines * flexibility)))
                # Add some randomness to flexibility
                num_eligible = min(num_machines, max(1, 
                    num_eligible + np.random.randint(-1, 2)))
                
                eligible_machines = np.random.choice(
                    num_machines, num_eligible, replace=False
                )
                
                # Generate processing times for each machine
                alternatives = {}
                for m in eligible_machines:
                    # Generate modal processing time
                    modal_time = self._generate_processing_time(
                        time_distribution, time_params
                    )
                    
                    # Generate fuzzy processing time
                    fuzzy_time = self._generate_fuzzy_time(
                        modal_time, fuzziness, fuzziness_type
                    )
                    
                    alternatives[int(m)] = fuzzy_time
                
                operations.append(Operation(
                    job_id=i, op_idx=j, alternatives=alternatives
                ))
            
            jobs.append(Job(job_id=i, operations=operations))
        
        return jobs
    
    def _generate_processing_time(self, distribution: str, params: Dict) -> float:
        """Generate a single modal processing time"""
        
        if distribution == 'uniform':
            return np.random.randint(params['min'], params['max'] + 1)
        
        elif distribution == 'normal':
            mean = params.get('mean', (params['min'] + params['max']) / 2)
            std = params.get('std', (params['max'] - params['min']) / 6)
            value = np.random.normal(mean, std)
            return max(params['min'], min(params['max'], int(value)))
        
        elif distribution == 'exponential':
            scale = params.get('scale', (params['min'] + params['max']) / 2)
            value = np.random.exponential(scale)
            return max(params['min'], min(params['max'], int(value)))
        
        else:
            # Default to uniform
            return np.random.randint(params['min'], params['max'] + 1)
    
    def _generate_fuzzy_time(
        self, modal: float, fuzziness: float, fuzziness_type: str
    ) -> FuzzyNumber:
        """
        Generate triangular fuzzy number from modal value
        
        Fuzziness types:
        - symmetric: L = M - δ, U = M + δ where δ = M * fuzziness
        - right_skewed: More uncertainty on upper side (pessimistic)
        - left_skewed: More uncertainty on lower side (optimistic)
        """
        
        if fuzziness_type == 'symmetric':
            delta = modal * fuzziness
            L = max(1, int(modal - delta))
            U = int(modal + delta)
        
        elif fuzziness_type == 'right_skewed':
            # More uncertainty upward (pessimistic scenarios)
            delta_l = modal * fuzziness * 0.5
            delta_u = modal * fuzziness * 1.5
            L = max(1, int(modal - delta_l))
            U = int(modal + delta_u)
        
        elif fuzziness_type == 'left_skewed':
            # More uncertainty downward (optimistic scenarios)
            delta_l = modal * fuzziness * 1.5
            delta_u = modal * fuzziness * 0.5
            L = max(1, int(modal - delta_l))
            U = int(modal + delta_u)
        
        else:
            # Default symmetric
            delta = modal * fuzziness
            L = max(1, int(modal - delta))
            U = int(modal + delta)
        
        return FuzzyNumber(L=L, M=int(modal), U=U)
    
    def _estimate_makespan(self, jobs: List[Job], num_machines: int) -> float:
        """Estimate makespan for PM window calculation"""
        
        total_work = 0
        for job in jobs:
            job_time = 0
            for op in job.operations:
                # Use average processing time across alternatives
                avg_time = np.mean([
                    ft.defuzzify_gmir() for ft in op.alternatives.values()
                ])
                job_time += avg_time
            total_work += job_time
        
        # Rough estimate: total work / machines (lower bound)
        # Multiply by factor for more realistic estimate
        return (total_work / num_machines) * 1.5
    
    def _generate_machines(
        self, num_machines: int, energy_params: Dict, 
        pm_params: Dict, estimated_makespan: float
    ) -> List[Machine]:
        """Generate machines with energy and PM parameters"""
        
        machines = []
        
        for k in range(num_machines):
            # Energy parameters
            power_proc = np.random.uniform(*energy_params['power_proc_range'])
            power_idle = np.random.uniform(*energy_params['power_idle_range'])
            
            # PM parameters
            pm_duration = np.random.uniform(*pm_params['duration_range'])
            pm_window_start = estimated_makespan * pm_params['window_start_ratio']
            pm_window_end = estimated_makespan * pm_params['window_end_ratio']
            
            machines.append(Machine(
                machine_id=k,
                power_processing=round(power_proc, 2),
                power_idle=round(power_idle, 2),
                pm_duration=round(pm_duration, 2),
                pm_window_start=round(pm_window_start, 2),
                pm_window_end=round(pm_window_end, 2)
            ))
        
        return machines


# ==============================================================================
# SECTION 3: DATA VALIDATION
# ==============================================================================

class DataValidator:
    """
    Validates synthetic instances for correctness and realism
    
    Validation checks:
    1. Structural validity (correct format, no missing data)
    2. Constraint satisfaction (L ≤ M ≤ U for fuzzy numbers)
    3. Parameter bounds (positive values, valid ranges)
    4. Statistical properties (similar to real benchmarks)
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_instance(self, instance: SyntheticInstance) -> Dict:
        """
        Perform comprehensive validation on an instance
        
        Returns:
        --------
        Dict with validation results including:
        - is_valid: Overall validity
        - checks: Individual check results
        - warnings: Non-critical issues
        - statistics: Instance statistics
        """
        
        checks = {}
        warnings = []
        errors = []
        
        # Check 1: Structural validity
        checks['structure'] = self._check_structure(instance, errors, warnings)
        
        # Check 2: Fuzzy number validity
        checks['fuzzy_validity'] = self._check_fuzzy_numbers(instance, errors, warnings)
        
        # Check 3: Parameter bounds
        checks['parameter_bounds'] = self._check_parameter_bounds(instance, errors, warnings)
        
        # Check 4: Flexibility consistency
        checks['flexibility'] = self._check_flexibility(instance, errors, warnings)
        
        # Check 5: PM window validity
        checks['pm_windows'] = self._check_pm_windows(instance, errors, warnings)
        
        # Calculate statistics
        statistics = self._calculate_statistics(instance)
        
        # Overall result
        is_valid = len(errors) == 0
        
        result = {
            'instance_name': instance.name,
            'is_valid': is_valid,
            'checks': checks,
            'errors': errors,
            'warnings': warnings,
            'statistics': statistics
        }
        
        self.validation_results.append(result)
        return result
    
    def _check_structure(self, instance, errors, warnings) -> bool:
        """Check structural validity"""
        
        valid = True
        
        # Check jobs exist
        if not instance.jobs:
            errors.append("No jobs defined")
            valid = False
        
        # Check machines exist
        if not instance.machines:
            errors.append("No machines defined")
            valid = False
        
        # Check each job has operations
        for job in instance.jobs:
            if not job.operations:
                errors.append(f"Job {job.job_id} has no operations")
                valid = False
            
            # Check operation indices are sequential
            for idx, op in enumerate(job.operations):
                if op.op_idx != idx:
                    warnings.append(f"Job {job.job_id}: operation index mismatch at position {idx}")
                
                # Check alternatives exist
                if not op.alternatives:
                    errors.append(f"Operation ({job.job_id}, {op.op_idx}) has no machine alternatives")
                    valid = False
        
        return valid
    
    def _check_fuzzy_numbers(self, instance, errors, warnings) -> bool:
        """Check all fuzzy numbers satisfy L ≤ M ≤ U"""
        
        valid = True
        
        for job in instance.jobs:
            for op in job.operations:
                for m_id, fuzzy in op.alternatives.items():
                    if not (fuzzy.L <= fuzzy.M <= fuzzy.U):
                        errors.append(
                            f"Invalid TFN at ({job.job_id}, {op.op_idx}, {m_id}): "
                            f"L={fuzzy.L}, M={fuzzy.M}, U={fuzzy.U}"
                        )
                        valid = False
                    
                    if fuzzy.L <= 0:
                        errors.append(
                            f"Non-positive lower bound at ({job.job_id}, {op.op_idx}, {m_id}): L={fuzzy.L}"
                        )
                        valid = False
        
        return valid
    
    def _check_parameter_bounds(self, instance, errors, warnings) -> bool:
        """Check parameter values are within reasonable bounds"""
        
        valid = True
        
        for m in instance.machines:
            # Power must be positive
            if m.power_processing <= 0:
                errors.append(f"Machine {m.machine_id}: non-positive processing power")
                valid = False
            
            if m.power_idle < 0:
                errors.append(f"Machine {m.machine_id}: negative idle power")
                valid = False
            
            # Idle power should be less than processing power
            if m.power_idle >= m.power_processing:
                warnings.append(f"Machine {m.machine_id}: idle power >= processing power")
            
            # PM duration must be positive
            if m.pm_duration <= 0:
                errors.append(f"Machine {m.machine_id}: non-positive PM duration")
                valid = False
        
        return valid
    
    def _check_flexibility(self, instance, errors, warnings) -> bool:
        """Check flexibility is consistent"""
        
        valid = True
        
        for job in instance.jobs:
            for op in job.operations:
                # Check machine IDs are valid
                for m_id in op.alternatives.keys():
                    if m_id < 0 or m_id >= instance.num_machines:
                        errors.append(
                            f"Invalid machine ID {m_id} at ({job.job_id}, {op.op_idx})"
                        )
                        valid = False
        
        return valid
    
    def _check_pm_windows(self, instance, errors, warnings) -> bool:
        """Check PM windows are valid"""
        
        valid = True
        
        for m in instance.machines:
            if m.pm_window_start < 0:
                errors.append(f"Machine {m.machine_id}: negative PM window start")
                valid = False
            
            if m.pm_window_end <= m.pm_window_start:
                errors.append(
                    f"Machine {m.machine_id}: PM window end <= start "
                    f"({m.pm_window_end} <= {m.pm_window_start})"
                )
                valid = False
            
            # Window must be large enough for PM
            window_size = m.pm_window_end - m.pm_window_start
            if window_size < m.pm_duration:
                errors.append(
                    f"Machine {m.machine_id}: PM window too small "
                    f"(window={window_size:.2f}, duration={m.pm_duration:.2f})"
                )
                valid = False
        
        return valid
    
    def _calculate_statistics(self, instance) -> Dict:
        """Calculate instance statistics"""
        
        # Processing time statistics
        all_times = []
        all_spreads = []
        flexibilities = []
        
        for job in instance.jobs:
            for op in job.operations:
                flexibilities.append(len(op.alternatives))
                for fuzzy in op.alternatives.values():
                    all_times.append(fuzzy.M)
                    all_spreads.append(fuzzy.relative_spread())
        
        # Machine statistics
        proc_powers = [m.power_processing for m in instance.machines]
        idle_powers = [m.power_idle for m in instance.machines]
        pm_durations = [m.pm_duration for m in instance.machines]
        
        return {
            'num_jobs': instance.num_jobs,
            'num_machines': instance.num_machines,
            'total_operations': instance.total_ops,
            'avg_ops_per_job': instance.avg_ops_per_job,
            'avg_flexibility': np.mean(flexibilities),
            'flexibility_std': np.std(flexibilities),
            'processing_time': {
                'min': min(all_times),
                'max': max(all_times),
                'mean': np.mean(all_times),
                'std': np.std(all_times)
            },
            'fuzziness': {
                'mean_relative_spread': np.mean(all_spreads),
                'std_relative_spread': np.std(all_spreads)
            },
            'energy': {
                'mean_proc_power': np.mean(proc_powers),
                'mean_idle_power': np.mean(idle_powers),
                'idle_to_proc_ratio': np.mean(idle_powers) / np.mean(proc_powers)
            },
            'maintenance': {
                'mean_pm_duration': np.mean(pm_durations)
            }
        }
    
    def print_validation_report(self, result: Dict):
        """Print formatted validation report"""
        
        print("\n" + "=" * 70)
        print(f"VALIDATION REPORT: {result['instance_name']}")
        print("=" * 70)
        
        # Overall status
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        print(f"\nOverall Status: {status}")
        
        # Individual checks
        print("\nValidation Checks:")
        print("-" * 40)
        for check_name, passed in result['checks'].items():
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {check_name}")
        
        # Errors
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for err in result['errors']:
                print(f"  ✗ {err}")
        
        # Warnings
        if result['warnings']:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warn in result['warnings']:
                print(f"  ⚠ {warn}")
        
        # Statistics
        stats = result['statistics']
        print("\nInstance Statistics:")
        print("-" * 40)
        print(f"  Size: {stats['num_jobs']} jobs × {stats['num_machines']} machines")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  Avg operations/job: {stats['avg_ops_per_job']:.2f}")
        print(f"  Avg flexibility: {stats['avg_flexibility']:.2f} machines/operation")
        print(f"  Processing time range: [{stats['processing_time']['min']}, {stats['processing_time']['max']}]")
        print(f"  Processing time mean: {stats['processing_time']['mean']:.2f} ± {stats['processing_time']['std']:.2f}")
        print(f"  Mean fuzziness (relative spread): {stats['fuzziness']['mean_relative_spread']:.2%}")


# ==============================================================================
# SECTION 4: COMPARISON WITH REAL BENCHMARKS
# ==============================================================================

class BenchmarkComparison:
    """
    Compare synthetic instances with real benchmark characteristics
    
    Reference benchmarks:
    - Brandimarte (1993): Mk01-Mk10 - FJSP
    - Lawrence (1984): LA01-LA40 - JSP
    """
    
    # Characteristics of real benchmarks
    BRANDIMARTE_STATS = {
        'Mk01': {'jobs': 10, 'machines': 6, 'ops': 55, 'avg_flex': 2.09},
        'Mk02': {'jobs': 10, 'machines': 6, 'ops': 58, 'avg_flex': 4.10},
        'Mk03': {'jobs': 15, 'machines': 8, 'ops': 150, 'avg_flex': 3.01},
        'Mk04': {'jobs': 15, 'machines': 8, 'ops': 90, 'avg_flex': 1.91},
        'Mk05': {'jobs': 15, 'machines': 4, 'ops': 106, 'avg_flex': 1.71},
        'Mk06': {'jobs': 10, 'machines': 15, 'ops': 150, 'avg_flex': 3.28},
        'Mk07': {'jobs': 20, 'machines': 5, 'ops': 100, 'avg_flex': 2.83},
        'Mk08': {'jobs': 20, 'machines': 10, 'ops': 225, 'avg_flex': 1.43},
        'Mk09': {'jobs': 20, 'machines': 10, 'ops': 240, 'avg_flex': 2.53},
        'Mk10': {'jobs': 20, 'machines': 15, 'ops': 240, 'avg_flex': 2.98},
    }
    
    LAWRENCE_STATS = {
        'LA01-LA05': {'jobs': 10, 'machines': 5, 'ops_per_job': 5, 'flexibility': 1.0},
        'LA06-LA10': {'jobs': 15, 'machines': 5, 'ops_per_job': 5, 'flexibility': 1.0},
        'LA11-LA15': {'jobs': 20, 'machines': 5, 'ops_per_job': 5, 'flexibility': 1.0},
        'LA16-LA20': {'jobs': 10, 'machines': 10, 'ops_per_job': 10, 'flexibility': 1.0},
        'LA21-LA25': {'jobs': 15, 'machines': 10, 'ops_per_job': 10, 'flexibility': 1.0},
        'LA26-LA30': {'jobs': 20, 'machines': 10, 'ops_per_job': 10, 'flexibility': 1.0},
        'LA31-LA35': {'jobs': 30, 'machines': 10, 'ops_per_job': 10, 'flexibility': 1.0},
        'LA36-LA40': {'jobs': 15, 'machines': 15, 'ops_per_job': 15, 'flexibility': 1.0},
    }
    
    @classmethod
    def compare_with_brandimarte(cls, instance: SyntheticInstance) -> Dict:
        """Compare instance characteristics with Brandimarte benchmarks"""
        
        # Calculate instance stats
        inst_stats = {
            'jobs': instance.num_jobs,
            'machines': instance.num_machines,
            'ops': instance.total_ops,
            'avg_flex': instance.avg_flexibility
        }
        
        # Find most similar Brandimarte instance
        best_match = None
        best_score = float('inf')
        
        for name, bm_stats in cls.BRANDIMARTE_STATS.items():
            # Normalized difference score
            score = (
                abs(inst_stats['jobs'] - bm_stats['jobs']) / 20 +
                abs(inst_stats['machines'] - bm_stats['machines']) / 15 +
                abs(inst_stats['ops'] - bm_stats['ops']) / 240 +
                abs(inst_stats['avg_flex'] - bm_stats['avg_flex']) / 4
            )
            
            if score < best_score:
                best_score = score
                best_match = name
        
        return {
            'instance_stats': inst_stats,
            'most_similar': best_match,
            'similarity_score': 1 - min(1, best_score / 4),  # 0-1 scale
            'brandimarte_stats': cls.BRANDIMARTE_STATS[best_match]
        }


# ==============================================================================
# SECTION 5: SAVE/LOAD FUNCTIONS
# ==============================================================================

def save_instance_json(instance: SyntheticInstance, filepath: str):
    """Save instance to JSON format"""
    
    data = {
        'name': instance.name,
        'seed': instance.seed,
        'generation_method': instance.generation_method,
        'parameters': instance.parameters,
        'num_jobs': instance.num_jobs,
        'num_machines': instance.num_machines,
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
    
    return filepath


def save_instance_txt(instance: SyntheticInstance, filepath: str):
    """Save instance in Brandimarte-like text format (for ACO solver)"""
    
    with open(filepath, 'w') as f:
        # First line: num_jobs num_machines
        f.write(f"{instance.num_jobs} {instance.num_machines}\n")
        
        # Each job
        for job in instance.jobs:
            line_parts = [str(len(job.operations))]  # num_ops
            
            for op in job.operations:
                line_parts.append(str(len(op.alternatives)))  # num_alternatives
                for m_id, fuzzy in op.alternatives.items():
                    # Use GMIR defuzzified value for deterministic solver
                    line_parts.append(str(m_id + 1))  # 1-based machine ID
                    line_parts.append(str(int(fuzzy.defuzzify_gmir())))
            
            f.write(' '.join(line_parts) + '\n')
    
    return filepath


# ==============================================================================
# SECTION 6: MAIN - DEMONSTRATION
# ==============================================================================

def main():
    """Demonstrate synthetic data generation and validation"""
    
    print("=" * 70)
    print("     SYNTHETIC DATA GENERATION FOR GF-FJSP-PM")
    print("     Demonstration and Validation")
    print("=" * 70)
    
    # Initialize generator with seed for reproducibility
    generator = SyntheticDataGenerator(seed=42)
    validator = DataValidator()
    
    # ==========================================================================
    # Generate instances with different characteristics
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("GENERATING TEST INSTANCES")
    print("-" * 70)
    
    instances = []
    
    # Instance 1: Small, high flexibility
    inst1 = generator.generate_instance(
        num_jobs=5, num_machines=3, ops_per_job=3,
        flexibility=0.8,
        time_distribution='uniform',
        time_params={'min': 10, 'max': 50},
        fuzziness=0.15,
        name="Small_HighFlex"
    )
    instances.append(inst1)
    print(f"Generated: {inst1.name} ({inst1.num_jobs}×{inst1.num_machines}, flex={inst1.avg_flexibility:.2f})")
    
    # Instance 2: Medium, moderate flexibility
    inst2 = generator.generate_instance(
        num_jobs=10, num_machines=5, ops_per_job=5,
        flexibility=0.5,
        time_distribution='normal',
        time_params={'min': 10, 'max': 100, 'mean': 50, 'std': 20},
        fuzziness=0.20,
        name="Medium_ModFlex"
    )
    instances.append(inst2)
    print(f"Generated: {inst2.name} ({inst2.num_jobs}×{inst2.num_machines}, flex={inst2.avg_flexibility:.2f})")
    
    # Instance 3: Medium, low flexibility (JSP-like)
    inst3 = generator.generate_instance(
        num_jobs=10, num_machines=5, ops_per_job=5,
        flexibility=0.25,
        time_distribution='uniform',
        time_params={'min': 20, 'max': 80},
        fuzziness=0.25,
        fuzziness_type='right_skewed',
        name="Medium_LowFlex"
    )
    instances.append(inst3)
    print(f"Generated: {inst3.name} ({inst3.num_jobs}×{inst3.num_machines}, flex={inst3.avg_flexibility:.2f})")
    
    # Instance 4: Large instance
    inst4 = generator.generate_instance(
        num_jobs=15, num_machines=8, ops_per_job=(4, 6),
        flexibility=0.4,
        time_distribution='uniform',
        time_params={'min': 10, 'max': 100},
        fuzziness=0.20,
        name="Large_VarOps"
    )
    instances.append(inst4)
    print(f"Generated: {inst4.name} ({inst4.num_jobs}×{inst4.num_machines}, flex={inst4.avg_flexibility:.2f})")
    
    # ==========================================================================
    # Validate all instances
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("VALIDATING INSTANCES")
    print("-" * 70)
    
    all_valid = True
    for inst in instances:
        result = validator.validate_instance(inst)
        validator.print_validation_report(result)
        if not result['is_valid']:
            all_valid = False
    
    # ==========================================================================
    # Compare with real benchmarks
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("COMPARISON WITH REAL BENCHMARKS")
    print("-" * 70)
    
    for inst in instances:
        comparison = BenchmarkComparison.compare_with_brandimarte(inst)
        print(f"\n{inst.name}:")
        print(f"  Most similar to: {comparison['most_similar']}")
        print(f"  Similarity score: {comparison['similarity_score']:.2%}")
        print(f"  Instance: {comparison['instance_stats']}")
        print(f"  Benchmark: {comparison['brandimarte_stats']}")
    
    # ==========================================================================
    # Save instances
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("SAVING INSTANCES")
    print("-" * 70)
    
    os.makedirs('synthetic_instances', exist_ok=True)
    
    for inst in instances:
        # Save as JSON (full information)
        json_path = save_instance_json(inst, f'synthetic_instances/{inst.name}.json')
        print(f"Saved: {json_path}")
        
        # Save as TXT (for ACO solver)
        txt_path = save_instance_txt(inst, f'synthetic_instances/{inst.name}.txt')
        print(f"Saved: {txt_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nGenerated {len(instances)} instances")
    print(f"All valid: {'✓ Yes' if all_valid else '✗ No'}")
    
    print("\nInstance Summary Table:")
    print("-" * 80)
    print(f"{'Name':<20} {'Size':<12} {'Ops':<8} {'Flex':<8} {'Fuzz':<8} {'Valid':<8}")
    print("-" * 80)
    
    for inst, result in zip(instances, validator.validation_results):
        size = f"{inst.num_jobs}×{inst.num_machines}"
        valid = "✓" if result['is_valid'] else "✗"
        fuzz = result['statistics']['fuzziness']['mean_relative_spread']
        print(f"{inst.name:<20} {size:<12} {inst.total_ops:<8} {inst.avg_flexibility:<8.2f} {fuzz:<8.2%} {valid:<8}")
    
    print("\n" + "=" * 70)
    print("For thesis: Use the validation report to argue that synthetic data is valid")
    print("=" * 70)


def solve_synthetic_instances():
    """
    Generate synthetic instances and solve them with MILP
    Shows complete solution results
    """
    
    print("=" * 70)
    print("     SYNTHETIC INSTANCE GENERATION AND SOLUTION")
    print("=" * 70)
    
    # Import MILP solver (assuming milp_verification.py is in same directory)
    try:
        from milp_verification import GF_FJSP_PM_MILP, GF_FJSP_PM_Instance as MILPInstance
        from milp_verification import Job as MILPJob, Operation as MILPOp, Machine as MILPMachine
        from milp_verification import FuzzyNumber as MILPFuzzy
        milp_available = True
    except ImportError:
        print("Note: milp_verification.py not found. Only validation will be shown.")
        milp_available = False
    
    # Initialize generator and validator
    generator = SyntheticDataGenerator(seed=42)
    validator = DataValidator()
    
    # Define test configurations
    test_configs = [
        {'name': 'Small_3x2', 'n': 3, 'm': 2, 'o': 2, 'f': 0.8, 'd': 0.15},
        {'name': 'Small_4x3', 'n': 4, 'm': 3, 'o': 2, 'f': 0.7, 'd': 0.20},
        {'name': 'Medium_5x3', 'n': 5, 'm': 3, 'o': 3, 'f': 0.6, 'd': 0.20},
        {'name': 'Medium_6x4', 'n': 6, 'm': 4, 'o': 3, 'f': 0.5, 'd': 0.20},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"INSTANCE: {config['name']}")
        print(f"{'='*70}")
        
        # Generate instance
        instance = generator.generate_instance(
            num_jobs=config['n'],
            num_machines=config['m'],
            ops_per_job=config['o'],
            flexibility=config['f'],
            fuzziness=config['d'],
            name=config['name']
        )
        
        # Validate
        validation = validator.validate_instance(instance)
        
        print(f"\n--- Instance Properties ---")
        print(f"  Jobs: {instance.num_jobs}")
        print(f"  Machines: {instance.num_machines}")
        print(f"  Total Operations: {instance.total_ops}")
        print(f"  Avg Flexibility: {instance.avg_flexibility:.2f} machines/operation")
        print(f"  Fuzziness: {config['d']*100:.0f}%")
        print(f"  Validation: {'✓ VALID' if validation['is_valid'] else '✗ INVALID'}")
        
        # Print instance details
        print(f"\n--- Job Details ---")
        for job in instance.jobs:
            print(f"  Job {job.job_id}: {len(job.operations)} operations")
            for op in job.operations:
                machines_str = ", ".join([
                    f"M{k}:({ft.L},{ft.M},{ft.U})" 
                    for k, ft in op.alternatives.items()
                ])
                print(f"    Op {op.op_idx}: {machines_str}")
        
        print(f"\n--- Machine Parameters ---")
        for m in instance.machines:
            print(f"  Machine {m.machine_id}: P_proc={m.power_processing:.1f}kW, "
                  f"P_idle={m.power_idle:.1f}kW, PM_dur={m.pm_duration:.1f}, "
                  f"PM_window=[{m.pm_window_start:.1f}, {m.pm_window_end:.1f}]")
        
        # Solve with MILP if available
        if milp_available:
            print(f"\n--- Solving with MILP ---")
            
            # Convert to MILP format
            milp_jobs = []
            for job in instance.jobs:
                milp_ops = []
                for op in job.operations:
                    milp_alts = {
                        k: MILPFuzzy(L=ft.L, M=ft.M, U=ft.U)
                        for k, ft in op.alternatives.items()
                    }
                    milp_ops.append(MILPOp(
                        job_id=op.job_id,
                        op_idx=op.op_idx,
                        alternatives=milp_alts
                    ))
                milp_jobs.append(MILPJob(job_id=job.job_id, operations=milp_ops))
            
            milp_machines = [
                MILPMachine(
                    machine_id=m.machine_id,
                    power_processing=m.power_processing,
                    power_idle=m.power_idle,
                    pm_duration=m.pm_duration,
                    pm_window_start=m.pm_window_start,
                    pm_window_end=m.pm_window_end
                )
                for m in instance.machines
            ]
            
            milp_instance = MILPInstance(
                name=instance.name,
                jobs=milp_jobs,
                machines=milp_machines,
                alpha=0.5,
                beta=0.5
            )
            
            # Solve
            milp_solver = GF_FJSP_PM_MILP(milp_instance)
            milp_solver.build_model()
            solution = milp_solver.solve(time_limit=120, verbose=False)
            
            print(f"  Status: {solution['status']}")
            print(f"  Objective: {solution['objective']:.2f}")
            print(f"  Makespan: {solution['makespan']:.2f}")
            print(f"  Solve Time: {solution['solve_time']:.2f} seconds")
            
            # Print schedule
            if solution['status'] in ['Optimal', 'Feasible']:
                print(f"\n--- Optimal Schedule ---")
                print(f"  {'Job':<6} {'Op':<6} {'Machine':<10} {'Start':<12} {'End':<12}")
                print(f"  {'-'*50}")
                
                schedule = []
                for (i, j), k in solution['assignments'].items():
                    schedule.append({
                        'job': i, 'op': j, 'machine': k,
                        'start': solution['start_times'][(i,j)],
                        'end': solution['completion_times'][(i,j)]
                    })
                
                schedule.sort(key=lambda x: (x['start'], x['job'], x['op']))
                for item in schedule:
                    print(f"  {item['job']:<6} {item['op']:<6} {item['machine']:<10} "
                          f"{item['start']:<12.2f} {item['end']:<12.2f}")
                
                print(f"\n--- PM Schedule ---")
                print(f"  {'Machine':<10} {'PM Start':<12} {'PM End':<12}")
                print(f"  {'-'*35}")
                for m in instance.machines:
                    k = m.machine_id
                    pm_start = solution['pm_times'].get(k, 0)
                    pm_end = pm_start + m.pm_duration
                    print(f"  {k:<10} {pm_start:<12.2f} {pm_end:<12.2f}")
                
                # Verify solution
                verification = milp_solver.verify_solution()
                print(f"\n--- Constraint Verification ---")
                if verification['valid']:
                    print(f"  ✓ All constraints satisfied!")
                else:
                    print(f"  ✗ Constraint violations found:")
                    for err in verification['errors']:
                        print(f"    - {err}")
                
                # Calculate energy
                E_proc = sum(
                    instance.machines[k].power_processing * 
                    instance.jobs[i].operations[j].alternatives[k].defuzzify_gmir()
                    for (i, j), k in solution['assignments'].items()
                )
                E_idle = sum(
                    instance.machines[m.machine_id].power_idle * solution['idle_times'].get(m.machine_id, 0)
                    for m in instance.machines
                )
                print(f"\n--- Energy Consumption ---")
                print(f"  Processing Energy: {E_proc:.2f} kWh")
                print(f"  Idle Energy: {E_idle:.2f} kWh")
                print(f"  Total Energy: {E_proc + E_idle:.2f} kWh")
                
                results.append({
                    'name': config['name'],
                    'size': f"{config['n']}×{config['m']}×{config['o']}",
                    'status': solution['status'],
                    'makespan': solution['makespan'],
                    'energy': E_proc + E_idle,
                    'objective': solution['objective'],
                    'time': solution['solve_time'],
                    'valid': verification['valid']
                })
    
    # Print summary table
    if results:
        print(f"\n\n{'='*80}")
        print(f"                        RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Instance':<15} {'Size':<12} {'Status':<10} {'Makespan':<12} {'Energy':<12} {'Objective':<12} {'Time(s)':<10} {'Valid':<8}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['name']:<15} {r['size']:<12} {r['status']:<10} {r['makespan']:<12.2f} "
                  f"{r['energy']:<12.2f} {r['objective']:<12.2f} {r['time']:<10.2f} {'✓' if r['valid'] else '✗':<8}")
        print(f"{'='*80}")
        
        # Statistics
        optimal_count = sum(1 for r in results if r['status'] == 'Optimal')
        valid_count = sum(1 for r in results if r['valid'])
        avg_time = np.mean([r['time'] for r in results])
        
        print(f"\nStatistics:")
        print(f"  Instances solved: {len(results)}")
        print(f"  Optimal solutions: {optimal_count}/{len(results)} ({100*optimal_count/len(results):.1f}%)")
        print(f"  Valid solutions: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
        print(f"  Average solve time: {avg_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--solve':
        # Run with MILP solving
        solve_synthetic_instances()
    else:
        # Run validation only (original behavior)
        main()
        print("\n" + "="*70)
        print("TIP: Run with --solve flag to see MILP solution results:")
        print("     python synthetic_data_generator.py --solve")
        print("="*70)
