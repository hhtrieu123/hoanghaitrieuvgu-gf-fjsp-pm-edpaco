# EDP-ACO: Enhanced Dual-Pheromone Ant Colony Optimization
## For Green Fuzzy Flexible Job Shop Scheduling with Preventive Maintenance

**Author:** Hoang Hai Trieu  
**Student ID:** VGU: 20623015  
**Advisors:** Prof. Dr. Nguyen Thi Viet Ly, Prof. Dr. Brian Boyd

---

## Project Structure

```
code/
├── Core Algorithm
│   ├── fjsp_aco_python.py      # Main EDP-ACO algorithm (Python)
│   ├── fjsp_aco.c              # EDP-ACO algorithm (C version)
│   ├── fjsp_aco.h              # C header file
│   ├── main.c                  # C main entry point
│   └── PARAMETERS.py           # Algorithm parameters
│
├── Experiments
│   ├── benchmark_comparison.py  # Compare with GA, PSO, SA, Basic-ACO
│   ├── hybrid_comparison.py     # Compare with HGATS, HGAVND, MA
│   ├── objective_weight_analysis.py  # Multi-objective weight tuning
│   ├── parameter_tuning.py      # ACO parameter optimization
│   ├── sensitivity_analysis.py  # Sensitivity analysis
│   └── run_all_experiments.py   # Run all experiments
│
├── Verification
│   ├── milp_verification.py     # MILP formulation (optimal solutions)
│   └── generate_thesis_results.py  # Generate thesis tables/figures
│
├── Data Generation
│   ├── synthetic_data_generator.py    # Generate test instances
│   ├── fuzzy_benchmark_generator.py   # Add fuzzy times to benchmarks
│   └── generate_thesis_benchmarks.py  # Create thesis benchmarks
│
├── Visualization
│   ├── visualization.py         # General plotting utilities
│   └── plot_convergence.py      # Convergence curve plots
│
├── Results (from experiments)
│   ├── basic_comparison/        # Results vs basic algorithms
│   ├── hybrid_comparison/       # Results vs hybrid algorithms
│   └── weight_analysis/         # Weight sensitivity results
│
├── instances/                   # Test problem instances
└── Makefile                     # Build C version
```

---

## Quick Start

### Requirements

```bash
pip install numpy matplotlib scipy
```

### Run Basic Demo

```python
python fjsp_aco_python.py
```

### Run Full Comparison Experiments

```python
# Basic metaheuristic comparison (EDP-ACO vs GA, PSO, SA)
python benchmark_comparison.py

# Hybrid algorithm comparison (EDP-ACO vs HGATS, HGAVND, MA)
python hybrid_comparison.py

# Weight sensitivity analysis
python objective_weight_analysis.py
```

---

## Core Algorithm: EDP-ACO

The main algorithm is in `fjsp_aco_python.py`. Key features:

### 1. Dual-Pheromone Mechanism
- **Attractive pheromone (τ⁺):** Reinforces good solutions
- **Repulsive pheromone (τ⁻):** Discourages poor solutions

### 2. Fuzzy Processing Times
- Triangular Fuzzy Numbers (TFN): (L, M, U)
- GMIR defuzzification for crisp comparisons

### 3. Multi-Objective Optimization
- Makespan minimization
- Energy consumption minimization
- Weighted sum approach: f = α·Cmax + β·Energy

### 4. Preventive Maintenance Integration
- Time window constraints [T_early, T_late]
- Machine unavailability during maintenance

---

## Key Classes and Functions

### `fjsp_aco_python.py`

```python
class FuzzyNumber:
    """Triangular fuzzy number (L, M, U)"""
    def __init__(self, L, M, U)
    def gmir(self)  # Graded Mean Integration Representation
    def __add__, __mul__, max()  # Fuzzy arithmetic

class Operation:
    """Single operation with fuzzy processing times"""
    job_id, op_id, eligible_machines, processing_times

class Job:
    """Collection of operations with precedence"""
    operations: List[Operation]

class Machine:
    """Machine with energy parameters and maintenance"""
    power_processing, power_idle, maintenance_window

class EDPACO:
    """Enhanced Dual-Pheromone ACO"""
    def __init__(self, jobs, machines, params)
    def solve(self, max_iterations)
    def construct_solution(self, ant)
    def local_search(self, solution)
    def update_pheromones(self, solutions)
    def calculate_objective(self, solution)
```

### Example Usage

```python
from fjsp_aco_python import EDPACO, Job, Operation, Machine, FuzzyNumber

# Create jobs with fuzzy processing times
jobs = [
    Job(job_id=0, operations=[
        Operation(0, 0, 
                  eligible_machines=[0, 1],
                  processing_times={
                      0: FuzzyNumber(8, 10, 12),
                      1: FuzzyNumber(6, 8, 10)
                  })
    ])
]

# Create machines with energy parameters
machines = [
    Machine(machine_id=0, power_processing=10.0, power_idle=2.0,
            maintenance_window=(40, 120))
]

# Run EDP-ACO
params = {
    'n_ants': 20,
    'alpha': 1.0,      # Pheromone importance
    'beta': 2.0,       # Heuristic importance
    'rho': 0.1,        # Evaporation rate
    'q0': 0.9,         # Exploitation probability
    'weight_makespan': 0.5,
    'weight_energy': 0.5
}

solver = EDPACO(jobs, machines, params)
best_solution, best_obj, history = solver.solve(max_iterations=100)

print(f"Best Makespan: {best_solution.makespan}")
print(f"Best Energy: {best_solution.energy}")
print(f"Best Objective: {best_obj}")
```

---

## Experimental Results

### Basic Comparison (EDP-ACO vs GA, PSO, SA)

| Instance | EDP-ACO | GA | PSO | SA |
|----------|---------|-----|-----|-----|
| Small_6x4x3 | **48.02** | 52.35 | 57.51 | 69.43 |
| Medium_10x5x4 | **83.72** | 98.55 | 112.96 | 114.47 |
| Medium_15x6x5 | **102.85** | 146.07 | 150.80 | 150.84 |
| Large_20x8x5 | **116.70** | 160.79 | 190.87 | 168.84 |

### Hybrid Comparison (EDP-ACO vs State-of-the-Art)

| Instance | EDP-ACO | HGATS | HGAVND | MA |
|----------|---------|-------|--------|-----|
| Small_6x4x3 | 155.59 | 155.59 | 162.36 | 156.02 |
| Medium_10x5x4 | 325.07 | 333.88 | 409.88 | **312.27** |
| Medium_15x6x5 | 479.36 | 494.56 | 619.03 | **476.39** |

---

## Algorithm Parameters

Default parameters (in `PARAMETERS.py`):

```python
PARAMS = {
    # ACO parameters
    'n_ants': 20,
    'max_iterations': 100,
    'alpha': 1.0,           # Pheromone importance
    'beta': 2.0,            # Heuristic importance  
    'rho': 0.1,             # Evaporation rate
    'q0': 0.9,              # Exploitation vs exploration
    
    # Dual pheromone
    'tau_min': 0.01,
    'tau_max': 10.0,
    'repulsive_weight': 0.3,
    
    # Multi-objective weights
    'weight_makespan': 0.5,
    'weight_energy': 0.5,
    
    # Local search
    'ls_iterations': 10,
    'ls_probability': 0.3
}
```

---

## References

1. **FJSP Review:** Dauzère-Pérès et al. (2024). "The flexible job shop scheduling problem: A review." EJOR.

2. **ACO Foundation:** Dorigo & Gambardella (1997). "Ant colony system." IEEE Trans. Evolutionary Computation.

3. **Fuzzy Scheduling:** Lei (2010). "A genetic algorithm for fuzzy flexible job shop scheduling." Applied Soft Computing.

4. **Energy-Efficient Scheduling:** Gahm et al. (2016). "Energy-efficient scheduling in manufacturing." Journal of Cleaner Production.

---

## License

This code is provided for academic purposes as part of a Master's thesis at Vietnamese-German University / Heilbronn University.

## Contact

- **Student:** Hoang Hai Trieu
- **VGU ID:** 20623015
- **Email:** [Your Email]
