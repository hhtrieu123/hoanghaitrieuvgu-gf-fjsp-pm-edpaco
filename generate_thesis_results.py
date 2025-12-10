#!/usr/bin/env python3
"""
Thesis Results Generator
=========================

Generate all results, tables, and figures for thesis Chapter 5.

This script:
1. Runs all experiments
2. Generates LaTeX tables
3. Creates all figures
4. Exports to thesis format

Usage:
    python generate_thesis_results.py

Author: Master's Thesis
"""

import numpy as np
import random
import time
import json
import os
from datetime import datetime


# =============================================================================
# IMPORT ALL MODULES
# =============================================================================

print("Loading modules...")

from run_all_experiments import (
    Instance, Job, Operation, Machine, FuzzyTime, Solution,
    generate_instance, decode, repair_precedence,
    ACO, GA, PSO, SA, solve_milp,
    run_experiments, export_results, export_instances
)

from sensitivity_analysis import (
    analyze_alpha_beta, analyze_fuzziness, 
    analyze_aco_parameters, analyze_scaling,
    export_sensitivity_results
)

from visualization import (
    generate_schedule, plot_gantt_chart, plot_machine_utilization,
    plot_energy_breakdown, plot_pareto_front, plot_convergence_comparison,
    plot_boxplot_comparison
)

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except:
    PLOT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'runs': 10,                    # Number of runs per algorithm
    'max_iter_small': 100,         # Iterations for small instances
    'max_iter_hard': 200,          # Iterations for hard instances
    'algorithms': ['ACO', 'GA', 'PSO', 'SA'],
    'output_dir': 'thesis_results',
    'seed': 42
}


# =============================================================================
# INSTANCE DEFINITIONS
# =============================================================================

SMALL_INSTANCES = [
    {'name': 'S1', 'jobs': 3, 'machines': 2, 'ops': 2, 'flex': 0.8, 'seed': 42},
    {'name': 'S2', 'jobs': 4, 'machines': 3, 'ops': 2, 'flex': 0.7, 'seed': 123},
    {'name': 'S3', 'jobs': 5, 'machines': 3, 'ops': 3, 'flex': 0.6, 'seed': 456},
    {'name': 'S4', 'jobs': 6, 'machines': 4, 'ops': 3, 'flex': 0.5, 'seed': 789},
]

MEDIUM_INSTANCES = [
    {'name': 'M1', 'jobs': 10, 'machines': 5, 'ops': 4, 'flex': 0.5, 'seed': 1001},
    {'name': 'M2', 'jobs': 10, 'machines': 6, 'ops': 5, 'flex': 0.5, 'seed': 1002},
    {'name': 'M3', 'jobs': 15, 'machines': 6, 'ops': 4, 'flex': 0.4, 'seed': 1003},
    {'name': 'M4', 'jobs': 15, 'machines': 8, 'ops': 5, 'flex': 0.4, 'seed': 1004},
    {'name': 'M5', 'jobs': 20, 'machines': 8, 'ops': 5, 'flex': 0.4, 'seed': 1005},
]

LARGE_INSTANCES = [
    {'name': 'L1', 'jobs': 20, 'machines': 10, 'ops': 5, 'flex': 0.3, 'seed': 2001},
    {'name': 'L2', 'jobs': 30, 'machines': 10, 'ops': 5, 'flex': 0.3, 'seed': 2002},
    {'name': 'L3', 'jobs': 30, 'machines': 10, 'ops': 6, 'flex': 0.3, 'seed': 2003},
    {'name': 'L4', 'jobs': 40, 'machines': 10, 'ops': 5, 'flex': 0.25, 'seed': 2004},
    {'name': 'L5', 'jobs': 50, 'machines': 10, 'ops': 5, 'flex': 0.25, 'seed': 2005},
]


# =============================================================================
# LATEX TABLE GENERATORS
# =============================================================================

def generate_latex_table_small(results, milp_results):
    """Generate LaTeX table for small instances with MILP comparison"""
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Comparison of metaheuristics with MILP optimal solutions on small instances}
\\label{tab:small_instances}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccccc@{}}
\\toprule
\\textbf{Instance} & \\textbf{Size} & \\textbf{MILP} & \\textbf{ACO} & \\textbf{GA} & \\textbf{PSO} & \\textbf{SA} & \\textbf{Best Gap (\\%)} \\\\
\\midrule
"""
    
    for inst_cfg in SMALL_INSTANCES:
        name = inst_cfg['name']
        size = f"{inst_cfg['jobs']}×{inst_cfg['machines']}×{inst_cfg['ops']}"
        
        milp_val = milp_results.get(name, '-')
        
        alg_vals = {}
        for alg in CONFIG['algorithms']:
            for r in results:
                if r['instance'] == name and r['algorithm'] == alg:
                    alg_vals[alg] = r['best_obj']
                    break
        
        if milp_val != '-' and alg_vals:
            best_meta = min(alg_vals.values())
            gap = (best_meta - milp_val) / milp_val * 100
        else:
            gap = '-'
        
        latex += f"{name} & {size} & "
        latex += f"{milp_val:.2f} & " if milp_val != '-' else "- & "
        
        for alg in CONFIG['algorithms']:
            val = alg_vals.get(alg, '-')
            latex += f"{val:.2f} & " if val != '-' else "- & "
        
        latex += f"{gap:.2f} \\\\\n" if gap != '-' else "- \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular*}
\\end{table}
"""
    return latex


def generate_latex_table_comparison(results, instance_type):
    """Generate LaTeX table for algorithm comparison"""
    
    instances = MEDIUM_INSTANCES if instance_type == 'medium' else LARGE_INSTANCES
    
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Algorithm comparison on {instance_type} instances}}
\\label{{tab:{instance_type}_instances}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}llcccccccc@{{}}}}
\\toprule
\\textbf{{Instance}} & \\textbf{{Size}} & \\textbf{{Ops}} & \\multicolumn{{2}}{{c}}{{\\textbf{{ACO}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{GA}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{PSO}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{SA}}}} \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}} \\cmidrule(lr){{8-9}} \\cmidrule(lr){{10-11}}
 & & & Best & Avg & Best & Avg & Best & Avg & Best & Avg \\\\
\\midrule
"""
    
    for inst_cfg in instances:
        name = inst_cfg['name']
        size = f"{inst_cfg['jobs']}×{inst_cfg['machines']}"
        ops = inst_cfg['jobs'] * inst_cfg['ops']
        
        latex += f"{name} & {size} & {ops}"
        
        for alg in CONFIG['algorithms']:
            for r in results:
                if r['instance'] == name and r['algorithm'] == alg:
                    latex += f" & {r['best_obj']:.1f} & {r['avg_obj']:.1f}"
                    break
        
        latex += " \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular*}
\\end{table}
"""
    return latex


def generate_latex_table_statistical(stats_results):
    """Generate LaTeX table for statistical tests"""
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Statistical comparison (Wilcoxon signed-rank test)}
\\label{tab:statistical}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}
\\toprule
\\textbf{Comparison} & \\textbf{W-statistic} & \\textbf{p-value} & \\textbf{Significant?} \\\\
\\midrule
"""
    
    for key, val in stats_results.items():
        if '_vs_' in key:
            latex += f"{key.replace('_', ' ')} & {val['statistic']:.2f} & {val['p_value']:.4f} & "
            latex += "Yes" if val['p_value'] < 0.05 else "No"
            latex += " \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular*}
\\end{table}
"""
    return latex


def generate_latex_table_sensitivity(sensitivity_results):
    """Generate LaTeX tables for sensitivity analysis"""
    
    tables = ""
    
    # Alpha/Beta table
    if 'alpha_beta' in sensitivity_results:
        tables += """\\begin{table}[htbp]
\\centering
\\caption{Sensitivity analysis: Effect of objective weights}
\\label{tab:sensitivity_weights}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccccc@{}}
\\toprule
$\\alpha$ & $\\beta$ & Avg Makespan & Avg Energy & Avg Objective \\\\
\\midrule
"""
        for r in sensitivity_results['alpha_beta']:
            tables += f"{r['alpha']:.2f} & {r['beta']:.2f} & {r['avg_makespan']:.2f} & "
            tables += f"{r['avg_energy']:.2f} & {r['avg_objective']:.2f} \\\\\n"
        
        tables += """\\bottomrule
\\end{tabular*}
\\end{table}

"""
    
    # Scaling table
    if 'scaling' in sensitivity_results:
        tables += """\\begin{table}[htbp]
\\centering
\\caption{Algorithm scaling with problem size}
\\label{tab:scaling}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccccc@{}}
\\toprule
\\textbf{Size} & \\textbf{Ops} & \\textbf{Avg Obj} & \\textbf{Std} & \\textbf{Time (s)} \\\\
\\midrule
"""
        for r in sensitivity_results['scaling']:
            size = f"{r['jobs']}×{r['machines']}×{r['ops_per_job']}"
            tables += f"{size} & {r['total_ops']} & {r['avg_obj']:.2f} & "
            tables += f"{r['std_obj']:.2f} & {r['avg_time']:.2f} \\\\\n"
        
        tables += """\\bottomrule
\\end{tabular*}
\\end{table}
"""
    
    return tables


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_all_thesis_experiments():
    """Run all experiments for thesis"""
    
    print("="*70)
    print("  THESIS RESULTS GENERATOR")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Create directories
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/tables", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/figures", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/data", exist_ok=True)
    
    all_results = {
        'small': [],
        'medium': [],
        'large': [],
        'milp': {},
        'sensitivity': {},
        'convergence': {}
    }
    
    # =========================================================================
    # PART 1: Small instances with MILP
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: Small Instances (MILP Comparison)")
    print("="*70)
    
    small_instances = []
    for cfg in SMALL_INSTANCES:
        inst = generate_instance(cfg['name'], cfg['jobs'], cfg['machines'], 
                                cfg['ops'], cfg['flex'], seed=cfg['seed'])
        small_instances.append(inst)
        
        # Solve with MILP
        print(f"\n{cfg['name']}: Solving MILP...", end=" ")
        milp_result = solve_milp(inst, time_limit=300)
        if milp_result:
            all_results['milp'][cfg['name']] = milp_result[0].objective
            print(f"Optimal = {milp_result[0].objective:.2f}")
        else:
            print("Failed/Timeout")
        
        # Run metaheuristics
        for alg in CONFIG['algorithms']:
            print(f"  {alg}...", end=" ")
            objs, times = [], []
            convergence_all = []
            
            for run in range(CONFIG['runs']):
                start = time.time()
                if alg == 'ACO':
                    solver = ACO(inst, max_iter=CONFIG['max_iter_small'])
                elif alg == 'GA':
                    solver = GA(inst, max_iter=CONFIG['max_iter_small'])
                elif alg == 'PSO':
                    solver = PSO(inst, max_iter=CONFIG['max_iter_small'])
                else:
                    solver = SA(inst, max_iter=CONFIG['max_iter_small']*10)
                
                solution, conv = solver.solve()
                objs.append(solution.objective)
                times.append(time.time() - start)
                convergence_all.append(conv)
            
            result = {
                'instance': cfg['name'],
                'algorithm': alg,
                'best_obj': min(objs),
                'avg_obj': np.mean(objs),
                'std_obj': np.std(objs),
                'avg_time': np.mean(times),
                'all_objs': objs
            }
            all_results['small'].append(result)
            all_results['convergence'][(cfg['name'], alg)] = np.mean(convergence_all, axis=0).tolist()
            
            print(f"Best={min(objs):.2f}, Avg={np.mean(objs):.2f}")
    
    # =========================================================================
    # PART 2: Medium instances
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: Medium Instances")
    print("="*70)
    
    for cfg in MEDIUM_INSTANCES:
        inst = generate_instance(cfg['name'], cfg['jobs'], cfg['machines'],
                                cfg['ops'], cfg['flex'], seed=cfg['seed'])
        
        print(f"\n{cfg['name']} ({cfg['jobs']}×{cfg['machines']}×{cfg['ops']}):")
        
        for alg in CONFIG['algorithms']:
            print(f"  {alg}...", end=" ")
            objs, times = [], []
            
            for run in range(CONFIG['runs']):
                start = time.time()
                if alg == 'ACO':
                    solver = ACO(inst, max_iter=CONFIG['max_iter_hard'])
                elif alg == 'GA':
                    solver = GA(inst, max_iter=CONFIG['max_iter_hard'])
                elif alg == 'PSO':
                    solver = PSO(inst, max_iter=CONFIG['max_iter_hard'])
                else:
                    solver = SA(inst, max_iter=CONFIG['max_iter_hard']*10)
                
                solution, _ = solver.solve()
                objs.append(solution.objective)
                times.append(time.time() - start)
            
            result = {
                'instance': cfg['name'],
                'algorithm': alg,
                'best_obj': min(objs),
                'avg_obj': np.mean(objs),
                'std_obj': np.std(objs),
                'avg_time': np.mean(times),
                'all_objs': objs
            }
            all_results['medium'].append(result)
            
            print(f"Best={min(objs):.2f}, Avg={np.mean(objs):.2f}, Time={np.mean(times):.1f}s")
    
    # =========================================================================
    # PART 3: Large instances
    # =========================================================================
    print("\n" + "="*70)
    print("PART 3: Large Instances")
    print("="*70)
    
    for cfg in LARGE_INSTANCES:
        inst = generate_instance(cfg['name'], cfg['jobs'], cfg['machines'],
                                cfg['ops'], cfg['flex'], seed=cfg['seed'])
        
        print(f"\n{cfg['name']} ({cfg['jobs']}×{cfg['machines']}×{cfg['ops']}):")
        
        for alg in CONFIG['algorithms']:
            print(f"  {alg}...", end=" ")
            objs, times = [], []
            
            for run in range(CONFIG['runs']):
                start = time.time()
                if alg == 'ACO':
                    solver = ACO(inst, max_iter=CONFIG['max_iter_hard'])
                elif alg == 'GA':
                    solver = GA(inst, max_iter=CONFIG['max_iter_hard'])
                elif alg == 'PSO':
                    solver = PSO(inst, max_iter=CONFIG['max_iter_hard'])
                else:
                    solver = SA(inst, max_iter=CONFIG['max_iter_hard']*10)
                
                solution, _ = solver.solve()
                objs.append(solution.objective)
                times.append(time.time() - start)
            
            result = {
                'instance': cfg['name'],
                'algorithm': alg,
                'best_obj': min(objs),
                'avg_obj': np.mean(objs),
                'std_obj': np.std(objs),
                'avg_time': np.mean(times),
                'all_objs': objs
            }
            all_results['large'].append(result)
            
            print(f"Best={min(objs):.2f}, Avg={np.mean(objs):.2f}, Time={np.mean(times):.1f}s")
    
    # =========================================================================
    # PART 4: Sensitivity Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PART 4: Sensitivity Analysis")
    print("="*70)
    
    base_inst = generate_instance("Base", 10, 5, 4, seed=42)
    
    print("\nAnalyzing alpha/beta weights...")
    all_results['sensitivity']['alpha_beta'] = analyze_alpha_beta(base_inst, runs=5)
    
    print("\nAnalyzing fuzziness levels...")
    all_results['sensitivity']['fuzziness'] = analyze_fuzziness(runs=5)
    
    print("\nAnalyzing ACO parameters...")
    all_results['sensitivity']['aco_parameters'] = analyze_aco_parameters(base_inst, runs=5)
    
    print("\nAnalyzing scaling...")
    all_results['sensitivity']['scaling'] = analyze_scaling(runs=5)
    
    # =========================================================================
    # EXPORT RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    
    # Export JSON
    json_safe = {
        'small': all_results['small'],
        'medium': all_results['medium'],
        'large': all_results['large'],
        'milp': all_results['milp'],
        'sensitivity': all_results['sensitivity']
    }
    with open(f"{CONFIG['output_dir']}/data/all_results.json", 'w') as f:
        json.dump(json_safe, f, indent=2, default=str)
    
    # Export LaTeX tables
    tables_dir = f"{CONFIG['output_dir']}/tables"
    
    # Small instances table
    with open(f"{tables_dir}/table_small.tex", 'w') as f:
        f.write(generate_latex_table_small(all_results['small'], all_results['milp']))
    
    # Medium instances table
    with open(f"{tables_dir}/table_medium.tex", 'w') as f:
        f.write(generate_latex_table_comparison(all_results['medium'], 'medium'))
    
    # Large instances table
    with open(f"{tables_dir}/table_large.tex", 'w') as f:
        f.write(generate_latex_table_comparison(all_results['large'], 'large'))
    
    # Sensitivity tables
    with open(f"{tables_dir}/table_sensitivity.tex", 'w') as f:
        f.write(generate_latex_table_sensitivity(all_results['sensitivity']))
    
    # Generate figures
    if PLOT_AVAILABLE:
        figures_dir = f"{CONFIG['output_dir']}/figures"
        
        # Convergence plots
        for name, alg in all_results['convergence'].keys():
            pass  # Individual convergence plots
        
        # Sample Gantt chart
        sample_inst = generate_instance("Sample", 5, 3, 3, seed=42)
        sample_schedule = generate_schedule(sample_inst)
        plot_gantt_chart(sample_inst, sample_schedule, None, f"{figures_dir}/gantt_sample.png")
        
        # Utilization
        plot_machine_utilization(sample_inst, sample_schedule, f"{figures_dir}/utilization.png")
        
        # Energy breakdown
        plot_energy_breakdown(sample_inst, sample_schedule, f"{figures_dir}/energy.png")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("THESIS RESULTS GENERATION COMPLETE")
    print("="*70)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {CONFIG['output_dir']}/")
    print("\nGenerated files:")
    print("  - data/all_results.json")
    print("  - tables/table_small.tex")
    print("  - tables/table_medium.tex")
    print("  - tables/table_large.tex")
    print("  - tables/table_sensitivity.tex")
    print("  - figures/*.png")
    
    # Print summary statistics
    print("\n" + "-"*70)
    print("RESULTS SUMMARY")
    print("-"*70)
    
    for category in ['small', 'medium', 'large']:
        print(f"\n{category.upper()} INSTANCES:")
        results = all_results[category]
        for alg in CONFIG['algorithms']:
            alg_results = [r for r in results if r['algorithm'] == alg]
            if alg_results:
                avg_best = np.mean([r['best_obj'] for r in alg_results])
                print(f"  {alg}: Avg Best = {avg_best:.2f}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_all_thesis_experiments()
