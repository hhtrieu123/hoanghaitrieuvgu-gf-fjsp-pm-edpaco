#ifndef FJSP_ACO_H
#define FJSP_ACO_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

// ----------------------------------------------------
// CORE DATA STRUCTURES
// ----------------------------------------------------

typedef struct {
    int machine_id;
    int time; // Processing time
} MachineAlt;

typedef struct {
    int num_alts;
    MachineAlt *alternatives; 
    double *pheromones;       // Pheromone levels for each machine (size: num_machines)
} OperationData;

typedef struct {
    int num_ops;
    OperationData *ops; 
} JobData;

typedef struct {
    int num_jobs;
    int num_machines;
    int total_ops;
    JobData *jobs; 
} FJSP_Data;

typedef struct {
    int job_id;
    int op_idx;
    int machine_id;
    int proc_time;
    int finish_time; 
} ScheduleOp;

// ----------------------------------------------------
// ALGORITHM & CONFIG STRUCTURES
// ----------------------------------------------------

typedef struct {
    int num_ants;
    int max_iter;
    int stop_patience;
    double alpha_start, alpha_end;
    double beta_start, beta_end;
    double rho_min, rho_max;
    double q0;
    int ls_max_attempts;
    int independent_runs;
} Config;

typedef struct {
    FJSP_Data *data;
    double alpha; 
    double beta;  
    double q0;    
    
    double makespan;
    int schedule_len;
    ScheduleOp *schedule;
    
    double *machine_times; 
    double *job_times;     
    int *op_counters;      
} Ant;

typedef struct {
    FJSP_Data *data;
    Config cfg;
    double min_tau;
    double max_tau;
} Optimizer;

// ----------------------------------------------------
// RUN RESULT STRUCTURE (for statistics)
// ----------------------------------------------------

typedef struct {
    char instance_name[64];
    int bks;
    int num_runs;
    double *run_scores;        // Score for each run
    double *run_times;         // Time for each run (seconds)
    double **convergence;      // Convergence curve for each run [run][iteration]
    double *convergence_times; // Time at each iteration for first run
    int convergence_len;       // Number of iterations
    
    // Statistics
    double best_score;
    double worst_score;
    double mean_score;
    double std_score;
    double gap_best;
    double gap_mean;
    double mean_time;
} RunResult;

// ----------------------------------------------------
// BENCHMARK INFO STRUCTURE
// ----------------------------------------------------

typedef struct {
    char name[20];
    int bks;
    char filepath[64];
} BenchmarkInstance;

// ----------------------------------------------------
// FUNCTION PROTOTYPES
// ----------------------------------------------------

// Utilities
double max_double(double a, double b);
double min_double(double a, double b);
ScheduleOp* copy_schedule(const ScheduleOp *source, int len);
void free_ant(Ant *ant);
void free_fjsp_data(FJSP_Data *data);
void free_optimizer(Optimizer *opt);
void free_run_result(RunResult *result);

// Core functions
int calculate_makespan(ScheduleOp *schedule_arr, int num_operations, int num_jobs, int num_machines);
FJSP_Data* load_fjsp_data(const char *file_path);
FJSP_Data* load_standard_jss(const char *file_path);  // For LA instances

// Ant functions
Ant* create_ant(FJSP_Data *data, double alpha, double beta, double q0);
void build_solution(Ant *ant);

// Local search
double stochastic_hill_climb(ScheduleOp **best_sched_ptr, int *sched_len_ptr, double current_mk, FJSP_Data *data, int attempts);

// Optimizer functions
Optimizer* create_optimizer(FJSP_Data *data, const Config *cfg);
double optimize_with_time(Optimizer *opt, double *convergence_curve, double *time_curve);

// Statistics
void calculate_statistics(RunResult *result);

// Output functions
void write_csv_results(const char *filename, RunResult *results, int num_results);
void write_text_report(const char *filename, RunResult *results, int num_results, const Config *cfg);
void write_convergence_csv(const char *filename, RunResult *result);

// Benchmark loaders
int get_brandimarte_benchmarks(BenchmarkInstance **benchmarks);
int get_lawrence_benchmarks(BenchmarkInstance **benchmarks);

#endif // FJSP_ACO_H
