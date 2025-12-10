#include "fjsp_aco.h"
#include <ctype.h>

// =========================================================
//  UTILITIES & MEMORY MANAGEMENT
// =========================================================

double max_double(double a, double b) { return (a > b) ? a : b; }
double min_double(double a, double b) { return (a < b) ? a : b; }

ScheduleOp* copy_schedule(const ScheduleOp *source, int len) {
    ScheduleOp *copy = (ScheduleOp*)malloc(len * sizeof(ScheduleOp));
    if (copy) {
        memcpy(copy, source, len * sizeof(ScheduleOp));
    }
    return copy;
}

void free_ant(Ant *ant) {
    if (!ant) return;
    if (ant->schedule) free(ant->schedule);
    if (ant->machine_times) free(ant->machine_times);
    if (ant->job_times) free(ant->job_times);
    if (ant->op_counters) free(ant->op_counters);
    free(ant);
}

void free_fjsp_data(FJSP_Data *data) {
    if (data == NULL) return;

    for (int j = 0; j < data->num_jobs; j++) {
        JobData *job = &data->jobs[j];
        for (int op = 0; op < job->num_ops; op++) {
            OperationData *op_data = &job->ops[op];
            if (op_data->alternatives) free(op_data->alternatives);
            if (op_data->pheromones) free(op_data->pheromones);
        }
        if (job->ops) free(job->ops);
    }
    if (data->jobs) free(data->jobs);
    free(data);
}

void free_optimizer(Optimizer *opt) {
    if (opt) free(opt);
}

void free_run_result(RunResult *result) {
    if (!result) return;
    if (result->run_scores) free(result->run_scores);
    if (result->run_times) free(result->run_times);
    if (result->convergence_times) free(result->convergence_times);
    if (result->convergence) {
        for (int i = 0; i < result->num_runs; i++) {
            if (result->convergence[i]) free(result->convergence[i]);
        }
        free(result->convergence);
    }
}

// =========================================================
//  FAST MAKESPAN CALCULATION
// =========================================================

int calculate_makespan(ScheduleOp *schedule_arr, int num_operations, int num_jobs, int num_machines) {
    
    double *m_times = (double*)calloc(num_machines, sizeof(double));
    double *j_times = (double*)calloc(num_jobs, sizeof(double));
    double mk = 0;

    for (int i = 0; i < num_operations; i++) {
        ScheduleOp *op = &schedule_arr[i];
        
        double start_t = max_double(j_times[op->job_id], m_times[op->machine_id]);
        double finish_t = start_t + op->proc_time;

        m_times[op->machine_id] = finish_t;
        j_times[op->job_id] = finish_t;
        op->finish_time = (int)round(finish_t); 
        
        if (finish_t > mk) {
            mk = finish_t;
        }
    }

    free(m_times);
    free(j_times);
    return (int)round(mk);
}

// =========================================================
//  DATA LOADING - BRANDIMARTE FORMAT (FJSP)
// =========================================================

FJSP_Data* load_fjsp_data(const char *file_path) {
    
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "ERROR: Could not open file: %s\n", file_path);
        return NULL;
    }

    FJSP_Data *data = (FJSP_Data*)malloc(sizeof(FJSP_Data));
    if (!data) {
        fclose(file);
        fprintf(stderr, "ERROR: Failed to allocate memory for FJSP_Data.\n");
        return NULL;
    }
    
    data->total_ops = 0;
    data->num_jobs = 0;
    data->num_machines = 0;

    // Read Header Line
    if (fscanf(file, "%d %d", &data->num_jobs, &data->num_machines) != 2) {
        fprintf(stderr, "ERROR: Failed to read problem dimensions from file %s.\n", file_path);
        free(data);
        fclose(file);
        return NULL;
    }
    
    // Skip rest of first line
    int c;
    while ((c = fgetc(file)) != '\n' && c != EOF);
    
    if (data->num_jobs <= 0 || data->num_machines <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions read: J=%d, M=%d in %s.\n", 
                data->num_jobs, data->num_machines, file_path);
        free(data);
        fclose(file);
        return NULL;
    }
    
    data->jobs = (JobData*)calloc(data->num_jobs, sizeof(JobData));
    if (!data->jobs) { 
        free(data);
        fclose(file);
        return NULL;
    }

    for (int j = 0; j < data->num_jobs; j++) {
        JobData *job = &data->jobs[j];
        
        if (fscanf(file, "%d", &job->num_ops) != 1) {
            fprintf(stderr, "ERROR: Failed to read operation count for Job %d in %s.\n", j, file_path);
            free_fjsp_data(data);
            fclose(file);
            return NULL;
        }
        
        job->ops = (OperationData*)calloc(job->num_ops, sizeof(OperationData));
        data->total_ops += job->num_ops;

        for (int op = 0; op < job->num_ops; op++) {
            OperationData *op_data = &job->ops[op];
            
            if (fscanf(file, "%d", &op_data->num_alts) != 1) {
                fprintf(stderr, "ERROR: Failed to read alternative count for Job %d, Op %d in %s.\n", j, op, file_path);
                free_fjsp_data(data);
                fclose(file);
                return NULL;
            }
            
            op_data->alternatives = (MachineAlt*)calloc(op_data->num_alts, sizeof(MachineAlt));
            op_data->pheromones = NULL; 
            
            for (int a = 0; a < op_data->num_alts; a++) {
                int m_id, t_time;
                if (fscanf(file, "%d %d", &m_id, &t_time) != 2) {
                    fprintf(stderr, "ERROR: Failed to read machine/time pair for Job %d, Op %d, Alt %d in %s.\n", j, op, a, file_path);
                    free_fjsp_data(data);
                    fclose(file);
                    return NULL;
                }
                op_data->alternatives[a].machine_id = m_id - 1; // 0-based
                op_data->alternatives[a].time = t_time;
            }
        }
    }
    
    fclose(file);
    return data;
}

// =========================================================
//  DATA LOADING - STANDARD JSP FORMAT (Lawrence LA instances)
//  Format: first line = jobs machines
//          each subsequent line = machine time machine time ... (for each operation)
// =========================================================

FJSP_Data* load_standard_jss(const char *file_path) {
    
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "ERROR: Could not open file: %s\n", file_path);
        return NULL;
    }

    FJSP_Data *data = (FJSP_Data*)malloc(sizeof(FJSP_Data));
    if (!data) {
        fclose(file);
        return NULL;
    }
    
    data->total_ops = 0;

    // Read header
    if (fscanf(file, "%d %d", &data->num_jobs, &data->num_machines) != 2) {
        fprintf(stderr, "ERROR: Failed to read dimensions from %s.\n", file_path);
        free(data);
        fclose(file);
        return NULL;
    }
    
    data->jobs = (JobData*)calloc(data->num_jobs, sizeof(JobData));
    if (!data->jobs) { 
        free(data);
        fclose(file);
        return NULL;
    }

    // Each job has num_machines operations (standard JSP)
    for (int j = 0; j < data->num_jobs; j++) {
        JobData *job = &data->jobs[j];
        job->num_ops = data->num_machines;
        job->ops = (OperationData*)calloc(job->num_ops, sizeof(OperationData));
        data->total_ops += job->num_ops;

        for (int op = 0; op < job->num_ops; op++) {
            OperationData *op_data = &job->ops[op];
            
            int m_id, t_time;
            if (fscanf(file, "%d %d", &m_id, &t_time) != 2) {
                fprintf(stderr, "ERROR: Failed to read machine/time for Job %d, Op %d in %s.\n", j, op, file_path);
                free_fjsp_data(data);
                fclose(file);
                return NULL;
            }
            
            // Standard JSP: each operation has exactly 1 machine alternative
            op_data->num_alts = 1;
            op_data->alternatives = (MachineAlt*)malloc(sizeof(MachineAlt));
            op_data->alternatives[0].machine_id = m_id;  // Already 0-based in standard format
            op_data->alternatives[0].time = t_time;
            op_data->pheromones = NULL;
        }
    }
    
    fclose(file);
    return data;
}

// =========================================================
//  ANT - CREATE AND BUILD SOLUTION
// =========================================================

Ant* create_ant(FJSP_Data *data, double alpha, double beta, double q0) {
    Ant *ant = (Ant*)malloc(sizeof(Ant));
    if (!ant) return NULL;
    
    ant->data = data;
    ant->alpha = alpha;
    ant->beta = beta;
    ant->q0 = q0;
    ant->makespan = 0.0;
    ant->schedule_len = 0;
    ant->schedule = NULL; 
    
    ant->machine_times = (double*)calloc(data->num_machines, sizeof(double));
    ant->job_times = (double*)calloc(data->num_jobs, sizeof(double));
    ant->op_counters = (int*)calloc(data->num_jobs, sizeof(int));
    
    if (!ant->machine_times || !ant->job_times || !ant->op_counters) {
        free_ant(ant); 
        return NULL;
    }
    return ant;
}

void build_solution(Ant *ant) {
    
    memset(ant->machine_times, 0, ant->data->num_machines * sizeof(double));
    memset(ant->job_times, 0, ant->data->num_jobs * sizeof(double));
    memset(ant->op_counters, 0, ant->data->num_jobs * sizeof(int));
    ant->makespan = 0.0;
    ant->schedule_len = 0;

    if (ant->schedule) free(ant->schedule);
    ant->schedule = (ScheduleOp*)malloc(ant->data->total_ops * sizeof(ScheduleOp)); 

    int total_ops_done = 0;
    
    while (total_ops_done < ant->data->total_ops) {
        
        ScheduleOp *move_options = NULL;
        double *probabilities = NULL;
        int num_options = 0;

        // 1. GATHER ALL POSSIBLE MOVES
        for (int j = 0; j < ant->data->num_jobs; j++) {
            int current_op_idx = ant->op_counters[j];
            if (current_op_idx >= ant->data->jobs[j].num_ops) continue;
            
            OperationData *op_data = &ant->data->jobs[j].ops[current_op_idx];
            
            for (int alt = 0; alt < op_data->num_alts; alt++) {
                int m_id = op_data->alternatives[alt].machine_id;
                int p_time = op_data->alternatives[alt].time;
                
                double start_t = max_double(ant->job_times[j], ant->machine_times[m_id]);
                double finish_t = start_t + p_time;
                
                num_options++;
                move_options = (ScheduleOp*)realloc(move_options, num_options * sizeof(ScheduleOp));
                probabilities = (double*)realloc(probabilities, num_options * sizeof(double));
                
                move_options[num_options - 1].job_id = j;
                move_options[num_options - 1].op_idx = current_op_idx;
                move_options[num_options - 1].machine_id = m_id;
                move_options[num_options - 1].proc_time = p_time;
                move_options[num_options - 1].finish_time = (int)finish_t;
                
                // Calculate probability
                double tau = (op_data->pheromones) ? op_data->pheromones[m_id] : 1.0;
                double eta = (p_time > 0) ? 1.0 / p_time : 1.0;
                probabilities[num_options - 1] = pow(tau, ant->alpha) * pow(eta, ant->beta);
            }
        }
        
        if (num_options == 0) break;

        // 2. SELECT A MOVE
        int selected_idx = 0;
        double q = (double)rand() / RAND_MAX;
        
        if (q < ant->q0) {
            // Exploitation: select best
            double max_prob = -1.0;
            for (int i = 0; i < num_options; i++) {
                if (probabilities[i] > max_prob) {
                    max_prob = probabilities[i];
                    selected_idx = i;
                }
            }
        } else {
            // Exploration: roulette wheel
            double sum = 0.0;
            for (int i = 0; i < num_options; i++) sum += probabilities[i];
            
            if (sum > 0) {
                double r = ((double)rand() / RAND_MAX) * sum;
                double cumsum = 0.0;
                for (int i = 0; i < num_options; i++) {
                    cumsum += probabilities[i];
                    if (r <= cumsum) {
                        selected_idx = i;
                        break;
                    }
                }
            }
        }
        
        // 3. APPLY THE SELECTED MOVE
        ScheduleOp selected_op = move_options[selected_idx];
        
        double start_t = max_double(ant->job_times[selected_op.job_id], ant->machine_times[selected_op.machine_id]);
        double finish_t = start_t + selected_op.proc_time;
        selected_op.finish_time = (int)finish_t;
        
        ant->machine_times[selected_op.machine_id] = finish_t;
        ant->job_times[selected_op.job_id] = finish_t;
        ant->op_counters[selected_op.job_id]++;
        
        ant->schedule[ant->schedule_len++] = selected_op;
        total_ops_done++;
        
        if (selected_op.finish_time > ant->makespan) ant->makespan = selected_op.finish_time;

        free(move_options);
        free(probabilities);
    }
}

// =========================================================
//  LOCAL SEARCH
// =========================================================

double stochastic_hill_climb(ScheduleOp **best_sched_ptr, int *sched_len_ptr, double current_mk, FJSP_Data *data, int attempts) {
    
    ScheduleOp *current_sched = *best_sched_ptr;
    int num_rows = *sched_len_ptr;
    double best_mk = current_mk;
    
    ScheduleOp *temp_sched = copy_schedule(current_sched, num_rows); 
    if (!temp_sched) return current_mk;
    
    for (int i = 0; i < attempts; i++) {
        int mode = rand() % 2; 

        if (mode == 1) { 
            // Sequence Swap
            int idx = rand() % (num_rows - 1);
            if (current_sched[idx].job_id != current_sched[idx + 1].job_id) {
                ScheduleOp temp = temp_sched[idx];
                temp_sched[idx] = temp_sched[idx + 1];
                temp_sched[idx + 1] = temp;
            }
        } else { 
            // Machine Change
            int target_idx = rand() % num_rows;
            int j = temp_sched[target_idx].job_id;
            int op = temp_sched[target_idx].op_idx;

            if (data->jobs[j].ops[op].num_alts > 1) {
                int alt_idx = rand() % data->jobs[j].ops[op].num_alts;
                temp_sched[target_idx].machine_id = data->jobs[j].ops[op].alternatives[alt_idx].machine_id;
                temp_sched[target_idx].proc_time = data->jobs[j].ops[op].alternatives[alt_idx].time;
            }
        }
        
        int new_mk = calculate_makespan(temp_sched, num_rows, data->num_jobs, data->num_machines);
        
        if (new_mk <= best_mk) {
            best_mk = (double)new_mk;
            free(*best_sched_ptr);
            *best_sched_ptr = copy_schedule(temp_sched, num_rows);
            if (!*best_sched_ptr) {
                free(temp_sched);
                return best_mk;
            }
            current_sched = *best_sched_ptr;
        } else {
            memcpy(temp_sched, current_sched, num_rows * sizeof(ScheduleOp));
        }
    }
    
    free(temp_sched);
    return best_mk;
}

// =========================================================
//  OPTIMIZER
// =========================================================

Optimizer* create_optimizer(FJSP_Data *data, const Config *cfg) {
    Optimizer *opt = (Optimizer*)malloc(sizeof(Optimizer));
    if (!opt) return NULL;
    
    opt->data = data;
    opt->cfg = *cfg;
    opt->min_tau = 0.1;
    opt->max_tau = 10.0;
    
    for (int j = 0; j < data->num_jobs; j++) {
        for (int op = 0; op < data->jobs[j].num_ops; op++) {
            OperationData *op_data = &data->jobs[j].ops[op];
            
            if (op_data->pheromones == NULL) {
                op_data->pheromones = (double*)malloc(data->num_machines * sizeof(double));
            }
            
            for (int m = 0; m < data->num_machines; m++) {
                op_data->pheromones[m] = 1.0;
            }
        }
    }
    return opt;
}

double optimize_with_time(Optimizer *opt, double *convergence_curve, double *time_curve) {
    
    clock_t start_clock = clock();
    
    double best_global_mk = DBL_MAX;
    ScheduleOp *best_global_sched = NULL;
    int best_global_sched_len = 0;
    int stagnation = 0;
    Ant *ants[opt->cfg.num_ants]; 
    
    memset(ants, 0, sizeof(Ant*) * opt->cfg.num_ants); 
    
    for (int it = 0; it < opt->cfg.max_iter; it++) {
        
        // Dynamic Parameters
        double ratio = (double)it / opt->cfg.max_iter;
        double alpha = opt->cfg.alpha_start + (opt->cfg.alpha_end - opt->cfg.alpha_start) * ratio;
        double beta = opt->cfg.beta_start + (opt->cfg.beta_end - opt->cfg.beta_start) * ratio;
        double rho = (stagnation > 10) ? opt->cfg.rho_max : opt->cfg.rho_min;

        double iter_best_mk = DBL_MAX;
        Ant *iter_best_ant = NULL; 
        double iter_worst_mk = 0.0;
        Ant *iter_worst_ant = NULL; 

        // ANT PHASE
        for (int i = 0; i < opt->cfg.num_ants; i++) {
            ants[i] = create_ant(opt->data, alpha, beta, opt->cfg.q0);
            if (!ants[i]) continue;
            
            build_solution(ants[i]);

            if (ants[i]->makespan < iter_best_mk) { 
                iter_best_mk = ants[i]->makespan; 
                iter_best_ant = ants[i]; 
            }
            if (ants[i]->makespan > iter_worst_mk) { 
                iter_worst_mk = ants[i]->makespan; 
                iter_worst_ant = ants[i]; 
            }
        }
        
        // LOCAL SEARCH
        if (iter_best_ant) {
            double ls_mk = stochastic_hill_climb(
                &iter_best_ant->schedule, 
                &iter_best_ant->schedule_len, 
                iter_best_ant->makespan, 
                opt->data, 
                opt->cfg.ls_max_attempts
            );
            iter_best_ant->makespan = ls_mk; 
            if (ls_mk < iter_best_mk) { 
                iter_best_mk = ls_mk; 
            }
        }
        
        // Global Update
        if (iter_best_mk < best_global_mk) {
            best_global_mk = iter_best_mk;
            if (best_global_sched) free(best_global_sched);
            
            if (iter_best_ant) {
                best_global_sched_len = iter_best_ant->schedule_len;
                best_global_sched = copy_schedule(iter_best_ant->schedule, best_global_sched_len);
            } else {
                best_global_sched_len = 0;
                best_global_sched = NULL;
            }
            stagnation = 0;
        } else {
            stagnation++;
        }

        convergence_curve[it] = best_global_mk;
        
        // Record time
        if (time_curve) {
            clock_t current_clock = clock();
            time_curve[it] = (double)(current_clock - start_clock) / CLOCKS_PER_SEC;
        }

        // PHEROMONE UPDATE
        double deposit = (best_global_mk > 0) ? 100.0 / best_global_mk : 0.0;
        double punishment = (iter_worst_mk > 0) ? 0.5 * (100.0 / iter_worst_mk) : 0.0; 
        
        // Evaporation
        for (int j = 0; j < opt->data->num_jobs; j++) {
            for (int op_idx = 0; op_idx < opt->data->jobs[j].num_ops; op_idx++) {
                OperationData *op_data = &opt->data->jobs[j].ops[op_idx];
                for (int m = 0; m < opt->data->num_machines; m++) {
                    double *tau = &op_data->pheromones[m];
                    *tau *= (1.0 - rho);
                    *tau = max_double(opt->min_tau, *tau);
                }
            }
        }
        
        // Reward
        if (best_global_sched) {
            for (int i = 0; i < best_global_sched_len; i++) {
                ScheduleOp op = best_global_sched[i];
                double *tau = &opt->data->jobs[op.job_id].ops[op.op_idx].pheromones[op.machine_id];
                *tau = min_double(opt->max_tau, *tau + deposit);
            }
        }
        
        // Punish
        if (iter_worst_ant) {
            for (int i = 0; i < iter_worst_ant->schedule_len; i++) {
                ScheduleOp op = iter_worst_ant->schedule[i];
                double *tau = &opt->data->jobs[op.job_id].ops[op.op_idx].pheromones[op.machine_id];
                double new_val = *tau - punishment;
                *tau = max_double(opt->min_tau, new_val);
            }
        }
        
        // Cleanup ants
        for (int i = 0; i < opt->cfg.num_ants; i++) {
            if (ants[i]) free_ant(ants[i]);
            ants[i] = NULL; 
        }

        if (stagnation >= opt->cfg.stop_patience) {
            // Fill remaining iterations with final value
            for (int fill = it + 1; fill < opt->cfg.max_iter; fill++) {
                convergence_curve[fill] = best_global_mk;
                if (time_curve) {
                    clock_t current_clock = clock();
                    time_curve[fill] = (double)(current_clock - start_clock) / CLOCKS_PER_SEC;
                }
            }
            break;
        }
    }
    
    if (best_global_sched) free(best_global_sched);
    return best_global_mk;
}

// =========================================================
//  STATISTICS CALCULATION
// =========================================================

void calculate_statistics(RunResult *result) {
    if (result->num_runs == 0) return;
    
    double sum = 0.0;
    result->best_score = DBL_MAX;
    result->worst_score = -DBL_MAX;
    
    for (int i = 0; i < result->num_runs; i++) {
        double score = result->run_scores[i];
        sum += score;
        if (score < result->best_score) result->best_score = score;
        if (score > result->worst_score) result->worst_score = score;
    }
    
    result->mean_score = sum / result->num_runs;
    
    // Standard deviation
    double var_sum = 0.0;
    for (int i = 0; i < result->num_runs; i++) {
        double diff = result->run_scores[i] - result->mean_score;
        var_sum += diff * diff;
    }
    result->std_score = sqrt(var_sum / result->num_runs);
    
    // Gap calculations
    if (result->bks > 0) {
        result->gap_best = ((result->best_score - result->bks) / result->bks) * 100.0;
        result->gap_mean = ((result->mean_score - result->bks) / result->bks) * 100.0;
    } else {
        result->gap_best = 0.0;
        result->gap_mean = 0.0;
    }
    
    // Mean time
    double time_sum = 0.0;
    for (int i = 0; i < result->num_runs; i++) {
        time_sum += result->run_times[i];
    }
    result->mean_time = time_sum / result->num_runs;
}

// =========================================================
//  OUTPUT FUNCTIONS
// =========================================================

void write_csv_results(const char *filename, RunResult *results, int num_results) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing.\n", filename);
        return;
    }
    
    // Header
    fprintf(f, "Instance,BKS,Best,Worst,Mean,Std,Gap_Best(%%),Gap_Mean(%%),Mean_Time(s)");
    
    // Find max runs for individual columns
    int max_runs = 0;
    for (int i = 0; i < num_results; i++) {
        if (results[i].num_runs > max_runs) max_runs = results[i].num_runs;
    }
    for (int r = 0; r < max_runs; r++) {
        fprintf(f, ",Run%d_Score,Run%d_Time", r+1, r+1);
    }
    fprintf(f, "\n");
    
    // Data rows
    for (int i = 0; i < num_results; i++) {
        RunResult *res = &results[i];
        fprintf(f, "%s,%d,%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.2f",
                res->instance_name, res->bks, res->best_score, res->worst_score,
                res->mean_score, res->std_score, res->gap_best, res->gap_mean, res->mean_time);
        
        for (int r = 0; r < max_runs; r++) {
            if (r < res->num_runs) {
                fprintf(f, ",%.0f,%.2f", res->run_scores[r], res->run_times[r]);
            } else {
                fprintf(f, ",,");
            }
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
    printf("Results saved to: %s\n", filename);
}

void write_text_report(const char *filename, RunResult *results, int num_results, const Config *cfg) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing.\n", filename);
        return;
    }
    
    fprintf(f, "================================================================================\n");
    fprintf(f, "       FJSP-ACO EXPERIMENTAL RESULTS - STATISTICAL REPORT\n");
    fprintf(f, "================================================================================\n\n");
    
    // Configuration
    fprintf(f, "ALGORITHM CONFIGURATION:\n");
    fprintf(f, "------------------------\n");
    fprintf(f, "  Number of ants:       %d\n", cfg->num_ants);
    fprintf(f, "  Max iterations:       %d\n", cfg->max_iter);
    fprintf(f, "  Stopping patience:    %d\n", cfg->stop_patience);
    fprintf(f, "  Alpha (start-end):    %.2f - %.2f\n", cfg->alpha_start, cfg->alpha_end);
    fprintf(f, "  Beta (start-end):     %.2f - %.2f\n", cfg->beta_start, cfg->beta_end);
    fprintf(f, "  Rho (min-max):        %.2f - %.2f\n", cfg->rho_min, cfg->rho_max);
    fprintf(f, "  q0:                   %.2f\n", cfg->q0);
    fprintf(f, "  Local search attempts:%d\n", cfg->ls_max_attempts);
    fprintf(f, "  Independent runs:     %d\n", cfg->independent_runs);
    fprintf(f, "\n");
    
    // Summary table
    fprintf(f, "SUMMARY RESULTS:\n");
    fprintf(f, "----------------\n");
    fprintf(f, "%-12s %6s %8s %8s %8s %8s %10s %10s %10s\n",
            "Instance", "BKS", "Best", "Worst", "Mean", "Std", "Gap_B(%)", "Gap_M(%)", "Time(s)");
    fprintf(f, "--------------------------------------------------------------------------------\n");
    
    double total_gap_best = 0.0;
    double total_gap_mean = 0.0;
    int optimal_count = 0;
    
    for (int i = 0; i < num_results; i++) {
        RunResult *res = &results[i];
        fprintf(f, "%-12s %6d %8.0f %8.0f %8.2f %8.2f %10.2f %10.2f %10.2f\n",
                res->instance_name, res->bks, res->best_score, res->worst_score,
                res->mean_score, res->std_score, res->gap_best, res->gap_mean, res->mean_time);
        
        total_gap_best += res->gap_best;
        total_gap_mean += res->gap_mean;
        if (res->gap_best <= 0.01) optimal_count++;
    }
    
    fprintf(f, "--------------------------------------------------------------------------------\n");
    fprintf(f, "%-12s %6s %8s %8s %8s %8s %10.2f %10.2f\n",
            "AVERAGE", "", "", "", "", "", total_gap_best / num_results, total_gap_mean / num_results);
    fprintf(f, "\n");
    
    // Summary statistics
    fprintf(f, "OVERALL STATISTICS:\n");
    fprintf(f, "-------------------\n");
    fprintf(f, "  Total instances tested:     %d\n", num_results);
    fprintf(f, "  Optimal solutions found:    %d (%.1f%%)\n", optimal_count, (100.0 * optimal_count) / num_results);
    fprintf(f, "  Average gap (best):         %.2f%%\n", total_gap_best / num_results);
    fprintf(f, "  Average gap (mean):         %.2f%%\n", total_gap_mean / num_results);
    fprintf(f, "\n");
    
    // Individual run details
    fprintf(f, "DETAILED RUN RESULTS:\n");
    fprintf(f, "---------------------\n\n");
    
    for (int i = 0; i < num_results; i++) {
        RunResult *res = &results[i];
        fprintf(f, "%s (BKS=%d):\n", res->instance_name, res->bks);
        for (int r = 0; r < res->num_runs; r++) {
            double gap = ((res->run_scores[r] - res->bks) / res->bks) * 100.0;
            fprintf(f, "  Run %2d: Score=%6.0f, Gap=%6.2f%%, Time=%.2fs\n",
                    r+1, res->run_scores[r], gap, res->run_times[r]);
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
    printf("Report saved to: %s\n", filename);
}

void write_convergence_csv(const char *filename, RunResult *result) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing.\n", filename);
        return;
    }
    
    // Header
    fprintf(f, "Iteration,Time(s)");
    for (int r = 0; r < result->num_runs; r++) {
        fprintf(f, ",Run%d", r+1);
    }
    fprintf(f, ",Mean\n");
    
    // Data
    for (int it = 0; it < result->convergence_len; it++) {
        double time_val = (result->convergence_times) ? result->convergence_times[it] : (double)it;
        fprintf(f, "%d,%.4f", it, time_val);
        
        double sum = 0.0;
        int count = 0;
        for (int r = 0; r < result->num_runs; r++) {
            if (result->convergence[r]) {
                fprintf(f, ",%.0f", result->convergence[r][it]);
                sum += result->convergence[r][it];
                count++;
            } else {
                fprintf(f, ",");
            }
        }
        
        double mean = (count > 0) ? sum / count : 0.0;
        fprintf(f, ",%.2f\n", mean);
    }
    
    fclose(f);
}

// =========================================================
//  BENCHMARK LOADERS
// =========================================================

int get_brandimarte_benchmarks(BenchmarkInstance **benchmarks) {
    static BenchmarkInstance mk_benchmarks[] = {
        {"Mk01", 40, "benchmarks/Mk01.txt"},
        {"Mk02", 26, "benchmarks/Mk02.txt"},
        {"Mk03", 204, "benchmarks/Mk03.txt"},
        {"Mk04", 60, "benchmarks/Mk04.txt"},
        {"Mk05", 172, "benchmarks/Mk05.txt"},
        {"Mk06", 57, "benchmarks/Mk06.txt"},
        {"Mk07", 139, "benchmarks/Mk07.txt"},
        {"Mk08", 523, "benchmarks/Mk08.txt"},
        {"Mk09", 307, "benchmarks/Mk09.txt"},
        {"Mk10", 197, "benchmarks/Mk10.txt"}
    };
    *benchmarks = mk_benchmarks;
    return 10;
}

int get_lawrence_benchmarks(BenchmarkInstance **benchmarks) {
    static BenchmarkInstance la_benchmarks[] = {
        // LA01-LA05: 10 jobs, 5 machines
        {"LA01", 666, "benchmarks/la01.txt"},
        {"LA02", 655, "benchmarks/la02.txt"},
        {"LA03", 597, "benchmarks/la03.txt"},
        {"LA04", 590, "benchmarks/la04.txt"},
        {"LA05", 593, "benchmarks/la05.txt"},
        // LA06-LA10: 15 jobs, 5 machines
        {"LA06", 926, "benchmarks/la06.txt"},
        {"LA07", 890, "benchmarks/la07.txt"},
        {"LA08", 863, "benchmarks/la08.txt"},
        {"LA09", 951, "benchmarks/la09.txt"},
        {"LA10", 958, "benchmarks/la10.txt"},
        // LA11-LA15: 20 jobs, 5 machines
        {"LA11", 1222, "benchmarks/la11.txt"},
        {"LA12", 1039, "benchmarks/la12.txt"},
        {"LA13", 1150, "benchmarks/la13.txt"},
        {"LA14", 1292, "benchmarks/la14.txt"},
        {"LA15", 1207, "benchmarks/la15.txt"},
        // LA16-LA20: 10 jobs, 10 machines
        {"LA16", 945, "benchmarks/la16.txt"},
        {"LA17", 784, "benchmarks/la17.txt"},
        {"LA18", 848, "benchmarks/la18.txt"},
        {"LA19", 842, "benchmarks/la19.txt"},
        {"LA20", 902, "benchmarks/la20.txt"},
        // LA21-LA25: 15 jobs, 10 machines
        {"LA21", 1046, "benchmarks/la21.txt"},
        {"LA22", 927, "benchmarks/la22.txt"},
        {"LA23", 1032, "benchmarks/la23.txt"},
        {"LA24", 935, "benchmarks/la24.txt"},
        {"LA25", 977, "benchmarks/la25.txt"},
        // LA26-LA30: 20 jobs, 10 machines
        {"LA26", 1218, "benchmarks/la26.txt"},
        {"LA27", 1235, "benchmarks/la27.txt"},
        {"LA28", 1216, "benchmarks/la28.txt"},
        {"LA29", 1152, "benchmarks/la29.txt"},
        {"LA30", 1355, "benchmarks/la30.txt"},
        // LA31-LA35: 30 jobs, 10 machines
        {"LA31", 1784, "benchmarks/la31.txt"},
        {"LA32", 1850, "benchmarks/la32.txt"},
        {"LA33", 1719, "benchmarks/la33.txt"},
        {"LA34", 1721, "benchmarks/la34.txt"},
        {"LA35", 1888, "benchmarks/la35.txt"},
        // LA36-LA40: 15 jobs, 15 machines
        {"LA36", 1268, "benchmarks/la36.txt"},
        {"LA37", 1397, "benchmarks/la37.txt"},
        {"LA38", 1196, "benchmarks/la38.txt"},
        {"LA39", 1233, "benchmarks/la39.txt"},
        {"LA40", 1222, "benchmarks/la40.txt"}
    };
    *benchmarks = la_benchmarks;
    return 40;
}
