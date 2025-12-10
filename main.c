#include "fjsp_aco.h"

// =========================================================
//  MAIN EXECUTION - COMPREHENSIVE TESTING
// =========================================================

int main(int argc, char *argv[]) {
    
    // Parse command line arguments
    int test_brandimarte = 1;
    int test_lawrence = 1;
    int mk_start = 1, mk_end = 10;
    int la_start = 1, la_end = 40;
    
    if (argc > 1) {
        if (strcmp(argv[1], "mk") == 0) {
            test_lawrence = 0;
            if (argc > 2) mk_start = atoi(argv[2]);
            if (argc > 3) mk_end = atoi(argv[3]);
        } else if (strcmp(argv[1], "la") == 0) {
            test_brandimarte = 0;
            if (argc > 2) la_start = atoi(argv[2]);
            if (argc > 3) la_end = atoi(argv[3]);
        }
    }
    
    // Configuration
    Config cfg = {
        .num_ants = 30,
        .max_iter = 100,
        .stop_patience = 50,
        .alpha_start = 1.0, .alpha_end = 4.0,
        .beta_start = 4.0, .beta_end = 1.0,
        .rho_min = 0.1, .rho_max = 0.3,
        .q0 = 0.5,
        .ls_max_attempts = 50000,
        .independent_runs = 10
    };
    
    srand(time(NULL));
    
    printf("================================================================================\n");
    printf("            FJSP-ACO COMPREHENSIVE BENCHMARK TESTING\n");
    printf("================================================================================\n\n");
    
    // Get benchmarks
    BenchmarkInstance *mk_benchmarks = NULL;
    BenchmarkInstance *la_benchmarks = NULL;
    int num_mk = get_brandimarte_benchmarks(&mk_benchmarks);
    int num_la = get_lawrence_benchmarks(&la_benchmarks);
    
    // Calculate total instances to test
    int total_instances = 0;
    if (test_brandimarte) total_instances += (mk_end - mk_start + 1);
    if (test_lawrence) total_instances += (la_end - la_start + 1);
    
    // Allocate results array
    RunResult *all_results = (RunResult*)calloc(total_instances, sizeof(RunResult));
    int result_idx = 0;
    
    // =========================================================
    //  TEST BRANDIMARTE INSTANCES (Mk01-Mk10)
    // =========================================================
    
    if (test_brandimarte) {
        printf("\n--- BRANDIMARTE INSTANCES (FJSP) ---\n\n");
        
        for (int i = mk_start - 1; i < mk_end && i < num_mk; i++) {
            BenchmarkInstance *bench = &mk_benchmarks[i];
            
            printf("Testing %s (BKS=%d)...\n", bench->name, bench->bks);
            
            // Load data
            FJSP_Data *data = load_fjsp_data(bench->filepath);
            if (!data) {
                fprintf(stderr, "  ERROR: Could not load %s\n", bench->filepath);
                continue;
            }
            
            // Initialize result
            RunResult *result = &all_results[result_idx++];
            strcpy(result->instance_name, bench->name);
            result->bks = bench->bks;
            result->num_runs = cfg.independent_runs;
            result->convergence_len = cfg.max_iter;
            result->run_scores = (double*)malloc(cfg.independent_runs * sizeof(double));
            result->run_times = (double*)malloc(cfg.independent_runs * sizeof(double));
            result->convergence = (double**)malloc(cfg.independent_runs * sizeof(double*));
            result->convergence_times = (double*)malloc(cfg.max_iter * sizeof(double));
            
            // Run experiments
            for (int r = 0; r < cfg.independent_runs; r++) {
                result->convergence[r] = (double*)malloc(cfg.max_iter * sizeof(double));
                double *time_curve = (r == 0) ? result->convergence_times : NULL;
                
                clock_t start = clock();
                
                Optimizer *opt = create_optimizer(data, &cfg);
                if (!opt) continue;
                
                double score = optimize_with_time(opt, result->convergence[r], time_curve);
                
                clock_t end = clock();
                double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
                
                result->run_scores[r] = score;
                result->run_times[r] = elapsed;
                
                double gap = ((score - bench->bks) / bench->bks) * 100.0;
                printf("  Run %2d: Score=%6.0f, Gap=%6.2f%%, Time=%.2fs\n", r+1, score, gap, elapsed);
                
                free_optimizer(opt);
            }
            
            // Calculate statistics
            calculate_statistics(result);
            
            printf("  Summary: Best=%.0f, Mean=%.2f, Gap_Best=%.2f%%, Gap_Mean=%.2f%%\n\n",
                   result->best_score, result->mean_score, result->gap_best, result->gap_mean);
            
            // Write convergence for this instance
            char conv_filename[128];
            snprintf(conv_filename, sizeof(conv_filename), "convergence_%s.csv", bench->name);
            write_convergence_csv(conv_filename, result);
            
            free_fjsp_data(data);
        }
    }
    
    // =========================================================
    //  TEST LAWRENCE INSTANCES (LA01-LA40)
    // =========================================================
    
    if (test_lawrence) {
        printf("\n--- LAWRENCE INSTANCES (JSP) ---\n\n");
        
        for (int i = la_start - 1; i < la_end && i < num_la; i++) {
            BenchmarkInstance *bench = &la_benchmarks[i];
            
            printf("Testing %s (BKS=%d)...\n", bench->name, bench->bks);
            
            // Load data (standard JSP format)
            FJSP_Data *data = load_standard_jss(bench->filepath);
            if (!data) {
                fprintf(stderr, "  ERROR: Could not load %s\n", bench->filepath);
                continue;
            }
            
            // Initialize result
            RunResult *result = &all_results[result_idx++];
            strcpy(result->instance_name, bench->name);
            result->bks = bench->bks;
            result->num_runs = cfg.independent_runs;
            result->convergence_len = cfg.max_iter;
            result->run_scores = (double*)malloc(cfg.independent_runs * sizeof(double));
            result->run_times = (double*)malloc(cfg.independent_runs * sizeof(double));
            result->convergence = (double**)malloc(cfg.independent_runs * sizeof(double*));
            result->convergence_times = (double*)malloc(cfg.max_iter * sizeof(double));
            
            // Run experiments
            for (int r = 0; r < cfg.independent_runs; r++) {
                result->convergence[r] = (double*)malloc(cfg.max_iter * sizeof(double));
                double *time_curve = (r == 0) ? result->convergence_times : NULL;
                
                clock_t start = clock();
                
                Optimizer *opt = create_optimizer(data, &cfg);
                if (!opt) continue;
                
                double score = optimize_with_time(opt, result->convergence[r], time_curve);
                
                clock_t end = clock();
                double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
                
                result->run_scores[r] = score;
                result->run_times[r] = elapsed;
                
                double gap = ((score - bench->bks) / bench->bks) * 100.0;
                printf("  Run %2d: Score=%6.0f, Gap=%6.2f%%, Time=%.2fs\n", r+1, score, gap, elapsed);
                
                free_optimizer(opt);
            }
            
            // Calculate statistics
            calculate_statistics(result);
            
            printf("  Summary: Best=%.0f, Mean=%.2f, Gap_Best=%.2f%%, Gap_Mean=%.2f%%\n\n",
                   result->best_score, result->mean_score, result->gap_best, result->gap_mean);
            
            // Write convergence for this instance
            char conv_filename[128];
            snprintf(conv_filename, sizeof(conv_filename), "convergence_%s.csv", bench->name);
            write_convergence_csv(conv_filename, result);
            
            free_fjsp_data(data);
        }
    }
    
    // =========================================================
    //  WRITE OUTPUT FILES
    // =========================================================
    
    printf("\n================================================================================\n");
    printf("                         WRITING OUTPUT FILES\n");
    printf("================================================================================\n\n");
    
    // Write CSV results
    write_csv_results("results.csv", all_results, result_idx);
    
    // Write text report
    write_text_report("report.txt", all_results, result_idx, &cfg);
    
    // =========================================================
    //  FINAL SUMMARY
    // =========================================================
    
    printf("\n================================================================================\n");
    printf("                         FINAL SUMMARY\n");
    printf("================================================================================\n\n");
    
    double total_gap_best = 0.0;
    double total_gap_mean = 0.0;
    int optimal_count = 0;
    
    printf("%-12s %6s %8s %8s %10s %10s\n", "Instance", "BKS", "Best", "Mean", "Gap_B(%)", "Gap_M(%)");
    printf("------------------------------------------------------------------------\n");
    
    for (int i = 0; i < result_idx; i++) {
        RunResult *res = &all_results[i];
        printf("%-12s %6d %8.0f %8.2f %10.2f %10.2f\n",
               res->instance_name, res->bks, res->best_score, 
               res->mean_score, res->gap_best, res->gap_mean);
        
        total_gap_best += res->gap_best;
        total_gap_mean += res->gap_mean;
        if (res->gap_best <= 0.01) optimal_count++;
    }
    
    printf("------------------------------------------------------------------------\n");
    printf("%-12s %6s %8s %8s %10.2f %10.2f\n",
           "AVERAGE", "", "", "", total_gap_best / result_idx, total_gap_mean / result_idx);
    printf("\nOptimal solutions: %d/%d (%.1f%%)\n", optimal_count, result_idx, (100.0*optimal_count)/result_idx);
    
    // Cleanup
    for (int i = 0; i < result_idx; i++) {
        free_run_result(&all_results[i]);
    }
    free(all_results);
    
    printf("\nAll experiments complete!\n");
    printf("Output files: results.csv, report.txt, convergence_*.csv\n");
    
    return 0;
}
