# Makefile for FJSP-ACO

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

# Target executable
TARGET = fjsp_aco

# Source files
SRCS = main.c fjsp_aco.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c fjsp_aco.h
	$(CC) $(CFLAGS) -c $< -o $@

# Run all benchmarks
run: $(TARGET)
	./$(TARGET)

# Run only Brandimarte instances
run_mk: $(TARGET)
	./$(TARGET) mk 1 10

# Run only Lawrence instances
run_la: $(TARGET)
	./$(TARGET) la 1 40

# Run specific range of Mk instances
run_mk_range: $(TARGET)
	./$(TARGET) mk $(START) $(END)

# Run specific range of LA instances
run_la_range: $(TARGET)
	./$(TARGET) la $(START) $(END)

# Generate plots (requires Python with matplotlib)
plot:
	python3 plot_convergence.py

# Combined plots
plot_combined:
	python3 plot_convergence.py --combined

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

# Clean all generated files
cleanall: clean
	rm -f results.csv report.txt convergence_*.csv
	rm -rf plots/

# Help
help:
	@echo "FJSP-ACO Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make            - Build the program"
	@echo "  make run        - Run all benchmarks (Mk01-Mk10 + LA01-LA40)"
	@echo "  make run_mk     - Run Brandimarte instances only"
	@echo "  make run_la     - Run Lawrence instances only"
	@echo "  make run_mk_range START=1 END=5 - Run Mk01 to Mk05"
	@echo "  make run_la_range START=1 END=10 - Run LA01 to LA10"
	@echo "  make plot       - Generate individual convergence plots"
	@echo "  make plot_combined - Generate combined convergence plots"
	@echo "  make clean      - Remove build files"
	@echo "  make cleanall   - Remove all generated files"
	@echo ""
	@echo "Output files:"
	@echo "  results.csv        - CSV with all results"
	@echo "  report.txt         - Detailed statistical report"
	@echo "  convergence_*.csv  - Convergence data for each instance"
	@echo "  plots/             - Generated plot images"

.PHONY: all run run_mk run_la run_mk_range run_la_range plot plot_combined clean cleanall help
