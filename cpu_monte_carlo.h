#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "csv_parser.h"

double rand_normal(double mean, double stddev);
void mc_pricer_cpu(
    const double* d_S0, 
    const double* d_K, 
    const double* d_T, 
    const double* d_r, 
    const double* d_sigma, 
    const double* d_results, 
    int num_paths, 
    int num_contracts, 
    double* d_prices
);