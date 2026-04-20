#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double rand_normal(double mean, double stddev);
void time_series_gen(double* time_series, double S0, double mu, double sigma, double T, int N);