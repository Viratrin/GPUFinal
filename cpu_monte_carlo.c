#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "csv_parser.h"

#define N_PATHS 100000

double rand_normal(double mean, double stddev) {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

void mc_pricer_cpu(
    const double* restrict d_S0, 
    const double* restrict d_K, 
    const double* restrict d_T, 
    const double* restrict d_r, 
    const double* restrict d_sigma, 
    const double* restrict d_results, 
    int num_paths, 
    int num_contracts,
    double* restrict d_prices) 
{
    for (int i = 0; i < num_contracts; i++) {
        double S0 = d_S0[i];
        double K = d_K[i];
        double T = d_T[i];
        double r = d_r[i];
        double sigma = d_sigma[i];
        double payoff_sum = 0.0;

        double drift = (r - 0.5 * sigma * sigma) * T;
        double vol = sigma * sqrt(T);

        for (int j = 0; j < num_paths; j++) {
            double Z = rand_normal(0, 1);
            double ST = S0 * exp(drift + vol * Z);
            double payoff = fmax(ST - K, 0.0);
            payoff_sum += payoff;
        }
        d_prices[i] = exp(-r * T) * (payoff_sum / (double)num_paths);
    }
}

int main(){
    // Example usage of the Monte Carlo pricer
    int num_contracts = 1;
    double S0 = 100.0; // Stock price
    double K = 100.0;  // Strike price
    double T = 1.0;    // Time to maturity in years
    double r = 0.05;   // Risk-free interest rate
    double sigma = 0.2; // Volatility

    double* d_S0 = (double*)malloc(num_contracts * sizeof(double));
    double* d_K = (double*)malloc(num_contracts * sizeof(double));
    double* d_T = (double*)malloc(num_contracts * sizeof(double));
    double* d_r = (double*)malloc(num_contracts * sizeof(double));
    double* d_sigma = (double*)malloc(num_contracts * sizeof(double));
    double* d_prices = (double*)malloc(num_contracts * sizeof(double));

    d_S0[0] = S0;
    d_K[0] = K;
    d_T[0] = T;
    d_r[0] = r;
    d_sigma[0] = sigma;

    mc_pricer_cpu(d_S0, d_K, d_T, d_r, d_sigma, NULL, N_PATHS, num_contracts, d_prices);

    printf("The price of the European call option is: %f\n", d_prices[0]);

    free(d_S0);
    free(d_K);
    free(d_T);
    free(d_r);
    free(d_sigma);
    free(d_prices);

    return 0;
}