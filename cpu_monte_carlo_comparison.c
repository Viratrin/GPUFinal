#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "csv_parser.h"

#define N_PATHS 1000000
#define MAX_ROWS 5000000

double rand_normal(double mean, double stddev) {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return z0 * stddev + mean;
}

void mc_pricer_cpu(
    const double *S0,
    const double *K,
    const double *T,
    const double *r,
    const double *sigma,
    int num_paths,
    int num_contracts,
    double *prices)
{
    for (int i = 0; i < num_contracts; i++) {
        double s0 = S0[i];
        double k = K[i];
        double t = T[i];
        double rate = r[i];
        double vol_input = sigma[i];

        double drift = (rate - 0.5 * vol_input * vol_input) * t;
        double vol = vol_input * sqrt(t);

        double payoff_sum = 0.0;

        for (int j = 0; j < num_paths; j++) {
            double Z = rand_normal(0.0, 1.0);
            double ST = s0 * exp(drift + vol * Z);
            double payoff = fmax(ST - k, 0.0);
            payoff_sum += payoff;
        }

        prices[i] = exp(-rate * t) * (payoff_sum / (double)num_paths);
    }
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s <Options.csv> [max_contracts]\n", argv[0]);
        return 1;
    }

    int max_contracts = MAX_ROWS;

    if (argc >= 3) {
        max_contracts = atoi(argv[2]);

        if (max_contracts <= 0 || max_contracts > MAX_ROWS) {
            printf("Invalid max_contracts value.\n");
            return 1;
        }
    }

    srand(1234);

    OptionContract *contracts = malloc(MAX_ROWS * sizeof(OptionContract));

    if (contracts == NULL) {
        printf("Error: could not allocate contracts array.\n");
        return 1;
    }

    int n = parse_spy_csv(argv[1], contracts, MAX_ROWS);

    if (n <= 0) {
        printf("No valid contracts found.\n");
        free(contracts);
        return 1;
    }

    if (n > max_contracts) {
        n = max_contracts;
    }

    double *S0 = malloc(n * sizeof(double));
    double *K = malloc(n * sizeof(double));
    double *T = malloc(n * sizeof(double));
    double *r = malloc(n * sizeof(double));
    double *sigma = malloc(n * sizeof(double));
    double *prices = malloc(n * sizeof(double));

    if (S0 == NULL || K == NULL || T == NULL || r == NULL || sigma == NULL || prices == NULL) {
        printf("Error: memory allocation failed.\n");

        free(contracts);
        free(S0);
        free(K);
        free(T);
        free(r);
        free(sigma);
        free(prices);

        return 1;
    }

    for (int i = 0; i < n; i++) {
        S0[i] = contracts[i].S0;
        K[i] = contracts[i].K;
        T[i] = contracts[i].T;
        r[i] = R_RISK_FREE;
        sigma[i] = contracts[i].sigma;
    }

    clock_t start = clock();

    mc_pricer_cpu(S0, K, T, r, sigma, N_PATHS, n, prices);

    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    double mae = 0.0;
    int in_spread = 0;

    for (int i = 0; i < n; i++) {
        double error = fabs(prices[i] - contracts[i].mid);
        mae += error / (double)n;

        if (prices[i] >= contracts[i].bid && prices[i] <= contracts[i].ask) {
            in_spread++;
        }
    }

    printf("\n=== CPU Monte Carlo Summary ===\n");
    printf("Total contracts priced : %d\n", n);
    printf("Paths per contract     : %d\n", N_PATHS);
    printf("CPU pricing time       : %.6f sec\n", elapsed);
    printf("Mean absolute error    : $%.4f\n", mae);
    printf("Prices inside spread   : %d / %d  (%.1f%%)\n",
           in_spread, n, 100.0 * in_spread / n);
    printf("Throughput             : %.2f contracts/sec\n", n / elapsed);
    printf("Path throughput        : %.2e paths/sec\n", ((double)n * N_PATHS) / elapsed);

    free(contracts);
    free(S0);
    free(K);
    free(T);
    free(r);
    free(sigma);
    free(prices);

    return 0;
}