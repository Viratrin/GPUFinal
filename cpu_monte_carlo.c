#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define NUM_SIMULATIONS 100000

double rand_normal(double mean, double stddev) {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

void time_series_gen(double* time_series, double S0, double mu, double sigma, double T, int N){
    double dt = T / N;
    time_series[0] = S0;
    for (int i = 1; i < N; i++) {
        double Z = rand_normal(0, 1);
        time_series[i] = time_series[i-1] * exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
    }
}

int main(){
    double S0 = 100.0; // Stock price
    double K = 100.0;  // Strike price
    int T = 1;         // Time to maturity in years
    int N = 252;       // Number of time steps
    double r = 0.05;   // Risk-free interest rate
    double v = 0.2;    // Volatility
    double payoff_sum = 0.0;

    srand(time(NULL));
    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        double* time_series = (double*)malloc(N * sizeof(double));
        time_series_gen(time_series, S0, r, v, T, N);
        double payoff = time_series[N-1] - K;
        payoff_sum += payoff;
        free(time_series);
    }
    printf("Monte Carlo Estimate: %f\n", payoff_sum / NUM_SIMULATIONS);
    return 0;
}