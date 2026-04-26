/*
 *
 * Compile:
 *      nvcc -O2 --use_fast_math -o main_v1_float main_v1_float.cu csv_parser.c -lm -lcurand
 *
 * Usage:
 *      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
 *      ./main_v1_float Options.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "csv_parser.h"


#define N_PATHS 1000000   /* Monte Carlo paths per contract */
#define BLOCK_SIZE 256   /* threads per block */
#define R_RISK_FREE 0.043f   /* must match parser */
#define MAX_ROWS 5000000
#define MAX_PREVIEW 20

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* 
 * One thread = one contract.
 * Simulates N_PATHS GBM paths and writes the discounted mean payoff to out_prices[idx].
 */
__global__ void mc_pricer_kernel(
    const float * __restrict__ d_S0,
    const float * __restrict__ d_K,
    const float * __restrict__ d_T,
    const float * __restrict__ d_sigma,
    float r,
    int n_paths,
    int n_contracts,
    float * __restrict__ d_prices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_contracts) return;

    // load this contract's parameters into registers
    float s0 = d_S0[idx];
    float k = d_K[idx];
    float t = d_T[idx];
    float sigma = d_sigma[idx];

    float drift = (r - 0.5f * sigma * sigma) * t;
    float vol = sigma * sqrtf(t);

    // initialise per-thread RNG state
    curandState state;
    curand_init(1234ULL + (unsigned long long)idx, 0ULL, 0ULL, &state);

    // simulate N_PATHS paths and accumulate payoffs
    float payoff_sum = 0.0f;
    for (int i = 0; i < n_paths; i++) {
        float Z = curand_normal(&state); /* Z ~ N(0,1)        */
        float ST = s0 * expf(drift + vol * Z); /* GBM terminal price */
        float payoff = ST - k;
        if (payoff > 0.0f) payoff_sum += payoff;
    }

    // discount the average payoff back to present value
    d_prices[idx] = expf(-r * t) * (payoff_sum / (float)n_paths);
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("\nNo CSV file provided.\n");
        printf("Usage: %s <spy_options.csv>\n", argv[0]);
        return 0;
    }

    // parse CSV
    printf("\n=== Kaggle SPY Dataset Pricing ===\n");

    OptionContract *contracts = (OptionContract *)malloc(MAX_ROWS * sizeof(OptionContract));
    if (!contracts) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    int n = parse_spy_csv(argv[1], contracts, MAX_ROWS);
    if (n <= 0) { free(contracts); return 1; }

    // extract the data from the struct-of-arrays
    float *h_S0 = (float *)malloc(n * sizeof(float));
    float *h_K = (float *)malloc(n * sizeof(float));
    float *h_T = (float *)malloc(n * sizeof(float));
    float *h_sigma = (float *)malloc(n * sizeof(float));
    float *h_prices= (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_S0[i] = (float)contracts[i].S0;
        h_K[i] = (float)contracts[i].K;
        h_T[i] = (float)contracts[i].T;
        h_sigma[i] = (float)contracts[i].sigma;
    }

    // allocate device memory
    float *d_S0, *d_K, *d_T, *d_sigma, *d_prices;
    CUDA_CHECK(cudaMalloc(&d_S0, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prices, n * sizeof(float)));

    // copy host -> device
    CUDA_CHECK(cudaMemcpy(d_S0, h_S0, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, h_T, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma, n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching kernel: %d blocks x %d threads = %d total threads\n",
           blocks, BLOCK_SIZE, blocks * BLOCK_SIZE);
    printf("Paths per contract: %d\n\n", N_PATHS);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    mc_pricer_kernel<<<blocks, BLOCK_SIZE>>>(
        d_S0, d_K, d_T, d_sigma,
        R_RISK_FREE, N_PATHS, n,
        d_prices
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());   /* catch any kernel launch errors */

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel finished in %.2f ms\n\n", ms);

    // copy results back
    CUDA_CHECK(cudaMemcpy(h_prices, d_prices, n * sizeof(float), cudaMemcpyDeviceToHost));

    // print results and compute error vs market mid
    int preview = n < MAX_PREVIEW ? n : MAX_PREVIEW;
    printf("\n=== Results (first %d contracts) ===\n", preview);
    printf("%-6s  %-8s  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "Idx", "S0", "K", "T(yrs)", "MC Price", "Market Mid", "Error", "In Spread?");
    printf("%s\n", "--------------------------------------------------------------------------------");

    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s_results.csv", argv[0]);
    FILE *csv_out = fopen(csv_path, "w");
    if (!csv_out) fprintf(stderr, "WARNING: could not open %s\n", csv_path);
    else          fprintf(csv_out, "Idx,S0,K,T_yrs,MC_Price,Market_Mid,Error,In_Spread\n");

    double mae = 0.0;
    int    in_spread   = 0;

    for (int i = 0; i < n; i++) {
        double mc = (double)h_prices[i];
        double mid = contracts[i].mid;
        double error = fabs(mc - mid);
        mae += error / n;

        int inside = (mc >= contracts[i].bid && mc <= contracts[i].ask);
        if (inside) in_spread++;

        if (csv_out)
            fprintf(csv_out, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n",
                    i, contracts[i].S0, contracts[i].K, contracts[i].T,
                    mc, mid, error, inside ? "YES" : "NO");

        if (i < preview) {
            printf("%-6d  %-8.4f  %-8.4f  %-8.4f  %-10.4f  %-10.4f  %-10.4f  %s\n",
                   i,
                   contracts[i].S0, contracts[i].K, contracts[i].T,
                   mc, mid, error,
                   inside ? "YES" : "NO");
        }
    }

    if (csv_out) { fclose(csv_out); printf("Results written to %s\n", csv_path); }

    printf("\n=== Summary ===\n");
    printf("Total contracts priced : %d\n", n);
    printf("Mean absolute error    : $%.4f\n", mae);
    printf("Prices inside spread   : %d / %d  (%.1f%%)\n", in_spread, n, 100.0 * in_spread / n);
    printf("Throughput             : %.2f contracts/sec\n", n / (ms / 1000.0));

    // cleanup
    cudaFree(d_S0);
    cudaFree(d_K);
    cudaFree(d_T);
    cudaFree(d_sigma);
    cudaFree(d_prices);
    free(h_S0);
    free(h_K);
    free(h_T);
    free(h_sigma);
    free(h_prices);
    free(contracts);

    return 0;
}


// === Summary ===
// Total contracts priced : 500000
// Mean absolute error    : $2.6841
// Prices inside spread   : 58662 / 500000  (11.7%)
// Throughput             : 484416.71 contracts/sec

// real    0m1.981s
// user    0m1.538s
// sys     0m0.401s