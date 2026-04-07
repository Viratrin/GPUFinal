#include <stdio.h>
#include <math.h>

double norm_cdf(double x){
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

double black_scholes(double S, double K, double T, double r, double v){
    double d1 = (log(S / K) + (r + 0.5 * v * v) * T) / (v * sqrt(T));
    double d2 = d1 - v * sqrt(T);
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
}

int main() {
    double S = 100.0; // Stock price
    double K = 100.0; // Strike price
    double T = 1.0;   // Time to maturity in years
    double r = 0.05;  // Risk-free interest rate
    double v = 0.2;   // Volatility

    double option_price = black_scholes(S, K, T, r, v);
    printf("The price of the European call option is: %f\n", option_price);

    return 0;
}