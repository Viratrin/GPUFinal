#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

#define R_RISK_FREE 0.043

typedef struct {
    double S0;
    double K;
    double T;
    double sigma;
    double r;
    double bid;
    double ask;
    double mid;
} OptionContract;

int parse_spy_csv(const char *filepath, OptionContract *contracts, int max_contracts);

#ifdef __cplusplus
}
#endif

#endif /* CSV_PARSER_H */