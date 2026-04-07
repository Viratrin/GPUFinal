#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_ROWS      500000
#define MAX_LINE_LEN  4096
#define MAX_COLS      64
#define R_RISK_FREE   0.043

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

// trim whitespace
static char *trim(char *s) {
    if (!s){
        return s;
    }
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n'){
        s++;
    }
    char *end = s + strlen(s) - 1;
    while (end > s && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')){
        *end-- = '\0';
    }
    return s;
}

// split csv line
static int split_csv_line(char *line, char **tokens, int max_tokens) {
    int count = 0;
    char *p = line;

    while (count < max_tokens) {
        if (*p == '"') {
            p++;
            tokens[count++] = p;
            while (*p && *p != '"'){
                p++;
            } 
            if (*p == '"'){
                *p++ = '\0';
            }
            if (*p == ','){
                p++;
            } 
        } else {
            tokens[count++] = p;
            while (*p && *p != ','){
                p++;
            }
            if (*p == ',') {
                *p++ = '\0';
            } else {
                break;
            }
        }
    }
    return count;
}

// find column by name
static int find_col(char **headers, int n, const char *name) {
    for (int i = 0; i < n; i++){
        if (strcmp(trim(headers[i]), name) == 0){
            return i; 
        }
    }
        
    return -1;
}


int parse_spy_csv(const char *filepath,
                  OptionContract *contracts,
                  int max_contracts)
{
    FILE *fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open '%s'\n", filepath); return -1; }

    char  line[MAX_LINE_LEN];
    char *tokens[MAX_COLS];

    /* header row */
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "ERROR: file is empty\n");
        fclose(fp); return -1;
    }

    char header_buf[MAX_LINE_LEN];
    strncpy(header_buf, line, MAX_LINE_LEN - 1);
    int n_headers = split_csv_line(header_buf, tokens, MAX_COLS);
    for (int i = 0; i < n_headers; i++) tokens[i] = trim(tokens[i]);

    int col_S0  = find_col(tokens, n_headers, "[UNDERLYING_LAST]");
    int col_K   = find_col(tokens, n_headers, "[STRIKE]");
    int col_DTE = find_col(tokens, n_headers, "[DTE]");
    int col_CIV = find_col(tokens, n_headers, "[C_IV]");
    int col_BID = find_col(tokens, n_headers, "[C_BID]");
    int col_ASK = find_col(tokens, n_headers, "[C_ASK]");

    struct { const char *name; int idx; } req[] = {
        {"[UNDERLYING_LAST]", col_S0 }, {"[STRIKE]", col_K  },
        {"[DTE]",             col_DTE}, {"[C_IV]",   col_CIV},
        {"[C_BID]",           col_BID}, {"[C_ASK]",  col_ASK},
    };
    int missing = 0;
    for (int i = 0; i < 6; i++) {
        if (req[i].idx == -1) {
            fprintf(stderr, "ERROR: column '%s' not found\n", req[i].name);
            missing = 1;
        }
    }
    if (missing) { fclose(fp); return -1; }

    /* highest column index we need — used to validate row width */
    int max_col = col_S0;
    if (col_K   > max_col) max_col = col_K;
    if (col_DTE > max_col) max_col = col_DTE;
    if (col_CIV > max_col) max_col = col_CIV;
    if (col_BID > max_col) max_col = col_BID;
    if (col_ASK > max_col) max_col = col_ASK;

    int loaded = 0, skipped = 0, row_num = 1;

    while (fgets(line, sizeof(line), fp) && loaded < max_contracts) {
        row_num++;
        if (strlen(trim(line)) == 0){
            continue;
        }

        char row_buf[MAX_LINE_LEN];
        strncpy(row_buf, line, MAX_LINE_LEN - 1);

        char *fields[MAX_COLS];
        int n_fields = split_csv_line(row_buf, fields, MAX_COLS);

        if (n_fields <= max_col) {
            fprintf(stderr, "WARNING: row %d has only %d fields, skipping\n",
                    row_num, n_fields);
            skipped++;
            continue;
        }

        double S0    = atof(trim(fields[col_S0]));
        double K     = atof(trim(fields[col_K]));
        double DTE   = atof(trim(fields[col_DTE]));
        double sigma = atof(trim(fields[col_CIV]));
        double bid   = atof(trim(fields[col_BID]));
        double ask   = atof(trim(fields[col_ASK]));

        // filter
        if (S0    <= 0.0) { skipped++; continue; }  /* bad underlying     */
        if (K     <= 0.0) { skipped++; continue; }  /* bad strike         */
        if (DTE   <= 0.0) { skipped++; continue; }  /* same-day / expired */
        if (sigma <= 0.0) { skipped++; continue; }  /* missing IV         */
        if (bid   <= 0.0) { skipped++; continue; }  /* no market          */
        if (ask   <= bid) { skipped++; continue; }  /* crossed spread     */

        OptionContract *c = &contracts[loaded++];
        c->S0    = S0;
        c->K     = K;
        c->T     = DTE / 365.0;
        c->sigma = sigma;
        c->r     = R_RISK_FREE;
        c->bid   = bid;
        c->ask   = ask;
        c->mid   = (bid + ask) / 2.0;
    }

    fclose(fp);
    printf("Done: %d loaded, %d skipped\n", loaded, skipped);
    return loaded;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <spy_options.csv>\n", argv[0]);
        return 1;
    }

    OptionContract *contracts = malloc(MAX_ROWS * sizeof(OptionContract));
    if (!contracts) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    int n = parse_spy_csv(argv[1], contracts, MAX_ROWS);
    if (n <= 0) { free(contracts); return 1; }

    int preview = n < 5 ? n : 5;
    printf("\nFirst %d contracts:\n", preview);
    printf("  %-10s %-10s %-8s %-8s %-8s %-8s %-8s\n",
           "S0", "K", "T(yrs)", "sigma", "bid", "ask", "mid");
    printf("  ------------------------------------------------------------------\n");
    for (int i = 0; i < preview; i++) {
        OptionContract *c = &contracts[i];
        printf("  %-10.4f %-10.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f\n",
               c->S0, c->K, c->T, c->sigma, c->bid, c->ask, c->mid);
    }

    free(contracts);
    return 0;
}