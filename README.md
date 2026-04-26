# GPUFinal

| Version | Change from baseline |
|---|---|
| `main_v1_double` | baseline — one thread/contract, `double`, no optimizations |
| `main_v1_float` | `double` → `float`, `--use_fast_math`, seed-based `curand_init` |
| `main_v2` | kernel restructured: one block/contract, stride loop + shared-memory reduction |
| `main_v3` | `malloc` → `cudaMallocHost` for all host arrays (pinned memory) |
| `main_v4` | v2 + v3 |
