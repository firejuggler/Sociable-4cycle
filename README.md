# Sociable-4cycle

A script to search for sociable 4-cycles (aliquot cycles of length 4) using the construction of Borho and Blankenagel, Borho and vom Stein, without starting from amicable pairs. The two seeds are constructed directly from a common part `G`:

```
a1 = G · u0 · p        a2 = G · u2
```

The program matches candidate pairs `(u0, u2)` by sorting `u / sigma(u)`, then checks the necessary divisibility conditions, primality of `p`, and Borho's Theorem 1. Cycles are verified before being written.

## 1. Files

| File | Role |
|---|---|
| `apparie_graines.py` | Main program |
| `borho4.py` | Borho theorems, cycle verification, and C-module loader |
| `borho4c.c` | GMP C module for candidate checks and Theorem 1 sieve |
| `c2io.py` | Reader for `c2_*` amicable-pair files |
| `apparie_gpu.cu` | Shared GPU/CPU core loop for candidate pairs |

Keep the files in the same directory.

## 2. Requirements

- Python 3.8+ with `numpy`, `sympy`, and `gmpy2`:
  ```bash
  pip install numpy sympy gmpy2
  ```
- `gcc` and GMP development headers (for example, `sudo apt install gcc libgmp-dev` on Debian/Ubuntu).
- For CUDA: an NVIDIA GPU and the CUDA toolkit (`nvcc`).
- For the CPU core loop: `g++` with OpenMP. The program compiles `libapparie_cpu.so` automatically when needed.

## 3. Compilation

The GMP C module is built automatically if missing or out of date. To build it manually:

```bash
gcc -O2 -shared -fPIC -o libborho4c.so borho4c.c -lgmp -lm
```

Build the CUDA core loop (change `sm_120` to match your GPU):

```bash
nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
```

Build the CPU core loop manually if desired:

```bash
g++ -O3 -fopenmp -fPIC -shared -x c++ -o libapparie_cpu.so apparie_gpu.cu
```

## 4. Quick test

```bash
python3 apparie_graines.py --G "3^2*5*13*19" --u 1e3 --p0 1e5
```

## 5. Usage

### Sources of common parts `G`

| Option | Meaning |
|---|---|
| `--G "3^4*11^2*13"` | Search one `G` value |
| `--liste file.txt` | Read one `G` per line (the first word is used) |
| `--depuis-catalogue sociable.txt` | Extract common parts from pure Borho 4-cycles; writes `G_catalogue.txt` |
| `--hors-catalogue 1e5` | Generate `G <= 1e5` with `1.8 <= sigma(G)/G < 2`, excluding the catalogue |
| `--depuis-paires c2_80*.txt.gz` | Extract exact common parts from amicable-pair files |
| `--impairs` | Exclude even `G` values |
| `--exclure files...` | Exclude `G` values found in the specified files |

`--exclure` accepts a mixture of ordinary `G` lists and `.faits` files. For list files, the first word on each line is read as `G` (including lines such as `G number`). For `.faits` files, every recorded `G` is excluded regardless of the run parameters recorded there. Thus, a `G` marked done with `--u 1e6` is also excluded from a search with `--u 1e7`; do not include old `.faits` files when you intend to search those values again with a different `--u`.

`G` values and duplicates are compared numerically, not by their written factorization: `3^2*5*19` and `19*3^2*5` are the same value. Files that provide or are generated as the current run's input list are ignored by `--exclure`, preventing the run from excluding its own search set.

Filters apply to the full input source before `--debut N --nombre M` selects a slice.

### Main options

| Option | Default | Meaning |
|---|---:|---|
| `--u` | `1e5` | Upper bound for `u0` and `u2` |
| `--p0` | `1e5` | Lower bound for the prime `p` |
| `--maxp` | `2000` | Number of `p` values tested per candidate pair above the minimum |
| `--K` | `1e7` | Bound `D <= K * G^2` |
| `--budget` | `60` | Seconds per `G`; interrupted searches are written to `<out>.incomplets` |
| `--jobs` | `1` | Number of worker processes |
| `--recyclage N` | `10` | Replace each worker after `N` values of `G` to release memory; with GPU mode, replacement reloads CUDA |
| `--gpu [lib]` | off | Use `libapparie_gpu.so` (or the named library); when enabled without a path, GPU is tried before the CPU library |
| `--sans-noyau` | off | Disable the fast CPU/GPU core loop and use the standard C path (for checking) |
| `--connus f1 f2 ...` | — | Known cycle catalogues used to label cycles as known/new |
| `--out` | `apparie_cycles.txt` | Cycle output file |
| `--refaire` | off | Re-run `G` already marked complete in `<out>.faits` |
| `--hmin H` | off | Exclude `G` for which `sigma(G)/G < H` (for example, `--hmin 1.85`) |

### Core-loop and process behavior

Without `--gpu`, the program uses the OpenMP CPU core loop (`libapparie_cpu.so`) by default and compiles it from `apparie_gpu.cu` if necessary. The number of OpenMP threads per worker is limited to the available CPU cores divided by `--jobs`, so multiple worker processes share the cores instead of each spawning a full thread team. Use `--sans-noyau` (or `--sans-c`) to compare against the standard C implementation. Use `--recyclage N` to periodically replace workers and return memory to the system; increase `N` when individual `G` searches are short to avoid excessive worker/CUDA restarts.

Example using the GPU:

```bash
python3 apparie_graines.py --liste my_G.txt --u 1e6 --p0 1000 --maxp 200 --budget 300 \
  --jobs 4 --gpu --connus sociable.txt --out my_run.txt
```

Example using the default CPU core loop with four workers:

```bash
python3 apparie_graines.py --liste my_G.txt --u 1e6 --p0 1000 --maxp 200 --budget 300 \
  --jobs 4 --out my_run_cpu.txt
```

## 6. Output and resuming

- Cycles are appended to `--out`, with members in cyclic order. A cycle already in the file is not written again.
- `<out>.faits` records completed `G` values together with the run parameters (`u`, `p0`, `K`, `maxp`). Re-running the same command skips completed values; use `--refaire` to repeat them. A single `--G` is always searched again.
- `<out>.incomplets` lists `G` cut off by the time budget.
- `<out>.erreurs` records `G` that raised an error; the run continues.
- Ctrl-C stops the run while preserving completed progress.

The program prints final statistics for candidate pairs, tested primes, theorem checks, and time spent in the filtering stages. Some runtime messages remain in French.

## 7. GPU limits

The GPU loop handles `G < 2^44`, `sigma(G) < 2^46`, `u < 2^24`, `sigma(u) < 2^26`, and `Emax < 2^63`. Other cases use the CPU/C path. For large `G`, explicitly enabling GPU mode may cap `E` at `2^63 - 1`; such a run is marked as bounded and is not recorded as fully completed in `.faits`.

## References

- W. Borho, *Über die Fixpunkte der k-fach iterierten Teilersummenfunktion*, Mitt. Math. Ges. Hamburg 9 (1969), 34–48.
- S. Blankenagel, W. Borho, A. vom Stein, *New amicable four-cycles*, Math. Comp. 72 (2003), 2071–2076.
- D. Moews, catalogue of sociable numbers, `djm.cc/sociable.txt`.
