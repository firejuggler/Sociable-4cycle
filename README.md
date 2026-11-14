[README_apparie.txt](https://github.com/user-attachments/files/34837372/README_apparie.txt)
# Sociable-4cycle
A script to find sociable4 cycle (aliquot)
# apparie_graines — searching for aliquot 4-cycles of Borho type

`apparie_graines.py` searches for Borho-type sociable 4-cycles without starting from amicable pairs.
The program builds the two seeds directly from a common part `G` and matches pairs `(u0, u2)` by the
value of `u / sigma(u)` before testing the final Borho conditions.

For a common part `G` (an odd integer with `1 < sigma(G) / G < 2`), the seeds are

```
a1 = G · u0 · p        a2 = G · u2
```

with `u0, u2 <= --u`, `gcd(u0, G) = gcd(u2, G) = 1`, and a large unknown prime `p >= --p0`.
The program filters candidate pairs `(u0, u2)` by their `y = u / sigma(u)` values, then checks the
exact divisibility conditions, the primality of `p`, and the full Borho Theorem 1 before emitting a cycle.

---

## 1. Files

| File | Role |
|---|---|
| `apparie_graines.py` | main program |
| `borho4.py` | Borho theorem code, cycle verification and C loader |
| `borho4c.c` | C implementation for the final sieve and checks |
| `c2io.py` | reader for `c2_*` amicable-pair files |
| `apparie_gpu.cu` | GPU/CPU core loop for the `(u0, u2, p)` search |

Put all files in the same directory.

## 2. Requirements

- Python 3.8+ with `numpy`, `sympy`, `gmpy2`
  ```bash
  pip install numpy sympy gmpy2
  ```
- `gcc` and the GMP headers (`libgmp-dev` on Debian/Ubuntu)
- CUDA toolkit for the GPU build (`nvcc`) if using the CUDA kernel
- `g++` with OpenMP for the CPU fallback build

## 3. Compilation

Build the C module (done automatically if missing):

```bash
gcc -O2 -shared -fPIC -o libborho4c.so borho4c.c -lgmp -lm
```

Build the GPU core loop:

```bash
nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
```

For CPU validation:

```bash
g++ -O3 -fopenmp -fPIC -shared -x c++ -o libapparie_cpu.so apparie_gpu.cu
```

## 4. Quick test

```bash
python3 apparie_graines.py --G "3^2*5*13*19" --u 1e3 --p0 1e5
```

This should return a 4-cycle quickly and is useful as a smoke test.

## 5. Usage

### Where the common parts G come from

| Option | Meaning |
|---|---|
| `--G "3^4*11^2*13"` | a single common part `G` |
| `--liste file.txt` | one `G` per line; the first word counts |
| `--depuis-catalogue sociable.txt` | list the `G` values found in the Borho 4-cycles of a catalogue |
| `--hors-catalogue 1e5` | all `G <= 1e5` with `1.8 <= sigma(G)/G < 2` not already in the catalogue |
| `--depuis-paires c2_80*.txt.gz` | exact common parts extracted from the amicable-pair lists |
| `--impairs` | exclude even `G` |
| `--exclure file1 file2 ...` | exclude `G` already present in these files |

`--exclure` accepts a mix of:

- plain `G` lists: the first word on each line is interpreted as a number or factorization,
- `.faits` files: any `G` recorded there is excluded regardless of the run parameters used to produce it.

Important notes:

- `G` values are compared numerically, not by their textual writing. For example,
  `3^2*5*19` and `19*3^2*5` are treated as the same `G`.
- Duplicates inside the same input list are removed by numeric value, not text.
- Files generated during the current run are ignored by the exclusion filter to avoid self-excluding the
  search set (for example generated `G_catalogue.txt`, `G_paires.txt`, or the active output list).

`--debut N --nombre M` restricts the scan to the slice of the list from line `N` to `N + M - 1`.

### Main options

| Option | Default | Meaning |
|---|---|---|
| `--u` | `1e5` | `u0, u2 <= u` |
| `--p0` | `1e5` | lower bound for the prime `p` |
| `--K` | `1e7` | keep `D <= K * G^2` |
| `--budget` | `60` | time budget per `G`; cut `G` are exported to `<out>.incomplets` |
| `--out` | `apparie_cycles.txt` | output file |
| `--jobs` | `1` | worker processes |
| `--gpu [lib]` | off | run the core loop on GPU (or use the specified library) |
| `--connus f1 f2 ...` | — | known cycle catalogues used for duplicate detection |
| `--refaire` | — | rerun `G` already recorded in `.faits` |
| `--hmin 1.85` | — | exclude `G` with `sigma(G)/G < 1.85` |

### Core-process improvement

The GPU/CPU core loop was tightened to reduce overhead in the repeated `(u0, u2, p)` filtering step:

- the work is staged in bounded chunks,
- output buffers are reused efficiently,
- the thread pool is kept under control to avoid runaway scheduling overhead,
- the C/CUDA core processes the fast arithmetic and only returns the surviving candidates to Python.

This keeps the expensive Borho checks concentrated in the low-level kernel while the Python layer remains
responsible only for verifying and writing the final cycles.

### Recommended settings

```bash
python3 apparie_graines.py --liste my_G.txt --u 1e6 --p0 1000 --maxp 200 --budget 300 \
    --jobs 4 --gpu --connus sociable.txt --out my_run.txt
```

For a second pass over promising `G` values, a larger `--u` often pays off:

```bash
python3 apparie_graines.py --liste productive_G.txt --u 1e7 --p0 1000 --maxp 200 --budget 600 \
    --jobs 4 --gpu --connus sociable.txt --out my_run_u1e7.txt
```

## 6. Output

- cycles are appended to `--out`
- completed `G` values are stored in `<out>.faits`
- timed-out or cut `G` values are written to `<out>.incomplets`
- run-time errors are written to `<out>.erreurs`
- the program prints a final statistics block with the counts of candidate pairs, tested primes, survivors,
  and time spent in the filtering stages

## 7. Practical advice

- prefer odd `G` with `sigma(G)/G` between `1.85` and `2`
- small `G` values are usually the most productive
- good searches often come from the exact common parts of known cycles and their nearby variants
- if the search stalls, try a different `--u`, `--p0`, or a narrower exclusion list

## 8. Checking a cycle independently

```python
from sympy import divisor_sigma
c = [n1, n2, n3, n4]
print(all(divisor_sigma(c[i]) - c[i] == c[(i + 1) % 4] for i in range(4)))
```

## References

- W. Borho, *Über die Fixpunkte der k-fach iterierten Teilersummenfunktion*,
  Mitt. Math. Ges. Hamburg 9 (1969), 34–48.
- S. Blankenagel, W. Borho, A. vom Stein, *New amicable four-cycles*,
  Math. Comp. 72 (2003), 2071–2076.
- D. Moews, catalogue of sociable numbers, `djm.cc/sociable.txt`.
