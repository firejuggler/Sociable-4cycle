[README_apparie.txt](https://github.com/user-attachments/files/34837372/README_apparie.txt)
# Sociable-4cycle
A script to search for sociable 4-cycles (aliquot cycles of length 4).

# apparie_graines — searching for aliquot 4-cycles of Borho type

`apparie_graines.py` searches for Borho-type sociable 4-cycles without starting from amicable pairs.
The two seeds are built directly from a common part `G`, and candidate pairs `(u0, u2)` are matched by
`u / sigma(u)` before the final Borho conditions are tested.

For a common part `G` (an odd integer with `1 < sigma(G) / G < 2`), the seeds are

```
a1 = G · u0 · p        a2 = G · u2
```

where `u0, u2 <= --u` are coprime to `G`, and `p >= --p0` is a large unknown prime.
The program finds candidate `(u0, u2)` pairs by sorting all `u <= U` by `y(u) = u / sigma(u)`, then
checks the exact divisibility condition (Theorem 2), the primality of `p`, and finally the full
Borho Theorem 1 before emitting a 4-cycle.

---

## 1. Files

| File | Role |
|---|---|
| `apparie_graines.py` | main program |
| `borho4.py` | Borho theorem code and cycle verification |
| `borho4c.c` | C module used in the sieve and final checks |
| `c2io.py` | reader for the `c2_*` amicable-pair files |
| `apparie_gpu.cu` | GPU / CPU inner loop for the `(u0, u2, p)` search |

Put all files in the same directory.

## 2. Requirements

- Python 3.8+ with `numpy`, `sympy`, and `gmpy2`
  ```bash
  pip install numpy sympy gmpy2
  ```
- `gcc` and the GMP headers (`libgmp-dev` on Debian/Ubuntu)
- CUDA toolkit (`nvcc`) if using the GPU build
- `g++` with OpenMP for the CPU fallback build

## 3. Compilation

Build the C module:

```bash
gcc -O2 -shared -fPIC -o libborho4c.so borho4c.c -lgmp -lm
```

Build the GPU core loop:

```bash
nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
```

For the CPU core loop:

```bash
g++ -O3 -fopenmp -fPIC -shared -x c++ -o libapparie_cpu.so apparie_gpu.cu
```

## 4. Quick test

```bash
python3 apparie_graines.py --G "3^2*5*13*19" --u 1e3 --p0 1e5
```

This is a quick smoke test and should finish in under a second.

## 5. Usage

### Where the common parts G come from

| Option | Meaning |
|---|---|
| `--G "3^2*5*13*19"` | a single common part `G` |
| `--liste G_liste.txt` | one `G` per line (first word used) |
| `--depuis-catalogue sociable.txt` | the `G` values from the pure Borho 4-cycles in a catalogue |
| `--hors-catalogue 1e5` | all `G <= 1e5` with `1.8 <= sigma(G)/G < 2` absent from the catalogue |
| `--depuis-paires c2_80*.txt.gz` | exact common parts extracted from amicable-pair files |

### Main options

| Option | Default | Meaning |
|---|---:|---|
| `--u` | `1e5` | limit `u0, u2 <= u` |
| `--p0` | `1e5` | lower bound for the prime `p` |
| `--K` | `1e7` | keep only `D <= K * G^2` |
| `--budget` | `60` | maximum time per `G`; cut `G` go to `<out>.incomplets` |
| `--jobs` | `1` | number of workers |
| `--out` | `apparie_cycles.txt` | output file |
| `--connus` | — | catalogues of known cycles |
| `--refaire` | — | re-run `G` already marked as done |

### Filter options

The search can be filtered before `--debut` / `--nombre` is applied:

| Option | Effect |
|---|---|
| `--impairs` | excludes even `G` |
| `--hmin 1.85` | excludes `G` with `sigma(G) / G < 1.85` |
| `--exclure fichiers...` | excludes `G` already present in these files |

`--exclure` accepts two formats, which can be mixed freely:

- a list of `G` values, one per line; the first word of each line is used, so files such as `G_paires_22.txt` and `G nombre`-style lists are accepted;
- `.faits` files: any `G` recorded there is excluded.

Important notes:

- `G` values are compared by their numeric value, not by their textual factorization.
  For example, `3^2*5*19` and `19*3^2*5` are treated as the same `G`.
- duplicate entries inside the same input list are removed by numeric value, not by string comparison.
- a `G` that was excluded in one run is also excluded in a later run with a different `--u` value,
  because the exclusion is by value, not by the exact run parameters.
- therefore, old `.faits` files should not be reused blindly when switching from one run configuration to another.

Example:

```bash
python3 apparie_graines.py --depuis-paires c2_22.txt.gz --impairs --hmin 1.85 --exclure G_*.txt ag_*.txt.faits
```

### Resuming a run

Each completed `G` is written to `<out>.faits` together with the run parameters (for example: `u`, `p0`, `K`, `maxp`).
Running the same command again skips those `G` values for the same parameters; use `--refaire` to redo them.
A single `--G` is always re-run even when already marked complete.

Ctrl-C stops the run cleanly and preserves progress. A `G` that raises an error does not abort the run; it is written to `<out>.erreurs`.

### GPU / CPU loop

The couple loop can run on GPU with `--gpu` and `libapparie_gpu.so`; without `--gpu`, the program uses
`libapparie_cpu.so` under OpenMP by default, compiled automatically if needed. This follows the same
calculation as the C module, but is often much faster. `--sans-noyau` disables the optimized loop and
uses the standard C path.

---

## 6. Output

- cycles are appended to `--out`
- completed `G` values are saved in `<out>.faits`
- interrupted or budget-cut `G` values go to `<out>.incomplets`
- `G` values that raised errors go to `<out>.erreurs`
- final statistics summarize the candidate counts and timing

## 7. Limits of the GPU loop

The GPU kernel works with 64/128-bit arithmetic and handles the usual domain of the search. For larger
values, the CPU path may be used instead, and very large `G` values are progressively bounded by the
`Emax` limit used in the fast arithmetic.

## 8. Practical advice (from experience)

These recommendations reflect observed runs and the current implementation; they are practical
heuristics, not mathematical guarantees.

### Choosing G

- Prefer odd `G`. Even values have not produced cycles in the runs so far.
- A useful range is `1.85 <= sigma(G)/G < 2`. Below `1.85`, the yield has been roughly halved.
- Smaller `G` values are generally both faster and more productive. A practical range is about `10^5`
  to `10^8`, with smaller values preferred.
- Start with exact common parts of known cycles and their one-change variants. Variants involving two
  changes have yielded much less.
- A ratio near `2` can make the number of candidate pairs explode. Values around `1.97` have been slow
  or cut short by the time budget.

### Size and arithmetic limits

- Above about `1.7 · 10^12`, the GPU has to bound `E` (the `emax_borne` statistic). The search is then
  incomplete, and that `G` is not marked as completed.
- Above `2^44 ≈ 1.76 · 10^13`, or when `sigma(G) > 2^46`, the GPU kernel rejects `G` and the CPU C
  path is used instead; this can be about 100 times slower.
- With `u = 10^7`, `G` above `10^12` is generally not worthwhile: runs tend to hit the 600-second
  budget without finding a cycle. For example, 36 budget-cut values ranged from `1.8 · 10^13` to
  `2 · 10^14`.

### Choosing `u`

| `u` | Practical range for `G` |
|---:|---|
| `10^6` | Values up to `10^12` are generally fast. |
| `10^7` | Prefer `G < 10^12`; larger values are usually not worthwhile. |
| `3 · 10^7` | Prefer `G < 10^6`; values up to `10^7` may be worth trying. |

The number of candidate pairs grows roughly as `u^2` and also increases with `G`, so the time per `G`
can rise quickly as either grows.

### Suggested default filters

For candidate lists, a reasonable starting point is odd `G` with
`1.85 <= sigma(G)/G < 1.96`, `G < 10^12` when `u = 10^7`, and `G < 10^7` when `u = 3 · 10^7`.
The upper ratio cutoff of `1.96` is tentative, not an established limit: two `G` values with ratio
`1.971` were slow, but it is not yet known whether they would have produced cycles if allowed to run
longer. Keeping values up to `2` is also reasonable if the available time budget permits it.

A larger `--u` often pays off much more than a larger `--maxp`, but its cost grows quickly; choose it
with the size of `G` in mind.

## 9. Checking a cycle independently

```python
from sympy import divisor_sigma
c = [n1, n2, n3, n4]
print(all(divisor_sigma(c[i]) - c[i] == c[(i + 1) % 4] for i in range(4)))
```

## References

- W. Borho, *Über die Fixpunkte der k-fach iterierten Teilersummenfunktion*, Mitt. Math. Ges. Hamburg 9 (1969), 34–48.
- S. Blankenagel, W. Borho, A. vom Stein, *New amicable four-cycles*, Math. Comp. 72 (2003), 2071–2076.
- D. Moews, catalogue of sociable numbers, `djm.cc/sociable.txt`.
