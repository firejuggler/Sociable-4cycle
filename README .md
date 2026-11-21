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
nvcc -O3 -arch=sm_120 -maxrregcount=64 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
```

`sm_120` is the architecture of RTX 50-series cards: replace it with yours (`sm_89` for RTX 40, `sm_86` for
RTX 30...), or use `-arch=native` to compile for the card present in the machine.

`-maxrregcount=64` matters: without it the kernel uses 96 registers per thread, which limits occupancy to 33 %
(profiled with Nsight Compute: the kernel is latency-bound, not memory-bound). With 64 registers it ran
1.3 to 1.5 times faster on an RTX 50-series card, with identical results; 48 or 40 registers gave no further gain.

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
| `--sans-premiers 2,3,5` | excludes these primes from every cofactor `u` and from `G` (for example, cycles coprime to 30); recorded in the resume key |

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

### Raising `u` without redoing work: `--u-deja`

`--u-deja U0` skips every pair (u0, u2) with **both** cofactors `<= U0`: they were already searched by a run
at `--u U0` with the same `--p0`, `--K` and `--maxp`. Only pairs where at least one cofactor exceeds `U0` are done.

```bash
python3 apparie_graines.py --liste G.txt --u 1.2e8 --u-deja 6e7 --gpu --out ag_u12e7.txt
```

The pairs below `U0` are only a quarter of the pairs when `u` doubles, but they are the most expensive ones
(they have the most values of `p` to test): going from `5e5` to `1e6` this way took 47 % less time, and the
two runs together found exactly the cycles of the full run. Use it only for `G` values that were **completed**
at `U0` (present in a `.faits` file with the same parameters); a `G` cut by the budget at `U0` would lose pairs.

### Two free primes: `--etendu` (experimental)

`--etendu V` searches seeds `a1 = G*u0*p` and `a2 = G*v*q` with `q` a large prime enumerated up to
`--etendu-max` (default `2^31 - 1`), `v <= V`, `--u < v*q`. This is the shape of the Borho cycles obtained from
amicable pairs (1276 of the 5553 4-cycles of the catalogue). `--etendu-premiers FILE` restricts `q` to a list of
primes, `--etendu-lisse B` to primes with `q + 1` B-smooth. In practice this mode only re-found known cycles.

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

The kernel (`apparie_gpu.cu`, same code for the GPU and the OpenMP CPU build) works in 64/128-bit
integers. A `G` is handled by the kernel when

- `G < 2^44 ≈ 1.76 · 10^13` and `sigma(G) < 2^46`;
- with `bG, bS, bu, bs` the bit lengths of `G`, `sigma(G)`, the largest `u` and the largest `sigma(u)`:
  `bu + bs + max(bG+bu, bS+bs) <= 125`, `bG + bs + 2·bu + 1 <= 125`, `bu <= 31`, `bs <= 33`.
  In practice `--u` can go well beyond `2^24` for small `G` (`3 · 10^7` and `6 · 10^7` have been used).

**Hybrid mode.** `E <= K · G^2 / sigma(G)` no longer has to fit in 63 bits. When that bound reaches
`2^63` (about `G > 1.7 · 10^12` with `K = 1e7`), the kernel tests `E < 2^63` in 64-bit arithmetic and
`2^63 <= E < 2^127` in 128-bit arithmetic. The search is complete and the `G` is recorded in `.faits`
(statistics: `emax_hybride`, and `p_128` = primes returned by the 128-bit path). This requires
`--gpu` (GPU, or `--gpu cpu` for the OpenMP build).

Outside this domain the standard C path (GMP) is used; it is far slower.

**Display card and Windows watchdog.** The kernel is called on slices of `2^28` pairs. On a card that also
drives the display, Windows resets the GPU when one call lasts more than about 2 seconds (TDR); under WSL the
program then hangs instead of getting an error (event `nvlddmkm` 153 in the Windows event log). Slices of
`2^31` pairs were about 8 % faster with the old static scheduling (this gain came mostly from the tail of
each call, which dynamic tiles have removed) but can hit this limit on heavy `G`: use `AG_PAS=31` only on a
card dedicated to computing.

**Kernel layout.** Each GPU thread alternates "take the next pair `(u0, u2)`" and "test one `p`", so a
thread never waits for the `p` loop of the 31 others in its warp. Measured on an RTX 50-series card:
about `1.3 · 10^10` pairs per second, 1.7 times the previous kernel, with identical results.
Compiling with `-DAG_SIMU` (CPU) runs the same warp logic on 32 simulated lanes, which allows the
scheduling to be checked without a GPU.

**Dynamic tiles.** Warps take tiles of 128 consecutive pairs from a global counter (`atomicAdd` by lane 0,
broadcast with `__shfl_sync`), on a persistent grid sized to fill the card. With fixed tiles of 2048 pairs per
warp, the expensive pairs (many `p` passing the divisibility test) are grouped in a few regions, and the whole
card waited for a few warps: Nsight Compute showed the SMs active only 24 % of the kernel duration.

**Montgomery primality test.** `premier64` (deterministic Miller-Rabin, 12 bases) now uses Montgomery
multiplication (`redc`) instead of `u128 % u64`, which is a software division on GPUs. For pairs where dozens of
`p` pass Theorem 2, the old test kept one warp busy for tens of millions of cycles. The division-based version is
kept as `premier64_div` for `n >= 2^63`.

Measured on an RTX 5070 (one slice of `2^28` pairs, `G = 3^2*5^2*13^2*41`, `u = 6e7`): 77 ms with the previous
kernel, 42 ms with dynamic tiles, and about 20 ms with the Montgomery test as well (per-warp timing). The results
are identical.

**Diagnostic build.** Compiling with `-DAG_CHRONO` prints, for the first call of the kernel, the duration of each
warp (min / mean / max, in cycles) and the 8 slowest tiles with their `t`, `u0`, `u2` and SM:

```bash
nvcc -O3 -arch=sm_120 -maxrregcount=64 -DAG_CHRONO -Xcompiler -fPIC -shared -o libapparie_gpu_chrono.so apparie_gpu.cu
```

### Memory (RAM and GPU)

Memory grows linearly with `--u`. Peak RAM used by a run with `--jobs 1` (system and other programs excluded;
measured on Linux with the CPU kernel, which builds the same arrays as the GPU run; the `1.2e8` figure matches
what was observed on a 16 GB WSL machine with an RTX 5070):

| `--u` | without `--u-deja` | with `--u-deja` (`U0 = u/2`) | GPU memory |
|---:|---:|---:|---:|
| `6e7` | about 2.9 GB | about 3.9 GB | about 1 GB (estimate) |
| `1.2e8` | - | about 7.7 GB | about 1.7 GB |

- The **main process** keeps, once for the whole run, the table of `sigma(u)` for `u <= U`, the cofactors sorted
  by `u/sigma(u)` and one mask per small prime: roughly 30 bytes per value of `u`.
- The **computing process** builds, for each `G`, the cofactors coprime with `G`, the windows of partners and the
  arrays sent to the card. With `--u-deja` the cofactors above `U0` are added once more.
- Memory freed after a `G` is not always returned to the system. The computing process is replaced every
  `--recyclage N` `G` (default 10); **at large `u`, use `--recyclage 1`**, otherwise the process grows from one `G`
  to the next. Without it, a run at `u = 1.2e8` saturated 16 GB of RAM and started swapping after about ten `G`.
- **Each extra job (`--jobs`) adds a full computing process**: at `u = 1.2e8`, `--jobs 2` does not fit in 16 GB.
- Do not run other heavy programs in parallel (`croise.py`, `paires_connues.py`...): under WSL, Linux and Windows
  share the same RAM. Check with `free -h`: if `available` drops below 1 GB or the swap grows, stop the run
  (Ctrl-C keeps every completed `G` in `.faits`) and restart it with `--recyclage 1`.

## 8. Practical advice (from experience)

These recommendations reflect observed runs and the current implementation; they are practical
heuristics, not mathematical guarantees.

### Choosing G

- Prefer odd `G`. Even values have not produced cycles in the runs so far.
- A useful range is `1.85 <= sigma(G)/G < 2`. Below `1.85`, the yield has been roughly halved.
- Most cycles come from `G` between about `10^5` and `10^8`, but larger `G` (up to `2^44`) are now
  cheap at `u = 10^7` and also produce cycles.
- Start with exact common parts of known cycles and their one-change variants. Variants involving two
  changes have yielded much less.
- A ratio near `2` makes the number of candidate pairs grow; such `G` are slow but productive.

### Size and arithmetic limits

- Up to `2^44 ≈ 1.76 · 10^13` the GPU handles `G` completely (hybrid mode above about `1.7 · 10^12`).
  With `u = 10^7`, a `G` near `10^13` takes a few seconds, and this range has produced a cycle
  (`G = 3^6·5^2·547·701·1093`).
- Above `2^44`, or when `sigma(G) >= 2^46`, the kernel rejects `G` and the CPU C path is used instead;
  this can be about 100 times slower.

### Choosing `u`

| `u` | Practical range for `G` |
|---:|---|
| `10^6` | Fast for every `G` the kernel accepts. |
| `10^7` | Fine up to `2^44`; a few seconds per `G` near `10^13`. |
| `3 · 10^7` | Best yield so far; prefer `G < 10^8`. Small `G` are the slow ones (more pairs). |
| `6 · 10^7` | About 4 times the cost of `3 · 10^7`; set `--budget` explicitly. |

The number of candidate pairs grows roughly as `u^2`. Cycles found at a larger `u` all have a cofactor
above the previous limit, so re-running productive `G` at a larger `u` finds new cycles. The default
`--budget` is 60 seconds per `G`: raise it for large `u`, otherwise slow `G` end up in `.incomplets`.

### Suggested default filters

For candidate lists, a reasonable starting point is odd `G` with
`1.85 <= sigma(G)/G < 2`, `G < 2^44` when `u = 10^7`, and `G < 10^8` when `u = 3 · 10^7`.
Do not cut the ratio below `2`: the `G` closest to `2` have been the most productive, even though
they are the slowest.

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

## License

This project is released under the GNU General Public License, version 3 or (at your option) any later
version (GPL-3.0-or-later). See the `LICENSE` file for the full text.

You may use, modify and redistribute it; any distributed modified version must remain under the same license
and come with its source code.

