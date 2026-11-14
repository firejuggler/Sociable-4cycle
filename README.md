[README_apparie.txt](https://github.com/user-attachments/files/34837372/README_apparie.txt)
# Sociable-4cycle
A script to find sociable4 cycle (aliquot)
# apparie_graines — searching for aliquot 4-cycles of Borho type

`apparie_graines.py` looks for **sociable 4-cycles** (aliquot cycles of length 4) with the
construction of Borho (1969) and Blankenagel, Borho and vom Stein (Math. Comp. 72, 2003),
*without* starting from amicable pairs: the two "seeds" of the theorem are built directly.

For a common part **G** (an odd integer with 1 < σ(G)/G < 2), the seeds are

```
a1 = G · u0 · p        a2 = G · u2
```

where `u0, u2 ≤ --u` are coprime to G and `p ≥ --p0` is a large unknown prime. With
`S = σ(G)` and `D = a1·a2 − τ(a1)·τ(a2)` (τ(n) = σ(n) − n), one has `D = S·E` with
`E = α·p + β`. A small positive D requires u0/σ(u0) + u2/σ(u2) to be just above σ(G)/G, so the
candidate couples (u0, u2) are found by **sorting** all u ≤ U by u/σ(u) and matching them,
then for each candidate p the program checks the necessary condition (Theorem 2), the primality
of p, and finally Theorem 1 in full, which yields the 4-cycles. Every cycle is verified
(σ recomputed) before it is written.

The inner loop over (u0, u2, p) runs on an NVIDIA GPU (CUDA), or on the CPU with OpenMP; the
exact checks are done in C with GMP, and Python only writes the cycles.

---

## 1. Files

| File | Role |
|---|---|
| `apparie_graines.py` | main program |
| `borho4.py` | Borho's theorems (Theorem 1 in Python, cycle verification), loader of the C module |
| `borho4c.c` | C module (GMP): Theorem 1 sieve for each surviving candidate → `libborho4c.so` |
| `c2io.py` | reader for the `c2_*` amicable-pair files (only needed by `--depuis-paires`, but imported) |
| `apparie_gpu.cu` | inner loop on GPU (CUDA) or CPU (OpenMP) → `libapparie_gpu.so` / `libapparie_cpu.so` |

Put all five files in the same directory.

## 2. Requirements

- Python ≥ 3.8 with `numpy`, `sympy`, `gmpy2`:
  ```bash
  pip install numpy sympy gmpy2
  ```
- `gcc` and the GMP headers for the C module (`sudo apt install gcc libgmp-dev` on Debian/Ubuntu).
  If `libgmp-dev` is missing, `borho4.py` falls back to the GMP library shipped with `gmpy2`.
- For the GPU: an NVIDIA card and the CUDA toolkit (`nvcc`). RTX 50 cards (compute capability
  12.0) need CUDA ≥ 12.8. Under Windows, WSL2 works.
- For the CPU version of the inner loop: `g++` with OpenMP.

## 3. Compilation

**C module** (compiled automatically at the first run if `libborho4c.so` is missing or older
than `borho4c.c`; you can also do it by hand):

```bash
gcc -O2 -shared -fPIC -o libborho4c.so borho4c.c -lgmp -lm
```

**GPU inner loop** — replace `sm_120` by the compute capability of your card:

```bash
nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
```

| Card | `-arch` |
|---|---|
| RTX 20xx | `sm_75` |
| RTX 30xx | `sm_86` |
| RTX 40xx | `sm_89` |
| RTX 50xx | `sm_120` |

**CPU inner loop** (no GPU, or to cross-check the GPU version):

```bash
g++ -O3 -fopenmp -fPIC -shared -x c++ -o libapparie_cpu.so apparie_gpu.cu
```

## 4. Quick test

```bash
python3 apparie_graines.py --G "3^2*5*13*19" --u 1e3 --p0 1e5
```

It takes about one second and finds a 4-cycle of 31 digits (number 471 of the catalogue
`sociable.txt` by David Moews).

## 5. Usage

### Where the common parts G come from

| Option | Meaning |
|---|---|
| `--G "3^4*11^2*13"` | a single G |
| `--liste file.txt` | one G per line (first word of the line); duplicate values are removed |
| `--depuis-catalogue sociable.txt` | the G of the "pure" Borho 4-cycles of a catalogue (writes `G_catalogue.txt`) |
| `--hors-catalogue 1e5` | all G ≤ 1e5 with 1.8 ≤ σ(G)/G < 2 that are not in `G_catalogue.txt` |
| `--depuis-paires c2_80*.txt.gz` | the exact common parts of the amicable pairs of those files |
| `--impairs` | exclude even G |
| `--hmin 1.85` | exclude G for which σ(G)/G is below 1.85 (replace 1.85 with the desired threshold) |
| `--exclure file1 file2 ...` | exclude G already present in the specified files; see details below |

`--exclure` accepts either or both of these file formats, which can be mixed:

- **G lists:** the first word on each line is read as G. This includes files such as
  `G_paires_22.txt`, whose lines have the form `G number`.
- **`.faits` files:** every G recorded in the file is excluded, regardless of the run parameters
  recorded there. For example, a G marked as done with `--u 1e6` is also excluded from a run with
  `--u 1e7`. Do not include `.faits` files from earlier runs when searching again with a different
  `--u` value.

G values are compared numerically, not by their written factorization, so `3^2*5*19` and
`19*3^2*5` are recognized as the same G. Duplicate removal in the input list also compares G
values rather than their text representations.

`--debut N --nombre M` processes only lines N to N+M−1 of the list.

### Main options

| Option | Default | Meaning |
|---|---|---|
| `--u` | 1e5 | u0, u2 ≤ u |
| `--p0` | 1e5 | smallest large prime p tried |
| `--maxp` | 2000 | number of values of p tried per couple above the smallest possible one |
| `--K` | 1e7 | D ≤ K·G² |
| `--budget` | 60 | maximum seconds per G (G cut by the budget go to `<out>.incomplets`) |
| `--jobs` | 1 | worker processes |
| `--gpu [lib]` | off | inner loop on GPU (`libapparie_gpu.so`, else `libapparie_cpu.so`, or the library given) |
| `--connus f1 f2 ...` | — | catalogues of known cycles: new cycles are tagged `[NOUVEAU ?]`, known ones `[deja connu]` |
| `--out` | `apparie_cycles.txt` | output file |
| `--refaire` | — | redo the G already marked as done |

### Recommended settings

These are the settings that produced several hundred new 4-cycles:

```bash
python3 apparie_graines.py --liste my_G.txt --u 1e6 --p0 1000 --maxp 200 --budget 300 \
    --jobs 4 --gpu --connus sociable.txt --out my_run.txt
```

For a second pass on G that are already known to carry cycles, a larger `--u` opens new
cofactors (between 10⁶ and 10⁷) and was by far the most productive step:

```bash
python3 apparie_graines.py --liste productive_G.txt --u 1e7 --p0 1000 --maxp 200 --budget 600 \
    --jobs 4 --gpu --connus sociable.txt --out my_run_u1e7.txt
```

## 6. Output

- **Cycles** are printed and appended to `--out`, one block per cycle, members in cyclic order,
  each line `number  factorization` with the factorization written `3E4.11E2.13` (= 3⁴·11²·13).
  A cycle already in `--out` is never written twice.
- **`<out>.faits`**: the G completely explored, with the parameters of the run
  (`u`, `p0`, `K`, `maxp`). Running the same command again skips them, so a run can be stopped
  (Ctrl-C) and resumed at any time. Changing `--u` or `--p0` gives a different key: the same G
  are then searched again with the new parameters.
- **`<out>.incomplets`**: G cut by `--budget` (rerun them with a larger budget).
- **`<out>.erreurs`**: G that raised an error (the run goes on).
- **Statistics** at the end (`statistiques : {...}`): `couples` candidate (u0, u2), `p_testes`
  values of p tried, `th2` couples passing Theorem 2, `premiers` with p prime,
  `t_crible` / `t_survivants` time in the GPU loop and in the C checks.

The program's messages are in French: *termine* = finished, *cycle(s) nouveaux* = new cycles
in the output file, *absent(s) des listes* = not found in the `--connus` catalogues,
*G coupe(s) par le budget* = G cut by the time budget.

## 7. Limits of the GPU loop

The GPU kernel works with 64/128-bit integers. A G is processed on the GPU when
G < 2⁴⁴, σ(G) < 2⁴⁶, u < 2²⁴ and σ(u) < 2²⁶; otherwise it is processed by the C module on
the CPU (much slower). For G above about 1.7·10¹², the bound on E is capped at 2⁶³ − 1
(statistic `emax_borne`); such G are not marked as done in `.faits`.

## 8. Practical advice (from experience)

- Choose **odd** G with σ(G)/G between **1.85 and 2**: below 1.85 the yield drops by half;
  even G gave nothing in our tests.
- **Small** G (10⁶ – 10⁹) are both the fastest and the most productive.
- Good lists come from the **exact common parts of known cycles** and their close
  **neighbours**: remove one prime, add one prime between 17 and 200, or change the exponent of
  3, 5, 7, 11 or 13 by ±1. One modification works much better than two.
- `--maxp` above 200 and `--p0` other than 1000 brought almost nothing; `--u 1e7` on G that
  already carry cycles brought a lot (about one new cycle every 20 to 50 G).
- Each new cycle can give more through **isotopes**: replacing a part of its common part by
  another part with the same σ(X)/X (for example 19²·127 ↔ 19⁴·151·911 or
  3¹⁰·107·3851 ↔ 3⁶·137·547·1093). This is not part of `apparie_graines.py`.

## 9. Checking a cycle independently

```python
from sympy import divisor_sigma
c = [n1, n2, n3, n4]          # the four members
print(all(divisor_sigma(c[i]) - c[i] == c[(i + 1) % 4] for i in range(4)))
```

## References

- W. Borho, *Über die Fixpunkte der k-fach iterierten Teilersummenfunktion*,
  Mitt. Math. Ges. Hamburg 9 (1969), 34–48.
- S. Blankenagel, W. Borho, A. vom Stein, *New amicable four-cycles*,
  Math. Comp. 72 (2003), 2071–2076.
- D. Moews, catalogue of sociable numbers, `djm.cc/sociable.txt`.

