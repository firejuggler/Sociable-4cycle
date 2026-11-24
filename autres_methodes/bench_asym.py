"""bench_asym.py -- mesure et verification du noyau de la recherche asymetrique (asym_gpu.cu).

1. Verification : la meme tranche est calculee par chaque bibliotheque (CPU OpenMP et GPU) ; les ensembles de
   candidats (i1, i2, i3, i4, C) doivent etre IDENTIQUES, ainsi que les compteurs (quadruplets, filtres).
2. Mesure : pour chaque bibliotheque et chaque --filtre, temps d'un appel (apres un appel de chauffe), debit en
   quadruplets par seconde. asym_run synchronise la carte avant de rendre la main : le temps mur est le bon.
   --filtre 0 donne le temps de l'appariement seul ; la difference avec les autres valeurs est le cout du
   pre-filtre.

Exemples :
  python3 bench_asym.py 4 3000001 3020000 --libs libasym_cpu.so libasym_gpu.so --filtre 0 128 2000
  python3 bench_asym.py 4 3000001 3020000 --libs libasym_gpu.so libasym_gpu64.so --filtre 2000
     (libasym_gpu64.so : meme source compilee avec -maxrregcount=64)
"""
import argparse, ctypes, math, os, sys, time
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import search_asym_gpu as S


def charger(chemin):
    if chemin == "libasym_cpu.so":
        lib, chemin = S.charger_lib(None)          # compile la version CPU si besoin
        return lib, chemin
    if os.path.sep not in chemin:
        chemin = os.path.join(ICI, chemin)
    lib = ctypes.CDLL(chemin)
    P = ctypes.c_void_p
    lib.asym_init.argtypes = [ctypes.c_int] * 3 + [P] * 13 + [ctypes.c_longlong] * 3
    lib.asym_init.restype = ctypes.c_int
    lib.asym_run.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_double, ctypes.c_longlong,
                             ctypes.c_double, ctypes.c_double, P, ctypes.c_longlong, P]
    lib.asym_run.restype = ctypes.c_longlong
    return lib, chemin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("G", type=int)
    ap.add_argument("ulo", type=float)
    ap.add_argument("uhi", type=float)
    ap.add_argument("--us", type=int, default=600)
    ap.add_argument("--pos", type=int, default=2)
    ap.add_argument("--cmax", type=float, default=2e5)
    ap.add_argument("--nmin", type=float, default=1e14)
    ap.add_argument("--cap", type=float, default=2e7)
    ap.add_argument("--libs", nargs="+", default=["libasym_cpu.so"])
    ap.add_argument("--filtre", type=float, nargs="+", default=[0, 128, 2000])
    ap.add_argument("--repete", type=int, default=2, help="appels mesures (apres 1 appel de chauffe)")
    a = ap.parse_args()
    G, US, ULO, UHI, CAP = a.G, a.us, int(a.ulo), int(a.uhi), int(a.cap)
    sG, NS, NB, UU, SS, X, Y, Z, Wd = S.tables(G, US, ULO, UHI)
    fv = X / Y; lf = np.log(fv); ly = np.log(Y.astype(np.float64)); f1p = fv + 1.0; GU = (G * UU).astype(np.float64)
    I, J = np.meshgrid(np.arange(NS), np.arange(NS), indexing="ij")
    I = I.ravel().astype(np.int32); J = J.ravel().astype(np.int32)
    SB = lf[I] + lf[J]; o = np.argsort(SB, kind="stable")
    Rs = np.ascontiguousarray(SB[o]); RI = np.ascontiguousarray(I[o]); RJ = np.ascontiguousarray(J[o])
    arrs = [np.ascontiguousarray(v) for v in (X, Y, Z, lf, fv, ly, f1p, GU)]
    UUc = np.ascontiguousarray(UU, np.int64); SSc = np.ascontiguousarray(SS, np.int64)
    u1max = UHI if a.pos == 1 else US
    EPS = min(0.06, float(np.max(Z / Wd)) * (G * u1max / (0.1 * a.nmin)) ** (1 / 3) * 1.01)
    LYmax = math.log(2.0 ** 63) + math.log(1.0 / EPS) - 1.0
    out = np.zeros((CAP, 5), np.int64); st = np.zeros(3, np.int64)
    print(f"G={G} pos={a.pos} petits={NS} grands={NB} (u {ULO}..{UHI}) cmax={a.cmax:g} eps={EPS:.4g}")
    ref = {}
    for nom in a.libs:
        lib, chemin = charger(nom)
        for fm in a.filtre:
            if lib.asym_init(NS, NB, len(Rs), *[S.ptr(v) for v in arrs], S.ptr(Rs), S.ptr(RI), S.ptr(RJ),
                             S.ptr(UUc), S.ptr(SSc), G, sG, int(fm)) != 0:
                sys.exit(f"{nom} : asym_init a echoue")
            args = (a.pos, 0, NB * NS, math.log(1 - EPS), int(a.cmax), LYmax, 0.1 * a.nmin, S.ptr(out), CAP, S.ptr(st))
            lib.asym_run(*args)                                    # chauffe (contexte CUDA, caches)
            ts = []
            for _ in range(a.repete):
                t = time.perf_counter(); k = lib.asym_run(*args); ts.append(time.perf_counter() - t)
            if k < 0 or k > CAP:
                sys.exit(f"{nom} filtre {fm:g} : k = {k} (erreur ou cap trop petit)")
            rows = out[:k].copy()
            rows = rows[np.lexsort(rows.T[::-1])]
            tmin = min(ts)
            print(f"  {os.path.basename(chemin):22s} filtre {fm:>6g} : {tmin:8.3f} s  "
                  f"{st[0] / tmin:10.3g} quads/s  quads {st[0]}  filtres {st[2]}  cand {k}")
            cle = int(fm)
            if cle in ref:
                r0, s0, n0 = ref[cle]
                same = rows.shape == r0.shape and np.array_equal(rows, r0) and tuple(st) == s0
                print(f"      identique a {n0} : {'OUI' if same else 'NON  <-- a examiner'}")
            else:
                ref[cle] = (rows, tuple(int(x) for x in st), os.path.basename(chemin))
        if hasattr(lib, "asym_free"):
            lib.asym_free()


if __name__ == "__main__":
    main()
