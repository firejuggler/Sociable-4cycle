"""search_asym_gpu.py -- recherche asymetrique a deux grands premiers, appariement sur GPU (ou CPU OpenMP),
decoupage automatique en tranches, reprise.

Forme cherchee : n1 = G*u1*p*q, n2 = G*u2*P2, n3 = G*u3*P3, n4 = G*u4*P4 (voir LISEZMOI.md), avec trois
cofacteurs petits (<= --us) et un grand (dans [Ulo, Uhi]) a la position --pos.

Difference avec search_asym.py : le seuil --cmax est FIXE (pas de tampon qui se reduit). Tout quadruplet de C
reduit <= cmax est traite, donc la couverture est complete jusqu'a cmax sur toute la plage. Le programme coupe
la plage du grand cofacteur en tranches ; si une tranche donne plus de --cap candidats, il la coupe en deux et
recommence ; si elle en donne peu, la tranche suivante est plus grande.

Exemples :
  python3 search_asym_gpu.py 1 3354000 3355000 --us 600 --pos 2 --out t123.log          (test : cycle n. 123)
  python3 search_asym_gpu.py 4 2000000 10000000 --pos 2 3 --gpu --jobs 8 --out g4.log
Bibliotheques :
  GPU : nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libasym_gpu.so asym_gpu.cu   (option --gpu)
  CPU : libasym_cpu.so, compilee automatiquement depuis asym_gpu.cu (g++ -fopenmp) si elle manque.
Reprise : chaque tranche terminee est notee dans <out>.faits ; relancer la meme commande reprend la ou on s'etait
arrete (mêmes G, us, cmax, nmin, pos). Ctrl-C est sans danger : la tranche en cours sera refaite.
"""
import argparse, ctypes, json, math, os, subprocess, sys, time
from math import gcd
from multiprocessing import Pool
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)


def sig_range(a, b):
    """sigma(u) pour u dans [a, b]."""
    s = np.zeros(b - a + 1, np.int64)
    for d in range(1, b + 1):
        lo = ((a + d - 1) // d) * d
        if lo <= b:
            s[lo - a::d] += d
    return s


def sigma(n):
    return int(sig_range(n, n)[0])


def tables(G, us, ulo, uhi):
    """Donnees par cofacteur (petits d'abord, puis grands), comme search_asym.py, en numpy vectorise."""
    sG = sigma(G)
    parts = []
    for a, b in ((1, us), (ulo, uhi)):
        u = np.arange(a, b + 1, dtype=np.int64); s = sig_range(a, b)
        ok = np.gcd(u, G) == 1
        u, s = u[ok], s[ok]
        x = sG * s - G * u; y = G * u
        ok = x > 0
        u, s, x, y = u[ok], s[ok], x[ok], y[ok]
        g = np.gcd(x, y); z = sG * s; w = G * u; g2 = np.gcd(z, w)
        parts.append((u, s, x // g, y // g, z // g2, w // g2))
    cat = [np.concatenate([parts[0][k], parts[1][k]]) for k in range(6)]
    UU, SS, X, Y, Z, Wd = cat
    return sG, len(parts[0][0]), len(parts[1][0]), UU, SS, X, Y, Z, Wd


def charger_lib(gpu):
    if gpu and gpu != "cpu":
        chemin = gpu if os.path.sep in gpu or gpu.endswith(".so") else os.path.join(ICI, "libasym_gpu.so")
    else:
        chemin = os.path.join(ICI, "libasym_cpu.so")
        src = os.path.join(ICI, "asym_gpu.cu")
        if not os.path.exists(chemin) or os.path.getmtime(chemin) < os.path.getmtime(src):
            print("compilation de libasym_cpu.so ...", flush=True)
            subprocess.check_call(["g++", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", "-x", "c++",
                                   "-o", chemin, src])
    lib = ctypes.CDLL(chemin)
    P = ctypes.c_void_p
    lib.asym_init.argtypes = [ctypes.c_int] * 3 + [P] * 13 + [ctypes.c_longlong] * 3
    lib.asym_init.restype = ctypes.c_int
    lib.asym_run.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_double, ctypes.c_longlong,
                             ctypes.c_double, ctypes.c_double, P, ctypes.c_longlong, P]
    lib.asym_run.restype = ctypes.c_longlong
    return lib, chemin


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("G", type=int)
    ap.add_argument("ulo", type=float, help="borne basse du grand cofacteur")
    ap.add_argument("uhi", type=float, help="borne haute du grand cofacteur")
    ap.add_argument("--us", type=int, default=600, help="borne des trois petits cofacteurs (600)")
    ap.add_argument("--pos", type=int, nargs="+", default=[2], help="position(s) 1..4 du grand cofacteur (2)")
    ap.add_argument("--cmax", type=float, default=2e5, help="seuil fixe du C reduit (2e5)")
    ap.add_argument("--nmin", type=float, default=1e14, help="taille minimale visee du cycle (1e14)")
    ap.add_argument("--eps", type=float, default=None, help="largeur de la fenetre 1 - prod f (auto)")
    ap.add_argument("--cap", type=float, default=4e6, help="candidats max par tranche (4e6)")
    ap.add_argument("--tranche", type=int, default=20000, help="taille initiale d'une tranche, en grands u (20000)")
    ap.add_argument("--gpu", nargs="?", const="auto", default=None,
                    help="noyau GPU (libasym_gpu.so ou chemin) ; sans option : CPU OpenMP")
    ap.add_argument("--filtre", type=float, default=128,
                    help="pre-filtre local dans le noyau : modules l^K <= FILTRE pour l = 3, 5, 7, 2 (128 ; 0 = aucun)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="processus pour crible et factorisation")
    ap.add_argument("--pari-sec", type=int, default=2, help="temps max PARI par N ; au-dela : 'hard' (2)")
    ap.add_argument("--out", default="asym.log")
    ap.add_argument("--refaire", action="store_true", help="ignorer <out>.faits")
    a = ap.parse_args()
    G, US, ULO, UHI = a.G, a.us, int(a.ulo), int(a.uhi)
    CMAX, CAP = int(a.cmax), int(a.cap)
    assert US < ULO <= UHI and all(1 <= p <= 4 for p in a.pos)

    t0 = time.time()
    sG, NS, NB, UU, SS, X, Y, Z, Wd = tables(G, US, ULO, UHI)
    n = len(UU)
    fv = X / Y; lf = np.log(fv); ly = np.log(Y.astype(np.float64)); f1p = fv + 1.0
    GU = (G * UU).astype(np.float64)
    I, J = np.meshgrid(np.arange(NS), np.arange(NS), indexing="ij")
    I = I.ravel().astype(np.int32); J = J.ravel().astype(np.int32)
    SB = lf[I] + lf[J]; o = np.argsort(SB, kind="stable")
    Rs = np.ascontiguousarray(SB[o]); RI = np.ascontiguousarray(I[o]); RJ = np.ascontiguousarray(J[o])
    arrs = [np.ascontiguousarray(v) for v in (X, Y, Z, lf, fv, ly, f1p, GU)]
    lib, chemin = charger_lib(a.gpu)
    if a.filtre and (sG * int(SS.max()) >= 2**62 or G * int(UU.max()) >= 2**62):
        print("pre-filtre desactive : G*u ou sigma(G)*sigma(u) depasse 64 bits", flush=True)
        a.filtre = 0
    UUc = np.ascontiguousarray(UU, np.int64); SSc = np.ascontiguousarray(SS, np.int64)
    if lib.asym_init(NS, NB, len(Rs), *[ptr(v) for v in arrs], ptr(Rs), ptr(RI), ptr(RJ), ptr(UUc), ptr(SSc),
                     G, sG, int(a.filtre)) != 0:
        sys.exit("asym_init a echoue (memoire GPU ?)")
    log = open(a.out, "a")

    def P(d):
        print(json.dumps(d), file=log, flush=True)

    faits_path = a.out + ".faits"
    faits = {}
    if os.path.exists(faits_path) and not a.refaire:
        for ligne in open(faits_path):
            try:
                d = json.loads(ligne)
            except ValueError:
                continue
            faits.setdefault(d["cle"], []).append((d["u_lo"], d["u_hi"]))
    fdone = open(faits_path, "a")
    P(dict(debut=True, G=G, us=US, ulo=ULO, uhi=UHI, pos=a.pos, cmax=CMAX, nmin=a.nmin, ns=NS, nb=NB,
           lib=os.path.basename(chemin), tables_s=round(time.time() - t0, 1)))
    print(f"G={G}  petits={NS}  grands={NB}  moitie B={len(Rs)}  bibliotheque={os.path.basename(chemin)}", flush=True)

    out = np.zeros((CAP, 5), np.int64); stats = np.zeros(3, np.int64)
    pool = Pool(a.jobs, initializer=_init_post, initargs=(G, sG, UU, SS, a.pari_sec))
    total = dict(quads=0, filtered=0, cand=0, zero=0, factored=0, found=0, hard=0)
    attente = None

    def finir(t):
        """Attend le traitement CPU d'une tranche, ecrit ses resultats et la marque comme faite."""
        if t is None:
            return
        res = t["res"].get() if t["res"] is not None else []
        nz = sum(r[0] for r in res); nf = sum(r[1] for r in res)
        cyc = [c for r in res for c in r[2]]; durs = [d for r in res for d in r[3]]
        vus = set()
        for c in cyc:
            if min(c["cycle"]) in vus:
                continue
            vus.add(min(c["cycle"])); P(c)
            print("CYCLE", c, flush=True)
        for d in durs:
            P(dict(hard=True, **d))
        tp = time.time() - t["t0"]
        P(dict(tranche=True, pos=t["pos"], u_lo=t["u_lo"], u_hi=t["u_hi"], quads=t["quads"], toobig=t["toobig"],
               filtered=t["filtered"], cand=t["cand"], zero=int(nz), factored=int(nf), found=len(vus), hard=len(durs),
               t_app=t["t_app"], t_post=round(tp, 1)))
        print(f"pos {t['pos']}  u {t['u_lo']}..{t['u_hi']}  quads {t['quads']:.3g}  filtres {t['filtered']}"
              f"  cand {t['cand']}  facto {nf}  cycles {len(vus)}  [GPU {t['t_app']:.1f}s | CPU {tp:.1f}s]", flush=True)
        fdone.write(json.dumps(dict(cle=t["cle"], u_lo=t["u_lo"], u_hi=t["u_hi"])) + "\n"); fdone.flush()
        for kk, v in (("quads", t["quads"]), ("filtered", t["filtered"]), ("cand", t["cand"]), ("zero", nz),
                      ("factored", nf), ("found", len(vus)), ("hard", len(durs))):
            total[kk] += v

    try:
        for pos in a.pos:
            u1max = UHI if pos == 1 else US
            EPS = a.eps or min(0.06, float(np.max(Z / Wd)) * (G * u1max / (0.1 * a.nmin)) ** (1 / 3) * 1.01)
            LYmax = math.log(2.0 ** 63) + math.log(1.0 / EPS) - 1.0
            cle = f"G={G} us={US} cmax={CMAX} nmin={a.nmin:g} eps={EPS:.6g} pos={pos}"
            couverts = faits.get(cle, [])
            b = 0; taille = max(1, a.tranche)
            while b < NB:
                ulo_t = int(UU[NS + b])
                # tranche deja faite ? (intervalles de u)
                saut = next((hi for lo, hi in couverts if lo <= ulo_t <= hi), None)
                if saut is not None:
                    b = int(np.searchsorted(UU[NS:], saut, side="right")); continue
                e = min(NB, b + taille)
                ta = time.time()
                k = lib.asym_run(pos, b * NS, e * NS, math.log(1 - EPS), CMAX, LYmax, 0.1 * a.nmin, ptr(out), CAP,
                                 ptr(stats))
                if k < 0:
                    sys.exit("asym_run a echoue")
                if k > CAP:                      # trop de candidats : on coupe la tranche en deux
                    if e - b == 1:
                        sys.exit(f"un seul grand u donne {k} candidats > cap : augmenter --cap ou baisser --cmax")
                    taille = max(1, (e - b) // 2)
                    print(f"  tranche trop pleine ({k} > cap) : taille -> {taille}", flush=True)
                    continue
                tm = time.time() - ta
                rows = out[:k].copy()
                # le CPU traite cette tranche pendant que la carte calcule la suivante
                finir(attente)
                lots = [rows[i::a.jobs] for i in range(a.jobs)] if k else []
                attente = dict(res=pool.map_async(_traiter, lots) if k else None, pos=pos, cle=cle,
                               u_lo=ulo_t, u_hi=int(UU[NS + e - 1]), quads=int(stats[0]), toobig=int(stats[1]),
                               filtered=int(stats[2]), cand=int(k), t_app=round(tm, 2), t0=time.time())
                b = e
                if k < CAP // 8:                 # peu de candidats : tranche suivante plus grande
                    taille = min(NB, taille * 2)
        finir(attente); attente = None
    except KeyboardInterrupt:
        print("interrompu : les tranches terminees sont dans", faits_path, flush=True)
    finally:
        pool.terminate()
    P(dict(done=True, G=G, **total, ttot=round(time.time() - t0, 1)))
    print("fin :", total, f"{time.time() - t0:.0f}s", flush=True)


def _init_post(*args):
    import asym_post
    asym_post.init(*args)


def _traiter(lot):
    import asym_post
    return asym_post.traiter(lot)


if __name__ == "__main__":
    main()
