#!/usr/bin/env python3
"""tests_croises.py — tests croises de apparie_graines.py et du noyau apparie_gpu.cu.

A lancer depuis le dossier des scripts (celui de apparie_graines.py), avec sociable.txt present :
    python3 tests/tests_croises.py                 (environ 5 a 10 minutes sur CPU)
    python3 tests/tests_croises.py --gpu ./libapparie_gpu.so   (ajoute le noyau GPU aux comparaisons)
    python3 tests/tests_croises.py --rapide        (sections 1 et 2 seulement, ~1 minute)

1. Arithmetique du noyau contre les entiers Python (valeurs aleatoires ET aux bornes) :
   redc, redc128, mr128, div_petit, div_petit64, mul64x32, premier64.
2. Un couple complet (init_couple + pas_couple, chemins 64 et 128 bits) contre une reference Python
   ecrite directement a partir des formules : memes p emis, meme nombre de p testes.
3. Chemins entiers : Python pur (--sans-c), module C GMP (--sans-noyau), noyau CPU, noyau CPU simulant les
   warps du GPU, et le GPU si --gpu : memes cycles et memes nombres de p premiers ; mode hybride force
   (AG_TEST_CAP) ; partition --u-deja (les deux moities redonnent le run complet).
4. Rappel sur le catalogue : chaque 4-cycle de sociable.txt de la forme a1 = G*u0*p, a2 = G*u2 avec
   u0, u2 <= U, p >= P0 et E <= Emax doit etre retrouve. C'est le test qui detecte un faux negatif.
Code de sortie 0 si tout passe, 1 sinon.
"""
import argparse, ctypes, math, os, random, re, subprocess, sys, tempfile, time
from math import gcd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gpu", default="", help="bibliotheque GPU a tester en plus (ex. ./libapparie_gpu.so)")
ap.add_argument("--rapide", action="store_true")
ap.add_argument("--graine", type=int, default=12345)
ap.add_argument("--catalogue", default="sociable.txt")
A = ap.parse_args()
random.seed(A.graine)
ECHECS = []
LIBS = []


def verifie(cond, msg):
    if not cond:
        ECHECS.append(msg)
        if len(ECHECS) <= 30:
            print("   [ECHEC]", msg, flush=True)
    return cond


def compiler(nom, defs):
    so = os.path.join(ICI, nom)
    src = os.path.join(ICI, "arith_wrap.cpp")
    dep = os.path.join(RACINE, "apparie_gpu.cu")
    if not os.path.exists(so) or os.path.getmtime(so) < max(os.path.getmtime(src), os.path.getmtime(dep)):
        subprocess.run(["g++", "-O2", "-fopenmp", "-fPIC", "-shared", "-x", "c++", *defs, "-o", so, src], check=True)
    return ctypes.CDLL(so)


u64, P64 = ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)
hl = lambda x: (x >> 64, x & ((1 << 64) - 1))


def preparer(L):
    L.t_redc128.argtypes = [u64] * 5 + [P64, P64]
    L.t_mr128.argtypes = [u64] * 5 + [P64, P64]
    L.t_redc.argtypes = [u64] * 3; L.t_redc.restype = u64
    L.t_div_petit.argtypes = [u64] * 4 + [P64, P64, P64]; L.t_div_petit.restype = ctypes.c_int
    L.t_div_petit64.argtypes = [u64, u64, P64, P64]; L.t_div_petit64.restype = ctypes.c_int
    L.t_mul64x32.argtypes = [u64, ctypes.c_uint32, P64, P64]
    L.t_premier64.argtypes = [u64]; L.t_premier64.restype = ctypes.c_int
    i64 = ctypes.c_int64
    L.t_couple.argtypes = [i64] * 6 + [u64] * 5 + [P64, ctypes.c_int, P64]; L.t_couple.restype = ctypes.c_int
    return L


def aleatoire_bornes(bits):
    """Valeurs aleatoires, et valeurs collees aux bornes, sur au plus bits bits."""
    v = [random.getrandbits(random.randint(1, bits)) for _ in range(6)]
    v += [(1 << bits) - 1, (1 << bits) - 2, (1 << (bits - 1)), (1 << (bits - 1)) + 1, 1, 2, 3]
    return v


# ------------------------------------------------------------------ 1. arithmetique
def section1(L, nom):
    t0 = time.time()
    n = 0
    a, b = u64(), u64()
    q, rh, rl = u64(), u64(), u64()
    R64 = 1 << 64
    for _ in range(4000):
        # redc 64 : n impair < 2^63, T < n*2^64
        m = random.choice(aleatoire_bornes(63)) | 1
        T = random.randrange(m << 64)
        r = L.t_redc(T >> 64, T & (R64 - 1), m)
        verifie(r < m and r == T * pow(R64, -1, m) % m, f"{nom} redc T={T} n={m} -> {r}")
        # redc128 : n impair < 2^127 ; T < n*2^64 (resultat < n), puis T < 2^127 quelconque, l'usage du noyau
        # (|X| d'un i128) : resultat < 2^63 + n, reduit ensuite par l'appelant
        m = random.choice(aleatoire_bornes(127)) | 1
        inv = pow(R64, -1, m)
        for T in (random.randrange(m << 64), random.randrange(1 << 127), (1 << 127) - 1):
            L.t_redc128(T & (R64 - 1), (T >> 64) & (R64 - 1), T >> 128, *hl(m), ctypes.byref(a), ctypes.byref(b))
            r = (a.value << 64) | b.value
            ok = r % m == T * inv % m and (r < m if T < (m << 64) else r < (1 << 63) + m)
            verifie(ok, f"{nom} redc128 T={T} n={m} -> {r}")
        # mr128 : a < n, b 64 bits
        x = random.randrange(m)
        y = random.choice(aleatoire_bornes(64))
        L.t_mr128(*hl(x), y, *hl(m), ctypes.byref(a), ctypes.byref(b))
        r = (a.value << 64) | b.value
        verifie(r < m and r == x * y * inv % m, f"{nom} mr128 a={x} b={y} n={m} -> {r}")
        # div_petit : n < 2^127, 0 < a < 2^96 ; si accepte, quotient et reste exacts ; accepte des que q < 2^19
        d = random.choice(aleatoire_bornes(96)) or 1
        qq = random.choice([random.randrange(1 << 20), random.randrange(1 << 19), random.randrange(1 << 25), 0, (1 << 20) - 1, 1 << 20])
        N = min(d * qq + random.randrange(d), (1 << 127) - 1)
        ok = L.t_div_petit(*hl(N), *hl(d), ctypes.byref(q), ctypes.byref(rh), ctypes.byref(rl))
        if ok:
            verifie(q.value == N // d and ((rh.value << 64) | rl.value) == N % d, f"{nom} div_petit n={N} a={d}")
        else:
            verifie(N // d >= 1 << 19, f"{nom} div_petit refuse a tort n={N} a={d} (q={N // d})")
        d = random.choice(aleatoire_bornes(64)) or 1
        N = random.choice([d * random.randrange(1 << 20) + random.randrange(d), random.getrandbits(64)]) % R64
        ok = L.t_div_petit64(N, d, ctypes.byref(q), ctypes.byref(rh))
        if ok:
            verifie(q.value == N // d and rh.value == N % d, f"{nom} div_petit64 n={N} a={d}")
        else:
            verifie(N // d >= 1 << 19, f"{nom} div_petit64 refuse a tort n={N} a={d}")
        # mul64x32
        x, y = random.choice(aleatoire_bornes(64)), random.choice(aleatoire_bornes(32))
        L.t_mul64x32(x, y, ctypes.byref(a), ctypes.byref(b))
        verifie(((a.value << 64) | b.value) == x * y, f"{nom} mul64x32 {x}*{y}")
        n += 1
    # premier64 : contre gmpy2, avec des composes pieges (Carmichael, pseudo-premiers forts, carres de premiers)
    import gmpy2
    pieges = [561, 1105, 1729, 2047, 3215031751, 2152302898747, 3474749660383, 341550071728321,
              3825123056546413051, 318665857834031151167461 % R64, 18446744073709551557, 18446744073709551615,
              4294967291 ** 2, 4294967279 * 4294967291, 1000000007 * 998244353]
    for x in pieges + [random.getrandbits(random.randint(2, 64)) for _ in range(20000)]:
        verifie(bool(L.t_premier64(x)) == bool(gmpy2.is_prime(x, 50)), f"{nom} premier64({x})")
    print(f"  1. arithmetique ({nom}) : {n * 7 + 20015} verifications, {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ 2. un couple complet
def ref_couple(G, S, u0, s0, u2, s2, P0, maxp, Emax, Evrai):
    """Reference Python des p emis pour un couple (meme convention que pas_couple : bit 63 = zone 128 bits)."""
    import gmpy2
    al = G * (s0 * u2 + s2 * u0) - S * s0 * s2
    be = s0 * (G * u2 - S * s2)
    if al <= 0 or be >= 0:
        return [], 0
    pmin = max(-be // al + 1, P0)
    borne = Evrai if Evrai else Emax
    M0 = abs(u0 * be + al * u2) * G * G * gcd(u0, u2) * s2
    out, nt = [], 0
    for c in range(maxp + 1):
        p = pmin + c
        E = al * p + be
        if E > borne:
            break
        nt += 1
        if M0 % E:
            continue
        if not (gmpy2.is_prime(p) and G % p and u0 % p and u2 % p):
            continue
        if E <= Emax:
            d = u0 * p - u2
            if (G * G * d * gcd(s0 * (p + 1), s2) * gcd(u0 * p, u2)) % E:
                continue
            out.append(p)
        else:
            out.append(p | (1 << 63))
    return out, nt


def section2(L, nom, ncouples):
    import numpy as np
    from sympy import divisor_sigma, factorint
    t0 = time.time()
    U = 200000
    sg = np.zeros(U + 1, dtype=np.int64)
    for d in range(1, U + 1):
        sg[d::d] += d
    u = np.arange(1, U + 1)
    y = u / sg[1:]
    o = np.argsort(y)
    ys, us = y[o], u[o]
    Gs = ["3*5*7", "3^2*5*7", "3^3*5*13", "3^4*7*11", "5^3*7^2*13*17*19*23*37", "3^2*5*13*19", "3*5^2*7*29",
          "3^6*5^2*547*701*1093"]
    buf = (u64 * 4096)()
    nt = u64()
    ncomp = nemis = n0 = 0
    for k in range(ncouples):
        gtxt = random.choice(Gs)
        f = {}
        for t in gtxt.split("*"):
            p, _, e = t.partition("^"); f[int(p)] = int(e or 1)
        G = math.prod(p ** e for p, e in f.items()); S = math.prod((p ** (e + 1) - 1) // (p - 1) for p, e in f.items())
        h = S / G
        P0 = random.choice([2, 100, 1000])
        # u2 au hasard (premier avec G), u0 pris dans la fenetre y0 dans (t, t + t/P0]
        while True:
            u2 = random.randrange(1, U)
            if gcd(u2, G) == 1:
                break
        t = h - u2 / sg[u2]
        lo, hi = np.searchsorted(ys, t), np.searchsorted(ys, t + t / P0 * 1.000001)
        if hi <= lo:
            continue
        u0 = int(us[random.randrange(lo, hi)])
        if gcd(u0, G) != 1:
            continue
        s0, s2 = int(sg[u0]), int(sg[u2])
        Emax = int(1e7 * G * G // S)
        Evrai = 0
        if Emax >= 1 << 63 or random.random() < 0.4:       # mode hybride : borne 64 bits abaissee ou reelle
            Evrai = Emax
            Emax = min((1 << 63) - 1, max(1, Emax // random.choice([1, 10, 1000, 10 ** 6])))
        maxp = random.choice([0, 5, 200])
        n = L.t_couple(G, S, u0, s0, u2, s2, P0, maxp, Emax, *hl(Evrai), buf, 4096, ctypes.byref(nt))
        emis = [buf[i] for i in range(min(n, 4096))]
        if 0 in emis:                                       # couple rendu au C (hors domaine) : pas compare
            n0 += 1
            continue
        ref, ntr = ref_couple(G, S, u0, s0, u2, s2, P0, maxp, Emax, Evrai)
        verifie(sorted(emis) == sorted(ref), f"{nom} couple G={gtxt} u0={u0} u2={u2} P0={P0} maxp={maxp} "
                                             f"Emax={Emax} Evrai={Evrai} : noyau {emis} / reference {ref}")
        verifie(nt.value == ntr, f"{nom} couple G={gtxt} u0={u0} u2={u2} : p testes {nt.value} / reference {ntr}")
        ncomp += 1
        nemis += len(ref)
    print(f"  2. couples ({nom}) : {ncomp} compares ({nemis} p emis au total, {n0} rendus au C), "
          f"{time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ 3. chemins entiers
def run(args, env=None, out=None):
    """apparie_graines.py ; renvoie (ensemble des plus petits membres des cycles, statistiques)."""
    d = tempfile.mkdtemp(prefix="tc_", dir=ICI)
    out = out or os.path.join(d, "o.txt")
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable, os.path.join(RACINE, "apparie_graines.py"), *args, "--out", out, "--budget", "0"],
                       capture_output=True, text=True, env=e, cwd=d)
    m = re.search(r"statistiques : (\{.*\})", r.stdout)
    st = eval(m.group(1)) if m else {}
    cyc = set()
    if os.path.exists(out):
        for b in open(out).read().split("#")[1:]:
            ns = [int(x) for x in re.findall(r"^\s*(\d{6,})\s", b, re.M)]
            if ns:
                cyc.add(min(ns))
    if not m:
        verifie(False, f"run {' '.join(args)} : pas de statistiques\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return cyc, st


def section3():
    t0 = time.time()
    L = os.path.join(ICI, "Gt.txt")
    open(L, "w").write("\n".join(["3*5*7", "3^2*5*7", "3^3*11", "3^2*5*13", "3*5*7*13", "3^3*5*13", "3^4*5^2",
                                  "3^2*7*11", "5^2*13", "3*5*11*13"]) + "\n")
    base = ["--liste", L, "--p0", "1000", "--maxp", "200", "--jobs", "1"]
    simu = os.path.join(ICI, "libapparie_simu.so")
    subprocess.run(["g++", "-O3", "-fopenmp", "-fPIC", "-shared", "-x", "c++", "-DAG_F32", "-DAG_SIMU", "-o", simu,
                    os.path.join(RACINE, "apparie_gpu.cu")], check=True)
    chemins = [("C (GMP)", ["--sans-noyau"]), ("noyau CPU", ["--gpu", "cpu"]), ("noyau CPU warps", ["--gpu", simu])]
    if A.gpu:
        chemins.append(("GPU", ["--gpu", os.path.abspath(A.gpu)]))
    for U in ("3e3", "1e5"):
        res = {}
        if U == "3e3":
            res["Python pur"] = run(base + ["--u", U, "--sans-c"])
        for nom, opt in chemins:
            res[nom] = run(base + ["--u", U] + opt)
        ref_c, ref_st = res["C (GMP)"]
        for nom, (c, st) in res.items():
            verifie(c == ref_c, f"u={U} {nom} : {len(c)} cycles / C {len(ref_c)} (difference {sorted(c ^ ref_c)[:4]})")
            verifie(st.get("premiers") == ref_st.get("premiers"),
                    f"u={U} {nom} : premiers {st.get('premiers')} / C {ref_st.get('premiers')}")
        print(f"     u={U} : {len(ref_c)} cycles, {ref_st.get('premiers')} p premiers, identiques sur {len(res)} chemins",
              flush=True)
    # mode hybride force : borne 64 bits abaissee, memes cycles
    U = "1e5"
    ref = run(base + ["--u", U, "--gpu", "cpu"])[0]
    for cap in ("100", "1000000"):
        for nom, opt in chemins[1:]:
            c, st = run(base + ["--u", U] + opt, env={"AG_TEST_CAP": cap})
            verifie(c == ref and st.get("emax_hybride"), f"hybride cap={cap} {nom} : {len(c)} cycles / {len(ref)}")
    # partition : --u-deja redonne exactement le run complet
    # (et les compteurs s'ajoutent exactement : chaque couple est vu une fois et une seule)
    a, sa = run(base + ["--u", "5e4", "--gpu", "cpu"])
    b, sb = run(base + ["--u", "1e5", "--u-deja", "5e4", "--gpu", "cpu"])
    _, sp = run(base + ["--u", "1e5", "--gpu", "cpu"])
    verifie(a | b == ref, f"--u-deja : {len(a)} + {len(b)} cycles, union {len(a | b)} / complet {len(ref)}")
    for k in ("couples", "p_testes", "th2", "premiers"):
        verifie(sa.get(k, 0) + sb.get(k, 0) == sp.get(k), f"--u-deja : {k} {sa.get(k)} + {sb.get(k)} != {sp.get(k)}")
    for cap in ("1000",):                              # --u-deja en mode hybride, chemin des warps du GPU
        b2, sb2 = run(base + ["--u", "1e5", "--u-deja", "5e4", "--gpu", simu], env={"AG_TEST_CAP": cap})
        verifie(b2 == b and sb2.get("premiers") == sb.get("premiers"), f"--u-deja hybride : {len(b2)} cycles / {len(b)}")
    print(f"  3. chemins entiers, hybride, --u-deja : {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ 4. rappel sur le catalogue
def section4(U=100000, P0=1000, maxp=200, K=1e7):
    import gmpy2
    t0 = time.time()
    if not os.path.exists(A.catalogue):
        print(f"  4. rappel : {A.catalogue} introuvable, section sautee")
        return
    def fac(s):
        f = {}
        for t in s.split("."):
            p, _, e = t.partition("E"); f[int(p)] = f.get(int(p), 0) + (int(e) if e else 1)
        return f
    sig = lambda f: math.prod((p ** (e + 1) - 1) // (p - 1) for p, e in f.items())
    val = lambda f: math.prod(p ** e for p, e in f.items())
    txt = lambda d: "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(d.items()))
    blocs, cur = [], []
    for l in open(A.catalogue, errors="replace"):
        m = re.match(r"^\s*\d*\s+(\d{4,})\s+([\dE.]+)\s*$", l)
        if m:
            cur.append((int(m.group(1)), m.group(2)))
        elif not l.strip():
            if len(cur) == 4:
                blocs.append(cur)
            cur = []
    attendus, graines = {}, []
    for b in blocs:
        F = {n: fac(s) for n, s in b}
        n = min(F); o = []
        while n in F and n not in o:
            o.append(n); n = sig(F[n]) - n
        if len(o) != 4:
            continue
        for k in range(4):
            n1, n2, n3, n4 = [o[(k + j) % 4] for j in range(4)]
            a1, a2 = gcd(n1, n2), gcd(n3, n4)
            if not (gmpy2.is_prime(n2 // a1) and gmpy2.is_prime(n4 // a2)):
                continue
            f1 = {p: e for p, e in F[n1].items() if a1 % p == 0}
            f2 = {p: e for p, e in F[n3].items() if a2 % p == 0}
            G = {p: e for p, e in f1.items() if f2.get(p) == e}
            if not G:
                continue
            g, S = val(G), sig(G)
            if not 1 < S / g < 2:
                continue
            for x, y in ((a1, a2), (a2, a1)):                 # x = G*u0*p, y = G*u2
                c0, u2 = x // g, y // g
                if u2 > U:
                    continue
                for p in factorint_simple(c0):
                    u0 = c0 // p
                    if p >= P0 and c0 % (p * p) and u0 <= U and gcd(u0, g) == 1 and gcd(u2, g) == 1:
                        s0, s2 = sig_int(u0), sig_int(u2)
                        al = g * (s0 * u2 + s2 * u0) - S * s0 * s2
                        be = s0 * (g * u2 - S * s2)
                        if al <= 0 or be >= 0:
                            continue
                        pmin = max(-be // al + 1, P0)
                        E = al * p + be
                        if pmin <= p <= pmin + maxp and 0 < E <= K * g * g // S:
                            attendus[min(o)] = txt(G)
                            graines.append((g, S, u0, s0, u2, s2, p, int(K * g * g // S)))
            break
    if not attendus:
        print("  4. rappel : aucun cycle du catalogue dans le domaine, section sautee")
        return
    # le noyau seul (CPU, et chemin GPU AG_F32) doit emettre le p de chaque graine attendue
    buf, nt = (u64 * 4096)(), u64()
    for nomL, Lk in LIBS:
        perdus = 0
        for g, S, u0, s0, u2, s2, p, Emax in graines:
            if g >= 1 << 44 or S >= 1 << 46:               # hors du domaine du noyau (chemin C seul)
                continue
            Ev = 0
            if Emax >= 1 << 63:
                Ev, Emax = Emax, (1 << 63) - 1
            n = Lk.t_couple(g, S, u0, s0, u2, s2, P0, maxp, Emax, *hl(Ev), buf, 4096, ctypes.byref(nt))
            em = {buf[i] & ((1 << 63) - 1) for i in range(min(n, 4096))}
            if p not in em and 0 not in em:
                perdus += 1
                verifie(False, f"rappel noyau {nomL} : p={p} non emis pour G={g} u0={u0} u2={u2}")
        nd = sum(1 for x in graines if x[0] < 1 << 44 and x[1] < 1 << 46)
        print(f"     noyau {nomL} : {nd - perdus}/{nd} graines du catalogue emettent leur p", flush=True)
    L = os.path.join(ICI, "Gcat.txt")
    open(L, "w").write("".join(g + "\n" for g in sorted(set(attendus.values()))))
    trouves, st = run(["--liste", L, "--u", str(U), "--p0", str(P0), "--maxp", str(maxp), "--jobs", "2", "--gpu", "cpu"])
    manques = [n for n in attendus if n not in trouves]
    verifie(not manques, f"rappel : {len(manques)} cycle(s) du catalogue NON retrouves, ex. {manques[:3]} "
                         f"(G {[attendus[n] for n in manques[:3]]})")
    print(f"  4. rappel : {len(attendus) - len(manques)}/{len(attendus)} cycles du catalogue retrouves "
          f"({len(set(attendus.values()))} G, u <= {U:g}), {time.time() - t0:.0f} s", flush=True)


def factorint_simple(n):
    from sympy import factorint
    return [int(p) for p in factorint(n)]


def sig_int(n):
    from sympy import divisor_sigma
    return int(divisor_sigma(n))


if __name__ == "__main__":
    os.chdir(RACINE)
    print("tests croises apparie_graines / apparie_gpu", flush=True)
    for nom, defs in (("CPU", []), ("chemin GPU (AG_F32)", ["-DAG_F32"])):
        L = preparer(compiler("libtests_arith" + ("_f32" if defs else "") + ".so", defs))
        LIBS.append((nom, L))
        section1(L, nom)
        section2(L, nom, 3000 if A.rapide else 20000)
    if not A.rapide:
        section3()
        section4()
    print(f"\n{'TOUT PASSE' if not ECHECS else str(len(ECHECS)) + ' ECHEC(S)'}")
    sys.exit(1 if ECHECS else 0)
