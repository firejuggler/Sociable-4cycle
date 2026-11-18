"""
apparie_graines.py — 4-cycles de Borho SANS paires amicales : graines construites par appariement
================================================================================================
Le theoreme de Borho (theoreme1 de borho4.py) n'a besoin que d'un couple de graines (a1, a2). Ici les
graines sont construites directement : a1 = G*u0*p, a2 = G*u2, avec G une partie commune, u0, u2 <= --u
premiers avec G, et p un GRAND premier inconnu (p >= --p0) calcule par la methode.

Principe (S = sigma(G), h = S/G, y(u) = u/sigma(u)) :
  D = a1*a2 - tau1*tau2 = S*E, et pour u1 = u0*p : E = alpha*p + beta, avec
  alpha = G*s0*u2 + G*s2*u0 - S*s0*s2, beta = s0*(G*u2 - S*s2) (s0 = sigma(u0), s2 = sigma(u2)).
  Un petit D (0 < D <= K*G^2, comme dans les cycles connus) demande y(u0) + y(u2) juste au-dessus de h :
  p ~ (h - y2)/eps, eps = y0 + y2 - h. Les couples (u0, u2) sont donc trouves par tri des y (appariement,
  comme les breeders de Chernykh), puis pour chaque p possible : theoreme 2 (D | ...), primalite de p,
  et theoreme 1 complet (borho4.theoreme1), qui donne les 4-cycles, verifies.

Listes de parties communes G :
  --G "3^2*5*13*19"                       une seule (test : redonne le cycle n° 471 de sociable)
  --liste G_liste.txt                      un G par ligne (premier mot)
  --depuis-catalogue sociable.txt          les G des 4-cycles de Borho « purs » du catalogue (ecrit G_catalogue.txt)
  --hors-catalogue 1e5                     tous les G <= 1e5 avec 1.8 <= h < 2 absents de G_catalogue.txt
                                           (ecrit G_hors_catalogue.txt ; G_catalogue.txt doit exister)
  --depuis-paires c2_80*.txt.gz            les parties communes exactes des paires amicales de ces fichiers
                                           (1.5 <= h < 2, --gmin <= G <= --gmax), les plus frequentes d'abord,
                                           hors G_catalogue.txt s'il existe (ecrit G_paires.txt) : grands G,
                                           donc grandes graines et grands cycles, region peu exploree

Usage (dans le dossier de borho4.py ; numpy, sympy, gmpy2) :
  python3 apparie_graines.py --G "3^2*5*13*19" --u 1e3 --p0 1e5                     (test, < 1 s)
  python3 apparie_graines.py --depuis-catalogue sociable.txt --u 1e4 --jobs 4 --connus sociable.txt mes_cycles_propre.txt
  python3 apparie_graines.py --liste G_catalogue.txt --debut 0 --nombre 200 --u 1e5 --jobs 4 --connus sociable.txt
  python3 apparie_graines.py --hors-catalogue 1e5 --u 1e5 --jobs 4 --connus sociable.txt mes_cycles_propre.txt
Options : --u (u0, u2 <= u, defaut 1e5), --p0 (p >= p0, defaut 1e5), --K (D <= K*G^2, defaut 1e7),
  --budget (secondes max par G, defaut 60 ; les G coupes vont dans <out>.incomplets), --out
  (defaut apparie_cycles.txt ; les cycles deja dans --out ne sont pas reecrits).
Filtres (sur toute source de G, avant --debut / --nombre) : --impairs (ecarte les G pairs), --hmin 1.85
  (ecarte les G d'abondance plus basse), --exclure fichiers... (ecarte les G deja dans ces listes ou .faits).
  Exemple : --depuis-paires c2_22.txt.gz --impairs --hmin 1.85 --exclure G_*.txt ag_*.txt.faits
Reprise : chaque G termine (non coupe par le budget) est note dans <out>.faits avec les parametres du run
  (u, p0, K, maxp) ; relancer la meme commande saute les G deja faits avec ces parametres (--refaire pour
  les relancer ; un --G unique est toujours refait). Ctrl-C arrete proprement, l'avancement est garde.
  Un G qui provoque une erreur n'arrete plus le run : il est note dans <out>.erreurs.
Boucle des couples : --gpu (libapparie_gpu.so) ; sans --gpu, libapparie_cpu.so (OpenMP, compilee au besoin
  depuis apparie_gpu.cu, coeurs partages entre les --jobs) est utilisee par defaut : meme calcul que le
  module C, plusieurs fois plus rapide. --sans-noyau : chemin C habituel (controle).
"""
import c2io                     # lecture des c2_* : .txt, .gz et .gz « factorisation seule »
import argparse
import os
import sys
import time
from math import gcd, isqrt
from multiprocessing import get_context

import gmpy2
import numpy as np
import sympy

import borho4 as B

SIG = None          # table sigma(u), u <= U (heritee par les processus au fork)
TRI = None          # (u, sigma(u), y = u/sigma(u)) tries par y, calcules une fois (herites au fork)
PETITS = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
MASQUES = {}        # q -> (u % q != 0) sur TRI[0], pour les petits premiers (herites au fork)


def tri():
    """u <= U tries une seule fois par y = u/sigma(u) (y ne depend pas de G)."""
    global TRI
    if TRI is None:
        u = np.arange(1, len(SIG), dtype=SIG.dtype)
        su = SIG[1:]
        y = u / su
        o = np.argsort(y, kind="stable")
        ty = np.int32 if len(SIG) < 2 ** 31 and int(su.max()) < 2 ** 31 else np.int64   # 32 bits : memoire / 2
        TRI = (u[o].astype(ty), su[o].astype(ty), y[o])   # y deja calcule : reordonne, pas recalcule
        del y
        # masques « u premier avec q » des petits premiers, une fois (partages par les processus au fork) :
        # le filtrage par G ne refait plus un modulo sur tout le tableau pour chacun d'eux
        MASQUES.clear()
        for q in PETITS:
            MASQUES[q] = (TRI[0] % q) != 0
    return TRI
P = {}
_LIB = None         # module C (libborho4c.so, fonction apparie_couples) ; False : Python pur


def lib_c():
    global _LIB
    if _LIB is None:
        _LIB = False
        if not P.get("sans_c") and B.charger_c():
            import ctypes
            try:
                lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(B.__file__)), "libborho4c.so"))
                L = ctypes.POINTER(ctypes.c_long)
                f2 = lib.apparie_couples2                          # boucle sur p + theoremes 2 et 1 en C
                f2.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, L, L, L, L, ctypes.c_ulong,
                               ctypes.c_ulong, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_ulong), ctypes.c_int, L]
                f2.restype = ctypes.c_int
                _LIB = f2
                try:                                       # variante « un p par couple » (survivants GPU)
                    fp = lib.apparie_couples_p
                    fp.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, L, L, L, L,
                                   ctypes.POINTER(ctypes.c_ulong), ctypes.c_char_p, ctypes.c_long,
                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong), ctypes.c_int, L]
                    fp.restype = ctypes.c_int
                    P["_fp"] = fp
                except AttributeError:
                    P["_fp"] = None
            except (OSError, AttributeError):
                _LIB = False
    return _LIB


_GPU = None        # (bibliotheque, chemin) : libapparie_gpu.so ou libapparie_cpu.so ; False : absente


def compiler_cpu(verbeux=False):
    """libapparie_cpu.so (boucle des couples sur CPU, OpenMP), compilee depuis apparie_gpu.cu si elle manque
    ou est plus ancienne que la source (comme libborho4c.so). Renvoie son chemin, ou None."""
    import subprocess
    ici = os.path.dirname(os.path.abspath(__file__))
    src, so = os.path.join(ici, "apparie_gpu.cu"), os.path.join(ici, "libapparie_cpu.so")
    if os.path.isfile(so) and (not os.path.isfile(src) or os.path.getmtime(so) >= os.path.getmtime(src)):
        return so
    if not os.path.isfile(src):
        return None
    cmd = ["g++", "-O3", "-fopenmp", "-fPIC", "-shared", "-x", "c++", "-o", so + ".tmp", src]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as ex:
        r = None
        err = str(ex)
    if r is not None and r.returncode == 0:
        os.replace(so + ".tmp", so)
        return so
    if verbeux:
        err = (r.stderr.strip().splitlines()[-1:] or ["?"])[0] if r is not None else err
        print(f"[i] compilation de libapparie_cpu.so impossible ({err}) : chemin C habituel", flush=True)
    return so if os.path.isfile(so) else None


def lib_gpu():
    """Boucle rapide des couples. P["gpu"] : chemin d'une bibliotheque, "auto" (--gpu : libapparie_gpu.so,
    sinon libapparie_cpu.so), "cpu" (defaut sans --gpu : libapparie_cpu.so, compilee au besoin) ou None."""
    global _GPU
    if _GPU is None:
        _GPU = False
        nom = P.get("gpu")
        if nom:
            import ctypes
            ici = os.path.dirname(os.path.abspath(__file__))
            if nom == "cpu":
                cands = [compiler_cpu()]
            elif nom == "auto":
                cands = [os.path.join(ici, "libapparie_gpu.so"), compiler_cpu()]
            else:
                cands = [nom]
            for c in cands:
                if c and os.path.isfile(c):
                    L = ctypes.CDLL(c)
                    i64, u64, ip = ctypes.c_int64, ctypes.c_uint64, ctypes.c_void_p
                    L.ag_charger.argtypes = [ip, ip, ctypes.c_int, ip, ip, ip, ctypes.c_int, i64, i64, u64, u64,
                                             u64, ctypes.c_int]
                    L.ag_charger.restype = ctypes.c_int
                    L.ag_scan.argtypes = [i64, i64, ip, ip, ip, ctypes.POINTER(ctypes.c_ulonglong),
                                          ctypes.POINTER(ctypes.c_ulonglong)]
                    L.ag_scan.restype = ctypes.c_int
                    L.ag_liberer.restype = None
                    if getattr(L, "ag_emax_vrai", None) is not None:   # noyau hybride : E > 2^63 rendu au C
                        L.ag_emax_vrai.argtypes = [u64, u64]
                        L.ag_emax_vrai.restype = None
                    fils = getattr(L, "ag_fils", None)     # version CPU : coeurs partages entre les --jobs
                    if fils is not None and P.get("fils"):
                        fils.argtypes = [ctypes.c_int]
                        fils.restype = None
                        fils(int(P["fils"]))
                    vrai = fils is None and "cpu" not in os.path.basename(c)   # ancienne version CPU : sans ag_fils
                    _GPU = (L, c, vrai)                     # (bibliotheque, chemin, vrai GPU ?)
                    break
    return _GPU


def sigma_tab(U):
    """sigma(u) pour u <= U. Chaque paire de diviseurs (d, j), d <= j, d*j = n, est ajoutee depuis le plus
    petit : boucle sur d <= racine(U) seulement (pour U = 1e7 : 3 162 operations numpy au lieu de 10 millions)."""
    s = np.zeros(U + 1, dtype=np.int32 if U <= 2 * 10 ** 8 else np.int64)   # sigma(u) < 2^31 pour u <= 2e8
    for d in range(1, isqrt(U) + 1):
        s[d * d::d] += d                               # n = d*j, j >= d : le diviseur d
        if d * d + d <= U:
            s[d * d + d::d] += np.arange(d + 1, U // d + 1, dtype=s.dtype)   # j > d : le codiviseur j
    return s


fmt = B.fmt


def rendre_memoire():
    """Rend au systeme la memoire liberee (glibc la garde sinon : fragmentation entre deux G)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def chercher(g):
    try:
        return _chercher(g)
    except Exception as ex:                                # un G anormal n'arrete pas le run
        return g, [], dict(erreur=1, _message=f"{type(ex).__name__} : {ex}")
    finally:
        rendre_memoire()


def _chercher(g):
    """Tous les 4-cycles pour la partie commune g (texte). Renvoie (g, cycles, stats)."""
    G = B.parse(g)
    U, P0, K, budget = P["U"], P["P0"], P["K"], P["budget"]
    Gv, S = B.val(G), B.sig(G)
    h = S / Gv
    st = dict(couples=0, p_testes=0, th2=0, premiers=0, incomplet=0)
    cyc = []
    if not 1 < h < 2:
        return g, cyc, st
    Ut, St, Yt = tri()
    ok = np.ones(len(Ut), dtype=bool)
    for q in G:
        if q in MASQUES:
            ok &= MASQUES[q]                           # petit premier : masque precalcule
        elif q <= U:                                   # un premier > U ne divise aucun u <= U
            ok &= (Ut % q) != 0                     # (et depasserait les entiers 32 bits)
    ix = np.flatnonzero(ok)                            # filtrer garde l'ordre des y
    del ok
    Uu, Ss, Y = Ut.take(ix), St.take(ix), Yt.take(ix)
    del ix
    Emax = int(K * Gv * Gv // S)
    Evrai = 0                                          # > 0 : noyau hybride (E <= 2^63 noyau, au-dela en C)
    if (Emax >= 1 << 63 and Emax < 1 << 127 and P.get("gpu") and lib_gpu()
            and getattr(lib_gpu()[0], "ag_emax_vrai", None) is not None):
        # noyau a jour : il traite E < 2^63 en 64 bits et 2^63 <= E <= K*G^2/S en 128 bits (p rendus avec
        # le bit 63 leve, theoreme 2 refait en C) ; seuls les couples hors domaine reviennent avec p = 0.
        # La recherche est donc complete et le G est note dans .faits.
        Evrai = Emax
        Emax = (1 << 63) - 1
        st["emax_hybride"] = 1
    elif os.environ.get("AG_TEST_CAP") and P.get("gpu") and lib_gpu() \
            and getattr(lib_gpu()[0], "ag_emax_vrai", None) is not None:
        # controle du chemin hybride sur de petits G : borne du noyau artificiellement abaissee
        Evrai = Emax
        Emax = max(1, Emax // int(os.environ["AG_TEST_CAP"]))
        st["emax_hybride"] = 1
    elif P.get("gpu") not in (None, "cpu") and Emax >= 1 << 63 and lib_gpu():
        # (seulement avec --gpu explicite : par defaut, ces G restent sur le chemin C, sans borne)
        # gros G (au-dela de ~1,7e12 pour K = 1e7) : le noyau GPU travaille avec E < 2^63. Plutot que de
        # passer tout le G sur CPU (des minutes), on borne E a 2^63 - 1 : D <= S * (2^63 - 1) au lieu de K G^2.
        Emax = (1 << 63) - 1
        st["emax_borne"] = 1
    if P.get("ext"):                                   # --etendu : u2 = v*q (grand premier q), u0 <= U
        return _chercher_etendu(g, G, Gv, S, h, Uu, Ss, Y, Emax, Evrai, st, cyc)
    if P.get("u_deja"):                                # --u-deja : sans les couples deja faits a un u plus petit
        return _chercher_suite(g, G, Gv, S, h, Uu, Ss, Y, Emax, Evrai, st, cyc)
    jx, lo, n = fenetres(Y, Y, h, P0)
    return _passe(g, G, Gv, S, Uu, Ss, jx, lo, n, Emax, Evrai, st, cyc, time.time())


def fenetres(Ycible, Ysrc, h, P0, decal=0):
    """Pour chaque u2 (y = Ysrc[k]) : fenetre [lo, lo + n) des u0 dans Ycible (trie), indices k + decal ;
    seuls les u2 qui ont des partenaires sont gardes."""
    jx, lo, n = [], [], []
    for c in range(0, len(Ysrc), 1 << 20):
        t = h - Ysrc[c:c + (1 << 20)]
        l_ = np.searchsorted(Ycible, t - 1e-15, side="right")  # marge : y arrondis a ~1e-16 ; alpha exact trie
        n_ = np.searchsorted(Ycible, t + t / P0 * 1.000001 + 1e-15, side="right") - l_
        k_ = np.nonzero(n_)[0]
        jx.append((k_ + c + decal).astype(np.int32)); lo.append(l_[k_].astype(np.int32)); n.append(n_[k_].astype(np.int32))
    if not jx:
        return (np.zeros(0, np.int32),) * 3
    return np.concatenate(jx), np.concatenate(lo), np.concatenate(n)


def _chercher_suite(g, G, Gv, S, h, Uu, Ss, Y, Emax, Evrai, st, cyc):
    """--u-deja Ud : seulement les couples (u0, u2) dont AU MOINS UN depasse Ud. Les couples ou les deux sont
    <= Ud ont ete faits par un run precedent a --u Ud (memes p0, K, maxp) : ils ne sont pas refaits.
      passe A : u2 > Ud, u0 quelconque ;  passe B : u2 <= Ud, u0 > Ud."""
    Ud, P0 = P["u_deja"], P["P0"]
    t0 = time.time()
    haut = Uu > Ud
    total = dict(st)

    def fusion(st1):
        for k, x in st1.items():
            total[k] = total.get(k, 0) + x if k not in ("incomplet", "gpu", "noyau_cpu", "emax_hybride", "emax_borne") \
                else max(total.get(k, 0), x)

    ia = np.flatnonzero(haut).astype(np.int64)
    jx, lo, n = fenetres(Y, Y[ia], h, P0)
    if len(jx):
        st1 = dict(couples=0, p_testes=0, th2=0, premiers=0, incomplet=0)
        # u2 = Uu[ia[jx]] : tableau etendu [Uu, Uu[ia]] pour que jx designe u2
        _passe(g, G, Gv, S, np.concatenate([Uu, Uu[ia]]), np.concatenate([Ss, Ss[ia]]),
               (jx + len(Uu)).astype(np.int32), lo, n, Emax, Evrai, st1, cyc, t0)
        fusion(st1)
    if not total.get("incomplet"):
        Uh, Sh, Yh = Uu[haut], Ss[haut], Y[haut]
        ib = np.flatnonzero(~haut)
        jx, lo, n = fenetres(Yh, Y[ib], h, P0)
        if len(jx):
            st1 = dict(couples=0, p_testes=0, th2=0, premiers=0, incomplet=0)
            _passe(g, G, Gv, S, np.concatenate([Uh, Uu[ib]]), np.concatenate([Sh, Ss[ib]]),
                   (jx + len(Uh)).astype(np.int32), lo, n, Emax, Evrai, st1, cyc, t0)
            fusion(st1)
    cyc[:] = list({repr(c): c for c in cyc}.values())
    return g, cyc, total


def _passe(g, G, Gv, S, Uu, Ss, jx, lo, n, Emax, Evrai, st, cyc, t0, noyau_seul=False):
    """Couples (u0 = Uu[lo[w] + k], u2 = Uu[jx[w]]), k < n[w] : crible des p puis theoremes 2 et 1."""
    U, P0, K, budget = P["U"], P["P0"], P["K"], P["budget"]
    tot = int(n.sum(dtype=np.int64))
    st["couples"] = tot

    def paquets(taille=1 << 20):
        """Couples (I = u0, J = u2) par paquets d'au plus ~taille : jamais tous en memoire (un G peut en
        avoir des milliards ; le budget de temps l'arrete alors proprement)."""
        cs = np.cumsum(n, dtype=np.int64)
        j0 = 0
        while j0 < len(n):
            base = cs[j0 - 1] if j0 else 0
            j1 = int(np.searchsorted(cs, base + taille, side="right"))
            j1 = max(j1, j0 + 1)
            nn = n[j0:j1]
            m = int(nn.sum())
            if m:
                J_ = np.repeat(jx[j0:j1], nn)
                I_ = (np.arange(m) - np.repeat(np.cumsum(nn, dtype=np.int64) - nn, nn)) + np.repeat(lo[j0:j1], nn)
                yield I_, J_
            j0 = j1
    f = lib_c()
    if f:
        import ctypes
        L = ctypes.POINTER(ctypes.c_long)
        cap = 1 << 16
        ok_ = (ctypes.c_int * cap)(); op_ = (ctypes.c_ulong * cap)(); sts = (ctypes.c_long * 8)()
        gfac, ss, emax = fmt(G).encode(), str(S).encode(), str(Emax).encode()
        emax_vrai = str(Evrai).encode() if Evrai else emax

        def appel(ii, jj, p0=None, mp=None, em=None):
            """Un paquet de couples en C ; si le tampon de sortie deborde, le paquet est coupe en deux."""
            p0 = P0 if p0 is None else p0
            mp = P.get("maxp", 2000) if mp is None else mp
            a_u0 = np.ascontiguousarray(Uu[ii], dtype=np.int64); a_s0 = np.ascontiguousarray(Ss[ii], dtype=np.int64)
            a_u2 = np.ascontiguousarray(Uu[jj], dtype=np.int64); a_s2 = np.ascontiguousarray(Ss[jj], dtype=np.int64)
            avant = list(sts)
            r = f(gfac, ss, len(ii), a_u0.ctypes.data_as(L), a_s0.ctypes.data_as(L), a_u2.ctypes.data_as(L),
                  a_s2.ctypes.data_as(L), p0, mp, emax if em is None else em, 10 ** 6, ok_, op_, cap, sts)
            if r == -2 and len(ii) > 1:
                for i in range(8):
                    sts[i] = avant[i]                      # le paquet est refait en deux moities
                h_ = len(ii) // 2
                return appel(ii[:h_], jj[:h_], p0, mp, em) + appel(ii[h_:], jj[h_:], p0, mp, em)
            if r < 0:
                raise RuntimeError(f"apparie_couples2 : code {r}")
            return [(int(a_u0[ok_[m]]), int(a_s0[ok_[m]]), int(a_u2[ok_[m]]), int(a_s2[ok_[m]]), int(op_[m]))
                    for m in range(r)]

        def appel_p(ii, jj, pk):
            """Survivants du GPU : couples (ii, jj) avec chacun SON p (pk), en C par paquets."""
            fp = P.get("_fp")
            if fp is None:                                 # libborho4c.so ancienne : un appel par survivant
                return [x for k in range(len(ii)) for x in appel(ii[k:k + 1], jj[k:k + 1], int(pk[k]), 0)]
            res = []
            for d in range(0, len(ii), 4096):
                i_, j_, p_ = ii[d:d + 4096], jj[d:d + 4096], pk[d:d + 4096]
                a_u0 = np.ascontiguousarray(Uu[i_], dtype=np.int64); a_s0 = np.ascontiguousarray(Ss[i_], dtype=np.int64)
                a_u2 = np.ascontiguousarray(Uu[j_], dtype=np.int64); a_s2 = np.ascontiguousarray(Ss[j_], dtype=np.int64)
                a_p = np.ascontiguousarray(p_, dtype=np.uint64)
                avant = list(sts)
                r = fp(gfac, ss, len(i_), a_u0.ctypes.data_as(L), a_s0.ctypes.data_as(L), a_u2.ctypes.data_as(L),
                       a_s2.ctypes.data_as(L), a_p.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)), emax_vrai, 10 ** 6,
                       ok_, op_, cap, sts)
                if r == -2:
                    for i in range(8):
                        sts[i] = avant[i]
                    h_ = len(i_) // 2
                    res += appel_p(i_[:h_], j_[:h_], p_[:h_]) + appel_p(i_[h_:], j_[h_:], p_[h_:])
                    continue
                if r < 0:
                    raise RuntimeError(f"apparie_couples_p : code {r}")
                res += [(int(a_u0[ok_[m]]), int(a_s0[ok_[m]]), int(a_u2[ok_[m]]), int(a_s2[ok_[m]]), int(op_[m]))
                        for m in range(r)]
            return res

        gp = lib_gpu()
        # domaine du noyau (128 bits) : avec bG, bS, bu, bs les tailles en bits de G, S, max u, max sigma(u),
        # |beta| < 2^(bs + max(bG+bu, bS+bs)), alpha < 2^(bG+bs+bu+1) et X = u0*beta + alpha*u2 doit tenir
        # dans un i128 : bu + bs + max(bG+bu, bS+bs) <= 125 et bG + bs + 2*bu + 1 <= 125.
        # (Ancienne borne fixe G < 2^44, S < 2^46, u < 2^24, s < 2^26 : un cas particulier.) Les petits G
        # passent ainsi sur GPU jusqu'a u ~ 2^30 au lieu de 2^24 = 1.68e7.
        dans = False
        if gp and len(Ss) and Emax < 1 << 63 and tot > 0 and Gv < 1 << 44 and S < 1 << 46 and len(Uu) < 1 << 31:
            bG, bS = int(Gv).bit_length(), int(S).bit_length()
            bu, bs = int(Uu.max()).bit_length(), int(Ss.max()).bit_length()
            dans = bu + bs + max(bG + bu, bS + bs) <= 125 and bG + bs + 2 * bu + 1 <= 125 and bu <= 31 and bs <= 33
        if noyau_seul and not dans:
            st["hors_noyau"] = 1                           # --etendu : pas de repli sur le chemin C (trop lent)
            return g, cyc, st
        if dans:
            Lg = gp[0]
            st["gpu" if gp[2] else "noyau_cpu"] = 1
            a_U = np.ascontiguousarray(Uu, dtype=np.int64); a_S = np.ascontiguousarray(Ss, dtype=np.int64)
            a_jx = np.ascontiguousarray(jx, dtype=np.int32); a_lo = np.ascontiguousarray(lo, dtype=np.int32)
            a_cs = np.ascontiguousarray(np.cumsum(n, dtype=np.int64))
            capg = 1 << 20
            if Lg.ag_charger(a_U.ctypes.data, a_S.ctypes.data, len(a_U), a_jx.ctypes.data, a_lo.ctypes.data,
                             a_cs.ctypes.data, len(a_cs), Gv, S, P0, P.get("maxp", 2000), Emax, capg) != 0:
                raise RuntimeError("ag_charger : erreur (voir le message CUDA)")
            if getattr(Lg, "ag_emax_vrai", None) is not None:
                Lg.ag_emax_vrai(Evrai >> 64, Evrai & ((1 << 64) - 1))
            oi = np.zeros(capg, dtype=np.int32); oj = np.zeros(capg, dtype=np.int32); op = np.zeros(capg, dtype=np.uint64)
            ntg = ctypes.c_ulonglong(0); n2g = ctypes.c_ulonglong(0)
            nt_gpu = n2_gpu = n_unit = n_128 = 0
            pas = 1 << int(os.environ.get("AG_PAS", "31"))   # couples par appel du noyau (2^28 avant : 8 % plus lent)
            t_ = 0
            try:
                while t_ < tot:
                    if budget and time.time() - t0 > budget:
                        st["incomplet"] = 1
                        break
                    t1_ = min(tot, t_ + pas)
                    tA = time.time()
                    r = Lg.ag_scan(t_, t1_, oi.ctypes.data, oj.ctypes.data, op.ctypes.data, ctypes.byref(ntg),
                                   ctypes.byref(n2g))
                    st["t_crible"] = st.get("t_crible", 0) + time.time() - tA
                    tA = time.time()
                    if r == -2:                            # trop de sorties : tranche plus petite
                        pas = max(1, pas // 8)
                        continue
                    if r < 0:
                        raise RuntimeError(f"ag_scan : code {r}")
                    nt_gpu += ntg.value
                    n2_gpu += n2g.value
                    n128 = int((op[:r] >> np.uint64(63)).sum())    # bit 63 : p de la suite 128 bits (E > 2^63),
                    op[:r] &= np.uint64((1 << 63) - 1)             # sans theoreme 2 fait par le noyau
                    st["p_128"] = st.get("p_128", 0) + n128
                    m_ = op[:r] != 0                       # (u0, u2, p) refaits en C, chacun avec son p
                    if not m_.all():                       # p = 0 : hors domaine, couple entier en C
                        st["rendus_C"] = st.get("rendus_C", 0) + int((~m_).sum())
                        for u0, s0, u2, s2, p in appel(oi[:r][~m_].copy(), oj[:r][~m_].copy(), em=emax_vrai):
                            cyc += cycles_de(G, Gv, S, u0, s0, u2, s2, p)
                    n_unit += int(m_.sum())
                    n_128 += n128
                    for u0, s0, u2, s2, p in appel_p(oi[:r][m_].copy(), oj[:r][m_].copy(), op[:r][m_].copy()):
                        cyc += cycles_de(G, Gv, S, u0, s0, u2, s2, p)
                    st["survivants"] = st.get("survivants", 0) + r
                    st["t_survivants"] = st.get("t_survivants", 0) + time.time() - tA
                    t_ = t1_
            finally:
                Lg.ag_liberer()
            st["p_testes"] = nt_gpu + int(sts[0]) - n_unit       # les survivants refaits en C comptent deux fois
            st["th2"], st["premiers"] = n2_gpu + int(sts[1]) - (n_unit - n_128), int(sts[2])
            st["hors_64_bits"] = int(sts[3])
            st["th1_C_aucun_d1"], st["th1_C_ok"], st["th1_C_trop_div"], st["th1_Python"] = (int(sts[4]), int(sts[5]),
                                                                                         int(sts[6]), int(sts[7]))
            if Evrai:
                cyc = list({repr(c): c for c in cyc}.values())
            return g, cyc, st
        for d, ii, jj in ((d, I_[d:d + 4096], J_[d:d + 4096]) for I_, J_ in paquets() for d in range(0, len(I_), 4096)):
            if budget and time.time() - t0 > budget:
                st["incomplet"] = 1
                break
            for u0, s0, u2, s2, p in appel(ii, jj, em=emax_vrai):   # chemin C : toujours la vraie borne
                cyc += cycles_de(G, Gv, S, u0, s0, u2, s2, p)
        st["p_testes"], st["th2"], st["premiers"] = int(sts[0]), int(sts[1]), int(sts[2])
        st["hors_64_bits"] = int(sts[3])
        st["th1_C_aucun_d1"], st["th1_C_ok"], st["th1_C_trop_div"], st["th1_Python"] = (int(sts[4]), int(sts[5]),
                                                                                     int(sts[6]), int(sts[7]))
        return g, cyc, st
    for k, (i, j) in enumerate((i, j) for I_, J_ in paquets() for i, j in zip(I_.tolist(), J_.tolist())):
        if budget and k % 256 == 0 and time.time() - t0 > budget:
            st["incomplet"] = 1
            break
        u0, s0, u2, s2 = int(Uu[i]), int(Ss[i]), int(Uu[j]), int(Ss[j])
        alpha = Gv * s0 * u2 + Gv * s2 * u0 - S * s0 * s2
        beta = s0 * (Gv * u2 - S * s2)
        if alpha <= 0 or beta >= 0:
            continue
        pmin = max(P0, -beta // alpha + 1)
        pmax = min(((Evrai or Emax) - beta) // alpha, pmin + P.get("maxp", 2000))
        if pmax < pmin:
            continue
        M0 = abs(u0 * beta + alpha * u2) * Gv * Gv * gcd(u0, u2) * s2      # E divise M0
        for p in range(pmin, pmax + 1):
            E = alpha * p + beta
            st["p_testes"] += 1
            if M0 % E:
                continue
            a1, a2 = Gv * u0 * p, Gv * u2
            sa1, sa2 = S * s0 * (p + 1), S * s2
            if ((a1 - a2) * gcd(sa1, sa2) * gcd(a1, a2)) % (S * E):
                continue
            st["th2"] += 1
            if not gmpy2.is_prime(p) or (Gv * u0 * u2) % p == 0:
                continue
            st["premiers"] += 1
            f1 = dict(G); f1.update({int(q): e for q, e in sympy.factorint(u0).items()}); f1[p] = 1
            f2 = dict(G); f2.update({int(q): e for q, e in sympy.factorint(u2).items()})
            cs, _ = B.theoreme1(f1, f2, 10 ** 6, (a1, sa1, a2, sa2))
            cyc += cs
    return g, cyc, st


PREMIERS = None     # premiers <= --etendu-max (uint32), calcules une fois (herites au fork)


def premiers_jusqua(X):
    """Tous les premiers <= X (crible segmente numpy), en uint32."""
    r = isqrt(X) + 1
    c = np.ones(r + 1, dtype=bool); c[:2] = False
    for i in range(2, isqrt(r) + 1):
        if c[i]:
            c[i * i::i] = False
    base = np.flatnonzero(c)
    out = [base[base <= X].astype(np.uint32)]
    pas = 1 << 24
    for a in range(r + 1, X + 1, pas):
        b = min(X + 1, a + pas)
        seg = np.ones(b - a, dtype=bool)
        for p in base:
            p = int(p)
            if p * p >= b:
                break
            seg[(-a) % p::p] = False
        out.append((np.flatnonzero(seg) + a).astype(np.uint32))
    return np.concatenate(out)


def garder_lisses(Pr, Bl):
    """Les premiers q de Pr (uint32) tels que q + 1 n'a que des facteurs premiers <= Bl.
    (Dans les cycles connus a deux premiers libres, 84 % des grands premiers ont q + 1 1000-lisse :
    sigma(q) = q + 1 doit partager beaucoup de facteurs avec sigma du reste.)"""
    petits = [int(p) for p in premiers_jusqua(Bl)]
    out = []
    for a in range(0, len(Pr), 1 << 22):
        m = Pr[a:a + (1 << 22)].astype(np.uint32) + np.uint32(1)
        for p in petits:
            p_ = np.uint32(p)
            while True:
                z = (m % p_) == 0
                if not z.any():
                    break
                m[z] //= p_
        out.append(Pr[a:a + (1 << 22)][m == 1])
    return np.concatenate(out)


def _chercher_etendu(g, G, Gv, S, h, Uu, Ss, Y, Emax, Evrai, st, cyc):
    """--etendu V : graines a1 = G*u0*p (u0 <= U, p cherche comme d'habitude) et a2 = G*v*q, ou q est un
    GRAND premier enumere (v <= V, U < v*q <= --etendu-max). Un premier libre de chaque cote : la forme
    des cycles tires des paires amicales (1276 des 5553 4-cycles du catalogue), hors de portee du run
    normal des que v*q > U. Les couples (u0, u2 <= U) du run normal ne sont PAS refaits ici.
    Les v pairs sont ecartes si G est impair (un cycle ne melange pas les parites)."""
    U, P0, V, X = P["U"], P["P0"], P["ext"], P["ext_max"]
    N = len(Uu)
    # domaine du noyau (voir _passe) avec bs <= bu + 2 : la borne de v*q est abaissee pour les gros G
    bu = min(31, (121 - int(S).bit_length()) // 3, (122 - int(Gv).bit_length()) // 3)
    X = min(X, (1 << bu) - 1)
    st["ext_vq_max"] = X
    if X <= U:
        st["hors_noyau"] = 1
        return g, cyc, st
    t0 = time.time()
    U64, S64 = Uu.astype(np.int64), Ss.astype(np.int64)
    vs = [v for v in range(1, V + 1) if gcd(v, Gv) == 1 and (Gv % 2 == 0 or v % 2 == 1)]
    sv = {v: int(SIG[v]) for v in vs} if SIG is not None and V < len(SIG) else \
        {v: int(sympy.divisor_sigma(v)) for v in vs}
    premiers_G = [q for q in G if q > 2]
    total = dict(st)
    cap = P.get("ext_tranche", 1 << 24)
    tu, ts, ty, m = [], [], [], 0

    def vider():
        nonlocal tu, ts, ty, m
        if not m:
            return True
        Ue, Se, Ye = np.concatenate(tu), np.concatenate(ts), np.concatenate(ty)
        tu, ts, ty, m = [], [], [], 0
        if int(Se.max()) >> (bu + 2):                      # sigma trop grand pour le noyau : ecarte (v tres abondant)
            ok_ = Se < (1 << (bu + 2))
            Ue, Se, Ye = Ue[ok_], Se[ok_], Ye[ok_]
        t = h - Ye
        l_ = np.searchsorted(Y, t - 1e-15, side="right")
        n_ = np.searchsorted(Y, t + t / P0 * 1.000001 + 1e-15, side="right") - l_
        k_ = np.nonzero(n_)[0]
        total["ext_elements"] = total.get("ext_elements", 0) + len(Ue)
        if not len(k_):
            return True
        Ue, Se = Ue[k_], Se[k_]                            # seuls les u2 qui ont des partenaires sont charges
        jx = (N + np.arange(len(k_))).astype(np.int32)
        st1 = dict(couples=0, p_testes=0, th2=0, premiers=0, incomplet=0)
        _passe(g, G, Gv, S, np.concatenate([U64, Ue]), np.concatenate([S64, Se]), jx, l_[k_].astype(np.int32),
               n_[k_].astype(np.int32), Emax, Evrai, st1, cyc, t0, noyau_seul=True)
        for k, x in st1.items():
            total[k] = total.get(k, 0) + x if k not in ("incomplet", "gpu", "noyau_cpu", "emax_hybride", "hors_noyau") \
                else max(total.get(k, 0), x)
        return not st1.get("incomplet") and not st1.get("hors_noyau")

    fini = True
    for v in vs:
        qa, qb = max(U // v, V, P0), X // v                # v*q > U, q > V >= v (premier avec v), q <= X/v
        if qb <= qa:
            continue
        # bornes en uint32 : avec un entier Python, numpy convertissait tout le tableau a chaque appel
        q = PREMIERS[PREMIERS.searchsorted(np.uint32(qa), side="right"):
                     PREMIERS.searchsorted(np.uint32(qb), side="right")]
        for pg in premiers_G:
            if qa < pg <= qb:
                q = q[q != pg]
        if not len(q):
            continue
        q = q.astype(np.int64)
        tu.append(q * v); ts.append((q + 1) * sv[v]); ty.append((v / sv[v]) * (q / (q + 1.0)))
        m += len(q)
        if m >= cap and not vider():
            fini = False
            break
    if fini and not vider():
        fini = False
    if not fini and not total.get("hors_noyau"):
        total["incomplet"] = 1
    if Evrai or True:
        cyc[:] = list({repr(c): c for c in cyc}.values())
    return g, cyc, total


def cycles_de(G, Gv, S, u0, s0, u2, s2, p):
    """Theoreme 1 complet (Python) pour les graines a1 = G*u0*p, a2 = G*u2 : les 4-cycles, verifies."""
    if os.environ.get("AG_TRACE"):                     # controle : graines (u0, u2, p) passees au theoreme 1
        with open(os.environ["AG_TRACE"], "a") as tr:
            tr.write(f"{Gv} {u0} {u2} {p}\n")
    a1, a2 = Gv * u0 * p, Gv * u2
    sa1, sa2 = S * s0 * (p + 1), S * s2
    f1 = dict(G); f1.update({int(q): e for q, e in sympy.factorint(u0).items()}); f1[p] = 1
    f2 = dict(G); f2.update({int(q): e for q, e in sympy.factorint(u2).items()})
    cs, _ = B.theoreme1(f1, f2, 10 ** 6, (a1, sa1, a2, sa2))
    return cs


def init(params):
    P.update(params)


def init_processus(params):
    """Processus de calcul : Ctrl-C est traite par le processus principal seul (qui les arrete ensuite) ;
    sans cela chaque processus affichait sa propre trace d'erreur a l'interruption."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    init(params)


def G_du_catalogue(path):
    """Parties communes G des 4-cycles de Borho « purs » (u1, u2 premiers avec G) de sociable.txt."""
    cyc, cur = [], []
    for l in open(path, encoding="utf-8", errors="replace"):
        t = l.split()
        nums = [x for x in t[:2] if x.isdigit() and len(x) >= 6]
        if nums:
            cur.append(int(nums[-1]))
        elif not l.strip() and cur:
            cyc.append(cur)
            cur = []
    if cur:
        cyc.append(cur)
    compte = {}
    for c in cyc:
        if len(c) != 4:
            continue
        for k in range(4):
            n1, n2, n3, n4 = c[k:] + c[:k]
            a1, a2 = gcd(n1, n2), gcd(n3, n4)
            r1, r2 = n2 // a1, n4 // a2
            if (gmpy2.is_prime(r1) and gmpy2.is_prime(r2) and not gmpy2.is_prime(n1 // a1)
                    and not gmpy2.is_prime(n3 // a2) and gcd(a1, r1) == 1 and gcd(a2, r2) == 1):
                Gv = gcd(a1, a2)
                if gcd(a1 // Gv, Gv) == 1 and gcd(a2 // Gv, Gv) == 1 and Gv < 10 ** 10:
                    compte[Gv] = compte.get(Gv, 0) + 1
                break
    return [fmt({int(p): e for p, e in sympy.factorint(Gv).items()}) for Gv, _ in
            sorted(compte.items(), key=lambda t: -t[1])]


def G_hors_catalogue(Gmax, deja):
    s = sigma_tab(Gmax)
    dv = {B.val(B.parse(g)) for g in deja}
    h = s[2:] / np.arange(2, Gmax + 1)
    c = np.nonzero((h >= 1.8) & (h < 2))[0]
    c = [int(g) + 2 for g in c[np.argsort(-h[c], kind="stable")] if int(g) + 2 not in dv]   # h decroissant
    return [fmt({int(p): e for p, e in sympy.factorint(g).items()}) for g in c]


def G_des_paires(fichiers, gmin, gmax):
    """[(G, nombre de paires)] : parties communes exactes (memes premiers, memes exposants) des paires."""
    import gzip
    compte = {}
    for path in fichiers:
        prev = None
        # lecture BRUTE : seule la factorisation (apres « = ») sert, et elle figure telle quelle dans tous les
        # formats (n=fact ; =fact et !ligne pour le format FZ). c2io.ouvrir_texte reconstruisait chaque nombre
        # n d'un fichier FZ (un produit de facteurs en Python par ligne) : c'etait l'essentiel du temps.
        fh = gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") else \
            open(path, encoding="utf-8", errors="replace")
        with fh:
            for l in fh:
                if l.startswith("!"):
                    l = l[1:]                          # FZ : ligne d'origine gardee telle quelle
                if "=" not in l:
                    continue
                if prev is None:
                    prev = l
                    continue
                j1 = {t[:-2] if t.endswith("^1") else t for t in prev.split("=", 1)[1].strip().replace(" ", "").split("*")}
                j2 = {t[:-2] if t.endswith("^1") else t for t in l.split("=", 1)[1].strip().replace(" ", "").split("*")}
                prev = None
                k = "*".join(sorted(j1 & j2, key=lambda t: int(t.split("^")[0])))
                if k:
                    compte[k] = compte.get(k, 0) + 1
        print(f"  {path} lu : {len(compte):,} parties communes distinctes", flush=True)
    deja = set()
    if os.path.exists("G_catalogue.txt"):
        deja = {B.val(B.parse(l.split()[0])) for l in open("G_catalogue.txt") if l.strip()}
    out = []
    for k, n in compte.items():
        f = B.parse(k)
        v = B.val(f)
        if not gmin <= v <= gmax or v in deja:
            continue
        if not all(gmpy2.is_prime(p) for p in f):
            continue
        if 1.5 <= B.sig(f) / v < 2:
            out.append((fmt(f), n))
    out.sort(key=lambda t: -t[1])
    return out


def main():
    global SIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--G")
    ap.add_argument("--liste")
    ap.add_argument("--depuis-catalogue")
    ap.add_argument("--hors-catalogue", type=float)
    ap.add_argument("--depuis-paires", nargs="+")
    ap.add_argument("--gmin", type=float, default=1e5)
    ap.add_argument("--gmax", type=float, default=1e30)
    ap.add_argument("--debut", type=int, default=0)
    ap.add_argument("--nombre", type=int, default=0)
    ap.add_argument("--u", type=float, default=1e5)
    ap.add_argument("--p0", type=float, default=1e5)
    ap.add_argument("--K", type=float, default=1e7)
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--maxp", type=int, default=2000, help="valeurs de p essayees par couple au-dessus du plus petit "
                    "possible (les cycles connus ont p tout pres de ce minimum ; 20 suffit presque toujours)")
    ap.add_argument("--u-deja", type=float, default=0, metavar="U0",
                    help="ne refait pas les couples (u0, u2) tous deux <= U0, deja faits par un run a --u U0 (memes "
                         "--p0, --K, --maxp) : en passant de 3e7 a 6e7, environ un quart du travail en moins")
    ap.add_argument("--etendu", type=int, default=0, metavar="V",
                    help="graines a deux premiers libres (un par cote) : a1 = G*u0*p, u0 <= --u, et a2 = G*v*q "
                         "avec v <= V et q grand premier, --u < v*q <= --etendu-max. Ne refait pas le run normal "
                         "(couples u0, u2 <= --u) ; avancement note a part dans .faits. Noyau seulement (--gpu)")
    ap.add_argument("--etendu-lisse", type=int, default=0, metavar="B",
                    help="avec --etendu : seulement les premiers q tels que q+1 n'a que des facteurs <= B "
                         "(ex. 1000 : 25 fois moins de premiers, 95 %% des cycles connus de cette forme ; "
                         "permet un --etendu V bien plus grand). 0 = tous les premiers")
    ap.add_argument("--etendu-premiers", default="", metavar="FICHIER",
                    help="avec --etendu : seulement les premiers q de ce fichier (un par ligne), par exemple "
                         "les grands premiers des graines de cycles connus")
    ap.add_argument("--etendu-max", type=float, default=2 ** 31 - 1,
                    help="borne de v*q (defaut et maximum du noyau : 2^31 - 1)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--recyclage", type=int, default=10, metavar="N",
                    help="chaque processus est remplace apres N G (memoire rendue au systeme ; avec --gpu, chaque "
                         "remplacement recharge CUDA) ; a augmenter si les G sont rapides (ex. 100)")
    ap.add_argument("--out", default="apparie_cycles.txt")
    ap.add_argument("--sans-c", action="store_true", help="tout en Python (verification, plus lent)")
    ap.add_argument("--connus", nargs="*", default=[])
    ap.add_argument("--refaire", action="store_true", help="refaire aussi les G deja notes dans <out>.faits")
    ap.add_argument("--impairs", action="store_true", help="ecarter les G pairs")
    ap.add_argument("--hmin", type=float, default=0, metavar="H",
                    help="ecarter les G dont l'abondance sigma(G)/G est < H (ex. 1.85)")
    ap.add_argument("--exclure", nargs="+", default=[], metavar="FICHIER",
                    help="ecarter les G presents dans ces fichiers : listes de G (premier mot de chaque ligne) ou "
                         "fichiers .faits (G apres la tabulation, QUELS QUE SOIENT les parametres du run)")
    ap.add_argument("--gpu", nargs="?", const="auto", default=None,
                    help="boucle des couples sur GPU (libapparie_gpu.so ; libapparie_cpu.so a defaut) ; "
                         "chemin de la bibliotheque en option. Conseille avec --jobs 1 ou 2")
    ap.add_argument("--sans-noyau", action="store_true",
                    help="ne pas utiliser la boucle rapide libapparie_cpu.so (utilisee par defaut sans --gpu) : "
                         "chemin C habituel, pour controle")
    args = ap.parse_args()
    if args.G:
        Gs = [args.G]
    elif args.liste:
        Gs = [l.split()[0] for l in open(args.liste) if l.strip()]
    elif args.depuis_catalogue:
        Gs = G_du_catalogue(args.depuis_catalogue)
        open("G_catalogue.txt", "w").write("".join(g + "\n" for g in Gs))
        print(f"{len(Gs)} parties communes tirees de {args.depuis_catalogue} -> G_catalogue.txt")
    elif args.hors_catalogue:
        if not os.path.exists("G_catalogue.txt"):
            sys.exit("lancer d'abord --depuis-catalogue sociable.txt (cree G_catalogue.txt)")
        Gs = G_hors_catalogue(int(args.hors_catalogue), [l.split()[0] for l in open("G_catalogue.txt")])
        open("G_hors_catalogue.txt", "w").write("".join(g + "\n" for g in Gs))
        print(f"{len(Gs)} parties communes hors catalogue (G <= {args.hors_catalogue:g}, 1.8 <= h < 2) -> G_hors_catalogue.txt")
    elif args.depuis_paires:
        Gs = G_des_paires(args.depuis_paires, args.gmin, args.gmax)
        open("G_paires.txt", "w").write("".join(f"{g} {n}\n" for g, n in Gs))
        print(f"{len(Gs)} parties communes tirees des paires (G dans [{args.gmin:g}, {args.gmax:g}], 1.5 <= h < 2) -> G_paires.txt")
        Gs = [g for g, _ in Gs]
    else:
        sys.exit("donner --G, --liste, --depuis-catalogue, --hors-catalogue ou --depuis-paires")
    if args.impairs or args.hmin or args.exclure:
        # filtres appliques AVANT --debut / --nombre (qui comptent donc dans la liste filtree) ; comparaison
        # par valeur de G, insensible a l'ecriture (« 3^2*5 » et « 5*3^2 » sont le meme G)
        exclus = set()
        # fichiers ecrits par CE run (liste des G tiree de la source) : jamais pris comme exclusion, sinon
        # tous les G s'excluraient eux-memes (ex. --depuis-paires ... --exclure G_*.txt, qui contient G_paires.txt)
        ecrits = {os.path.abspath(n) for n, oui in (("G_paires.txt", args.depuis_paires),
                                                     ("G_catalogue.txt", args.depuis_catalogue),
                                                     ("G_hors_catalogue.txt", args.hors_catalogue)) if oui}
        if args.liste:
            ecrits.add(os.path.abspath(args.liste))
        for f in args.exclure:
            if os.path.abspath(f) in ecrits:
                print(f"[i] --exclure : {f} ignore (c'est la liste de G de ce run)", flush=True)
                continue
            for l in open(f, encoding="utf-8", errors="replace"):
                l = l.strip()
                if not l or l.startswith("#"):
                    continue
                g = l.split("\t", 1)[1] if "\t" in l else l.split()[0]
                try:
                    exclus.add(B.val(B.parse(g)))
                except ValueError:
                    pass                               # ligne qui n'est pas un G : ignoree
        n0, garde, ec = len(Gs), [], dict(pair=0, abondance=0, exclu=0)
        for g in Gs:
            f = B.parse(g)
            if args.impairs and 2 in f:
                ec["pair"] += 1
            elif args.hmin and B.sig(f) < args.hmin * B.val(f):
                ec["abondance"] += 1
            elif B.val(f) in exclus:
                ec["exclu"] += 1
            else:
                garde.append(g)
        Gs = garde
        print(f"filtres : {n0} G -> {len(Gs)} gardes (ecartes : {ec['pair']} pairs, {ec['abondance']} sous "
              f"h = {args.hmin:g}, {ec['exclu']} deja dans les fichiers exclus)", flush=True)
    Gs = Gs[args.debut:args.debut + args.nombre] if args.nombre else Gs[args.debut:]
    vus_g, uniques = set(), []                             # un G en double n'est fait qu'une fois, meme ecrit
    for g in Gs:                                           # autrement (« 3^2*5*19 » et « 19*3^2*5 »)
        try:
            v = B.val(B.parse(g))
        except ValueError:
            v = g
        if v not in vus_g:
            vus_g.add(v)
            uniques.append(g)
    Gs = uniques
    # reprise : G termines avec les memes parametres (une ligne « parametres <tab> G » par G)
    f_faits = args.out + ".faits"
    cle = f"u={int(args.u)},p0={int(args.p0)},K={args.K:g},maxp={args.maxp}"
    cles = [cle]                                       # un G fait par un run complet au meme u est fait aussi
    if args.u_deja:
        cle += f",deja={int(args.u_deja)}"
        cles.append(cle)
    if args.etendu:                                    # run a part : n'interfere pas avec l'avancement normal
        cle += f",etendu={args.etendu},vq={int(args.etendu_max)}"
        if args.etendu_lisse:
            cle += f",lisse={args.etendu_lisse}"
        if args.etendu_premiers:
            cle += f",premiers={os.path.basename(args.etendu_premiers)}"
    if os.path.exists(f_faits) and not args.refaire and not args.G:
        faits = {l.rstrip("\n").split("\t", 1)[1] for l in open(f_faits, encoding="utf-8")
                 if l.split("\t", 1)[0] in (cles if args.u_deja else [cle])}
        avant = len(Gs)
        Gs = [g for g in Gs if g not in faits]
        if avant != len(Gs):
            print(f"reprise ({f_faits}, {cle}) : {avant - len(Gs)} G deja fait(s), saute(s) ; --refaire pour les relancer")
    connus = set()
    for c in args.connus:
        for l in open(c, encoding="utf-8", errors="replace"):
            if not l.lstrip().startswith("#"):
                for m in l.replace("=", " ").split()[:2]:
                    if m.isdigit() and len(m) >= 6:
                        connus.add(int(m))
    deja = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8", errors="replace"):
            m = l.split()
            if m and m[0].isdigit() and len(m[0]) >= 6:
                deja.add(int(m[0]))
    U = int(args.u)
    t0 = time.time()
    SIG = sigma_tab(U)
    tri()                                                  # une fois, avant le fork
    SIG = None                                             # plus utile : la table triee suffit
    print(f"{len(Gs)} partie(s) commune(s) ; u0, u2 <= {U:g}, p >= {args.p0:g}, D <= {args.K:g} G^2 ; "
          f"table sigma {time.time() - t0:.0f} s", flush=True)
    # boucle rapide des couples : --gpu, sinon libapparie_cpu.so par defaut (meme calcul que le chemin C,
    # ~4 fois plus rapide par coeur) ; --sans-noyau ou --sans-c : chemin C habituel (controle)
    mode = None if (args.sans_noyau or args.sans_c) else (args.gpu or "cpu")
    try:
        ncoeurs = len(os.sched_getaffinity(0))
    except AttributeError:
        ncoeurs = os.cpu_count() or 1
    params = dict(U=U, P0=int(args.p0), K=args.K, budget=args.budget, sans_c=args.sans_c, maxp=args.maxp,
                  gpu=mode, fils=max(1, ncoeurs // max(1, args.jobs)),
                  ext=args.etendu, ext_max=min(int(args.etendu_max), 2 ** 31 - 1), u_deja=int(args.u_deja))
    if args.etendu:
        global PREMIERS
        t_ = time.time()
        if args.etendu_premiers:
            qs = {int(l.split()[0]) for l in open(args.etendu_premiers) if l.strip() and l.split()[0].isdigit()}
            PREMIERS = np.array(sorted(q for q in qs if q <= params["ext_max"] and gmpy2.is_prime(q)), dtype=np.uint32)
        else:
            PREMIERS = premiers_jusqua(params["ext_max"])
        if args.etendu_lisse:                          # calcul long (minutes pour B = 1000) : garde sur disque
            cache = f"premiers_lisses_{args.etendu_lisse}_{params['ext_max']}" \
                    + (f"_{os.path.basename(args.etendu_premiers)}" if args.etendu_premiers else "") + ".npy"
            if os.path.exists(cache):
                PREMIERS = np.load(cache)
            else:
                PREMIERS = garder_lisses(PREMIERS, args.etendu_lisse)
                np.save(cache, PREMIERS)
        print(f"[i] --etendu {args.etendu} : {len(PREMIERS):,} premiers q <= {params['ext_max']:,}"
              + (f" avec q+1 {args.etendu_lisse}-lisse" if args.etendu_lisse else "")
              + (f" (liste {args.etendu_premiers})" if args.etendu_premiers else "") + f" ({time.time() - t_:.0f} s)", flush=True)
    P["sans_c"] = args.sans_c
    print("[i] module C : " + ("oui" if lib_c() else "non (Python pur, plus lent)"), flush=True)
    if mode:
        P["gpu"] = mode
        if mode == "cpu":
            compiler_cpu(verbeux=True)                     # une fois, avant le fork
        gp = lib_gpu()
        print("[i] boucle des couples : " + (os.path.basename(gp[1]) + ("" if gp[2] else f", {params['fils']} fil(s) "
                                             "par processus") if gp else "bibliotheque introuvable, module C seul"),
              flush=True)
        globals()["_GPU"] = None                           # rechargee dans chaque processus (CUDA apres fork)
    init(params)
    ncyc = nin = ninc = nerr = nfaits = 0
    f_inc = args.out + ".incomplets"
    deja_inc = {l.strip() for l in open(f_inc)} if os.path.exists(f_inc) else set()
    total = {}
    interrompu = False
    pool = get_context("fork").Pool(max(1, args.jobs), initializer=init_processus, initargs=(params,),
                                  maxtasksperchild=max(1, args.recyclage))
    try:
        for k, (g, cyc, st) in enumerate(pool.imap_unordered(chercher, Gs), 1):
            nfaits = k
            message = st.pop("_message", None)
            for x, v in st.items():
                total[x] = total.get(x, 0) + v
            if st.get("erreur"):
                nerr += 1
                print(f"\n  [!] G = {g} : {message} -> {args.out}.erreurs", flush=True)
                with open(args.out + ".erreurs", "a", encoding="utf-8") as fe:
                    fe.write(f"{g}\t{message}\n")
            elif not st.get("incomplet") and not st.get("emax_borne"):
                with open(f_faits, "a", encoding="utf-8") as ff:     # G entierement explore avec ces parametres
                    ff.write(f"{cle}\t{g}\n")
            if st.get("incomplet"):
                ninc += 1
                if g not in deja_inc:                      # pas de doublon d'un run a l'autre
                    deja_inc.add(g)
                    with open(f_inc, "a") as fi:
                        fi.write(g + "\n")
            for c in cyc:
                ns = [B.val(f) for f in c]
                if min(ns) in deja:
                    continue
                inedit = bool(connus) and not any(x in connus for x in ns)
                tag = " [NOUVEAU ?]" if inedit else (" [deja connu]" if connus else "")
                if B.ecrire(c, f"apparie : G = {g}{tag}", args.out, deja):
                    ncyc += 1
                    nin += inedit
            if k % 20 == 0 or cyc:
                print(f"\r  {k}/{len(Gs)} G, {ncyc} cycle(s)" + (f" dont {nin} absent(s) des listes" if connus else "")
                      + (f", {ninc} G coupe(s) par le budget" if ninc else "") + f", {time.time() - t0:.0f} s   ",
                      end="", flush=True)
        pool.close()
    except KeyboardInterrupt:
        interrompu = True
    finally:
        for _ in range(3):                                 # un second Ctrl-C pendant l'arret ne casse rien
            try:
                pool.terminate()
                pool.join()
                break
            except KeyboardInterrupt:
                interrompu = True
    print(f"\n{'interrompu' if interrompu else 'termine'} : {nfaits if interrompu else len(Gs)} G"
          + (f" sur {len(Gs)} (relancer la meme commande reprend ; avancement dans {f_faits})" if interrompu else "")
          + f", {ncyc} cycle(s) nouveaux dans {args.out}"
          + (f", dont {nin} ABSENT(S) de {' '.join(args.connus)}" if connus else "")
          + (f" ; {ninc} G coupe(s) par le budget -> {args.out}.incomplets" if ninc else "")
          + (f" ; {nerr} G en erreur -> {args.out}.erreurs" if nerr else "")
          + f" ; {time.time() - t0:.0f} s")
    print("statistiques :", total)


if __name__ == "__main__":
    main()
