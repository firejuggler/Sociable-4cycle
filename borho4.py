"""
borho4.py — construction de 4-cycles aliquotes (Borho 1969 ; Blankenagel, Borho, vom Stein 2003)
==================================================================================================
Theoreme 1 (Math. Comp. 72 (2003) 2071-2076), avec tau(n) = sigma(n) - n :
  a1 != a2, D := a1*a2 - tau(a1)*tau(a2). Pour toute factorisation d1*d2 = a1*a2 :
     p_ij := ( tau(a_{i+1}) sigma(a_i) + d_j sigma(a_{i+1}) ) / D      (a_3 := a_1)
     r_i  := tau(a_i p_i1 p_i2) / a_i
  Si les six nombres sont premiers, p_ij ne divise pas a_i, r_i ne divise pas a_i et
  p_i1 != p_i2, alors  n1 = a1 p11 p12,  n2 = a1 r1,  n3 = a2 p21 p22,  n4 = a2 r2
  est un 4-cycle aliquote.
Theoreme 2 (condition necessaire) : (a1 - a2) * f * g = 0 mod D, f = pgcd(sigma(a1), sigma(a2)),
  g = pgcd(a1, a2). Teste avant d'enumerer les d1.

Graines (section 7 de l'article : a1, a2 choisis comme « breeders » de paires amicales) :
  pour une paire n1 = G*m1, n2 = G*m2 (G = partie commune exacte), on prend
  a1 = G * (m1 sans k1 de ses premiers), a2 = G * (m2 sans k2 de ses premiers), 1 <= k <= --retire.
  Pour une paire, (sigma(n1)/n1 - 1)(sigma(n2)/n2 - 1) = 1 exactement : D(n1, n2) = 0, et les
  « breeders » voisins ont un D petit, ce qui donne une chance aux p_ij d'etre entiers.

Usage :
  python3 borho4.py --test                                   (exemples de l'article)
  python3 borho4.py --a1 "3^3*5*7" --a2 "3^3*5*11"
  python3 borho4.py --paires c2_22.txt c2_23.txt ... [--retire 1] [--jobs 8] [--max-div 1e6] [--out borho4_cycles.txt]
Chaque cycle trouve est re-verifie (sigma recalcule) avant d'etre ecrit.
Reprise : l'avancement (par fichier, pour un --retire et un --max-div donnes) est enregistre dans
--reprise (borho4_reprise.json) toutes les 10 000 paires et a l'interruption (Ctrl-C) ; relancer la
meme commande reprend la ou on s'etait arrete, sans re-ecrire les cycles deja dans --out.
"""
import c2io                     # lecture des c2_* : .txt, .gz et .gz « factorisation seule »
import argparse
import gzip
import io
import re
import sys
import threading
import time
from itertools import combinations
from math import gcd, prod

import gmpy2

is_prime = gmpy2.is_prime


def parse(s):
    f = {}
    for t in s.replace(" ", "").split("*"):
        p, _, e = t.partition("^")
        f[int(p)] = f.get(int(p), 0) + (int(e) if e else 1)
    return f


def val(f):
    return prod(p ** e for p, e in f.items())


def sig(f):
    return prod((p ** (e + 1) - 1) // (p - 1) for p, e in f.items())


def fmt(f):
    return "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))


def divisors(f):
    ds = [1]
    for p, e in f.items():
        ds = [d * p ** i for d in ds for i in range(e + 1)]
    return ds


def _congruence(a, b, D):
    """Solutions de a*x = b (mod D) : (c, M) avec x = c (mod M), ou None."""
    g = gcd(a, D)
    if b % g:
        return None
    M = D // g
    return ((b // g) * pow(a // g, -1, M) % M if M > 1 else 0), M


def _crt(c1, m1, c2, m2):
    g = gcd(m1, m2)
    if (c2 - c1) % g:
        return None
    l = m1 // g * m2
    k = ((c2 - c1) // g * pow(m1 // g, -1, m2 // g)) % (m2 // g) if m2 // g > 1 else 0
    return (c1 + m1 * k) % l, l


MAX_ATTENDUS = 2                                      # meme valeur que dans borho4c.c


def diviseurs_congrus(Nf, c, L, max_cote, symetrique=False):
    """Diviseurs d de N = value(Nf) avec d = c (mod L), par « rencontre au milieu » :
    premiers de N divisant L a part (partie b, peu de combinaisons), les autres coupes en deux
    moities A, B ; pour chaque b et chaque x de A, les y de B voulus sont dans une seule classe
    modulo L/pgcd(b, L) (x inversible). Coute ~ |A| + |B| au lieu de |A|*|B|.
    Leve OverflowError si une moitie depasse max_cote diviseurs.
    symetrique=True : seulement les d dont le cofacteur N/d peut lui aussi etre = c (mod L) (theoreme1
    l'exige : d1 = d2 = c). La valuation de N/d en chaque premier p de L obeit a la meme regle que celle
    de d : on n'enumere que v dans [k, e - k], ou v = w avec e = 2w (meme regle que borho4c.c, pour que
    les deux abandonnent « trop de diviseurs » exactement dans les memes cas)."""
    mauvais = {p: e for p, e in Nf.items() if L % p == 0}
    bons = sorted(((p, e) for p, e in Nf.items() if L % p), key=lambda t: -(t[1] + 1))
    A, B, ta, tb = {}, {}, 1, 1
    for p, e in bons:                                  # equilibrage glouton des deux moities
        if ta <= tb:
            A[p] = e; ta *= e + 1
        else:
            B[p] = e; tb *= e + 1
    # Pour p | L (p^k || L), d1 = c (mod p^k) fixe la valuation v de d1 en p : v = v_p(c mod p^k)
    # si c n'est pas divisible par p^k, sinon v >= k. On n'enumere que ces valuations-la.
    permis = []
    gsym = 1
    for p, e in mauvais.items():
        k, pk = 0, 1
        while L % (pk * p) == 0:
            k += 1
            pk *= p
        cm = c % pk
        if cm == 0:
            vs = list(range(k, (e - k if symetrique else e) + 1))
        else:
            w = 0
            while cm % p == 0:
                cm //= p
                w += 1
            vs = [w] if (2 * w == e if symetrique else w <= e) else []
        if not vs:
            return
        permis.append([p ** v for v in vs])
        gsym *= p ** min(vs[0], k)
    if symetrique and c % gsym == 0:
        # pgcd(b, L) = gsym pour tout b permis. d1 = c et d2 = c (mod L) donnent pour x*y deux classes
        # modulo Mp = L / gsym, qui coincident ssi (c/g)^2 = N / g^2 (mod Mp), avec c/g inversible :
        # condition independante de b, testee une fois (comme congrus() de borho4c.c)
        Mp, cg = L // gsym, c // gsym
        if Mp > 1 and (gcd(cg, Mp) != 1 or (cg * cg - val(Nf) // (gsym * gsym)) % Mp):
            return
    nb = prod(len(v) for v in permis)
    if max(ta * nb, tb) > max_cote:
        raise OverflowError(max(ta * nb, tb))
    bs = [1]
    for vs in permis:
        bs = [b * q for b in bs for q in vs]
    # diviseurs attendus dans la classe : ta*tb/Mp pour chaque b ; si L est petit (D petit), presque
    # toutes les combinaisons passent et le travail devient ta*tb : abandon (comme le module C)
    attendus = sum(ta * tb / (L // gcd(b, L)) for b in bs if c % gcd(b, L) == 0)
    if attendus > MAX_ATTENDUS * max_cote:
        raise OverflowError(int(attendus))
    la, lb = divisors(A), divisors(B)
    inv_la = None                                      # inverses des x modulo L, calcules une fois
    tables = {}
    for b in bs:
        g = gcd(b, L)
        if c % g:
            continue
        Mp = L // g
        if Mp == 1:
            for x in la:
                for y in lb:
                    yield b * x * y
            continue
        z = (c // g) * pow(b // g, -1, Mp) % Mp
        if gcd(z, Mp) != 1:                            # x*y est inversible mod Mp : aucune solution
            continue
        t = tables.get(Mp)
        if t is None:
            t = {}
            for y in lb:
                t.setdefault(y % Mp, []).append(y)
            tables[Mp] = t
        if inv_la is None:
            inv_la = [pow(x, -1, L) for x in la]       # inverse mod L, donc aussi mod Mp (Mp | L)
        for x, ix in zip(la, inv_la):
            for y in t.get(z * ix % Mp, ()):
                yield b * x * y


def theoreme1(a1f, a2f, max_div=10**6, pre=None):
    """Tous les 4-cycles du theoreme 1 pour (a1, a2). Renvoie (liste de cycles, statut).
    pre = (a1, sigma(a1), a2, sigma(a2)) deja calcules (evite de les recalculer pour chaque couple)."""
    a1, s1, a2, s2 = pre if pre else (val(a1f), sig(a1f), val(a2f), sig(a2f))
    if a1 == a2:
        return [], "a1 = a2"
    t1, t2 = s1 - a1, s2 - a2
    D = a1 * a2 - t1 * t2
    if D <= 0:
        return [], "D <= 0"
    if ((a1 - a2) * gcd(s1, s2) * gcd(a1, a2)) % D:
        return [], "theoreme 2"
    # p11 et p21 premiers IMPAIRS : N11 = t2*s1 + d1*s2 = D (mod 2D) et N21 = t1*s2 + d1*s1 = D (mod 2D),
    # soit deux congruences sur d1 modulo 2D (deux fois plus selectives que l'integralite seule, mod D).
    # Le cas d'un premier egal a 2 (N = 2D exactement) ne donne qu'une valeur de d1 : ajoutee a part.
    N = a1 * a2
    D2 = 2 * D
    c1 = _congruence(s2, D - t2 * s1, D2)
    c3 = _congruence(s1, D - t1 * s2, D2)
    cl = _crt(*c1, *c3) if c1 and c3 else None
    extras = set()
    for num, den in ((D2 - t2 * s1, s2), (D2 - t1 * s2, s1)):   # p11 = 2 ou p21 = 2 (d1), p12 / p22 = 2 (d2)
        if num > 0 and num % den == 0 and N % (num // den) == 0:
            extras.update((num // den, N // (num // den)))
    # Filtre du couple : p12 et p22 impairs imposent les MEMES congruences a d2 (memes seconds
    # membres), donc d1 = d2 = c (mod L) et N = d1*d2 = c^2 (mod L). Si N != c^2 (mod L), aucune
    # solution a quatre premiers impairs : seuls les extras (un premier = 2) restent a tester.
    if cl is not None and (N - cl[0] * cl[0]) % cl[1]:
        cl = None
    if cl is None and not extras:
        return [], "aucun d1"
    out = []
    if cl is not None:
        prodf = dict(a1f)
        for p, e in a2f.items():
            prodf[p] = prodf.get(p, 0) + e
        try:
            cands = set(diviseurs_congrus(prodf, cl[0], cl[1], max_div, symetrique=True))
        except OverflowError as ex:
            return [], f"trop de diviseurs ({ex})"
        cands |= extras
    else:
        cands = extras
    A1, A2 = t2 * s1, t1 * s2                          # invariants de la boucle
    for d1 in cands:
        d2 = N // d1
        if d1 > d2:
            continue                                   # (d1, d2) et (d2, d1) : meme cycle
        if cl is not None and d1 not in extras and d2 % cl[1] != cl[0]:
            continue                                   # d2 hors classe : p12 ou p22 ne sera pas impair
        n11, n12 = A1 + d1 * s2, A1 + d2 * s2
        if n11 % D or n12 % D:
            continue
        n21, n22 = A2 + d1 * s1, A2 + d2 * s1
        if n21 % D or n22 % D:
            continue
        p11, p12, p21, p22 = n11 // D, n12 // D, n21 // D, n22 // D
        if p11 == p12 or p21 == p22 or min(p11, p12, p21, p22) < 2:
            continue
        if any(a % p == 0 for a, p in ((a1, p11), (a1, p12), (a2, p21), (a2, p22))):
            continue
        if not all(is_prime(p) for p in (p11, p12, p21, p22)):
            continue
        f1 = dict(a1f); f1[p11] = 1; f1[p12] = 1
        f3 = dict(a2f); f3[p21] = 1; f3[p22] = 1
        q1, q2 = sig(f1) - val(f1), sig(f3) - val(f3)             # tau(a_i p_i1 p_i2)
        if q1 % a1 or q2 % a2:
            continue
        r1, r2 = q1 // a1, q2 // a2
        if r1 < 2 or r2 < 2 or a1 % r1 == 0 or a2 % r2 == 0 or not (is_prime(r1) and is_prime(r2)):
            continue
        f2 = dict(a1f); f2[r1] = 1
        f4 = dict(a2f); f4[r2] = 1
        cyc = [f1, f2, f3, f4]
        ns = [val(f) for f in cyc]
        if len(set(ns)) == 4 and all(sig(cyc[i]) - ns[i] == ns[(i + 1) % 4] for i in range(4)):
            out.append(cyc)
    return out, "ok"


LIGNE = re.compile(r"^\s*(\d+)\s*=\s*([\d^*\s]+?)\s*$")


def ouvrir(path):
    """Fichier binaire en lecture, decompression transparente des .gz."""
    return c2io.ouvrir(path)                           # .txt, .gz classique ou .gz FZ


def _avancement(v):
    """Entree du fichier de reprise : {"n": paires faites, "off": octet de reprise} (ancien format : n)."""
    if isinstance(v, dict):
        return v.get("n", 0), v.get("off")
    return (v or 0), None


def lire_paires(paths, faits=None, avec_pos=False):
    """(fichier, rang, s1, s2[, position]) lus en flux dans plusieurs fichiers « n=factorisation »
    (deux lignes par paire). Reprise : on saute directement a l'octet enregistre (instantane) ;
    avec l'ancien format (nombre de paires seulement), les paires deja faites sont relues et sautees."""
    faits = faits or {}
    for path in paths:
        v = faits.get(path)
        if isinstance(v, dict) and v.get("fini"):
            continue                                   # fichier termine (ex. compresse) : rien a lire
        saut, off = _avancement(v)
        prev, i, pos = None, 0, 0
        attente = None                                 # derniere paire lue, emise avec un temps de retard
        with ouvrir(path) as fh:
            if off is not None:
                fh.seek(off)
                pos, i = off, saut
            for raw in fh:
                pos += len(raw)
                if b"=" not in raw:
                    continue
                mo = LIGNE.match(raw.decode("utf-8", "replace"))
                if not mo:
                    continue
                if prev is None:
                    prev = mo.group(2)
                else:
                    if i >= saut:
                        if attente:
                            yield attente
                        attente = (path, i, prev, mo.group(2), pos) if avec_pos else (path, i, prev, mo.group(2))
                    i += 1
                    prev = None
        if attente:
            # derniere paire du fichier : si rien d'autre ne suit (lignes vides...), sa position de
            # reprise est la FIN du fichier, pour que le fichier soit reconnu comme termine
            if avec_pos and prev is None:
                attente = attente[:4] + ((pos, True),)     # (position, fichier termine)
            yield attente
        if off is None and path in faits and i <= saut:
            # fichier entierement saute avec l'ancien format : on enregistre sa position finale,
            # pour que les reprises suivantes le sautent sans le relire
            faits[path] = {"n": i, "off": pos}


def _bases(f, m, retire):
    """Bases f sans k premiers de m (1 <= k <= retire) : (k, factorisation, valeur, sigma),
    valeur et sigma obtenues par division (une seule fois par base)."""
    n, s = val(f), sig(f)
    out = []
    for k in range(1, min(retire, len(m)) + 1):
        for S in combinations(sorted(m), k):
            q = prod(p ** m[p] for p in S)
            sq = prod((p ** (m[p] + 1) - 1) // (p - 1) for p in S)
            out.append((k, {p: e for p, e in f.items() if p not in S}, n // q, s // sq))
    return out


def bases_paire(f1, f2, retire):
    """Bases des deux membres (meme ordre que le module C)."""
    G = {p: e for p, e in f1.items() if f2.get(p) == e}
    m1 = {p: e for p, e in f1.items() if p not in G}
    m2 = {p: e for p, e in f2.items() if p not in G}
    return _bases(f1, m1, retire), _bases(f2, m2, retire)


def graines_paire(f1, f2, retire, total=0, sauf=0, max_graine=0):
    """(a1, a2, pre) pour une paire ; pre = (a1, sigma(a1), a2, sigma(a2)).
    max_graine > 0 : seulement les couples dont les deux graines ont au plus max_graine chiffres."""
    b1, b2 = bases_paire(f1, f2, retire)
    lim = 10 ** max_graine if max_graine > 0 else 0
    for k1, a1, v1, s1 in b1:
        if lim and v1 >= lim:
            continue
        for k2, a2, v2, s2 in b2:
            if lim and v2 >= lim:
                continue
            if total and k1 + k2 > total:
                continue
            if sauf and k1 <= sauf and k2 <= sauf:
                continue                               # couple deja traite par un run --retire sauf
            yield a1, a2, (v1, s1, v2, s2)


def ecrire(cyc, origine, out, deja):
    ns = [val(f) for f in cyc]
    i = ns.index(min(ns))
    cyc = cyc[i:] + cyc[:i]
    if min(ns) in deja:
        return False
    deja.add(min(ns))
    print(f"\n  *** 4-CYCLE ({len(str(min(ns)))} chiffres) <- {origine}", flush=True)
    for f in cyc:
        print(f"      {val(f)} = {fmt(f)}")
    if out:
        with open(out, "a", encoding="utf-8") as fo:
            fo.write(f"# 4-cycle Borho, {origine}\n")
            for f in cyc:
                fo.write(f"      {val(f)}  {'.'.join(f'{p}E{e}' if e > 1 else str(p) for p, e in sorted(f.items()))}\n")
            fo.write("\n")
    return True


def main():
    ap = argparse.ArgumentParser(description="4-cycles par le theoreme de Borho")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--a1")
    ap.add_argument("--a2")
    ap.add_argument("--paires", nargs="+", help="un ou plusieurs fichiers de paires (format n=factorisation)")
    ap.add_argument("--jobs", type=int, default=0, help="processus (defaut : tous les coeurs)")
    ap.add_argument("--retire", type=int, default=2, help="nb max de premiers retires de chaque membre")
    ap.add_argument("--retire-total", type=int, default=0,
                    help="nb max de premiers retires au total (ex. 3 avec --retire 2 : saute le cas (2,2), le plus "
                         "couteux) ; 0 = pas de limite")
    ap.add_argument("--deja-retire", type=int, default=0,
                    help="saute les couples deja traites par un run --retire N (ex. --retire 3 --deja-retire 2 : "
                         "seulement les couples ou 3 premiers sont retires d'au moins un membre)")
    ap.add_argument("--max-div", type=float, default=1e6)
    ap.add_argument("--max-graine", type=int, default=0, metavar="CHIFFRES",
                    help="seulement les couples dont les deux graines ont au plus CHIFFRES chiffres (ex. 25 : "
                         "passage rapide sur les grandes paires ; au-dela de ~25 chiffres les graines ne donnent "
                         "presque plus de cycles). 0 = pas de limite")
    ap.add_argument("--out", default="borho4_cycles.txt")
    ap.add_argument("--sans-c", action="store_true", help="ne pas utiliser le module C (Python pur)")
    ap.add_argument("--reprise", default="borho4_reprise.json",
                    help="fichier d'avancement (relancer la meme commande reprend ou on s'etait arrete)")
    args = ap.parse_args()
    deja = set()
    if args.test:
        ex = [("3^3*5*7", "3^3*5*11", [3**4 * 5 * 7 * 83 * 359 // 3]),          # article, exemple du theoreme 1
              ("2*7*223*5*11", "2*7*223*23", [2 * 7 * 223 * 5 * 11 * 3121 * 73589]),   # cycle 4)
              ("3^3*5^3*7*59", "3^3*5^3*89", []),                                  # cycle 5)
              ("2^3*13*3079", "2^3*19*293", [])]                                   # cycle 18)
        for a1, a2, attendu in ex:
            cyc, st = theoreme1(parse(a1), parse(a2), int(args.max_div))
            mins = [min(val(f) for f in c) for c in cyc]
            print(f"a1 = {a1}, a2 = {a2} : {len(cyc)} cycle(s) [{st}]", end="")
            print(f"  -> attendu retrouve : {all(any(val(c[0]) == x or x in [val(f) for f in c] for c in cyc) for x in attendu)}"
                  if attendu else "")
            for c in cyc:
                print("     ", " -> ".join(fmt(f) for f in c))
        return
    if args.a1 and args.a2:
        cyc, st = theoreme1(parse(args.a1), parse(args.a2), int(args.max_div))
        print(f"statut : {st} ; {len(cyc)} cycle(s)")
        for c in cyc:
            ecrire(c, f"a1 = {args.a1}, a2 = {args.a2}", args.out, deja)
        return
    if not args.paires:
        sys.exit("donner --test, --a1/--a2 ou --paires")
    import json
    import os
    from multiprocessing import Pool
    jobs = args.jobs or os.cpu_count() or 1
    paths = [os.path.abspath(p) for p in args.paires]
    if args.deja_retire and args.deja_retire >= args.retire:
        sys.exit(f"--deja-retire {args.deja_retire} >= --retire {args.retire} : aucun couple nouveau")
    cle = (f"retire={args.retire},max_div={int(args.max_div)}" + (f",total={args.retire_total}" if args.retire_total else "")
           + (f",sauf={args.deja_retire}" if args.deja_retire else "")
           + (f",max_graine={args.max_graine}" if args.max_graine else ""))
    etat = {}
    if os.path.exists(args.reprise):
        etat = json.load(open(args.reprise, encoding="utf-8"))
    faits = etat.setdefault(cle, {})
    deja_faits = sum(_avancement(faits.get(p))[0] for p in paths)
    if deja_faits:
        print(f"reprise ({args.reprise}, {cle}) : {deja_faits:,} paire(s) deja traitee(s), sautees", flush=True)
    if os.path.exists(args.out):                      # cycles deja ecrits : pas de doublon a la reprise
        for line in open(args.out, encoding="utf-8", errors="replace"):
            m = re.match(r"^\s*(\d{2,})\s+[0-9E.]+\s*$", line)
            if m:
                deja.add(int(m.group(1)))

    def sauver():
        # On relit le fichier et on ne remplace QUE la cle de ce run : deux runs (ex. --retire 2 et
        # --retire 3) peuvent partager le meme fichier de reprise sans effacer l'avancement de l'autre.
        disque = {}
        try:
            with open(args.reprise, encoding="utf-8") as fi:
                disque = json.load(fi)
        except (OSError, ValueError):
            disque = dict(etat)
        disque[cle] = faits
        tmp = f"{args.reprise}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fo:
            json.dump(disque, fo, indent=1)
        os.replace(tmp, args.reprise)

    t0 = time.time()
    npaires = ncouples = ncyc = 0
    courant = None                                     # fichier en cours (affiche a chaque changement)
    stats = {}
    # File bornee : Pool.imap lit les taches aussi vite qu'il peut (sans attendre les processus) ; quand le
    # traitement est rapide (--max-graine) ou qu'une paire bloque l'ordre, tout le fichier finissait en RAM.
    # Le generateur attend donc des que EN_VOL paires sont lues et pas encore rendues.
    EN_VOL = max(4096, 64 * 16 * jobs)
    places = threading.Semaphore(EN_VOL)
    arret = threading.Event()

    def generer():
        for p, i, s1, s2, pos in lire_paires(paths, faits, avec_pos=True):
            places.acquire()
            if arret.is_set():
                return
            yield (p, i, s1, s2, args.retire, int(args.max_div), (args.retire_total, args.deja_retire, args.max_graine), pos)
    taches = generer()
    global _C
    if args.sans_c:
        _C = False
    else:
        charger_c(verbeux=True)                        # compile/charge une fois, avant le fork des processus
    pool = Pool(jobs)
    try:
        # imap ordonne : l'avancement enregistre est exact (toutes les paires avant lui sont faites)
        for path, i, found, st, nc, src, pos in pool.imap(traiter_paire, taches, chunksize=64):
            places.release()
            if path != courant:
                courant = path
                print(f"\n  [fichier] {os.path.basename(path)}" + (f" (reprise a la paire {i + 1:,})" if i else ""),
                      flush=True)
            npaires += 1
            ncouples += nc
            pos, fin = pos if isinstance(pos, tuple) else (pos, False)
            faits[path] = {"n": i + 1, "off": pos}
            if fin and path.endswith(".gz"):
                faits[path]["fini"] = True             # .gz termine : saute sans decompression a la reprise
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
            for c, orig in found:
                ncyc += ecrire(c, f"{orig} (paire {src})", args.out, deja)
            for k in st:
                if k.startswith("SUSPECTE"):
                    with open("paires_suspectes.txt", "a", encoding="utf-8") as fs_:
                        fs_.write(f"{os.path.basename(path)}  paire n° {i + 1}  plus petit membre {src}  : {k}\n")
                    print(f"\n  [!] paire suspecte ({k}) : {os.path.basename(path)} n° {i + 1}, {src[:40]}...", flush=True)
            if npaires % 10000 == 0:
                sauver()
                print(f"\r  {os.path.basename(path)} : paire {i + 1:,} | total {npaires:,} paires, {ncouples:,} couples "
                      f"(a1, a2), {ncyc} cycle(s), {time.time() - t0:.0f} s   ", end="", flush=True)
    except KeyboardInterrupt:
        arret.set()                                    # debloque le generateur (sinon terminate attend)
        for _ in range(EN_VOL):
            places.release()
        pool.terminate()
        sauver()
        print(f"\n[!] interrompu : avancement enregistre dans {args.reprise}, relancer la meme commande")
        return
    pool.close()
    pool.join()
    sauver()
    print(f"\n{npaires:,} paires, {ncouples:,} couples (a1, a2) en {time.time() - t0:.0f} s ; "
          f"{ncyc} 4-cycle(s){' -> ' + args.out if ncyc else ''}")
    print("statuts :", ", ".join(f"{k} {v:,}" for k, v in sorted(stats.items(), key=lambda t: -t[1])))


def _rho(n, budget=1 << 21):
    """Un facteur non trivial de n compose (Pollard-Brent). ArithmeticError si rien n'est trouve en
    ~budget iterations par constante (facteurs jusqu'a ~13 chiffres) : sans cette borne, un compose de
    40 a 60 chiffres sans petit facteur bloquait le run indefiniment."""
    if n % 2 == 0:
        return 2
    for c in range(1, 4):
        y, r, q, g, x, ys = 2, 1, 1, 1, 2, 2
        while g == 1:
            if r > budget:
                raise ArithmeticError(f"rho : pas de facteur en {budget} iterations ({len(str(n))} chiffres)")
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(128, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += 128
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if g != n:
            return g
    raise ArithmeticError(f"rho : echec sur {n}")


def deplier(f, max_chiffres=60):
    """Remplace les facteurs composes d'une factorisation par leurs facteurs premiers.
    Renvoie (factorisation, nb de facteurs deplies) ; None si un compose est trop gros."""
    out, n_dep = {}, 0
    for p, e in f.items():
        if p >= 2 and is_prime(p):
            out[p] = out.get(p, 0) + e
            continue
        if p < 2 or len(str(p)) > max_chiffres:
            return None, 0
        n_dep += 1
        pile = [p]
        while pile:
            x = pile.pop()
            if is_prime(x):
                out[x] = out.get(x, 0) + e
                continue
            r = int(gmpy2.isqrt(x))
            try:
                d = r if r * r == x else _rho(x)
            except ArithmeticError:
                return None, 0                         # compose non factorisable vite : paire ecartee
            pile += [d, x // d]
    return out, n_dep


# --- module C (borho4c.c) : filtre rapide des couples, facultatif ------------------------------
_C = None          # None : pas encore charge ; False : indisponible (Python pur)


def charger_c(verbeux=False):
    """Charge libborho4c.so (compile borho4c.c s'il le faut, avec le GMP du systeme ou, a defaut,
    celui fourni avec gmpy2). Renvoie la fonction C, ou False (Python pur)."""
    global _C
    if _C is not None:
        return _C
    import ctypes
    import glob
    import os
    import subprocess
    _C = False
    ici = os.path.dirname(os.path.abspath(__file__))
    src, so = os.path.join(ici, "borho4c.c"), os.path.join(ici, "libborho4c.so")
    if not os.path.exists(src):
        if verbeux:
            print("[i] borho4c.c absent : calcul en Python pur", flush=True)
        return _C
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        base = ["gcc", "-O2", "-shared", "-fPIC", "-o", so + ".tmp", src]
        essais = [base + ["-lgmp", "-lm"]]
        gd = os.path.dirname(gmpy2.__file__)                # GMP embarque dans la roue pip de gmpy2
        for lib in glob.glob(os.path.join(gd + ".libs", "libgmp*.so*")) + glob.glob(os.path.join(gd, "libgmp*.so*")):
            essais.append(base + [f"-I{gd}", f"-L{os.path.dirname(lib)}", f"-l:{os.path.basename(lib)}",
                                  f"-Wl,-rpath,{os.path.dirname(lib)}", "-lm"])
        err = ""
        for cmd in essais:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                os.replace(so + ".tmp", so)
                break
            err = r.stderr.strip().splitlines()[-1:] or ["?"]
        else:
            if verbeux:
                print(f"[i] compilation de borho4c.c impossible ({err[0]}) : calcul en Python pur "
                      f"(pour l'acceleration : sudo apt install gcc libgmp-dev)", flush=True)
            return _C
    try:
        lib = ctypes.CDLL(so)
        f = lib.borho_paire
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long,
                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        g = getattr(lib, "borho_set_max_graine", None)
        if g is not None:
            g.restype = None
            g.argtypes = [ctypes.c_int]
        _C = (f, ctypes.c_int * 6, ctypes.c_int * 8192, g)
        if verbeux:
            print("[i] module C charge (libborho4c.so)", flush=True)
    except OSError as ex:
        if verbeux:
            print(f"[i] libborho4c.so inutilisable ({ex}) : calcul en Python pur", flush=True)
    return _C


def _paire_c(s1, s2, retire, max_div, total):
    """Crible C : tous les couples de la paire y sont cribles ; seuls ceux qui ont un candidat
    (quatre p_ij premiers) ou trop de diviseurs sont refaits ici avec theoreme1().
    Renvoie (found, st, nc), ou None si la paire doit etre traitee entierement en Python."""
    f, Stats, Out, g = _C
    maxg = total[2] if len(total) > 2 else 0
    if g is None:
        if maxg:
            return None                                # bibliotheque ancienne : chemin Python (filtre)
    else:
        g(maxg)
    stats, out = Stats(), Out()
    n = f(s1.encode(), s2.encode(), retire, total[0], total[1], max_div, stats, out, 4096)
    if n < 0:
        return None
    st = {}
    for k, v in zip(("a1 = a2", "D <= 0", "theoreme 2", "aucun d1", "ok"), stats[1:]):
        if v:
            st[k] = v
    found = []
    if n:
        b1, b2 = bases_paire(parse(s1), parse(s2), retire)
        for i in range(n):
            _, a1, v1, sa = b1[out[2 * i]]
            _, a2, v2, sb = b2[out[2 * i + 1]]
            cyc, s = theoreme1(a1, a2, max_div, (v1, sa, v2, sb))
            st[s] = st.get(s, 0) + 1
            for c in cyc:
                found.append((c, f"a1 = {fmt(a1)}, a2 = {fmt(a2)}"))
    return found, st, stats[0]


def traiter_paire(t):
    t0 = time.time()
    r = _traiter_paire(t)
    dt = time.time() - t0
    if dt > 60:
        print(f"\n  [lent] {dt:.0f} s pour la paire {t[2][:60]}... ({len(t[2].split('*'))} facteurs) : "
              f"{', '.join(f'{k} {v}' for k, v in r[3].items())}", flush=True)
    return r


def _traiter_paire(t):
    path, rang, s1, s2, retire, max_div, total, pos = t
    if not isinstance(total, tuple):
        total = (total, 0)                             # (retire-total, deja-retire)
    if _C is None:
        charger_c()
    if _C:
        try:
            r = _paire_c(s1, s2, retire, max_div, total)
        except Exception:
            r = None                                   # repli : chemin Python complet ci-dessous
        if r is not None:
            src = str(min(val(parse(s1)), val(parse(s2)))) if r[0] else ""   # utile seulement si cycle
            return (path, rang) + r + (src, pos)
    f1, f2 = parse(s1), parse(s2)
    found, st, nc = [], {}, 0
    src = str(min(val(f1), val(f2)))
    # la construction suppose des facteurs PREMIERS : un facteur compose (bloc ecrit d'un seul
    # tenant) est deplie ; s'il est trop gros pour etre factorise ici, la paire est ecartee
    if not all(p >= 2 and is_prime(p) for p in list(f1) + list(f2)):
        g1, d1 = deplier(f1)
        g2, d2 = deplier(f2)
        if g1 is None or g2 is None or val(g1) != val(f1) or val(g2) != val(f2):
            st["SUSPECTE : facteur non premier"] = 1
            return path, rang, found, st, nc, src, pos
        f1, f2 = g1, g2
        st["corrigee : facteur compose deplie"] = 1
    try:
        for a1, a2, (v1, s1, v2, s2) in graines_paire(f1, f2, retire, *total):
            # un seul sens : (a2, a1) donne le meme cycle que (a1, a2), demarre en n3 (theoreme symetrique)
            for x, y, pre in ((a1, a2, (v1, s1, v2, s2)),):
                nc += 1
                cyc, s = theoreme1(x, y, max_div, pre)
                st[s] = st.get(s, 0) + 1
                for c in cyc:
                    found.append((c, f"a1 = {fmt(x)}, a2 = {fmt(y)}"))
    except Exception as ex:                            # une paire anormale n'arrete jamais le run
        st[f"SUSPECTE : erreur {type(ex).__name__}"] = 1
    return path, rang, found, st, nc, src, pos


if __name__ == "__main__":
    main()
