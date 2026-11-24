"""Traitement des candidats de la recherche asymetrique (meme logique que la fin de search3.py) :
crible local (rho), factorisation de N = D^2 + C*E par PARI, enumeration des diviseurs, verification exacte.
Fonctions de module pour multiprocessing : init() une fois par processus, puis traiter(lot)."""
import numpy as np
from sympy import primefactors
from resid import icoeffs, rho
from core import chain_ok

_G = _sG = None
_UU = _SS = None
_pari = None
_PSEC = 2
_pfc = {}
SMALL = (2, 3, 5, 7, 11, 13)


def init(G, sG, UU, SS, psec):
    global _G, _sG, _UU, _SS, _pari, _PSEC
    _G, _sG, _UU, _SS, _PSEC = G, sG, UU, SS, psec
    import cypari2
    _pari = cypari2.Pari()
    _pari.allocatemem(2 * 10**8, silent=True)


def _pfs(x):
    if x not in _pfc:
        if len(_pfc) > 200000:
            _pfc.clear()
        _pfc[x] = [int(l) for l in primefactors(x)]   # int : sympy + gmpy2 renvoie des mpz, illisibles par numba
    return _pfc[x]


def _vv(x, l):
    v = 0
    while x % l == 0:
        x //= l; v += 1
    return v


def _score(s, u, C, D, E, maxM=8192):
    ls = set(SMALL)
    for x in u:
        ls.update(_pfs(x))
    ls.update(_pfs(_G))
    r = 1.0; sa = np.array(s, np.int64); ua = np.array(u, np.int64)
    for l in sorted(ls, key=lambda l: (l == 2, l)):
        v = sum(_vv(_G * x, l) for x in u)
        K = max(v + 1, 2 if l < 50 else 1)
        while l ** K > maxM and K > 1:
            K -= 1
        if l ** K > maxM:
            r *= float(l) ** (-sum(_vv(x, l) for x in u[1:])); continue
        M = l ** K
        a, b = rho(int(l), int(K), int(C % M), int(D % M), int(E % M), int(_sG), int(_G), sa, ua)
        if a == 0:
            return 0.0
        r *= a / b
    return r


def _un(row, alarm, AlarmInterrupt, cancel_alarm, cycles, durs):
    """Un candidat : 0 = rejete par le crible, 1 = N factorise, 2 = N trop dur pour PARI."""
    ii = [int(k) for k in row[:4]]
    u = [int(_UU[k]) for k in ii]; s = [int(_SS[k]) for k in ii]
    C, D, E = (int(x) for x in icoeffs(_sG, _G, s, u))
    if _score(s, u, C, D, E) == 0:
        return 0
    N = D * D + C * E
    if N <= 0:
        return 0
    try:
        alarm(_PSEC); f = _pari.factor(N); cancel_alarm()
    except AlarmInterrupt:
        durs.append(dict(G=_G, u=u, C=C, D=D, E=E, N=str(N), Nbits=N.bit_length()))
        return 2
    ps = [int(x) for x in f[0]]; es = [int(x) for x in f[1]]
    divs = [1]
    for pr, e in zip(ps, es):
        divs = [x * pr ** j for x in divs for j in range(e + 1)]
    for s1 in divs:
        s2 = N // s1
        if s1 > s2 or (s1 + D) % C or (s2 + D) % C:
            continue
        p = (s1 + D) // C; q = (s2 + D) // C
        r = chain_ok(_G, u, p, q)
        if r:
            cycles.append(dict(G=_G, u=u, p=int(p), q=int(q), cycle=[int(x) for x in r], C=C))
    return 1


def traiter(lot):
    """lot : liste de (i1, i2, i3, i4, C).  Renvoie (nzero, nfacto, cycles, durs).
    Un candidat qui leve une erreur est mis dans durs avec la cle 'erreur' au lieu d'arreter tout le calcul."""
    from cysignals.alarm import alarm, AlarmInterrupt, cancel_alarm
    nzero = nfac = 0; cycles = []; durs = []
    for row in lot:
        try:
            r = _un(row, alarm, AlarmInterrupt, cancel_alarm, cycles, durs)
        except Exception as ex:                     # noqa: BLE001 -- on note et on continue
            try:
                cancel_alarm()
            except Exception:
                pass
            durs.append(dict(G=_G, erreur=repr(ex)[:300], u=[int(_UU[int(k)]) for k in row[:4]]))
            continue
        if r == 0:
            nzero += 1
        elif r == 1:
            nfac += 1
    return nzero, nfac, cycles, durs
