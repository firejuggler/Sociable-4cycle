"""
breed4.py -- mother -> daughter breeding of aliquot 4-cycles (te Riele transposed to cycles).

Daughter shape (rotation chosen so that member 0 carries the two new primes):
    n0 = A0*p*q,  n1 = A1*r1,  n2 = A2*r2,  n3 = A3*r3,   p,q,r_k new primes, coprime to their A.
Each A_k is taken from the mother member m_k: A_k = m_k ("full", new prime multiplied in, te Riele's
'a*u*q') or A_k = m_k / (one exponent-1 prime) ("stripped"); A0 may also be m_0/(two primes).
With t(A)=sigma(A)-A:  n1 = sigma(A0)(p+1)(q+1) - A0 pq ;  n_{k+1} = (t(A_k)/A_k) n_k + sigma(A_k).
Composing k=1..3:  n0 = lam n1 + mu  =>  alpha pq - beta(p+q) = gamma  with
    alpha = A0 - lam t(A0), beta = lam sigma(A0), gamma = beta + mu,
    (alpha p - beta)(alpha q - beta) = alpha gamma + beta^2 = N   (integers after scaling).
Enumerate the (finitely many) divisor pairs of N; test primality and integrality of r_k.
"""
import sys, itertools, time
from fractions import Fraction as F
from math import lcm, gcd, prod
import gmpy2
from sympy import factorint, pollard_rho, divisor_sigma
isp = gmpy2.is_prime

def parsef(s):
    f = {}
    for t in s.split('.'):
        p, _, e = t.partition('E'); f[int(p)] = f.get(int(p), 0) + (int(e) if e else 1)
    return f
def val(f): return prod(p**e for p, e in f.items())
def sigf(f): return prod((p**(e+1)-1)//(p-1) for p, e in f.items())

from sympy import primerange
SMALLP = list(primerange(2, 30000))
def factor_bounded(N, steps=4000):
    """trial division + short Pollard rho. Returns (f, rest): f = prime factors found,
    rest = list of composite cofactors that resisted (empty -> complete)."""
    f = {}
    for p in SMALLP:
        if p * p > N: break
        if N % p == 0:
            e = 0
            while N % p == 0: N //= p; e += 1
            f[p] = e
    stack = [N] if N > 1 else []; rest = []
    while stack:
        x = stack.pop()
        if isp(x): f[x] = f.get(x, 0) + 1; continue
        if gmpy2.is_square(x):
            r = int(gmpy2.isqrt(x)); stack += [r, r]; continue
        if x > 10**36: rest.append(x); continue
        try:
            d = pollard_rho(x, retries=2, max_steps=steps)
        except Exception:
            d = None
        if not d: rest.append(x); continue
        stack += [d, x // d]
    return f, rest

def divisors(f):
    res = [1]
    for p, e in f.items(): res = [d * p**i for d in res for i in range(e + 1)]
    return res

def solve(Af, maxdiv=3000000, maxdig=70):
    """Af = 4 factor dicts. Returns (cycles, status)."""
    A = [val(f) for f in Af]; S = [sigf(f) for f in Af]
    lam, mu = F(1), F(0)
    for k in (1, 2, 3):
        a = F(S[k] - A[k], A[k]); lam, mu = a * lam, a * mu + S[k]
    alpha = A[0] - lam * (S[0] - A[0]); beta = lam * S[0]; gamma = beta + mu
    D = lcm(alpha.denominator, beta.denominator, gamma.denominator)
    al, be, ga = int(alpha * D), int(beta * D), int(gamma * D)
    if al == 0: return [], 'degenerate'
    N = al * ga + be * be
    if N == 0: return [], 'N=0'
    if len(str(abs(N))) > maxdig:          # too big for rho here: strip small primes, log for outside factoring
        f, M = {}, abs(N)
        for p in SMALLP:
            if M % p == 0:
                e = 0
                while M % p == 0: M //= p; e += 1
                f[p] = e
        if M == 1 or isp(M):
            if M > 1: f[M] = 1
            return enumerate_pq(Af, A, S, al, be, N, f, maxdiv)
        solve.last = dict(N=N, al=al, be=be, known=f, rest=[M] if M > 1 else [],
                          A=[{int(p): e for p, e in g.items()} for g in Af])
        return [], 'unfactored' if M > 1 else 'ok'
    fN, rest = factor_bounded(abs(N))
    if rest:
        solve.last = dict(N=N, al=al, be=be, known=fN, rest=rest,
                          A=[{int(p): e for p, e in f.items()} for f in Af])
        return [], 'unfactored'
    return enumerate_pq(Af, A, S, al, be, N, fN, maxdiv)

def enumerate_pq(Af, A, S, al, be, N, fN, maxdiv=200000):
    if prod(e + 1 for e in fN.values()) > maxdiv: return [], 'toomany'
    out = []
    for d in divisors(fN):
        for sd in (d, -d):
            e = N // sd
            if (sd + be) % al or (e + be) % al: continue
            p = (sd + be) // al; q = (e + be) // al
            if p < 2 or q <= p or A[0] % p == 0 or A[0] % q == 0: continue
            if not (isp(p) and isp(q)): continue
            n = [A[0] * p * q]; x = S[0] * (p + 1) * (q + 1) - n[0]; ok = True; rs = [(p, q)]
            for k in (1, 2, 3):
                if x % A[k]: ok = False; break
                r = x // A[k]
                if r < 2 or not isp(r) or A[k] % r == 0: ok = False; break
                n.append(x); rs.append((r,)); x = (S[k] - A[k]) * r + S[k]
            if ok and x == n[0] and len(set(n)) == 4:
                facs = [dict(Af[k]) for k in range(4)]
                for k in range(4):
                    for pr in rs[k]: facs[k][int(pr)] = facs[k].get(int(pr), 0) + 1
                out.append(([int(v) for v in n], facs))
    return out, 'ok'

def options(f, double):
    o = [dict(f)]
    ones = sorted(p for p, e in f.items() if e == 1)[-4:]   # cap: 4 largest exponent-1 primes
    for p in ones:
        g = dict(f); del g[p]; o.append(g)
    if double:
        for p, q in itertools.combinations(ones, 2):
            g = dict(f); del g[p]; del g[q]; o.append(g)
    return o

def verify_sympy(n):
    return all(int(divisor_sigma(n[i])) - n[i] == n[(i + 1) % 4] for i in range(4))

def canon(c):
    i = c.index(min(c)); return tuple(c[i:] + c[:i])

def borho_shape(facs):
    """facs: list of 4 factor dicts (cyclic). Borho Th.1 shape: some rotation with
    n1=a1 p q, n2=a1 r, n3=a2 p' q', n4=a2 r' (p,q,r,... primes not dividing a_i, p!=q)."""
    for rot in range(4):
        m = facs[rot:] + facs[:rot]; good = True
        for x, y in ((m[0], m[1]), (m[2], m[3])):
            a = {p: min(e, y.get(p, 0)) for p, e in x.items() if p in y}
            u = {p: e - a.get(p, 0) for p, e in x.items() if e - a.get(p, 0) > 0}
            v = {p: e - a.get(p, 0) for p, e in y.items() if e - a.get(p, 0) > 0}
            if not (len(u) == 2 and all(e == 1 for e in u.values()) and all(p not in a for p in u)
                    and len(v) == 1 and list(v.values()) == [1] and all(p not in a for p in v)):
                good = False; break
        if good: return True
    return False
