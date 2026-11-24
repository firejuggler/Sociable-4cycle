# Recherche ASYMETRIQUE a deux grands premiers : trois cofacteurs petits (<= Us), un grand (dans [Ulo, Uhi]).
# Forme : n1 = G*u1*p*q, n2 = G*u2*P2, n3 = G*u3*P3, n4 = G*u4*P4 (meme algebre que search3.py / core.py).
# Appariement au milieu : moitie A = (petit, GRAND) parcourue a la volee, moitie B = (petit, petit) triee.
# usage : python3 search_asym.py G Us Ulo Uhi pos cap worker nworkers log NMIN tbudget [pari_sec] [eps]
#   pos = position (1..4) du grand cofacteur dans (u1, u2, u3, u4) ; u1 porte p*q.
#   Test (cycle n. 123) : python3 search_asym.py 1 600 3354000 3355000 2 4000000 0 1 t123.log 1e14 300
import sys, time, json, math, numpy as np, numba as nb
from math import gcd
from sympy import primefactors
G = int(sys.argv[1]); US = int(sys.argv[2]); ULO = int(sys.argv[3]); UHI = int(sys.argv[4]); POS = int(sys.argv[5])
CAP = int(sys.argv[6]); W = int(sys.argv[7]); NW = int(sys.argv[8]); OUT = sys.argv[9]; NMIN = float(sys.argv[10])
TBUD = float(sys.argv[11]); PSEC = int(sys.argv[12]) if len(sys.argv) > 12 else 2
EPS_USER = float(sys.argv[13]) if len(sys.argv) > 13 else None
assert 1 <= POS <= 4 and US < ULO <= UHI
import cypari2; pari = cypari2.Pari(); pari.allocatemem(10**9, silent=True)
from cysignals.alarm import alarm, AlarmInterrupt, cancel_alarm
from core import chain_ok
from resid import icoeffs, rho


def sig_range(a, b):
    # boucle sur tous les diviseurs d <= b : O(b log b), suffisant jusqu'a ~2e7
    s = np.zeros(b - a + 1, np.int64)
    for d in range(1, b + 1):
        lo = ((a + d - 1) // d) * d
        if lo <= b:
            s[lo - a::d] += d
    return s


sG = int(sig_range(G, G)[0]) if G > 1 else 1
X = []; Y = []; Z = []; Wd = []; UU = []; SS = []


def ajoute(us, ss):
    for u, s in zip(us, ss):
        u = int(u); s = int(s)
        if gcd(u, G) != 1:
            continue
        x = sG * s - G * u; y = G * u
        if x <= 0:
            continue
        g = gcd(x, y); z = sG * s; w = G * u; g2 = gcd(z, w)
        X.append(x // g); Y.append(y // g); Z.append(z // g2); Wd.append(w // g2); UU.append(u); SS.append(s)


ajoute(range(1, US + 1), sig_range(1, US))
NS = len(UU)
ajoute(range(ULO, UHI + 1), sig_range(ULO, UHI))
NB = len(UU) - NS
X = np.array(X, np.int64); Y = np.array(Y, np.int64); Z = np.array(Z, np.int64); UU = np.array(UU, np.int64)
n = len(UU); fv = X / Y; lf = np.log(fv); ly = np.log(Y.astype(np.float64)); f1p = fv + 1.0; GU = (G * UU).astype(np.float64)
U1MAX = UHI if POS == 1 else US
EPS = EPS_USER or min(0.06, float(np.max(Z / np.array(Wd))) * (G * U1MAX / (0.1 * NMIN)) ** (1 / 3) * 1.01)
LYmax = math.log(2.0 ** 63) + math.log(1.0 / EPS) - 1.0
# moitie B : (petit, petit), triee par log f
I, J = np.meshgrid(np.arange(NS), np.arange(NS), indexing='ij'); I = I.ravel().astype(np.int32); J = J.ravel().astype(np.int32)
SB = lf[I] + lf[J]; o = np.argsort(SB); Rs = SB[o].copy(); RI = I[o].copy(); RJ = J[o].copy()


@nb.njit(cache=True)
def gcd64(a, b):
    while b:
        a, b = b, a % b
    return a


@nb.njit(cache=True)
def run(POS, NS, NB, Rs, RI, RJ, X, Y, Z, ly, lo_eps, cmax, LYmax, W, NW, out, cap, fv, f1p, GU, nmin):
    cnt = 0; nq = 0; nbig = 0; tot = 0
    q = np.zeros(4, np.int64)
    nA = NS * NB
    for t in range(W, nA, NW):
        ib = NS + t // NS; isml = t % NS          # A = (grand, petit)
        # positions de la moitie A et de la moitie B dans (u1, u2, u3, u4)
        if POS == 1:
            a0 = ib; a1 = isml
        elif POS == 2:
            a0 = isml; a1 = ib
        elif POS == 3:
            a0 = ib; a1 = isml
        else:
            a0 = isml; a1 = ib
        sA = math.log(fv[a0]) + math.log(fv[a1])
        lo = np.searchsorted(Rs, lo_eps - sA); hi = np.searchsorted(Rs, -sA)
        for r in range(lo, hi):
            b0 = RI[r]; b1 = RJ[r]; nq += 1
            if POS <= 2:
                q[0] = a0; q[1] = a1; q[2] = b0; q[3] = b1
            else:
                q[0] = b0; q[1] = b1; q[2] = a0; q[3] = a1
            i1 = q[0]; i2 = q[1]; i3 = q[2]; i4 = q[3]
            pf = fv[i1] * fv[i2] * fv[i3] * fv[i4]
            if pf >= 1.0:
                continue
            rr = f1p[i1] * fv[i2] * fv[i3] * fv[i4] / (1.0 - pf)
            if rr * rr * rr * GU[i1] < nmin:
                continue
            if ly[i1] + ly[i2] + ly[i3] + ly[i4] > LYmax:
                nbig += 1; continue
            C = Y[i1] * Y[i2] * Y[i3] * Y[i4] - X[i1] * X[i2] * X[i3] * X[i4]
            if C <= 0:
                continue
            for v in (Y[i1], Z[i1], X[i2], X[i3], X[i4]):
                g = gcd64(C, v)
                if g > 1:
                    C //= g
                if C <= cmax:
                    break
            if C <= cmax:
                out[cnt, 0] = i1; out[cnt, 1] = i2; out[cnt, 2] = i3; out[cnt, 3] = i4; out[cnt, 4] = C
                cnt += 1; tot += 1
                if cnt == cap:
                    oo = np.argsort(out[:, 4]); tmp = out[oo[:cap // 2]].copy()
                    out[:cap // 2] = tmp; cnt = cap // 2; cmax = out[cap // 2 - 1, 4] - 1
    return cnt, nq, nbig, tot, cmax


out = np.zeros((CAP, 5), np.int64)
t0 = time.time()
cnt, nq, nbig, tot, cmx = run(POS, NS, NB, Rs, RI, RJ, X, Y, Z, ly, math.log(1 - EPS), 10**12, LYmax, W, NW, out, CAP,
                              fv, f1p, GU, 0.1 * NMIN)
t1 = time.time()
log = open(OUT, 'a')


def P(d):
    print(json.dumps(d), file=log, flush=True)


P(dict(G=G, Us=US, Ulo=ULO, Uhi=UHI, pos=POS, eps=EPS, nmin=NMIN, w=W, nw=NW, ns=NS, nb=NB, quads=int(nq),
       toobig=int(nbig), cand=int(cnt), candtot=int(tot), cmax_eff=int(cmx), tmitm=round(t1 - t0, 1)))
U = UHI
# ---- suite identique a search3.py : crible local, factorisation PARI, verification
rows=out[:cnt]; rows=rows[np.argsort(rows[:,4])]
# ---- local density scoring
pfc={}
def pfs(x):
    if x not in pfc: pfc[x]=primefactors(x)
    return pfc[x]
def vv(x,l):
    v=0
    while x%l==0: x//=l; v+=1
    return v
SMALL=(2,3,5,7,11,13)
def score(s,u,C,D,E,maxM=8192):
    ls=set(SMALL)
    for x in u: ls.update(pfs(x))
    ls.update(pfs(G))
    r=1.0; sa=np.array(s,np.int64); ua=np.array(u,np.int64)
    for l in sorted(ls,key=lambda l:(l==2,l)):
        v=sum(vv(G*x,l) for x in u)
        K=max(v+1,2 if l<50 else 1)
        while l**K>maxM and K>1: K-=1
        if l**K>maxM:
            r*=float(l)**(-sum(vv(x,l) for x in u[1:])); continue
        M=l**K
        a,b=rho(l,K,C%M,D%M,E%M,sG,G,sa,ua)
        if a==0: return 0.0
        r*=a/b
    return r
t2=time.time(); scored=[]; nzero=0
for row in rows:
    if time.time()-t2>TBUD*0.85: break
    ii=[int(k) for k in row[:4]]
    u=[int(UU[k]) for k in ii]; s=[SS[k] for k in ii]
    C,D,E=icoeffs(sG,G,s,u)
    r=score(s,u,C,D,E)
    if r==0: nzero+=1; continue
    scored.append((r/C,C,D,E,u))
nsc=len(scored)+nzero
scored.sort(key=lambda t:-t[0])
P(dict(w=W,scored=nsc,zero=nzero,lastC_scored=int(rows[nsc-1,4]) if nsc else None,tscore=round(time.time()-t2,1)))
# ---- factor in order of expected value
t3=time.time(); nfac=0; hard=[]; seen=set(); nf=0; last=None
for k,(sc,C,D,E,u) in enumerate(scored):
    if time.time()-t3>TBUD*0.15+60: last=sc; break
    N=D*D+C*E
    try:
        alarm(PSEC); f=pari.factor(N); cancel_alarm()
    except AlarmInterrupt:
        hard.append(dict(G=G,u=u,C=C,D=D,E=E,N=str(N),Nbits=N.bit_length(),score=sc)); continue
    nfac+=1
    ps=[int(x) for x in f[0]]; es=[int(x) for x in f[1]]
    divs=[1]
    for pr,e in zip(ps,es): divs=[x*pr**j for x in divs for j in range(e+1)]
    for s1 in divs:
        s2=N//s1
        if s1>s2 or (s1+D)%C or (s2+D)%C: continue
        p=(s1+D)//C; q=(s2+D)//C
        r=chain_ok(G,u,p,q)
        if r and min(r) not in seen:
            seen.add(min(r)); nf+=1; P(dict(G=G,u=u,p=p,q=q,cycle=r,C=C))
for h in hard: P(dict(hard=True,**h))
P(dict(done=True,G=G,U=U,w=W,factored=nfac,hard=len(hard),found=nf,stopped_score=last,nscored=len(scored),ttot=round(time.time()-t0,1)))
