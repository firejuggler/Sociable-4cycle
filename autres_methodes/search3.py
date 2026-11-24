# Two-large-prime search v3: MITM on reduced-C lower bound + local-density (rho) ranking + PARI/yafu
# usage: python3 search3.py G U cap worker nworkers log NMIN tbudget_factor [pari_sec]
import sys, time, json, math, numpy as np, numba as nb
from math import gcd
from sympy import divisor_sigma, primefactors
G=int(sys.argv[1]); U=int(sys.argv[2]); CAP=int(sys.argv[3]); W=int(sys.argv[4]); NW=int(sys.argv[5])
OUT=sys.argv[6]; NMIN=float(sys.argv[7]); TBUD=float(sys.argv[8]); PSEC=int(sys.argv[9]) if len(sys.argv)>9 else 2
import cypari2; pari=cypari2.Pari(); pari.allocatemem(10**9,silent=True)
from cysignals.alarm import alarm, AlarmInterrupt, cancel_alarm
from core import chain_ok
from resid import icoeffs, rho
sG=int(divisor_sigma(G))
X=[];Y=[];Z=[];Wd=[];UU=[];SS=[]
for u in range(1,U+1):
    if gcd(u,G)!=1: continue
    s=int(divisor_sigma(u)); x=sG*s-G*u; y=G*u
    if x<=0: continue
    g=gcd(x,y); z=sG*s; w=G*u; g2=gcd(z,w)
    X.append(x//g);Y.append(y//g);Z.append(z//g2);Wd.append(w//g2);UU.append(u);SS.append(s)
X=np.array(X,np.int64);Y=np.array(Y,np.int64);Z=np.array(Z,np.int64);UU=np.array(UU,np.int64)
n=len(UU); fv=X/Y; lf=np.log(fv); ly=np.log(Y.astype(np.float64)); f1p=fv+1.0; GU=(G*UU).astype(np.float64)
EPS=min(0.06,float(np.max(Z/np.array(Wd)))*(G*U/(0.1*NMIN))**(1/3)*1.01)
LYmax=math.log(2.0**63)+math.log(1.0/EPS)-1.0   # C=py-px exact by wraparound while true C<2^63
I,J=np.meshgrid(np.arange(n),np.arange(n),indexing='ij'); I=I.ravel().astype(np.int32);J=J.ravel().astype(np.int32)
S12=(lf[I]+lf[J]); o=np.argsort(S12); Rs=S12[o].copy(); RI=I[o].copy(); RJ=J[o].copy()
@nb.njit(cache=True)
def gcd64(a,b):
    while b: a,b=b,a%b
    return a
@nb.njit(cache=True)
def run(I,J,S,Rs,RI,RJ,X,Y,Z,ly,lo_eps,cmax,LYmax,W,NW,out,cap,fv,f1p,GU,nmin):
    cnt=0; nq=0; nbig=0; tot=0
    for t in range(W,S.shape[0],NW):
        i1=I[t]; i2=J[t]
        lo=np.searchsorted(Rs,lo_eps-S[t]); hi=np.searchsorted(Rs,-S[t])
        for r in range(lo,hi):
            i3=RI[r]; i4=RJ[r]; nq+=1
            pf=fv[i1]*fv[i2]*fv[i3]*fv[i4]
            if pf>=1.0: continue
            rr=f1p[i1]*fv[i2]*fv[i3]*fv[i4]/(1.0-pf)
            if rr*rr*rr*GU[i1]<nmin: continue
            if ly[i1]+ly[i2]+ly[i3]+ly[i4]>LYmax: nbig+=1; continue
            C=Y[i1]*Y[i2]*Y[i3]*Y[i4]-X[i1]*X[i2]*X[i3]*X[i4]
            if C<=0: continue
            for v in (Y[i1],Z[i1],X[i2],X[i3],X[i4]):
                g=gcd64(C,v)
                if g>1: C//=g
                if C<=cmax: break
            if C<=cmax:
                out[cnt,0]=i1;out[cnt,1]=i2;out[cnt,2]=i3;out[cnt,3]=i4;out[cnt,4]=C
                cnt+=1; tot+=1
                if cnt==cap:
                    oo=np.argsort(out[:,4]); tmp=out[oo[:cap//2]].copy()
                    out[:cap//2]=tmp; cnt=cap//2; cmax=out[cap//2-1,4]-1
    return cnt,nq,nbig,tot,cmax
out=np.zeros((CAP,5),np.int64)
t0=time.time()
cnt,nq,nbig,tot,cmx=run(I,J,S12,Rs,RI,RJ,X,Y,Z,ly,math.log(1-EPS),10**12,LYmax,W,NW,out,CAP,fv,f1p,GU,0.1*NMIN)
t1=time.time()
log=open(OUT,'a')
def P(d): print(json.dumps(d),file=log,flush=True)
P(dict(G=G,U=U,eps=EPS,nmin=NMIN,w=W,nw=NW,n=n,quads=int(nq),toobig=int(nbig),cand=int(cnt),candtot=int(tot),cmax_eff=int(cmx),tmitm=round(t1-t0,1)))
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
