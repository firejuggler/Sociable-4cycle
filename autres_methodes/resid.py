# residue (local) feasibility of the two-prime chain, and fast integer coefficients
import numba as nb, numpy as np
from math import gcd
from sympy import primefactors
from sympy import symbols
import os
Estr=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'icoef.txt')).read().split('E=')[1].strip()
_S,_g,_s1,_s2,_s3,_s4,_u1,_u2,_u3,_u4=symbols('S g s1 s2 s3 s4 u1 u2 u3 u4')
Efun=eval('lambda S,g,s1,s2,s3,s4,u1,u2,u3,u4: '+Estr)
def icoeffs(S,g,s,u):
    X=[S*s[i]-g*u[i] for i in range(4)]
    C=g**4*u[0]*u[1]*u[2]*u[3]-X[0]*X[1]*X[2]*X[3]
    D=S*s[0]*X[1]*X[2]*X[3]
    E=Efun(S,g,s[0],s[1],s[2],s[3],u[0],u[1],u[2],u[3])
    t=gcd(gcd(C,D),E)
    return C//t,D//t,E//t
@nb.njit(cache=True)
def vl(x,l):
    v=0
    while x%l==0 and x>0: x//=l; v+=1
    return v
@nb.njit(cache=True)
def inv(a,m):
    # a unit mod m
    t,nt,r,nr=0,1,m,a%m
    while nr:
        qq=r//nr; t,nt=nt,t-qq*nt; r,nr=nr,r-qq*nr
    return t%m
@nb.njit(cache=True)
def rho(l,K,Cm,Dm,Em,S,g,s,u):
    """fraction of unit residues p mod l^K for which q (from bilinear eq) and the chain are locally OK.
    returns (npass, ntotal).  Unknown (precision / non-invertible) counts as pass."""
    M=1
    for _ in range(K): M*=l
    vs=np.zeros(4,np.int64); us=np.zeros(4,np.int64); pw=np.ones(4,np.int64)
    for i in range(4):
        vs[i]=vl(g*u[i],l)
        for _ in range(vs[i]): pw[i]*=l
        us[i]=(g*u[i])//pw[i]
    if Cm%l==0 and Dm%l==0: return 0,1
    npass=0; tot=0
    for p in range(M):
        if p%l==0: continue
        tot+=1
        den=(Cm*p-Dm)%M
        if den%l==0: npass+=1; continue
        q=((Dm*p+Em)%M)*inv(den,M)%M
        if q%l==0: continue
        n1=(g*u[0]%M)*(p*q%M)%M
        n=( (S*s[0]%M)*((p+1)*(q+1)%M)%M - n1)%M
        Mc=M; ok=True; unk=False
        for i in (1,2,3,0):
            # n = G*u_i*P_i  (i==0: closure n == n1)
            if i==0:
                if (n-n1)%Mc!=0: ok=False
                break
            if Mc<=pw[i]:
                unk=True; break
            if n%pw[i]!=0: ok=False; break
            Mc2=Mc//pw[i]
            x=(n//pw[i])%Mc2
            if x%l==0: ok=False; break   # v_l too big -> P_i divisible by l
            P=x*inv(us[i]%Mc2,Mc2)%Mc2
            Mc=Mc2
            n=((S*s[i]%Mc)*((P+1)%Mc)%Mc-(g*u[i]%Mc)*P%Mc)%Mc
        if unk or ok: npass+=1
    return npass,tot
SMALL=[2,3,5,7,11,13]
def score(S,g,s,u,C,D,E,maxM=20000):
    """local density rho (product over primes) ; 0 => impossible"""
    ls=set(SMALL)
    for x in u: ls|=set(primefactors(x))
    ls|=set(primefactors(g))
    r=1.0
    sa=np.array(s,np.int64); ua=np.array(u,np.int64)
    for l in sorted(ls):
        v=sum(_v(g*x,l) for x in u)
        K=max(v+1,2 if l<50 else 1)
        while l**K>maxM and K>1: K-=1
        if l**K>maxM:  # crude approx for big primes
            r*=float(l)**(-sum(_v(x,l) for x in u[1:])); continue
        M=l**K
        a,b=rho(l,K,C%M,D%M,E%M,S%M,g%M if False else g,sa,ua)
        if a==0: return 0.0
        r*=a/b
    return r
def _v(x,l):
    v=0
    while x%l==0: x//=l; v+=1
    return v
