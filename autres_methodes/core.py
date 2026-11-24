# exact solver for the two-large-prime variant
from fractions import Fraction as Fr
from math import gcd, prod
from sympy import divisor_sigma, isprime, factorint, divisors
def sig(n): return int(divisor_sigma(n))
def coeffs(G,us):
    """integer (C,D,E) with C pq - D(p+q) - E = 0, reduced by gcd"""
    sG=sig(G); h=Fr(sG,G); u1,u2,u3,u4=us; s=[sig(u) for u in us]
    A=[h*s[i]-us[i] for i in range(4)]; B=[h*s[i] for i in range(4)]
    H=B[0]
    al=A[1]*A[2]*A[3]/(u2*u3*u4)
    beta=A[3]*(A[2]*B[1]+B[2]*u3)/(u3*u4)+B[3]
    c=u1*(1+al)-al*H; d=al*H; e=al*H+beta
    L=1
    for x in (c,d,e): L=L*x.denominator//gcd(L,x.denominator)
    C,Dd,E=int(c*L),int(d*L),int(e*L)
    g=gcd(gcd(C,Dd),E)
    return C//g,Dd//g,E//g
def s_of(n):
    return sig(n)-n
def chain_ok(G,us,p,q):
    """check p,q -> P2,P3,P4 integral primes and closure; return cycle or None"""
    sG=sig(G); h=Fr(sG,G); u1,u2,u3,u4=us
    if p==q or not (isprime(p) and isprime(q)): return None
    if gcd(p*q,G*u1)!=1: return None
    m=[u1*p*q]
    M=h*sig(u1)*(p+1)*(q+1)-u1*p*q
    P=[]
    for i in (1,2,3):
        x=M/us[i]
        if x.denominator!=1: return None
        x=int(x)
        if x<2 or not isprime(x) or gcd(x,G*us[i])!=1: return None
        P.append(x); m.append(us[i]*x)
        M=(h*sig(us[i])-us[i])*x+h*sig(us[i])
    if M!=m[0]: return None
    ns=[G*x for x in m]
    # exact verification
    for i in range(4):
        if s_of(ns[i])!=ns[(i+1)%4]: return None
    if len(set(ns))<4: return None
    return ns
def solve(G,us,flim=None):
    C,Dd,E=coeffs(G,us)
    if C<=0: return [],None
    N=Dd*Dd+C*E
    if N<=0: return [],N
    if flim is not None:
        f=flim(N)
        if f is None: return [],N
    else: f=factorint(N)
    # enumerate divisors s with s = -D mod C
    divs=[1]
    for pr,e in f.items():
        divs=[x*pr**k for x in divs for k in range(e+1)]
    out=[]
    for s1 in divs:
        if (s1+Dd)%C: continue
        s2=N//s1
        if s1>s2: continue
        if (s2+Dd)%C: continue
        p=(s1+Dd)//C; q=(s2+Dd)//C
        r=chain_ok(G,us,p,q)
        if r: out.append(r)
    return out,N
