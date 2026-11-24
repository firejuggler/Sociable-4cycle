import sys, time, numpy as np, numba as nb
from math import gcd
from sympy import divisor_sigma, isprime, factorint
Gv=int(sys.argv[1]); U=int(sys.argv[2]); eps=float(sys.argv[3])
SG=int(divisor_sigma(Gv)); h=SG/Gv
us=[u for u in range(1,U+1) if gcd(u,Gv)==1]
su=np.array([int(divisor_sigma(u)) for u in us],dtype=np.float64)
ua=np.array(us,dtype=np.float64)
f=h*su/ua-1.0
keep=f>0
ua,su,f=ua[keep],su[keep],f[keep]
n=len(ua); print('n u',n)
I,J=np.meshgrid(np.arange(n),np.arange(n),indexing='ij'); I=I.ravel();J=J.ravel()
lf=np.log(f)
S=lf[I]+lf[J]
o=np.argsort(S); Rs=S[o]; RI=I[o]; RJ=J[o]
@nb.njit
def run(S,I,J,Rs,RI,RJ,ua,su,h,eps,out,cap):
    le=np.log(1-eps); cnt=0
    for t in range(S.shape[0]):
        lo=np.searchsorted(Rs,le-S[t]); hi=np.searchsorted(Rs,-S[t])
        i1=I[t]; i2=J[t]
        for q in range(lo,hi):
            i3=RI[q]; i4=RJ[q]
            idx=(i1,i2,i3,i4)
            a=1.0; b=0.0
            for k in range(4):
                u=ua[idx[k]]; s=su[idx[k]]; un=ua[idx[(k+1)%4]]
                A=h*s-u; B=h*s
                a=a*A/un; b=(A*b+B)/un
            if 1-a<=0: continue
            x=b/(1-a)
            if x<3 or x>1e13: continue
            ok=True; y=x
            for k in range(4):
                if abs(y-round(y))>1e-4*max(1.0,y*1e-9): ok=False; break
                u=ua[idx[k]]; s=su[idx[k]]; un=ua[idx[(k+1)%4]]
                y=((h*s-u)*y+h*s)/un
            if ok and cnt<cap:
                out[cnt,0]=i1;out[cnt,1]=i2;out[cnt,2]=i3;out[cnt,3]=i4;cnt+=1
    return cnt
out=np.zeros((200000,4),dtype=np.int64)
t=time.time(); c=run(S,I,J,Rs,RI,RJ,ua,su,h,eps,out,200000); print('cand',c,time.time()-t)
def s(m):
    r=1
    for p,e in factorint(m).items(): r*=(p**(e+1)-1)//(p-1)
    return r-m
found=set()
for row in out[:c]:
    uu=[int(ua[k]) for k in row]
    from fractions import Fraction as Fr
    hh=Fr(SG,Gv); a=Fr(1); b=Fr(0)
    for k in range(4):
        u=uu[k]; sg=int(divisor_sigma(u)); un=uu[(k+1)%4]
        A=hh*sg-u; B=hh*sg; a=a*A/un; b=(A*b+B)/un
    x=b/(1-a)
    if x.denominator!=1: continue
    P=[int(x)]
    for k in range(3):
        u=uu[k]; sg=int(divisor_sigma(u)); y=((hh*sg-u)*P[-1]+hh*sg)/uu[k+1]
        if y.denominator!=1: break
        P.append(int(y))
    if len(P)<4 or not all(isprime(p) for p in P): continue
    ns=[Gv*u*p for u,p in zip(uu,P)]
    c2=[ns[0]]
    for k in range(4): c2.append(s(c2[-1]))
    if c2[4]==ns[0] and len(set(c2[:4]))==4:
        found.add(min(c2[:4]))
print('cycles',sorted(found))
