# symbolic derivation / check of the two-large-prime closing condition
from sympy import symbols, simplify, expand, factor, Poly
p,q,P2,h,s1,s2,s3,s4,u1,u2,u3,u4=symbols('p q P2 h s1 s2 s3 s4 u1 u2 u3 u4',positive=True)
H=h*s1
M2=H*(p+1)*(q+1)-u1*p*q          # m2 = h*sigma(u1 p q) - u1 p q
P2e=M2/u2
A=lambda s,u: h*s-u
B=lambda s: h*s
P3=(A(s2,u2)*P2e+B(s2))/u3
P4=(A(s3,u3)*P3+B(s3))/u4
m1=A(s4,u4)*P4+B(s4)             # closing: must equal u1 p q
E=expand((m1-u1*p*q)*u2*u3*u4)
Pl=Poly(E,p,q)
cpq=Pl.coeff_monomial(p*q); cp=Pl.coeff_monomial(p); cq=Pl.coeff_monomial(q); c0=Pl.coeff_monomial(1)
print('coef pq :',factor(cpq)); print('coef p  :',factor(cp)); print('coef q-coef p:',simplify(cq-cp)); print('const :',factor(c0))
# claimed form: alpha = A2A3A4/(u2u3u4), beta = composite constant ; c=u1(1+alpha)-alpha*H, d=alpha*H, e=alpha*H+beta
al=A(s2,u2)*A(s3,u3)*A(s4,u4)/(u2*u3*u4)
be=expand(m1.subs({p:0,q:0}) - al*H)   # m1 at p=q=0 is alpha*H+beta? check:
beta=(A(s4,u4)*(A(s3,u3)*B(s2)+B(s3)*u3)/(u3*u4)+B(s4))
c=u1*(1+al)-al*H; d=al*H; e=al*H+beta
K=u2*u3*u4
print('check c:',simplify(-cpq-c*K), ' d:',simplify(cp-d*K),' e:',simplify(c0-e*K))
# identity (cp-d)(cq-d)=d^2+ce whenever c pq - d(p+q) - e = 0
cc,dd,ee=symbols('c d e'); print('te Riele:',expand((cc*p-dd)*(cc*q-dd)-dd**2-cc*ee - cc*(cc*p*q-dd*(p+q)-ee)))
# positivity: c>0 <=> f1 f2 f3 f4 < 1 with f_i = h s_i/u_i - 1
f=[h*s/u-1 for s,u in ((s1,u1),(s2,u2),(s3,u3),(s4,u4))]
print('c/u1 - (1+alpha)(1-f1f2f3f4/(1+... )):',simplify(c/(u1) - (1+al - al*(f[0]+1))), ' ; alpha==f2f3f4:',simplify(al-f[1]*f[2]*f[3]))
