"""Exact rational multiquadratic arithmetic; no numerical tolerances."""
from fractions import Fraction as F
from math import gcd

def require(test,message):
    if not test: raise RuntimeError(message)

class Q:
    def __init__(self,value=0):
        self.a={k:F(v) for k,v in value.items() if v} if isinstance(value,dict) else ({1:F(value)} if value else {})
    def __add__(self,b):
        b=b if isinstance(b,Q) else Q(b); out=dict(self.a)
        for k,v in b.a.items():out[k]=out.get(k,F(0))+v
        return Q(out)
    __radd__=__add__
    def __neg__(self):return Q({k:-v for k,v in self.a.items()})
    def __sub__(self,b):return self+-Q.coerce(b)
    def __rsub__(self,b):return Q.coerce(b)+-self
    @staticmethod
    def coerce(b):return b if isinstance(b,Q) else Q(b)
    def __mul__(self,b):
        b=Q.coerce(b);out={}
        for k,v in self.a.items():
            for l,w in b.a.items():
                g=gcd(k,l);t=k*l//(g*g)
                out[t]=out.get(t,F(0))+g*v*w
        return Q(out)
    __rmul__=__mul__
    def __truediv__(self,b):return self*F(1,b)
    def __eq__(self,b):return self.a==Q.coerce(b).a
    def serial(self):return {str(k):str(v) for k,v in sorted(self.a.items())}
    def __repr__(self):return str(self.serial())

def sqrt(v):
    v=F(v);require(v>=0,'negative radicand')
    if not v:return Q()
    n=v.numerator*v.denominator;s=1;d=2
    while d*d<=n:
        while n%(d*d)==0:n//=d*d;s*=d
        d+=1
    return Q({n:F(s,v.denominator)})

def dot(v,w):return sum(a*b for a,b in zip(v,w))
def canonical(v):
    g=gcd(*v);require(g!=0,'zero ray')
    v=tuple(x//g for x in v)
    return tuple(-x for x in v) if next(x for x in v if x)<0 else v
def rank(vs):
    rows=[[F(x) for x in v] for v in vs];k=0
    for j in range(4):
        pivot=next((i for i in range(k,len(rows)) if rows[i][j]),None)
        if pivot is None:continue
        rows[k],rows[pivot]=rows[pivot],rows[k]
        rows[k]=[x/rows[k][j] for x in rows[k]]
        for i in range(k+1,len(rows)):
            t=rows[i][j]
            if t:rows[i]=[a-t*b for a,b in zip(rows[i],rows[k])]
        k+=1
    return k

def unit(v):return [x*sqrt(F(1,dot(v,v))) for x in v]
def outer(v):return [[a*b for b in v] for a in v]
