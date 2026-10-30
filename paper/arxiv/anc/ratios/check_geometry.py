"""Independent closure, nonorthogonal, prime and conference certificates."""
from pathlib import Path
from itertools import product,combinations,permutations
from fractions import Fraction as F
import json
from exact import require,dot,canonical,rank
HERE=Path(__file__).resolve().parent

def main():
    data=json.loads((HERE/'closure.json').read_text())
    rays=sorted({canonical(v) for v in product(range(-2,3),repeat=4) if any(v)})
    require(rays==[tuple(v) for v in data['rays']] and len(rays)==272,'height-two ray enumeration')
    adj={v:{w for w in rays if dot(v,w)==0} for v in rays}
    bases=[]
    for a in rays:
        for b in sorted(w for w in adj[a] if w>a):
            for c in sorted(w for w in adj[a]&adj[b] if w>b):
                for d in sorted(w for w in adj[a]&adj[b]&adj[c] if w>c):bases.append((a,b,c,d))
    require(bases==[tuple(tuple(v) for v in b) for b in data['bases']] and len(bases)==380,'height-two bases')
    seed=json.loads((HERE/'game-3.json').read_text())['seeds']
    known={canonical(v) for b in seed for v in b}
    require(len(known)==24,'seed ray count')
    seedset=set(known)
    # Left multiplication by 1+i, 1+j, 1+k; the omitted scalar does not alter rays.
    def left(v,k):
        a,b,c,d=v
        return [(a-b,a+b,c-d,c+d),(a-c,b+d,a+c,d-b),(a-d,b-c,b+c,a+d)][k]
    for k in range(3):require({canonical(left(v,k)) for v in seedset}==seedset,'quaternion seed symmetry')
    for perm in permutations(range(4)):
        for signs in product((-1,1),repeat=4):
            require({canonical([signs[j]*v[perm[j]] for j in range(4)]) for v in seedset}==seedset,'signed coordinate symmetry')
    available={tuple(tuple(v) for v in b) for b in seed};rounds=[]
    while len(known)<272:
        ready={v for v in rays if rank(list(adj[v]&known))==3}
        new=[b for b in bases if b not in available and all(v in ready for v in b)]
        require(new,'closure stalled')
        count=len(known);known.update(v for b in new for v in b);available.update(new)
        rounds.append([len(new),len(known)-count])
    require(rounds==data['rounds']==[[146,64],[212,152],[16,32]],'closure rounds')
    witnesses=json.loads((HERE/'nonorthogonal.json').read_text());checked=[]
    for item in witnesses:
        r=F(item['ratio']);w=item['w'];v=item['v'];ds=[]
        for x,y in product(range(2),repeat=2):
            t=dot(w[x],v[y]);require(t!=0,'zero witness overlap');ds.append(t)
            require(F(dot(w[x],w[x])*dot(v[y],v[y]),t*t)==r,'witness squared overlap')
        require(ds[0]*ds[1]*ds[2]*ds[3]<0,'witness frustration')
        require(dot(w[0],w[1])!=0 or dot(v[0],v[1])!=0,'witness is not nonorthogonal')
        checked.append(str(r))
    primes=[]
    for p in range(3,500,2):
        if any(p%d==0 for d in range(2,int(p**.5)+1)):continue
        pair=next(((a,b) for a,b in product(range((p+1)//2),repeat=2) if (2+a*a+b*b)%p==0),None)
        require(pair is not None,'prime residue witness')
        a,b=pair;k=(2+a*a+b*b)//p
        require(0<k<p,'prime induction bound');primes.append(dict(p=p,a=a,b=b,k=k))
    require(len(primes)==94,'prime count')
    # Directly exhibit the signed row/column equivalence of the manuscript basis.
    R=[[1,1,0,1],[1,-1,1,0],[-1,0,1,1],[0,1,1,-1]]
    C=[[0,1,1,1],[-1,0,1,-1],[-1,-1,0,1],[-1,1,-1,0]]
    equivalence=None
    for perm in permutations(range(4)):
        for rs in product((-1,1),repeat=4):
            for cs in product((-1,1),repeat=4):
                if all(rs[i]*R[perm[i]][j]*cs[j]==C[i][j] for i,j in product(range(4),repeat=2)):
                    equivalence=dict(row_permutation=perm,row_signs=rs,column_signs=cs);break
            if equivalence:break
        if equivalence:break
    require(equivalence is not None,'conference equivalence missing')
    require(all(C[i][j]==-C[j][i] for i,j in product(range(4),repeat=2)),'conference skewness')
    require(all(dot(C[i],C[j])==(3 if i==j else 0) for i,j in product(range(4),repeat=2)),'conference Gram matrix')
    summary=dict(status='PASS',closure_rays=272,closure_bases=380,rounds=rounds,nonorthogonal_witnesses=checked,prime_witnesses=primes,conference_equivalence=equivalence)
    (HERE/'geometry-checks.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8',newline='\n')
    print('PASS: closure, ten nonorthogonal witnesses, 94 primes and conference equivalence')
if __name__=='__main__':main()
