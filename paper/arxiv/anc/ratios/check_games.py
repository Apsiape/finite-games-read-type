"""Reconstruct all six games, PVMs and event probabilities exactly."""
from pathlib import Path
from itertools import product
from fractions import Fraction as F
import argparse,json,re
from exact import Q,sqrt,require,dot,canonical,rank,unit,outer

HERE=Path(__file__).resolve().parent
def mm(A,B):return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def kron(A,B):return [[A[i//2][j//2]*B[i%2][j%2] for j in range(4)] for i in range(4)]
I=[[1,0],[0,1]];X=[[0,1],[1,0]];Z=[[1,0],[0,-1]];J=[[0,-1],[1,0]]
O=[[kron(X,I),kron(I,X),kron(X,X)],[kron(I,Z),kron(Z,I),kron(Z,Z)],[kron(X,Z),kron(Z,X),[[-x for x in row] for row in kron(J,J)]]]

def check(data,out=None):
    r=F(data['ratio']);require(r>=2,'invalid bridge parameter')
    seeds=[[tuple(v) for v in b] for b in data['seeds']]
    contexts=O+[[O[i][j] for i in range(3)] for j in range(3)]
    require(len(seeds)==6,'six seed contexts required')
    for n,(basis,ops) in enumerate(zip(seeds,contexts)):
        require(len(basis)==4 and rank(basis)==4,'seed basis rank')
        signs=[]
        for v in basis:
            ev=[]
            for A in ops:
                Av=[dot(row,v) for row in A]
                s=1 if Av==list(v) else -1
                require(Av==[s*x for x in v],'seed is not Pauli eigenray')
                ev.append(s)
            require(ev[0]*ev[1]*ev[2]==(-1 if n==5 else 1),'Pauli parity')
            signs.append(ev)
        require(len({tuple(s) for s in signs})==4,'duplicate seed eigenvalue')
    known={canonical(v) for b in seeds for v in b};bases=list(seeds)
    for step,item in enumerate(data['chain']):
        b=[tuple(v) for v in item['basis']]
        require(len(b)==4 and rank(b)==4,'added basis rank')
        require(all(dot(u,v)==0 for i,u in enumerate(b) for v in b[i+1:]),'nonorthogonal added basis')
        require(len(item['spans'])==4,'four certificates required')
        for v,span in zip(b,item['spans']):
            require(len(span)==3 and rank(span)==3,'certificate must have rank three')
            require(all(canonical(w) in known for w in span),'certificate uses a later ray')
            require(all(dot(v,w)==0 for w in span),'certificate is not perpendicular')
        # Add only after every certificate in this question has passed.
        known.update(canonical(v) for v in b);bases.append(b)
    ws=[tuple(v) for v in data['w']];vs=[tuple(v) for v in data['v']]
    require(len(ws)==len(vs)==2,'two rays per bridge side')
    c=sqrt(1/r)
    for x,y in product(range(2),repeat=2):
        require(dot(unit(ws[x]),unit(vs[y]))==((-1)**(x*y))*c,'cross-overlap sign or magnitude')
    calibrations=[]
    for side in ('w','v'):
        b=[tuple(v) for v in data[side+'_context']]
        require(b in bases,'calibrating basis is missing')
        require(all(canonical(v) in [canonical(z) for z in b] for v in data[side]),'selected ray missing in calibration')
        calibrations.append(bases.index(b))
    flag=[Q(1),Q(),Q(),Q(),Q()]
    A={};B={}
    for k,b in enumerate(bases):
        A[k]=[flag]+[[Q()]+unit(v) for v in b];B[k]=A[k]
    residuals={}
    for side,family,selected in [('w',A,ws),('v',B,vs)]:
        cb=calibrations[side=='v'];basis=bases[cb]
        for i in range(2):
            residual=[j for j,v in enumerate(basis) if canonical(v)!=canonical(selected[i])]
            u=unit(selected[i]);family['TU'[side=='v']+str(i)]=[[sqrt(F(1,2))]+[s*z*sqrt(F(1,2)) for z in u] for s in (1,-1)]+[[Q()]+unit(basis[j]) for j in residual]
            residuals[side,i]=residual
    for family in (A,B):
        for name,vecs in family.items():
            require(len(vecs)==5,'five outcomes required')
            for i,u in enumerate(vecs):
                for j,v in enumerate(vecs):require(dot(u,v)==(i==j),'PVM orthogonality')
            for i,j in product(range(5),repeat=2):require(sum(v[i]*v[j] for v in vecs)==(i==j),'PVM completeness')
    def wins(q,s,a,b):
        if isinstance(q,int) and isinstance(s,int):
            if a==0 or b==0:return a==b
            return dot(bases[q][a-1],bases[s][b-1])!=0
        if isinstance(q,str) and s==calibrations[0]:
            x=int(q[-1]);selected=next(j for j,v in enumerate(bases[s]) if canonical(v)==canonical(ws[x]))
            return b in (0,selected+1) if a<2 else b==residuals['w',x][a-2]+1
        if q==calibrations[1] and isinstance(s,str):
            y=int(s[-1]);selected=next(j for j,v in enumerate(bases[q]) if canonical(v)==canonical(vs[y]))
            return a in (0,selected+1) if b<2 else a==residuals['v',y][b-2]+1
        if isinstance(q,str) and isinstance(s,str) and a<2 and b<2:return (1-2*a)*(1-2*b)==(-1)**(int(q[-1])*int(s[-1]))
        return True
    cells=[];probabilities=[];forbidden=0
    for q,s in product(A,B):
        total=Q();bad=[];probs=[]
        for a,b in product(range(5),repeat=2):
            u,v=A[q][a],B[s][b]
            amplitude=c*u[0]*v[0]+dot(u[1:],v[1:])
            p=amplitude*amplitude/(1/r+4)
            total+=p;probs.append(p.serial())
            if not wins(q,s,a,b):
                require(amplitude==0,'nonzero forbidden amplitude at '+str((q,s,a,b)))
                forbidden+=1;bad.append([a,b])
        require(total==1,'pair normalization at '+str((q,s)))
        cells.append(dict(alice=q,bob=s,forbidden=bad,weight=str(F(1,len(A)*len(B)))))
        probabilities.append(dict(alice=q,bob=s,probabilities=probs))
    record=dict(ratio=str(r),status='PASS',questions=len(A),pairs=len(cells),events=25*len(cells),forbidden=forbidden,added_bases=len(data['chain']),all_PVMs_exact=True,all_forbidden_amplitudes_zero=True,all_pair_totals_one=True)
    expected={F(3):(9,732),F(5,2):(13,1638),F(6):(9,732),F(7):(13,1642),F(10):(13,1638),F(25,4):(17,2902)}
    require((len(A),forbidden)==expected[r],'game size or forbidden-event count changed')
    if out:
        for suffix,d in [('predicate',dict(ratio=str(r),answers=list(range(5)),cells=cells)),('probabilities',probabilities)]:
            target=out/f'game-{str(r).replace("/","-")}-{suffix}.json'
            if target.exists():require(json.loads(target.read_text(encoding='utf8'))==d,'stored '+suffix+' disagrees with exact reconstruction')
            target.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8',newline='\n')
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    files=[args.data] if args.data else sorted(HERE.glob('game-*.json'))
    if not args.data:files=[f for f in files if re.fullmatch(r'game-\d+(?:-\d+)?\.json',f.name)]
    require(len(files)==(1 if args.data else 6),'wrong game-data file count')
    records=[check(json.loads(p.read_text(encoding='utf8')),None if args.check_only else HERE) for p in files]
    if not args.data:
        require(sorted(x['questions'] for x in records)==[9,9,13,13,13,17],'unexpected game sizes')
        (HERE/'game-checks.json').write_text(json.dumps(dict(status='PASS',games=records),indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
