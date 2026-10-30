"""Finite diagnostics for rank/defect geometry and composition; not type proofs."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from bisect import bisect_right
import argparse,json,math,random
HERE=Path(__file__).resolve().parent

def require(condition,message):
    if not condition:raise RuntimeError(message)

def defect(groups,r):
    ordered=sorted(groups,key=lambda x:-x[1])
    require(all(m>0 and p>0 for m,p in ordered),'positive matrix spectrum required')
    total=math.fsum(m*p for m,p in ordered)
    values=[p/total for m,p in ordered];edges=[0.0]
    for m,p in ordered:edges.append(edges[-1]+m)
    points=sorted(set(edges+[x/r for x in edges]))
    def value(t):
        k=bisect_right(edges,t)-1
        return values[k] if 0<=k<len(values) else 0.0
    H2=math.fsum((b-a)*(math.sqrt(value((a+b)/2))-math.sqrt(r*value(r*(a+b)/2)))**2 for a,b in zip(points,points[1:]))
    require(math.isfinite(H2) and H2>0,'invalid defect')
    return math.sqrt(H2),sum(m for m,p in ordered)

def profile(r,L,noise=0,seed=0):
    rng=random.Random(seed)
    c=[math.sin(math.pi*(j+1)/(L+2))*(1+noise*rng.uniform(-1,1)) for j in range(L+1)]
    norm=math.fsum(x*x for x in c)
    return [(r**j,c[j]**2/(norm*r**j)) for j in range(L+1)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cells',type=Path,default=HERE/'cells.json');args=parser.parse_args()
    cells=json.loads(args.cells.read_text(encoding='utf8'))
    expected={'finite':['filled','empty','empty','empty','empty','empty'],'I_infinity':['empty','filled','open','open','open','empty'],'II':['empty','open','filled','open','filled','filled'],'III_only':['empty','open','open','open','filled','filled']}
    require(cells['columns']==['B','L','P','SPE','E','gap'] and cells['rows']==expected,'type/price cells changed')
    statuses=[x for row in cells['rows'].values() for x in row]
    require(statuses.count('open')==8 and len(statuses)==24,'settled/open cell count')
    arcsine=[];a=2/(math.sqrt(2)-1)**2;b=2*(math.sqrt(2)-1)*a
    for i in range(2000):
        H=math.exp(math.log(1e-7)+(math.log(.05)-math.log(1e-7))*i/1999)
        factor=math.sqrt((1+b*H*H)/(1-a*H**4));z=H*factor/2
        require(factor<=1+4.9*H*H+3e-15,'square-root remainder')
        require(2*math.asin(z)<=H*(1+5*H*H)+3e-15,'arcsine remainder')
        remainder=math.pi/H-math.pi/(2*math.asin(z))
        require(remainder<1,'rank remainder exceeds one')
        arcsine.append(remainder)
    profiles=[]
    for r in (2,3):
        for L in (12,24,48,96,192):
            H,R=defect(profile(r,L),r)
            row=dict(base=r,levels=L+1,H=H,log_rank=math.log(R,r),scaled_coefficient=math.log(R,r)*H/math.pi,in_range=H<=.05)
            if H<=.05:
                slack=math.log(R,r)-(math.pi/H-2*math.log(1/H,r)-3)
                require(slack>=-1e-10,'sine matrix violates rank inequality');row['slack']=slack
            profiles.append(row)
        for L,seed in product((80,120,160),range(3)):
            H,R=defect(profile(r,L,.002,seed),r)
            require(H<=.05,'perturbed diagnostic outside range')
            slack=math.log(R,r)-(math.pi/H-2*math.log(1/H,r)-3)
            require(slack>=-1e-10,'perturbed matrix violates rank inequality')
            profiles.append(dict(base=r,levels=L+1,seed=seed,H=H,slack=slack,in_range=True))
    flats=[]
    for r,d in product((2,3),(1,2,8,64)):
        H,R=defect([(d,1/d)],r);target=2*(1-r**(-.5))
        require(abs(H*H-target)<1e-13,'flat-spectrum identity')
        flats.append(dict(base=r,rank=d,H=H,H2=H*H))
    sine=[];coefficient=math.pi*math.sqrt(2/243)
    for L in (32,64,128,256,512,1024):
        loss=8*math.sin(math.pi/(2*L))**2/243
        log_dimension=math.log2(9*(2**(L+1)-1))
        scaled=log_dimension*math.sqrt(loss)
        require(scaled>coefficient and scaled/coefficient<1.2,'sine-template coefficient')
        sine.append(dict(L=L,loss=loss,log2_dimension=log_dimension,scaled_coefficient=scaled))
    require(abs(sine[-1]['scaled_coefficient']/coefficient-1)<.01,'sine-template convergence diagnostic')
    tails=[]
    for lam,n in product((F(2),F(3,2)),(2,10)):
        for eps in (F(1,10),F(1,100),F(1,1000)):
            K=1
            while lam**(-K)>eps/2:K+=1
            require(lam**(-K)<=eps/2<lam**(-(K-1)),'inverse-tail threshold')
            tails.append(dict(lam=str(lam),multiplicity=n,eps=str(eps),K=K,rank=n*K,tail=str(lam**(-K))))
    truth=[]
    for p_halts,q_halts in product((False,True),repeat=2):
        hosted=(not p_halts) or (not q_halts);semi=not q_halts;iii=hosted and not semi
        require(iii==(q_halts and not p_halts),'difference reduction polarity')
        truth.append(dict(P_halts=p_halts,Q_halts=q_halts,hosted=hosted,semifinite=semi,III_only=iii))
    # A disconnected product graph becomes connected using exactly the two stars.
    left=[('A',x,y) for x,y in product(range(2),repeat=2)];right=[('B',x,y) for x,y in product(range(2),repeat=2)]
    am=('A','M');bm=('B','M');edges=list(zip(left,right))+[(am,v) for v in right]+[(v,bm) for v in left]
    seen={am}
    while True:
        added={b for a,b in edges if a in seen}|{a for a,b in edges if b in seen}
        if added<=seen:break
        seen|=added
    require(len(seen)==10,'mode stars do not connect disconnected product questions')
    summary=dict(status='PASS',scope='Finite numerical and Boolean diagnostics only; no proof of hosting or complexity imports',cells=dict(settled=16,open=8),arcsine_samples=2000,max_arcsine_remainder=max(arcsine),profiles=profiles,flat_spectra=flats,sine_limit=coefficient,sine_templates=sine,inverse_tail=tails,reduction_truth_table=truth,connected_mode_graph_vertices=len(seen))
    (HERE/'phase-checks.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8',newline='\n')
    print('PASS: rank remainder, sorted profiles, flat spectra, sine templates, tails, mode graph, reductions and 24 cells')
if __name__=='__main__':main()
