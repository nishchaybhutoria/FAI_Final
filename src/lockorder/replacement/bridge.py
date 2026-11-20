"""One temporary *booking total*, exact at a fixed clock, with reliable serial releases.
No extension of the lossy direct probability law to bridges is implied.
"""
from dataclasses import dataclass,replace
from itertools import combinations,pairwise
from math import inf,isfinite
from ..models.physics import t_minus,t_plus
from .model import EPS,ancestry,compile_update,select
from .protocol import Drivers,Ledger


def _times(f,k):
    job=next(j for j in f.snap.jobs() if j.id==k);a=f.lane[k];t=f.snap.t
    lo=max(t,job.floor,job.earliest,*(f.snap.last[b]+f.snap.H[b][a] for b in (0,1)));hi=job.latest
    if f.kin and k in f.heads:
        p,v=f.heads[k].state(t);lo=max(lo,t+t_minus(p,v,f.kin));hi=min(hi,t+t_plus(p,v,f.kin))
    if hi<lo-EPS:return
    cuts={lo,f.old[k],f.new[k]}
    if isfinite(hi):cuts.add(hi)
    for j in f.old:
        if j==k:continue
        b=f.lane[j];ab=f.snap.H[a][b];ba=f.snap.H[b][a]
        if a==b:ab=ba=max(ab,f.snap.TL)
        for x in (f.old[j],f.new[j]):cuts.update((x-ab,x+ba))
    cuts=sorted(x for x in cuts if lo<=x<=hi)
    pts=cuts+[(x+y)/2 for x,y in pairwise(cuts)]
    if not isfinite(hi):pts.append(cuts[-1]+max(1.,f.snap.H[a][a]))
    for x in sorted(pts):
        if x>t+EPS and min(abs(x-f.old[k]),abs(x-f.new[k]))>EPS:yield x


def _order(graph):
    depth,_=ancestry(graph)
    return sorted(depth,key=lambda z:(depth[z],z)) if len(depth)==len(graph) else None


def complete(f):
    """Witness to COMPLETE this target, or None if impossible with at most one detour."""
    c=compile_update(f,0.);D=f.changed
    if any(not f.certified(i,f.snap.t) for i in D):return None
    direct=_order(c.parents)
    if direct is not None:return [(i,f.new[i]) for i in direct]
    for k in sorted(D-c.depth.keys()):
        # The parking holder must hit every remaining cycle.
        if _order({i:ps-{k} for i,ps in c.parents.items() if i!=k}) is None:continue
        for x in _times(f,k):
            if any(not f.compatible(k,x,j,f.old[j]) for j in f.old.keys()-D):continue
            out=(k,0);back=(k,1);g={(i,int(i==k)):{(j,0) for j in ps} for i,ps in c.parents.items()};g[out]=set();g[back].add(out)
            for j in D-{k}:
                if not f.compatible(k,x,j,f.old[j]):g[out].add((j,0))
                if not f.compatible(k,x,j,f.new[j]):g[j,0].add(back)
            seq=_order(g)
            if seq is not None:return [(i,x if (i,phase)==out else f.new[i]) for i,phase in seq]
    return None


@dataclass
class Result:
    path:list
    exact:bool
    full:bool


def best(f,cap=10):
    """Best reachable old/target hybrid for a NOMINALLY OPTIMAL input target.
    A full witness attains the snapshot bound. Fallback is labeled if enumeration capped.
    """
    path=complete(f)
    if path is not None:return Result(path,True,True)
    c=compile_update(f,0.);sel=select(f,c,1,1);path=[(i,f.new[i]) for i in sorted(sel,key=lambda i:(c.depth[i],i))]
    D=sorted(f.changed);value=sum(f.old[i]-s for i,s in path)
    if len(D)>cap:return Result(path,False,False)
    for n in range(1,len(D)):
        for tup in combinations(D,n):
            S=set(tup);gain=sum(f.old[i]-f.new[i] for i in S)
            if gain<=value+EPS or any(not c.parents[i]<=S for i in S):continue
            partial=replace(f,new=f.old|{i:f.new[i] for i in S});p=complete(partial)
            if p is not None:path,value=p,gain
    return Result(path,True,False)


def replay(f,path):
    """Reliable serial zero-time handoffs. Carry actual profiles, per-car versions and
    manager's confirmed singleton slots across moves; never manufacture a release.
    """
    w=Drivers(f);known=dict(f.old);versions={i:f.epoch-1 for i in known};stages={i:0 for i in known};temp=None
    for i,s in path:
        if stages[i] or s!=f.new[i]:
            if (stages[i]==0 and temp is None and s!=f.old[i]):temp=i
            elif not (stages[i]==1 and temp==i and s==f.new[i]):raise ValueError('more than one temporary substitution')
        if s==known[i] or i not in f.changed:raise ValueError('not a changed slot')
        leg=replace(f,old=known.copy(),new=known|{i:s},heads=dict(w.profiles),epoch=versions[i]+1)
        m=Ledger(leg);g=m.issue(i,f.snap.t);w.f=leg
        if not g or not w.receive(g,f.snap.t) or w.switch(g,f.snap.t)!=g or not m.ack(g):raise AssertionError('invalid bridge handoff')
        known[i],=m.possible[i];versions[i]=g.epoch;stages[i]+=1
    if known!=w.active or any(known[i] not in (f.old[i],f.new[i]) for i in known):raise AssertionError('unfinished bridge')
    w.advance(inf);return w.entered
