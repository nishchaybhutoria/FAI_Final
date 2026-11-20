"""Finite replacement of two feasible endpoint schedules; no channel draws here."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import math

from ..models.physics import Kinematics, Profile, make_profile, t_minus, t_plus
from ..scheduling.core import Plan, Snapshot, validate_plan

EPS = 1e-8


@dataclass
class Fork:
    snap: Snapshot
    old: dict[int,float]
    new: dict[int,float]
    kin: Kinematics | None
    heads: dict[int,Profile]       # actual old profiles, not freshly interpolated surrogates
    epoch: int
    pred: dict = field(init=False,repr=False)
    lane: dict = field(init=False,repr=False)
    pos: dict = field(init=False,repr=False)

    def __post_init__(self):
        if any(j.need for j in self.snap.jobs()):
            raise ValueError('external order locks are not part of the direct replacement contract')
        self.pred, self.lane, self.pos = {},{},{}
        for l,q in enumerate(self.snap.lanes):
            for k,j in enumerate(q):
                self.pred[j.id]=q[k-1].id if k else None
                self.lane[j.id], self.pos[j.id]=l,k
        if self.old.keys()!=self.new.keys() or self.old.keys()!=self.lane.keys():
            raise ValueError('endpoints must cover the same live vehicles')
        # Give equality one canonical representation in the authorization ledger.
        self.new={i:(s if abs(s-self.old[i])>EPS else self.old[i]) for i,s in self.new.items()}
        for mapping in (self.old,self.new):
            errors=validate_plan(self.snap,Plan(sorted(mapping,key=mapping.get),mapping,0.),tol=EPS)
            if errors: raise ValueError(errors)
        if self.kin is not None and any(i not in self.heads for i,p in self.pred.items() if p is None):
            raise ValueError('physical forks require an actual profile for each active head')

    @property
    def changed(self):
        return {i for i,s in self.new.items() if s!=self.old[i]}

    def compatible(self,i,x,j,y):
        a,b=self.lane[i],self.lane[j]; H=self.snap.H
        if a==b:
            gap=y-x if self.pos[i]<self.pos[j] else x-y
            return gap>=max(H[a][a],self.snap.TL)-EPS
        return y-x>=H[a][b]-EPS or x-y>=H[b][a]-EPS

    def certified(self,i,t):
        """Sufficient for every old trajectory at reception; strict pre-entry deadline."""
        O,N=self.old[i],self.new[i]
        if t>=min(O,N)-EPS: return False
        if self.kin is None: return True
        k=self.kin; p=self.pred[i]
        if p is None:
            profiles=[self.heads[i]]
        else:
            # At launch the parked follower tracks O_i, unless it already switched.
            # An incompatible predecessor time cannot have been used while i holds O_i.
            profiles=[make_profile(z,k.L,0.,O,k) for z in {self.old[p],self.new[p]}
                      if z<=t and self.compatible(p,z,i,O)]
        for prof in profiles:
            distance,v=prof.state(t)
            if not (distance>0 and t+t_minus(distance,v,k)-EPS<=N<=t+t_plus(distance,v,k)+EPS):
                return False
        return True   # if still staged, the ledger enforces the actual future launch gap


def ancestry(parents):
    """Peel sources. Remaining vertices are precisely cycles and their descendants."""
    depth,anc={},{}
    while wave:=[i for i,ps in parents.items() if i not in depth and ps<=depth.keys()]:
        for i in wave:
            ps=parents[i]; depth[i]=1+max((depth[j] for j in ps),default=-1)
            anc[i]=set(ps).union(*(anc[j] for j in ps))
    return depth,anc


@dataclass
class Certificate:
    parents: dict[int,set]
    depth: dict[int,int]
    anc: dict[int,set]
    eligible: set[int]


def compile_update(f:Fork,d:float,offset:float=0.) -> Certificate:
    parents={i:{j for j,s in f.old.items() if j!=i and not f.compatible(i,f.new[i],j,s)} for i in f.changed}
    depth,anc=ancestry(parents); eligible=set()
    for i,k in sorted(depth.items(),key=lambda pair:pair[1]):
        if parents[i]<=eligible and f.certified(i,f.snap.t+offset+(2*k+1)*d): eligible.add(i)
    return Certificate(parents,depth,anc,eligible)


def best_closed(parents,weights):
    """Exact small-case maximum closure. Enumerate *negative vertices*, not all jobs.
    For given forbidden negatives all allowable nonnegative descendants can be included.
    Worst case exponential; a standard min cut is preferable for a large mixed-sign DAG.
    """
    depth,anc=ancestry(parents)
    if len(depth)!=len(parents): raise ValueError('closure input must be the eligible DAG')
    neg=[i for i,w in weights.items() if w<0]; best=set(); value=0.
    for k in range(len(neg)+1):
        for xs in combinations(neg,k):
            banned=set(xs)
            s={i for i in parents if i not in banned and not (banned & anc[i])}
            z=math.fsum(weights[i] for i in s)
            if z>value or (z==value and len(s)<len(best)): best,value=s,z
    return best


def select(f,c,u,v):
    return best_closed({i:c.parents[i] for i in c.eligible},
                       {i:(f.old[i]-f.new[i])*u*(u*v)**len(c.anc[i]) for i in c.eligible})
