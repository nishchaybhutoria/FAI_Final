"""One nonce per vehicle/fork, non-Byzantine messages. No release on timeout."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import time

from ..models.physics import InfeasibleArrival,make_profile
from ..scheduling.core import Plan,validate_plan
from .model import EPS,Fork,compile_update,select


@dataclass(frozen=True)
class Grant:
    epoch: int
    i: int
    base: float
    target: float
    at: float


class Ledger:
    """Manager information, independent of driver state or delivery draws."""
    def __init__(self,f):
        self.f=f; self.possible={i:{s} for i,s in f.old.items()}; self.issued={};self.released=set()

    def issue(self,i,at):
        f=self.f; s=f.new[i]
        if i in self.issued or f.old[i]==s: return None
        if not all(f.compatible(i,s,j,y) for j,ys in self.possible.items() if j!=i for y in ys): return None
        g=Grant(f.epoch,i,f.old[i],s,at);self.issued[i]=g;self.possible[i].add(s)
        return g

    def ack(self,g):
        if self.issued.get(g.i)!=g or g.i in self.released: return False
        self.possible[g.i]={g.target};self.released.add(g.i);return True


class Drivers:
    """Local actions and actual entries; a buffered grant is not yet an ACK."""
    def __init__(self,f:Fork):
        self.f=f;self.active=dict(f.old); self.version={i:f.epoch-1 for i in f.old}; self.inbox={}
        self.q=[[j.id for j in q] for q in f.snap.lanes]; self.last=list(f.snap.last)
        self.profiles=dict(f.heads); self.entered={};self.guards={j.id:j.floor for j in f.snap.jobs()}

    def receive(self,g,now):
        i=g.i
        if (i not in self.active or g.epoch!=self.f.epoch or self.version[i]!=g.epoch-1
                or g.base!=self.active[i] or i in self.entered or now>g.at+EPS):return False
        if i in self.inbox: return self.inbox[i]==g
        self.inbox[i]=g;return True

    def _retime(self,i,s,t):
        if self.f.kin and self.q[self.f.lane[i]][0]==i:
            p,v=self.profiles[i].state(t)
            self.profiles[i]=make_profile(t,p,v,s,self.f.kin)
        self.active[i]=s

    def switch(self,g,now):
        i=g.i
        if self.inbox.get(i)!=g or abs(now-g.at)>EPS or i in self.entered: return None
        del self.inbox[i]
        if now>=min(self.active[i],g.target)-EPS or self.version[i]!=g.epoch-1: return None
        try: self._retime(i,g.target,now)
        except InfeasibleArrival: return None   # keep guaranteed old profile
        self.version[i]=g.epoch;return g

    def advance(self,until):
        while self.q[0] or self.q[1]:
            t,i,l=min((self.active[q[0]],q[0],l) for l,q in enumerate(self.q) if q)
            if t>until: break
            f=self.f
            if (t<self.guards[i]-EPS or any(t<self.last[a]+f.snap.H[a][l]-EPS for a in (0,1))):
                raise AssertionError(('actual entry',i,t,self.last))
            if f.kin:
                pr=self.profiles.pop(i)
                if abs(pr.arrival-t)>1e-7 or pr.state(t)[0]>1e-7: raise AssertionError(('missed local slot',i,t))
            self.last[l]=t;self.entered[i]=t;self.q[l].pop(0)
            if self.q[l] and f.kin:
                j=self.q[l][0];self.profiles[j]=make_profile(t,f.kin.L,0.,self.active[j],f.kin)


@dataclass(frozen=True)
class Channel:
    d: float         # one-way bound (s); grant effective at end of downstream slot
    u: float         # grant delivery probability
    v: float         # return ACK delivery probability

    def __post_init__(self):
        if self.d<0 or not 0<=self.u<=1 or not 0<=self.v<=1: raise ValueError('invalid channel')


POLICIES=('fixed','atomic','global','wave','one_wave','eager','scoped')


@dataclass
class Outcome:
    entries: dict
    metrics: dict


def execute(f:Fork,policy:str,link:Channel,tape) -> Outcome:
    """One fork. Random uniforms keyed externally by epoch, ID and leg, never by policy.
    0,1: delivery indicators; 2,3: grant/ACK latency fractions; 4,5: survey legs.
    Survey consumes one fixed round; no permission is changed by a survey.
    """
    if policy not in POLICIES:raise ValueError(policy)
    d,u,v=link.d,link.u,link.v; t0=f.snap.t
    m=Ledger(f);w=Drivers(f);changed=f.changed;offset=2*d if policy=='global' else 0.
    z=dict(changed=len(changed),cycle_blocked=0,deadline_blocked=0,max_depth=0,attempted=0,
           adopted=0,acks=0,required_acks=0,messages=0,span=0.,unused_old=0,protected_s=0.,
           eligible=0,chosen=0,expected_gain=0.,selection_us=0.,violations=0)
    switched={};returned={};needed=set()
    if changed and policy=='atomic':
        for i in changed: w._retime(i,f.new[i],t0)
        z['adopted']=len(changed)
    elif changed and policy!='fixed':
        clock=time.perf_counter_ns();c=compile_update(f,d,offset)
        z.update(cycle_blocked=len(changed)-len(c.depth),deadline_blocked=len(c.depth)-len(c.eligible),
                 max_depth=max(c.depth.values(),default=0),eligible=len(c.eligible))
        chosen=c.eligible.copy()
        if policy=='one_wave': chosen={i for i in chosen if not c.parents[i] and f.new[i]<f.old[i]}
        elif policy=='scoped': chosen=select(f,c,u,v)
        z['selection_us']=(time.perf_counter_ns()-clock)/1000
        if policy=='global':
            # Full readiness survey is nonbinding, distinct from a release. If the target cannot
            # be completely serialized even on reliable links, the global attempt is not opened.
            if chosen!=changed: chosen=set()
            elif chosen:
                z['messages']+=len(f.old)+sum(int(tape[i,4]<u) for i in f.old)
                z['span']=offset
                if not all(tape[i,4]<u and tape[i,5]<v for i in f.old): chosen=set()
        z['chosen']=len(chosen)
        if policy in ('eager','scoped','one_wave'):
            z['expected_gain']=sum((f.old[i]-f.new[i])*u*(u*v)**len(c.anc[i]) for i in chosen)
        prev=set()
        for k in sorted({c.depth[i] for i in chosen}):
            if policy in ('global','wave') and not prev<=m.released:break
            send=t0+offset+2*k*d;at=t0+offset+(2*k+1)*d;w.advance(send)
            targets=sorted(i for i in chosen if c.depth[i]==k and c.parents[i]<=m.released)
            events=[];prev=set(targets)
            for i in targets:
                g=m.issue(i,at)
                if g is None:raise AssertionError('compiled release cone disagrees with ledger')
                needed.update(c.parents[i]);z['attempted']+=1;z['messages']+=1
                # An attempt reserves both possibilities even when the packet is dropped.
                if tape[i,0]<u:
                    heapq.heappush(events,(min(at,send+d*tape[i,2]),0,i,g))
                    heapq.heappush(events,(at,1,i,g))
            if targets:z['span']=offset+(2*k+2)*d  # budget until this ACK deadline, including silent timeout
            while events:
                now,kind,i,g=heapq.heappop(events);w.advance(now)
                if kind==0:
                    if not w.receive(g,now): raise AssertionError('valid timely reception rejected')
                elif kind==1:
                    if w.switch(g,now)!=g:raise AssertionError('claimed physical certificate failed')
                    switched[i]=now;z['messages']+=1
                    if tape[i,1]<v:heapq.heappush(events,(now+d*tape[i,3],2,i,g))
                else:
                    if not m.ack(g):raise AssertionError('matching ACK rejected')
                    returned[i]=now
                if w.active[i] not in m.possible[i]:raise AssertionError('manager lost a live authorization')
        z.update(adopted=len(switched),acks=len(returned),required_acks=len(needed),
                 unused_old=sum(returned.get(i,math.inf)>=f.old[i] for i in switched),
                 protected_s=sum(min(f.old[i],returned.get(i,math.inf))-t for i,t in switched.items()))
    w.advance(math.inf)
    errs=validate_plan(f.snap,Plan(sorted(w.entered,key=w.entered.get),w.entered,0.))
    if errs:raise AssertionError(('executed hybrid failed independent validator',errs))
    z['gain']=sum(f.old[i]-s for i,s in w.entered.items())
    z['full']=int(bool(changed) and z['adopted']==len(changed));z['partial']=int(0<z['adopted']<len(changed))
    z['abandoned']=int(bool(changed) and not z['adopted']);return Outcome(w.entered,z)
