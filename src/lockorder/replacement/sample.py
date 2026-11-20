"""Fork actual rolling decisions. Targets never receive future signs, true marks or tape."""
from dataclasses import dataclass
import numpy as np
from ..simulation.engine import Engine
from ..simulation.scenario import SPLITS
from ..policies.base import Rolling
from .model import Fork,EPS


@dataclass
class Observation:
    fork: Fork
    tape: np.ndarray
    # Evaluator only. Kept physically separate from scheduling/transport interfaces.
    marks: dict
    solve_ms: float


class Sampler(Engine):
    def __init__(self,sc):
        super().__init__(sc,Rolling());self.observations=[]
        if sc.cfg.tie_eta!=0:raise ValueError('fork objective must be unpenalized sum of start times')

    def decide(self,resolve):
        ids=set(self.queue[0]+self.queue[1]);old=self.incumbent;t=self.t
        # In Rolling, identical live IDs at consecutive information calls imply a pure
        # signal event (no birth or entry). No conditioning on gains or private marks.
        take=bool(resolve and ids and old and ids==old.times.keys() and self.cfg.warmup<=t<self.cfg.win_end)
        if take:snap=self.build_snapshot();heads=dict(self.profile)
        super().decide(resolve)
        if not take:return
        new=self.incumbent
        if new.approx or new.source!='dp':raise AssertionError('invalid exact rolling reference')
        O=old.times;N=new.times
        gain=sum(O.values())-sum(N.values())
        if gain<-1e-6:raise AssertionError('optimizer worse than feasible incumbent')
        if gain<=EPS:N=O   # no transport just for an equal-cost shuffle
        # Label the public signal (slot in the reference stream), not a policy's number of sends.
        reveals=[(i,int(self.krev[i])) for i in ids if self.krev[i] and
                 abs(t-(self.a[i]+self.cfg.reveal_offset+self.krev[i]*self.cfg.dq))<EPS]
        if len(reveals)!=1:raise AssertionError('study expects isolated public sign epochs')
        i,k=reveals[0];epoch=1+i*self.cfg.K+k
        f=Fork(snap,O,N,self.kin,heads,epoch)
        study,split,rep=self.sc.key
        rng=np.random.default_rng(np.random.SeedSequence([study,SPLITS[split],rep,7,epoch]))
        tape=rng.random((max(ids)+1,8))
        self.observations.append(Observation(f,tape,{i:float(self.sc.r[i]) for i in ids},self.stats['solve_ms'][-1]))
