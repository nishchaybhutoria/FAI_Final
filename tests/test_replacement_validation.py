"""Independent small-state, local-motion and common-randomness checks."""
from itertools import combinations,product
import numpy as np
import pytest
from lockorder.replacement import Channel,Drivers,Ledger,compile_update,execute
from lockorder.models.physics import Kinematics
from lockorder.scheduling.core import Plan,validate_plan
from lockorder.simulation.scenario import Scenario,SimConfig
from lockorder.replacement.sample import Sampler
from test_replacement import fork


def test_unsupported_precedence_contract_is_rejected_before_granting():
    from dataclasses import replace
    # Both endpoints satisfy the extra A<B lock. A direct, clearance-compatible
    # hybrid A=19,B=10 would violate it. The extension is for unlocked snapshots.
    f=fork((5,10),(19,22));f.snap.lanes[1][0].need=1
    with pytest.raises(ValueError,match='lock'):
        replace(f)


def test_every_hybrid_and_single_switch_reachable_set_against_independent_schedule_validator():
    rng=np.random.default_rng(1057)
    for _ in range(70):
        lanes=(0,1,0,1,0)
        def endpoint():
            q=[[0,2,4],[1,3]];s=1.;out=[0.]*5
            while any(q):
                l=int(rng.integers(2)) if all(q) else int(not q[0]);s+=rng.integers(2,6);out[q[l].pop(0)]=s
            return out
        f=fork(endpoint(),endpoint(),lanes,h=2);c=compile_update(f,0.);changes=f.changed
        visited={frozenset()}
        for k in range(len(changes)+1):
            for item in combinations(changes,k):
                S=frozenset(item);times=f.old|{i:f.new[i] for i in S}
                safe=not validate_plan(f.snap,Plan(sorted(times,key=times.get),times,0))
                assert safe==all(c.parents[i]<=S for i in S)
                if safe and any(S-{i} in visited for i in S):visited.add(S)
                assert (S in visited)==(safe and S<=c.depth.keys())


def test_a_lost_ack_blocks_only_its_successor_and_never_forces_a_rollback():
    # Two independent promotion chains. Old times 8,11 and 30,33;
    # proximity in a connected lane queue is not by itself a dependency.
    f=fork((8,11,30,33),(6,9,28,31),lanes=(0,1,0,1));c=compile_update(f,.1)
    assert c.parents=={0:set(),1:{0},2:set(),3:{2}}
    tape=np.zeros((4,8));tape[0,1]=.99;tape[0,2]=.99;tape[2,2]=.01
    a=execute(f,'scoped',Channel(.1,1,.8),tape);b=execute(f,'wave',Channel(.1,1,.8),tape)
    assert a.entries=={0:6,1:11,2:28,3:31} and b.entries=={0:6,1:11,2:28,3:33}
    assert a.metrics['unused_old']==b.metrics['unused_old']==1


def test_follower_launches_from_its_actual_predecessor_entry_and_can_retime_while_moving():
    f=fork((7,15),(3.8,8),lanes=(0,0),kin=Kinematics());d=1.4
    assert compile_update(f,d).eligible=={0,1}
    tape=np.zeros((2,8));a=execute(f,'scoped',Channel(d,1,1),tape)
    assert a.entries=={0:3.8,1:8}
    tape[0,1]=.99;b=execute(f,'scoped',Channel(d,1,.9),tape)
    assert b.entries=={0:3.8,1:15} and b.metrics['unused_old']==1


def test_unsafe_motion_switch_is_denied_without_discarding_old_profile():
    f=fork((9.,),(3.,),lanes=(0,),kin=Kinematics());m=Ledger(f);w=Drivers(f)
    g=m.issue(0,2.);assert w.receive(g,1.5) and not w.switch(g,2.)
    w.advance(20);assert w.entered==f.old and m.possible[0]=={3,9}


def test_forks_are_keyed_replayable_and_target_never_uses_unrevealed_marks():
    cfg=SimConfig(rho=.5,sigma=1.,kin=Kinematics(),t_arr_end=15,warmup=.1,win_end=8,tie_eta=0.,audit=True)
    def first(bit):
        a=np.array([0.,.25]);lanes=np.array([0,1]);sg=np.array([[-1,bit,bit,bit],[bit,bit,bit,bit]])
        s=Sampler(Scenario(a,lanes,sg,a+cfg.lead+sg.sum(axis=1), (7319,'tune',91),cfg));assert not s.run().violations
        return s.observations[0]
    a,b,c=first(1),first(1),first(-1)
    assert a.fork.snap.t==1 and a.marks!=c.marks
    for obs in (b,c):
        assert a.fork.old==obs.fork.old and a.fork.new==obs.fork.new and np.array_equal(a.tape,obs.tape)
        assert execute(a.fork,'scoped',Channel(.1,.9,.9),a.tape).entries==execute(obs.fork,'scoped',Channel(.1,.9,.9),obs.tape).entries
