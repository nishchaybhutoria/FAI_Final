"""Independent authorization/physical witnesses, not assertions about plotted data."""
from importlib.util import find_spec
from itertools import combinations, product
import math
import numpy as np
import pytest
from lockorder.scheduling.core import Job, Snapshot, Plan, validate_plan
from lockorder.models.physics import Kinematics, make_profile, launch_time


def api():
    assert find_spec('lockorder.replacement') is not None, 'a direct-update transport is required; the legacy plan copy is not a release'
    from lockorder import replacement as r
    return r


def fork(O=(10.,16.),N=(12.,9.),lanes=(0,1),h=3.,kin=None,t=0.,profiles=None):
    r=api(); qs=([],[])
    for i,l in enumerate(lanes):qs[l].append(Job(i,l,0.,earliest=t))
    snap=Snapshot(t,qs,((h,h),(h,h)),0. if kin is None else launch_time(kin))
    O,N=dict(enumerate(O)),dict(enumerate(N))
    if kin and profiles is None:
        profiles={q[0].id:make_profile(t,kin.L,0,O[q[0].id],kin) for q in qs if q}
    return r.Fork(snap,O,N,kin,profiles or {},epoch=7)


def test_lost_ack_does_not_retire_permission_and_stale_messages_do_not_resurrect_it():
    from dataclasses import replace
    r=api(); f=fork(); m=r.Ledger(f); w=r.Drivers(f)
    assert m.issue(1,1.) is None
    g=m.issue(0,1.); assert g and not m.ack(replace(g,epoch=6))
    assert not w.receive(replace(g,epoch=6),.4) and not w.receive(g,1.1)
    assert w.receive(g,.3) and w.active[0]==10 and m.issue(1,1.) is None
    a=w.switch(g,1.); assert a==g and w.active[0]==12
    assert m.possible[0]=={10,12} and m.issue(1,2.) is None  # lost/delayed ACK
    assert m.ack(a) and not m.ack(a) and m.possible[0]=={12}
    assert not w.receive(g,.9)  # delayed duplicate cannot roll back
    gb=m.issue(1,2.); assert gb and w.receive(gb,1.9) and w.switch(gb,2.)
    w.advance(math.inf);assert w.entered=={1:9.,0:12.}


def test_swap_has_no_first_move_even_with_perfect_links():
    r=api();f=fork((10.,12.),(12.,10.),h=2.)
    c=r.compile_update(f,0.); assert c.parents=={0:{1},1:{0}} and not c.depth
    for pol in ['eager','scoped','wave','global','one_wave']:
        out=r.execute(f,pol,r.Channel(0,1,1),np.zeros((2,8)))
        assert out.entries==f.old
    assert r.execute(f,'atomic',r.Channel(0,1,1),np.zeros((2,8))).entries==f.new


def test_same_lane_mixed_slots_can_be_separated_but_in_reverse_fifo_order():
    r=api();f=fork((4.,8.),(12.,16.),lanes=(0,0)); c=r.compile_update(f,0.)
    assert c.parents[0]=={1} and c.parents[1]==set()


def test_probability_threshold_is_about_relinquishment_not_receipt_of_own_ack():
    r=api(); f=fork(); signs=[]
    for u,v,want in [(.5,.5,set()),(.9,.9,{0,1})]:
        c=r.compile_update(f,.2); assert r.select(f,c,u,v)==want
        total=0.
        for downA,ackA,downB in product([0,1],repeat=3):
            tape=np.zeros((2,8));tape[0,0]=0 if downA else .999;tape[0,1]=0 if ackA else .999;tape[1,0]=0 if downB else .999
            out=r.execute(f,'eager',r.Channel(.2,u,v),tape)
            mass=(u if downA else 1-u)*(v if ackA else 1-v)*(u if downB else 1-u)
            total+=mass*(26-sum(out.entries.values()))
        assert total==pytest.approx(u*(-2+7*u*v));signs.append(np.sign(total))
    assert signs==[-1,1]


def test_coordinate_only_realizable_physical_switch_at_delivery_not_at_proposal():
    r=api();kin=Kinematics();f=fork((9.,),(3.,),lanes=(0,),kin=kin)
    assert r.compile_update(f,0.).eligible=={0}
    # Still before both slots but cannot promote after two seconds tracking 9.
    assert r.compile_update(f,2.).eligible==set()
    out=r.execute(f,'eager',r.Channel(2.,1,1),np.zeros((1,8))); assert out.entries=={0:9.}


def test_distinct_ancestors_and_best_closure_against_bruteforce():
    r=api();rng=np.random.default_rng(252)
    for _ in range(120):
        n=6; parents={i:{j for j in range(i) if rng.random()<.38} for i in range(n)}
        weight=dict(enumerate(rng.normal(size=n))); S=r.best_closed(parents,weight)
        feasible=[set(c) for k in range(n+1) for c in combinations(range(n),k) if all(parents[i]<=set(c) for i in c)]
        assert S in feasible and sum(weight[i] for i in S)==pytest.approx(max(sum(weight[i] for i in q) for q in feasible))
    assert r.ancestry({0:set(),1:{0},2:{0},3:{1,2}})[1][3]=={0,1,2}
