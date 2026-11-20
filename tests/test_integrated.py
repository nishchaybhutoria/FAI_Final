"""Small exact references independent of the target scorer and bridge event-graph solver."""
from importlib.util import find_spec
from importlib import import_module
from itertools import product
import numpy as np
import pytest
from lockorder.scheduling.core import Job,Snapshot,Plan,validate_plan,solve_dp
from lockorder.replacement import Fork,Channel,execute
from lockorder.models.physics import Kinematics,launch_time,make_profile


def mod(name):
    path='lockorder.replacement.'+name
    assert find_spec(path) is not None, f'{name} contract not available in the direct fixed-target transport'
    return import_module(path)


def fixture(O=(10.,14.),N=(12.,9.),floors=(10.,9.),latest=None,lanes=(0,1),kin=None,h=3.):
    q=([],[]);heads={};old=dict(enumerate(O));latest=latest or (np.inf,)*len(O)
    for i,l in enumerate(lanes):
        if not q[l] and kin:heads[i]=make_profile(0.,kin.L,0.,old[i],kin)
        q[l].append(Job(i,l,floors[i],earliest=0.,latest=latest[i]))
    return Fork(Snapshot(0.,q,((h,h),(h,h)),launch_time(kin) if kin else 0.),old,dict(enumerate(N)),kin,heads,4)


def test_nominal_optimum_can_have_zero_value_while_old_order_retiming_has_value():
    m=mod('targets');f=fixture();link=Channel(.1,.9,.9)
    assert solve_dp(f.snap).times==f.new
    opt=m.options(f);name,scores=m.rank(opt,link)
    assert scores['nominal']==0 and scores[name]==pytest.approx(.9)
    assert opt[name].new=={0:10.,1:13.} and f.new=={0:12.,1:9.}


def test_deliberate_boundary_wait_can_beat_both_earliest_orders_and_matches_all_delivery_histories():
    m=mod('targets');f=fixture((12,21),(9,12),(9,0),lanes=(0,0));ln=Channel(.1,.75,.75)
    opt=m.options(f);k,z=m.rank(opt,ln);assert opt[k].new=={0:9,1:15} and z[k]>z['nominal']
    for name,T in opt.items():
        total=0.
        for bits in product((False,True),repeat=4):
            tape=np.zeros((2,8));p=1.
            for (i,leg),bit in zip(product((0,1),(0,1)),bits):
                tape[i,leg]=0 if bit else .99;p*=.75 if bit else .25
            total+=p*execute(T,'scoped',ln,tape).metrics['gain']
        assert z[name]==pytest.approx(total)


def test_one_legal_temporary_equality_point_can_unlock_exactly_the_full_plan():
    b=mod('bridge');f=fixture(latest=(12,15),kin=Kinematics())
    path=b.complete(f);assert path==[(1,15),(0,12),(1,9)]
    assert b.replay(f,path)==f.new
    assert b.complete(fixture(latest=(12,14),kin=Kinematics())) is None
    with pytest.raises((AssertionError,ValueError)):
        b.replay(f,[(1,9),(0,12)]) # no early use of a still-occupied permission


def test_tight_pair_bound_over_many_non_earliest_and_nominally_worse_targets():
    from dataclasses import replace
    m=mod('targets');b=mod('bridge');link=Channel(0,.8,.7)
    for s in (0.,.5,1.5):
        f=fixture((10.,13.+s),(12.,9.));vals=[]
        for a in np.arange(10.,20.,.5):
            for x in np.arange(9.,21.,.5):
                if abs(a-x)>=3:vals.append(m.score(replace(f,new={0:a,1:x}),link))
        assert max(vals)==pytest.approx(link.u*s)
        assert b.replay(f,b.complete(f))==f.new


def test_narrow_physical_window_removes_otherwise_available_buffer():
    from dataclasses import replace
    b=mod('bridge'); bare=fixture((.422,.462),(.462,.422),(0,0),h=.04)
    assert b.complete(bare) is not None
    kin=Kinematics();profiles={i:make_profile(0,2.5,6.,s,kin) for i,s in bare.old.items()}
    moving=replace(bare,kin=kin,heads=profiles)
    assert all(moving.certified(i,0) for i in bare.old) and b.complete(moving) is None


def test_one_temporary_move_is_not_one_reusable_buffer_for_two_disjoint_cycles():
    b=mod('bridge');f=fixture((10,14,20,24),(12,9,22,19),(10,9,20,19),lanes=(0,1,0,1))
    assert b.complete(f) is None
    p=b.best(f);s=b.replay(f,p.path);assert p.exact and sum(f.old.values())-sum(s.values())==pytest.approx(3.)


def explicit(f):
    # Independent state search, INTEGER source times/windows and clearance => every
    # cell has endpoint or half-integer witness. Finite latest=18 in these cases.
    O=tuple(f.old.values());N=tuple(f.new.values());ids=list(f.old);seen={(O,False)};todo=list(seen)
    def ok(S):
        times=dict(zip(ids,S));return not validate_plan(f.snap,Plan(sorted(ids,key=times.get),times,0.))
    while todo:
        s,used=todo.pop()
        if s==N:return True
        for i in ids:
            if O[i]==N[i] or s[i]==N[i]:continue
            opts=[(N[i],used)]
            if not used:opts += [(float(x)/2,True) for x in range(2,37) if x/2 not in (O[i],N[i])]
            for val,flag in opts:
                q=s[:i]+(val,)+s[i+1:];node=(q,flag)
                if node not in seen and ok(q):seen.add(node);todo.append(node)
    return False


def test_continuous_bridge_characterization_agrees_with_enumeration_of_intermediate_states():
    b=mod('bridge');rng=np.random.default_rng(8921)
    for _ in range(65):
        O=tuple(sorted(rng.choice([4,7,10,13,16],3,replace=False)));n=sorted(rng.choice([4,7,10,13,16],3,replace=False));j=rng.integers(3);second=n.pop(j);N=(n[0],second,n[1]);f=fixture(O,N,(2,2,2),(18,18,18),(0,1,0),h=3)
        path=b.complete(f);assert (path is not None)==explicit(f)
        if path is not None:assert b.replay(f,path)==f.new
