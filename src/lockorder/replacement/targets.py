"""Rank public endpoints by what the existing scoped transport can actually install."""
from dataclasses import replace
from itertools import combinations
from math import fsum

from ..scheduling.core import forward_times
from .model import EPS, compile_update, select


def options(f):
    order=sorted(f.old,key=f.old.get)
    retime=forward_times(f.snap,order)
    if retime is None:raise AssertionError('incumbent order was feasible')
    opts={'nominal':f,'old_order':replace(f,new=retime.times),'retain':replace(f,new=f.old)}
    seen={tuple(x.new[i] for i in sorted(f.old)) for x in opts.values()}
    for label in ('nominal','old_order'):
        base=opts[label];seq=sorted(base.new,key=base.new.get)
        for pos,i in enumerate(seq):
            for j in seq[:pos]:
                if f.compatible(i,base.new[i],j,f.old[j]):continue
                a,b=f.lane[j],f.lane[i];gap=f.snap.H[a][b]
                if a==b:gap=max(gap,f.snap.TL)
                bound=f.old[j]+gap
                lanes=tuple([replace(job,floor=max(bound,job.floor)) if job.id==i else job for job in q] for q in f.snap.lanes)
                p=forward_times(replace(f.snap,lanes=lanes),seq)
                if p is None:continue
                key=tuple(p.times[k] for k in sorted(f.old))
                if key not in seen:
                    opts[f'{label}:{j}>{i}']=replace(f,new=p.times);seen.add(key)
    return opts


def all_earliest(f):
    """All feasible earliest-time FIFO interleavings. Not all real-valued schedules."""
    A,B=([job.id for job in q] for q in f.snap.lanes);out={}
    for num,slots in enumerate(combinations(range(len(f.old)),len(A))):
        a,b=iter(A),iter(B);seq=[next(a) if j in slots else next(b) for j in range(len(f.old))]
        p=forward_times(f.snap,seq)
        if p is not None:out[f'order:{num}']=replace(f,new=p.times)
    return out


def score(f,link):
    c=compile_update(f,link.d);chosen=select(f,c,link.u,link.v)
    return fsum((f.old[i]-f.new[i])*link.u*(link.u*link.v)**len(c.anc[i]) for i in chosen)


def rank(opts,link):
    """Ties (within numerical EPS in vehicle-seconds) prefer input order, nominal first."""
    cache={};scores={}
    for k,f in opts.items():
        key=tuple(f.new[i] for i in sorted(f.old))
        if key not in cache:cache[key]=score(f,link)
        scores[k]=cache[key]
    return next(k for k,s in scores.items() if s>=max(scores.values())-EPS),scores
