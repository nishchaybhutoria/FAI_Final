"""Fork evaluation for the regime sweep (study 5807): every installation arm at one rolling decision.

For one observed rolling decision and one channel, every arm sees the same public
snapshot, the same issued slots O and nominal optimum N, and the same packet
uniforms (archived tape for first legs, keyed stream 8 for a return leg). Private
readiness marks enter only the delay measurement, after every choice is made.

Arms
  keep       retain every issued slot
  atomic     install N at once (oracle comparator, not a protocol)
  nominal    N by release-gated direct substitutions, Eq. (4) subset (archived)
  menu       best direct target from the archived menu (archived ``rank``)
  oblivious  the perfect-link one-detour witness for N, all certified moves sent
  aware      best of the direct menu and every detour, by expected realized value
"""
from __future__ import annotations

from math import fsum

from lockorder.replacement import execute as direct_execute
from lockorder.replacement import handoff as H
from lockorder.replacement.bridge import complete
from lockorder.replacement.targets import options, rank, score

EPS = 1e-8


def temp_of(f, path):
    temps = [(i, x) for i, x in path if min(abs(x - f.old[i]), abs(x - f.new[i])) > EPS]
    assert len(temps) <= 1
    return temps[0] if temps else None


def evaluate(key, ob, links, extra=None):
    """Rows (one per link and arm) for a single fork. ``extra`` is copied into each row."""
    f = ob.fork
    ids = sorted(f.old)
    n = len(ids)
    marks = ob.marks
    tape2 = H.second_tape(key, f.epoch, len(ob.tape))
    ideal = fsum(f.old.values()) - fsum(f.new.values())
    reorder = ideal > EPS and sorted(ids, key=f.old.get) != sorted(ids, key=f.new.get)
    witness = complete(f)
    wtemp = temp_of(f, witness) if witness else None
    wh = next(h for h in H.detours(f) if h.temp == wtemp) if wtemp else None
    menu = options(f)
    base = dict(rep=key[2], epoch=f.epoch, t=f.snap.t, n=n, changed=len(f.changed), ideal=ideal / n,
                reorder=reorder, rescuable=wtemp is not None, **(extra or {}))
    rows = []

    def row(link, arm, entries, expected, label="", temp=None, stranded=False):
        saved = fsum(f.old[i] - entries[i] for i in ids)
        assert saved <= ideal + 1e-7, (arm, saved, ideal)
        assert all(entries[i] >= marks[i] - 1e-7 for i in ids), arm
        delays = [entries[i] - marks[i] for i in ids]
        rows.append(dict(base, d=link.d, u=link.u, v=link.v, arm=arm, label=label,
                         expected=expected / n, saved=saved / n, delay=fsum(delays) / n, max_delay=max(delays),
                         temp_shift=(temp[1] - f.old[temp[0]]) if temp else float("nan"), stranded=stranded))

    for link in links:
        row(link, "keep", dict(f.old), 0.)
        row(link, "atomic", dict(f.new), ideal)
        # archived direct transport for N; the move-graph executor must reproduce it exactly
        nom = direct_execute(f, "scoped", link, ob.tape).entries
        hn = H.direct(f)
        vn, cn, _ = H.value(f, hn, link)
        assert abs(vn - score(f, link)) < 1e-9
        assert H.execute(f, hn, cn, link, ob.tape, tape2).entries == nom
        row(link, "nominal", nom, vn, "nominal")
        name, scores = rank(menu, link)
        row(link, "menu", direct_execute(menu[name], "scoped", link, ob.tape).entries, scores[name], name)
        if wh is None:
            row(link, "oblivious", nom, vn, "nominal")
        else:
            ev, chosen, _ = H.eager(f, wh, link)
            r = H.execute(f, wh, chosen, link, ob.tape, tape2)
            row(link, "oblivious", r.entries, ev, wh.label, wh.temp, r.stranded)
        c = H.choose(f, link)
        r = H.execute(c.target, c.handoff, c.chosen, link, ob.tape, tape2)
        used = c.handoff.temp if c.handoff.temp and any(e[0] == c.handoff.temp[0] for e in c.chosen) else None
        row(link, "aware", r.entries, c.value, c.handoff.label, used, r.stranded)
    return rows
