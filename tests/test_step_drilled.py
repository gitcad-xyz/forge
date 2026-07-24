"""STEP export of drilled solids — exact cylindrical holes, not facets.

A hole is emitted as a real CYLINDRICAL_SURFACE with CIRCLE edges, so any CAD
system reads it back as a hole (with a diameter you can query) rather than a
polygon soup. The oracle here is topological: in a valid MANIFOLD_SOLID_BREP
every edge is used exactly twice, once in each direction — an orientation slip
anywhere in the caps, walls or blind disks breaks that immediately.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from forgekernel.brep import Solid
from forgekernel.quadric import Cyl, DrilledSolid
from forgekernel.stepio import write_step_planar_solid


def _entities(text):
    ent = {}
    for m in re.finditer(r"^#(\d+) = ([A-Z0-9_]+)\((.*)\);$", text, re.M):
        ent[int(m.group(1))] = (m.group(2), m.group(3))
    return ent


def _refs(s):
    return [int(x) for x in re.findall(r"#(\d+)", s)]


def _edge_uses(text):
    """edge id -> list of traversal senses across every face loop."""
    ent = _entities(text)
    use = defaultdict(list)
    for fid, (t, args) in ent.items():
        if t != "ADVANCED_FACE":
            continue
        for b in _refs(args):
            if ent.get(b, ("", ""))[0] not in ("FACE_OUTER_BOUND", "FACE_BOUND"):
                continue
            for lp in _refs(ent[b][1]):
                if ent.get(lp, ("", ""))[0] != "EDGE_LOOP":
                    continue
                for oe in _refs(ent[lp][1]):
                    if ent.get(oe, ("", ""))[0] != "ORIENTED_EDGE":
                        continue
                    use[_refs(ent[oe][1])[0]].append(
                        ".T." in ent[oe][1].split(",")[-1])
    return use


def _assert_manifold(text):
    use = _edge_uses(text)
    assert use, "no edges found"
    for eid, senses in use.items():
        assert len(senses) == 2, f"edge #{eid} used {len(senses)}x, want 2"
        assert senses[0] != senses[1], f"edge #{eid} traversed the same way twice"


def _assert_refs_resolve(text):
    ids = {int(m) for m in re.findall(r"^#(\d+) =", text, re.M)}
    used = {int(m) for m in re.findall(r"#(\d+)", text)}
    assert not (used - ids), f"dangling refs: {sorted(used - ids)[:5]}"


CASES = {
    "through hole": DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)]),
    "blind from top": DrilledSolid(Solid.box(30, 30, 10), [Cyl(15, 15, 3, 4, 10)]),
    "blind from bottom": DrilledSolid(Solid.box(30, 30, 10), [Cyl(15, 15, 3, 0, 6)]),
    "two holes": DrilledSolid(Solid.box(60, 30, 5),
                              [Cyl(15, 15, 3, 0, 5), Cyl(45, 15, 3, 0, 5)]),
    "counterbore": DrilledSolid(Solid.box(30, 30, 10),
                                [Cyl(15, 15, 2, 0, 10), Cyl(15, 15, 4, 7, 10)]),
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_drilled_step_is_a_closed_oriented_brep(label) -> None:
    solid = CASES[label]
    text = write_step_planar_solid(solid.base, bores=solid.bores)
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    _assert_refs_resolve(text)
    _assert_manifold(text)


def test_a_hole_is_an_exact_cylinder_not_facets() -> None:
    solid = DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])
    text = write_step_planar_solid(solid.base, bores=solid.bores)
    # one analytic cylinder of the right radius on the right axis...
    cyls = re.findall(r"CYLINDRICAL_SURFACE\('',#(\d+),([0-9.]+)\)", text)
    assert len(cyls) == 1 and float(cyls[0][1]) == pytest.approx(4.0)
    # ...two rim circles, same radius
    circles = re.findall(r"CIRCLE\('',#(\d+),([0-9.]+)\)", text)
    assert len(circles) == 2
    assert all(float(r) == pytest.approx(4.0) for _, r in circles)
    # ...and the axis really is at the bore centre
    ent = _entities(text)
    place = ent[int(cyls[0][0])][1]
    origin = ent[_refs(place)[0]][1]
    xs = [float(v) for v in re.findall(r"-?[0-9.]+", origin)]
    assert xs[0] == pytest.approx(20.0) and xs[1] == pytest.approx(10.0)
    # the caps carry the hole as an INNER bound (FACE_BOUND), not an outer one
    assert len(re.findall(r"= FACE_BOUND\(", text)) == 2


def test_planar_export_is_unchanged_by_the_bore_support() -> None:
    box = Solid.box(20, 20, 5)
    assert write_step_planar_solid(box) == write_step_planar_solid(box, bores=[])
    _assert_manifold(write_step_planar_solid(box))
