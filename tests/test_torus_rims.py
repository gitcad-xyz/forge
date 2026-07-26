"""#130: a lathe's Torus faces must carry their boundary rims as loops.

``lathe_body`` emitted every arc segment as ``Face(Torus(...), (), True)`` —
zero loops. The volume term never noticed (a torus sector's sweep is carried
on the SURFACE, k0/span), but the topology audit was blind: the two rim
circles where the fillet meets its neighbours were each used by ONE face
(the neighbour), so ``manifold_violations`` reported every filleted lathe as
open, and the answer audit had to carry a blanket exemption for any body
containing a torus — which exempted exactly the bodies most likely to be
hand-assembled wrong.
"""
from fractions import Fraction as F

from forgekernel import body as B
from forgekernel.polypi import PiPoly


def _filleted_cylinder() -> B.Body:
    """d=10 x 12 cylinder, both rims rounded at 1 mm — two quarter tori."""
    segs = [("line", (F(0), F(0)), (F(4), F(0))),
            ("arc", (F(4), F(1)), (F(4), F(0)), (F(5), F(1))),
            ("line", (F(5), F(1)), (F(5), F(11))),
            ("arc", (F(4), F(11)), (F(5), F(11)), (F(4), F(12))),
            ("line", (F(4), F(12)), (F(0), F(12)))]
    return B.lathe_body(segs, F(0), F(0))


def test_torus_faces_carry_their_rims() -> None:
    body = _filleted_cylinder()
    tori = [f for f in body.faces if isinstance(f.surface, B.Torus)]
    assert len(tori) == 2
    for f in tori:
        assert len(f.loops) == 1
        edges = f.loops[0].edges
        assert len(edges) == 2
        assert all(isinstance(e.curve, B.Circle) and e.v0 == e.v1
                   for e in edges)
    # the rims (4@0, 5@1, 5@11, 4@12) previously showed used-by-1
    assert B.manifold_violations(body) == []


def test_rims_change_no_measured_number() -> None:
    """The loops are topology only: volume (Green by hand: pi*850/3 + 4pi^2),
    bbox and the mesh must not move."""
    body = _filleted_cylinder()
    assert B.volume(body) == PiPoly([0, F(850, 3), 4])
    lo, hi = B.bbox(body)
    assert (round(lo[2], 9), round(hi[2], 9)) == (0.0, 12.0)
    mesh = B.tessellate(body, 0.05)
    seen: dict = {}
    for tri in mesh["triangles"]:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            seen[(a, b)] = seen.get((a, b), 0) + 1
    unpaired = sum(1 for (a, b), n in seen.items()
                   if seen.get((b, a), 0) != n)
    assert unpaired == 0


def test_a_dome_arc_touching_the_axis_emits_one_rim() -> None:
    """An arc whose endpoint sits ON the axis has a degenerate rim of radius
    zero — a singular point on no edge (the pointed-cone precedent), so only
    the non-degenerate rim becomes a loop edge."""
    segs = [("line", (F(0), F(0)), (F(3), F(0))),
            ("line", (F(3), F(0)), (F(3), F(9))),
            ("arc", (F(0), F(9)), (F(3), F(9)), (F(0), F(12)))]
    body = B.lathe_body(segs, F(0), F(0))
    tori = [f for f in body.faces if isinstance(f.surface, B.Torus)]
    assert len(tori) == 1
    assert len(tori[0].loops) == 1
    assert len(tori[0].loops[0].edges) == 1
    assert B.manifold_violations(body) == []
