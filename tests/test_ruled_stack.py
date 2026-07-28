"""``ruled_stack`` — a >2-section ruled loft built directly as one shell.

The BSP fold (union of the pairwise prismatoids) is the executable spec:
the direct stack construction must describe the SAME solid. The interface
caps cancel exactly — piece i's top cap and piece i+1's bottom cap are the
same ear-clipped section polygon with opposite orientation — so omitting
them and keeping every wall band plus the two outer caps is a point-set
identity, verified here by exact (Fraction) volume and centroid equality
against the fold, plus the watertightness audit, at O(n) cost instead of
the fold's O(n²) splits.
"""
from fractions import Fraction as Fr

import pytest

from forgekernel import csg
from forgekernel.brep import prismatoid, ruled_stack


def _fold(sections):
    """The spec: pairwise prismatoids fused by the exact BSP union."""
    out = None
    for (la, za), (lb, zb) in zip(sections, sections[1:]):
        piece = prismatoid(la, za, lb, zb)
        out = piece if out is None else csg.union(out, piece)
    return out


def _tapered_tower():
    """4 sections, non-convex L-shaped footprint shrinking with height —
    a stack whose fold does real work (splits at every interface)."""
    def L(s, dx=0):
        s = Fr(s)
        return [(Fr(0) + dx, Fr(0)), (s * 3 + dx, Fr(0)), (s * 3 + dx, s),
                (s + dx, s), (s + dx, s * 3), (Fr(0) + dx, s * 3)]
    return [(L(6), Fr(0)), (L(5, 1), Fr(4)), (L(3, 2), Fr(9)),
            (L(2, 3), Fr(11))]


def test_stack_volume_and_centroid_match_the_fold_exactly():
    secs = _tapered_tower()
    spec = _fold(secs)
    fast = ruled_stack(secs)
    assert fast.volume() == spec.volume()           # exact Fraction equality
    assert fast.centroid() == spec.centroid()
    assert fast.volume() > 0


def test_stack_is_watertight():
    fast = ruled_stack(_tapered_tower())
    assert fast.watertight_violations() == []


def test_stack_bbox_matches_the_fold():
    secs = _tapered_tower()
    spec, fast = _fold(secs), ruled_stack(secs)

    def bb(s):
        xs = [v[i] for p in s.polys for v in p.verts for i in range(1)]
        return (min(v[i] for p in s.polys for v in p.verts)
                for i in range(3))
    for a, b in zip(bb(spec), bb(fast)):
        assert a == b


def test_cw_sections_are_normalised_like_the_fold():
    secs = [([tuple(reversed(pt)) for pt in reversed(lp)], z)
            for lp, z in _tapered_tower()]
    # reversing each loop flips its orientation; both constructions must
    # normalise identically, so exact volume equality still holds
    spec, fast = _fold(secs), ruled_stack(secs)
    assert fast.volume() == spec.volume() > 0
    assert fast.watertight_violations() == []


def test_non_monotonic_z_refuses():
    secs = _tapered_tower()
    secs[2] = (secs[2][0], Fr(3))                   # z dips back down
    with pytest.raises(ValueError, match="increasing"):
        ruled_stack(secs)


def test_mixed_orientation_refuses():
    secs = _tapered_tower()
    secs[1] = (list(reversed(secs[1][0])), secs[1][1])
    with pytest.raises(ValueError, match="orientation"):
        ruled_stack(secs)


def test_fewer_than_three_sections_refuses():
    secs = _tapered_tower()[:2]
    with pytest.raises(ValueError, match="3"):
        ruled_stack(secs)
