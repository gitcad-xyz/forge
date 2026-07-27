"""Coincident-contact booleans must refuse structured, never crash.

Two lofts sharing an exactly coincident wall (A's x=2 face IS B's x=2
face, opposite outward normals) drive SSI cell resolution into the
border-pinning last resort with ALL FOUR parameters hugging a domain
border: a corner cell of A's wall pairs with a corner cell of B's.
``_refine_constrained`` used to raise a raw
``ValueError("all four parameters fixed — nothing to solve")`` that
escaped every structured except-list (released 0.9.8 wheels crash,
chip task_f68af885).

The fully-pinned candidate is not an error: the point EXISTS — there is
nothing left to solve, only the exact residual certificate to check
(the same idiom as the matched-corner probes, 0 Newton iterations).
With that evaluation in place the coincident family lands in the
existing structured refusal (:class:`SsiCellUncertified` — tangential
contact is genuinely uncertifiable by subdivision, which never prunes
coincident surfaces).

These tests pin the CONTRACT: structured refusal by name, never a bare
ValueError. They do not pin a merge — union/cut of coincident-contact
operands stays refused until the coincident-face merge lands.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import boolean_trimmed
from forgekernel.loft import LoftSolid
from forgekernel.nurbs import bezier_surface
from forgekernel.ssi import SsiCellUncertified, _refine_constrained


def _sq(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _coincident_lofts():
    """A: x∈[0,2], B: x∈[2,4], identical z stack — shared wall x=2."""
    a = LoftSolid([(_sq(0, 2, 0, 2), 0), (_sq(0, 2, 0, 2), 1),
                   (_sq(0, 2, 0, 2), 2)])
    b = LoftSolid([(_sq(2, 4, 0, 2), 0), (_sq(2, 4, 0, 2), 1),
                   (_sq(2, 4, 0, 2), 2)])
    return a, b


@pytest.mark.parametrize("op", ["union", "cut"])
def test_coincident_wall_refuses_structured_never_raw_valueerror(op):
    a, b = _coincident_lofts()
    with pytest.raises(SsiCellUncertified):
        boolean_trimmed(op, a.to_patches(), b.to_patches(), depth=4)


@pytest.mark.parametrize("op", ["union", "cut"])
def test_coincident_wall_never_escapes_as_bare_valueerror(op):
    """The zero-crash rule, stated directly: whatever the refusal type,
    it must be one of the structured (subclassed) refusals — a bare
    ``ValueError`` is a crash escaping the kernel's contract."""
    a, b = _coincident_lofts()
    try:
        boolean_trimmed(op, a.to_patches(), b.to_patches(), depth=4)
    except ValueError as exc:
        assert type(exc) is not ValueError, (
            f"raw ValueError escaped boolean_trimmed({op!r}): {exc}")


def test_refine_constrained_fully_pinned_point_is_evaluated_not_raised():
    """All four parameters fixed = a fully-determined candidate: the
    exact residual certificate decides. Coincident corner → ok (residual
    exactly 0); separated corner → not ok, with the true residual."""
    sq = [[(F(0), F(0), F(0)), (F(0), F(1), F(0))],
          [(F(1), F(0), F(0)), (F(1), F(1), F(0))]]
    a = bezier_surface(sq)
    b = bezier_surface(sq)                      # exactly coincident
    pt = (F(0), F(0), F(0), F(0))
    fixed = {0: F(0), 1: F(0), 2: F(0), 3: F(0)}
    out, ok, res2 = _refine_constrained(a, b, pt, fixed)
    assert out == (F(0), F(0), F(0), F(0))
    assert ok
    assert res2 == 0

    far = [[(F(5), F(0), F(0)), (F(5), F(1), F(0))],
           [(F(6), F(0), F(0)), (F(6), F(1), F(0))]]
    c = bezier_surface(far)
    out, ok, res2 = _refine_constrained(a, c, pt, fixed)
    assert out == (F(0), F(0), F(0), F(0))
    assert not ok
    assert res2 == 25                           # |(0,0,0)-(5,0,0)|² exactly
