"""A wider field must not be read as its rational part (regression).

``body._as_fraction`` duck-typed a value as rational by reading ``.b``::

    b = getattr(x, "b", None)
    if b is not None and b == 0:
        return x.a

That is right for ``SurdVal`` (a + b√d) and silently WRONG for ``BiSurd``
(a + b√p + c√q + e√pq), which also carries ``.a`` and ``.b``. Any biquadratic
value whose √p coefficient happens to be zero — ``960 − 20√3`` is stored as
``b=0, c=−20`` — was read as the plain rational ``960`` and the radical was
discarded. ``body.volume`` then reported the rational part of the answer as if
it were the answer.

Found by an independent Monte-Carlo oracle on ``cut(box@45°, slab@60°)``: the
kernel said exactly 960.000, the truth is 960 − 20√3 = 925.359, and the float
integral over the kernel's OWN output polygons agreed with the truth — so the
geometry was right and only the number was wrong. Unreachable before #127,
because a pair of fields never met; promotion made this path live.

Violated invariant: no float and no truncation may decide a reported value
(ADR-0019). Losing a term is worse than refusing — a refusal is visible.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from forgekernel.bisurd import BiSurd
from forgekernel.body import _as_fraction, _pi_value
from forgekernel.surd import SurdVal


def test_a_biquadratic_value_is_not_rational():
    """The verbatim shape of the bug: b == 0 but c != 0."""
    v = BiSurd(960, 0, -20, 0, 2, 3)                  # 960 − 20√3
    assert float(v) == pytest.approx(960 - 20 * math.sqrt(3), abs=1e-9)
    assert _as_fraction(v) is None, "read as its rational part 960"


@pytest.mark.parametrize("a,b,c,e", [
    (960, 0, -20, 0),        # the found case: only √q present
    (0, 0, 0, 1),            # only √pq present
    (7, 0, 0, -3),
    (0, 1, 0, 0),            # only √p — the one shape that DID refuse before
    (5, 2, 3, 4),            # everything at once
])
def test_no_nonzero_radical_is_ever_dropped(a, b, c, e):
    assert _as_fraction(BiSurd(a, b, c, e, 2, 3)) is None


def test_a_biquadratic_value_that_really_is_rational_still_collapses():
    """The other half: a widened tag on a rational value must NOT start
    refusing, or every 45°-rotated body's rational coordinates would."""
    assert _as_fraction(BiSurd(7, 0, 0, 0, 2, 3)) == Fraction(7)
    assert _as_fraction(BiSurd(Fraction(-3, 4), 0, 0, 0, 5, 7)) == Fraction(-3, 4)


def test_the_surdval_contract_is_unchanged():
    assert _as_fraction(SurdVal(5, 0, 2)) == Fraction(5)
    assert _as_fraction(SurdVal(5, 1, 2)) is None
    assert _as_fraction(Fraction(3, 2)) == Fraction(3, 2)
    assert _as_fraction(4) == Fraction(4)
    assert _as_fraction("nope") is None


def test_a_biquadratic_volume_term_is_carried_not_refused():
    """``_pi_value`` gate-kept on ``hasattr(part, 'd')`` — the SurdVal
    radicand. BiSurd has (p, q) instead, so a term PiPoly can hold perfectly
    well was turned away. Both halves of the same duck-typing mistake."""
    v = _pi_value(BiSurd(960, 0, -20, 0, 2, 3), Fraction(0), "planar face")
    assert float(v) == pytest.approx(960 - 20 * math.sqrt(3), abs=1e-9)
