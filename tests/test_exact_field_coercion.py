"""`F()` admits wider exact fields, and still refuses to wave a float through.

This is the single coercion point every coordinate in the canonical B-rep goes
through, so it is also the single place exactness could be lost. It used to
force everything to `Fraction`, which made whole families of solid
inexpressible: any shape whose faces meet at an IRRATIONAL height — a napkin
ring's bore circles at z = ±√(R²−r²), a shelled cone's offset profile, a
torus-eroded blind bore — could not be built at all, because the height is not
in ℚ.

The fix widens the DOMAIN without dropping the CHECK, and those are different
things. A value already exact in ℚ[√d] or ℚ[√d][π] passes through untouched; a
float still converts to its exact binary Fraction rather than being admitted as
itself. If that second half ever regresses, the charter's central rule is gone
and nothing else would notice — hence this file.
"""

from fractions import Fraction

import pytest

from forgekernel.exact import F
from forgekernel.polypi import PiPoly
from forgekernel.quadric import PiVal
from forgekernel.surd import SurdVal


def test_rationals_are_still_normalised_to_Fraction() -> None:
    for x in (3, -7, Fraction(2, 3)):
        assert isinstance(F(x), Fraction)


def test_a_float_is_still_converted_never_admitted_as_itself() -> None:
    """The half that must never regress. A float reaching the b-rep as a float
    is exactly what ADR-0019 forbids; converting it to its exact binary value
    keeps every later decision exact relative to the given input."""
    got = F(0.5)
    assert isinstance(got, Fraction) and got == Fraction(1, 2)
    assert not isinstance(F(0.1), float)
    assert F(0.1) == Fraction(0.1)          # exact binary value, not 1/10


WIDER = [
    ("Q[sqrt d]", SurdVal(0, 1, 35)),
    ("Q[pi]", PiVal(0, Fraction(1))),
    ("Q[sqrt d][pi]", PiPoly([0, SurdVal(0, 1, 35)])),
]


@pytest.mark.parametrize("label,value", WIDER, ids=[w[0] for w in WIDER])
def test_wider_exact_fields_pass_through_untouched(label, value) -> None:
    """Coercing these to Fraction would either raise or, worse, lose the
    radical — a silently wrong coordinate."""
    assert F(value) is value


def test_an_irrational_height_can_now_reach_the_canonical_brep() -> None:
    """The concrete thing this unlocks: a circle at z = √35, which is where a
    napkin ring's bore meets its sphere. Before, this raised
    `TypeError: argument should be a string or a Rational instance`."""
    from forgekernel import body as B

    h = SurdVal(0, 1, 35)
    circle = B._circle_at(F(0), F(0), h, F(1))
    assert circle.c[2] == h
    assert circle.r == 1
