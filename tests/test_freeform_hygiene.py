"""Freeform hygiene (#96 gap 11): a freeform solid handed to the canonical
B-rep machinery must refuse BY NAME, never leak a raw AttributeError.

``tessellate(LoftSolid)`` used to die with ``'LoftSolid' object has no
attribute 'faces'`` — a crash wearing the mesher's clothes — and ``to_body``
named only the missing converter, not the stage that will bring it. Under the
charter a structured refusal is a finished answer; a raw AttributeError is
not an answer at all.
"""

import pytest

from forgekernel import body as B
from forgekernel.loft import LoftSolid

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]
IN_ = [(1, 1), (9, 1), (9, 9), (1, 9)]


def _loft() -> LoftSolid:
    return LoftSolid([(SQ, 0), (IN_, 6), (SQ, 12)])


def test_tessellate_on_a_loft_refuses_by_name() -> None:
    """A ValueError naming the representation and the stage that will bring
    its display mesh (K7's trimmed-patch tessellator) — not an
    AttributeError about ``.faces``."""
    with pytest.raises(ValueError) as e:
        B.tessellate(_loft())
    msg = str(e.value)
    assert "LoftSolid" in msg
    assert "K7" in msg
    assert "attribute" not in msg          # no leaked CPython text


def test_tessellate_on_any_non_body_refuses_by_name() -> None:
    """The guard is generic: whatever non-Body arrives, the answer names the
    type and points at to_body(), instead of crashing on ``.faces``."""
    with pytest.raises(ValueError) as e:
        B.tessellate(42)                   # type: ignore[arg-type]
    msg = str(e.value)
    assert "int" in msg
    assert "to_body" in msg


def test_to_body_on_a_loft_names_the_freeform_stage() -> None:
    """The refusal already named the type; now it must also say WHAT brings
    the capability — the K7 trimmed-patch shell is the canonical form a
    freeform solid converts to (ADR-0021)."""
    with pytest.raises(ValueError) as e:
        B.to_body(_loft())
    msg = str(e.value)
    assert "LoftSolid" in msg
    assert "K7" in msg
    assert "ADR-0021" in msg
