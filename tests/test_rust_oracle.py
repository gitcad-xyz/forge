"""W-I oracle: the Rust port (forgekernel_rs) must reproduce the Python
reference (forgekernel) EXACTLY — same exact volume AND same canonical
face-set. ref is forge's oracle exactly as OCCT is ref's. Skipped when
the Rust extension is not built (it needs a cargo toolchain)."""

from fractions import Fraction as Fr

import pytest

rs = pytest.importorskip("forgekernel_rs")

from forgekernel.brep import Solid  # noqa: E402
import forgekernel.csg as csg       # noqa: E402


def _canon(solid) -> str:
    faces = set()
    for p in solid.polys:
        vs = sorted(
            f"{v[0].numerator}/{v[0].denominator},"
            f"{v[1].numerator}/{v[1].denominator},"
            f"{v[2].numerator}/{v[2].denominator}"
            for v in p.verts)
        faces.add("|".join(vs))
    return ";".join(sorted(faces))


def _vol6(solid) -> str:
    v = solid.volume6()
    return f"{v.numerator}/{v.denominator}"


def test_box_identical() -> None:
    b = Solid.box(10, 10, 10, "box")
    rv, rc = rs.box_form("10/1", "10/1", "10/1")
    assert rv == _vol6(b)
    assert rc == _canon(b)


@pytest.mark.parametrize("op", ["union", "cut", "intersect"])
def test_boolean_identical_to_ref(op) -> None:
    A = Solid.box(10, 10, 10, "A")
    B = Solid.box(10, 10, 10, "B").translated((Fr(5), Fr(5), Fr(5)))
    ref = {"union": csg.union, "cut": csg.cut, "intersect": csg.intersect}[op](A, B)
    rv, rc = rs.box_boolean("10/1", "10/1", "10/1", "10/1", "10/1", "10/1",
                            "5/1", "5/1", "5/1", op)
    assert rv == _vol6(ref), f"{op}: rust volume6 {rv} != ref {_vol6(ref)}"
    assert rc == _canon(ref), f"{op}: rust faces differ from ref"


@pytest.mark.parametrize("tx", ["3/1", "7/2", "13/4"])
def test_boolean_identical_various_offsets(tx) -> None:
    A = Solid.box(12, 8, 6, "A")
    B = Solid.box(5, 5, 20, "B").translated((Fr(tx), Fr(2), Fr(-1)))
    ref = csg.cut(A, B)
    rv, rc = rs.box_boolean("12/1", "8/1", "6/1", "5/1", "5/1", "20/1",
                            tx, "2/1", "-1/1", "cut")
    assert rv == _vol6(ref)
    assert rc == _canon(ref)
