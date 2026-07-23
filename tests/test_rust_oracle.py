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


# -- extended ops: prism / scale / mirror / rotate / chamfer / prismatoid -----

def _rr(x):
    return f"{Fr(x).numerator}/{Fr(x).denominator}"


def _canon_rs(pysolid):
    return pysolid.canonical()


def test_prism_identical() -> None:
    L = [(0, 0), (30, 0), (30, 8), (8, 8), (8, 25), (0, 25)]
    ref = Solid.prism(L, 12, "prism")
    rp = rs.make_prism([_rr(c) for xy in L for c in xy], "12/1", "prism")
    assert rp.volume6_str() == _vol6(ref)
    assert rp.canonical() == _canon(ref)


def test_scale_mirror_rotate_identical() -> None:
    base = Solid.box(10, 20, 30, "box").translated((Fr(1), Fr(2), Fr(3)))
    rb = rs.make_box("10/1", "20/1", "30/1", "box").translate("1/1", "2/1", "3/1")
    assert rb.scale("2/1", "1/1", "1/1").canonical() == _canon(base.scaled(Fr(2), Fr(1), Fr(1)))
    assert rb.mirror(0).canonical() == _canon(base.mirrored("x"))
    assert rb.rotate_quarter(2, 1).canonical() == _canon(base.rotated_quarter("z", 1))


def test_chamfer_identical() -> None:
    from forgekernel.brep import chamfer_corners, chamfer_planar, logical_edges

    b = Solid.box(30, 20, 10, "box")
    e = logical_edges(b)
    ref = chamfer_corners(chamfer_planar(b, 2, e), 2, e)
    rp = rs.make_box("30/1", "20/1", "10/1", "box").chamfer("2/1")
    assert rp.volume6_str() == _vol6(ref)
    assert rp.canonical() == _canon(ref)


def test_prismatoid_identical() -> None:
    from forgekernel.brep import prismatoid

    bot = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    top = [(-4, -4), (4, -4), (4, 4), (-4, 4)]
    ref = prismatoid(bot, 0, top, 25)
    rp = rs.make_prismatoid([_rr(c) for xy in bot for c in xy], "0/1",
                            [_rr(c) for xy in top for c in xy], "25/1", "prismatoid")
    assert rp.volume6_str() == _vol6(ref)
    assert rp.canonical() == _canon(ref)
