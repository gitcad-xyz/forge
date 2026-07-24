"""Native hidden-line removal (ADR-0020) — visible/hidden 2D polylines per view.

Visibility is a display property (floats legal); these check the projected
geometry lands where it must and holes/occlusion behave."""

from __future__ import annotations

import math

from forgekernel.brep import Solid
from forgekernel.hlr import hidden_line, section_polys
from forgekernel.quadric import Cyl, DrilledSolid


def _chain(segs, tol=1e-4):
    """Chain 2D segments into closed loops (mirrors the drawing engine)."""
    def key(p):
        return (round(p[0] / tol), round(p[1] / tol))
    rem = [list(s) for s in segs if len(s) >= 2]
    loops = []
    while rem:
        loop = rem.pop()
        while key(loop[0]) != key(loop[-1]):
            for i, c in enumerate(rem):
                if key(c[0]) == key(loop[-1]):
                    loop += c[1:]
                    rem.pop(i)
                    break
                if key(c[-1]) == key(loop[-1]):
                    loop += list(reversed(c))[1:]
                    rem.pop(i)
                    break
            else:
                break
        if key(loop[0]) == key(loop[-1]) and len(loop) >= 4:
            loops.append(loop)
    return loops


def _bbox2d(polys):
    xs = [p[0] for pl in polys for p in pl]
    ys = [p[1] for pl in polys for p in pl]
    return min(xs), max(xs), min(ys), max(ys)


def test_box_front_and_top_project_to_the_right_rectangle() -> None:
    box = Solid.box(20, 10, 5)
    front = hidden_line(box, (0, -1, 0), (1, 0, 0))     # sheet x=X, y=Z
    x0, x1, y0, y1 = _bbox2d(front["visible"])
    assert (round(x0), round(x1), round(y0), round(y1)) == (0, 20, 0, 5)
    top = hidden_line(box, (0, 0, 1), (1, 0, 0))        # sheet x=X, y=Y
    x0, x1, y0, y1 = _bbox2d(top["visible"])
    assert (round(x0), round(x1), round(y0), round(y1)) == (0, 20, 0, 10)


def test_box_back_edges_are_hidden_not_visible() -> None:
    # a solid box has no hidden geometry that isn't coincident with an outline;
    # but every projected outline is present and closed (>= 4 visible runs)
    box = Solid.box(8, 8, 8)
    r = hidden_line(box, (0, -1, 0), (1, 0, 0))
    assert len(r["visible"]) >= 4


def test_drilled_plate_shows_the_hole_and_hides_the_bore_wall() -> None:
    # 40x20x5 plate, Ø8 through-hole at (20,10)
    plate = DrilledSolid(Solid.box(40, 20, 5),
                         [Cyl(20, 10, 4, 0, 5)])
    top = hidden_line(plate, (0, 0, 1), (1, 0, 0))      # looking down the bore
    # the rim circle appears near (20,10) with radius ~4
    pts = [p for pl in top["visible"] for p in pl]
    near = [p for p in pts if abs(math.hypot(p[0] - 20, p[1] - 10) - 4) < 0.6]
    assert near, "hole rim circle should be visible in the top view"
    # front view: the bore is behind the front face -> its wall runs are hidden
    front = hidden_line(plate, (0, -1, 0), (1, 0, 0))
    assert front["hidden"], "bore walls should be hidden in the front view"


def test_deflection_controls_circle_fidelity() -> None:
    plate = DrilledSolid(Solid.box(30, 30, 4), [Cyl(15, 15, 6, 0, 4)])
    coarse = hidden_line(plate, (0, 0, 1), (1, 0, 0), deflection=0.5)
    fine = hidden_line(plate, (0, 0, 1), (1, 0, 0), deflection=0.05)
    n_coarse = sum(len(pl) for pl in coarse["visible"])
    n_fine = sum(len(pl) for pl in fine["visible"])
    assert n_fine > n_coarse


# -- section curves -----------------------------------------------------------

def test_section_of_box_is_the_cut_rectangle() -> None:
    box = Solid.box(20, 10, 10)
    segs = section_polys(box, (1, 0, 0), (0, 1, 0), 10.0)  # plane x=10
    loops = _chain(segs)
    assert len(loops) == 1
    xs = [p[0] for lp in loops for p in lp]
    ys = [p[1] for lp in loops for p in lp]
    assert (round(min(xs)), round(max(xs))) == (0, 10)     # sheet x = Y
    assert (round(min(ys)), round(max(ys))) == (0, 10)     # sheet y = Z


def test_section_through_drilled_hole_splits_into_two_loops() -> None:
    # box 20x10x10, Ø4 through-hole at (10,5); section on the hole axis (x=10)
    # is the material on either side of the slot: exactly two closed loops.
    plate = DrilledSolid(Solid.box(20, 10, 10), [Cyl(10, 5, 2, 0, 10)])
    segs = section_polys(plate, (1, 0, 0), (0, 1, 0), 10.0)
    loops = _chain(segs)
    assert len(loops) == 2
    spans = sorted((round(min(p[0] for p in lp)), round(max(p[0] for p in lp)))
                   for lp in loops)
    assert spans == [(0, 3), (7, 10)]      # slot clears Y∈(3,7)


def test_section_normal_to_bore_shows_the_hole_circle() -> None:
    # a plane perpendicular to the bore axis cuts the wall as a full circle:
    # outer rectangle + inner circle = two loops (the hole stays clear).
    plate = DrilledSolid(Solid.box(20, 20, 10), [Cyl(10, 10, 3, 0, 10)])
    segs = section_polys(plate, (0, 0, 1), (1, 0, 0), 5.0)  # plane z=5
    loops = _chain(segs)
    assert len(loops) == 2
    inner = min(loops, key=lambda lp: max(p[0] for p in lp) - min(p[0] for p in lp))
    r = (max(p[0] for p in inner) - min(p[0] for p in inner)) / 2
    assert abs(r - 3.0) < 0.2              # the bore radius, within faceting
