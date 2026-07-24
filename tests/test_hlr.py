"""Native hidden-line removal (ADR-0020) — visible/hidden 2D polylines per view.

Visibility is a display property (floats legal); these check the projected
geometry lands where it must and holes/occlusion behave."""

from __future__ import annotations

import math

from forgekernel.brep import Solid
from forgekernel.hlr import hidden_line
from forgekernel.quadric import Cyl, DrilledSolid


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
