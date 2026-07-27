"""K7 gap 3 — classify_point_in_shell: certified ray-parity membership.

A point is inside a closed TrimmedShell iff a ray from it crosses the
boundary an odd number of times. Every ingredient of the count is
certified:

  * candidate cells come from Bernstein-hull pruning of the two plane
    forms f1, f2 that cut out the ray line (a cell whose hull excludes
    zero provably contains no ray hit);
  * the crossing parity per surviving cell is the exact topological
    degree of (f1, f2) on the cell boundary (winding number, computed by
    exact 1D Bernstein root isolation on the cell edges) — parity needs
    no transversality certificate because the local degree IS the local
    crossing parity;
  * a surviving cell that touches the face's SSI trim strip REFUSES and
    the classifier retries with a jittered ray, rather than guessing
    which side of the trim boundary the hit lies on;
  * cells behind the ray origin are pruned by the exact τ-form hull; a
    cell that cannot be certified fully in front refuses (a point ON the
    shell refuses on every ray — the honest answer).

Verified on the hand-built dome ∩ half-space cap against the exact
closed-form membership 1 < z < 24·u(1-u)·v(1-v)·... of the column.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.nurbs import bezier_surface
from forgekernel.raycast import PointClassifyUncertified, classify_point_in_shell
from forgekernel.ssi import ssi_strips, ssi_trim_loops
from forgekernel.trimshell import ShellFace, TrimmedShell, TrimVertex

DOME = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                       [(2, 0, 0), (2, 2, 6), (2, 4, 0)],
                       [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
PLANE = bezier_surface([[(0, 0, 1), (0, 4, 1)], [(4, 0, 1), (4, 4, 1)]])


def _above(u, v) -> bool:
    return DOME.eval(u, v)[2] > 1


def _cap_shell(depth: int = 5) -> TrimmedShell:
    out = ssi_trim_loops(DOME, PLANE, depth=depth)
    chain = out["loops"][0]["points"]
    a_cells, b_cells = ssi_strips(DOME, PLANE, depth=depth)
    verts = [TrimVertex() for _ in chain]
    top = ShellFace(DOME, +1, a_cells, _above)
    bot = ShellFace(PLANE, -1, b_cells, _above)
    top.add_loop([(vx, (u, v)) for vx, (u, v, _, _) in zip(verts, chain)])
    bot.add_loop([(vx, (s, t)) for vx, (_, _, s, t) in zip(verts, chain)])
    return TrimmedShell([top, bot])


SHELL = _cap_shell(depth=5)


def _z_dome(x, y):
    """Exact dome height over the (x, y) footprint: 24·u(1-u)·v(1-v)."""
    u, v = F(x) / 4, F(y) / 4
    return 24 * u * (1 - u) * v * (1 - v)


def _truth(x, y, z) -> str:
    z = F(z)
    return "in" if (1 < z < _z_dome(x, y)) else "out"


# -- membership against the exact closed form ---------------------------------

@pytest.mark.parametrize("p", [
    (F(6, 5), F(17, 10), F(11, 10)),     # inside the cap (1 dome hit)
    (F(6, 5), F(17, 10), F(1, 2)),       # below the plane (2 hits: even)
    (F(6, 5), F(17, 10), 2),             # above the dome (0 hits)
    (2, 2, F(6, 5)),                     # center column: dyadic → jitters
    (2, 2, 3),                           # far above everything
    (F(1, 2), F(1, 2), F(21, 20)),       # dome too low here: not in the cap
    (F(29, 10), F(19, 10), F(11, 10)),   # off-center, inside
])
def test_classify_matches_exact_membership(p) -> None:
    assert classify_point_in_shell(SHELL, p) == _truth(*p)


def test_first_ray_through_strip_refuses_with_budget_one() -> None:
    # the +z ray from under the rim passes through the trim strip: with a
    # single-ray budget the classifier must refuse by name, not guess
    p = (F(17, 20), 2, F(1, 2))
    with pytest.raises(PointClassifyUncertified) as ei:
        classify_point_in_shell(SHELL, p, max_rays=1)
    assert "ray" in str(ei.value)


def test_near_rim_point_resolves_with_jittered_rays() -> None:
    # same point, full budget: some jittered ray clears the strip and the
    # certified parity answers — the point is below the plane, so "out"
    p = (F(17, 20), 2, F(1, 2))
    assert classify_point_in_shell(SHELL, p) == "out"


def test_point_on_the_shell_refuses_on_every_ray() -> None:
    # exactly on the bottom face (z = 1, inside the disk): the τ-form
    # straddles zero at the hit cell on every ray — honest refusal
    p = (2, 2, 1)
    with pytest.raises(PointClassifyUncertified):
        classify_point_in_shell(SHELL, p, max_rays=4)
