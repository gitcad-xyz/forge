"""K3.1 — NURBS / B-spline curve evaluation via de Boor (ADR-0018/0019).

The groundwork for free-form geometry: STEP AP214 curves, general
sweeps/lofts, and — the crown jewel — surface–surface intersection.

The exactness charter reaches further here than one might expect. A
B-spline with **rational** control points, knots, and weights, evaluated
at a **rational** parameter, comes out *exactly rational*: de Boor's
recurrence is nothing but convex combinations — ``+``, ``×``, and ``÷``
by rationals — so no irrationality enters. Such a curve is ``exact``,
not merely certified, and ``eval`` returns a point in ℚ³ that OCCT can
only approximate.

The certified-interval path (ADR-0019) is reserved for the genuinely
irrational cases: an irrational weight (the √2/2 of a true circular
NURBS arc) or an irrational parameter. There ``eval_ci`` carries each
homogeneous coordinate as a ``CInterval`` and the division by the
weight stays enclosure-preserving.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.interval import CInterval

F = Fraction


class BSplineCurve:
    """A NURBS curve: degree ``p``, control points, clamped/open knot
    vector, optional weights (default 1 → a polynomial B-spline).

    Control points are 3-tuples; knots and weights are scalars. Anything
    rational stays exact through evaluation."""

    def __init__(self, degree: int, control_points, knots, weights=None) -> None:
        self.p = int(degree)
        self.cp = [tuple(F(c) for c in pt) for pt in control_points]
        self.U = [F(u) for u in knots]
        n = len(self.cp)
        if weights is None:
            self.w = [F(1)] * n
            self.rational = False
            self.exact_weights = True
        else:
            # weights may be rational (exact eval available) or CInterval
            # (a genuinely irrational weight, e.g. √2/2 for a true circle —
            # only the certified path eval_ci applies)
            self.w = [x if isinstance(x, CInterval) else F(x) for x in weights]
            self.exact_weights = all(not isinstance(x, CInterval) for x in self.w)
            self.rational = not self.exact_weights or \
                any(x != self.w[0] for x in self.w)
        if len(self.w) != n:
            raise ValueError("weights and control points differ in count")
        if len(self.U) != n + self.p + 1:
            raise ValueError(
                f"knot vector must have n+p+1={n + self.p + 1} entries, "
                f"got {len(self.U)}")

    # -- span location --------------------------------------------------------

    def _span(self, u: F) -> int:
        """Knot span index k with U[k] <= u < U[k+1] (clamped to the last
        non-empty span at the right end). Exact rational comparisons."""
        n = len(self.cp) - 1
        if u >= self.U[n + 1]:
            return n
        if u <= self.U[self.p]:
            return self.p
        lo, hi = self.p, n + 1
        while hi - lo > 1:                  # binary search, exact
            mid = (lo + hi) // 2
            if u < self.U[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    # -- exact rational evaluation (de Boor in homogeneous coords) ------------

    def eval(self, t):
        """Exact point in ℚ³ at parameter ``t`` (rational in, rational out).
        Raises if any weight is irrational — use :meth:`eval_ci` for that."""
        if not self.exact_weights:
            raise ValueError(
                "curve has irrational weights — use eval_ci (certified)")
        u = F(t)
        k = self._span(u)
        p = self.p
        # homogeneous control points (w·x, w·y, w·z, w) for the active span
        d = []
        for j in range(p + 1):
            i = k - p + j
            wi = self.w[i]
            x, y, z = self.cp[i]
            d.append([wi * x, wi * y, wi * z, wi])
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = k - p + j
                denom = self.U[i + p - r + 1] - self.U[i]
                a = F(0) if denom == 0 else (u - self.U[i]) / denom
                b = 1 - a
                d[j] = [b * d[j - 1][c] + a * d[j][c] for c in range(4)]
        hx, hy, hz, hw = d[p]
        return (hx / hw, hy / hw, hz / hw)

    # -- certified evaluation (for irrational weights/parameters) -------------

    def eval_ci(self, t):
        """Certified point (three ``CInterval``s) — the enclosure-preserving
        path for irrational weights or parameters. Accepts rational or
        ``CInterval`` inputs and never loses the bracket."""
        u = t if isinstance(t, CInterval) else CInterval.exact(F(t))
        # span from the interval midpoint (a location choice, not a decision
        # that affects the certified value — the recurrence is continuous)
        k = self._span(u.mid)
        p = self.p
        d = []
        for j in range(p + 1):
            i = k - p + j
            wi = _ci(self.w[i])
            x, y, z = (_ci(v) for v in self.cp[i])
            d.append([wi * x, wi * y, wi * z, wi])
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = k - p + j
                denom = self.U[i + p - r + 1] - self.U[i]
                if denom == 0:
                    continue
                a = (u - _ci(self.U[i])) * _ci(F(1) / denom)
                b = _ci(1) - a
                d[j] = [b * d[j - 1][c] + a * d[j][c] for c in range(4)]
        hx, hy, hz, hw = d[p]
        inv = _ci_reciprocal(hw)
        return (hx * inv, hy * inv, hz * inv)

    # -- float evaluation (tessellation) --------------------------------------

    def eval_f(self, t: float):
        x, y, z = self.eval(F(t).limit_denominator(10 ** 9))
        return (float(x), float(y), float(z))

    def domain(self):
        return (float(self.U[self.p]), float(self.U[len(self.cp)]))


# -- constructors -------------------------------------------------------------

def bezier(control_points, weights=None) -> BSplineCurve:
    """A Bézier curve as a clamped B-spline on [0, 1]."""
    n = len(control_points)
    p = n - 1
    knots = [F(0)] * (p + 1) + [F(1)] * (p + 1)
    return BSplineCurve(p, control_points, knots, weights)


def _ci(x) -> CInterval:
    return x if isinstance(x, CInterval) else CInterval.exact(F(x))


def _ci_reciprocal(x: CInterval) -> CInterval:
    """1/x for an interval that strictly excludes zero (certified)."""
    s = x.sign()                                    # raises if straddles 0
    lo, hi = (F(1) / x.hi, F(1) / x.lo) if s > 0 else (F(1) / x.hi, F(1) / x.lo)
    return CInterval(min(lo, hi), max(lo, hi))
