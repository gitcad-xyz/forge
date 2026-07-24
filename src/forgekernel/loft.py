"""K3.7 — smooth multi-section loft, exact volume.

A smooth loft interpolates a stack of same-count section polygons with a
natural cubic spline in the section parameter v. Its volume is *exactly
rational*: each vertex coordinate x_j(v), y_j(v) and the height z(v) are
piecewise-cubic polynomials with **rational** coefficients (the natural-
spline tridiagonal system is solved in ℚ), so the cross-section area

    A(v) = ½ Σ_j (x_j y_{j+1} − x_{j+1} y_j)          (shoelace)

is a polynomial in v, and

    V = ∫ A(v) z'(v) dv

integrates exactly per spline segment. OCCT skins a B-spline surface and
Gauss-quadratures the volume; forge returns a Fraction.
"""

from __future__ import annotations

from fractions import Fraction

F = Fraction


def natural_spline_M(vals):
    """Second derivatives M_k of the natural cubic spline through ``vals``
    at unit-spaced knots (M_0 = M_n = 0). Exact rational Thomas solve."""
    n = len(vals) - 1
    if n < 1:
        return [F(0)] * len(vals)
    if n == 1:
        return [F(0), F(0)]
    # tridiagonal: M_{k-1} + 4M_k + M_{k+1} = 6(v_{k+1}-2v_k+v_{k-1})
    a = [F(1)] * (n - 1)          # sub-diagonal
    b = [F(4)] * (n - 1)          # diagonal
    c = [F(1)] * (n - 1)          # super-diagonal
    d = [6 * (vals[k + 1] - 2 * vals[k] + vals[k - 1]) for k in range(1, n)]
    # forward sweep
    for i in range(1, n - 1):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = [F(0)] * (n - 1)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 3, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return [F(0)] + x + [F(0)]


def _seg_cubic(v0, v1, M0, M1):
    """Coefficients [c0,c1,c2,c3] (power basis in local s∈[0,1]) of the
    natural-spline segment with endpoints v0,v1 and 2nd derivs M0,M1
    (unit knot spacing h=1). S(s)=v0(1-s)+v1 s + ((s³-s)M1 + ((1-s)³-(1-s))M0)/6."""
    # expand to power basis in s
    # term A = v0(1-s) + v1 s = v0 + (v1-v0)s
    c = [v0, v1 - v0, F(0), F(0)]
    # term B = M1(s³-s)/6 : +M1/6 s³ - M1/6 s
    c[3] += M1 / 6
    c[1] += -M1 / 6
    # term C = M0((1-s)³-(1-s))/6.  (1-s)³ = 1-3s+3s²-s³ ; (1-s)=1-s
    # (1-s)³-(1-s) = -2s+3s²-s³  → times M0/6
    c[1] += M0 / 6 * (-2)
    c[2] += M0 / 6 * 3
    c[3] += M0 / 6 * (-1)
    return c


def _poly_mul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _poly_add(a, b):
    out = [F(0)] * max(len(a), len(b))
    for i, ai in enumerate(a):
        out[i] += ai
    for j, bj in enumerate(b):
        out[j] += bj
    return out


def _poly_sub(a, b):
    return _poly_add(a, [-c for c in b])


def _poly_deriv(a):
    return [a[i] * i for i in range(1, len(a))] or [F(0)]


def _poly_integ01(a):
    return sum(ci / (i + 1) for i, ci in enumerate(a))


class LoftSolid:
    """Exact-volume smooth loft through ``sections`` = [(loop, z), …] with
    equal vertex counts. ``loop`` is a list of (x, y)."""

    provenance = "exact"

    def __init__(self, sections) -> None:
        self.sections = [([tuple(F(c) for c in pt) for pt in loop], F(z))
                         for loop, z in sections]
        counts = {len(loop) for loop, _ in self.sections}
        if len(self.sections) < 2 or len(counts) != 1:
            raise ValueError("loft: ≥2 sections of equal vertex count")
        self.m = len(self.sections[0][0])           # verts per section
        self.n = len(self.sections)                  # section count

    def _splines(self):
        """Per-vertex x,y splines (M arrays) + the z spline."""
        xs = [[self.sections[k][0][j][0] for k in range(self.n)]
              for j in range(self.m)]
        ys = [[self.sections[k][0][j][1] for k in range(self.n)]
              for j in range(self.m)]
        zs = [self.sections[k][1] for k in range(self.n)]
        Mx = [natural_spline_M(xs[j]) for j in range(self.m)]
        My = [natural_spline_M(ys[j]) for j in range(self.m)]
        Mz = natural_spline_M(zs)
        return xs, ys, zs, Mx, My, Mz

    def _moments(self):
        """Signed volume and the three first moments (∫x dV, ∫y dV, ∫z dV),
        all exact ℚ. Volume ``V = ∫ A(v) z'(v) dv`` with A the shoelace
        cross-section area; the moments add the polygon area-moments
        ``Qx = ∫∫ x dA`` and ``Qy = ∫∫ y dA`` (exact for a polygon) and the
        ``z·A`` integrand. The section-loop orientation cancels in every
        centroid ratio because A, Qx, Qy all carry the same signed factor."""
        xs, ys, zs, Mx, My, Mz = self._splines()
        Vs = Ix = Iy = Iz = F(0)
        for seg in range(self.n - 1):
            xpoly = [_seg_cubic(xs[j][seg], xs[j][seg + 1],
                                Mx[j][seg], Mx[j][seg + 1]) for j in range(self.m)]
            ypoly = [_seg_cubic(ys[j][seg], ys[j][seg + 1],
                                My[j][seg], My[j][seg + 1]) for j in range(self.m)]
            zpoly = _seg_cubic(zs[seg], zs[seg + 1], Mz[seg], Mz[seg + 1])
            zprime = _poly_deriv(zpoly)
            area = [F(0)]           # A(s) = ½ Σ cross_j
            qx = [F(0)]             # ∫∫ x dA = ⅙ Σ (x_j+x_k)·cross_j
            qy = [F(0)]             # ∫∫ y dA = ⅙ Σ (y_j+y_k)·cross_j
            for j in range(self.m):
                k = (j + 1) % self.m
                cross = _poly_sub(_poly_mul(xpoly[j], ypoly[k]),
                                  _poly_mul(xpoly[k], ypoly[j]))
                area = _poly_add(area, cross)
                qx = _poly_add(qx, _poly_mul(_poly_add(xpoly[j], xpoly[k]), cross))
                qy = _poly_add(qy, _poly_mul(_poly_add(ypoly[j], ypoly[k]), cross))
            area = [c / 2 for c in area]
            qx = [c / 6 for c in qx]
            qy = [c / 6 for c in qy]
            Vs += _poly_integ01(_poly_mul(area, zprime))
            Ix += _poly_integ01(_poly_mul(qx, zprime))
            Iy += _poly_integ01(_poly_mul(qy, zprime))
            Iz += _poly_integ01(_poly_mul(_poly_mul(zpoly, area), zprime))
        return Vs, Ix, Iy, Iz

    def volume(self) -> Fraction:
        return abs(self._moments()[0])

    def centroid(self):
        """Exact centroid in ℚ³ — the true first-moment centroid, not the
        bbox centre. Signed volume and moments share orientation, so the
        ratio is orientation-independent."""
        Vs, Ix, Iy, Iz = self._moments()
        if Vs == 0:
            raise ValueError("loft: degenerate (zero signed volume)")
        return (Ix / Vs, Iy / Vs, Iz / Vs)

    def bbox_f(self):
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for loop, z in self.sections:
            for x, y in loop:
                lo[0], hi[0] = min(lo[0], float(x)), max(hi[0], float(x))
                lo[1], hi[1] = min(lo[1], float(y)), max(hi[1], float(y))
            lo[2], hi[2] = min(lo[2], float(z)), max(hi[2], float(z))
        return tuple(lo), tuple(hi)

    def centroid_f(self):
        """Float centroid, derived from the exact :meth:`centroid` (not the
        bbox centre — that is only correct for a symmetric loft)."""
        return tuple(float(c) for c in self.centroid())
