"""K3.4 — STEP geometry reader: exact decimal→rational import."""

from __future__ import annotations

from fractions import Fraction

from forgekernel.stepio import StepFile, read_step_geometry

F = Fraction

_SIMPLE = """
ISO-10303-21;
DATA;
#1 = CARTESIAN_POINT('',(0.,0.,0.));
#2 = CARTESIAN_POINT('',(1.,3.,0.));
#3 = CARTESIAN_POINT('',(3.,3.,0.));
#4 = CARTESIAN_POINT('',(4.,0.,0.));
#10 = B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4),.UNSPECIFIED.,.F.,.F.,
  (4,4),(0.,1.),.PIECEWISE_BEZIER_KNOTS.);
ENDSEC;
END-ISO-10303-21;
"""

_RATIONAL = """
DATA;
#1 = CARTESIAN_POINT('',(1.,0.,0.));
#2 = CARTESIAN_POINT('',(1.,1.,0.));
#3 = CARTESIAN_POINT('',(0.,1.,0.));
#20 = ( BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
  B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.) CURVE()
  GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE((1.,0.75,1.))
  REPRESENTATION_ITEM('') );
ENDSEC;
"""


def test_simple_bspline_curve_parses_exactly() -> None:
    geo = read_step_geometry(_SIMPLE)
    assert len(geo["curves"]) == 1
    c = geo["curves"][0]
    assert c.p == 3
    assert c.cp[1] == (1, 3, 0)
    assert all(isinstance(v, Fraction) for v in c.cp[2])
    # cubic Bézier midpoint, exact: (P0+3P1+3P2+P3)/8
    assert c.eval(F(1, 2)) == (F(2), F(9, 4), F(0))


def test_rational_complex_entity_parses_with_weights() -> None:
    sf = StepFile(_RATIONAL)
    ids = sf.bspline_curves()
    assert ids == [20]
    c = sf.curve(20)
    assert c.rational
    assert c.w == [F(1), F(3, 4), F(1)]      # 0.75 → exactly 3/4
    # rational quadratic still evaluates exactly (weights are rational)
    pt = c.eval(F(1, 2))
    assert all(isinstance(v, Fraction) for v in pt)


def test_decimal_text_is_exact_not_float() -> None:
    # 0.1 is not a binary float — but it IS the exact rational 1/10 here
    step = """DATA;
#1 = CARTESIAN_POINT('',(0.1,0.2,0.3));
#2 = CARTESIAN_POINT('',(1.,0.,0.));
#3 = B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,
  (2,2),(0.,1.),.UNSPECIFIED.);
ENDSEC;"""
    c = read_step_geometry(step)["curves"][0]
    assert c.cp[0] == (F(1, 10), F(1, 5), F(3, 10))   # exact, unlike a double
