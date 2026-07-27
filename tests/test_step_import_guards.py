"""Two shipped STEP-import defects, pinned failing-first.

1. CHORDED CURVED EDGE (silent wrong number): the planar topology walk
   keyed its freeform refusal on the SURFACE type only and read nothing
   but vertex points off each ``EDGE_CURVE`` — so a planar face bounded
   by a genuinely curved edge (arc, B-spline) imported with the edge
   silently replaced by its chord. A bulged box imported at volume 24
   with no refusal; any rounded-corner plate would import with a
   silently wrong volume. The basis-curve guard refuses non-LINE edges
   on the planar path BY NAME (the curved-edge topology arrives with
   K3.7's trimmed import).

2. ENTITY REGEX SPLIT INSIDE STRINGS (silent entity loss):
   ``StepFile`` split entities at the first ``;`` even inside a quoted
   string, so ``PRODUCT('a;b')`` VANISHED from the entity map (the
   parse resynced mid-string) and surfaced — if at all — as a KeyError
   three modules from the cause. The tokenizer is now quote-aware,
   including STEP's doubled-quote escape (``'it''s'``).
"""

from __future__ import annotations

import re

import pytest

from forgekernel.brep import Solid
from forgekernel.stepio import (StepFile, read_step_planar_solid,
                                write_step_planar_solid)


def _bulged_box_step() -> str:
    """A kernel-exported box with one straight edge's basis curve swapped
    for an in-plane bulging B-spline (same endpoints — the chord is what
    the old walk silently kept)."""
    step = write_step_planar_solid(Solid.box(2, 3, 4))
    m = re.search(r"#(\d+)\s*=\s*LINE\([^;]*\);", step)
    assert m, "writer no longer emits LINE entities — fixture needs updating"
    eid = m.group(1)
    curved = (f"#{eid}=B_SPLINE_CURVE_WITH_KNOTS('',2,"
              "(#9001,#9002,#9003),.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),"
              ".UNSPECIFIED.);\n"
              "#9001=CARTESIAN_POINT('',(0.,0.,0.));\n"
              "#9002=CARTESIAN_POINT('',(5.,5.,5.));\n"
              "#9003=CARTESIAN_POINT('',(1.,1.,1.));")
    return step.replace(m.group(0), curved)


def test_curved_edge_on_planar_face_refuses_by_name():
    with pytest.raises(ValueError, match="curved edge"):
        read_step_planar_solid(_bulged_box_step())


def test_straight_box_still_round_trips():
    s = read_step_planar_solid(write_step_planar_solid(Solid.box(2, 3, 4)))
    assert float(s.volume()) == 24.0


# -- 2: quote-aware entity tokenizer ------------------------------------------


def test_semicolon_inside_string_does_not_split_the_entity():
    txt = ("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
           "#1=PRODUCT('a;b','desc','',(#2));\n"
           "#2=CARTESIAN_POINT('',(1.,2.,3.));\n"
           "ENDSEC;\nEND-ISO-10303-21;\n")
    sf = StepFile(txt)
    assert sorted(sf.entities) == [1, 2]
    types, args = sf.entities[1]
    assert types == ["PRODUCT"]
    assert "'a;b'" in args


def test_doubled_quote_escape_with_semicolon_survives():
    txt = ("DATA;\n"
           "#1=PRODUCT_CONTEXT('it''s; escaped',#2,'x');\n"
           "#2=CARTESIAN_POINT('a;b',(1.,2.,3.));\n"
           "ENDSEC;\n")
    sf = StepFile(txt)
    assert sorted(sf.entities) == [1, 2]
    assert "'it''s; escaped'" in sf.entities[1][1]
    # the point still parses despite its decorated name
    assert sf.point("#2") == (1, 2, 3)


def test_planar_import_survives_semicolon_in_entity_names():
    step = write_step_planar_solid(Solid.box(2, 3, 4))
    step = step.replace("CLOSED_SHELL('", "CLOSED_SHELL('lid;top ", 1)
    s = read_step_planar_solid(step)
    assert float(s.volume()) == 24.0
