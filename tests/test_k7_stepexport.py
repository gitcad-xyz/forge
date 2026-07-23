"""K7.0c — native STEP AP214 export of planar solids (OCCT-free exchange)."""

from __future__ import annotations

from forgekernel.brep import Solid
from forgekernel.stepio import read_step_planar_solid, write_step_planar_solid


def test_step_export_self_roundtrips_exactly() -> None:
    for dims in ((6, 4, 3), (10, 10, 10), (2, 7, 5)):
        s = Solid.box(*dims, "b")
        back = read_step_planar_solid(write_step_planar_solid(s))
        assert back.volume() == s.volume()          # exact (terminating decimals)
        assert not back.watertight_violations()


def test_step_export_is_valid_ap214_structure() -> None:
    txt = write_step_planar_solid(Solid.box(3, 4, 5, "b"))
    for token in ("ISO-10303-21", "MANIFOLD_SOLID_BREP", "CLOSED_SHELL",
                  "SHAPE_DEFINITION_REPRESENTATION", "PRODUCT_DEFINITION",
                  "ADVANCED_BREP_SHAPE_REPRESENTATION", "END-ISO-10303-21"):
        assert token in txt
