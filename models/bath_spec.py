"""Unified electrolyte chemistry state for the cell-physics stack.

The repository has rich chemistry modules for sulfate speciation, chloride
speciation, ammonium buffering, dissolved oxygen, Fe³⁺ shuttle losses, and
surface-state HER.  Historically those modules were invoked from separate
runners, while :mod:`models.cell_physics` accepted only a narrow
``BathRecipe`` (FeSO₄/Na₂SO₄/H₂SO₄/H₃BO₃/pH).  ``BathSpec`` is the shared
state object that lets one operating-point solve see the same bath inventory
and decide which chemistry corrections should be attached.

Scope
-----
This is a reduced-order integration object, not a full equilibrium database.
It deliberately exposes the important totals and delegates actual chemistry to
existing modules:

* sulfate activity/conductivity -> :func:`models.speciation.solve_speciation`
* chloride activity/conductivity -> :func:`models.fe_chloride_speciation.solve_chloride_speciation`
* ammonium buffer/ammine diagnostics -> :class:`models.ammonium_buffer.AmmoniumBufferModel`
* O₂ solubility / ORR / Fe²⁺ autoxidation -> :mod:`models.dissolved_oxygen`

The object is therefore useful both as a central cell-physics input and as a
machine-readable summary of which chemistry is present in a run record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Literal

from .speciation import SolutionComposition, solve_speciation
from .fe_chloride_speciation import ChlorideBathComposition, solve_chloride_speciation
from .ammonium_buffer import AmmoniumBufferModel
from .dissolved_oxygen import (
    DissolvedOxygenParams,
    cathodic_orr_limiting_current_A_m2,
    dissolved_oxygen_solubility_M,
    homogeneous_fe2_oxidation_rate_M_s,
)

BathFamily = Literal["sulfate", "chloride", "mixed"]


@dataclass
class BathSpec:
    """Versionable bath inventory shared by speciation, transport and control.

    Parameters are molar concentrations unless noted otherwise.  The field
    names intentionally use species totals (``fe2_M``, ``chloride_M``,
    ``ammonium_total_M``) rather than salt bottle names where possible so a
    measured bath can be represented without reverse-engineering how it was
    prepared.  Salt-specific fields remain for common recipes and for mapping
    to the existing sulfate/chloride solvers.
    """

    name: str = "reference_sulfate"
    family: BathFamily = "sulfate"

    # Core iron/redox inventory
    fe2_M: float = 1.0
    fe3_M: float = 0.0

    # Sulfate-route recipe fields
    na2so4_M: float = 0.5
    h2so4_M: float = 0.01
    h3bo3_M: float = 0.4

    # Chloride-route recipe fields.  ``fecl2_M`` defaults to zero so sulfate
    # users only need ``fe2_M``; chloride factory constructors set both.
    fecl2_M: float = 0.0
    licl_M: float = 0.0
    nacl_M: float = 0.0
    hcl_M: float = 0.0

    # Alternative buffers / ligands and gas exposure
    ammonium_total_M: float = 0.0
    dissolved_o2_fraction_sat: float = 0.0

    # Declared/measured boundary pH and default temperature for standalone
    # diagnostics.  ``CellPhysics`` overrides temperature with
    # ``ProcessConditions.temperature_C`` so one bath can be swept in T.
    pH: float = 2.0
    temperature_C: float = 50.0

    # Machine-readable extras used by calibration/run-record code.  Values are
    # totals in mol/L unless an entry documents otherwise.
    additives_M: Mapping[str, float] = field(default_factory=dict)
    impurities_M: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in ("sulfate", "chloride", "mixed"):
            raise ValueError("family must be 'sulfate', 'chloride', or 'mixed'")
        for field_name in (
            "fe2_M", "fe3_M", "na2so4_M", "h2so4_M", "h3bo3_M",
            "fecl2_M", "licl_M", "nacl_M", "hcl_M", "ammonium_total_M",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.fe2_total_M <= 0.0:
            raise ValueError("BathSpec requires a positive Fe(II) inventory")
        if self.dissolved_o2_fraction_sat < 0.0:
            raise ValueError("dissolved_o2_fraction_sat must be non-negative")
        if self.pH < 0.0 or self.pH > 14.5:
            raise ValueError("pH must be in a physically meaningful aqueous range")
        # Copy mutable mappings so callers cannot mutate the chemistry behind
        # a running twin by holding a reference to the original dict.
        object.__setattr__(self, "additives_M", dict(self.additives_M))
        object.__setattr__(self, "impurities_M", dict(self.impurities_M))
        object.__setattr__(self, "metadata", dict(self.metadata))

    # ------------------------------------------------------------------
    # Constructors for common repository cases
    # ------------------------------------------------------------------
    @classmethod
    def reference_sulfate(
        cls,
        *,
        fe2_M: float = 1.0,
        na2so4_M: float = 0.5,
        h2so4_M: float = 0.01,
        h3bo3_M: float = 0.4,
        pH: float = 2.0,
        temperature_C: float = 50.0,
        dissolved_o2_fraction_sat: float = 0.0,
        fe3_M: float = 0.0,
        ammonium_total_M: float = 0.0,
        additives_M: Mapping[str, float] | None = None,
        impurities_M: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        name: str = "reference_sulfate",
    ) -> "BathSpec":
        """Canonical FeSO₄/Na₂SO₄/borate bath used by ``CellPhysics``."""
        return cls(
            name=name,
            family="sulfate",
            fe2_M=fe2_M,
            fe3_M=fe3_M,
            na2so4_M=na2so4_M,
            h2so4_M=h2so4_M,
            h3bo3_M=h3bo3_M,
            pH=pH,
            temperature_C=temperature_C,
            dissolved_o2_fraction_sat=dissolved_o2_fraction_sat,
            ammonium_total_M=ammonium_total_M,
            additives_M={} if additives_M is None else additives_M,
            impurities_M={} if impurities_M is None else impurities_M,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def aware_chloride(
        cls,
        *,
        fecl2_M: float = 1.0,
        licl_M: float = 10.0,
        hcl_M: float = 0.01,
        pH: float = 2.0,
        temperature_C: float = 60.0,
        dissolved_o2_fraction_sat: float = 0.0,
        fe3_M: float = 0.0,
        additives_M: Mapping[str, float] | None = None,
        impurities_M: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        name: str = "aware_chloride",
    ) -> "BathSpec":
        """AWARE-style concentrated chloride bath inventory."""
        return cls(
            name=name,
            family="chloride",
            fe2_M=fecl2_M,
            fe3_M=fe3_M,
            na2so4_M=0.0,
            h2so4_M=0.0,
            h3bo3_M=0.0,
            fecl2_M=fecl2_M,
            licl_M=licl_M,
            hcl_M=hcl_M,
            pH=pH,
            temperature_C=temperature_C,
            dissolved_o2_fraction_sat=dissolved_o2_fraction_sat,
            additives_M={} if additives_M is None else additives_M,
            impurities_M={} if impurities_M is None else impurities_M,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_legacy_recipe(
        cls,
        recipe: Any,
        *,
        temperature_C: float = 50.0,
        name: str = "legacy_bath_recipe",
    ) -> "BathSpec":
        """Create a ``BathSpec`` from the old ``cell_physics.BathRecipe``.

        ``Any`` is used to avoid a module import cycle; duck typing is enough
        because the legacy recipe is a tiny dataclass.
        """
        return cls.reference_sulfate(
            name=name,
            fe2_M=float(recipe.c_FeSO4_M),
            na2so4_M=float(recipe.c_Na2SO4_M),
            h2so4_M=float(recipe.c_H2SO4_M),
            h3bo3_M=float(recipe.c_H3BO3_M),
            pH=float(recipe.pH),
            temperature_C=temperature_C,
        )

    # ------------------------------------------------------------------
    # Derived inventory and compatibility helpers
    # ------------------------------------------------------------------
    @property
    def fe2_total_M(self) -> float:
        """Total dissolved Fe(II) represented by the bath."""
        return float(self.fecl2_M if self.family == "chloride" and self.fecl2_M > 0 else self.fe2_M)

    @property
    def chloride_total_M(self) -> float:
        """Total nominal chloride from FeCl₂, LiCl/NaCl and HCl."""
        return float(2.0 * self.fecl2_M + self.licl_M + self.nacl_M + self.hcl_M)

    @property
    def sulfate_total_M(self) -> float:
        """Total nominal sulfate from FeSO₄/Na₂SO₄/H₂SO₄ equivalents."""
        if self.family == "chloride":
            return float(self.na2so4_M + self.h2so4_M)
        return float(self.fe2_M + self.na2so4_M + self.h2so4_M)

    def to_legacy_bath_recipe_kwargs(self) -> Dict[str, float]:
        """Keyword mapping for ``cell_physics.BathRecipe``.

        The current reactive diffusion film is sulfate-native.  For chloride
        baths this mapping is intentionally a compatibility fallback: Fe(II)
        and pH are preserved, sulfate/borate are set to zero unless supplied,
        and chloride-specific activity/conductivity remains available through
        :meth:`solve_bulk_speciation`.  The diagnostics flag this limitation.
        """
        return {
            "c_FeSO4_M": self.fe2_total_M,
            "c_Na2SO4_M": self.na2so4_M if self.family != "chloride" else 0.0,
            "c_H2SO4_M": self.h2so4_M if self.family != "chloride" else 0.0,
            "c_H3BO3_M": self.h3bo3_M if self.family != "chloride" else 0.0,
            "pH": self.pH,
        }

    def to_solution_composition(self, T_C: float | None = None) -> SolutionComposition:
        """Sulfate ``SolutionComposition`` view used by the Pitzer solver."""
        return SolutionComposition(
            c_FeSO4=self.fe2_total_M,
            c_Na2SO4=self.na2so4_M,
            c_H2SO4=self.h2so4_M,
            c_H3BO3=self.h3bo3_M,
            T_C=self.temperature_C if T_C is None else float(T_C),
        )

    def to_chloride_composition(self, T_C: float | None = None) -> ChlorideBathComposition:
        """Chloride ``ChlorideBathComposition`` view used by the Fe-Cl solver."""
        return ChlorideBathComposition(
            c_FeCl2=self.fe2_total_M if self.fecl2_M <= 0 else self.fecl2_M,
            c_LiCl=self.licl_M,
            c_NaCl=self.nacl_M,
            c_HCl=self.hcl_M,
            T_C=self.temperature_C if T_C is None else float(T_C),
        )

    def surface_state_bath_type(self) -> str:
        """Map the inventory to ``DiffusionLayer1D.bath_type`` keys."""
        if self.family == "chloride" or self.chloride_total_M >= 5.0:
            return "aware"
        if self.chloride_total_M > 0.05 and self.sulfate_total_M > 0.05:
            return "mixed"
        return "sulfate"

    # ------------------------------------------------------------------
    # Diagnostics delegated to existing chemistry modules
    # ------------------------------------------------------------------
    def solve_bulk_speciation(self, T_C: float | None = None) -> Dict[str, Any]:
        """Return one merged, machine-readable bath chemistry dictionary."""
        T = self.temperature_C if T_C is None else float(T_C)
        if self.family == "chloride":
            out: Dict[str, Any] = solve_chloride_speciation(self.to_chloride_composition(T))
            out["transport_solver_basis"] = (
                "chloride bulk speciation; sulfate-native film uses BathSpec "
                "compatibility view until concentrated-solution chloride film is added"
            )
        else:
            out = solve_speciation(self.to_solution_composition(T), model="pitzer")
            if self.chloride_total_M > 0.0:
                # Mixed sulfate/chloride route: keep sulfate as the dominant
                # transport basis, but report chloride speciation separately.
                out["chloride_diagnostics"] = solve_chloride_speciation(
                    self.to_chloride_composition(T)
                )
                out["transport_solver_basis"] = "sulfate Pitzer film with mixed-chloride diagnostics"
            else:
                out["transport_solver_basis"] = "sulfate Pitzer film"

        out.update({
            "bath_spec_name": self.name,
            "bath_family": self.family,
            "declared_pH": float(self.pH),
            "c_Fe2_total_M": self.fe2_total_M,
            "c_Fe3_total_M": float(self.fe3_M),
            "c_sulfate_total_M": self.sulfate_total_M,
            "c_chloride_total_M": self.chloride_total_M,
            "c_NH4_total_M": float(self.ammonium_total_M),
            "dissolved_o2_fraction_sat": float(self.dissolved_o2_fraction_sat),
            "additives_M": dict(self.additives_M),
            "impurities_M": dict(self.impurities_M),
            "metadata": dict(self.metadata),
        })

        if self.ammonium_total_M > 0.0:
            amm = AmmoniumBufferModel(T).solve_speciation(
                self.pH, self.fe2_total_M, self.ammonium_total_M
            )
            out["ammonium"] = {
                "pKa": float(AmmoniumBufferModel(T).pka),
                "free_fe2_M": float(amm.free_fe2_M),
                "free_nh3_M": float(amm.free_nh3_M),
                "nh4_M": float(amm.nh4_M),
                "complexed_fe_M": float(sum(amm.fe_ammine_M)),
                "saturation_ratio_FeOH2": float(amm.saturation_ratio_FeOH2),
                "is_hydroxide_precipitated": bool(amm.is_hydroxide_precipitated),
                "screening_note": "diagnostic ligand speciation; not yet a transported film species",
            }

        if self.dissolved_o2_fraction_sat > 0.0:
            ionic_strength = float(out.get("ionic_strength_M", out.get("ionic_strength_molal", 0.0)))
            do_params = DissolvedOxygenParams(
                temperature_C=T,
                ionic_strength_M=ionic_strength,
                pH=self.pH,
                fe2_M=self.fe2_total_M,
            )
            sat = dissolved_oxygen_solubility_M(T, ionic_strength)
            out["dissolved_oxygen"] = {
                "saturation_M": float(sat),
                "bulk_M": float(sat * self.dissolved_o2_fraction_sat),
                "fraction_sat": float(self.dissolved_o2_fraction_sat),
                "orr_limiting_current_A_m2_at_100um": float(
                    cathodic_orr_limiting_current_A_m2(do_params, self.dissolved_o2_fraction_sat)
                ),
                "fe3_generation_M_s": float(
                    homogeneous_fe2_oxidation_rate_M_s(do_params, self.dissolved_o2_fraction_sat)
                ),
            }

        return out

    def oxygen_params(
        self,
        *,
        T_C: float,
        ionic_strength_M: float,
        boundary_layer_m: float,
    ) -> DissolvedOxygenParams:
        """Build dissolved-O₂ parameters consistent with this bath and film."""
        return DissolvedOxygenParams(
            temperature_C=float(T_C),
            ionic_strength_M=max(float(ionic_strength_M), 0.0),
            pH=float(self.pH),
            fe2_M=self.fe2_total_M,
            delta_um=float(boundary_layer_m) * 1e6,
        )

    def feature_inventory(self) -> Dict[str, bool]:
        """Boolean chemistry inventory useful for summaries and validation."""
        return {
            "sulfate_present": self.sulfate_total_M > 0.0,
            "chloride_present": self.chloride_total_M > 0.0,
            "ammonium_present": self.ammonium_total_M > 0.0,
            "borate_present": self.h3bo3_M > 0.0,
            "dissolved_oxygen_present": self.dissolved_o2_fraction_sat > 0.0,
            "fe3_present": self.fe3_M > 0.0,
            "additives_present": bool(self.additives_M),
            "impurities_present": bool(self.impurities_M),
        }

    def to_record(self) -> Dict[str, Any]:
        """JSON-friendly bath inventory for run records and reports."""
        return {
            "name": self.name,
            "family": self.family,
            "fe2_M": self.fe2_M,
            "fe3_M": self.fe3_M,
            "na2so4_M": self.na2so4_M,
            "h2so4_M": self.h2so4_M,
            "h3bo3_M": self.h3bo3_M,
            "fecl2_M": self.fecl2_M,
            "licl_M": self.licl_M,
            "nacl_M": self.nacl_M,
            "hcl_M": self.hcl_M,
            "ammonium_total_M": self.ammonium_total_M,
            "dissolved_o2_fraction_sat": self.dissolved_o2_fraction_sat,
            "pH": self.pH,
            "temperature_C": self.temperature_C,
            "additives_M": dict(self.additives_M),
            "impurities_M": dict(self.impurities_M),
            "metadata": dict(self.metadata),
        }


__all__ = ["BathFamily", "BathSpec"]
