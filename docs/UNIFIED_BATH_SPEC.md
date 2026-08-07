# Unified BathSpec and Rich Cell-Physics Coupling

**Status:** implemented screening integration, not gate evidence  
**Code:** `models/bath_spec.py`, `models/cell_physics.py`  
**Tests:** `tests/test_bath_spec.py`, existing `tests/test_cell_physics.py`

## What changed

The cell-physics path can now consume a single electrolyte inventory object,
`BathSpec`, instead of only the legacy sulfate-only `BathRecipe`.

`BathSpec` represents:

- Fe(II) and Fe(III) inventory;
- sulfate route salts: FeSO₄ equivalent, Na₂SO₄, H₂SO₄, boric acid;
- chloride route salts: FeCl₂, LiCl/NaCl, HCl;
- ammonium total for boric-free buffering diagnostics;
- dissolved-O₂ fraction of saturation;
- additive and impurity totals;
- run/twin metadata such as catholyte volume and active area.

It delegates chemistry to the existing modules rather than duplicating them:

| Chemistry | Delegated module |
|---|---|
| sulfate activities, water activity, conductivity | `models.speciation` / `models.pitzer` |
| Fe-Cl / AWARE-type chloride speciation | `models.fe_chloride_speciation` |
| ammonium/ammonia and Fe-ammine diagnostics | `models.ammonium_buffer` |
| dissolved-O₂ solubility, ORR and Fe²⁺ autoxidation | `models.dissolved_oxygen` |
| Fe³⁺ cathodic shuttle estimate | `models.fe3_shuttle` |
| surface-state HER, FeSO₄⁰ pairing, Fe(OH)₂ film | `models.diffusion_layer_1d` options |

## How to use it

Legacy usage is unchanged:

```python
from models.cell_physics import BathRecipe, CellPhysics

point = CellPhysics(BathRecipe()).solve_at_j(100.0)
```

The richer coupled path is opt-in:

```python
from models.bath_spec import BathSpec
from models.cell_physics import CellPhysics, ProcessConditions

bath = BathSpec.reference_sulfate(
    fe2_M=1.0,
    na2so4_M=0.5,
    h3bo3_M=0.4,
    pH=2.0,
    dissolved_o2_fraction_sat=0.25,
    fe3_M=1e-6,
    metadata={"cathode_area_m2": 1.0e-3, "catholyte_volume_L": 0.5},
)

point = CellPhysics(
    bath,
    conditions=ProcessConditions.rich(boundary_layer_m=50e-6),
).solve_at_j(100.0)

print(point.current_efficiency)          # Fe current / applied current
print(point.transport_current_efficiency) # Fe / (Fe + HER) before ORR/Fe3 branches
print(point.current_breakdown_A_m2)
print(point.chemistry_diagnostics)
```

`ProcessConditions.rich()` enables the following central-path corrections:

1. surface-state HER correction;
2. FeSO₄⁰ contact-pair transport correction;
3. Fe(OH)₂ passivation-film resistance when precipitation is active;
4. mass-transfer-limited cathodic ORR from dissolved oxygen;
5. Fe³⁺ shuttle reduction current from measured Fe³⁺ or an O₂-driven screening estimate.

Individual features can also be enabled in otherwise legacy mode, for example:

```python
conditions = ProcessConditions(dissolved_oxygen=True)
```

## Current accounting

The rich solve now reports a current ledger:

```text
j_applied = j_Fe_deposition + j_HER + j_ORR + j_Fe3_shuttle + residual
```

The Fe/HER diffusion layer is solved on the current left after ORR and Fe³⁺
parasitics.  The reported `current_efficiency` is therefore the applied-current
Faradaic efficiency relevant to energy and production:

```text
FE = j_Fe_deposition / j_applied
```

The legacy Fe/(Fe+HER) value is retained as
`transport_current_efficiency` so residual dashboards can identify whether a
missed experiment is a cathode-kinetics error or a side-reaction/redox error.

## Important limitations

- The reactive diffusion film is still sulfate-native.  Chloride `BathSpec`
  objects use the Fe-Cl bulk speciation and surface-state bath type, but a true
  concentrated-solution chloride Nernst-Planck/Maxwell-Stefan film remains a
  future upgrade.
- Ammonium is currently a merged bulk/interfacial diagnostic; NH₄⁺/NH₃/ammine
  species are not yet transported as independent film variables.
- ORR and Fe³⁺ shuttle branches are screening estimates.  They are useful for
  current-ledger honesty but are not gate evidence without measured dissolved
  O₂, Fe³⁺, gas and concentration data.
- `ProcessConditions.rich()` is intended for model-development and sensitivity
  work.  Gate decisions remain measurement-only via `models.process_gates`.

## Why this matters

Previously, several chemistry modules could be correct in isolation while the
headline `CellPhysics` result still saw only a simplified sulfate bath.  The
new object makes the missing side branches visible in every operating point and
provides a clean place to add the next upgrades: transported ammonium species,
chloride concentrated-solution transport, impurity codeposition in the current
ledger, and measured-run residual dashboards.
