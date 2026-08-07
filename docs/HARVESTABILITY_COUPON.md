# Q5 combined harvestability coupon — adhesion/release + residual-stress + hydrogen on candidate Ti surfaces

**Status:** sanctioned L0 spec (kanban task t_0a2c8d9b, aqueous-steel board).
Carried out via `models/harvestability_coupon.py` (+ CLI `aq-steel-harvestability-coupon`).
**Purpose:** the physical branch + harvestability test from `docs/NEXT_STEPS.md`
§2 items **5** (adhesion/release coupon on candidate titanium release surfaces,
measure peel/strip force) and **6** (bent-strip curvature → residual stress,
plus hydrogen uptake/bake-out on representative deposits).
**Protects against:** building a machine that makes **unharvestable** or
**unsafe** material.
**Not gate evidence.** Everything here is L0 screening scaffold composed from the
existing `adhesion_peel`, `internal_stress`, and `hydrogen_embrittlement`
screening models; there is still no wet-lab iron data in the repository.

---

## 1. Why a single combined coupon

The three quantities in the title were already modelled in three separate
modules, each of which ships its own coupon protocol:

- `adhesion_peel.coupon_test_protocol` — the **interface**: peel/strip force,
  plastic amplification, branch decision (foil vs flake).
- `internal_stress.coupon_curvature_protocol` — the **film**: bent-strip /
  Stoney curvature → residual stress, GUM uncertainty budget.
- `hydrogen_embrittlement` — **hydrogen**: IPZ uptake,
  `bakeout_time_hr`/`bakeout_schedule`, Troiano susceptibility index.

What none of them did was run the three against **the same candidate-Ti coupon
matrix and a single operating point**. That sharing is the point of Q5:

1. **Cross-modal integrity.** A single (j, FE, t, T, pH) point must
   consistently drive peel force, film stress, and hydrogen content. If the
   three measurements contradict the same operating point, the models are
   internally inconsistent and the coupon data arbitrates — exactly the
   cross-modal criterion the twin uses to judge model completeness.
2. **One lab session, three gates.** NEXT_STEPS #2.5/#2.6 are cheap enough to
   run in the same Day-1 session as the Hull cell and divided-cell sets; a
   shared coupon matrix makes that literal.
3. **A combined verdict.** The program wants to know, per surface: can I strip
   it (foil / flake / not at all) **and** is it safe (hydrogen baked out or
   not). `harvestability_coupon.assess_surface` returns exactly that verdict.

---

## 2. The candidate-Ti coupon matrix

Ordered in the spec; the passive-TiO₂ drum surface is the reference.

| id | material / condition | role |
|----|----------------------|------|
| `ti_passive_tio2` | titanium + passive TiO₂ (copper-foil drum practice) | **reference drum surface** |
| `ti_bare_etched` | titanium, etched / de-passivated (metallic contact) | de-passivation failure mode of the reference |
| `stainless_316_passive` | 316L stainless, passive Cr₂O₃ | known-good industrial batch-strip blank |
| `chromium_plated` | hard-chromium mandrel | electroforming release-mandrel alternative |
| `copper_substrate` | copper cathode | **negative control** (expect strong bond) |

All are electrically conductive cathodes; the non-conductive PTFE idea is
excluded on physics (`adhesion_peel` rejects it) rather than by omission. Each
coupon is plated **3 replicates** at the shared operating point.

---

## 3. The shared operating point (cross-modal anchor)

The default matches the Day-1 reference bath (docs/FIRST_LAB_DAY.md bath B0) and
the adhesion/stress coupon sets, so results are directly comparable:

| quantity | value |
|----------|-------|
| current density | 100 mA/cm² |
| current efficiency (Fe) | 85 % |
| deposition time | 1800 s (30 min) → ~38 µm foil |
| bath temperature | 60 °C |
| bath pH | 3.0 |

The derived deposit state (thickness by Faraday, diffusible H by IPZ, grain
size) is computed **once** from this point in
`harvestability_coupon._derived_deposit` and shared by all three channels —
that is the cross-modal integrity guarantee.

---

## 4. The three measurements (NEXT_STEPS #2.5 + #2.6)

For every coupon surface, `assess_surface` returns:

**4.1 Adhesion / release (#2.5)** — from `adhesion_peel.evaluate_peel`:
- peel/strip force per width (N/m), interface outcome (`clean_peel`,
  `marginal_peel`, `spontaneous_delamination`, `bonded_no_release`,
  `cohesive_failure_in_film`, `tears_before_peel`), web stress and tear margin,
  critical self-release thickness.
- Replaces in the model: `plastic_amplification` (the least-constrained peel
  parameter).

**4.2 Residual stress (#2.6a)** — from `internal_stress`
(`deposit_stress_from_conditions`): intrinsic (Hoffman) + hydrogen-effusion +
thermal-mismatch decomposition, total σ, and the bent-strip / Stoney curvature
coupon that measures it.
- Replaces in the model: `HOFFMAN_DELTA_M` and the whole forward intrinsic
  estimate.

**4.3 Hydrogen uptake + bake-out (#2.6b)** — from `hydrogen_embrittlement`:
- IPZ diffusible-H (ppm) and absorption fraction;
- Troiano susceptibility index I_HE (as-plated);
- a full bake-out schedule at 120/150/170/200/250 °C to 0.1 ppm, with the gate
  point taken at **170 °C**.

---

## 5. The combined verdict

`assess_surface` returns one verdict per surface:

| verdict | meaning |
|---------|---------|
| `harvestable_foil` | controllable peel, intact strip — foil branch survives |
| `harvestable_flake` | spontaneous self-release — flake/powder path (redirect) |
| `unharvestable` | will not come off cleanly on this surface at this point |
| `bakeout_required` | H cannot be baked to safe in a production cadence (hard stop), **or** a feasible-but-nonzero bake is still owed before melt/ship |

Decision order (each maps to a NEXT_STEPS rule):

1. **Hard safety stop** — bake-out infeasible (`> 24 hr` at 170 °C): the
   material is unsafe however it releases.
2. **Unharvestable** — peel outcome says it will not come off the drum cleanly.
3. **Bake owed** — H bakes out in a feasible-but-nonzero time (`> 1 hr`): safe
   *after* a bake step (verdict `bakeout_required` as an obligation).
4. Else **harvestable_foil / harvestable_flake** from the peel outcome.

An as-plated critical I_HE is **never silently erased** even when bake is fast:
it is surfaced in `reasons` so the report cannot present a hydrogen-heavy
deposit as cleanly harvestable without the bake step being visible.

### Reference-point dry-run result (L0, synthetic)

At the Day-1 operating point the diffusion models give a thin (~38 µm) deposit
whose H bakes out in **< 1 hr** at 170 °C — so no hard safety stop fires, and
the verdicts track the peel outcome:

| surface | verdict | peel outcome |
|---------|---------|--------------|
| ti_passive_tio2 | harvestable_flake | spontaneous_delamination |
| ti_bare_etched | harvestable_foil | clean_peel |
| stainless_316_passive | harvestable_flake | spontaneous_delamination |
| chromium_plated | harvestable_flake | spontaneous_delamination |
| copper_substrate | harvestable_foil | clean_peel |

This is **not** a prediction that iron peels or flakes — it is what the three
*existing screening models* already said, now rendered as a single per-surface
Q5 verdict. The coupon measurements replace the estimates. Note especially the
cross-modal tension the dry run exposes: the reference TiO₂ drum surface
self-releases (→ flake), while the same deposit on etched Ti or copper is a
controlled peel — the same operating point, different release behaviour, which
is precisely what the coupon is designed to discriminate with real data.

---

## 6. Wiring and the dry run (OFF-by-default)

`harvestability_coupon.py` is a pure composition/reporting layer. It adds **no
new physics** and does not modify any of the three composing modules. The CLI:

```bash
.venv/bin/python -m models.run_harvestability_coupon
```

runs `run_harvestability_dryrun` — `assess_surface` for every coupon in
`COUPON_SURFACE_IDS` — and prints the summary table (surface, verdict, σ_total,
C_H, bake time, peel force). `coupon_spec()` returns the full physical
protocol; `model_scope()` states the is / is-not. `tests/test_harvestability_coupon.py`
locks the composition, the shared-point consistency, the verdict classes, and
the never-silently-harvestable-H rule.

---

## 7. Acceptance criteria

This spec is complete when:
1. one candidate-Ti coupon matrix is specified (§2) at one shared operating
   point (§3);
2. the three NEXT_STEPS measurements are composed per surface (§4);
3. a combined harvestability + safety verdict exists with named decision rules
   (§5), implemented OFF-by-default with a deterministic dry-run (§6);
4. tests lock the composition and the cross-modal consistency.
