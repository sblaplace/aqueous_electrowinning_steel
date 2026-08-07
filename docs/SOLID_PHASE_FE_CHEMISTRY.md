# Solid-phase Fe chemistry in the deposit (Tier 1.2)

Companion note to `CHEM_PHYS_REVIEW.md` §1.2 ("Solid-phase Fe chemistry in
the deposit is treated as pure Fe").  Two screening modules were added to
close that gap without touching existing module behaviour:

* `models/oxygen_in_iron.py` — oxygen in the deposit
* `models/bath_impurity_codeposition.py` — S / P / Mn / Si / B co-deposition
  and AISI 10xx routing

Both carry `SCREENING_FLAG = "unvalidated (L1)"` and are **not** gate
evidence: the numbers are screening central values to be replaced by
inert-gas-fusion (LECO), combustion/OES analysis of real deposits.

---

## What oxygen does to the yield and cold-rolling ceiling

Electrolytic iron always contains oxygen, carried in as co-deposited
Fe(OH)₂ / FeOOH inclusion particles (not as a substitutional alloying
element).  The mechanical model had used Ni and C as the only
solid-solution strengtheners; `oxygen_in_iron.py` adds the O channel:

1. **Oxygen budget.** A precipitation flux (mol Fe(OH)₂ / m²·s) — from the
   diffusion-layer precipitation / pH model, `DiffusionLayer1D.solve(j)
   .precipitation_flux_mol_m2_s`, or estimated from the pulse waveform via
   `surface_pH_from_pulse` — is converted to a deposit O content (ppm).
   Only a fraction of the precipitated Fe is mechanically captured; the rest
   is sludge / redissolves.

2. **Upper-bound yield strength.** Oxygen from a hard oxide dispersion
   raises the deposit's upper-bound yield (Orowan-type), `Δσ_O ≈ 120 MPa per
   1 000 ppm O`.  The `H=P`/Ni/C model gives a *lower* bound; O is why a real
   as-deposited foil is harder and more brittle than the alloying-only
   estimate.

3. **Cold-rolling ceiling.** Above ~1 000 ppm O the oxide network makes the
   as-deposited foil edge-crack on cold rolling (not cold-rollable without
   prior processing); below ~400 ppm it is freely cold-rollable on O grounds.
   `cold_rollability()` returns the verdict.

   The default operating point (100 mA/cm², pH 3.5, sulfate) yields
   ~700–1 300 ppm O depending on waveform — i.e. sitting at or above the
   rolling ceiling — which is exactly why the pulse waveform (and PRE's
   anodic oxide dissolution) matters for producing a rollable foil.

## What the impurity module routes to

`bath_impurity_codeposition.py` extends the `impurity_codeposition.py`
`BathKinetics` framework to Mn (metallic, Butler–Volmer, and — like Zn —
less noble than Fe so it stays trace in the deposit) and to S, P, Si, B
(anionic/oxy-anion, Langmuir adsorption).  Its `route_steel_grade(c, mn, p,
s, si)` maps a deposit composition to:

* **AISI 1005** — extra-low carbon, low Mn (C ≤ 0.06 %, Mn ≤ 0.35 %);
* **AISI 1018** — low carbon general purpose (C 0.15–0.20 %, Mn ≤ 0.90 %);
* **low-sulfur deep-drawing** — S ≤ 0.02 % and P ≤ 0.02 % (clean enough to
  deep draw); anything above the general S/P ceilings routes to
  "resulfurized / not deep-drawing".

This is what routes a foil toward deep-drawing (clean, low S) versus a
plain 10xx carbon grade.  It is a placeholder by design: real routing needs
measured deposit composition, not the screening estimate.
