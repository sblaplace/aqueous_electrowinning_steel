# Chemistry & Physics Improvements — Round 3 (Beyond Reviews 1 & 2)

> **Scope**: Advanced physical and chemical mechanisms identifying critical gaps in aqueous iron electrowinning modeling **beyond** what is covered in `CHEM_PHYS_REVIEW.md` and `docs/CHEM_PHYS_IMPROVEMENTS_V2.md`.
>
> Where Review 1 focused on screening-to-design leaps in baseline electrochemistry and Review 2 addressed cross-module coupling loops (such as stress–H and neutral pairs), this document addresses **fundamental chemical thermodynamics, electrocrystallization physics, stack-scale electrical hydrodynamics, defect metallurgy, and hydrometallurgical upstream phenomena**.

---

## 1. Aqueous Phase Equilibria & Solution Thermodynamics

### 1.1 Retrograde (Inverse) $\text{FeSO}_4$ Solubility and High-Temperature Szomolnokite Scaling
* **Code Reference**: `models/fe_sulfate_solubility.py`, `tests/test_fe_sulfate_solubility.py`
* **The Physics/Chemistry**:
  Operating electrowinning cells at elevated temperatures (60–90 °C) reduces electrolyte Ohmic resistance and activation overpotentials. However, the binary $\text{FeSO}_4\text{--}\text{H}_2\text{O}$ system possesses a critical thermodynamic phase transition at **56.7 °C**:
  - Below 56.7 °C: **Melanterite** ($\text{FeSO}_4\cdot 7\text{H}_2\text{O}$) is the stable solid, exhibiting normal *prograde* solubility ($\Delta H_{\text{diss}} > 0$).
  - Above 56.7 °C: Melanterite dehydrates to **szomolnokite** ($\text{FeSO}_4\cdot\text{H}_2\text{O}$), which has an exothermic heat of dissolution ($\Delta H_{\text{diss}} < 0$) and exhibits **retrograde (inverse) solubility**:
    $$C_{\text{sat}}(60\text{ }^\circ\text{C}) \approx 1.99\text{ M} \quad \longrightarrow \quad C_{\text{sat}}(75\text{ }^\circ\text{C}) \approx 1.45\text{ M} \quad \longrightarrow \quad C_{\text{sat}}(90\text{ }^\circ\text{C}) \approx 1.00\text{ M}$$
  - **Common-ion depression**: Supporting salts ($\text{Na}_2\text{SO}_4$, $(\text{NH}_4)_2\text{SO}_4$) depress the maximum soluble $\text{Fe}^{2+}$ concentration via the ion product $K'_{\text{sp}}(T) = [\text{Fe}^{2+}] [\text{SO}_4^{2-}] \gamma_\pm^2$.
  - **Heat exchanger fouling**: Heating surfaces operate at $T_{\text{wall}} > T_{\text{bulk}}$. Because of retrograde solubility, **the hottest surface has the lowest solubility**, causing spontaneous crystallization of hard szomolnokite scale on heat exchanger tubes and submerged electric heating elements.
* **Model Deliverable**: `assess_heat_exchanger_scaling()` predicts the maximum safe wall temperature $T_{\text{wall,crit}}$ and supersaturation ratio $S_{\text{wall}} = C_{\text{bulk}} / C_{\text{sat}}(T_{\text{wall}})$.

### 1.2 Transmembrane Chemical Osmosis Driven by Water Activity Gradients ($\Delta a_w$)
* **Code Reference**: `models/chemical_osmosis.py`, `tests/test_chemical_osmosis.py`
* **The Physics/Chemistry**:
  In divided-cell electrowinning, the catholyte (1.5 M $\text{FeSO}_4$ + 0.5 M $\text{Na}_2\text{SO}_4$, $a_w \approx 0.925$) and anolyte (0.5–1.0 M $\text{H}_2\text{SO}_4$, $a_w \approx 0.965$) maintain a significant water chemical potential difference:
  $$\Delta \pi = -\frac{R T}{\bar{V}_w} \ln\left(\frac{a_{w,\text{cath}}}{a_{w,\text{ano}}}\right) \approx 40\text{--}70\text{ bar}$$
  - **Electro-osmotic drag (EOD)** carries water from catholyte to anolyte with migrating protons: $J_{w,\text{eod}} = n_w (j/F) \bar{V}_w$ ($n_w \approx 2.0\text{--}3.0\text{ H}_2\text{O}/\text{H}^+$).
  - **Chemical osmosis** drives water in the reverse direction (anolyte to catholyte): $J_{w,\text{osm}} = L_p \sigma_{\text{refl}} \Delta \pi$.
  - At low current densities ($j < j_{\text{zero}} \approx 14\text{--}25\text{ mA/cm}^2$), chemical osmosis dominates, causing **catholyte volume swelling and dilution**. At high current densities ($j > 50\text{ mA/cm}^2$), EOD dominates, **drying out the catholyte**.
* **Model Deliverable**: `solve_transmembrane_water_flux()` computes the exact isovolemic zero-net-flux current density $j_{\text{zero}}$, closing the long-term cell water balance.

---

## 2. Fundamental Cathode Microkinetics & Electrocrystallization

### 2.1 Bockris–Dražić–Despić (BDD) / Epelboin–Wiart Multi-Step Iron Deposition Kinetics
* **Code Reference**: `models/bdd_kinetics.py`, `tests/test_bdd_kinetics.py`
* **The Physics/Chemistry**:
  Rather than an unphysical elementary 2-electron transfer, cathodic iron electrodeposition proceeds via a sequential catalytic mechanism involving the hydroxo-intermediate $(\text{FeOH})_{\text{ads}}$:
  1. $\text{Fe}^{2+} + \text{H}_2\text{O} \rightleftharpoons \text{FeOH}^+ + \text{H}^+$ ($\log K_{\text{hyd}} \approx -9.50$ at 25 °C)
  2. $\text{FeOH}^+ + e^- \rightleftharpoons (\text{FeOH})_{\text{ads}}$ (adsorption, transfer coefficient $\beta_1$)
  3. $(\text{FeOH})_{\text{ads}} + e^- \rightarrow \text{Fe} + \text{OH}^-$ (crystallization, transfer coefficient $\beta_2$)
* **Kinetic Predictions**:
  - **Positive Reaction Order in Hydroxide**: $p_{\text{OH}^-} = +1.0$ (or $p_{\text{H}^+} = -1.0$). Local boundary-layer alkalization (pH 2.0 $\rightarrow$ 3.5) directly accelerates iron discharge.
  - **Dual Tafel Slopes**: Low overpotentials ($|\eta| < 80\text{ mV}$) yield a slope of **$b \approx 2.303 RT / ((1+\beta_2)F) \approx 40\text{ mV/dec}$**; high overpotentials yield **$b \approx 120\text{ mV/dec}$**.
  - **EIS Inductive Loops**: Surface intermediate relaxation time $\tau_{\theta} = 1 / (k_1^f [\text{FeOH}^+] + k_1^r + k_2^f)$ generates low-frequency inductive loops in impedance spectra.

### 2.2 Substrate Lattice Misfit Strain & Heteroepitaxial Nucleation
* **Code Reference**: `models/adhesion_peel.py`, `models/internal_stress.py`
* **The Physics/Chemistry**:
  Electrodeposition of bcc $\alpha$-Fe ($a = 0.2866\text{ nm}$) onto hcp $\alpha$-Ti drums ($a = 0.2950\text{ nm}$) generates an initial coherency lattice mismatch $f = (a_{\text{Fe}} - a_{\text{sub}})/a_{\text{sub}} \approx -2.8\%$.
  - In the pseudomorphic island growth stage ($h < h_c$), elastic coherency stress reaches **$\sigma_{\text{misfit}} \approx 2.5\text{--}3.2\text{ GPa (tensile)}$**.
  - Beyond the Matthews–Blakeslee critical thickness ($h_c \approx 8\text{--}15\text{ nm}$), misfit dislocations nucleate, relaxing long-range strain while establishing the interfacial fracture energy $G_c$ required for clean peel harvesting.

---

## 3. Pulse Power Electronics & Hydrodynamics

### 3.1 Double-Layer Capacitive Charging ($RC$ Filtering) in Pulse Plating
* **Code Reference**: `models/pulse_rc_filter.py`, `tests/test_pulse_rc_filter.py`
* **The Physics/Chemistry**:
  The cell interface contains a double-layer capacitance $C_{\text{dl}}$ ($20\text{--}60\text{ }\mu\text{F/cm}^2$) in series with uncompensated solution resistance $R_\Omega$. The cell time constant $\tau_{\text{cell}} = R_\Omega C_{\text{dl}} \approx 0.05\text{--}1.0\text{ ms}$ rounds off rectangular current pulses:
  $$j_F(t) = j_{\text{peak}} \left[1 - \exp\left(-\frac{t}{\tau_{\text{cell}}}\right)\right] + j_{\text{min}} \exp\left(-\frac{t}{\tau_{\text{cell}}}\right)$$
  - Above the 3dB cutoff frequency $f_{\text{cutoff}} = 1 / (2\pi R_\Omega C_{\text{dl}})$, the pulse collapses into attenuated ripple-DC.
* **Model Deliverable**: `max_practical_frequency_Hz()` bounds the pulse optimization search space to physically realizable frequencies where Faradaic peak current fidelity $\ge 85\%$.

### 3.2 Coupled Solutal Buoyancy & Mixed Convection ($Gr_m / Re^2$)
* **Code Reference**: `models/boundary_layer.py`, `models/transport.py`
* **The Physics/Chemistry**:
  Depletion of dense $\text{Fe}^{2+}$ ($M = 55.85\text{ g/mol}$) at the vertical cathode reduces local electrolyte density by $\Delta \rho \approx -30\text{ to }-70\text{ kg/m}^3$, driving strong upward solutal natural convection ($Gr_m = g \beta_c \Delta C H^3 / \nu^2$).
  - In downward forced-flow cells, opposing buoyant and forced flows create **flow reversal, stagnation zones, and local starvation** when the Richardson number $Ri_m = Gr_m / Re^2 \approx 1$.

---

## 4. Stack & Reactor Engineering

### 4.1 Manifold Electrical Shunt Currents in Multi-Cell Stacks
* **Code Reference**: `models/shunt_currents.py`, `tests/test_shunt_currents.py`
* **The Physics/Chemistry**:
  In a 50-cell series crate ($V_{\text{stack}} = 100\text{--}150\text{ V}$), common electrolyte manifolds act as conductive shunt paths (Kaminski–Gileadi / Waha resistor network).
  - Drives Faradaic bypass loss and causes current maldistribution between center and end cells.
  - Generates high exit port current densities ($j_{\text{port}} > 25\text{ mA/cm}^2$), triggering severe **electrolytic pitting and port perforation** at high-potential end cells.
* **Model Deliverable**: `solve_stack_shunt_currents()` solves the tridiagonal resistor network and sizes port aspect ratios ($L_p / A_p$) to mitigate port corrosion when $j_{\text{port}} > 25\text{ mA/cm}^2$.

---

## 5. Physical Metallurgy & Post-Processing Physics

### 5.1 Hierarchical Hydrogen Traps & McNabb–Foster Bakeout Kinetics
* **Code Reference**: `models/hydrogen_trapping.py`, `tests/test_hydrogen_trapping.py`
* **The Physics/Chemistry**:
  Hydrogen in electrodeposited bcc iron is partitioned between:
  1. **Reversible (Diffusible) Traps** ($E_b = 20\text{--}35\text{ kJ/mol}$): Dislocations and low-angle grain boundaries that exchange hydrogen with the lattice at room temperature and cause delayed cracking.
  2. **Irreversible (Deep) Traps** ($E_b = 55\text{--}95\text{ kJ/mol}$): Cementite and oxide interfaces that retain hydrogen harmlessly below 350 °C.
  - During de-embrittlement baking (150–220 °C, ASTM F519), hydrogen transport obeys the **McNabb–Foster / Oriani diffusion-effusion equations** with temperature-dependent effective diffusivity $D_{\text{eff}}(T)$.
* **Model Deliverable**: `compute_bakeout_schedule()` calculates the exact baking time required to reduce mobile diffusible hydrogen below $C_{H,\text{diff}} < 0.10\text{ ppm wt}$.

---

## 6. Upstream Hydrometallurgy & Primary Feedstock

### 6.1 Primary Ore Leaching Kinetics & Reductive Leaching Mechanics
* **Code Reference**: `models/ore_leaching.py`, `tests/test_ore_leaching.py`
* **The Physics/Chemistry**:
  Primary iron ores (hematite $\alpha\text{-Fe}_2\text{O}_3$, magnetite $\text{Fe}_3\text{O}_4$, goethite $\alpha\text{-FeOOH}$) dissolve in acid via the **Shrinking Core Model (SCM)**:
  $$1 - (1 - X)^{1/3} = \frac{k_{\text{chem}} C_{\text{acid}}^n}{\rho_m r_0} t \quad (\text{chemical reaction control})$$
  - Direct acid leaching of hematite is sluggish ($E_a \approx 80\text{ kJ/mol}$).
  - **Reductive leaching** with scrap Fe⁰ or $\text{SO}_2$ reduces surface $\text{Fe}^{3+} \rightarrow \text{Fe}^{2+}$, accelerating dissolution rates by **10–50×** and directly generating $\text{Fe}^{2+}$ feed.
* **Model Deliverable**: `simulate_ore_leaching()` sizes leach vessel residence times ($t_{\text{res}} = 2\text{--}6\text{ hr}$) and acid stoichiometry.

---

## Priority & Implementation Summary

| Module | Core Physics Delivered | Primary Metric Impact |
|---|---|---|
| `fe_sulfate_solubility.py` | Retrograde szomolnokite solubility & common-ion limits | Prevents heat-exchanger fouling at $T > 56.7\text{ }^\circ\text{C}$ |
| `pulse_rc_filter.py` | Double-layer capacitive charging & cutoff frequency | Bounds pulse frequency to maintain Faradaic fidelity |
| `bdd_kinetics.py` | Multi-step BDD catalytic FeOH⁺ mechanism | Dual Tafel slopes (40/120 mV/dec) & pH reaction order |
| `shunt_currents.py` | Discrete resistor ladder network in crate stacks | Stack Faradaic efficiency & port corrosion mitigation |
| `hydrogen_trapping.py` | Reversible/irreversible trap hierarchy & Oriani bakeout | De-embrittlement baking time for structural steel |
| `ore_leaching.py` | Shrinking core kinetics & reductive leaching | Sizing of primary ore dissolution reactors in Dark Mill |
| `chemical_osmosis.py` | Water activity gradient ($\Delta a_w$) & osmotic flux | Long-term cell volume stability & zero-net-flux window |
| `tempering_kinetics.py` | 4-stage tempering, LSW Ostwald ripening & Orowan bypass | Predicts carbide coarsening, DBTT, and Charpy impact energy |
| `solutal_convection.py` | Solutal density depletion & Grashof mixed convection | Predicts flow reversal and critical downflow velocity |

— *Round 3 Physics & Chemistry Review, August 2026.*
