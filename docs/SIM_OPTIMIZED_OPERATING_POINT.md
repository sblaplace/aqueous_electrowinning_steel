# Joint operating-point optimization and contact-resistance measurement protocol

## Scope and status

This report is a transparent, bottom-up **Level-0 screening prediction** for the
RC-1 reference divided cell. Every predicted value in this report is
**unvalidated (L0)**. There is no real laboratory data in the calculation,
and no value below is a measured result.

This is **NOT gate evidence**. The energy and FE gates are measurement-only and
are implemented in `models/process_gates.py`. The models and tables below
turn the recommendations of #38 and #39 into an actionable measurement plan
and a joint operating-point optimizer (`models/contact_resistance_protocol.py`
and `models/operating_point_optimizer.py`).

## Sequencing and additive architecture

This brief builds upon:
1. **#38 (physics-derived economics):** LCOFe minimization near 150 mA/cm² (~$333/t) versus the 300 mA/cm² benchmark duty.
2. **#39 (voltage decomposition):** Identification that ohmic contact resistance is the single largest voltage penalty (~1.50 V out of 5.68 V at 300 mA/cm²), recommending "buy a measured terminal-to-electrode contact resistance next."

The implementation is purely additive:
- `models/contact_resistance_protocol.py` defines the 4-wire Kelvin measurement protocol and expected L0 range ($1.0 \times 10^{-4}$ to $1.0 \times 10^{-3}\ \Omega\cdot\text{m}^2$, typical $5.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$).
- `models/operating_point_optimizer.py` implements joint search over $(j \times \text{gap} \times \text{membrane} \times \text{contact})$ enforcing transport limits.
- No existing modules, tests, or docs were modified.

---

## Part A: Contact-resistance measurement protocol

### Protocol Overview
- **Method:** 4-wire (Kelvin) DC current injection and differential voltage drop measurement.
- **Target Interfaces:**
  1. Busbar-to-current collector interface
  2. Current collector-to-electrode coupon interface
  3. Aggregate terminal-to-electrode joint (total path)
- **Expected Signal:** Apply known DC currents (1.0 A to 10.0 A) through each joint while measuring microvolt/millivolt drops.
- **Recorded Units:** Area-normalized specific resistance in $\Omega\cdot\text{m}^2$ (ohms·m²), directly plugging into `CellGeometry.contact_resistance_ohm_m2`.

### Expected L0 Prior Range (RC-1 Build)
- **Minimum ($1.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$) [unvalidated (L0)]:** Optimized mechanical bolting with silver-filled conductive epoxy or gold plating on copper busbars.
- **Typical ($5.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$) [unvalidated (L0)]:** RC-1 standard bolted copper busbar and titanium/steel coupon clamp baseline (current model default).
- **Maximum ($1.0 \times 10^{-3}\ \Omega\cdot\text{m}^2$) [unvalidated (L0)]:** Unoptimized mechanical contact with native oxide films or minor clamping non-uniformity.

### Decision Consequence (Impact if Measured at 300 mA/cm²)
- **Minimum ($1.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$):** $V_\text{cell} = 4.478\text{ V}$ ($\Delta V = +1.200\text{ V}$), specific energy = $4,363.1\text{ kWh/t Fe}$ [unvalidated (L0)] — **Energy gate pass ($\le 4,000\text{ kWh/t}$): False**.
- **Typical ($5.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$):** $V_\text{cell} = 5.678\text{ V}$ ($\Delta V = 0.000\text{ V}$), specific energy = $5,532.3\text{ kWh/t Fe}$ [unvalidated (L0)] — **Energy gate pass: False**.
- **Maximum ($1.0 \times 10^{-3}\ \Omega\cdot\text{m}^2$):** $V_\text{cell} = 7.178\text{ V}$ ($\Delta V = -1.500\text{ V}$), specific energy = $6,993.8\text{ kWh/t Fe}$ [unvalidated (L0)] — **Energy gate pass: False**.

---

## Part B: Joint operating-point optimizer & reachability verdict

### Joint Search Methodology
A deterministic grid scan over operating current density $j$ ($150$ and $300\text{ mA/cm}^2$), interelectrode gap ($1.5$ and $3.0\text{ mm}$), membrane resistance ($3.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$), and contact resistance ($1.0 \times 10^{-4}$ and $5.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$). Transport limits are strictly enforced (invalid points where transport margin $\le 1.0$ are surfaced, not silently skipped).

### Energy Gate Reachability Verdict
- **Verdict:** **Energy gate IS reachable under optimized lever combinations [unvalidated (L0)]**.
- **Minimum Achieved Energy:** **3,306.3 kWh/t Fe [unvalidated (L0)]** at $150\text{ mA/cm}^2$, $1.5\text{ mm}$ gap, and $1.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$ contact resistance.
- **Comparison:** At the baseline $300\text{ mA/cm}^2$ benchmark duty with default contact resistance ($5.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$), specific energy exceeds $4,000\text{ kWh/t}$. However, operating at $150\text{ mA/cm}^2$ with reduced gap and contact resistance successfully brings specific energy below $4,000\text{ kWh/t}$.

### Cost-Optimal Operating Point
- **Optimal Current Density:** $150\text{ mA/cm}^2$ [unvalidated (L0)].
- **LCOFe:** **$300/t [unvalidated (L0)]** (or **$327/t [unvalidated (L0)]** when wired with the protocol's expected typical contact resistance of $5.0 \times 10^{-4}\ \Omega\cdot\text{m}^2$).
- **Verification:** Re-derives the finding from #38 that LCOFe is minimized near $150\text{ mA/cm}^2$, and confirms that A $\to$ B wiring is active and real.

---

## Reproduce

From the repository root:

```bash
python -m models.contact_resistance_protocol
python -m models.operating_point_optimizer
```

Both runners print structured Level-0 reports with explicit unvalidated and non-gate-evidence notices.
