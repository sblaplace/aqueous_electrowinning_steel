#!/usr/bin/env python3
"""H adsorption on Fe surfaces using Skala ML XC functional.

Computes ΔG_ads(H) on BCC Fe(110), Fe(100), and Fe(111) at all
high-symmetry adsorption sites. Uses PySCF + Skala (CPU).

Reference:
  Jiang & Carter (2004) Surface Science 570:167 — ΔG_ads(H) on Fe(110) ≈ -0.40 eV
  Hammer & Nørskov (1995) — d-band model for H adsorption on transition metals

Usage:
    python3 scripts/dft_h_adsorption_fe.py [--slab-size 2x2] [--quick]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# PySCF imports
from pyscf import gto, scf
from skala.pyscf import SkalaKS


# ─── Physical Constants ─────────────────────────────────────────

BOHR_TO_ANG = 0.529177
HARTREE_TO_EV = 27.2114
HARTREE_TO_KCAL = 627.509
ZPE_H2_EV = 0.27  # ZPE of H2 ≈ 0.27 eV (experimental)
ENTROPY_H2_60C_EV = 0.41  # -TΔS for H2 at 60°C, 1 atm ≈ 0.41 eV
# For adsorbed H: ZPE ≈ 0.06 eV (hindered vibration on surface)
ZPE_H_ADS_EV = 0.06


# ─── Fe BCC Slab Builder ────────────────────────────────────────

# BCC Fe lattice constant: 2.87 Å
FE_LATTICE_A = 2.87  # Å

# BCC(110): rectangular unit cell
# a1 = a/sqrt(2) along [1-10], a2 = a along [001]
# Surface atom positions within the rectangular cell:
# Layer 0: (0,0), (a1/2, a2/2) — two atoms per layer
# Layer 1: (a1/2, 0), (0, a2/2) — shifted
# Layer 2: same as layer 0

def build_fe_slab(miller: int, nx: int = 2, ny: int = 2, nlayers: int = 4,
                  vacuum: float = 12.0) -> gto.Mole:
    """Build a Fe surface slab as a PySCF Mole (cluster model with vacuum).

    This uses ASE to generate the atomic positions, then converts to PySCF.
    The vacuum is large enough that periodic images don't interact.
    """
    from ase.build import bcc110, bcc100, bcc111

    builders = {110: bcc110, 100: bcc100, 111: bcc111}
    if miller not in builders:
        raise ValueError(f"Unsupported Miller index: {miller}")

    slab = builders[miller]('Fe', size=(nx, ny, nlayers), vacuum=vacuum,
                             a=FE_LATTICE_A)

    return slab


def slab_to_pyscf_mol(slab, basis: str = 'def2-svp', charge: int = 0) -> gto.Mole:
    """Convert ASE Atoms to PySCF Mole."""
    atom_str = ''
    for sym, pos in zip(slab.get_chemical_symbols(), slab.get_positions()):
        atom_str += f'{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}; '

    mol = gto.M(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=0,  # high-spin Fe will be handled by UKS if needed
        unit='Angstrom',
    )
    return mol


def add_h_adatom(slab, site: str, miller: int, height: float = 1.7):
    """Add H adatom to a slab at a specified high-symmetry site.

    Returns a new ASE Atoms object with H added.
    """
    from ase.build import add_adsorbate
    import copy

    s = copy.deepcopy(slab)

    # Get surface cell vectors
    cell = s.cell
    a1 = cell[0]
    a2 = cell[1]

    if miller == 110:
        if site == 'top':
            # On top of a surface Fe atom
            surf_pos = s.positions[s.positions[:, 2].argmax()]
            h_pos = surf_pos + [0, 0, height]
        elif site == 'long_bridge':
            # Bridge between two Fe along [1-10] direction
            top_z = s.positions[:, 2].max()
            top_layer = s.positions[s.positions[:, 2] > top_z - 0.5]
            if len(top_layer) >= 2:
                h_pos = (top_layer[0] + top_layer[1]) / 2 + [0, 0, height]
            else:
                h_pos = [a1[0]/2, 0, top_z + height]
        elif site == 'short_bridge':
            # Bridge between two Fe along [001] direction
            top_z = s.positions[:, 2].max()
            h_pos = [0, a2[1]/2, top_z + height]
        elif site == 'hollow':
            # 4-fold hollow site
            top_z = s.positions[:, 2].max()
            h_pos = [a1[0]/2, a2[1]/2, top_z + height]
        else:
            raise ValueError(f"Unknown site: {site}")
    elif miller == 100:
        top_z = s.positions[:, 2].max()
        if site == 'top':
            surf_pos = s.positions[s.positions[:, 2].argmax()]
            h_pos = surf_pos + [0, 0, height]
        elif site == 'bridge':
            top_layer = s.positions[s.positions[:, 2] > top_z - 0.5]
            if len(top_layer) >= 2:
                h_pos = (top_layer[0] + top_layer[1]) / 2 + [0, 0, height]
            else:
                h_pos = [a1[0]/2, 0, top_z + height]
        elif site == 'hollow':
            h_pos = [a1[0]/2, a2[1]/2, top_z + height]
        else:
            raise ValueError(f"Unknown site: {site}")
    elif miller == 111:
        top_z = s.positions[:, 2].max()
        if site == 'top':
            surf_pos = s.positions[s.positions[:, 2].argmax()]
            h_pos = surf_pos + [0, 0, height]
        elif site == 'bridge':
            top_layer = s.positions[s.positions[:, 2] > top_z - 0.5]
            if len(top_layer) >= 2:
                h_pos = (top_layer[0] + top_layer[1]) / 2 + [0, 0, height]
            else:
                h_pos = [a1[0]/2, 0, top_z + height]
        elif site == 'fcc':
            h_pos = [a1[0]/3, a2[1]/3, top_z + height]
        elif site == 'hcp':
            h_pos = [2*a1[0]/3, 2*a2[1]/3, top_z + height]
        else:
            raise ValueError(f"Unknown site: {site}")
    else:
        raise ValueError(f"Unsupported Miller: {miller}")

    # Add H atom
    s.append('H')
    s.positions[-1] = h_pos
    return s


# ─── Calculation Engine ──────────────────────────────────────────

def run_scf(slab_atoms, basis: str = 'def2-svp', label: str = '',
            quick: bool = False) -> dict:
    """Run Skala SCF on an ASE Atoms object.

    Returns dict with energy, convergence status, timing.
    """
    mol = slab_to_pyscf_mol(slab_atoms, basis=basis)

    # Use unrestricted KS for Fe (open-shell d-electrons)
    ks = SkalaKS(mol, xc='skala-1.1')
    if quick:
        ks.max_cycle = 50
        ks.conv_tol = 1e-6
    else:
        ks.max_cycle = 200
        ks.conv_tol = 1e-8

    t0 = time.time()
    try:
        E = ks.kernel()
        dt = time.time() - t0
        converged = ks.converged
        return {
            'energy_hartree': float(E),
            'energy_eV': float(E * HARTREE_TO_EV),
            'converged': bool(converged),
            'time_seconds': dt,
            'label': label,
            'n_atoms': len(slab_atoms),
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            'error': str(e),
            'converged': False,
            'time_seconds': dt,
            'label': label,
            'n_atoms': len(slab_atoms),
        }


def compute_h2_reference(basis: str = 'def2-svp') -> dict:
    """Compute H2 energy as reference."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74',
        basis=basis,
        spin=0,
        unit='Angstrom',
    )
    ks = SkalaKS(mol, xc='skala-1.1')
    E = ks.kernel()
    return {
        'energy_hartree': float(E),
        'energy_eV': float(E * HARTREE_TO_EV),
        'converged': bool(ks.converged),
        'label': 'H2',
    }


def compute_adsorption_energy(E_slab_H: float, E_slab: float, E_H2: float) -> float:
    """ΔE_ads = E(slab+H) - E(slab) - E(H2)/2

    Negative = exothermic adsorption.
    """
    return E_slab_H - E_slab - E_H2 / 2.0


def compute_gibbs_adsorption(dE_ads_eV: float) -> float:
    """ΔG_ads = ΔE_ads + ZPE(H_ads) - ZPE(H2)/2 - (-TΔS_H2/2)

    Accounts for zero-point energy and entropy at 60°C.
    """
    dG = (dE_ads_eV
          + ZPE_H_ADS_EV           # ZPE of adsorbed H (hindered)
          - ZPE_H2_EV / 2          # ZPE of gas-phase H2 (split between 2 H)
          + ENTROPY_H2_60C_EV / 2  # -TΔS term for H2 (favors desorption)
    )
    return dG


# ─── Main Calculation ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='H adsorption on Fe surfaces')
    parser.add_argument('--miller', type=int, nargs='+', default=[110, 100, 111],
                        help='Miller indices to compute')
    parser.add_argument('--nx', type=int, default=2, help='Supercell x')
    parser.add_argument('--ny', type=int, default=2, help='Supercell y')
    parser.add_argument('--nlayers', type=int, default=4, help='Number of layers')
    parser.add_argument('--basis', type=str, default='def2-svp', help='Basis set')
    parser.add_argument('--quick', action='store_true', help='Loose convergence')
    parser.add_argument('--output', type=str, default='experiments/data/dft_h_adsorption.json')
    args = parser.parse_args()

    results = {
        'method': 'Skala-1.1 / PySCF',
        'basis': args.basis,
        'slab_size': f'{args.nx}x{args.ny}x{args.nlayers}',
        'lattice_constant_A': FE_LATTICE_A,
        'T_K': 333.15,  # 60°C
        'h2': {},
        'surfaces': {},
    }

    # Step 1: H2 reference
    print("Computing H2 reference...")
    h2 = compute_h2_reference(args.basis)
    results['h2'] = h2
    print(f"  H2: {h2['energy_hartree']:.6f} Ha ({h2['energy_eV']:.4f} eV), "
          f"converged={h2['converged']}")

    # Step 2: Bare slabs
    for miller in args.miller:
        print(f"\n{'='*60}")
        print(f"Fe({miller}) bare slab {args.nx}x{args.ny}x{args.nlayers}")
        slab = build_fe_slab(miller, args.nx, args.ny, args.nlayers)
        slab_result = run_scf(slab, args.basis, f'Fe({miller}) bare', args.quick)
        print(f"  E = {slab_result.get('energy_hartree', 'FAILED'):.6f} Ha, "
              f"converged={slab_result.get('converged')}, "
              f"time={slab_result.get('time_seconds', 0):.0f}s")

        if not slab_result.get('converged'):
            print(f"  WARNING: bare slab did not converge!")
            results['surfaces'][str(miller)] = {'bare': slab_result, 'sites': {}}
            continue

        E_slab = slab_result['energy_hartree']
        results['surfaces'][str(miller)] = {'bare': slab_result, 'sites': {}}

        # Step 3: H adsorption at each site
        if miller == 110:
            sites = ['top', 'long_bridge', 'short_bridge', 'hollow']
        elif miller == 100:
            sites = ['top', 'bridge', 'hollow']
        else:
            sites = ['top', 'bridge', 'fcc', 'hcp']

        for site in sites:
            print(f"\n  H on Fe({miller}) @ {site}...")
            slab_h = add_h_adatom(slab, site, miller)
            slab_h_result = run_scf(slab_h, args.basis,
                                    f'Fe({miller})+H @ {site}', args.quick)
            print(f"    E = {slab_h_result.get('energy_hartree', 'FAILED'):.6f} Ha, "
                  f"converged={slab_h_result.get('converged')}, "
                  f"time={slab_h_result.get('time_seconds', 0):.0f}s")

            if slab_h_result.get('converged'):
                dE = compute_adsorption_energy(
                    slab_h_result['energy_hartree'], E_slab, h2['energy_hartree']
                )
                dG = compute_gibbs_adsorption(dE * HARTREE_TO_EV)
                slab_h_result['dE_ads_eV'] = dE * HARTREE_TO_EV
                slab_h_result['dG_ads_eV'] = dG
                print(f"    ΔE_ads = {dE * HARTREE_TO_EV:.4f} eV")
                print(f"    ΔG_ads = {dG:.4f} eV  (ZPE + TΔS corrected, 60°C)")

            results['surfaces'][str(miller)]['sites'][site] = slab_h_result

    # Save results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {out}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Site':<25} {'ΔE_ads (eV)':>12} {'ΔG_ads (eV)':>12} {'Time (s)':>10}")
    print(f"{'='*70}")
    for miller_str, surface in results['surfaces'].items():
        print(f"\nFe({miller_str}):")
        print(f"  {'bare slab':<23} {surface['bare'].get('energy_hartree', 0):>12.6f} Ha"
              f"{' '*12}{surface['bare'].get('time_seconds', 0):>10.0f}")
        for site, r in surface.get('sites', {}).items():
            if r.get('converged'):
                print(f"  {site:<23} {r.get('dE_ads_eV', 0):>12.4f} "
                      f"{r.get('dG_ads_eV', 0):>12.4f} "
                      f"{r.get('time_seconds', 0):>10.0f}")
            else:
                print(f"  {site:<23} {'FAILED':>12} {'':>12} "
                      f"{r.get('time_seconds', 0):>10.0f}")


if __name__ == '__main__':
    main()
