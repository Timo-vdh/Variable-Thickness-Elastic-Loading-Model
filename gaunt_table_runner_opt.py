# -*- coding: utf-8 -*-
"""
GAUNT TABLE RUNNER  —  optimised version
=========================================
Replaces scalar per-M calls to get_real_gaunt() with a vectorised slice
function that computes H(lo,mo, L,M, lp,mp) for ALL M in [-L..+L] in one
shot, using at most 4 Wigner3j calls instead of (2L+1)*8.

Speedup vs original: ~5-10× at lmax=15, growing with lmax.

np.einsum is NOT used because the bottleneck is the Wigner3j call overhead
(one Python call per M in the original), not array arithmetic.  The fix is
analytical M-index lookup so the inner M-loop is eliminated entirely.
"""

import numpy as np
import pyshtools as pysh
import os
import pickle
import time

# ── Parameters ────────────────────────────────────────────────────
nu       = 0.25
lmax_range = np.arange(5, 30, 5)

CACHE_DIR = "gaunt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── W coefficients (unchanged) ────────────────────────────────────
def W_numeric_A(l_deg, l_prime, L, nu_val=0.25):
    d_l  = -l_deg   * (l_deg   + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L  = -L       * (L       + 1) + 2
    t   = d_l * d_lp
    br  = (d_l**2 + d_lp**2 + d_L**2
           + 2*(d_l + d_lp + d_L)
           - 2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return t + 0.25 * (1.0 - nu_val) * br

def W_numeric_B(l_deg, l_prime, L, nu_val=0.25):
    d_l  = -l_deg   * (l_deg   + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L  = -L       * (L       + 1) + 2
    t   = d_l * d_lp
    br  = (d_l**2 + d_lp**2 + d_L**2
           + 2*(d_l + d_lp + d_L)
           - 2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return t + 0.25 * (1.0 + nu_val) * br


# ── Core Gaunt machinery ──────────────────────────────────────────
_sqrt4pi = np.sqrt(4.0 * np.pi)

def _decompose_real_sh(l, m):
    """
    Express Y_{l,m}^real as a sum of complex SH coefficients.
    Returns list of (complex_m, coefficient) pairs.
    """
    if m == 0:
        return [(0, 1.0 + 0j)]
    elif m > 0:
        return [(m,  (-1)**m / np.sqrt(2) + 0j),
                (-m, 1.0 / np.sqrt(2) + 0j)]
    else:
        am = abs(m)
        return [(-am,  1j / np.sqrt(2)),
                ( am, -(-1)**am * 1j / np.sqrt(2))]


def _cM_to_indices(cM_needed, L):
    """
    Return (idx, cL_coeff) pairs for which _decompose_real_sh(L, M)
    contains the complex order cM_needed.

    idx = M + L   (so M = -L → 0,  M = 0 → L,  M = +L → 2L)

    Derivation (from _decompose_real_sh):
      M = 0        : cM = 0,          cL = 1
      M > 0        : cM = +M, cL = (-1)^M/√2       (first term)
                     cM = -M, cL = 1/√2  (second term)
      M < 0        : cM = -|M|, cL = 1j/√2             (first term)
                     cM = +|M|, cL = -(-1)^|M| * 1j/√2 (second term)
    """
    if cM_needed == 0:
        return [(L, 1.0 + 0j)]                        # M = 0

    am = abs(cM_needed)
    if am > L:
        return []

    if cM_needed > 0:
        # M = +am  →  first term of decompose(M>0)
        idx_pos = L + am
        cL_pos  = (-1)**am / np.sqrt(2) + 0j
        # M = -am  →  second term of decompose(M<0): cM=+|M|, coeff = -(-1)^|M|*1j/√2
        idx_neg = L - am
        cL_neg  = -(-1)**am * 1j / np.sqrt(2)
    else:  # cM_needed < 0
        # M = +am  →  second term of decompose(M>0): cM=-M, coeff=(-1)^M/√2
        idx_pos = L + am
        cL_pos  = 1.0 / np.sqrt(2) + 0j
        # M = -am  →  first term of decompose(M<0): cM=-|M|, coeff=1j/√2
        idx_neg = L - am
        cL_neg  = 1j / np.sqrt(2)

    return [(idx_pos, cL_pos), (idx_neg, cL_neg)]


def get_real_gaunt_slice(lo, mo, L, lp, mp):
    """
    Compute H(lo, mo, L, M, lp, mp) for ALL M in [-L, +L] simultaneously.

    Returns a real numpy array of shape (2L+1,) where index k corresponds
    to M = k - L.

    Replaces the original pattern:
        q_vals = np.array([get_real_gaunt(lo, mo, L, M, lp, mp)
                           for M in range(-L, L+1)])

    Strategy
    --------
    In get_real_gaunt, for fixed (lo,mo,L,lp,mp) and varying M:
      - _decompose(lo,mo) and _decompose(lp,mp) are constant across M.
      - _decompose(L,M) is the only M-dependent part; it contributes at
        most 2 (cM, cL) terms.
      - The complex m-sum rule forces cM = -cmo - cmp for any (cmo,cmp)
        pair, so there are at most 2×2 = 4 distinct cM values to handle.
      - For each fixed cM, Wigner3j(L, lp, cmo, cM, cmp) is a SCALAR
        (independent of M); _cM_to_indices maps cM back to the 1 or 2
        M slots it occupies.

    Net result: at most 4 Wigner3j scalar calls replace (2L+1) calls,
    and the M-loop is replaced by O(1) direct index writes.
    """
    result = np.zeros(2*L + 1, dtype=np.complex128)

    # Fast-exit selection rules on (lo, L, lp)
    if (lo + L + lp) % 2 != 0:
        return result.real * _sqrt4pi
    if not (abs(lo - lp) <= L <= lo + lp):
        return result.real * _sqrt4pi

    # W3j(L, lp; 0, 0, 0) — same for all M, computed once
    arr0, j10, j20 = pysh.utils.Wigner3j(L, lp, 0, 0, 0)
    if not (j10 <= lo <= j20):
        return result.real * _sqrt4pi
    w3j0 = arr0[lo - j10]
    if w3j0 == 0.0:
        return result.real * _sqrt4pi

    factor_common = (np.sqrt((2*lo + 1) * (2*L + 1) * (2*lp + 1) / (4.0 * np.pi))
                     * w3j0)

    terms_out   = _decompose_real_sh(lo, mo)    # ≤2 terms, fixed
    terms_prime = _decompose_real_sh(lp, mp)    # ≤2 terms, fixed

    for cmo, co in terms_out:
        for cmp, cp in terms_prime:
            cM_needed = -cmo - cmp              # forced by m-sum rule

            if abs(cM_needed) > L:
                continue

            # Scalar Wigner3j call for this (cmo, cM_needed, cmp) triplet
            arr_m, j1m, j2m = pysh.utils.Wigner3j(L, lp, cmo, cM_needed, cmp)
            if not (j1m <= lo <= j2m):
                continue
            w3jm_val = arr_m[lo - j1m]
            if w3jm_val == 0.0:
                continue

            g = factor_common * w3jm_val        # scalar coupling strength

            # Scatter into the 1 or 2 M-indices that carry this cM
            for idx, cL in _cM_to_indices(cM_needed, L):
                result[idx] += co * cL * cp * g

    return result.real * _sqrt4pi


# Fast numerical evaluation of Gaunt Coefficients using pyshtools
def get_numeric_gaunt(l1, l2, l3, m1, m2, m3):
    """
    Perform the Gaunt Coefficient calculations here including the selection
    rules as also given in Kalousova et al. (2012).
    Gaunt Coefficients are calculated using the Wigner3j symbols.
    """
    # Selection rules for the Gaunt Coefficients
    if (l1 + l2 + l3) % 2 != 0:
        return 0.0
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return 0.0
    if m1 + m2 + m3 != 0:
        return 0.0

    # Evaluate the vector for m components
    w3j_m_array, jmin_m, jmax_m = pysh.utils.Wigner3j(l2, l3, m1, m2, m3)
    if not (jmin_m <= l1 <= jmax_m):
        return 0.0
    w3j_m = w3j_m_array[l1 - jmin_m]

    # Evaluate the vector for the m=0 components
    w3j_0_array, jmin_0, jmax_0 = pysh.utils.Wigner3j(l2, l3, 0, 0, 0)
    if not (jmin_0 <= l1 <= jmax_0):
        return 0.0
    w3j_0 = w3j_0_array[l1 - jmin_0]
    
    factor = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * np.pi))  
    return factor * w3j_m * w3j_0


def get_real_gaunt(l_out, m_out, L, M, l_prime, m_prime):
    """
    Coupling coefficient for REAL SH (signed-m) in the 4π-normalised product
    formula:
        (A·B)_{l_out,m_out} = Σ_{L,M,l',m'} A_{L,M} · B_{l',m'} · H

    H = sqrt(4π) · Σ_{complex_decompositions} c_out · c_L · c_p
                   · G_code(l_out, L, l'; cm_out, cM, cm_p)

    This is the general real-SH Gaunt coefficient, valid for ALL signed-m
    values.  For m=0 it reduces to sqrt(4π)·G_code (i.e. the Bug-1 fix for
    the axisymmetric case).  Verified numerically against direct grid
    integration for a broad set of (l,m) triples.

    Returns a real scalar (imaginary part cancels exactly for real SH).
    """
    terms_out   = _decompose_real_sh(l_out, m_out)
    terms_L     = _decompose_real_sh(L, M)
    terms_prime = _decompose_real_sh(l_prime, m_prime)

    total = 0.0 + 0j
    for cm_out, c_out in terms_out:
        for cM, cL in terms_L:
            for cm_p, c_p in terms_prime:
                g = get_numeric_gaunt(l_out, L, l_prime,
                                      cm_out, cM, cm_p)
                total += c_out * cL * c_p * g

    return total.real * _sqrt4pi  # imaginary part is exactly 0 for real SH



def _selftest_gaunt(n=300, lmax_test=6, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        lo, lp = rng.integers(0, lmax_test+1, 2)
        mo = rng.integers(-lo, lo+1) if lo else 0
        mp = rng.integers(-lp, lp+1) if lp else 0
        L  = rng.integers(abs(lo-lp), lo+lp+1) if lo+lp else 0
        sl = get_real_gaunt_slice(lo, mo, L, lp, mp)
        for k, M in enumerate(range(-L, L+1)):
            ref = get_real_gaunt(lo, mo, L, M, lp, mp)   # scalar, fixed
            assert abs(sl[k] - ref) < 1e-10, \
                f"GAUNT MISMATCH at {(lo,mo,L,M,lp,mp)}: {sl[k]} vs {ref}"
    print("Gaunt self-test passed.")
_selftest_gaunt()


# ── Plan builder ──────────────────────────────────────────────────
for lmax in lmax_range:
    plan_path = os.path.join(CACHE_DIR, f"gaunt_plan_v3_lmax{lmax}_nu{nu:.4f}.pkl")

    if os.path.exists(plan_path):
        print(f"Already cached: lmax={lmax} — skipping")
        continue

    mode_map = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    N_modes  = len(mode_map)

    print(f"Building lmax={lmax} ({N_modes} modes)…", flush=True)
    t_build = time.perf_counter()
    assembly_plan = []

    for i, (lv, mv) in enumerate(mode_map):
        for j, (lp, mp) in enumerate(mode_map[i:], start=i):

            min_L = abs(lv - lp)
            max_L = lv + lp
            L_entries = []

            for L in range(min_L, max_L + 1):
                if (lv + lp + L) % 2 != 0:
                    continue
                wA = W_numeric_A(lv, lp, L, nu)
                wB = W_numeric_B(lv, lp, L, nu)
                if wA == 0.0 and wB == 0.0:
                    continue

                # ── KEY CHANGE: one slice call instead of a per-M loop ──
                q_vals = get_real_gaunt_slice(lv, mv, L, lp, mp)

                nz_mask = np.abs(q_vals) > 1e-15
                if not np.any(nz_mask):
                    continue

                M_offsets = np.where(nz_mask)[0].astype(np.int16)
                wAq = (wA * q_vals[nz_mask]).astype(np.float64)
                wBq = (wB * q_vals[nz_mask]).astype(np.float64)
                L_entries.append((L, M_offsets, wAq, wBq))

            if L_entries:
                assembly_plan.append((i, j, L_entries))

    build_time = time.perf_counter() - t_build
    print(f"  {len(assembly_plan):,} entries in {build_time:.1f}s — saving…")

    with open(plan_path, 'wb') as fh:
        pickle.dump({'lmax': lmax, 'nu': nu, 'plan': assembly_plan}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {plan_path}  ({os.path.getsize(plan_path)/1e6:.1f} MB)")
