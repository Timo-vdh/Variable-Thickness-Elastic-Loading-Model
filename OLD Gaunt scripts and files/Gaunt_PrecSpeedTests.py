# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:28:58 2026

@author: Timov
"""

"""  
GAUNT TABLES LOADING/BUILDING TEST SCRIPT
-----
Test different strategies for building/loading in the Gaunt pickle files:
    - disable vs enable garbace collector of pickle
    - Reduce floating point precision of 

A .txt file is available in the same folder showing outputs for lmax=20 and
lmax=30, so that they dont have to be ran again.



RESULTS ON TESTS LMAX=20 & LMAX=30:
Using the main code gaunt functions:       
    %% At lmax=20:
        PRECISION (np.float16 vs. np.float 32 vs. np.float64):
            - No consistent gain in building time (~503s all, +- 8s)
            - No consistent gain in saving time (float64 saved fastest)
            - No difference in loading time (~identical)
            - Memory difference is (95.4 MB, 101.8 MB, 114.3 MB)
            - np.float16 gives RunTime warning, it cannot store the values due 
              to too large size
        
        GARBAGE COLLECTION DISABLED PICKLING VS NORMAL PICKLING:
            - Consistently much faster loading for gb collection disabled
              (3.4 seconds vs. 7.8-9.4 seconds)
            - No difference in number of loaded entries (no entries missed)
                       
    %% At lmax=30 (performed on pf-merkur DLR computer):
        PRECISION (np.float16 vs np.float 32 vs np.float64):
            - For float16, building=7311s, saving=108s
            - For float32, building=7105s, saving=91s
            - For float64, building=7126s, saving=93s
            - Memory difference is (653.9 MB, 699.0 MB, 788.1 MB)
            
        GARBAGE COLLECTION DISABLED PICKLING VS NORMAL PICKLING:
            - No difference in number of loaded entries (no entries missed)
            - For float16, loading in 31s vs. 61s
            - For float32, loading in 29s vs. 62s
            - For float64, loading in 29s vs. 63s


Using the optimized gaunt table runner functions:
    %% At lmax=20:
        PRECISION (np.float16 vs. np.float 32 vs. np.float64):
            - No consistent gain in building time (~42s all, +- 2s)
            - No consistent gain in saving time (float32 saved slowest)
            - No difference in loading time (~identical)
            - Memory difference is (91.5 MB, 97.9 MB, 110.7MB)
            - np.float16 gives RunTime warning, it cannot store the values due 
              to too large size

        GARBAGE COLLECTION DISABLED PICKLING VS NORMAL PICKLING:
            - Consistently much faster loading for gb collection disabled
              (4.8-5.1 seconds vs. 10.3-11.0 seconds)
            - No difference in number of loaded entries (no entries missed)

            
    %% At lmax=30:
        PRECISION (np.float16 vs np.float 32 vs np.float64):
            - For float16, building=329s, saving=134s
            - For float32, building=374s, saving=302s
            - For float64, building=337s, saving=318s
            - Memory difference is (636.2 MB, 681.3 MB, 771.5 MB)
            
        GARBAGE COLLECTION DISABLED PICKLING VS NORMAL PICKLING:
            - No difference in number of loaded entries (no entries missed)
            - For float16, loading in 141s vs. 523s
            - For float32, loading in 211s vs. 750s
            - For float64, loading in 223s vs. 758s
            
CONCLUSIONS:
    - Precision makes no difference in loading time
    - Precision cannot be done in float16 due to size of Gaunt coefficients
    - File size ('memory') difference between precisions is small for lmax=20, 
      but may become more significant at higher lmax
        - TBD what the effect is on final results of model (power spectra, 
          2D deflection) 
    - Garbage collection disabled is consistently faster for loading of files 
      without consequence on loaded values
    - Memory difference is not the biggest contributor to slow loading/
      building/saving of files

"""


import os
import time
import pickle
import pyshtools as pysh
import numpy as np
import gc





def make_mode_map(lmax):
    # Build a flat sequence list matching the grid ordering approach
    mode_map = []

    for l_idx in range(lmax + 1):
        for m_idx in range(-l_idx, l_idx + 1):
            mode_map.append((l_idx, m_idx))
    return mode_map

# %% BEUTHE/KALOUSOVA MATRIX FUNCTIONS

# Fast numeric W-coefficients evaluation
def W_numeric_A(l_deg, l_prime, L, nu_val=0.25):
    """
    The W-term of Matrix A is the large term in square brackets of Kalousova 
    et al. (2012) equation A18.
    """
    d_l = -l_deg * (l_deg + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L = -L * (L + 1) + 2
    
    term1 = d_l * d_lp
    bracket = (d_l**2 + d_lp**2 + d_L**2 + 
               2*(d_l + d_lp + d_L) - 
               2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return term1 + 0.25 * (1.0 - nu_val) * bracket

def W_numeric_B(l_deg, l_prime, L, nu_val=0.25):
    """
    The W-term of Matrix A is the large term in square brackets of Kalousova 
    et al. (2012) equation A18. Only difference with W_numeric_A is in the 
    final 'return'-term, (1 + nu) vs (1 - nu)
    """
    d_l = -l_deg * (l_deg + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L = -L * (L + 1) + 2
    
    term1 = d_l * d_lp
    bracket = (d_l**2 + d_lp**2 + d_L**2 + 
               2*(d_l + d_lp + d_L) - 
               2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return term1 + 0.25 * (1.0 + nu_val) * bracket

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


# Get the real SH gaunt coefficients using complex decomposition
def _decompose_real_sh(l, m):
    """
    Express Y_{l,m}^{4π real} as a linear combination of Y_{l,cm}^{4π complex}.
    Returns a list of (complex_m_index, coefficient) pairs.
    Convention (pyshtools):
      m = 0  →  Y_{l,0}^real = Y_{l,0}^cmplx
      m > 0  →  Y_{l,+m}^real = (Y_{l,m}^c + (−1)^m Y_{l,−m}^c) / √2  [cosine]
      m < 0  →  Y_{l,m}^real  = i(Y_{l,m}^c − (−1)^|m| Y_{l,−m}^c) / √2  [sine]
    """
    if m == 0:
        return [(0, 1.0+0j)]
    elif m > 0:
        return [(m,  (-1)**m/np.sqrt(2)+0j),
                (-m,  1.0/ np.sqrt(2)+0j)]
    else:
        absm = abs(m)
        return [(-absm,  1j / np.sqrt(2)),
                ( absm, -(-1)**absm * 1j / np.sqrt(2))]

# Conversion to 4pi normalization
_sqrt4pi = np.sqrt(4.0 * np.pi)

# Use numeric (complex) gaunt and decompose real SH to compute the real gaunt
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


# %% GAUNT COEFFICIENT TABLES PRE-CALCULATION OR LOADING
"""
The Gaunt coefficient calculations in the main Beuthe loop take by far the 
longest time to calculate, which explodes when increasing lmax.
Since these values only depend on lmax (specifically l, l', L, m, m' M, which
are defined up to lmax) and the Poisson's ratio nu, these values can be 
precalculated, stored in a subfolder of this repository, and loaded for 
specific lmax and Poisson's ratios.

This section precomputes the (non-zero) coefficients if they do not exist in
the cache yet, or loads them from the cache directory otherwise. 
"""



def build_or_load_gaunt(lmax, nu, plan_path, precision_val = 64, precision_class=np.float64):
   
    mode_map = make_mode_map(lmax)
    
    # Load a Gaunt plan if it exists in the cache
    if os.path.exists(plan_path):
        
        # Test loading time using disabled garbage collector
        print(f"Loading Gaunt plan from cache using disabled gb collector: {plan_path}")
        t_load_disabledgc = time.perf_counter()
        gc.disable()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        gc.enable()
        disabled_gc_load_time = time.perf_counter()-t_load_disabledgc

        
        # Test loading time using normal pickle
        print(f"Loading Gaunt plan from cache using pickle: {plan_path}")
        t_load_normal = time.perf_counter()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        normal_load_time = time.perf_counter()-t_load_normal
        
        print(f"  Using disabled gb collector, loaded {len(assembly_plan):,} entries in "
              f"{disabled_gc_load_time:.2f}s with precision=fl{precision_val}")
        print(f"  Using normal pickle, loaded {len(assembly_plan):,} entries in "
              f"{normal_load_time:.2f}s with precision=fl{precision_val} \n")
        

        
    # If not existing yet, calculate all the coefficients and save them to cache
    else:
        print(f"\nBuilding Gaunt plan (first run for this lmax (={lmax}) — will be cached)...")

        
        t_build = time.perf_counter()
        assembly_plan = []
    
        for i, (l_val, m_val) in enumerate(mode_map):
            for j, (l_prime, m_prime) in enumerate(mode_map[i:], start=i):
                L_entries = []
                min_L = abs(l_val - l_prime)
                max_L = l_val + l_prime
    
                for L in range(min_L, max_L + 1):
                    if (l_val + l_prime + L) % 2 != 0:
                        continue
                    w_coef_A = W_numeric_A(l_val, l_prime, L, nu)
                    w_coef_B = W_numeric_B(l_val, l_prime, L, nu)
                    if w_coef_A == 0.0 and w_coef_B == 0.0:
                        continue
    
                    # Evaluate Gaunt for all M at once, keep only nonzero (nz) values
                    M_vals = np.arange(-L, L + 1)
                    q_vals = np.array([get_real_gaunt(l_val, m_val, L, M, l_prime, m_prime)
                                       for M in M_vals])
                    nz_mask = np.abs(q_vals) > 1e-15
                    if not np.any(nz_mask):
                        continue
    
                    # Store (M_offset_into_slice, w_coef_A*q, w_coef_B*q) for the nonzero M
                    M_offsets = np.where(nz_mask)[0].astype(np.int16)
                    wAq = (w_coef_A * q_vals[nz_mask]).astype(precision_class)
                    wBq = (w_coef_B * q_vals[nz_mask]).astype(precision_class)
                    L_entries.append((L, M_offsets, wAq, wBq))
    
                if L_entries:
                    assembly_plan.append((i, j, L_entries))
        
        build_time = time.perf_counter() - t_build
        print(f"  Built {len(assembly_plan):,} entries in {build_time:.1f}s," 
              f" precision=fl{precision_val} — saving...")
        
        t_save = time.perf_counter()
        with open(plan_path, 'wb') as fh:
            pickle.dump({'lmax': lmax, 'nu': nu, 'plan': assembly_plan}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        save_time = time.perf_counter() - t_save
        print(f"  Saved in {save_time:.1f}s at precision=fl{precision_val},"
              f" to {plan_path} ({os.path.getsize(plan_path)/1e6:.1f} MB)\n")

    return assembly_plan


# %% TEST LOOP

# Make/identify gaunt cache directory to save or load gaunt coefficient tables
CACHE_DIR = "gaunt_cache_PrecTests"
os.makedirs(CACHE_DIR, exist_ok=True)


precision_tests = [(16, np.float16), (32, np.float32), (64, np.float64)]

lmax = 20
nu = 0.25

for i, (precision_val, precision_class) in enumerate(precision_tests):
    # Plan path specifically for these tests
    # NOTE: .pkl REMOVED FROM END OF STRING!!
    plan_path = os.path.join(CACHE_DIR, f"gaunt_plan_SoA_PrecTests_lmax{lmax}_nu{nu:.2f}_precision=fl{precision_val}")
    
    # Save the new precision file
    build_or_load_gaunt(lmax, nu, plan_path, precision_val=precision_val, precision_class=precision_class )
    
    # Directly load it in to check the time required for loading
    build_or_load_gaunt(lmax, nu, plan_path, precision_val=precision_val, precision_class=precision_class )














# %% STRUCTURE FROM GAUNT TABLE RUNNER OPT SCRIPT


import numpy as np
import pyshtools as pysh
import os
import pickle
import time
import gc

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








# import scipy.sparse as sparse
# import scipy.sparse.linalg as spla
# ## INPUTS
# nu = 0.25       # Poisson's ratio
# E = 100.0e9     # Young's Modulus
# rho_c = 2900.   # Density of the crustal manterial
# rho_m = 3500.   # Density of the mantle material
# rho_l = rho_c
# drho = rho_m - rho_c


# # Set all lmax runs
# LMAX_RUNS = [35]

# # Set whether rotation of inputs is applied or not - Verification method
# rotate_angles = (0.0, 0.0, 0.0)

# # Set whether output figures are saved or not
# Save_Figs = False

# # Set color maps
# from palettable import scientific as scm
# from cmcrameri import cm

# cmap1 = scm.diverging.Vik_20.mpl_colormap
# cmap2 = cm.davos

# # Set Te max resolution and tapering cut & width degrees
# lmax_Te_fit = 60    # LSQ fit resolution (top ~5 degrees never trusted)
# l_cut       = 40    # working bandlimit of the Te field
# taper_width = 10    # fade-out width in degrees

# # Map out how the pyshstools array is structured
# def find_custom_element(l_param, m_param, xlm_unstr):
#     # Find the starting index of degree l in the shtools array (which is l^2)
#     block_start = l_param**2
#     if m_param == 0:
#         offset = 0
#     elif m_param > 0:
#         offset = m_param
#     else:
#         offset = l_param + abs(m_param)
#     return xlm_unstr[block_start + offset]

# def solve_beuthe(topo_clm, geoid_clm, D_clm, a_clm, assembly_plan, lmax,
#                  R, T_e_0, g0):
#     """ 
#     TBD: Extensive testing the difference in dense and sparse matrices on total runtime & memory
#     """
    
    
#     mode_map = make_mode_map(lmax)
#     N_modes = len(mode_map)
    
#     # Calculate the buoyancy term used in Matrix A, 
#     # and the two scaling factors of the two matrices
#     Re = R - T_e_0 / 2
#     buoy = (Re / T_e_0)**3 * (Re / E) * g0 * (rho_m - rho_c)
#     scaler_A = 1.0 / (E * T_e_0**3)
#     scaler_B = Re
    
    
#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     # Convert the D and alpha coefficients into a vector for the computations
#     Dlm_unstr = pysh.shio.SHCilmToVector(D_clm.coeffs)
#     alm_unstr = pysh.shio.SHCilmToVector(a_clm.coeffs)

#     # Pre-extract D / alpha coefficient slices per degree L
#     # One array of length (2L+1) per L, indexed M = -L … +L.
#     # Avoids repeated find_custom_element() calls inside the fill loop.
#     D_slices = {}
#     a_slices = {}
#     for L in range(2*lmax + 1):
#         block = L * L
#         idx_list = [0 if M == 0 else (M if M > 0 else L + abs(M))
#                     for M in range(-L, L + 1)]
#         flat_idx = np.array([block + off for off in idx_list], dtype=np.int32)
#         D_slices[L] = Dlm_unstr[flat_idx]
#         a_slices[L] = alm_unstr[flat_idx]
        
#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     sparse_timer_start = time.perf_counter()
    
#     print("Initializing SPARSE matrices...")
#     diag_a = np.zeros(N_modes, dtype=np.float64)
#     diag_b = np.zeros(N_modes, dtype=np.float64)
    
#     for i, (l_val, m_val) in enumerate(mode_map):
#         d_l = -l_val * (l_val + 1) + 2
#         diag_a[i] = ((Re / T_e_0)**3 / E) * d_l
#         diag_b[i] = -1.0 * d_l
    
#     matrix_a_l_sparse = sparse.diags(diag_a, format="lil")
#     matrix_b_l_sparse = sparse.diags(diag_b, format="lil")
    
#     matrix_A_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)
#     matrix_B_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)    

#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     print("Filling matrices A and B...")
#     for i, j, L_entries in assembly_plan:
#         cell_A = 0.0
#         cell_B = 0.0
#         for L, M_offsets, wAq, wBq in L_entries:
#             D_sel = D_slices[L][M_offsets]
#             a_sel = a_slices[L][M_offsets]
#             cell_A += float(np.dot(D_sel, wAq))   # BLAS dot — no Python loop
#             cell_B += float(np.dot(a_sel, wBq))

#         val_A = cell_A * scaler_A + (buoy if i == j else 0.0)
#         val_B = cell_B * scaler_B

#         if val_A != 0.0:
#             matrix_A_sparse[i, j] = val_A
#             if i != j:
#                 matrix_A_sparse[j, i] = val_A      
#         if val_B != 0.0:
#             matrix_B_sparse[i, j] = val_B
#             if i != j:
#                 matrix_B_sparse[j, i] = val_B      


#     print("Combining sub-matrices into a sparse 2N x 2N architecture...")
#     M_system_sparse = sparse.bmat([
#         [matrix_A_sparse,     matrix_a_l_sparse],
#         [matrix_b_l_sparse,   matrix_B_sparse]
#     ], format="lil")
    
#     print("Setting degree 0 and 1 of large matrix to zero...")
#     for idx, (l_val, _) in enumerate(mode_map):
#         if l_val == 0 or l_val == 1:
#             M_system_sparse[idx, :] = 0.0
#             M_system_sparse[idx, idx] = 1.0
#             M_system_sparse[idx + N_modes, :] = 0.0
#             M_system_sparse[idx + N_modes, idx + N_modes] = 1.0
    
#     # Convert to CSR (compressed sparse row) format for faster calculations
#     M_system_csr = M_system_sparse.tocsr()
    
    
    
    
#     # Calculate the RHS components
#     print(f"Solving structural displacement vector for lmax={lmax}")
#     factors_y_lm = (Re / T_e_0)**3 * (rho_c * g0 * Re) / E
       
#     # True topographic loading case, negative to match displacement
#     y_lm_topo = -factors_y_lm * (topo_clm.coeffs - geoid_clm.coeffs)
#     y_lm_unstr = pysh.shio.SHCilmToVector(y_lm_topo)
#     y_lm_str = np.array([find_custom_element(l_v, m_v, y_lm_unstr) for l_v, m_v in mode_map])
    
#     rhs_dense = np.concatenate([y_lm_str, np.zeros(N_modes)])
    
#     print("Setting degree 0 and 1 of rhs vector to zero...")
#     for idx, (l_val, m_val) in enumerate(mode_map):
#         if l_val == 0 or l_val == 1:
#             rhs_dense[idx] = 0.0
#             rhs_dense[idx + N_modes] = 0.0
    
#     # Run linear solver
#     sol_vector = spla.spsolve(M_system_csr, rhs_dense)
    
#     sparse_timer_stop = time.perf_counter() - sparse_timer_start
    
#     print(f'sparse system calculation time at l={lmax} : {sparse_timer_stop:.2f}s \n')
    
    
    
#     w_sol = sol_vector[:N_modes]
#     # F_sol = sol_vector[N_modes:]
    
#     # Map flat 1D solution back into 3D SH shape
#     w_coeffs_np = np.zeros((2, lmax + 1, lmax + 1))
#     for idx, (l_val, m_val) in enumerate(mode_map):
#         if m_val >= 0:
#             w_coeffs_np[0, l_val, m_val] = float(w_sol[idx])
#         else:
#             w_coeffs_np[1, l_val, abs(m_val)] = float(w_sol[idx])
     
#     # Finally, transform the Beuthe solution vector into pysh coefficient and grid format
#     w_sol_clm_beuthe = pysh.SHCoeffs.from_array(w_coeffs_np, normalization='4pi')

#     return w_sol_clm_beuthe





# Make/identify gaunt cache directory to save or load gaunt coefficient tables
CACHE_DIR = "gaunt_cache_PrecTests"
os.makedirs(CACHE_DIR, exist_ok=True)

precision_tests = [(16, np.float16), (32, np.float32), (64, np.float64)]

lmax = 30
nu = 0.25

print('\nNow running for the same precision vals, lmax and nu, using the '
      'slicing method of the wigner3j as used in the optimized runner script.')

for i, (precision_val, precision_class) in enumerate(precision_tests):
    # Plan path specifically for these tests
    plan_path = os.path.join(CACHE_DIR, 
                             f"gaunt_plan_runnerOpt_PrecTests_lmax{lmax}_"
                             f"nu{nu:.2f}_precision=fl{precision_val}.pkl")

    
    if os.path.exists(plan_path):
        # Test loading time using disabled garbage collector
        print(f"Loading Gaunt plan from cache using disabled gb collector: {plan_path}")
        t_load_disabledgc = time.perf_counter()
        gc.disable()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        gc.enable()
        disabled_gc_load_time = time.perf_counter()-t_load_disabledgc

        
        # Test loading time using normal pickle
        print(f"Loading Gaunt plan from cache using pickle: {plan_path}")
        t_load_normal = time.perf_counter()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        normal_load_time = time.perf_counter()-t_load_normal
        
        print(f"  Using disabled gb collector, loaded {len(assembly_plan):,} entries in "
              f"{disabled_gc_load_time:.2f}s with precision=fl{precision_val}")
        print(f"  Using normal pickle, loaded {len(assembly_plan):,} entries in "
              f"{normal_load_time:.2f}s with precision=fl{precision_val} \n")
    
    
    
    else:
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
                    wAq = (wA * q_vals[nz_mask]).astype(precision_class)
                    wBq = (wB * q_vals[nz_mask]).astype(precision_class)
                    L_entries.append((L, M_offsets, wAq, wBq))
        
                if L_entries:
                    assembly_plan.append((i, j, L_entries))
        

        build_time = time.perf_counter() - t_build
        print(f"  Built {len(assembly_plan):,} entries in {build_time:.1f}s," 
              f" precision=fl{precision_val} — saving...")
        
        t_save = time.perf_counter()
        with open(plan_path, 'wb') as fh:
            pickle.dump({'lmax': lmax, 'nu': nu, 'plan': assembly_plan}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        save_time = time.perf_counter() - t_save
        print(f"  Saved in {save_time:.1f}s at precision=fl{precision_val},"
              f" to {plan_path} ({os.path.getsize(plan_path)/1e6:.1f} MB)\n")



        print(f"Loading Gaunt plan from cache using disabled gb collector: {plan_path}")
        t_load_disabledgc = time.perf_counter()
        gc.disable()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        gc.enable()
        disabled_gc_load_time = time.perf_counter()-t_load_disabledgc
    
        
        # Test loading time using normal pickle
        print(f"Loading Gaunt plan from cache using pickle: {plan_path}")
        t_load_normal = time.perf_counter()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        normal_load_time = time.perf_counter()-t_load_normal
        
        print(f"  Using disabled gb collector, loaded {len(assembly_plan):,} entries in "
              f"{disabled_gc_load_time:.2f}s with precision=fl{precision_val}")
        print(f"  Using normal pickle, loaded {len(assembly_plan):,} entries in "
              f"{normal_load_time:.2f}s with precision=fl{precision_val} \n")


