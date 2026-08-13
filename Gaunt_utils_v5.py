# -*- coding: utf-8 -*-
"""
Beuthe (2008) variable-thickness flexure solver
Gaunt functions assisting file

This file contains the Gaunt builder, saver, loader and reader functions that
are used in the flexure models of this thesis work. These functions are 
separated from the model themselves for conciseness and clarity.

"""

import numpy as np
import pyshtools as pysh
import os
import time
from concurrent.futures import ProcessPoolExecutor

CACHE_DIR  = "gaunt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def make_mode_map(lmax):
    """
    Flatten all combinations of l,m into a flat array based on input lmax.
    """
    return [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]


# %% GAUNT FUNCTIONS
# Conversion factor to go to 4pi normalization
_sqrt4pi = np.sqrt(4.0*np.pi)

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
    bracket = (d_l**2 + d_lp**2 + d_L**2 
               + 2*(d_l + d_lp + d_L) 
               - 2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return term1 + 0.25 * (1.0 - nu_val) * bracket

def W_numeric_B(l_deg, l_prime, L, nu_val=0.25):
    """
    The W-term of Matrix B is the large term in square brackets of Kalousova 
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

def W_numeric_Aonly(l, lp, L):
    """
    Spectral weight of Beuthe's PURE A operator (Beuthe 2008 eq 33;
    Kalousova 2012 eq A9): the coupling A(field; operand) alone, without the
    Delta'(field*Delta' operand) part. 
    
    Follows directly from Kalousova A18:
    the -(1-nu)*A(D;w) piece contributes -(1-nu)/4 * br, hence
        W_Aonly(l, lp, L) = -br/4
    with the SAME symmetric bracket br used in W_numeric_A/B.
    
    Convention (as W_numeric_A): 
        l = output degree, 
        lp = operand degree,
        L = field degree. 
    Symmetric in (l, lp). 
    
    Monopole-field reduction:
        L=0 (dL=2), 
        l=lp  ->  br = -4*dl  ->  W_Aonly = dl = Delta',
    i.e. A(const; f) = const * Delta' f, as required.
    
    Used (in decomposed per-term/per-cell form) to build the ETA_FULL
    coupling blocks R^3*A(eta;F) and -(1/R)*A(eta;w) of Beuthe eqs (58)/(66).
    """
    dl  = -l  * (l  + 1) + 2
    dlp = -lp * (lp + 1) + 2
    dL  = -L  * (L  + 1) + 2
    
    bracket = (dl**2 + dlp**2 + dL**2 
               + 2*(dl+dlp+dL) 
               - 2*(dl*dlp+dl*dL+dlp*dL) - 8)
    return -0.25*bracket

# Define two functions for the transformation between real and complex 
# coefficients. Required for calculation of Wigner3j: real degrees & orders go 
# into model, transform to complex for Wigner3j (& thus Gaunt) computation, 
# transform back to real domain for interpretation of results
def _decompose_real_sh(l, m):
    """
    Converts a real harmonic function Y_lm to its sum of two complex harmonic
    functions. For m>0 (cosine-like), it's a combination of complex order +m 
    and -m with no imaginary term. For m<0 (sine-like) the same, but flipped
    and with the imaginary unit. For m=0 the real and imaginary term are equal.
    
    More elaborately:
    Express Y_{l,m}^{4π real} as a linear combination of Y_{l,cm}^{4π complex}.
    Returns a list of (complex_m_index, coefficient) pairs.
    Convention (pyshtools):
      m = 0  →  Y_{l,0}^real = Y_{l,0}^cmplx
      m > 0  →  Y_{l,+m}^real = (Y_{l,m}^c + (−1)^m Y_{l,−m}^c) / √2  [cosine]
      m < 0  →  Y_{l,m}^real  = i(Y_{l,m}^c − (−1)^|m| Y_{l,−m}^c) / √2  [sine]
      
    This function together with _cM_to_indices is required for the 
    computation of the gaunt coefficients only, since the Wigner3j functions 
    are defined in the complex domain while all other functions are defined in 
    the real domain. Complex form of the sh functions therefore only occurs 
    shortly during the calculation of the gaunt coefficients.
    """
    if m == 0:  
        return [(0, 1.0+0j)]
    elif m > 0:   
        return [(m, (-1)**m/np.sqrt(2)+0j), 
                (-m, 1.0/np.sqrt(2)+0j)]
    else:
        am = abs(m)
        return [(-am, 1j/np.sqrt(2)), 
                (am, -(-1)**am*1j/np.sqrt(2))]

def _cM_to_indices(cM, L):
    """
    Converts the complex harmonic functions back to a real harmonic function.
    Deposits a complex coupling from order cM at location k=M+L in the 
    real-basis output array of length 2L+1. 
    
    For the cM > 0 case (second term of Y_l^+m in the real-complex SH function),
    two real-basis slots carry complex order cM: 
        - The M = +cM slot (slot L + cM, the cosine term), with the cosine 
          coefficient term [(-1)**|M|*Y_l^|M| / sqrt(2)]
        - The M = -cM slot (slot L - cM, the sine term), with the sine 
          coefficient term [i*(-1)**|M|*Y_l^|M| / sqrt(2)]
          
    For the cM < 0 case (first term of Y_l^+m in the real-complex SH function), 
    the terms have the same indexing as above, but different coefficients:
        - The M = +|cM| slot (slot L + cM, the cosine term), with the cosine 
          coefficient term [1 / sqrt(2)]
        - The M = -|cM| slot (slot L - cM, the sine term), with the sine 
          coefficient term [i / sqrt(2)]     
    
    This function together with _decompose_real_sh is required for the 
    computation of the gaunt coefficients only, since the Wigner3j functions 
    are defined in the complex domain while all other functions are defined in 
    the real domain. Complex form of the sh functions therefore only occurs 
    shortly during the calculation of the gaunt coefficients.
    """
    if cM == 0: 
        return [(L, 1.0+0j)]
    am = abs(cM)
    if am > L:  
        return []
    if cM > 0:  
        return [(L+am, (-1)**am/np.sqrt(2)+0j), 
                (L-am, -(-1)**am*1j/np.sqrt(2))]
    return [(L+am, 1.0/np.sqrt(2)+0j), 
            (L-am, 1j/np.sqrt(2))]

def get_real_gaunt_slice(l_out, m_out, L, l_prime, m_prime):
    """
    Optimized computation of the Gaunt coefficients. Instead of looping over
    all complex orders cM, now feed only those cM values that agree with the 
    selection rule for orders (m1+m2+m3=0). Drastically reduced number of 
    Wigner3j calls.
    All M in [-L,L] at once; <=4 Wigner3j calls. Index k <-> M = k-L.
    """
    res = np.zeros(2*L+1, dtype=np.complex128)
    
    # Rule of even sum of l-values
    if (l_out+L+l_prime) % 2 != 0:              
        return res.real*_sqrt4pi
    
    # Triangle rule, L is between [0, 2*lmax]
    if not (abs(l_out-l_prime) <= L <= l_out+l_prime):  
        return res.real*_sqrt4pi
    
    # Compute zero-order Wigner3j (no complex orders m needed)
    w3j_0_array, jmin_0, jmax_0 = pysh.utils.Wigner3j(L, l_prime, 0, 0, 0)
    # l_out must be between 0 (min) and 2*l_out (max)
    if not (jmin_0 <= l_out <= jmax_0):          
        return res.real*_sqrt4pi
    
    w3j_0 = w3j_0_array[l_out-jmin_0]
    if w3j_0 == 0.0:                       
        return res.real*_sqrt4pi
    
    # Precompute the prefactor of the Gaunt coefficient * zero-order Wigner3j
    gaunt_factor = (np.sqrt((2*l_out+1)*(2*L+1)*(2*l_prime+1)/(4.0*np.pi)) 
                    * w3j_0)
    
    for cm_out, c_out in _decompose_real_sh(l_out, m_out):
        for cm_prime, c_prime in _decompose_real_sh(l_prime, m_prime):
            
            # "cM + cM_out + cM_prime = 0" Rule
            cM = -cm_out - cm_prime
            if abs(cM) > L: # M must be smaller or equal to L
                continue
            
            # Compute full Wigner3j using complex orders of m's
            w3j_m_array, jmin_m, jmax_m = (
                pysh.utils.Wigner3j(L, l_prime, cm_out, cM, cm_prime))
            # l_out must be between 0 (min) and 2*l_out (max)
            if not (jmin_m <= l_out <= jmax_m): 
                continue
            
            # Only interested in the first value of the array
            w3j_m = w3j_m_array[l_out-jmin_m]
            if w3j_m == 0.0: 
                continue
            
            # Compute the final (complex) gaunt value for this specific 
            # combination of l_out, l_prime, L, m_out, m_prime, M
            gaunt = gaunt_factor * w3j_m
            
            # Transform back to the real domain by summing over all L for
            # specific combination of l_out, m_out and l_prime, m_prime
            for idx, cL in _cM_to_indices(cM, L):
                res[idx] += c_out*cL*c_prime*gaunt
    # Return the final Gaunt value in real domain and 4pi normalized
    return res.real*_sqrt4pi

def get_numeric_gaunt(l_out, L, l_prime, m_out, M, m_prime):
    """
    Perform the Gaunt Coefficient calculations here including the selection
    rules as also given in Kalousova et al. (2012).
    Gaunt Coefficients are calculated using the Wigner3j symbols, which are 
    complex values.
    """
    if (l_out+L+l_prime)%2: 
        return 0.0
    if not (abs(l_out-L)<=l_prime<=l_out+L): 
        return 0.0
    if m_out + M + m_prime != 0: 
        return 0.0
    
    # Evaluate the vector for m components
    w3j_m_array, jmin_m, jmax_m = pysh.utils.Wigner3j(L, l_prime, 
                                                      m_out, M, m_prime)
    if not (jmin_m <= l_out <= jmax_m):
        return 0.0
    w3j_m = w3j_m_array[l_out - jmin_m]

    # Evaluate the vector for the m=0 components
    w3j_0_array, jmin_0, jmax_0 = pysh.utils.Wigner3j(L, l_prime, 0, 0, 0)
    if not (jmin_0 <= l_out <= jmax_0):
        return 0.0
    w3j_0 = w3j_0_array[l_out - jmin_0]
    
    factor = np.sqrt((2*l_out+1) * (2*L+1) * (2*l_prime+1) / (4.0*np.pi))  
    return factor * w3j_m * w3j_0

# Use numeric (complex) gaunt and decompose real SH to compute the real gaunt
def get_real_gaunt(l_out, m_out, L, M, l_prime, m_prime):
    """
    Coupling coefficient for REAL SH (signed-m) in the 4pi-normalised product
    formula:
        (A·B)_{l_out,m_out} = SUM_{L,M,l',m'} A_{L,M} · B_{l',m'} · H

    H = sqrt(4pi) · SUM_{complex_decompositions} c_out · c_L · c_p
                   · G_code(l_out, L, l'; cm_out, cM, cm_p)

    This is the general real-SH Gaunt coefficient, valid for ALL signed-m
    values.  For m=0 it reduces to sqrt(4pi)*G_code.  
    Verified numerically against direct grid
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
                gaunt_val = get_numeric_gaunt(l_out, L, l_prime,
                                      cm_out, cM, cm_p)
                total += c_out * cL * c_p * gaunt_val

    return total.real * _sqrt4pi  # imaginary part is exactly 0 for real SH

# Perform a selftest of the gaunt coefficients
def selftest_gaunt(n=300, lmax_test=6, seed=0):
    """
    Perform a selftest of the get_real_gaunt_slice method by commparing it with 
    the slow but intuitive get_real_gaunt function to see if outputs are equal.
    This selftest is the reason that get_real_gaunt and get_numeric_gaunt are
    kept in the code.
    """
    
    rng = np.random.default_rng(seed)
    for _ in range(n):
        lo,lp = rng.integers(0,lmax_test+1,2)
        mo = rng.integers(-lo,lo+1) if lo else 0
        mp = rng.integers(-lp,lp+1) if lp else 0
        L  = rng.integers(abs(lo-lp),lo+lp+1) if lo+lp else 0
        sl = get_real_gaunt_slice(lo,mo,L,lp,mp)
        for k,M in enumerate(range(-L,L+1)):
            assert abs(sl[k]-get_real_gaunt(lo,mo,L,M,lp,mp))<1e-15, \
                f"GAUNT MISMATCH {(lo,mo,L,M,lp,mp)}"
    print("Gaunt self-test passed.")

# %% SoA FUNCTIONS
"""
SoA plan format: the Gaunt coupling plan is stored as 8 flat arrays instead of 
millions of nested (i, j, [(L, offsets, wA, wB), ...]) objects.

  cell_i[c], cell_j[c]   : matrix (row, col) of nonzero cell number c
  cell_start[c]          : first term index belonging to cell c
                          (cell c owns terms [cell_start[c] : cell_start[c+1]])
  cell_nterms[c]         : number of terms owned by cell c
  term_L[t]              : SH degree L of term t
  term_off[t]            : M-offset into the (2L+1) D/alpha slice for term t
  term_gaunt_bare[t]     : the bare real Gaunt coefficient q for term t

PLAN v5 -- term_wA / term_wB ARE NO LONGER STORED.
They were W_numeric_A/B(l, l', L) * q, i.e. the bare Gaunt coefficient scaled
by a weight that depends on nothing but the three DEGREES. Those degrees are
already in the plan: l from cell_i, l' from cell_j (via mode_map), L from
term_L. Storing the products cost two float64 arrays of length n_terms --
36.6 GB at lmax=90, half the file -- to save ~15 flops per term that run ONCE
per solve (the matrix is LU-cached). They are now recomputed on the fly by
term_weights_AB(). Net effect at lmax=90: file 74 GB -> 37 GB, load time
roughly halved, recompute cost a few tens of seconds once.

Reconstruction of one matrix cell c:
    A_ij = scaler_A * sum_{t in cell c} D_{term_L,term_off} * term_wA[t] 
            (+buoy if i==j)
This whole sum, for ALL cells at once, is done with one fancy-index gather 
plus np.add.reduceat -- no Python loop over terms (that loop was the lmax=50 
                                                   RAM killer).
"""

PLAN_VERSION = 5      # v5: term_wA/term_wB dropped, recomputed by term_weights_AB


def mode_degrees(lmax):
    """
    Spherical-harmonic degree l of every entry of make_mode_map(lmax), as an
    int32 array of length (lmax+1)**2. Needed to recover l and l' per term from
    cell_i / cell_j once the stored weights are gone. Tiny: 8281 entries at
    lmax = 90.
    """
    return np.asarray([lv for lv, _ in make_mode_map(lmax)], dtype=np.int32)


def term_weights_AB(l_of, cell_i, cell_j, cell_nterms, term_L, nu):
    """
    Recompute W_numeric_A and W_numeric_B for every term of a BLOCK of cells.

    Replaces the stored term_wA / term_wB. Both are the same expression apart
    from (1 - nu) vs (1 + nu), so term1 and bracket are formed once and reused:

        wA = term1 + 0.25*(1 - nu)*bracket
        wB = term1 + 0.25*(1 + nu)*bracket

    Arguments are already sliced to the block. `term_L` must be exactly the
    terms owned by cells [cell_i, cell_j], i.e. length == cell_nterms.sum().

    Returns (wA, wB), float64, length == term_L.size. Note these are the BARE
    weights -- the caller still multiplies by term_gaunt_bare, exactly as the
    old stored arrays (which were weight * q) implied.
    """
    d_l  = -(l_of[cell_i].astype(np.float64) * (l_of[cell_i] + 1.0)) + 2.0
    d_lp = -(l_of[cell_j].astype(np.float64) * (l_of[cell_j] + 1.0)) + 2.0
    d_l  = np.repeat(d_l,  cell_nterms)
    d_lp = np.repeat(d_lp, cell_nterms)

    Lf  = term_L.astype(np.float64)
    d_L = -(Lf * (Lf + 1.0)) + 2.0
    del Lf

    term1   = d_l * d_lp
    bracket = (d_l*d_l + d_lp*d_lp + d_L*d_L
               + 2.0*(d_l + d_lp + d_L)
               - 2.0*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8.0)
    del d_l, d_lp, d_L

    wA = term1 + 0.25*(1.0 - nu)*bracket
    wB = term1 + 0.25*(1.0 + nu)*bracket
    return wA, wB


def build_plan_serial(lmax, nu):
    """
    Single-process plan build. Same output as build_plan_soa_parallel but with
    no ProcessPoolExecutor, so it works from an interactive session and on
    Windows, where spawned workers re-import __main__ and cannot see functions
    defined by running this file as a script. Only for SMALL lmax -- the
    parallel builder exists because this is O(N^2) in Python.
    """
    frag = _build_chunk_flat((0, len(make_mode_map(lmax)), lmax, nu))
    cell_start = np.empty(frag['cell_nterms'].size + 1, np.int64)
    cell_start[0] = 0
    np.cumsum(frag['cell_nterms'], out=cell_start[1:])
    return {'cell_i': frag['cell_i'], 'cell_j': frag['cell_j'],
            'cell_nterms': frag['cell_nterms'], 'cell_start': cell_start,
            'term_L': frag['term_L'], 'term_off': frag['term_off'],
            'term_gaunt_bare': frag['term_gaunt_bare']}


def selftest_term_weights(lmax=8, nu=0.25):
    """
    Verify that term_weights_AB reproduces W_numeric_A/B term by term, i.e.
    that dropping term_wA/term_wB from the plan changes nothing.

    Builds its plan with build_plan_serial: no process pool, so this runs from
    an interactive session. The reference side is a pure-Python loop over every
    term, so keep lmax small -- lmax = 8 is ~1 s, lmax = 20 is minutes.
    """
    plan = build_plan_serial(lmax, nu)
    l_of = mode_degrees(lmax)
    wA, wB = term_weights_AB(l_of, plan['cell_i'], plan['cell_j'],
                             plan['cell_nterms'], plan['term_L'], nu)
    ci = np.repeat(plan['cell_i'], plan['cell_nterms'])
    cj = np.repeat(plan['cell_j'], plan['cell_nterms'])
    eA = np.array([W_numeric_A(int(l_of[i]), int(l_of[j]), int(L), nu)
                   for i, j, L in zip(ci, cj, plan['term_L'])])
    eB = np.array([W_numeric_B(int(l_of[i]), int(l_of[j]), int(L), nu)
                   for i, j, L in zip(ci, cj, plan['term_L'])])
    assert np.abs(wA - eA).max() == 0.0 and np.abs(wB - eB).max() == 0.0, \
        'term_weights_AB does not reproduce W_numeric_A/B exactly'
    print(f'term_weights_AB self-test passed at lmax={lmax}: '
          f'{wA.size:,} terms, max|diff| = 0 exactly.')


def save_plan_soa(plan, lmax, nu, path):
    """ 
    Save SoA plan to directory.
    """
    np.savez(path, lmax=np.int64(lmax), nu=np.float64(nu),
             plan_version=np.int64(PLAN_VERSION), **plan)

def load_plan_soa(path):
    """
    Load in SoA plan, using disabled gar
    """
    with np.load(path) as z:
        version = int(z['plan_version']) if 'plan_version' in z.files else 1
        if version != PLAN_VERSION:
            raise ValueError(
                f'plan at {path} is version {version}, this code expects '
                f'v{PLAN_VERSION}. v1 stored term_wA/term_wB; v5 recomputes '
                f'them. Delete the file and let it rebuild.')
        return {k: z[k] for k in z.files
                if k not in ('lmax', 'nu', 'plan_version')}



def _build_chunk_flat(args):
    """
    Build outer indices [i_lo, i_hi) and return FLAT arrays for that chunk.
    Returns a dict of 7 arrays (no sentinel here; the parent stitches starts).
    """
    i_lo, i_hi, lmax, nu = args
    mode_map = make_mode_map(lmax)
    Nmode = len(mode_map)
 
    cell_i, cell_j, cell_nterms = [], [], []   # per-cell: row, col, #terms
    term_L, term_off, term_gaunt_bare = [], [], []
 
    for i in range(i_lo, i_hi):
        lv, mv = mode_map[i]
        for j in range(i, Nmode):
            lp, mp = mode_map[j]
            nterms_cell = 0
            for L in range(abs(lv - lp), lv + lp + 1):
                if (lv + lp + L) % 2:
                    continue
                # The weights are NOT stored (see the v5 note above), but they
                # are still evaluated here because `wA == wB == 0` is a term
                # SELECTION criterion. Dropping the test would change which
                # terms exist and break bit-compatibility with older plans.
                wA = W_numeric_A(lv, lp, L, nu)
                wB = W_numeric_B(lv, lp, L, nu)
                if wA == 0.0 and wB == 0.0:
                    continue
                q = get_real_gaunt_slice(lv, mv, L, lp, mp)
                nz = np.abs(q) > 1e-15
                if not np.any(nz):
                    continue
                idx = np.where(nz)[0]
                n = idx.size
                term_L.append(np.full(n, L, np.int32))
                term_off.append(idx.astype(np.int32))
                term_gaunt_bare.append(q[nz].astype(np.float64))
                nterms_cell += n
            if nterms_cell > 0:
                cell_i.append(i) 
                cell_j.append(j)
                cell_nterms.append(nterms_cell)
 
    cat = lambda lst, dt: (np.concatenate(lst) if lst else np.empty(0, dt))
    return {
        'cell_i':      np.asarray(cell_i, np.int32),
        'cell_j':      np.asarray(cell_j, np.int32),
        'cell_nterms': np.asarray(cell_nterms, np.int64),   # per-cell term count
        'term_L':   cat(term_L,  np.int32),
        'term_off': cat(term_off, np.int32),
        'term_gaunt_bare':  cat(term_gaunt_bare, np.float64),
    }
 
 
def build_plan_soa_parallel(lmax, nu, nproc=16, chunks_per_core=4):
    """Parallel, low-memory SoA build. Workers return flat arrays; the parent
    concatenates flat arrays only (never a nested plan)."""
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 1))
    N = len(make_mode_map(lmax))
    nproc = min(nproc, N)
    # More, smaller chunks than cores -> better load balance (low-i chunks are
    # heavier because inner j runs i..N). Order is still preserved by ex.map.
    nchunks = min(N, nproc * max(1, chunks_per_core))
    bounds = np.linspace(0, N, nchunks + 1).astype(int)
    tasks = [(int(bounds[k]), int(bounds[k + 1]), lmax, nu)
             for k in range(nchunks) if bounds[k + 1] > bounds[k]]
    print(f"  parallel build: {len(tasks)} chunks on {nproc} cores (N={N})")
 
    # Collect flat fragments in submission order (ex.map preserves it).
    frags = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for frag in ex.map(_build_chunk_flat, tasks):
            frags.append(frag)
 
    # Concatenate flat arrays. cell ordering is i-ascending across chunks, so
    # simple concatenation keeps the plan in canonical order (no sort).
    cell_i          = np.concatenate([f['cell_i']      for f in frags])
    cell_j          = np.concatenate([f['cell_j']      for f in frags])
    cell_nterms     = np.concatenate([f['cell_nterms'] for f in frags])
    term_L          = np.concatenate([f['term_L']   for f in frags])
    term_off        = np.concatenate([f['term_off'] for f in frags])
    term_gaunt_bare = np.concatenate([f['term_gaunt_bare']  for f in frags])
    del frags  # release fragment memory immediately
 
    # Build cell_start from per-cell term counts: start[0]=0, cumulative, plus
    # a trailing sentinel = total term count (so reduceat segments close).
    cell_start = np.empty(cell_nterms.size + 1, np.int64)
    cell_start[0] = 0
    np.cumsum(cell_nterms, out=cell_start[1:])
 
    return {
        'cell_i': cell_i, 
        'cell_j': cell_j, 
        'cell_start': cell_start,
        'term_L': term_L, 
        'term_off': term_off,
        'cell_nterms': cell_nterms,
        'term_gaunt_bare': term_gaunt_bare
    }
 
 
# ---------------------------------------------------------------------------
#  CHUNKED GAUNT REDUCTIONS
#
#  Every quantity the assembly needs from the plan has the form
#      cell_c = sum_{t in cell c}  field[gidx_t] * weight_t
#  The direct expression `field[gidx] * weight` materialises TWO arrays of
#  length n_terms. At lmax = 90 that is 2 x 18.3 GB per call, and there are
#  about ten such calls per assembly -- which is where 270 GB of peak RSS came
#  from, on top of a 74 GB plan.
#
#  Chunking on CELL boundaries keeps every reduceat segment inside one block,
#  so the summation order is unchanged and the result is BITWISE identical.
# ---------------------------------------------------------------------------

CELLS_PER_CHUNK = 2_000_000        # ~1 GB of working buffer at lmax = 90


def _chunk_bounds(starts, nterm, ncell, cells_per_chunk):
    """Yield (c0, c1, t0, t1) blocks of cells and the terms they own."""
    for c0 in range(0, ncell, cells_per_chunk):
        c1 = min(c0 + cells_per_chunk, ncell)
        t0 = int(starts[c0])
        t1 = int(starts[c1]) if c1 < ncell else int(nterm)
        yield c0, c1, t0, t1


def _reduceat_block(blk, starts, c0, c1, t0):
    """np.add.reduceat over one block, with the empty-trailing-cell guard."""
    loc = (starts[c0:c1] - t0).astype(np.int64)
    np.clip(loc, 0, blk.size - 1, out=loc)
    return np.add.reduceat(blk, loc)


def cell_sums(field_unstr, gidx, weight, starts, seg_len,
              cells_per_chunk=CELLS_PER_CHUNK):
    """
    Per-cell sums of  field_unstr[gidx_t] * weight_t,  chunked over cells.
    `weight` is a full-length array (e.g. plan['term_gaunt_bare']).
    """
    ncell = starts.size
    out = np.zeros(ncell)
    for c0, c1, t0, t1 in _chunk_bounds(starts, weight.size, ncell,
                                        cells_per_chunk):
        if t1 <= t0:
            continue
        blk = field_unstr[gidx[t0:t1]]
        blk *= weight[t0:t1]                       # in place: no second array
        out[c0:c1] = _reduceat_block(blk, starts, c0, c1, t0)
    out[seg_len == 0] = 0.0
    return out


def cell_sums_AB(Dlm, alm, gidx, plan, l_of, starts, seg_len, nu,
                 cells_per_chunk=CELLS_PER_CHUNK):
    """
    The A and B cell sums, with W_numeric_A/B recomputed per block instead of
    read from the plan (see the PLAN v5 note in Gaunt_utils):

        cellA_c = sum_t D[gidx_t] * wA_t * q_t
        cellB_c = sum_t a[gidx_t] * wB_t * q_t

    Both in one pass, since term1 and bracket are shared between wA and wB.
    """
    ncell = starts.size
    q = plan['term_gaunt_bare']
    cellA, cellB = np.zeros(ncell), np.zeros(ncell)
    for c0, c1, t0, t1 in _chunk_bounds(starts, q.size, ncell, cells_per_chunk):
        if t1 <= t0:
            continue
        wA, wB = term_weights_AB(l_of, plan['cell_i'][c0:c1],
                                 plan['cell_j'][c0:c1],
                                 plan['cell_nterms'][c0:c1],
                                 plan['term_L'][t0:t1], nu)
        qb = q[t0:t1]
        wA *= qb;  wA *= Dlm[gidx[t0:t1]]
        cellA[c0:c1] = _reduceat_block(wA, starts, c0, c1, t0)
        del wA
        wB *= qb;  wB *= alm[gidx[t0:t1]]
        cellB[c0:c1] = _reduceat_block(wB, starts, c0, c1, t0)
        del wB
    cellA[seg_len == 0] = 0.0
    cellB[seg_len == 0] = 0.0
    return cellA, cellB


def build_gidx(plan, chunk=200_000_000):
    """
    Flat index into the Dlm / alm coefficient vectors for every term.

    int32 THROUGHOUT. gidx maxes at lmax**2 + 2*lmax = 8280 at lmax = 90, four
    orders of magnitude inside int32. The previous version cast term_L and
    term_off to int64 and then built gidx through ~9 more int64 temporaries of
    the same length -- roughly 169 GB of allocation at lmax = 90, several alive
    simultaneously.
    """
    term_L, k_offset = plan['term_L'], plan['term_off']
    gidx = np.empty(term_L.size, np.int32)
    for t0 in range(0, term_L.size, chunk):
        t1 = min(t0 + chunk, term_L.size)
        L  = term_L[t0:t1]
        Mv = k_offset[t0:t1] - L                        # signed order M = k - L
        off = np.where(Mv == 0, 0,
                       np.where(Mv > 0, Mv, L + np.abs(Mv)))   # shtools offset
        gidx[t0:t1] = L*L + off
    return gidx


def cell_sums_eta_dL(field_unstr, gidx, plan, starts, seg_len,
                     cells_per_chunk=CELLS_PER_CHUNK):
    """
    The three per-cell sums weighted by powers of dL = -L(L+1) + 2:

        g0_c = sum_t f[gidx_t] * q_t
        g1_c = sum_t f[gidx_t] * q_t * dL_t
        g2_c = sum_t f[gidx_t] * q_t * dL_t**2

    Used for the eta coupling A(eta; .), whose weight splits into a per-TERM
    part (powers of dL) and a per-CELL part (dl, dl'). dL is derived from
    plan['term_L'] per block rather than stored.

    The direct form built f[gidx]*q, then that times dL and times dL**2 -- five
    arrays of length n_terms, 5 x 18.3 GB at lmax = 90.
    """
    q = plan['term_gaunt_bare']
    ncell = starts.size
    g0, g1, g2 = (np.zeros(ncell) for _ in range(3))
    for c0, c1, t0, t1 in _chunk_bounds(starts, q.size, ncell, cells_per_chunk):
        if t1 <= t0:
            continue
        Lb  = plan['term_L'][t0:t1].astype(np.float64)
        dLb = -(Lb*(Lb + 1.0)) + 2.0
        eb  = field_unstr[gidx[t0:t1]] * q[t0:t1]
        g0[c0:c1] = _reduceat_block(eb,         starts, c0, c1, t0)
        g1[c0:c1] = _reduceat_block(eb*dLb,     starts, c0, c1, t0)
        g2[c0:c1] = _reduceat_block(eb*dLb*dLb, starts, c0, c1, t0)
        del Lb, dLb, eb
    for arr in (g0, g1, g2):
        arr[seg_len == 0] = 0.0
    return g0, g1, g2


def build_conv_matrix(field_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N):
    """ C_X[i,j] = sum_{t in cell(i,j)} X[gidx_t] * gaunt_bare_t   (single field
        spectral convolution operator, symmetric)."""
    # Chunked: the direct `field_unstr[gidx] * gaunt_bare` allocated two
    # arrays of length n_terms (2 x 18.3 GB at lmax = 90) per call.
    cell = cell_sums(field_unstr, gidx, gaunt_bare, starts, seg_len)
    # Vectorised scatter: the old Python loop ran ci.size times -- 34 million
    # iterations per call at lmax = 90, and this is called ~6 times per
    # assembly. C[i,j] and C[j,i] receive the same value, so the i == j case
    # needs no special handling.
    C = np.zeros((N, N))
    C[ci, cj] = cell
    C[cj, ci] = cell
    return C
 


def build_or_load_gaunt(lmax, nu, nproc=16):
    """ 
    If a plan at the input lmax and Poisson's ratio exists in the cache 
    directory that is set at the top of the code, then this function
    will load it in. If it is not found, it will build it using multiple 
    processors. Number of processors can be set manually to either speed up
    building, or prevent overloading of computer.
    """
    
    path = os.path.join(CACHE_DIR, f"gaunt_plan_v{PLAN_VERSION}_lmax{lmax}_nu{nu:.4f}.npz")
    if os.path.exists(path):
        print(f"Start loading in SoA plan, lmax={lmax}")
        t = time.perf_counter()
        plan = load_plan_soa(path)
        print(f"Loaded SoA plan lmax={lmax}: {plan['cell_i'].size:,} cells, "
              f"{plan['term_L'].size:,} terms in {time.perf_counter()-t:.2f}s")
    else:
        print(f"Building SoA plan lmax={lmax} (parallel v2, first time)…", 
              flush=True)
        t = time.perf_counter()
        plan = build_plan_soa_parallel(lmax, nu, nproc=nproc)
        save_plan_soa(plan, lmax, nu, path)
        print(f"  Built {plan['cell_i'].size:,} cells / "
              f"{plan['term_L'].size:,} terms "
              f"in {time.perf_counter()-t:.1f}s -> "
              f"{os.path.getsize(path)/1e6:.1f} MB")
    return plan