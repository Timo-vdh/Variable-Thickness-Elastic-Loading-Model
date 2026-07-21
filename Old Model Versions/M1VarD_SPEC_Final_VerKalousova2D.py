# -*- coding: utf-8 -*-
"""
Beuthe (2008) variable-thickness flexure solver — Model 1 (M1) - Verification
with Kalousova 1D displacements

Model for the variable thickness deformations of a thin elastic spherical shell.
Current model works with Beuthe's equations 75 and 76. 
Model 1 does not include:
    - Tangential loading (Omega=V=0)
    - Geoid self-consistency solving
    - Crustal thickness variations
    - Mantle density variations

Following Beuthe's model requires implementation of the differential operator 
A(a;b). Beuthe (2008) does not give a spectral method for this, but in Beuthe
(2010) this spectral notation is made. Kalousova et al. (2012) describe the 
system of equations 75 and 76 in full spectral notation. This exact system of
equations is solved in this Model 1.
"""

import numpy as np
import pyshtools as pysh
import os, time, gc
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# %% INPUTS

# Kalousova uses E=65 GPa & rho_m=3400 (their Table 1)
nu, E      = 0.25, 65.0e9    
rho_c, rho_m = 2900., 3400.

LMAX_RUNS  = [45]        # last entry is the reference resolution
rotate_angles = (0.0, 0.0, 0.0)
lmax_Te_fit = 30
CACHE_DIR  = "gaunt_cache"
cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cm.davos
os.makedirs(CACHE_DIR, exist_ok=True)

model = 'MIII'

SaveFigs = False
SavePath = "Plots/M1VarD_SPEC_FinalPlots"

# %% BASIC FUNCTION DEFINITIONS

def make_mode_map(lmax):
    """
    Flatten all combinations of l,m into a flat array based on input lmax.
    """
    return [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]

def truncate(clm, lmax):
    """
    Truncate any SHCoeffs object coefficients up to degree lmax.
    """
    return pysh.SHCoeffs.from_array(clm.coeffs[:, :lmax+1, :lmax+1].copy(),
                                    normalization='4pi')


# %% INPUT LOADERS
     
def make_synthetic_Te_model(lmax, model):
    """
    Recreate the synthetic Te distributions as done in Kalousova et al. (2012).
    """
    # Make a colatitude range for the harmonic T_e functions to be created over
    theta_range = np.linspace(0, 180, 2*(lmax+1)+1)
    
    # Make T_e distribution - Model I of Kalousova
    T_e_I = []
    start_trans = 80
    stop_trans = 100
    phi_trans = np.pi * (theta_range - start_trans)/(stop_trans - start_trans)
    transition_T_e_I = 125e3 + 75e3*np.cos(phi_trans)

    for i, theta in enumerate(theta_range):
        if theta <= start_trans:
            T_e_I.append(200e3)
        elif theta >= 100:
            T_e_I.append(50e3)
        else:
            T_e_I.append(transition_T_e_I[i])
    T_e_I = np.array(T_e_I)

    # Make harmonic T_e distribution - Model II of Kalousova
    T_e_II = 100e3 + 50e3*np.cos(2*np.radians(theta_range))

    # Make harmonic T_e distribution - Model III of Kalousova
    T_e_III = 100e3 + 50e3*np.cos(10*np.radians(theta_range))
    
    if model == 'MI':
        T_e_array = np.tile(T_e_I.reshape(-1, 1), (1, 4*(lmax+1)+1))
    elif model == 'MII':
        T_e_array = np.tile(T_e_II.reshape(-1, 1), (1, 4*(lmax+1)+1))
    elif model == 'MIII':
        T_e_array = np.tile(T_e_III.reshape(-1, 1), (1, 4*(lmax+1)+1))
    
    return pysh.SHGrid.from_array(T_e_array).expand()
    
def load_inputs(lmax, model):
    """
    Load in the GMM3 potential and MOLA topography up to lmax. Use these to 
    obtain mean planetary radius R, geoid (pot*R) and g0. Also loads in Te map
    up to lmax_Te_fit (which )
    """
    pot  = pysh.datasets.Mars.GMM3(lmax=lmax)
    topo = pysh.datasets.Mars.MOLA_shape(lmax=lmax)
    # R = topo.coeffs[0,0,0]
    R = 3395e3  # Kalousova-defined value
    pot = pot.change_ref(r0=R)
    geoid = pot*R
    # gm = pot.gm; 
    # g0 = gm/R**2
    g0 = 3.8  # Kalousova-defined value
    
    percent_C20 = 0.0
    print(f'\nSetting C20 of topo and geoid to {percent_C20}% of original value')
    topo.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo.coeffs[0, 2, 0]
    geoid.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid.coeffs[0, 2, 0]
    
    print(f'Loading Kalousova Te model {model} at lmax={lmax_Te_fit}')
    T_e_parent = make_synthetic_Te_model(lmax_Te_fit, model)
    print('Te map loaded in')
    return topo, geoid, T_e_parent, R, g0

def derive_D_a(T_e_parent, lmax):
    """
    Compute the flexural rigidity D and parameter alpha using the parent Te.
    Function first expands the parent Te map to a fine grid of 3*lmax, which
    is then used to compute D and alpha coefficients. D and alpha are then
    truncated to 2*lmax+1 because the coupling coefficients contain degrees
    up to the sum of two input degrees (the sum over LM goes from l-l' to l+l',
    i.e. 2*l).
    """
    Te_grid_exp_factor = 3
    g = T_e_parent.expand(lmax=Te_grid_exp_factor*lmax)
    print(f'Computing D and alpha using Te grid expanded to '
          f'lmax={Te_grid_exp_factor}*lmax')
    D = pysh.SHGrid.from_array(E*g.data**3/(12*(1-nu**2))).expand()
    D = pysh.SHCoeffs.from_array(D.coeffs[:, :2*lmax+1, :2*lmax+1])
    a = pysh.SHGrid.from_array(1.0/(E*g.data)).expand()
    a = pysh.SHCoeffs.from_array(a.coeffs[:, :2*lmax+1, :2*lmax+1])
    print('D and alpha computed\n')
    return D, a

def rotate_inputs(rot_angles, T_e_parent, D_clm, a_clm, topo_clm, geoid_clm):
    """
    Rotate the input topography, geoid, Te map and the resulting D and alpha 
    maps using the user-input rotation angles based on the rotation convention
    as used by pyshtools.
    """
    print(f"Rotating inputs with angles {rot_angles}...")
    alpha, beta, gamma = rot_angles
    T_e_parent = T_e_parent.rotate(alpha, beta, gamma)
    D_clm = D_clm.rotate(alpha, beta, gamma)
    a_clm = a_clm.rotate(alpha, beta, gamma)
    topo_clm = topo_clm.rotate(alpha, beta, gamma)
    geoid_clm = geoid_clm.rotate(alpha, beta, gamma)

    return T_e_parent, D_clm, a_clm, topo_clm, geoid_clm


# %% GAUNT FUNCTIONS
# Conversion factor to go to 4pi normalization
_sqrt4pi = np.sqrt(4.0*np.pi)

# Fast numeric W-coefficients evaluation
def W_numeric_A(l, lp, L, nu_val=0.25):
    """
    The W-term of Matrix A is the large term in square brackets of Kalousova 
    et al. (2012) equation A18.
    """
    dl=-l*(l+1)+2; dlp=-lp*(lp+1)+2; dL=-L*(L+1)+2
    br = dl**2+dlp**2+dL**2 + 2*(dl+dlp+dL) - 2*(dl*dlp+dl*dL+dlp*dL) - 8
    return dl*dlp + 0.25*(1.0-nu_val)*br

def W_numeric_B(l, lp, L, nu_val=0.25):
    """
    The W-term of Matrix B is the large term in square brackets of Kalousova 
    et al. (2012) equation A21. Only difference with W_numeric_A is in the 
    final 'return'-term, (1 + nu) vs (1 - nu)
    """
    dl=-l*(l+1)+2; dlp=-lp*(lp+1)+2; dL=-L*(L+1)+2
    br = dl**2+dlp**2+dL**2 + 2*(dl+dlp+dL) - 2*(dl*dlp+dl*dL+dlp*dL) - 8
    return dl*dlp + 0.25*(1.0+nu_val)*br


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
    all complex orders 
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
SoA plan format (see flatten_plan): the Gaunt coupling plan is stored as 8 flat
arrays instead of millions of nested (i, j, [(L, offsets, wA, wB), ...]) objects.

  cell_i[c], cell_j[c]   : matrix (row, col) of nonzero cell number c
  cell_start[c]          : first term index belonging to cell c
                          (cell c owns terms [cell_start[c] : cell_start[c+1]])
  term_L[t]              : SH degree L of term t
  term_off[t]            : M-offset into the (2L+1) D/alpha slice for term t
  term_wA[t], term_wB[t] : W_numeric_A/B * realGaunt for term t  (float64)

Reconstruction of one matrix cell c:
    A_ij = scaler_A * sum_{t in cell c} D_{term_L,term_off} * term_wA[t] 
            (+buoy if i==j)
This whole sum, for ALL cells at once, is done with one fancy-index gather 
plus np.add.reduceat -- no Python loop over terms (that loop was the lmax=50 
                                                   RAM killer).
"""

def flatten_plan(assembly_plan):
    """Convert nested [(i,j,[(L,off,wA,wB),...])] -> 8 flat arrays (see header)."""
    cell_i, cell_j, cell_start = [], [], []
    term_L, term_off, term_wA, term_wB = [], [], [], []
    cur = 0
    for i, j, L_entries in assembly_plan:
        cell_i.append(i); cell_j.append(j); cell_start.append(cur)
        for L, off, wAq, wBq in L_entries:
            n = len(off)
            term_L.append(np.full(n, L, np.int32))
            term_off.append(np.asarray(off, np.int32))
            term_wA.append(np.asarray(wAq, np.float64))
            term_wB.append(np.asarray(wBq, np.float64))
            cur += n
    cat = lambda lst, dt: (np.concatenate(lst) if lst else np.empty(0, dt))
    return {
        'cell_i':     np.asarray(cell_i, np.int32),
        'cell_j':     np.asarray(cell_j, np.int32),
        # append total term count as the final "start" so reduceat segments close
        'cell_start': np.asarray(cell_start + [cur], np.int64),
        'term_L':   cat(term_L,  np.int32),
        'term_off': cat(term_off,np.int32),
        'term_wA':  cat(term_wA, np.float64),
        'term_wB':  cat(term_wB, np.float64),
    }

def save_plan_soa(plan, lmax, nu, path):
    np.savez(path, lmax=np.int64(lmax), nu=np.float64(nu), **plan)

def load_plan_soa(path):
    gc.disable()
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files if k not in ('lmax','nu')}
    finally:
        gc.enable()


def _build_chunk_flat(args):
    """Build outer indices [i_lo, i_hi) and return FLAT arrays for that chunk.
    Returns a dict of 7 arrays (no sentinel here; the parent stitches starts)."""
    i_lo, i_hi, lmax, nu = args
    mode_map = make_mode_map(lmax)
    Nmode = len(mode_map)
 
    cell_i, cell_j, cell_nterms = [], [], []   # per-cell: row, col, #terms
    term_L, term_off, term_wA, term_wB = [], [], [], []
 
    for i in range(i_lo, i_hi):
        lv, mv = mode_map[i]
        for j in range(i, Nmode):
            lp, mp = mode_map[j]
            nterms_cell = 0
            for L in range(abs(lv - lp), lv + lp + 1):
                if (lv + lp + L) % 2:
                    continue
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
                term_wA.append((wA * q[nz]).astype(np.float64))
                term_wB.append((wB * q[nz]).astype(np.float64))
                nterms_cell += n
            if nterms_cell > 0:
                cell_i.append(i); cell_j.append(j); cell_nterms.append(nterms_cell)
 
    cat = lambda lst, dt: (np.concatenate(lst) if lst else np.empty(0, dt))
    return {
        'cell_i':      np.asarray(cell_i, np.int32),
        'cell_j':      np.asarray(cell_j, np.int32),
        'cell_nterms': np.asarray(cell_nterms, np.int64),   # per-cell term count
        'term_L':   cat(term_L,  np.int32),
        'term_off': cat(term_off, np.int32),
        'term_wA':  cat(term_wA, np.float64),
        'term_wB':  cat(term_wB, np.float64),
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
    cell_i      = np.concatenate([f['cell_i']      for f in frags])
    cell_j      = np.concatenate([f['cell_j']      for f in frags])
    cell_nterms = np.concatenate([f['cell_nterms'] for f in frags])
    term_L   = np.concatenate([f['term_L']   for f in frags])
    term_off = np.concatenate([f['term_off'] for f in frags])
    term_wA  = np.concatenate([f['term_wA']  for f in frags])
    term_wB  = np.concatenate([f['term_wB']  for f in frags])
    del frags  # release fragment memory immediately
 
    # Build cell_start from per-cell term counts: start[0]=0, cumulative, plus
    # a trailing sentinel = total term count (so reduceat segments close).
    cell_start = np.empty(cell_nterms.size + 1, np.int64)
    cell_start[0] = 0
    np.cumsum(cell_nterms, out=cell_start[1:])
 
    return {
        'cell_i': cell_i, 'cell_j': cell_j, 'cell_start': cell_start,
        'term_L': term_L, 'term_off': term_off,
        'term_wA': term_wA, 'term_wB': term_wB,
    }
 
 
def build_or_load_gaunt(lmax, nu, nproc=16):
    path = os.path.join(CACHE_DIR, f"gaunt_plan_v4_lmax{lmax}_nu{nu:.4f}.npz")
    if os.path.exists(path):
        t = time.perf_counter()
        plan = load_plan_soa(path)
        print(f"Loaded SoA plan lmax={lmax}: {plan['cell_i'].size:,} cells, "
              f"{plan['term_L'].size:,} terms in {time.perf_counter()-t:.2f}s")
    else:
        print(f"Building SoA plan lmax={lmax} (parallel v2, first time)…", flush=True)
        t = time.perf_counter()
        plan = build_plan_soa_parallel(lmax, nu, nproc=nproc)
        save_plan_soa(plan, lmax, nu, path)
        print(f"  Built {plan['cell_i'].size:,} cells / {plan['term_L'].size:,} "
              f"terms in {time.perf_counter()-t:.1f}s -> {os.path.getsize(path)/1e6:.1f} MB")
    return plan




# %% BEUTHE MODEL SOLVER

def solve_beuthe(topo_clm, geoid_clm, D_clm, a_clm, plan, lmax, R, T_e_0, g0):
    mode_map = make_mode_map(lmax)
    N = len(mode_map)
    Re   = R - T_e_0/2
    buoy = (Re/T_e_0)**3 * (Re/E) * g0 * (rho_m-rho_c)
    scaler_A = 1.0/(E*T_e_0**3)
    scaler_B = Re

    Dlm = pysh.shio.SHCilmToVector(D_clm.coeffs)
    alm = pysh.shio.SHCilmToVector(a_clm.coeffs)

    # ---- SoA FILL: gather + reduceat, no per-term Python loop -------------
    term_L  = plan['term_L'].astype(np.int64)
    k_offset = plan['term_off'].astype(np.int64)     # slice position k, 0..2L (k=L is M=0)
    # convert slice position k -> signed order M -> shtools coefficient offset
    Mvals = k_offset - term_L                            # M = k - L
    shtools_offset = np.where(Mvals == 0, 0,
            np.where(Mvals  > 0, Mvals, term_L + np.abs(Mvals)))   # shtools offset
    gidx  = term_L*term_L + shtools_offset                         # flat index into Dlm/alm
    prodA = Dlm[gidx] * plan['term_wA']
    prodB = alm[gidx] * plan['term_wB']
    starts = plan['cell_start'][:-1]
    cellA = np.add.reduceat(prodA, starts) * scaler_A
    cellB = np.add.reduceat(prodB, starts) * scaler_B
    seg_len = np.diff(plan['cell_start'])
    cellA[seg_len == 0] = 0.0; cellB[seg_len == 0] = 0.0

    ci, cj = plan['cell_i'], plan['cell_j']

    # ---- scatter per-cell values into dense blocks (loop over CELLS) ------
    A = np.zeros((N, N)); B = np.zeros((N, N))
    for c in range(ci.size):
        i, j = int(ci[c]), int(cj[c])
        vA = cellA[c] + (buoy if i == j else 0.0)
        vB = cellB[c]
        A[i, j] = vA;  B[i, j] = vB
        if i != j:
            A[j, i] = vA;  B[j, i] = vB     # operators are symmetric

    # diagonal coupling blocks a_l, b_l (the R^3 Delta' and -1/R Delta')
    d_l = np.array([-l*(l+1)+2 for l,_ in mode_map])
    diag_a = ((Re/T_e_0)**3 / E) * d_l
    diag_b = -d_l

    # assemble 2N x 2N dense system
    M = np.zeros((2*N, 2*N))
    M[:N, :N]   = A
    M[N:, N:]   = B
    M[:N, N:]   = np.diag(diag_a)
    M[N:, :N]   = np.diag(diag_b)

    # pin degree 0 and 1 (rigid-body / translation freedom)
    for idx,(l,_) in enumerate(mode_map):
        if l in (0,1):
            M[idx, :] = 0.0; M[idx, idx] = 1.0
            M[idx+N, :] = 0.0; M[idx+N, idx+N] = 1.0

    # RHS: topographic load
    fac = (Re/T_e_0)**3 * (rho_c*g0*Re)/E
    y_topo = -fac*(topo_clm.coeffs - geoid_clm.coeffs)
    y_unstr = pysh.shio.SHCilmToVector(y_topo)
    def elem(l,m,v):
        off = 0 if m==0 else (m if m>0 else l+abs(m))
        return v[l*l+off]
    y = np.array([elem(l,m,y_unstr) for l,m in mode_map])
    rhs = np.concatenate([y, np.zeros(N)])
    for idx,(l,_) in enumerate(mode_map):
        if l in (0,1): rhs[idx] = 0.0; rhs[idx+N] = 0.0

    sol = np.linalg.solve(M, rhs)
    w_sol = sol[:N]

    w_coeffs = np.zeros((2, lmax+1, lmax+1))
    for idx,(l,m) in enumerate(mode_map):
        if m >= 0: w_coeffs[0,l,m]     = w_sol[idx]
        else:      w_coeffs[1,l,abs(m)] = w_sol[idx]
    return pysh.SHCoeffs.from_array(w_coeffs, normalization='4pi')

# %% MAIN LOOP & PLOTTING

if __name__ == "__main__":
    t_begin = time.perf_counter()
    selftest_gaunt()
    LMAX_REF = max(LMAX_RUNS)
    topo_p, geoid_p, T_e_parent, R, g0 = load_inputs(LMAX_REF, model)
    T_e_0 = T_e_parent.coeffs[0,0,0]
    D_clm, a_clm  = derive_D_a(T_e_parent, LMAX_REF)

    
    solutions = {}
    fig, ax = plt.subplots(figsize=(10,7))
    for lmax_run in LMAX_RUNS:
        topo_clm  = truncate(topo_p,  lmax_run)
        geoid_clm = truncate(geoid_p, lmax_run)
        plan  = build_or_load_gaunt(lmax_run, nu)        
        
        do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
        for rotation in ([0, 1] if do_rotation_check else [0]):
            linestyle = 'solid' if rotation == 0 else 'dashed'
            if rotation == 1:
                T_e_rot, D_use, a_use, topo_use, geoid_use = rotate_inputs(
                    rotate_angles, T_e_parent, D_clm, a_clm, 
                    topo_clm, geoid_clm)
            else:
                D_use, a_use, topo_use, geoid_use = (D_clm, a_clm, 
                                                     topo_clm, geoid_clm)

            print('Start solving of system')
            t = time.perf_counter()
            w = solve_beuthe(topo_use, geoid_use, D_use, a_use, plan, 
                             lmax_run, R, T_e_0, g0)
            print(f'Finished solving of system in {(time.perf_counter()-t):.1f}s\n')
            solutions[lmax_run, rotation] = w
            w.plot_spectrum(ax=ax, show=False, 
                            legend=(
                                f'lmax={lmax_run}'+ 
                                (f', rotated {rotate_angles}' 
                                 if rotation else '')), 
                            plot_dict={'linestyle': linestyle})
    ax.set_title('Power spectra of w (Beuthe-model, Plesa Te Map, M1)')
    ax.legend(); plt.tight_layout()
    if SaveFigs:
        plt1_title = (f'Power spectra w, M1, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath1 = os.path.join(SavePath, plt1_title)
        plt.savefig(FigPath1, dpi=200)
    plt.show() 
    plt.close()


    # Residual vs reference, plot only if there are more than one lmax runs
    if len(LMAX_RUNS)>1:
        S_ref = solutions[LMAX_REF, 0].spectrum()
        fig2, ax2 = plt.subplots(figsize=(8,5))
        for lmax_run in LMAX_RUNS[:-1]:
            do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
            for rotation in ([0, 1] if do_rotation_check else [0]):
                linestyle = 'solid' if rotation == 0 else 'dashed'
                S = solutions[lmax_run, rotation].spectrum()
                l = np.arange(2, lmax_run+1)
                ax2.plot(l, np.abs(S[2:]/S_ref[2:lmax_run+1]-1.0)*100, '.-', 
                         label=(f'lmax={lmax_run} vs {LMAX_REF}'
                                + (f', rotated {rotate_angles}' 
                                   if rotation else '')), 
                         linestyle=linestyle)
        ax2.set_xlabel('degree l'); ax2.set_ylabel(r'$|S_l/S_l^{ref}-1|$*100%')
        ax2.legend(); ax2.grid(True)
        ax2.set_title(f'Residual vs lmax_ref={LMAX_REF}')
        plt.tight_layout(); 
        if SaveFigs:
            plt2_title = (f'Residuals w power, M1, lmax_run={LMAX_RUNS}, '
                          f'lmaxTe={lmax_Te_fit}'
                          + (f', rotated {rotate_angles}' if rotation else '') 
                          + '.png')
            FigPath2 = os.path.join(SavePath, plt2_title)
            plt.savefig(FigPath2, dpi=200)
        plt.show(); plt.close()


    # Plot inputs Te, D and alpha
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(14,12))
    do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    if do_rotation_check:
        T_e_use_clm, D_use_clm, a_use_clm, _, _ = rotate_inputs(
            rotate_angles, T_e_parent, D_clm, a_clm, 
            topo_clm, geoid_clm)
    else:
        T_e_use_clm, D_use_clm, a_use_clm = T_e_parent, D_clm, a_clm
    T_e_parent_km = T_e_use_clm.expand(lmax=LMAX_REF)
    T_e_parent_km.data = T_e_parent_km.data/1e3
    T_e_parent_km.plot(ax=ax1, 
                       cmap=cmap2, 
                       colorbar='right', 
                       cb_label=r'$T_e \ [m]$')
    ax1.set_title(f'Te input map (Plesa et al. 2018), exp. to lmax={LMAX_REF}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    D_use_clm.expand(lmax=LMAX_REF).plot(ax=ax2, 
                                        cmap=cmap2, 
                                        colorbar='right', 
                                        cb_label=r'$D \ [N\cdot m]$')  
    ax2.set_title(f'Flexural rigidity D (Te-derived), exp. to lmax={LMAX_REF}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    a_use_clm.expand(lmax=LMAX_REF).plot(ax=ax3, 
                                        cmap=cmap2, 
                                        colorbar='right', 
                                        cb_label=r'$E \ [m/N$]') 
    ax3.set_title(f'Parameter alpha (Te-derived), exp. to lmax={LMAX_REF}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    plt.tight_layout()
    if SaveFigs:
        plt3_title = (f'Inputs Te, D and alpha, M1, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath3 = os.path.join(SavePath, plt3_title)
        plt.savefig(FigPath3, dpi=200)
    plt.show(); plt.close()
    
        

    # 2D deflection map + difference between lmax runs
    do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    if do_rotation_check:
        w_fine = pysh.SHGrid.from_array(
                solutions[LMAX_REF, 1].expand(lmax=LMAX_REF).data/1e3)
        if len(LMAX_RUNS)>1:
            lo = LMAX_RUNS[-2] 
            d = (solutions[LMAX_REF, 1].coeffs[:, :lo+1, :lo+1] 
                 - solutions[lo, 1].coeffs[:, :lo+1, :lo+1])
            w_diff = pysh.SHCoeffs.from_array(d).expand(lmax=LMAX_REF)
    else:
        w_fine = pysh.SHGrid.from_array(
                solutions[LMAX_REF, 0].expand(lmax=LMAX_REF).data/1e3)        
        if len(LMAX_RUNS)>1:
            lo = LMAX_RUNS[-2] 
            d = (solutions[LMAX_REF, 0].coeffs[:, :lo+1, :lo+1] 
                 - solutions[lo, 0].coeffs[:, :lo+1, :lo+1])
            w_diff = pysh.SHCoeffs.from_array(d).expand(lmax=LMAX_REF)
            
    if len(LMAX_RUNS)>1:
        fig3, (a1,a2) = plt.subplots(2,1, figsize=(12,10))
        w_fine.plot(ax=a1, cmap=cmap1, colorbar='right', cb_label='w [km]')
        a1.set_title(f'Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
                     + (f', rot={rotate_angles}' if rotation else ''))

        a1.contour(w_fine.data>0, 
                   levels=[0.99], 
                   extent=(0,360,-90,90), 
                   colors='k', 
                   origin='upper')
        w_diff.plot(ax=a2, cmap=cmap1, colorbar='right', cb_label='w diff [m]')
        a2.set_title(f'Residual w: lmax={LMAX_REF} minus lmax={lo}'
                     + (f', rot={rotate_angles}' if rotation else ''))
        
    else:
        fig3, a1 = plt.subplots(figsize=(12,10))
        w_fine.plot(ax=a1, cmap=cmap1, colorbar='right', cb_label='w [km]')
        a1.set_title(f'Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
                     + (f', rot={rotate_angles}' if rotation else ''))

        a1.contour(w_fine.data>0, 
                   levels=[0.99], 
                   extent=(0,360,-90,90), 
                   colors='k', 
                   origin='upper')
    
    plt.tight_layout()
    if SaveFigs:
        plt4_title = (f'Displacement w 2D map, M1, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath4 = os.path.join(SavePath, plt4_title)
        plt.savefig(FigPath4, dpi=200)
        print(f"Saved Figures to subfolder: {SavePath}")
    plt.show(); plt.close()

print(f'\nTotal model runtime: {time.perf_counter() - t_begin}s')