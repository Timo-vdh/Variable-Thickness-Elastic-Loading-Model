# -*- coding: utf-8 -*-
"""
Beuthe (2008) variable-thickness flexure solver — Model 3 (M3)

Model for the variable thickness deformations of a thin elastic spherical shell,
including consoidal term of the tangential loading (the surface gradient of 
a scalar potential Omega).
Current model (M3) works with:
    - Beuthe (2008)'s equations 75 and 76 for the vertical displacement w and the 
      stress function F. 
    - Banerdt (1986)/Broquet & Andrews-Hanna (2023) equation for tangential
      loading potential Omega (with zero dc and zero drho).
    - Geoid self-consistency solving
    - Crustal thickness variations

Model 3 does not include:
    - Toroidal loading (V=0)
    - Mantle density variations

Following Beuthe's model requires implementation of the differential operator 
A(a;b). Beuthe (2008) does not give a spectral method for this, but in Beuthe
(2010) this spectral notation is made. Kalousova et al. (2012) describe the 
system of equations 75 and 76 in full spectral notation. This system of
equations is solved in Model 1, and extended here for the inclusion of 
tangential loading potential Omega.

"""

import numpy as np
import pyshtools as pysh
import os
import time
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# %% INPUTS

nu      = 0.25
E       = 100.0e9
rho_l = 2900.0
rho_c = 2900.0 
rho_m = 3500.0
drho = rho_m - rho_c
drhol = rho_c - rho_l
T_c = 65e3                 # Arbitrary crustal thickness value, TBC


LMAX_RUNS  = [45]        # last entry is the reference resolution
rotate_angles = (0.0, 0.0, 0.0)
lmax_Te_fit = 45
CACHE_DIR  = "gaunt_cache"
cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cm.davos
cmap3 = cm.roma_r
os.makedirs(CACHE_DIR, exist_ok=True)

omega_On = True
strain = 14      # Set which Te map is used, strain-14, strain-17, or
                 # strain-0 (returns constant Te map with Te=average of Te-14)


SaveFigs = False
SavePath = "Plots/M3VarD_SPEC_FinalPlots"
os.makedirs(SavePath, exist_ok=True)

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

def load_Temap(lmax_Te_fit, strain=14):
    """
    Load in a Te map from a data file stored in the same directory as this 
    script. Return the LSQ-derived coefficients of the Te data array, using
    the data file's latitude and longitude values.
    
    Currently loads in the Plesa et al. (2018) data set 1 file,
    specifically the Te map from a strain rate of 1e-14 1/s. 
    A 1e-17 1/s strain rate Te map is also available
    """
    subfolder_Te_maps = "Elastic Thickness Input Maps"
    Te_filename = "grl58258-sup-0002-data_set_1.dat"
    Te_file_path = os.path.join(subfolder_Te_maps, Te_filename)
    df = pd.read_csv(Te_file_path, sep=r'\s+', comment='#',
                     header=None,
                     names=['longitude','latitude','crustal_thickness_km',
                            'heat_flow_mW_M3','Te_1e-14_km','Te_1e-17_km',
                            'T_150km_K','depth_1370km_km'],
                     usecols=['longitude','latitude','Te_1e-14_km','Te_1e-17_km'])
    Te_14 = df['Te_1e-14_km'].values*1e3
    Te_17 = df['Te_1e-17_km'].values*1e3
    
    print(f'Computing Te SHCoeffs from Te map up to lmax={lmax_Te_fit}')
    if strain == 14:
        print(f'For input strain=14, Te_mean = {np.mean(Te_14)/1e3:.2f} km')
        return pysh.SHCoeffs.from_least_squares(Te_14, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_14
    elif strain == 17:
        print(f'For input strain=17, Te_mean = {np.mean(Te_17)/1e3:.2f} km')
        return pysh.SHCoeffs.from_least_squares(Te_17, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_17
    elif strain == 0:
        Te_constant = np.mean(Te_14)
        print(f'For input strain=0, Te_mean = {Te_constant/1e3:.2f} km')
        Te_constant_array = Te_constant * np.ones([64800])
        return pysh.SHCoeffs.from_least_squares(Te_constant_array, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_14
    else:
        print('ERROR: Input strain rate of 1e-{strain} 1/s is not available'
              ' for data files of Plesa et al. (2018). Please select strain'
              ' rate exponent of 14 or 17, or change the data file manually.')
        
def load_inputs(lmax, strain=14):
    """
    Load in the GMM3 potential and MOLA topography up to lmax. Use these to 
    obtain mean planetary radius R, geoid (pot*R) and g0. Also loads in Te map
    up to lmax_Te_fit (which )
    """
    pot  = pysh.datasets.Mars.GMM3(lmax=lmax)
    topo = pysh.datasets.Mars.MOLA_shape(lmax=lmax)
    R = topo.coeffs[0,0,0]
    pot = pot.change_ref(r0=R)
    geoid = pot*R
    gm = pot.gm; 
    g0 = gm/R**2
    G = pysh.constants.G.value  # Gravitational constant
    mass = gm / G  # Mass of the planet

    percent_C20 = 0.0
    print(f'\nSetting C20 of topo and geoid to {percent_C20}% of original value')
    topo.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo.coeffs[0, 2, 0]
    geoid.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid.coeffs[0, 2, 0]
        
    print(f'Loading Te map at lmax={lmax_Te_fit}')
    T_e_parent,_ = load_Temap(lmax_Te_fit, strain)
    if strain == 0:
        T_e_0 = T_e_parent.coeffs[0,0,0]
        T_e_parent.coeffs[:,:,:] = 0
        T_e_parent.coeffs[0,0,0] = T_e_0
    print('Te map loaded in')
    
    return topo, geoid, T_e_parent, R, g0, mass

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
    grid = T_e_parent.expand(lmax=Te_grid_exp_factor*lmax)
    print(f'Computing D and alpha using Te grid expanded to '
          f'lmax={Te_grid_exp_factor}*lmax')
    if strain == 0:
        D = E*T_e_parent.coeffs[0,0,0]**3/(12*(1-nu**2))
        D_coef = grid.expand().copy()
        D_coef.coeffs[:,:,:] = 0
        D_coef.coeffs[0,0,0] = D
        D = pysh.SHCoeffs.from_array(D_coef.coeffs[:, :2*lmax+1, :2*lmax+1])
        
        a = 1.0/(E*T_e_parent.coeffs[0,0,0])
        a_coef = grid.expand().copy()
        a_coef.coeffs[:,:,:] = 0
        a_coef.coeffs[0,0,0] = a
        a = pysh.SHCoeffs.from_array(a_coef.coeffs[:, :2*lmax+1, :2*lmax+1])
    else:
        D = pysh.SHGrid.from_array(E*grid.data**3/(12*(1-nu**2))).expand()
        D = pysh.SHCoeffs.from_array(D.coeffs[:, :2*lmax+1, :2*lmax+1])
        a = pysh.SHGrid.from_array(1.0/(E*grid.data)).expand()
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
    all complex orders cM, now feed only those cM values that agree with the 
    selection rule for orders (m1+M3+m3=0). Drastically reduced number of 
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

def save_plan_soa(plan, lmax, nu, path):
    """ 
    Save SoA plan to directory.
    """
    np.savez(path, lmax=np.int64(lmax), nu=np.float64(nu), **plan)

def load_plan_soa(path):
    """
    Load in SoA plan, using disabled gar
    """
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k not in ('lmax','nu')}



def _build_chunk_flat(args):
    """
    Build outer indices [i_lo, i_hi) and return FLAT arrays for that chunk.
    Returns a dict of 7 arrays (no sentinel here; the parent stitches starts).
    """
    i_lo, i_hi, lmax, nu = args
    mode_map = make_mode_map(lmax)
    Nmode = len(mode_map)
 
    cell_i, cell_j, cell_nterms = [], [], []   # per-cell: row, col, #terms
    term_L, term_off, term_wA, term_wB, term_gaunt_bare = [], [], [], [], []
 
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
        'term_wA':  cat(term_wA, np.float64),
        'term_wB':  cat(term_wB, np.float64),
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
    term_wA         = np.concatenate([f['term_wA']  for f in frags])
    term_wB         = np.concatenate([f['term_wB']  for f in frags])
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
        'term_wA': term_wA, 
        'term_wB': term_wB,
        'term_gaunt_bare': term_gaunt_bare
    }
 
 
def build_or_load_gaunt(lmax, nu, nproc=16):
    path = os.path.join(CACHE_DIR, f"gaunt_plan_v4_lmax{lmax}_nu{nu:.4f}_tangential.npz")
    if os.path.exists(path):
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


# %% OMEGA-TERMS EQUATIONS (SUBSTITUTIONS INTO w-F EQUATIONS)

def Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, lmax, R, T_e_0, g0, mass):
    """
    Full set of terms for the Omega parameters of the first equation in the
    system of two equations. 
    
    At current stage, the SH function products are done by expansion
    into the spatial domain, performing the multiplication there and then
    transforming back to spatial domain. It should be possible to perform this
    product using the Gaunt coefficients too, which may be implemented at a 
    next stage.
    """
    
    Re = R - T_e_0/2
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    eps0  = 12.0*Re**2/T_e_0**2
    eta0  = eps0/(1.0 + eps0)          # = 0.9994349 for your run
    corr1 = eta0*Re/R                  # eq-1 correction  = 0.959903
    
    # Laplacian array for degrees
    lap_by_degree = np.array([-l * (l + 1) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ rhobar*(2*l+1) for l in range(2 * lmax + 1)])
    
    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    topo_grid = topo_clm.expand(lmax=3*lmax).data - R
    geoid_grid = geoid_clm.expand(lmax=3*lmax).data - R
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
     
    
    # Field RHS 1a: Te*H grid
    TeH_grid = T_e_parent_grid * topo_grid
    TeH_clm = pysh.SHGrid.from_array(TeH_grid).expand()
    TeH_clm = pysh.SHCoeffs.from_array(TeH_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field RHS ab: Te**3 * topo grid
    Te2_grid = T_e_parent_grid**2    

    
    # PREFACTORS OF THE RHS OMEGA TERMS
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    factor1a_omega = -2.0*Re**3 *rho_l *g0 *nu/(1.0-nu) * Kalousova_scaler1 * corr1
    factor1b_omega = rho_l * g0 * (Re/12.0) * nu/(1.0-nu) * Kalousova_scaler1 * corr1
    # Term b to be multiplied with Laplacian
    
    factorRHS_omega1_dc1 = 2*Re**3*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc2 = -Re/12*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc3 = -2*Re**3*( nu/(1-nu)*g_m / 3) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc4 = Re/12 * ( nu/(1-nu)*g_m / 3 ) * Kalousova_scaler1 * corr1

    
    
    # Perform multiplication with laplacian for 1b only, by multiplying it with 
    # the Te3*topo coefficients for the degrees l only
    TeH_lap = TeH_clm.copy()
    for l in range(TeH_lap.coeffs.shape[1]):
        TeH_lap.coeffs[:, l, :] *= lap_by_degree[l]
    TeH_lap_grid = TeH_lap.expand(lmax=3*lmax)
    TeH_lap_Te2_grid = TeH_lap_grid.data * Te2_grid.data
    TeH_lap_Te2_clm = pysh.SHGrid.from_array(TeH_lap_Te2_grid).expand()
    
    
    
    
    # pre-weighted topo  H' = H / phi^(l+2)
    Hp = pysh.SHGrid.from_array(topo_grid).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]):
        Hp.coeffs[:, l, :] *= 1.0 / RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=3*lmax).data
     
    # pre-weighted geoid  G' = rhobar(2l+1)/phi^(l+2) * G
    Gp = pysh.SHGrid.from_array(geoid_grid).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]):
        Gp.coeffs[:, l, :] *= rhobar2l1[l] / RTcR_l2[l]
    Gp_grid = Gp.expand(lmax=3*lmax).data
    
    # dc1 :  max * H'                          (no Laplacian)
    dc1_clm = pysh.SHGrid.from_array(TeTc_grid * Hp_grid).expand()
    dc1_clm = pysh.SHCoeffs.from_array(dc1_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # dc3 :  max * G'                          (no Laplacian)
    dc3_clm = pysh.SHGrid.from_array(TeTc_grid * Gp_grid).expand()
    dc3_clm = pysh.SHCoeffs.from_array(dc3_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # dc2 :  Te^2 * Laplacian( max * H' )      (Laplacian on the INNER product, as in 1b)
    tmp = dc1_clm.copy()
    for l in range(tmp.coeffs.shape[1]):
        tmp.coeffs[:, l, :] *= lap_by_degree[l]
    dc2_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=3*lmax).data).expand()
    dc2_clm = pysh.SHCoeffs.from_array(dc2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # dc4 :  Te^2 * Laplacian( max * G' )
    tmp = dc3_clm.copy()
    for l in range(tmp.coeffs.shape[1]):
        tmp.coeffs[:, l, :] *= lap_by_degree[l]
    dc4_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=3*lmax).data).expand()
    dc4_clm = pysh.SHCoeffs.from_array(dc4_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    
    # TeH_clm.coeffs[:,0,:] = 0
    # dc1_clm.coeffs[:,0,:] = 0
    # dc2_clm.coeffs[:,0,:] = 0
    # dc3_clm.coeffs[:,0,:] = 0
    # dc4_clm.coeffs[:,0,:] = 0
    
    def _omrhs1_term(coeffs_arr, label, factor):
        v = factor * coeffs_arr
        msg = []
        for _l in range(2, min(9, lmax + 1)):
            msg.append(f"l{_l}:{np.max(np.abs(v[:, _l, :_l+1])):.2e}")
        print(f"   {label:4s}: " + "  ".join(msg))
    print("\n  --- Omega_RHS1 per-term  (max|.| by degree) ---")
    _omrhs1_term(TeH_clm.coeffs,         "1a",  factor1a_omega)
    _omrhs1_term(TeH_lap_Te2_clm.coeffs, "1b",  factor1b_omega)
    _omrhs1_term(dc1_clm.coeffs,         "dc1", factorRHS_omega1_dc1)
    _omrhs1_term(dc2_clm.coeffs,         "dc2", factorRHS_omega1_dc2)
    _omrhs1_term(dc3_clm.coeffs,         "dc3", factorRHS_omega1_dc3)
    _omrhs1_term(dc4_clm.coeffs,         "dc4", factorRHS_omega1_dc4)
    print("  (sum of these = OmRHS1; the term ~1e6 at l=2 is the offender)")
    print("  ----------------------------------------------\n")
    
    # assembly: prefactors stay (they already hold the constants, /3 etc.); the
    # per-degree factors now live in the fields, so DO NOT multiply them again.
    Omega_RHS1_coeffs = ( factor1a_omega       * TeH_clm.coeffs[:, :lmax+1, :lmax+1]
                        + factor1b_omega       * TeH_lap_Te2_clm.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega1_dc1 * dc1_clm.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega1_dc2 * dc2_clm.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega1_dc3 * dc3_clm.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega1_dc4 * dc4_clm.coeffs[:, :lmax+1, :lmax+1] )
   
    # Then transform to an 'unstructured' vector (structure same as that of y in
    # solve_beuthe) 
    Omega_RHS1_unstr = pysh.shio.SHCilmToVector(Omega_RHS1_coeffs)
    
    return Omega_RHS1_unstr
   
 
    
def Omega_eq1_LHS(T_e_parent, lmax, R, T_e_0, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the first equation.
    
    A number of Te and alpha products occur in the LHS terms. These can be
    simplified, since alpha = 1/(E*Te), thus reducing to 1/E for the product.
    """
    
    Re = R - T_e_0/2
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    eps0  = 12.0*Re**2/T_e_0**2
    eta0  = eps0/(1.0 + eps0)          # = 0.9994349 for your run
    corr1 = eta0*Re/R                  # eq-1 correction  = 0.959903
    
    RTeR = (R - T_e_0) / R
    if T_e_0 <= T_c:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_c / rhobar) / RTeR**2
    else:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_m / rhobar) / RTeR**2
        
    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    # Te - Tc field
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data


    # Field 1a: Te
    Te_grid = T_e_parent_grid 
    Te_clm = pysh.SHGrid.from_array(Te_grid).expand()
    Te_clm = pysh.SHCoeffs.from_array(Te_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 1b: none, constant term for diagonal only
    
    
    # Field 1c: max(Te-Tc,0) 
    TeTc_clm = pysh.SHGrid.from_array(TeTc_grid).expand()
    TeTc_clm = pysh.SHCoeffs.from_array(TeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 1d: Te^2 * laplacian of Te
    # Te_lap = T_e_parent.copy()
    # for l in range(Te_lap.coeffs.shape[1]):
    #     Te_lap.coeffs[:, l, :] *= lap_by_degree[l]
    # Te_lap_grid = Te_lap.expand(lmax=3*lmax)
    
    # Te2_lap_Te_grid = T_e_parent_grid**2 * Te_lap_grid.data
    # Te2_lap_Te_clm = pysh.SHGrid.from_array(Te2_lap_Te_grid).expand()
    # Te2_lap_Te_clm = pysh.SHCoeffs.from_array(Te2_lap_Te_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    Te3_grid = T_e_parent_grid**3
    Te3_clm = pysh.SHGrid.from_array(Te3_grid).expand()
    Te3_clm = pysh.SHCoeffs.from_array(Te3_clm.coeffs[:, :2*lmax+1, :2*lmax+1])


    # Field 1e: Te^2 * laplacian ??? 
    # TODO: Find out to which SH function laplacian is applied here
    Te2_grid = T_e_parent_grid**2 
    Te2_clm = pysh.SHGrid.from_array(Te2_grid).expand()
    Te2_clm = pysh.SHCoeffs.from_array(Te2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 1f: Te^2 * laplacian of max(Te-Tc,0)
    # TeTc_lap = TeTc_clm.copy()
    # for l in range(TeTc_lap.coeffs.shape[1]):
    #     TeTc_lap.coeffs[:, l, :] *= lap_by_degree[l]
    # TeTc_lap_grid = TeTc_lap.expand(lmax=3*lmax)
    
    # Te2_lap_TeTc_grid = T_e_parent_grid**2 * TeTc_lap_grid.data 
    # Te2_lap_TeTc_clm = pysh.SHGrid.from_array(Te2_lap_TeTc_grid).expand()
    # Te2_lap_TeTc_clm = pysh.SHCoeffs.from_array(Te2_lap_TeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    Te2TeTc_grid = T_e_parent_grid**2 * TeTc_grid 
    Te2TeTc_clm = pysh.SHGrid.from_array(Te2TeTc_grid).expand()
    Te2TeTc_clm = pysh.SHCoeffs.from_array(Te2TeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    # Field dc1: max(Te-Tc,0)
    

    # Field dc2: Te^2 * laplacian of max(Te-Tc,0)



    # Te_clm.coeffs[:,0,:] = 0
    # TeTc_clm.coeffs[:,0,:] = 0
    # Te3_clm.coeffs[:,0,:] = 0
    # Te2_clm.coeffs[:,0,:] = 0
    # Te2TeTc_clm.coeffs[:,0,:] = 0

    
    # Calculate the six prefactor Omega-terms of matrix A
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    
    # 1a & 1d change to take difference in rho_l and rho_c and the surface gravity
    # 1c and 1f changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    # factorLHS_omega1a = -2*Re**3*rho_c*g_m*nu/(1-nu) * Kalousova_scaler1
    factorLHS_omega1a = -2*Re**3*drhol*g0*nu/(1-nu) * Kalousova_scaler1 *corr1

    factorLHS_omega1b = 2*Re**3*rho_c*g_m*T_c * Kalousova_scaler1 *corr1
    
    # factorLHS_omega1c = 2*Re**3*g_m*rho_m * Kalousova_scaler1
    factorLHS_omega1c = 2*Re**3*gTe*rho_m * Kalousova_scaler1 *corr1

    # factorLHS_omega1d = Re/12*rho_c*g_m*nu/(1-nu) * Kalousova_scaler1  # *Laplacian!
    factorLHS_omega1d = Re/12*drhol*g0*nu/(1-nu) * Kalousova_scaler1 *corr1  # *Laplacian!

    factorLHS_omega1e = -Re/12*g_m*rho_c*T_c * Kalousova_scaler1  *corr1      # *Laplacian!
    
    # factorLHS_omega1f = -Re/12*g_m*rho_m * Kalousova_scaler1           # *Laplacian!
    factorLHS_omega1f = -Re/12*gTe*rho_m * Kalousova_scaler1  *corr1          # *Laplacian!
 
    # factorLHS_omega1_dc1 = -2*Re**3*nu/(1-nu)*g_m*drho * Kalousova_scaler1
    # factorLHS_omega1_dc2 = Re/12*nu/(1-nu)*g_m*drho * Kalousova_scaler1                   # *Laplacian! 
    factorLHS_omega1_dc1 = 0
    factorLHS_omega1_dc2 = 0

    
    # Transform into SHtools vectorformat again
    Omega_LHS_1a_unstr = factorLHS_omega1a * pysh.shio.SHCilmToVector(Te_clm.coeffs)
    Omega_LHS_1b_unstr = factorLHS_omega1b
    Omega_LHS_1c_unstr = factorLHS_omega1c * pysh.shio.SHCilmToVector(TeTc_clm.coeffs)
    # Omega_LHS_1d_unstr = factorLHS_omega1d * pysh.shio.SHCilmToVector(Te2_lap_Te_clm.coeffs)
    Omega_LHS_1d_unstr = factorLHS_omega1d * pysh.shio.SHCilmToVector(Te3_clm.coeffs)
    Omega_LHS_1e_unstr = factorLHS_omega1e * pysh.shio.SHCilmToVector(Te2_clm.coeffs)
    # Omega_LHS_1f_unstr = factorLHS_omega1f * pysh.shio.SHCilmToVector(Te2_lap_TeTc_clm.coeffs)
    Omega_LHS_1f_unstr = factorLHS_omega1f * pysh.shio.SHCilmToVector(Te2TeTc_clm.coeffs)

    Omega_LHS_dc1_unstr = factorLHS_omega1_dc1 *  pysh.shio.SHCilmToVector(TeTc_clm.coeffs)
    Omega_LHS_dc2_unstr = factorLHS_omega1_dc2 * pysh.shio.SHCilmToVector(Te2TeTc_clm.coeffs)

    return (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr, 
            Omega_LHS_1d_unstr, Omega_LHS_1e_unstr, Omega_LHS_1f_unstr,
            Omega_LHS_dc1_unstr, Omega_LHS_dc2_unstr)
    
 
    
def Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_clm, lmax, R, T_e_0, g0, mass):
    """
    Full set of terms for the Omega parameters of the second equation in the
    system of two equations.
    
    At current stage, the SH function products are done by expansion
    into the spatial domain, performing the multiplication there and then
    transforming back to spatial domain. It should be possible to perform this
    product using the Gaunt coefficients too, which may be implemented at a 
    next stage.
    """
    
    Re = R - T_e_0/2
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    

    eps0  = 12.0*Re**2/T_e_0**2
    eta0  = eps0/(1.0 + eps0)          # = 0.9994349 for your run
    corr1 = eta0*Re/R                  # eq-1 correction  = 0.959903

    # Laplacian array for degrees (incl +2 term)
    lap2_by_degree = np.array([-l * (l + 1) +2 for l in range(2 * lmax + 1)])
    
    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ rhobar*(2*l+1) for l in range(2 * lmax + 1)])

    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    topo_grid = topo_clm.expand(lmax=3*lmax).data - R
    geoid_grid = geoid_clm.expand(lmax=3*lmax).data - R
    alpha_grid = a_clm.expand(lmax=3*lmax).data
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
    
    
    # Field RHS 2a: Te*H*alpha grid
    TeHa_grid = T_e_parent_grid * topo_grid * alpha_grid
    TeHa_clm = pysh.SHGrid.from_array(TeHa_grid).expand()
    TeHa_clm = pysh.SHCoeffs.from_array(TeHa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    # # Field RHS 2 dc1: alpha grid * max(Te-Tc) * H
    # TeTcHa_grid = TeTc_grid * topo_grid * alpha_grid
    # TeTcHa_clm = pysh.SHGrid.from_array(TeTcHa_grid).expand()
    # TeTcHa_clm = pysh.SHCoeffs.from_array(TeTcHa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # # Field RHS 2 dc2: alpha grid * max(Te-Tc) * G
    # TeTcGa_grid = TeTc_grid * geoid_grid * alpha_grid
    # TeTcGa_clm = pysh.SHGrid.from_array(TeTcGa_grid).expand()
    # TeTcGa_clm = pysh.SHCoeffs.from_array(TeTcGa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    # PREFACTORS OF THE RHS OMEGA TERMS
    Kalousova_scaler2 = eta0*Re**2/R
    factor2a_omega = -1.0* nu * rho_l * g0 * Kalousova_scaler2
    # Term to be multiplied with Laplacian+2
    
    factorRHS_omega2_dc1 = nu*g_m*rho_l * Kalousova_scaler2
    factorRHS_omega2_dc2 = -nu*g_m/3 * Kalousova_scaler2
    
    # Perform multiplication with laplacian2, by multiplying it with 
    # the TeHa coefficients for the degrees l only
    TeHa_lap = TeHa_clm.copy()
    # TeTcHa_lap = TeTcHa_clm.copy()
    # TeTcGa_lap = TeTcGa_clm.copy()
    for l in range(TeHa_lap.coeffs.shape[1]):
        TeHa_lap.coeffs[:, l, :] *= lap2_by_degree[l]
        # TeTcHa_lap.coeffs[:, l, :] *= (lap2_by_degree[l] / ReTcRe_l2[l])
        # TeTcGa_lap.coeffs[:, l, :] *= (lap2_by_degree[l] * rhobar2l1[l] / ReTcRe_l2[l])
    
    
    # # Make coeffs array of size lmax+1 for the RHS
    # Omega_RHS2_coeffs = (factor2a_omega * TeHa_lap.coeffs[:, :lmax+1, :lmax+1]
    #                     + (factorRHS_omega2_dc1 
    #                        * TeTcHa_lap.coeffs[:, :lmax+1, :lmax+1]) 
    #                     + (factorRHS_omega2_dc2
    #                        * TeTcGa_lap.coeffs[:, :lmax+1, :lmax+1]) 
    #                     ) 
      
    
    # (same H', G'; here each product also carries alpha and the Laplacian is +2)
    Hp = pysh.SHGrid.from_array(topo_grid).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]): Hp.coeffs[:, l, :] *= 1.0/RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=3*lmax).data
    Gp = pysh.SHGrid.from_array(geoid_grid).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]): Gp.coeffs[:, l, :] *= rhobar2l1[l]/RTcR_l2[l]
    Gp_grid = Gp.expand(lmax=3*lmax).data
     
    d_dc1 = pysh.SHGrid.from_array(TeTc_grid * Hp_grid * alpha_grid).expand()   # max*H'*alpha
    d_dc1 = pysh.SHCoeffs.from_array(d_dc1.coeffs[:, :2*lmax+1, :2*lmax+1])
    d_dc2 = pysh.SHGrid.from_array(TeTc_grid * Gp_grid * alpha_grid).expand()   # max*G'*alpha
    d_dc2 = pysh.SHCoeffs.from_array(d_dc2.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(d_dc1.coeffs.shape[1]):       # Laplacian+2 on the inner product (as in 2a)
        d_dc1.coeffs[:, l, :] *= lap2_by_degree[l]
        d_dc2.coeffs[:, l, :] *= lap2_by_degree[l]
        
    # TeHa_lap.coeffs[:,0,:] = 0
    # d_dc1.coeffs[:,0,:] = 0
    # d_dc2.coeffs[:,0,:] = 0
    
    Omega_RHS2_coeffs = ( factor2a_omega       * TeHa_lap.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega2_dc1 * d_dc1.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega2_dc2 * d_dc2.coeffs[:, :lmax+1, :lmax+1] )
    
    
    # Then transform to an 'unstructured' vector (structure same as that of y in
    # solve_beuthe) 
    Omega_RHS2_unstr = pysh.shio.SHCilmToVector(Omega_RHS2_coeffs)
    
    return Omega_RHS2_unstr 
    
    

def Omega_eq2_LHS(T_e_parent, a_clm, lmax, R, T_e_0, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the second equation.
    
    A number of Te and alpha products occur in the LHS terms. These can be
    simplified, since alpha = 1/(E*Te), thus reducing to 1/E for the product.
    """
    
    Re = R - T_e_0/2
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    eps0  = 12.0*Re**2/T_e_0**2
    eta0  = eps0/(1.0 + eps0)          # = 0.9994349 for your run
    corr1 = eta0*Re/R                  # eq-1 correction  = 0.959903

    RTeR = (R - T_e_0) / R
    if T_e_0 <= T_c:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_c / rhobar) / RTeR**2
    else:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_m / rhobar) / RTeR**2
    
    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    a_grid = a_clm.expand(lmax=3*lmax).data
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data


    # Field 2a: Te * alpha
    Tea_grid = T_e_parent_grid * a_grid
    Tea_clm = pysh.SHGrid.from_array(Tea_grid).expand()
    Tea_clm = pysh.SHCoeffs.from_array(Tea_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 2b: alpha
    a_clm_copy = a_clm.copy()
    a_clm_copy.coeffs = a_clm_copy.coeffs[:, :2*lmax+1, :2*lmax+1]
    
    # Field 2c: max(Te-Tc,0) * alpha
    TeTca_grid = TeTc_grid * a_grid
    TeTca_clm = pysh.SHGrid.from_array(TeTca_grid).expand()
    TeTca_clm = pysh.SHCoeffs.from_array(TeTca_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    # Field 2D - dc:  max(Te-Tc,0) * alpha
    
    
    # Tea_clm.coeffs[:,0,:] = 0
    # a_clm_copy.coeffs[:,0,:] = 0
    # TeTca_clm.coeffs[:,0,:] = 0
    
    
    # Calculate the three prefactor Omega-terms of matrix A
    Kalousova_scaler2 = eta0*Re**2/R
    # factorLHS_omega2a = -rho_c*g_m*nu * Kalousova_scaler2          # *Laplacian+2!
    # 2a change to take difference in rho_l and rho_c and the surface gravity
    factorLHS_omega2a = -drhol*g0*nu * Kalousova_scaler2         # *Laplacian+2!
    factorLHS_omega2b = (1-nu)*rho_c*g_m*T_c * Kalousova_scaler2 # *Laplacian+2!
    # factorLHS_omega2c = (1-nu)*rho_m*g_m * Kalousova_scaler2 # *Laplacian+2!
    
    # 2c changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    factorLHS_omega2c = (1-nu)*rho_m*gTe * Kalousova_scaler2 # *Laplacian+2! 
    
    # the fourth prefactor term for crustal thickness variations
    # factorLHS_omega2d_dc = -nu*drho*g_m*Kalousova_scaler2
    factorLHS_omega2d_dc = 0

    # Transform into SHtools vectorformat again
    Omega_LHS_2a_unstr = factorLHS_omega2a * pysh.shio.SHCilmToVector(Tea_clm.coeffs)
    Omega_LHS_2b_unstr = factorLHS_omega2b * pysh.shio.SHCilmToVector(a_clm_copy.coeffs)
    Omega_LHS_2c_unstr = factorLHS_omega2c * pysh.shio.SHCilmToVector(TeTca_clm.coeffs)
    
    Omega_LHS_2d_dc_unstr = factorLHS_omega2d_dc * pysh.shio.SHCilmToVector(TeTca_clm.coeffs)


    return Omega_LHS_2a_unstr, Omega_LHS_2b_unstr, Omega_LHS_2c_unstr, Omega_LHS_2d_dc_unstr



def q_lm(topo_clm, geoid_clm, lmax, R, T_e_0, g0, mass):
    
    Re = R - T_e_0/2
    
    topo_clm.coeffs[0,0,0] = 0
    geoid_clm.coeffs[0,0,0] = 0
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2   
    
    
    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(l+1) for degrees l
    RTcR_l1 = np.array([((R-T_c)/R)**(l) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(-l+1) for degrees l
    RTcR_negl1 = np.array([((R-T_c)/R)**(-l-1) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ rhobar*(2*l+1) for l in range(2 * lmax + 1)])
    
    # q_lm = 
    #   g0 * rho_l * topo_clm 
    # + g0 * rho_l * - geoid_clm
    # + gmoho * drho *- dc_lm
    # + gmoho * drho *- Gc_lm
    
    # dc_lm = 1/(drho * RTcRl2) * (rhoc*topo_clm - (rhobar*(2l+1)/3) * G_lm) + w_lm
    # Gc_lm = 3/(rhobar*(2l+1)) * (rhoc*RTcRl1*topo_clm 
    #                               + drho*RTcR3*w_lm 
    #                               + (-drho*RTcR3*/(rhoc * RTcRl2) * (rhoc*topo_clm 
    #                                                                  - (rhobar*(2l+1)/3) * G_lm) 
    #                                                                  + w_lm))
    
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    q_topo_term1 = g0*rho_c 
    q_topo_term2 = -g_m*rho_c         # / RTcRl2
    q_topo_term3 = -3*g0*drho*rho_c  # * RTcRl1/rhobar2l1
    q_topo_term4 = 3*g0*drho*rho_c   # * RTcR_nl1/rhobar2l1
    
    q_geoid_term1 = -g0*rho_c 
    q_geoid_term2 = g_m/3             # *rhobar2l1 / RTcRl2
    q_geoid_term3 = -g0*drho         # * RTcR_nl1


    # Perform the multiplications with degree-dependent terms
    ReTcRe_l2_Hlap = topo_clm.coeffs.copy()
    RTcR_l1_over_rhobar2l1_Hlap = topo_clm.coeffs.copy()
    RTcR_negl1_over_rhobar2l1_Hlap = topo_clm.coeffs.copy()
    rhobar2l1_over_ReTcRe_l2_Glap = geoid_clm.coeffs.copy()
    RTcR_negl1_Glap = geoid_clm.coeffs.copy()
    for l in range(ReTcRe_l2_Hlap.shape[1]):
        ReTcRe_l2_Hlap[:, l, :] *= (1/RTcR_l2[l])
        RTcR_l1_over_rhobar2l1_Hlap[:, l, :] *= (RTcR_l1[l] / rhobar2l1[l])
        RTcR_negl1_over_rhobar2l1_Hlap[:, l, :] *= (RTcR_negl1[l]/rhobar2l1[l])
        rhobar2l1_over_ReTcRe_l2_Glap[:, l, :] *= (rhobar2l1[l] / RTcR_l2[l])
        RTcR_negl1_Glap[:, l, :] *= RTcR_negl1[l]
    
    
    # topo_clm.coeffs[:,0,:] = 0
    # ReTcRe_l2_Hlap[:,0,:] = 0
    # RTcR_l1_over_rhobar2l1_Hlap[:,0,:] = 0
    # RTcR_negl1_over_rhobar2l1_Hlap[:,0,:] = 0
    # geoid_clm.coeffs[:,0,:] = 0
    # rhobar2l1_over_ReTcRe_l2_Glap[:,0,:] = 0
    # RTcR_negl1_Glap[:,0,:] = 0
    
    
    
    # Make coeffs array of size lmax+1 for the RHS
    q_coeffs = -Re**4 * Kalousova_scaler1 * (
                  q_topo_term1 
                   * topo_clm.coeffs[:, :lmax+1, :lmax+1]
                + q_topo_term2 
                   * ReTcRe_l2_Hlap[:, :lmax+1, :lmax+1] 
                + q_topo_term3
                   * RTcR_l1_over_rhobar2l1_Hlap[:, :lmax+1, :lmax+1] 
                + q_topo_term4 
                   * RTcR_negl1_over_rhobar2l1_Hlap[:, :lmax+1, :lmax+1] 
                + q_geoid_term1
                   * geoid_clm.coeffs[:, :lmax+1, :lmax+1] 
                + q_geoid_term2 
                   * rhobar2l1_over_ReTcRe_l2_Glap[:, :lmax+1, :lmax+1] 
                + q_geoid_term3
                   * RTcR_negl1_Glap[:, :lmax+1, :lmax+1] 
                ) 


    q_lm_unstr = pysh.shio.SHCilmToVector(q_coeffs)
        
    return q_lm_unstr



# %% A-TILDE DOUBLE CONVOLUTIONS FUNCTIONS

def build_conv_matrix(field_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N):
    """ C_X[i,j] = sum_{t in cell(i,j)} X[gidx_t] * gaunt_bare_t   (single field
        spectral convolution operator, symmetric)."""
    fld  = field_unstr[gidx] * gaunt_bare
    cell = np.add.reduceat(fld, starts); cell[seg_len == 0] = 0.0
    C = np.zeros((N, N))
    for c in range(ci.size):
        i, j = int(ci[c]), int(cj[c])
        C[i, j] = cell[c]
        if i != j:
            C[j, i] = cell[c]
    return C
 
def build_A_tilde_group2(Te_unstr, Te2_unstr, max_unstr,
                         f1d, f1e, f1f,
                         gidx, gaunt_bare, starts, seg_len, ci, cj, mode_map, N):
    """ Correct  Te^2 * Delta'( X * w )  operators, returned as a (generally
        NON-symmetric) dense N x N matrix to ADD into A_tilde group-2."""
    C_Te  = build_conv_matrix(Te_unstr,  gidx, gaunt_bare, starts, seg_len, ci, cj, N)
    C_Te2 = build_conv_matrix(Te2_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
    C_max = build_conv_matrix(max_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
 
    dl   = np.array([-l*(l+1) for l, _ in mode_map])      # Delta (no +2 in eq.1)
    Dlap = np.diag(dl)
 
    Te2_Lap = C_Te2 @ Dlap                                # Te^2 . Delta  (reused)
    M_1d = f1d * (Te2_Lap @ C_Te)                         # Te^2 Delta (Te  w)
    M_1e = f1e * (Te2_Lap)                                # Te^2 Delta (    w)  (Tc in f1e)
    M_1f = f1f * (Te2_Lap @ C_max)                        # Te^2 Delta (max w)
    return M_1d + M_1e + M_1f                             # NOT symmetric -- keep full




# %% FINAL OMEGA & dc EQUATIONS (COMPUTED AFTER w_lm IS KNOWN)

def compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, g0, R, T_e_0, lmax):
    """
    Equation for tangential loading potential Omega, following the definition
    as given in Broquet & Andrews-Hanna (2022), which is derived from Banerdt
    (1986). 
    
    In this M3, this equation has been rewritten into w-terms in order
    to maintain a 2Nx2N block matrix system, neglecting effects of crustal 
    thickness variations dc and mantle density variations dm. The solution for 
    Omega itself can therefore be obtained using the result for w_lm.
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    # Grids
    w_grid = w_clm.expand(lmax=lmax).data
    T_e_parent_grid = T_e_parent.expand(lmax=lmax).data
    topo_grid = topo_clm.expand(lmax=lmax).data - R
    geoid_grid = geoid_clm.expand(lmax=lmax).data - R
    # Te - Tc field
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data


    # gravity at the elastic base (depth Te) for the mantle column term
    RTeR = (R - T_e_0) / R
    if T_e_0 <= T_c:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_c / rhobar) / RTeR**2
    else:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_m / rhobar) / RTeR**2

    TeH_grid = T_e_parent_grid * topo_grid
    
    TeTcH_grid = TeTc_grid*topo_grid
    TeTcH_coeffs_RTcR = pysh.SHGrid.from_array(TeTcH_grid).expand()
    for l in range(TeTcH_coeffs_RTcR.coeffs.shape[1]):
        TeTcH_coeffs_RTcR.coeffs[:, l, :] *= 1/RTcR**(l+2)
    TeTcH_RTcR_grid = TeTcH_coeffs_RTcR.expand(lmax=lmax).data
    
    TeTcG_grid = TeTc_grid * geoid_grid
    TeTcG_rhobar_RTcR_coeffs = pysh.SHGrid.from_array(TeTcG_grid).expand()
    for l in range(TeTcG_rhobar_RTcR_coeffs.coeffs.shape[1]):
       TeTcG_rhobar_RTcR_coeffs.coeffs[:, l, :] *= rhobar*(2*l+1)/(3 * RTcR**(l+2) )
    TeTcG_rhobar_RTcR_grid = TeTcG_rhobar_RTcR_coeffs.expand(lmax=lmax).data
    
    # Compute Re*Omega_lm as the term Omega_lm (required in conversion between
    # Banerdt and Beuthe's formulations).  w-coefficient corrected to match the
    # solve: surface -> drhol*g0 (vanishes for rho_l=rho_c), mantle -> gTe.
    
    term_1 = nu/(1-nu)*rho_l*g0*TeH_grid
    term_2 = + nu/(1-nu)*g_m*rho_l * TeTcH_RTcR_grid
    term_3 = -drhol*g0*nu/(1-nu)*T_e_parent_grid *w_grid
    term_4 = rho_c*g_m*T_c *w_grid
    term_5 = rho_m*gTe*TeTc_grid *w_grid
    term_6 = - nu/(1-nu)*g_m*drhol * TeTc_grid * w_grid
    term_7 = - nu/(1-nu)*g_m * TeTcG_rhobar_RTcR_grid
    
    
    
    Omega_grid_data = (  term_1 
                       + term_2 
                       + term_3 
                       + term_4 
                       + term_5 
                       + term_6 
                       + term_7)
        
    
    Omega_grid = pysh.SHGrid.from_array(Omega_grid_data)
    Omega_clm = Omega_grid.expand()
    
    return Omega_clm


def compute_dc(w_clm, topo_clm, geoid_clm, R, T_e_0, lmax):
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
     
    geoid_clm_copy = geoid_clm.copy()
    for l in range(geoid_clm_copy.coeffs.shape[1]):
       geoid_clm_copy.coeffs[:, l, :] *= rhobar*(2*l+1)/3
    
    dc_clm = 1/drho * (rho_l*topo_clm + drhol*w_clm - geoid_clm_copy)
        
    for l in range(dc_clm.coeffs.shape[1]):
       dc_clm.coeffs[:, l, :] *= 1/(RTcR**(l+2)) 
        
    dc_clm = dc_clm + w_clm
    
    return dc_clm


def compute_Gc(w_clm, dc_clm, topo_clm, R, T_e_0, lmax):
    rhobar = mass * 3.0 / (4.0*np.pi) / R**3
    RTcR = (R-T_c)/R

    wmdc = w_clm - dc_clm                      # (w - dc)

    H_term = (rho_l*topo_clm + drhol*w_clm)    # (rho_l*H + drhol*w) * phi^(l+1)
    for l in range(H_term.coeffs.shape[1]):
        H_term.coeffs[:, l, :] *= RTcR**(l+1)

    wmdc_term = wmdc.copy()                    # drho * (w-dc) * phi^3
    wmdc_term.coeffs *= drho * RTcR**3         # phi^3 is degree-independent -> scalar, fine

    Gc_clm = H_term + wmdc_term                # bracket
    for l in range(Gc_clm.coeffs.shape[1]):    # times 3/(rhobar(2l+1))
        Gc_clm.coeffs[:, l, :] *= 3.0/(rhobar*(2*l+1))

    Gc_grid = Gc_clm.expand(lmax=lmax)
    return Gc_grid, Gc_clm


# %% STRESS AND STRAIN FIELDS


def O1(SH_function, lmax):
    """ Differential operator O_1 in 2D spherical geometry. """
    SH_function_grid = SH_function.expand(lmax=lmax)
    dtheta_grid = SH_function.gradient(lmax=lmax).theta
    dtheta_sh = dtheta_grid.expand()
    dtheta2_grid = dtheta_sh.gradient(lmax=lmax).theta
    return dtheta2_grid.data + SH_function_grid.data

def O2(SH_function, lmax):
    """ Differential operator O_2 in 2D spherical geometry. """
    SH_function_grid = SH_function.expand(lmax=lmax)
    dtheta_grid = SH_function.gradient(lmax=lmax).theta

    dphi_grid = SH_function.gradient(lmax=lmax).phi
    dphi_sh = dphi_grid.expand()
    dphi2_grid = dphi_sh.gradient(lmax=lmax).phi
    
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1)+1))
    cot_theta = 1/np.tan(theta_range)
    cot_theta[0] = 0; cot_theta[-1] = 0
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))
    
    return dphi2_grid.data + cot_theta_grid * dtheta_grid.data + SH_function_grid.data

def O3(SH_function, lmax):
    """ Differential operator O_3 in 2D spherical geometry. """
    dphi_grid = SH_function.gradient(lmax=lmax).phi

    dtheta_grid = SH_function.gradient(lmax=lmax).theta
    dtheta_sh = dtheta_grid.expand()
    dthetaphi_grid = dtheta_sh.gradient(lmax=lmax).phi

    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1)+1))
    cot_theta = 1/np.tan(theta_range)
    cot_theta[0] = 0; cot_theta[-1] = 0
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

    return dthetaphi_grid.data - cot_theta_grid * dphi_grid.data



def stress_fields(w_sol, F_sol, Omega_grid, T_e_parent, R, T_e_0, lmax):
    """
    Beuthe eqs (73). MISSING THE TOROIDAL STRESS FUNCTION H TERMS!
    """
    # Laplacian array for degrees
    lap2_by_degree = np.array([(-l * (l + 1) + 2) for l in range(2 * lmax + 1)])
    w_lap2 = w_sol.copy()
    for l in range(w_lap2.coeffs.shape[1]):
        w_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
    w_lap2_grid = w_lap2.expand(lmax=lmax)
    
    Te_grid = T_e_parent.expand(lmax=lmax)
    
    O1F = O1(F_sol, lmax); O2F = O2(F_sol, lmax); O3F = O3(F_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax) 
    
    xi = 12*R**2/Te_grid.data**2
    eta = xi/(1+xi)
    zeta = Te_grid.data/2
    Re = R - T_e_0/2
    
    sigma_tt = (eta/Te_grid.data * (O2F + Omega_grid.data) 
                + E/(Re*(1-nu**2))*(eta/xi - zeta/(Re+zeta)) * (w_lap2_grid.data - (1-nu)*O2w)
                )
    sigma_pp = (eta/Te_grid.data * (O1F + Omega_grid.data) 
                + E/(Re*(1-nu**2))*(eta/xi - zeta/(Re+zeta)) * (w_lap2_grid.data - (1-nu)*O1w)
                )
    sigma_tp = (eta/Te_grid.data * -O3F 
                + E/(Re*(1+nu))*(eta/xi - zeta/(Re+zeta)) *O3w
                )
    
    sigma_tt = pysh.SHGrid.from_array(sigma_tt)
    sigma_pp = pysh.SHGrid.from_array(sigma_pp)
    sigma_tp = pysh.SHGrid.from_array(sigma_tp)
    
    return sigma_tt, sigma_pp, sigma_tp


def strain_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0):
    """
    Beuthe Eqs (69). MISSING THE TOROIDAL DISPLACEMENT POTENTIAL T TERMS!
    """
    Re = R - T_e_0/2
    
    # Return diff operator applied S and w terms, in grid.data format
    O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    
    S_grid  = S_sol.expand(lmax=lmax)
    w_grid  = w_sol.expand(lmax=lmax)
    Te_grid = T_e_parent.expand(lmax=lmax)

    eps_t    = 1/Re * (O1S - S_grid.data + w_grid.data)    
    eps_p    = 1/Re * (O2S - S_grid.data + w_grid.data)
    gamma_tp = 1/Re * (2*O3S)
    
    kappa_t = -1/Re**2 * O1w
    kappa_p = -1/Re**2 * O2w
    tau     = -2/Re**2 * O3w
    
    zeta = Te_grid.data/2
    tot_strain_pref = zeta / (1+zeta/Re)
    
    tot_eps_tt = eps_t    + tot_strain_pref*kappa_t
    tot_eps_pp = eps_p    + tot_strain_pref*kappa_p
    tot_eps_tp = (gamma_tp + tot_strain_pref*tau)/2.0
    
    
    tot_eps_tt = pysh.SHGrid.from_array(tot_eps_tt)
    tot_eps_pp = pysh.SHGrid.from_array(tot_eps_pp)
    tot_eps_tp = pysh.SHGrid.from_array(tot_eps_tp)
    
    return tot_eps_tt, tot_eps_pp, tot_eps_tp


def cons_disp_S(w_sol, F_sol, Omega_grid, T_e_parent, a_clm, R, T_e_0, lmax):
    
    lap_by_degree = np.array([(-l * (l + 1)) for l in range(2 * lmax + 1)])
    lap2_by_degree = np.array([(-l * (l + 1) + 2) for l in range(2 * lmax + 1)])
    F_lap2 = F_sol.copy()
    w_lap2 = w_sol.copy()
    for l in range(F_lap2.coeffs.shape[1]):
        F_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
        w_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
    F_lap2_grid = F_lap2.expand(lmax=lmax)
    w_lap2_grid = w_lap2.expand(lmax=lmax)
    
    w_grid = w_sol.expand(lmax=lmax)
    a_grid = a_clm.expand(lmax=lmax)
    Te_grid = T_e_parent.expand(lmax=lmax)

    xi = 12*R**2/Te_grid.data**2
    eta = xi/(1+xi)
    Re = R - T_e_0/2
    
    lapl_S_grid = (Re*eta*a_grid.data*(1-nu)*(F_lap2_grid.data + 2*Omega_grid.data) 
              + eta/xi * w_lap2_grid.data  
              - 2*w_grid.data)
    
    lapl_S_lm = pysh.SHGrid.from_array(lapl_S_grid).expand()
     
    S_lm = lapl_S_lm.copy()
    S_lm.coeffs[:, 0, :] = 0.0     
    for l in range(1, S_lm.coeffs.shape[1]):
        S_lm.coeffs[:, l, :] /= lap_by_degree[l]
        
    return S_lm


def Principal_strainstress_angle(s_theta, s_phi, s_theta_phi):
    """
    Calculate principal strains, stresses, and
    their principal angles.

    Returns
    -------
    min_strain : array, size same as input arrays
        Array with the minimum principal horizontal strain or stress.
    max_strain : array, size same as input arrays
        Array with the maximum principal horizontal strain or stress.
    sum_strain : array, size same as input arrays
        Array with the sum of the principal horizontal strain or stress.
    principal_angle : array, size same as input arrays
        Array with the principal strain or stress direction in degrees.

    Parameters
    ----------
    s_theta : array, float, size(nlat, nlon)
        Array of the colatitude component of the stress or strain field.
    s_phi : array, float, size(nlat, nlon)
        Array of the longitude component of the stress or strain field.
    s_theta_phi : array, float, size(nlat, nlon)
        Array of the colatitude and longitude component of the stress or strain field.
    """

    min_strain = 0.5 * (
        (s_theta + s_phi) - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    max_strain = 0.5 * (
        (s_theta + s_phi) + np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    sum_strain = min_strain + max_strain
    principal_angle = 0.5 * np.arctan2(2 * s_theta_phi, s_theta - s_phi) * 180.0 / np.pi

    return min_strain, max_strain, sum_strain, principal_angle


# %% BEUTHE MODEL SOLVER

def solve_beuthe(topo_clm, geoid_clm, T_e_parent, D_clm, a_clm, plan, lmax, R,
                 T_e_0, g0, mass, rhs_override=None, omega_on=True):
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
    
    # Calculate the original A and B terms
    prodA = Dlm[gidx] * plan['term_wA']
    prodB = alm[gidx] * plan['term_wB']
    starts = plan['cell_start'][:-1]
    cellA = np.add.reduceat(prodA, starts) * scaler_A
    cellB = np.add.reduceat(prodB, starts) * scaler_B
    seg_len = np.diff(plan['cell_start'])
    cellA[seg_len == 0] = 0.0
    cellB[seg_len == 0] = 0.0

    ci, cj = plan['cell_i'], plan['cell_j']

    # Calculate the Omega LHS terms for equation 1
    (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr, 
     Omega_LHS_1d_unstr, Omega_LHS_1e_unstr, Omega_LHS_1f_unstr,
     Omega_LHS_dc1_unstr, Omega_LHS_dc2_unstr) = (
                Omega_eq1_LHS(T_e_parent, lmax, R, T_e_0, g0, mass))
    
    # group 1: terms 1a,1c, dc1 -- NO output Laplacian
    field_ac = (Omega_LHS_1a_unstr[gidx] + Omega_LHS_1c_unstr[gidx]
                + Omega_LHS_dc1_unstr[gidx]) * plan['term_gaunt_bare']
    cell_ac = np.add.reduceat(field_ac, starts)
    cell_ac[seg_len == 0] = 0.0
    
    # # group 2: terms 1d,1e,1f, dc2 -- carry output-degree Laplacian -l(l+1)
    # field_def = (Omega_LHS_1d_unstr[gidx] + Omega_LHS_1e_unstr[gidx]
    #              + Omega_LHS_1f_unstr[gidx]
    #              + Omega_LHS_dc2_unstr[gidx]) * plan['term_gaunt_bare']
    # cell_def = np.add.reduceat(field_def, starts)
    # cell_def[seg_len == 0] = 0.0
    
    # group 3: term 1b is a scalar, only applied on diagonal
    # see line vA
    
    
    # # apply output-degree Laplacian to group 2 using the OUTPUT degree (cell_i's l)
    # out_deg = np.array([mode_map[int(ci[c])][0] for c in range(ci.size)])
    # lap_out = -out_deg*(out_deg+1)
    # cell_def = cell_def * lap_out
    
    cellA_tilde = cell_ac #+ cell_def 
    cellA_tilde[seg_len == 0] = 0.0


    # Calculate the Omega LHS terms for equation 2
    (Omega_LHS_2a_unstr, Omega_LHS_2b_unstr, 
     Omega_LHS_2c_unstr, Omega_LHS_2d_dc_unstr) = (
        Omega_eq2_LHS(T_e_parent, a_clm, lmax, R, T_e_0, g0, mass))
    
    # terms 2a, 2b, 2c, 2d -- carry output-degree Laplacian -l(l+1)+2
    field_2a2b2c2d = (Omega_LHS_2a_unstr[gidx] 
                  + Omega_LHS_2b_unstr[gidx] 
                  + Omega_LHS_2c_unstr[gidx]
                  + Omega_LHS_2d_dc_unstr[gidx]) * plan['term_gaunt_bare']
    cell_2a2b2c2d = np.add.reduceat(field_2a2b2c2d, starts)
    cell_2a2b2c2d[seg_len == 0] = 0.0
    
    # apply output-degree Laplacian to group 2 using the OUTPUT degree (cell_i's l)
    out_deg = np.array([mode_map[int(ci[c])][0] for c in range(ci.size)])
    lap_out = -out_deg*(out_deg+1)+2
    cellb_tilde = cell_2a2b2c2d * lap_out



    # ---- scatter per-cell values into dense blocks (loop over CELLS) ------
    A = np.zeros((N, N))
    A_tilde = np.zeros((N, N))
    B = np.zeros((N, N))
    b_tilde = np.zeros((N, N))
    for c in range(ci.size):
        i, j = int(ci[c]), int(cj[c])
        vA = cellA[c] + (buoy if i == j and omega_on == False else 0.0)
        vA_tilde = cellA_tilde[c] + (Omega_LHS_1b_unstr if i == j else 0.0)
        vB = cellB[c]
        vb_tilde = cellb_tilde[c]
        
        A[i, j] = vA  
        A_tilde[i, j] = vA_tilde 
        B[i, j] = vB
        b_tilde[i, j] = vb_tilde
        if i != j:
            A[j, i] = vA 
            A_tilde[j, i] = vA_tilde
            B[j, i] = vB     # operators are symmetric
            b_tilde[j, i] = vb_tilde


    # diagonal coupling blocks a_l, b_l (the R^3 Delta' and -1/R Delta')
    d_l2 = np.array([-l*(l+1)+2 for l,_ in mode_map])
    diag_a = ((Re/T_e_0)**3 / E) * d_l2
    diag_b = -d_l2

    a = np.diag(diag_a)
    b = np.diag(diag_b)
    

    if omega_on:
        # ------------------------------------------------------------------
        # FIX (operator ordering): build the correctly-ordered group-2 LHS
        # operator  Te^2 . Delta'( X . w )  (1d: X=Te, 1e: X=1 with Tc folded
        # into its factor, 1f & dc2: X=max -- dc2 shares 1f's structure so its
        # factor is added to 1f).  Recompute the bare Te/Te^2/max fields and
        # the same scalar prefactors used in Omega_eq1_LHS (kept local so that
        # function is left untouched).  Returned matrix is generally
        # NON-symmetric, which is expected for a two-stage  conv . Lap . conv
        # operator; it is added straight into A.
        # NOTE: build_A_tilde_group2 forms three dense N x N convolution
        # matrices and two matmuls (~O(N^3)); a few seconds at lmax~45.
        # ------------------------------------------------------------------
        rhobar_loc = mass * 3.0 / 4.0 / np.pi / R**3
        RTcR_loc   = (R - T_c) / R
        g_m_loc    = g0 * (1.0 + (RTcR_loc**3 - 1.0) * rho_c / rhobar_loc) / RTcR_loc**2
        RTeR = (R - T_e_0) / R
        rhobar = mass * 3.0 / 4.0 / np.pi / R**3
        eps0  = 12.0*Re**2/T_e_0**2
        eta0  = eps0/(1.0 + eps0)          # = 0.9994349 for your run
        corr1 = eta0*Re/R                  # eq-1 correction  = 0.959903
        if T_e_0 <= T_c:
            gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_c / rhobar) / RTeR**2
        else:
            gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_m / rhobar) / RTeR**2
             
        Te_grid_loc   = T_e_parent.expand(lmax=3*lmax).data
        TeTc_grid_loc = np.array(Te_grid_loc - T_c)
        TeTc_grid_loc[TeTc_grid_loc < 0.0] = 0.0    # max(Te-Tc,0)
 
        Te_clm_loc  = pysh.SHGrid.from_array(Te_grid_loc).expand()
        Te_clm_loc  = pysh.SHCoeffs.from_array(Te_clm_loc.coeffs[:, :2*lmax+1, :2*lmax+1])
        Te2_clm_loc = pysh.SHGrid.from_array(Te_grid_loc**2).expand()
        Te2_clm_loc = pysh.SHCoeffs.from_array(Te2_clm_loc.coeffs[:, :2*lmax+1, :2*lmax+1])
        max_clm_loc = pysh.SHGrid.from_array(TeTc_grid_loc).expand()
        max_clm_loc = pysh.SHCoeffs.from_array(max_clm_loc.coeffs[:, :2*lmax+1, :2*lmax+1])
 
        Te_unstr  = pysh.shio.SHCilmToVector(Te_clm_loc.coeffs)
        Te2_unstr = pysh.shio.SHCilmToVector(Te2_clm_loc.coeffs)
        max_unstr = pysh.shio.SHCilmToVector(max_clm_loc.coeffs)
 
        # same prefactors as Omega_eq1_LHS (scaler_A == Kalousova_scaler1):
        f1d   =  Re/12 * drhol * g0 * nu/(1-nu)      * scaler_A * corr1   # surface: drhol*g0 (matches 1a/1d)
        f1e   = -Re/12 * g_m_loc * rho_c * T_c       * scaler_A * corr1   # crust column: g_m (matches 1e)
        f1f   = -Re/12 * gTe * rho_m                 * scaler_A * corr1   # mantle: gTe (matches 1f)
        # f_dc2 =  Re/12 * nu/(1-nu) * g_m_loc * drho  * scaler_A   # dc2 ~ 1f structure
        f_dc2 =  0                                                # dc2 ~ 1f structure
        # 1f and dc2 multiply the SAME operator Te^2.Delta'(max.w) -> sum factors
        A_tilde_group2 = build_A_tilde_group2(
            Te_unstr, Te2_unstr, max_unstr,
            f1d, f1e, f1f + f_dc2,
            gidx, plan['term_gaunt_bare'], starts, seg_len, ci, cj, mode_map, N)
 
        A = A + A_tilde + A_tilde_group2
        b = b + b_tilde

    # assemble 2N x 2N dense system
    M = np.zeros((2*N, 2*N))
    M[:N, :N]   = A
    M[N:, N:]   = B
    M[:N, N:]   = a
    M[N:, :N]   = b

    # pin degree 0 and 1 (rigid-body / translation freedom)
    for idx,(l,_) in enumerate(mode_map):
        if l in (0,1):
            M[idx, :] = 0.0
            M[idx, idx] = 1.0
            M[idx+N, :] = 0.0
            M[idx+N, idx+N] = 1.0

    # RHS: topographic load
    factor_y = (Re/T_e_0)**3 * (rho_c*g0*Re)/E
    y_topo = -factor_y*(topo_clm.coeffs - geoid_clm.coeffs)
    y1_unstr = pysh.shio.SHCilmToVector(y_topo)
    q_lm_unstr = q_lm(topo_clm, geoid_clm, lmax, R, T_e_0, g0, mass)
    Omega_RHS1_unstr = Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, lmax, R, T_e_0, g0, mass)
    Omega_RHS2_unstr = Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, 
                                           a_clm, lmax, R, T_e_0, g0, mass)
    
    def elem(l,m,v):
        off = 0 if m==0 else (m if m>0 else l+abs(m))
        return v[l*l+off]
    
    y1 = np.array([elem(l,m,y1_unstr) for l,m in mode_map])
    q = np.array([elem(l,m,q_lm_unstr) for l,m in mode_map])
    Omega_RHS1 = np.array([elem(l,m,Omega_RHS1_unstr) for l,m in mode_map])
    Omega_RHS2 = np.array([elem(l,m,Omega_RHS2_unstr) for l,m in mode_map])


    if omega_on:
        y1 = q + Omega_RHS1
        y2 = Omega_RHS2
    else:
        y1 = y1
        y2 = np.zeros(N)
    
    rhs = np.concatenate([y1, y2])
    for idx,(l,_) in enumerate(mode_map):
        if l in (0,1): rhs[idx] = 0.0; rhs[idx+N] = 0.0

    sol = np.linalg.solve(M, rhs)
    w_sol = sol[:N]
    F_sol = sol[N:]

    w_coeffs = np.zeros((2, lmax+1, lmax+1))
    F_coeffs = np.zeros((2, lmax+1, lmax+1))
    q_coeffs = np.zeros((2, lmax+1, lmax+1))
    for idx,(l,m) in enumerate(mode_map):
        if m >= 0: 
            w_coeffs[0,l,m]     = w_sol[idx]
            F_coeffs[0,l,m]     = F_sol[idx]
            q_coeffs[0,l,m]     = q[idx]
        else:      
            w_coeffs[1,l,abs(m)] = w_sol[idx]
            F_coeffs[1,l,abs(m)] = F_sol[idx]
            q_coeffs[1,l,abs(m)] = q[idx]
    return (pysh.SHCoeffs.from_array(w_coeffs, normalization='4pi'), 
            pysh.SHCoeffs.from_array(F_coeffs, normalization='4pi'),
            pysh.SHCoeffs.from_array(q_coeffs, normalization='4pi'),)



# %% MAIN LOOP & PLOTTING

if __name__ == "__main__":
    t_begin = time.perf_counter()
    selftest_gaunt()
    LMAX_REF = max(LMAX_RUNS)
    topo_p, geoid_p, T_e_parent, R, g0, mass = load_inputs(LMAX_REF, strain=strain)
    T_e_0 = T_e_parent.coeffs[0,0,0]
    print(f'T_e_0 = {T_e_0/1e3:.2f} km')
    D_clm, a_clm  = derive_D_a(T_e_parent, LMAX_REF)

    solutions_w = {}
    solutions_F = {}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,9))
    for lmax_run in LMAX_RUNS:
        topo_clm  = truncate(topo_p,  lmax_run)
        geoid_clm = truncate(geoid_p, lmax_run)
        plan  = build_or_load_gaunt(lmax_run, nu)        
                
        do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
        for rotation in ([0, 1] if do_rotation_check else [0]):
            linestyle = 'solid' if rotation == 0 else 'dashed'
            if rotation == 1:
                T_e_use, D_use, a_use, topo_use, geoid_use = rotate_inputs(
                    rotate_angles, T_e_parent, D_clm, a_clm, 
                    topo_clm, geoid_clm)
            else:
                T_e_use, D_use, a_use, topo_use, geoid_use = (
                    T_e_parent, D_clm, a_clm, topo_clm, geoid_clm)
                
            print('Start solving of system')
            t = time.perf_counter()
            w, F, q = solve_beuthe(topo_use, geoid_use, T_e_use, D_use, a_use, plan, 
                             lmax_run, R, T_e_0, g0, mass, omega_on=omega_On)
            print(f'Finished solving of system in {(time.perf_counter()-t):.1f}s\n')
            solutions_w[lmax_run, rotation] = w
            solutions_F[lmax_run, rotation] = F
            w.plot_spectrum(ax=ax1, show=False, 
                            legend=(
                                f'lmax={lmax_run}'+ 
                                (f', rotated {rotate_angles}' 
                                 if rotation else '')), 
                            plot_dict={'linestyle': linestyle})
            F.plot_spectrum(ax=ax2, show=False, 
                            legend=(
                                f'lmax={lmax_run}'+ 
                                (f', rotated {rotate_angles}' 
                                 if rotation else '')), 
                            plot_dict={'linestyle': linestyle})
    ax1.set_title('M3 - Power spectra of w (Beuthe-model, Plesa Te Map)')
    ax1.legend()
    ax1.set_ylim(1e-2)
    plt.tight_layout()
    if SaveFigs:
        plt1_title = (f'M3 - Power spectra w, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath1 = os.path.join(SavePath, plt1_title)
        plt.savefig(FigPath1, dpi=200)
    ax2.set_title('M3 - Stress function F (Beuthe-model, Plesa Te Map)')
    ax2.legend()
    plt.tight_layout()
    plt.show() 
    plt.close()



# %% PLOTS - POWER SPECTRUM OF RESIDUAL BETWEEN LAST TWO LMAX RUNS


    # # Residual vs reference, plot only if there are more than one lmax runs
    # if len(LMAX_RUNS)>1:
    #     S_ref = solutions_w[LMAX_REF, 0].spectrum()
    #     fig2, ax2 = plt.subplots(figsize=(8,5))
    #     for lmax_run in LMAX_RUNS[:-1]:
    #         do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    #         for rotation in ([0, 1] if do_rotation_check else [0]):
    #             linestyle = 'solid' if rotation == 0 else 'dashed'
    #             S = solutions_w[lmax_run, rotation].spectrum()
    #             l = np.arange(2, lmax_run+1)
    #             ax2.plot(l, np.abs(S[2:]/S_ref[2:lmax_run+1]-1.0)*100, '.-', 
    #                      label=(f'lmax={lmax_run} vs {LMAX_REF}'
    #                             + (f', rotated {rotate_angles}' 
    #                                if rotation else '')), 
    #                      linestyle=linestyle)
    #     ax2.set_xlabel('degree l'); ax2.set_ylabel(r'$|S_l/S_l^{ref}-1|$*100%')
    #     ax2.legend(); ax2.grid(True)
    #     ax2.set_title(f'M3 - Residual vs lmax_ref={LMAX_REF}')
    #     plt.tight_layout(); 
    #     if SaveFigs:
    #         plt2_title = (f'M3 - Residuals w power, lmax_run={LMAX_RUNS}, '
    #                       f'lmaxTe={lmax_Te_fit}'
    #                       + (f', rotated {rotate_angles}' if rotation else '') 
    #                       + '.png')
    #         FigPath2 = os.path.join(SavePath, plt2_title)
    #         plt.savefig(FigPath2, dpi=200)
    #     plt.show(); plt.close()


# %% PLOTS - INPUTS TOPOGRAPHY, Te, D AND alpha


    # # Plot inputs topography, Te, D and alpha
    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,9))
    # do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    # if do_rotation_check:
    #     T_e_use_clm, D_use_clm, a_use_clm, topo_use_clm, _ = rotate_inputs(
    #         rotate_angles, T_e_parent, D_clm, a_clm, 
    #         topo_clm, geoid_clm)
    # else:
    #     T_e_use_clm, D_use_clm, a_use_clm, topo_use_clm = T_e_parent, D_clm, a_clm, topo_clm
    
    # topography_km = topo_use_clm.expand(lmax=3*LMAX_REF)
    # topography_km.data = (topography_km.data - R)/1e3
    # topography_km.plot(ax=ax0, 
    #                    cmap=cmap2, 
    #                    colorbar='right', 
    #                    cb_label='Topographic height [km]')
    # ax0.set_title(f'M3 - MOLA topography map, exp. to lmax={LMAX_REF}'
    #               + (f', rot={rotate_angles}' if rotation else ''))
    
    # T_e_parent_km = T_e_use_clm.expand(lmax=3*lmax_Te_fit)
    # T_e_parent_km.data = T_e_parent_km.data/1e3
    # T_e_parent_km.plot(ax=ax1, 
    #                    cmap=cmap2, 
    #                    colorbar='right', 
    #                    cb_label=r'$T_e \ [km]$')
    # ax1.set_title(f'M3 - Te input map (Plesa et al. 2018), exp. to lmax={lmax_Te_fit}'
    #               + (f', rot={rotate_angles}' if rotation else ''))
    # D_use_clm.expand(lmax=3*lmax_Te_fit).plot(ax=ax2, 
    #                                     cmap=cmap2, 
    #                                     colorbar='right', 
    #                                     cb_label=r'$D \ [N\cdot m]$')  
    # ax2.set_title(f'M3 - Flexural rigidity D (Te-derived), exp. to lmax={lmax_Te_fit}'
    #               + (f', rot={rotate_angles}' if rotation else ''))
    # a_use_clm.expand(lmax=3*lmax_Te_fit).plot(ax=ax3, 
    #                                     cmap=cmap2, 
    #                                     colorbar='right', 
    #                                     cb_label=r'$\alpha \ [m/N$]') 
    # ax3.set_title(f'M3 - Parameter alpha (Te-derived), exp. to lmax={lmax_Te_fit}'
    #               + (f', rot={rotate_angles}' if rotation else ''))
    # plt.suptitle('M3 - Input maps topography & Te, and derived parameters D and $\\alpha$')
    # plt.tight_layout()
    # if SaveFigs:
    #     plt3_title = (f'M3 - Inputs Te, D and alpha, lmax_run={LMAX_RUNS}, '
    #                   f'lmaxTe={lmax_Te_fit}'
    #                   + (f', rotated {rotate_angles}' if rotation else '') 
    #                   + '.png')
    #     FigPath3 = os.path.join(SavePath, plt3_title)
    #     plt.savefig(FigPath3, dpi=200)
    # plt.show(); plt.close()
        
# %% PLOTS - 2D DEFLECTION MAP + RESIDUAL BETWEEN LAST TWO LMAX RUNS

    # # 2D deflection map + difference between lmax runs
    # do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    # if do_rotation_check:
    #     w_fine = pysh.SHGrid.from_array(
    #             solutions_w[LMAX_REF, 1].expand(lmax=3*LMAX_REF).data/1e3)
    #     if len(LMAX_RUNS)>1:
    #         lo = LMAX_RUNS[-2] 
    #         d = (solutions_w[LMAX_REF, 1].coeffs[:, :lo+1, :lo+1] 
    #              - solutions_w[lo, 1].coeffs[:, :lo+1, :lo+1])
    #         w_diff = pysh.SHCoeffs.from_array(d).expand(lmax=3*LMAX_REF)
    # else:
    #     w_fine = pysh.SHGrid.from_array(
    #             solutions_w[LMAX_REF, 0].expand(lmax=3*LMAX_REF).data/1e3)        
    #     if len(LMAX_RUNS)>1:
    #         lo = LMAX_RUNS[-2] 
    #         d = (solutions_w[LMAX_REF, 0].coeffs[:, :lo+1, :lo+1] 
    #              - solutions_w[lo, 0].coeffs[:, :lo+1, :lo+1])
    #         w_diff = pysh.SHCoeffs.from_array(d).expand(lmax=3*LMAX_REF)
            
    # if len(LMAX_RUNS)>1:
    #     fig3, (a1,a2) = plt.subplots(2,1, figsize=(12,10))
    #     w_fine.plot(ax=a1, cmap=cmap1, colorbar='right', cb_label='w [km]',
    #                 # cmap_limits=[-24,11]
    #                 )
    #     a1.set_title(f'M3 - Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
    #                  + (f', rot={rotate_angles}' if rotation else ''))

    #     a1.contour(w_fine.data>0, 
    #                levels=[0.99], 
    #                extent=(0,360,-90,90), 
    #                colors='k', 
    #                origin='upper')
    #     w_diff.plot(ax=a2, cmap=cmap1, colorbar='right', cb_label='w diff [m]', 
    #                 # cmap_limits=[-320,200]
    #                 )
    #     a2.set_title(f'M3 - Residual w: lmax={LMAX_REF} minus lmax={lo}'
    #                  + (f', rot={rotate_angles}' if rotation else ''))
        
    # else:
    #     fig3, a1 = plt.subplots(figsize=(12,10))
    #     w_fine.plot(ax=a1, cmap=cmap1, colorbar='right', cb_label='w [km]')
    #     a1.set_title(f'M3 - Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
    #                  + (f', rot={rotate_angles}' if rotation else ''))

    #     a1.contour(w_fine.data>0, 
    #                levels=[0.99], 
    #                extent=(0,360,-90,90), 
    #                colors='k', 
    #                origin='upper')
    
    # plt.tight_layout()
    # if SaveFigs:
    #     plt4_title = (f'M3 - Displacement w 2D map, lmax_run={LMAX_RUNS}, '
    #                   f'lmaxTe={lmax_Te_fit}'
    #                   + (f', rotated {rotate_angles}' if rotation else '') 
    #                   + '.png')
    #     FigPath4 = os.path.join(SavePath, plt4_title)
    #     plt.savefig(FigPath4, dpi=200)
    #     print(f"Saved Figures to subfolder: {SavePath}")
    # plt.show(); plt.close()

    
    
# %% PLOTS - OMEGA MAP

    # # 2D Omega map + Omega power spectrum
    # w_clm = solutions_w[LMAX_REF, 0]
    # geoid_clm = truncate(geoid_clm, LMAX_REF)
    # Omega_coeffs = compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, g0, R, T_e_0, LMAX_REF)
    # Omega_grid = Omega_coeffs.expand(lmax=3*LMAX_REF)
    
    # fig4, (a1,a2) = plt.subplots(2, 1, figsize=(12,10))
    # Omega_grid.plot(ax=a1,cmap=cmap1, cmap_limits=[-5e6, 3.5e6], colorbar='right', cb_label=r'$\Omega \ [N/m]$')
    # a1.set_title(f'M3 - Consoidal load potential Omega 2D map (lmax={LMAX_REF})'
    #              + (f', rot={rotate_angles}' if rotation else '')
    #                 )
    # Omega_grid.expand().plot_spectrum(ax=a2, 
    #                 legend=(f'lmax={LMAX_REF}'+ 
    #                     (f', rotated {rotate_angles}' 
    #                      if rotation else '')))
    # a2.set_title(f'M3 - Consoidal load potential Omega power spectrum (lmax={LMAX_REF})'
    #              + (f', rot={rotate_angles}' if rotation else '')
    #                 )
    # plt.tight_layout()
    # a2.set_ylim(1e7, 1e13)
    # plt.show()



# %% PLOTS - DSP-TYPE RESULTS OVERVIEW


    # w_clm = solutions_w[LMAX_REF, 0]
    # w_fine = pysh.SHGrid.from_array(
    #         solutions_w[LMAX_REF, 1].expand(lmax=3*LMAX_REF).data/1e3)
    
    # # Crustal thickness
    # dc_clm = compute_dc(w_clm, topo_clm, geoid_clm, R, T_e_0, LMAX_REF)
    # dc_clm_zeroed = dc_clm.copy()
    # dc_clm_zeroed.coeffs[0,0,0] = 0
    # dc_grid = dc_clm_zeroed.expand(lmax=3*LMAX_REF)/1e3
    # topo_grid = topo_clm.expand(lmax=3*LMAX_REF)/1e3 - R/1e3
    # T_c_grid = topo_grid.data + dc_grid.data - w_fine.data + T_c*np.ones((2*(3*LMAX_REF+1)+1, 4*(3*LMAX_REF+1)+1))/1e3
    # T_c_grid = pysh.SHGrid.from_array(T_c_grid)

    # # Make a final overview plot that is structured like Adrien's DSP plot
    # fig4, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8), dpi=100)
    # w_fine.plot(ax=ax1, 
    #             cmap=cmap3, 
    #             # cmap_limits=[-6.55, 3.6],
    #             tick_interval=[45, 30],
    #             colorbar='bottom', 
    #             cb_label='Upward displacement (km)')
    # ax1.contour(w_fine.data>0, 
    #            levels=[0.99], 
    #            extent=(0,360,-90,90), 
    #            colors='k', 
    #            origin='upper')
    # dc_grid.plot(ax=ax2, 
    #              cmap=cmap3,
    #              # cmap_limits=[-60, 40],
    #              tick_interval=[45, 30],
    #              ylabel=None,
    #              colorbar='bottom', 
    #              cb_label='Crustal root variations (km)')
    # ax2.yaxis.tick_right()
    # T_e_parent_km = T_e_use_clm.expand(lmax=3*lmax_Te_fit)
    # T_e_parent_km.data = T_e_parent_km.data/1e3
    # T_e_parent_km.plot(ax=ax3, 
    #                    cmap=cmap3,
    #                    tick_interval=[45, 30],
    #                    colorbar='bottom', 
    #                    cb_label='Elastic thickness map T_e (km)')
    # cax = ax3.figure.axes[-1]
    # pos = cax.get_position()
    # cax.xaxis.labelpad = 15
    # # 4. Modify the y-position (e.g., lower it by subtracting 0.05)
    # new_pos = [pos.x0, pos.y0 - 10, pos.width, pos.height]
    # cax.set_position(new_pos)

    # T_c_grid.plot(ax=ax4, 
    #               cmap=cmap3,
    #               # cmap_limits=[35,150],
    #               tick_interval=[45, 30],
    #               colorbar='bottom', 
    #               cb_label='Crustal thickness (km)')

    # plt.suptitle('M3 - Result maps', y=0.9)
    # # plt.tight_layout()
    # plt.show()
    # plt.close()




# %% PLOTS - RESIDUALS MAPS BETWEEN M2 AND M3 

    # # Residual map between M2 and M3
    # w_fine = pysh.SHGrid.from_array(
    #         solutions_w[LMAX_REF, 0].expand(lmax=LMAX_REF).data/1e3)
    # w_M2 = pysh.SHGrid.from_file('M2_result_files/M2_w_expandedto45_pySHGrid_lmax=45_rot=False_lmaxTeFit=45')
    # w_M2_lmax = 45
    # rotate_angles_M2 = (0.0, 0.0, 0.0)
    # fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,10))
    # w_M2.plot(ax=ax1, cmap_limits=[-25, 10], cmap=cmap3, colorbar='right', cb_label='w [km]')
    # ax1.set_title(f'M2 - Transverse displacement w Beuthe-model (lmax={w_M2_lmax})'
    #              + (f', rot={rotate_angles_M2}' if rotation else ''))
    # ax1.contour(w_M2.data>0, 
    #            levels=[0.99], 
    #            extent=(0,360,-90,90), 
    #            colors='k', 
    #            origin='upper')

    # w_fine.plot(ax=ax2, cmap_limits=[-25, 10], cmap=cmap3, colorbar='right', cb_label='w [km]')
    # ax2.set_title(f'M3 - Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
    #              + (f', rot={rotate_angles}' if rotation else ''))
    # ax2.contour(w_fine.data>0, 
    #            levels=[0.99], 
    #            extent=(0,360,-90,90), 
    #            colors='k', 
    #            origin='upper')

    # w_diff_M2M3 = w_M2.copy()
    # w_diff_M2M3.data = w_M2.data - w_fine.data
    # w_diff_M2M3.plot(ax=ax3, cmap=cmap3, colorbar='right', cb_label='w [km]')
    # ax3.set_title('Transverse displacement w residual M2 - M3')

    # plt.tight_layout()
    # plt.show()
    # plt.close()


# %% PLOTS - RESIDUALS MAPS BETWEEN DSP AND M3
    # Residual map between DSP and M3
    w_fine = pysh.SHGrid.from_array(
            solutions_w[LMAX_REF, 0].expand(lmax=3*LMAX_REF).data/1e3)
    w_clm = solutions_w[LMAX_REF, 0]
    dc_clm = compute_dc(w_clm, topo_clm, geoid_clm, R, T_e_0, LMAX_REF)
    dc_clm_zeroed = dc_clm.copy()
    dc_clm_zeroed.coeffs[0,0,0] = 0
    dc_grid = dc_clm_zeroed.expand(lmax=3*LMAX_REF)/1e3
    
    topo_grid = topo_clm.expand(lmax=3*LMAX_REF)/1e3 - R/1e3
    
    T_c_grid = topo_grid.data + dc_grid.data - w_fine.data + T_c*np.ones((2*(3*LMAX_REF+1)+1, 4*(3*LMAX_REF+1)+1))/1e3
    T_c_grid = pysh.SHGrid.from_array(T_c_grid)
    
    lmax_DSP = 3*LMAX_REF

    # w_DSP = pysh.SHGrid.from_file('DSP_result_files/DSP_w_expandedto135_pySHGrid_lmax=45_Tc=100000.0_Te=61220.0')
    # dc_DSP = pysh.SHGrid.from_file('DSP_result_files/DSP_dc_expandedto135_pySHGrid_lmax=45_Tc=100000.0_Te=61220.0')
    # Tc_DSP = pysh.SHGrid.from_file('DSP_result_files/DSP_Tc_expandedto135_pySHGrid_lmax=45_Tc=100000.0_Te=61220.0')
    # Tc_val_DSP = 100.0e3
    # Te_val_DSP = 61.22e3
    
    # w_DSP = pysh.SHGrid.from_file('DSP_result_files/DSP_w_expandedto135_pySHGrid_lmax=45_Tc=100000.0_Te=268120.0')
    # dc_DSP = pysh.SHGrid.from_file('DSP_result_files/DSP_dc_expandedto135_pySHGrid_lmax=45_Tc=100000.0_Te=268120.0')
    # Tc_DSP = pysh.SHGrid.from_file('DSP_result_files/DSP_Tc_expandedto135_pySHGrid_lmax=45_Tc=100000.0_Te=268120.0')
    # Tc_val_DSP = 100.0e3
    # Te_val_DSP = 268.12e3

    w_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_w_expandedto{3*LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te={T_e_0:.1f}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    dc_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_dc_expandedto{3*LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te={T_e_0:.1f}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    Tc_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_Tc_expandedto{3*LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te={T_e_0:.1f}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    Tc_val_DSP = 65.0e3
    Te_val_DSP = 268.12e3
    
    fig, ((ax1, ax4, ax7),
          (ax2, ax5, ax8),
          (ax3, ax6, ax9),) = plt.subplots(3,3, figsize=(20,11))
    w_DSP.plot(ax=ax1, 
               # cmap_limits=[-25, 10], 
               cmap=cmap3, colorbar='right', cb_label='w [km]')
    ax1.set_title(f'DSP - Transverse displacement w Beuthe-model (lmax={lmax_DSP})')
    ax1.contour(w_DSP.data>0, 
               levels=[0.99], 
               extent=(0,360,-90,90), 
               colors='k', 
               origin='upper')

    w_fine.plot(ax=ax2, 
                # cmap_limits=[-25, 10], 
                cmap=cmap3, colorbar='right', cb_label='w [km]')
    ax2.set_title(f'M3 - Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
                 + (f', rot={rotate_angles}' if rotation else ''))
    ax2.contour(w_fine.data>0, 
               levels=[0.99], 
               extent=(0,360,-90,90), 
               colors='k', 
               origin='upper')

    w_diff_DSPM3 = w_DSP.copy()
    w_diff_DSPM3.data = w_DSP.data - w_fine.data
    w_diff_DSPM3.plot(ax=ax3, cmap=cmap3, colorbar='right', cb_label='w [km]')
    ax3.set_title('Transverse displacement w residual M2 - M3')



    dc_DSP.plot(ax=ax4, cmap=cmap3, colorbar='right', cb_label='dc [km]')
    ax4.set_title('DSP - Crustal thickness variations')
    dc_grid.plot(ax=ax5, cmap=cmap3, colorbar='right', cb_label='dc [km]')
    ax5.set_title('M3 - Crustal thickness variations')

    dc_diff_DSPM3 = dc_DSP.copy()
    dc_diff_DSPM3.data = dc_DSP.data - dc_grid.data
    dc_diff_DSPM3.plot(ax=ax6, cmap=cmap3, colorbar='right', cb_label='dc [km]')
    ax6.set_title('Crustal thickness variations residual DSP - M3')

    
    Tc_DSP.plot(ax=ax7, cmap=cmap3, colorbar='right', cb_label='Tc [km]')
    ax7.set_title('DSP - Crustal thickness')
    T_c_grid.plot(ax=ax8, cmap=cmap3, colorbar='right', cb_label='Tc [km]')
    ax8.set_title('M3 - Crustal thickness')

    Tc_diff_DSPM3 = Tc_DSP.copy()
    Tc_diff_DSPM3.data = Tc_DSP.data - T_c_grid.data
    Tc_diff_DSPM3.plot(ax=ax9, cmap=cmap3, colorbar='right', cb_label='Tc [km]')
    ax9.set_title('Crustal thickness residual DSP - M3')

    plt.suptitle(f'Residual checks DSP and M3 for DSP Te={T_e_0/1e3} km, '
                f'constant Tc={Tc_val_DSP/1e3} km, expanded to lmax={lmax_DSP}\n'
                f'rho_c = {rho_c}, rho_l = {rho_l}, rho_m = {rho_m}')
    plt.tight_layout()
    plt.show()
    if SaveFigs:
        plt_savetitle = f'Residual_checks_DSP_M3_Te={Te_val_DSP/1e3}km_Tc={Tc_val_DSP/1e3}km_lmax={lmax_DSP}.png'
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=200)
    plt.close()
    
    
# %%
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,9))

    solutions_w[lmax_run, rotation].plot_spectrum(ax=ax1, show=False, 
                    legend=('M3 coeffs'), plot_dict={'linestyle': linestyle})
    w_coeffs_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_w_coeffs_expandedto{3*LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te={T_e_0:.1f}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')

    w_coeffs_DSP.coeffs[0,0,0] = 0
    w_coeffs_DSP.plot_spectrum(ax=ax1, show=False, 
                    legend=('DSP coeffs'), plot_dict={'linestyle': '--'})

    w_spectrum_diff = solutions_w[lmax_run, rotation].spectrum() - w_coeffs_DSP.spectrum()    
    l = np.arange(2,(LMAX_REF+3))
    ax2.plot(l, w_spectrum_diff, 
             label=('M3 - DSP'), 
             linestyle=linestyle)
    plt.tight_layout()
    plt.grid()
    # ax1.set_ylim(1e-5)
    ax2.set_xlim(0,44)
    plt.show()
    
    
    
# %%

    # from m3_admittance_diagnostic import admittance_diagnostic

    # res = admittance_diagnostic(
    #     w_M3   = solutions_w[LMAX_REF, 0],   # SHCoeffs, metres
    #     topo   = topo_clm,                   # your solver input
    #     geoid  = geoid_clm,                  # your solver input
    #     R      = R,
    #     lmax   = LMAX_REF,
    #     w_DSP  = w_DSP,                      # SHGrid loaded from file
    #     dsp_units_to_m = 1.0e3,             # your DSP files are in km -> 1e3
    #     lmin   = 3,
    #     make_plots = True,
    #     )

    # P_ratio = res['P_M3']/res['P_DSP']
    # c1 = np.nanmedian(P_ratio[30:])          # high-l constant = Signal 1
    # e = P_ratio/c1                            # residual shape = Signal 2
    # # for l in range(2,16): print(l, round(e[l],3))
    
    # phi_M3  = (R - T_c)/R
    # phi_DSP = (R - 65e3)/R
    # geom = (phi_M3/phi_DSP)**np.arange(LMAX_REF+1)   # continuation-exponent mismatch
    # for l in range(2,16): print(l, round(e[l],3), round(geom[l],3))
    
    
# %% PLOTS - STRESS AND STRAIN FIELDS

    lmax_stress_strain = 3*LMAX_REF

    w_clm = solutions_w[LMAX_REF, 0].expand(lmax=lmax_stress_strain).expand()
    F_clm = solutions_F[LMAX_REF, 0].expand(lmax=lmax_stress_strain).expand()

    Omega_coeffs = compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, g0, R, T_e_0, lmax_stress_strain)
    Omega_grid = Omega_coeffs.expand(lmax=lmax_stress_strain)
    sigma_tt, sigma_pp, sigma_tp = stress_fields(w_clm, F_clm, Omega_grid, T_e_parent, R, T_e_0, lmax_stress_strain)

    S_clm = cons_disp_S(w_clm, F_clm, Omega_grid, T_e_parent, a_clm, R, T_e_0, lmax_stress_strain)
    S_clm.coeffs[0,0,0] = 0
    w_clm.coeffs[0,0,0] = 0
    eps_tt, eps_pp, eps_tp = strain_fields(S_clm, w_clm, T_e_parent, lmax_stress_strain, R, T_e_0)

    eps_tt.data = eps_tt.data*1e3; eps_pp.data = eps_pp.data*1e3; eps_tp.data = eps_tp.data*1e3

    fig, ((ax1, ax4), 
          (ax2, ax5),
          (ax3, ax6)) = plt.subplots(3, 2, figsize=(12,11), dpi=100)
    sigma_tt.plot(ax=ax1, cmap=cmap1, tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field $\\sigma_{\\theta \\theta}$')
    sigma_pp.plot(ax=ax2, cmap=cmap1, tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field $\\sigma_{\\phi \\phi}$')
    sigma_tp.plot(ax=ax3, cmap=cmap1, tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field $\\sigma_{\\theta \\phi}$')
    
    eps_tt.plot(  ax=ax4, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field $\\hat \\epsilon_{\\theta \\theta}$ (x$10^{-3}$)')
    eps_pp.plot(  ax=ax5, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field $\\hat \\epsilon_{\\phi \\phi}$ (x$10^{-3}$)')
    eps_tp.plot(  ax=ax6, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field $\\hat \\epsilon{\\theta \\phi}$ (x$10^{-3}$)')

    plt.suptitle('M3 - Stress & strain fields')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    eps_tt.data = eps_tt.data/1e3; eps_pp.data = eps_pp.data/1e3; eps_tp.data = eps_tp.data/1e3


    (
        min_strain,
        max_strain,
        sum_strain,
        principal_angle,
    ) = Principal_strainstress_angle(-eps_tt.data, -eps_pp.data, -eps_tp.data)

    args_plot = dict(
        tick_interval=[45, 30],
        colorbar="bottom",
        cmap=cm.vik,
        cb_tick_interval=1,
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,11), dpi=100)
    
    pysh.SHGrid.from_array(min_strain * 1e3).plot(
        ax=ax1,
        ticks="WSne",
        cb_label="Minimum principal horizontal strain ($\\times 10^{-3}$)",
        cmap_limits=[-4, 4],
        **args_plot,
    )
    pysh.SHGrid.from_array(max_strain * 1e3).plot(
        ax=ax2,
        cb_label="Maximum principal horizontal strain ($\\times 10^{-3}$)",
        ticks="wSnE",
        ylabel=None,
        cmap_limits=[-4, 4],
        **args_plot,
    )
    pysh.SHGrid.from_array(sum_strain * 1e3).plot(
        ax=ax3,
        cb_label="Sum of principal horizontal strains ($\\times 10^{-3}$)",
        cmap_limits=[-3, 3],
        ticks="WSne",
        **args_plot,
    )
    pysh.SHGrid.from_array(principal_angle).plot(
        ax=ax4,
        cb_label="Principal angle (°)",
        ticks="wSnE",
        ylabel=None,
        cmap_limits=[-90, 90],
        tick_interval=[45, 30],
        colorbar="bottom",
        cmap=cm.vikO,
    )
    
    # Plot strain direction
    skip_i = int(LMAX_REF / 2)
    skip = (slice(None, None, skip_i), slice(None, None, skip_i))
    grid_long, grid_lat = np.meshgrid(
        pysh.SHGrid.from_array(principal_angle).lons(),
        pysh.SHGrid.from_array(principal_angle).lats(),
    )
    ones = np.ones(np.shape(principal_angle))
    ax4.quiver(
        grid_long[skip],
        grid_lat[skip],
        ones[skip],
        ones[skip],
        scale=5e1,
        angles=principal_angle[skip],
        color="g",
    )
    
    
    plt.show()
    
    
# %%
    

    # lmax_stress_strain = 3*LMAX_REF

    # w_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_w_expandedto{3*LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te=268120.0_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # Omega_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_Omega_expandedto{3*LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te=268120.0_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')

    # sigma_tt_DSP, sigma_pp_DSP, sigma_tp_DSP = stress_fields(w_DSP.expand(), F_clm, Omega_DSP, T_e_parent, R, T_e_0, lmax_stress_strain)

    # S_clm_DSP = cons_disp_S(w_DSP.expand(), F_clm, Omega_DSP, T_e_parent, a_clm, R, T_e_0, lmax_stress_strain)
    # eps_tt_DSP, eps_pp_DSP, eps_tp_DSP = strain_fields(S_clm_DSP, w_DSP.expand(), T_e_parent, lmax_stress_strain, R, T_e_0)

    # eps_tt_DSP.data = eps_tt_DSP.data*1e3; eps_pp_DSP.data = eps_pp_DSP.data*1e3; eps_tp_DSP.data = eps_tp_DSP.data*1e3

    # fig, ((ax1, ax4), 
    #       (ax2, ax5),
    #       (ax3, ax6)) = plt.subplots(3, 2, figsize=(12,11), dpi=100)
    # sigma_tt_DSP.plot(ax=ax1, cmap=cmap1, tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field DSP $\\sigma_{\\theta \\theta}$')
    # sigma_pp_DSP.plot(ax=ax2, cmap=cmap1, tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field DSP $\\sigma_{\\phi \\phi}$')
    # sigma_tp_DSP.plot(ax=ax3, cmap=cmap1, tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field DSP $\\sigma_{\\theta \\phi}$')
    
    # eps_tt_DSP.plot(  ax=ax4, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field DSP $\\hat \\epsilon_{\\theta \\theta}$ (x$10^{-3}$)')
    # eps_pp_DSP.plot(  ax=ax5, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field DSP $\\hat \\epsilon_{\\phi \\phi}$ (x$10^{-3}$)')
    # eps_tp_DSP.plot(  ax=ax6, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field DSP $\\hat \\epsilon{\\theta \\phi}$ (x$10^{-3}$)')

    # plt.suptitle('M3 DSP inputs - Stress & strain fields')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    # eps_tt_DSP.data = eps_tt_DSP.data/1e3; eps_pp_DSP.data = eps_pp_DSP.data/1e3; eps_tp_DSP.data = eps_tp_DSP.data/1e3


    # (
    #     min_strain_DSP,
    #     max_strain_DSP,
    #     sum_strain_DSP,
    #     principal_angle_DSP,
    # ) = Principal_strainstress_angle(-eps_tt_DSP.data, -eps_pp_DSP.data, -eps_tp_DSP.data)

    # args_plot = dict(
    #     tick_interval=[45, 30],
    #     colorbar="bottom",
    #     cmap=cm.vik,
    #     cb_tick_interval=1,
    # )
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,11), dpi=100)
    
    # pysh.SHGrid.from_array(min_strain_DSP * 1e3).plot(
    #     ax=ax1,
    #     ticks="WSne",
    #     cb_label="Minimum principal horizontal strain ($\\times 10^{-3}$)",
    #     cmap_limits=[-4, 4],
    #     **args_plot,
    # )
    # pysh.SHGrid.from_array(max_strain_DSP * 1e3).plot(
    #     ax=ax2,
    #     cb_label="Maximum principal horizontal strain ($\\times 10^{-3}$)",
    #     ticks="wSnE",
    #     ylabel=None,
    #     cmap_limits=[-4, 4],
    #     **args_plot,
    # )
    # pysh.SHGrid.from_array(sum_strain_DSP * 1e3).plot(
    #     ax=ax3,
    #     cb_label="Sum of principal horizontal strains ($\\times 10^{-3}$)",
    #     cmap_limits=[-3, 3],
    #     ticks="WSne",
    #     **args_plot,
    # )
    # pysh.SHGrid.from_array(principal_angle_DSP).plot(
    #     ax=ax4,
    #     cb_label="Principal angle (°)",
    #     ticks="wSnE",
    #     ylabel=None,
    #     cmap_limits=[-90, 90],
    #     tick_interval=[45, 30],
    #     colorbar="bottom",
    #     cmap=cm.vikO,
    # )
    
    # # Plot strain direction
    # skip_i = int(LMAX_REF / 2)
    # skip = (slice(None, None, skip_i), slice(None, None, skip_i))
    # grid_long, grid_lat = np.meshgrid(
    #     pysh.SHGrid.from_array(principal_angle_DSP).lons(),
    #     pysh.SHGrid.from_array(principal_angle_DSP).lats(),
    # )
    # ones = np.ones(np.shape(principal_angle_DSP))
    # ax4.quiver(
    #     grid_long[skip],
    #     grid_lat[skip],
    #     ones[skip],
    #     ones[skip],
    #     scale=5e1,
    #     angles=principal_angle_DSP[skip],
    #     color="g",
    # )
    
    
    # plt.show()
    
    
    
# %%
    

    # 2D Omega map + Omega power spectrum
    q_grid = q.expand(lmax=3*LMAX_REF)

    fig4, (a1,a2, ax3) = plt.subplots(3, 1, figsize=(12,10))
    q_grid.plot(ax=a1,cmap=cm.vik, colorbar='right', cb_label=r'$q_{lm} \ [N]$')
    a1.set_title(f'M3 - q_lm')
    q.plot_spectrum(ax=a2, 
                    legend=(f'lmax={LMAX_REF}'))
    a2.set_title(f'M3 - q_lm')
    
    q_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_q_expandedto{LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te=268120.0_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    q_DSP.plot_spectrum(ax=a2)
    q_diff = pysh.SHCoeffs.from_array(q_DSP.coeffs - q.coeffs)
    q_diff.plot_spectrum(ax=ax3)
    
    
    plt.tight_layout()
    a2.set_ylim(1e7, 1e19)
    a2.set_xlim(0,45)
    plt.show()
    
    
    
    
    
    
    
# %%
    
    w_clm = solutions_w[LMAX_REF, 0]
    dc_clm = compute_dc(w_clm, topo_clm, geoid_clm, R, T_e_0, LMAX_REF)
    _, Gc_clm = compute_Gc(w_clm, dc_clm, topo_clm, R, T_e_0, LMAX_REF)
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2   

    q_clm = rho_c*g0*(topo_clm.coeffs - geoid_clm.coeffs) + g_m * drho*(w_clm.coeffs - dc_clm.coeffs - Gc_clm.coeffs)
    q_clm = pysh.SHCoeffs.from_array(q_clm)
    q_clm.coeffs[0,0,0] = 0
    
    q_comp_grid = q_clm.expand(lmax=3*LMAX_REF, grid='DH')
    
    fig_q, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(12,12))
    q_comp_grid.plot(ax=ax1,cmap=cm.vik, colorbar='right')
    ax1.set_title('M3 q_lm computed')
    q_clm.plot_spectrum(ax=ax2, show=False, legend='q computed M3')
    ax2.set_title('M3 q_lm computed power spectrum')
    
    q_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_q_expandedto{LMAX_REF}_pySHGrid_lmax={LMAX_REF}_Tc=65000.0_Te=268120.0_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    q_DSP.plot_spectrum(ax=ax2, legend = 'q_lm DSP')
    q_diff = pysh.SHCoeffs.from_array(q_DSP.coeffs - q_clm.coeffs)
    q_diff.plot_spectrum(ax=ax3, legend = 'q DSP - q M3')
    
    plt.tight_layout()
    ax2.set_ylim(1e11,1e14)
    ax2.set_xlim(0,45)
    plt.show()
    
    
    
    

    
    
    print(f'\nTotal model runtime: {(time.perf_counter() - t_begin):.1f}s')