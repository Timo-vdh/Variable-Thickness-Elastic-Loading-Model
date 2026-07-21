# -*- coding: utf-8 -*-
"""
Beuthe (2008) variable-thickness flexure solver — Model 4 (M4)

Model for the variable thickness deformations of a thin elastic spherical shell,
including consoidal term of the tangential loading (the surface gradient of 
a scalar potential Omega).
Current model (M4) works with:
    - Beuthe (2008)'s equations 75 and 76 for the vertical displacement w and the 
      stress function F. 
    - Banerdt (1986)/Broquet & Andrews-Hanna (2023) equation for tangential
      loading potential Omega (with zero dc and zero drho).
    - Geoid self-consistency solving
    - Crustal root variations
    - Mantle density variations

Model 4 does not include:
    - Toroidal loading (V=0)

Following Beuthe's model requires implementation of the differential operator 
A(a;b). Beuthe (2008) does not give a spectral method for this, but in Beuthe
(2010) this spectral notation is made. Kalousova et al. (2012) describe the 
system of equations 75 and 76 in full spectral notation. This system of
equations is solved in Model 1, and extended here for the inclusion of 
tangential loading potential Omega_lm, crustal root variations dc_lm and 
mantle density variations drho_lm.

This model includes the thin-shell approximation factor eta that Beuthe and
Kalousova neglect in their final equations (Beuthe does include it in 
equations 58 and 66). 

"""

import numpy as np
import pyshtools as pysh
import os
import time
from palettable import scientific as scm
from cmcrameri import cm as cmc
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(1, 'C:/Users/Timov/Displacement_strain_planet/Displacement_strain_planet')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# %% INPUTS

nu    = 0.25
E     = 100.0e9
rho_l = 2900.0
rho_c = 2900.0 
rho_m = 3500.0
drho = rho_m - rho_c
drhol = rho_c - rho_l
T_c = 50e3                 # Arbitrary crustal thickness value, TBC
Te_input = 268.12e3

# Top and bottom depth of density variations drho_lm
Mt = 0
Mb = T_c

lmax = 45
# lmax_Te_fit = Resolution at which Te map is loaded, can be made higher than 
# lmax for finer resolution of inputs Te, D (=f{Te^3}) and alpha (=f{1/Te})
lmax_Te_fit = lmax      
CACHE_DIR  = "gaunt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cmc.broc
cmap3 = cmc.roma_r

strain = 0      # Set which Te map is used, strain-14, strain-17, or
                 # strain-0 (returns constant Te map with Te=Te_input)

# Select whether solving for crustal root or mantle density variations
# solve_for = 'drho_lm'
solve_for = 'dc_lm'

# %% BASIC FUNCTION DEFINITIONS

def make_mode_map(lmax):
    """
    Flatten all combinations of l,m into a flat array based on input lmax.
    """
    return [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]


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
        Te_constant_array = Te_input * np.ones([64800])
        return pysh.SHCoeffs.from_least_squares(Te_constant_array, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_14
    else:
        print('ERROR: Input strain rate of 1e-{strain} 1/s is not available'
              ' for data files of Plesa et al. (2018). Please select strain'
              ' rate exponent of 0, 14 or 17, or change the data file manually.')
        
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
    
    # Ensure all coefficients apart from 0,0 are truly zero in constant Te case
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

    ETA_FULL extension (Beuthe eqs 58/66, unsimplified): additionally returns
    the eta-weighted fields (eta*D, eta*alpha) -- used by the A/B operator
    convolutions -- and the eta field itself -- used by the A(eta;F) and
    A(eta;w) coupling blocks in solve_beuthe. The products are formed on the
    GRID (coefficient-vector products would be wrong: (eta*D)_lm != eta_lm*D_lm).
    The PLAIN a_clm is still returned and must keep feeding Omega_eq2_LHS/RHS
    and cons_disp_S: those carry their own eta bookkeeping (cons_disp_S
    already implements Beuthe's full eq 71; the omega wiring's exact
    constant-Te decomposition requires the eq-2 scale Y = Re^2 built from
    plain alpha with Kalousova_scaler2 = Re^2/R).

    eta convention: eta = 1/(1 + Te^2/(12*Re^2)) with Re = R - T_e_0/2 built
    from the reference (mean) thickness, matching DSP's eps/beta/eta
    definitions (Beuthe writes R for his mid-surface radius; DSP's Re
    convention is the one the benchmark established).

    Returns: D_clm, a_clm, D_eta_clm, a_eta_clm, eta_clm
    (the last three are None when ETA_FULL is False).
    """
    Te_grid_exp_factor = 3
    grid = T_e_parent.expand(lmax=Te_grid_exp_factor*lmax)
    print(f'Computing D and alpha using Te grid expanded to '
          f'lmax={Te_grid_exp_factor}*lmax')
    Te0_loc = T_e_parent.coeffs[0,0,0]
    Re_loc  = R - Te0_loc/2
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

        eta0_loc = 1.0/(1.0 + Te0_loc**2/(12.0*Re_loc**2))
        D_eta = pysh.SHCoeffs.from_array(D.coeffs.copy());  D_eta.coeffs *= eta0_loc
        a_eta = pysh.SHCoeffs.from_array(a.coeffs.copy());  a_eta.coeffs *= eta0_loc
        eta_coef = grid.expand().copy()
        eta_coef.coeffs[:,:,:] = 0
        eta_coef.coeffs[0,0,0] = eta0_loc
        eta_clm = pysh.SHCoeffs.from_array(eta_coef.coeffs[:, :2*lmax+1, :2*lmax+1])

    else:
        D = pysh.SHGrid.from_array(E*grid.data**3/(12*(1-nu**2))).expand()
        D = pysh.SHCoeffs.from_array(D.coeffs[:, :2*lmax+1, :2*lmax+1])
        a = pysh.SHGrid.from_array(1.0/(E*grid.data)).expand()
        a = pysh.SHCoeffs.from_array(a.coeffs[:, :2*lmax+1, :2*lmax+1])

        eta_grid = 1.0/(1.0 + grid.data**2/(12.0*Re_loc**2))
        D_eta = pysh.SHGrid.from_array(eta_grid*E*grid.data**3/(12*(1-nu**2))).expand()
        D_eta = pysh.SHCoeffs.from_array(D_eta.coeffs[:, :2*lmax+1, :2*lmax+1])
        a_eta = pysh.SHGrid.from_array(eta_grid/(E*grid.data)).expand()
        a_eta = pysh.SHCoeffs.from_array(a_eta.coeffs[:, :2*lmax+1, :2*lmax+1])
        eta_clm = pysh.SHGrid.from_array(eta_grid).expand()
        eta_clm = pysh.SHCoeffs.from_array(eta_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    print('D and alpha computed \n')
    return D, a, D_eta, a_eta, eta_clm

# %% drho_lm HELPER FUNCTION

def drho_layer(lmax, R, g0, mass):
    """
    Density-anomaly-layer kernels, transcribed verbatim from upstream DSP
    (Displacement_strain_planet/B1986_nmax.py). SINGLE SOURCE OF TRUTH --
    this block was previously duplicated in four functions and had drifted.

      M    : layer thickness (Mb - Mt)
      g_M  : gravity at the layer  (DSP's `gdrho`)
      B_1  : eq-(1) geoid kernel   = Cbar * R/(l+3) * [(Rt/R)^(l+3) - (Rb/R)^(l+3)]
      B_2  : eq-(2) moho-geoid kernel = Cbar * R/(l+3) * [RtRCl - RbRCl]
             (the (g0/g_m) prefactor of DSP eq (2) is applied OUTSIDE, by the
             caller, exactly as in DSP.)

    FIXES vs the previous in-line copies:
      * R_drho_mid was  (Mt + Mb)/2  -- a DEPTH where a RADIUS is required.
        Upstream DSP has  R_drho_mid = (R_top_drho + R_top_drho)/2, i.e.
        R_top_drho written twice, so the "mid" radius IS the top radius
        R - Mt. That (probable) upstream typo is reproduced verbatim here so
        the benchmark matches DSP bit-for-bit. With Mt=0, Mb=Tc the old code
        gave g_M too large by ~1e4 ((R_mid/R)^-2 ratio = 9.2e-5).
      * B_2 used  ((Rc/Rt)^(l+1) - Rt^3)/(Rc*R^2)  -- a DIFFERENCE of a
        dimensionless number and a length^3. DSP has a PRODUCT:
        RtRCl = [(Rt/Rc)^l if Rt <= Rc else (Rc/Rt)^(l+1)] * Rt^3/(Rc*R^2).
        The Rt <= Rc branch (i.e. Mt >= Tc) was also missing entirely.
    """
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    M      = Mb - Mt
    R_top  = R - Mt
    R_base = R - Mb
    R_c    = R - T_c
    R_mid  = (R_top + R_base) / 2.0
    rho_d  = rho_c if Mt <= T_c else rho_m
    g_M    = g0 * (1.0 + ((R_mid/R)**3 - 1.0) * rho_d / rhobar) / (R_mid/R)**2

    degs = np.arange(2*lmax + 1, dtype=float)
    Cp   = 3.0 / (rhobar * (2.0*degs + 1.0))
    Rl3  = R / (degs + 3.0)

    B_1 = Cp * Rl3 * ((R_top/R)**(degs + 3.0) - (R_base/R)**(degs + 3.0))

    RtRCl = ((R_top/R_c)**degs if R_top <= R_c else (R_c/R_top)**(degs + 1.0))
    RbRCl = ((R_base/R_c)**degs if R_base <= R_c else (R_c/R_base)**(degs + 1.0))
    RtRCl = RtRCl * R_top**3  / (R_c * R**2)
    RbRCl = RbRCl * R_base**3 / (R_c * R**2)
    B_2 = Cp * Rl3 * (RtRCl - RbRCl)

    return dict(M=M, g_M=g_M, B_1=B_1, B_2=B_2, Cp=Cp, Rl3=Rl3, rhobar=rhobar)



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
    """ 
    If a plan at the input lmax and Poisson's ratio exists in the cache 
    directory that is set at the top of the code, then this function
    will load it in. If it is not found, it will build it using multiple 
    processors. Number of processors can be set manually to either speed up
    building, or prevent overloading of computer.
    """
    
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

def Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, lmax, R, T_e_0, Re, g0, mass):
    """
    Full set of terms for the Omega parameters of the first equation in the
    system of two equations. 
    
    At current stage, the SH function products are done by expansion
    into the spatial domain, performing the multiplication there and then
    transforming back to spatial domain. It should be possible to perform this
    product using the Gaunt coefficients too, which may be implemented at a 
    next stage.
    """
        
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    # Thin-shell correction factors
    eps0  = 12.0*Re**2/T_e_0**2
    eta0  = eps0/(1.0 + eps0)
    corr1 = eta0*Re/R
    
    # Laplacian array for degrees
    lap_by_degree = np.array([-l * (l + 1) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ (rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])
    
    
    # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------    
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    topo_grid = topo_clm.expand(lmax=3*lmax).data - R
    geoid_grid = geoid_clm.expand(lmax=3*lmax).data - R
    Te2_grid = T_e_parent_grid**2    
    
    # max(Te - Tc, 0) field
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
     
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
    
    
    # ------- THE FIELDS FOR EACH TERM -------
    # Field RHS 1a: Te*H grid
    TeH_grid = T_e_parent_grid * topo_grid
    TeH_clm = pysh.SHGrid.from_array(TeH_grid).expand()
    TeH_clm = pysh.SHCoeffs.from_array(TeH_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field RHS 1b: Te**2 * Laplacian(Te * topo)
    # (Laplacian on the INNER product)
    TeH_lap = TeH_clm.copy()
    for l in range(TeH_lap.coeffs.shape[1]):
        TeH_lap.coeffs[:, l, :] *= lap_by_degree[l]
    TeH_lap_grid = TeH_lap.expand(lmax=3*lmax)
    TeH_lap_Te2_grid = TeH_lap_grid.data * Te2_grid.data
    TeH_lap_Te2_clm = pysh.SHGrid.from_array(TeH_lap_Te2_grid).expand()
    
    
    # Field dc1 :  max * H'    
    # (no Laplacian)
    dc1_clm = pysh.SHGrid.from_array(TeTc_grid * Hp_grid).expand()
    dc1_clm = pysh.SHCoeffs.from_array(dc1_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # Field dc2 :  Te^2 * Laplacian( max * H' )      
    # (Laplacian on the INNER product, as in 1b)
    tmp = dc1_clm.copy()
    for l in range(tmp.coeffs.shape[1]):
        tmp.coeffs[:, l, :] *= lap_by_degree[l]
    dc2_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=3*lmax).data).expand()
    dc2_clm = pysh.SHCoeffs.from_array(dc2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # dc3 :  max * G'   
    # (no Laplacian)  
    dc3_clm = pysh.SHGrid.from_array(TeTc_grid * Gp_grid).expand()
    dc3_clm = pysh.SHCoeffs.from_array(dc3_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # dc4 :  Te^2 * Laplacian( max * G' )
    # (Laplacian on the INNER product, as in 1b)
    tmp = dc3_clm.copy()
    for l in range(tmp.coeffs.shape[1]):
        tmp.coeffs[:, l, :] *= lap_by_degree[l]
    dc4_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=3*lmax).data).expand()
    dc4_clm = pysh.SHCoeffs.from_array(dc4_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    # ------- drho_lm VARIABLES AND FIELDS -------
    #  definitions of g_M, B_1 and B_2, and for the two fixes it encodes.)
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']
    g_M   = _dl['g_M']
    B_1   = _dl['B_1']
    Cp    = _dl['Cp']
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)
    
    # =====================================================================
    # drho_lm RHS fields (omega's drhom part; solve_for == 'drho_lm')
    # ---------------------------------------------------------------------
    # omega gains  P_hat * drhom / R, and with dc = 0 the eq-(1) elimination
    # gives, PER DEGREE,
    #     drhom_H,lm = -Cbar_l * rho_l * H_lm / B_1_l
    #     drhom_G,lm = +           G_lm / B_1_l
    # so the omega content is the SPATIAL product
    #     P_hat(theta,phi) * drhom(theta,phi).
    #
    # STRUCTURAL FIXES vs the previous version:
    #  1. P_hat was multiplied into H/G COEFFICIENT-WISE
    #     (`H.coeffs[:,l,:] *= P_frac.coeffs[:,l,:]`). That is not a product
    #     of two functions: (P*H)_lm != P_lm * H_lm. The product must be
    #     formed on the GRID and re-expanded.
    #  2. 1/B_1 was folded into P_hat's OWN coefficients, i.e. applied at the
    #     FIELD degree. It belongs to drhom, i.e. to the degree of H/G --
    #     it is applied to the H/G coefficients here, before the grid product.
    #  3. The c2-half Laplacian was applied to the weighted H/G BEFORE
    #     multiplying by P_hat. The operator is  Te^2 * Delta( P_hat * drhom ),
    #     so Delta acts on the PRODUCT.
    # =====================================================================
    Phat_g = MTeMt * TeMt0        # pure grid field
    
    def _drhom_part(source_coeffs, weights):
        """ 
        Per-degree weight on the source coefficients. Return grid for calcs. 
        """
        weighted_coeffs = pysh.SHCoeffs.from_array(np.array(source_coeffs.coeffs[:, :2*lmax+1, :2*lmax+1]))
        for l in range(weighted_coeffs.coeffs.shape[1]):
            weighted_coeffs.coeffs[:, l, :] *= weights[l]
        return weighted_coeffs.expand(lmax=3*lmax).data
    
    def _c1_c2(prod_grid):
        """
        Each Omega_eq1_RHS term has two halves due to two omega terms in the eq.
        Calculate each drho-term half and return as coeffs.
        c1-half = P_hat*drhom 
        c2-half = Te^2 * Delta(P_hat*drhom)
        """
        p_clm = pysh.SHGrid.from_array(prod_grid).expand()
        p_clm = pysh.SHCoeffs.from_array(p_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
        lap = pysh.SHCoeffs.from_array(np.array(p_clm.coeffs))
        for l in range(lap.coeffs.shape[1]):
            lap.coeffs[:, l, :] *= lap_by_degree[l]
        lap_g = Te2_grid.data * lap.expand(lmax=3*lmax).data
        lap_clm = pysh.SHGrid.from_array(lap_g).expand()
        lap_clm = pysh.SHCoeffs.from_array(lap_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
        return p_clm, lap_clm
    
    # degree-weights for two H-field terms
    wH_drho = np.array([-Cp[l] / B_1[l] for l in range(2*lmax+1)])
    # degree-weights for two G-field terms
    wG_drho = np.array([ 1.0 / B_1[l]   for l in range(2*lmax+1)])
    
    _topo_clm  = pysh.SHGrid.from_array(topo_grid).expand()
    _geoid_clm = pysh.SHGrid.from_array(geoid_grid).expand()
    field_drho1, field_drho2 = _c1_c2(Phat_g * _drhom_part(_topo_clm,  wH_drho))
    field_drho3, field_drho4 = _c1_c2(Phat_g * _drhom_part(_geoid_clm, wG_drho))




    # ------- THE PREFACTORS OF THE EQ1 RHS OMEGA TERMS -------
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    factor1a_omega = -2.0*Re**3 *rho_l *g0 *nu/(1.0-nu) * Kalousova_scaler1 * corr1
    factor1b_omega = rho_l * g0 * (Re/12.0) * nu/(1.0-nu) * Kalousova_scaler1 * corr1
    # Term b to be multiplied with Laplacian
    
    # factors from dc_lm inclusion
    factorRHS_omega1_dc1 = -2*Re**3*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc2 = Re/12*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc3 = 2*Re**3*( nu/(1-nu)*g_m ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc4 = -Re/12 * ( nu/(1-nu)*g_m ) * Kalousova_scaler1 * corr1
    
    # factors from drho_lm inclusion
    # RHS = -K1*R*(c1 + c2*Delta)*omega_content, and the drhom weights
    # (-Cbar*rho_l/B_1 for H, +1/B_1 for G) are now inside field_drho1..4,
    # so only the operator halves remain here:
    #   c1-half: -2Re^3*K1*corr1        c2-half: +Re/12*K1*corr1
    factorRHS_omega1_drho1 = -2*Re**3 * (-0.5*nu/(1-nu)*g_M*rho_l) * Kalousova_scaler1 * corr1  # * P_hat*drhom_H
    factorRHS_omega1_drho2 =  Re/12   * (-0.5*nu/(1-nu)*g_M*rho_l) * Kalousova_scaler1 * corr1  # * Te2*lapl(P_hat*drhom_H)
    factorRHS_omega1_drho3 = -2*Re**3 * (-0.5*nu/(1-nu)*g_M) * Kalousova_scaler1 * corr1  # * P_hat*drhom_G
    factorRHS_omega1_drho4 =  Re/12   * (-0.5*nu/(1-nu)*g_M) * Kalousova_scaler1 * corr1  # * Te2*lapl(P_hat*drhom_G)

    
    
    # ------- ASSEMBLY -------
    Omega_RHS1_coeffs = ( factor1a_omega       * TeH_clm.coeffs[:, :lmax+1, :lmax+1]
                        + factor1b_omega       * TeH_lap_Te2_clm.coeffs[:, :lmax+1, :lmax+1]
                        
                        + (factorRHS_omega1_dc1 * dc1_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega1_dc2 * dc2_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega1_dc3 * dc3_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega1_dc4 * dc4_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        
                        + (factorRHS_omega1_drho1 * field_drho1.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        + (factorRHS_omega1_drho2 * field_drho2.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0) 
                        + (factorRHS_omega1_drho3 * field_drho3.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        + (factorRHS_omega1_drho4 * field_drho4.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        )
   
    # Then transform to an 'unstructured' vector (structure same as that of y in
    # solve_beuthe) 
    Omega_RHS1_unstr = pysh.shio.SHCilmToVector(Omega_RHS1_coeffs)
    
    return Omega_RHS1_unstr
   
def Omega_eq1_LHS(T_e_parent, lmax, R, T_e_0, Re, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the first equation.
    """
        
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    # Thin-shell approximations/corrections
    eps0  = 12.0*Re**2/T_e_0**2
    eta0  = eps0/(1.0 + eps0)
    corr1 = eta0*Re/R
    
    # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------   
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    
    # Field Te
    Te_grid = T_e_parent_grid 
    Te_clm = pysh.SHGrid.from_array(Te_grid).expand()
    Te_clm = pysh.SHCoeffs.from_array(Te_clm.coeffs[:, :2*lmax+1, :2*lmax+1])  
    
    # Field Te^2 
    Te2_grid = T_e_parent_grid**2 
    Te2_clm = pysh.SHGrid.from_array(Te2_grid).expand()
    Te2_clm = pysh.SHCoeffs.from_array(Te2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field max(Te-Tc,0)
    TeTc_grid = T_e_parent_grid - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid = np.array(TeTc_grid.data)
    TeTc_grid[TeTc_grid < 0.0] = 0  
    TeTc_clm = pysh.SHGrid.from_array(TeTc_grid).expand()
    TeTc_clm = pysh.SHCoeffs.from_array(TeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field gTe (variable-Te fix)
    # gravity at the LOCAL shell-base depth,
    # mantle branch only -- every gTe-carrying term also carries max(Te-Tc,0),
    # which is zero exactly where the density branch would switch. Monopole at
    # constant Te => benchmark preserved.
    RTeR_grid = (R - T_e_parent_grid) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid <= T_c, rho_c, rho_m)
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2    
    
    # Field Tc if Tc < Te else 0
    Tcind_grid_1 = np.where(T_e_parent_grid > T_c, T_c, 0.0)
    Tcind_clm_1  = pysh.SHGrid.from_array(Tcind_grid_1).expand()
    Tcind_clm_1  = pysh.SHCoeffs.from_array(Tcind_clm_1.coeffs[:, :2*lmax+1, :2*lmax+1])


    # ------- THE PREFACTORS OF THE EQ1 RHS OMEGA TERMS -------
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    
    # 1a & 1d change to take difference in rho_l and rho_c and the surface gravity
    # 1c and 1f changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    factorLHS_omega1a = -2*Re**3*drhol*g0*nu/(1-nu) * Kalousova_scaler1 *corr1
    
    
    # Te<Tc FIX (DSP eq 5: rhoc*gmoho*(Tc if Tc < Te else 0)): the crustal-
    # column term must vanish wherever the crust-mantle interface lies BELOW
    # the elastic shell. Implemented as the indicator field
    #     Tcind(theta,phi) = Tc * 1[Te > Tc],
    # which reduces to the constant Tc (or 0) monopole at constant Te, and
    # handles variable-Te maps that locally dip below Tc. NOTE: for maps
    # crossing Tc the indicator is discontinuous -> spectral ringing near
    # the Te = Tc contour is inherent; consider a smooth Te map or a
    # tapered indicator if that contour matters.
    factorLHS_omega1b = 2*Re**3*rho_c*g_m * Kalousova_scaler1 *corr1  # Tc in field
    factorLHS_omega1c = 2*Re**3*rho_m * Kalousova_scaler1 *corr1      # gTe in field

    # =====================================================================
    # GROUP-2 factors (1d/1e/1f) -- these are now the LIVE definitions,
    # handed to solve_beuthe in the g2 dict below. There is no second copy.
    # They are FACTORS ONLY: group 2 is the ordered operator
    #     Te^2 . Delta( X . w ),
    # so the factor and the inner field X must stay separate -- a single
    # pre-multiplied vector (Te^2 * X) cannot express it, because Te^2 sits
    # OUTSIDE the Laplacian and X INSIDE. Hence the dict, not *_unstr.
    #   1d: X = Te        
    #   1e: X = Tcind (Tc dropped from the factor)
    #   1f: X = gTe*max   (gTe dropped from the factor)
    # =====================================================================
    factorLHS_omega1d = Re/12*drhol*g0*nu/(1-nu) * Kalousova_scaler1 *corr1   # *Laplacian!
    factorLHS_omega1e = -Re/12*g_m*rho_c * Kalousova_scaler1  *corr1          # *Laplacian! (Tc -> Tcind field)
    factorLHS_omega1f = -Re/12*rho_m * Kalousova_scaler1  *corr1              # *Laplacian! (gTe -> gTemax field)

    # drhol EXTENSION -- reinstated dc w-coupling (zero iff rho_l == rho_c);
    # pairs with 1b/1c (c1-half, +2Re^3) and 1e/1f (c2-half, -Re/12).
    # =====================================================================
    # DOUBLE-COUNTING FIX (rho_l != rho_c, BOTH branches).
    # The reinstated dc w-coupling is
    #     + v1v*g_m*drhol*max(Te-Tc,0) * phi^-(l'+2) * w
    # with the weight phi^-(l'+2) on the OPERAND degree l'. That operand
    # weight can only be applied as a diagonal to the RIGHT of the max-field
    # convolution, i.e. inside build_A_tilde_group2 (fdc1_w/fdc2_w + Pw).
    # It therefore must NOT also appear in the group-1 field_ac sum, which is
    # a plain (unweighted) convolution.
    # Previously ONE variable, factorLHS_omega1_dc1, fed BOTH:
    #   * Omega_LHS_dc1_unstr -> field_ac   (no Pw -> wrong operator), and
    #   * g2['fdc1_w']        -> group 2    (with Pw -> correct),
    # so the term was counted twice, once with the wrong weight. It cancelled
    # out at rho_l == rho_c only because drhol = 0 there -- which is exactly
    # why every rho_l == rho_c benchmark passed while rho_l != rho_c failed
    # in BOTH branches. The names are now separated.
    # =====================================================================
    factor_fdc1_w =  2*Re**3 * nu/(1-nu) * g_m * drhol * Kalousova_scaler1 * corr1
    factor_fdc2_w = -Re/12   * nu/(1-nu) * g_m * drhol * Kalousova_scaler1 * corr1

    # Wrap gTe in the factor fields for 1c and 1f
    gTeTeTc_clm = pysh.SHGrid.from_array(gTe_grid * TeTc_grid).expand()
    gTeTeTc_clm = pysh.SHCoeffs.from_array(gTeTeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    # REMOVED: gTeTe2TeTc_clm = gTe*Te^2*max  was a PRE-MULTIPLIED single
    # field for term 1f. That cannot express group 2: the operator is
    # Te^2 . Delta( gTe*max . w ), i.e. Te^2 OUTSIDE the Laplacian and
    # gTe*max INSIDE. Convolving with one merged field (Te^2*gTe*max) is a
    # different operator. 1f is now built in build_A_tilde_group2 from the
    # separate Te2_unstr (outer) and gTemax_unstr (inner) vectors in g2.


    # ------- drho_lm VARIABLES AND FIELDS -------
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']  
    g_M = _dl['g_M']
    B_1   = _dl['B_1']
    B_2 = _dl['B_2']
    Cp    = _dl['Cp']
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)
 
    
    # =====================================================================
    # drho_lm LHS w-couplings  (solve_for == 'drho_lm'; dc = 0)
    # ---------------------------------------------------------------------
    # Eliminating drhom from eq (1) with dc = 0:
    #   drhom_l = [ G_l - Cbar_l*( rho_l*H_l + (drhol + drho*phi^(l+2))*w_l ) ]/B_1_l
    # so drhom's w-coefficient is the PER-DEGREE scalar
    #   Dw_l = -Cbar_l*( drhol + drho*phi^(l+2) ) / B_1_l .
    #
    # STRUCTURAL FIX: the previous version built `ones([2,2L+1,2L+1])` arrays
    # and scaled them per degree, then fed them in as convolution FIELDS.
    # A per-degree scalar weight is NOT a field: convolving with a "field"
    # whose coefficients happen to equal f(l) is a completely different
    # operator from multiplying w's coefficients by f(l). Per-degree weights
    # must be DIAGONAL matrices acting on the OPERAND degree. Only P_hat
    # (Te-dependent) is a genuine spatial field and hence a convolution.
    # Likewise 1/B_1 must NOT be folded into P_hat's own coefficients (that
    # applies B_1 at the FIELD degree, not the operand degree) -- it belongs
    # in Dw_l.
    #
    # Two contributions, both gated on solve_for in solve_beuthe:
    #  (a) q's w-coupling  -> pure per-degree diagonal, no Gaunt:
    #        Lam_q_drho_l = g0*drhol + g_m*drho
    #                       - g0*drho*Cbar_l*drhol*phi^l
    #                       - g0*drho^2*Cbar_l*phi
    #                       + ( g_M*M - g0*drho*B_2_l ) * Dw_l
    #      entering A as  +Re^4*K1*Lam_q_drho_l  on the diagonal.
    #  (b) omega's drhom term  P_hat*drhom/R  -> conv(P_hat) @ diag(Dw):
    #        c1-half:  2Re^3*corr1*K1 * C_Phat @ diag(Dw)
    #        c2-half: -Re/12*corr1*K1 * C_Te2 @ Lap @ C_Phat @ diag(Dw)
    #
    # Also: with dc = 0 the omega term  v1v*drho*g_m*max*(dc-w)/R  loses its
    # +dc half, so its -w half now STANDS ALONE (in the dc branch the two
    # cancel down to the small drhol coupling). Its factors are exactly the
    # historical commented-out dc1/dc2 values -- correct here, wrong there:
    #   fdrho_w1 = -2Re^3*v1v*g_m*drho*K1*corr1   (field: max(Te-Tc,0))
    #   fdrho_w2 = +Re/12*v1v*g_m*drho*K1*corr1   (field: max, c2-half)
    # =====================================================================
    RTcR_l2 = np.array([RTcR**(l+2) for l in range(2*lmax+1)])
    RTcR_l  = np.array([RTcR**l     for l in range(2*lmax+1)])
    Dw_arr = -Cp * (drhol + drho*RTcR_l2) / B_1

    Lam_q_drho_arr = ( g0*drhol 
                       + g_m*drho
                       - g0*drho*Cp*drhol*RTcR_l
                       - g0*drho**2*Cp*RTcR
                       + (g_M*M - g0*drho*B_2) * Dw_arr )

    # STEP 1: dc-branch q w-coupling, precomputed here (was inline in
    # solve_beuthe). Lam_q_dc = q_H with rho_l -> drhol; zero iff rho_l==rho_c.
    RTcR_nl1     = np.array([RTcR**(-(l+1)) for l in range(2*lmax+1)])
    RTcR_negl2   = np.array([RTcR**(-(l+2)) for l in range(2*lmax+1)])
    Lam_q_dc_arr = ( g0*drhol 
                     - g_m*drhol*RTcR_negl2
                     - g0*drho*drhol*Cp*(RTcR_l - RTcR_nl1) )

    # P_hat as a pure GRID field (no 1/B_1 baked in -- that lives in Dw_arr)
    Phat_g   = TeMt0 * MTeMt
    Phat_clm = pysh.SHGrid.from_array(Phat_g).expand()
    Phat_clm = pysh.SHCoeffs.from_array(Phat_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    fdrho_w1 = -2*Re**3 * nu/(1-nu) * g_m * drho * Kalousova_scaler1 * corr1
    fdrho_w2 =  Re/12   * nu/(1-nu) * g_m * drho * Kalousova_scaler1 * corr1
    fdrho_om1 = 2*Re**3 * -0.5 * nu/(1-nu) * g_M * Kalousova_scaler1 * corr1
    fdrho_om2 = -Re/12  * -0.5 * nu/(1-nu) * g_M * Kalousova_scaler1 * corr1

    # ---- group-1 LHS vectors (restored: these sat inside the replaced span)
    Omega_LHS_1a_unstr = factorLHS_omega1a * pysh.shio.SHCilmToVector(Te_clm.coeffs)
    Omega_LHS_1b_unstr = factorLHS_omega1b * pysh.shio.SHCilmToVector(Tcind_clm_1.coeffs)
    Omega_LHS_1c_unstr = factorLHS_omega1c * pysh.shio.SHCilmToVector(gTeTeTc_clm.coeffs)

    # ---- group-2 ingredients: FACTORS + separate inner field vectors ------
    gTemax_clm = pysh.SHGrid.from_array(gTe_grid * TeTc_grid).expand()
    gTemax_clm = pysh.SHCoeffs.from_array(gTemax_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    g2 = dict(
        f1d          = factorLHS_omega1d,
        f1e          = factorLHS_omega1e,
        f1f          = factorLHS_omega1f,
        fdc1_w       = factor_fdc1_w,
        fdc2_w       = factor_fdc2_w,
        Te_unstr     = pysh.shio.SHCilmToVector(Te_clm.coeffs),      # X = Te   (1d)
        Te2_unstr    = pysh.shio.SHCilmToVector(Te2_clm.coeffs),     # outer Te^2
        max_unstr    = pysh.shio.SHCilmToVector(TeTc_clm.coeffs),    # X = max  (dc pair)
        Tcind_unstr  = pysh.shio.SHCilmToVector(Tcind_clm_1.coeffs), # X = Tcind (1e)
        gTemax_unstr = pysh.shio.SHCilmToVector(gTemax_clm.coeffs),  # X = gTe*max (1f)
        # shared scalars/grids -- single source of truth for solve_beuthe
        g_m          = g_m,
        rhobar       = rhobar,
        RTcR         = RTcR,
        TeTc_grid    = TeTc_grid,
        # ---- drho_lm branch ingredients (unused when solve_for=='dc_lm') --
        Dw           = Dw_arr,             # per-degree: drhom's w-coefficient
        Lam_q_drho   = Lam_q_drho_arr,     # per-degree: q's w-coupling (diagonal)
        Lam_q_dc     = Lam_q_dc_arr,       # per-degree: q's w-coupling (dc branch)
        # eq-2 dc coupling factor (scalar); field/weights assembled in solver
        fdc_2d       = nu * g_m * drhol * (Re**2/R),
        Phat_unstr   = pysh.shio.SHCilmToVector(Phat_clm.coeffs),  # genuine field
        fdrho_w1     = fdrho_w1,           # -w half of (dc-w), c1 (field: max)
        fdrho_w2     = fdrho_w2,           # -w half of (dc-w), c2 (field: max)
        fdrho_om1    = fdrho_om1,          # omega/drhom c1 scale
        fdrho_om2    = fdrho_om2,          # omega/drhom c2 scale
    )

    return (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr,
            g2)

 
    
def Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_clm, lmax, R, T_e_0, Re, g0, mass):
    """
    Full set of terms for the Omega parameters of the second equation in the
    system of two equations.
    
    At current stage, the SH function products are done by expansion
    into the spatial domain, performing the multiplication there and then
    transforming back to spatial domain. It should be possible to perform this
    product using the Gaunt coefficients too, which may be implemented at a 
    next stage.
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    

    # Laplacian array for degrees (incl +2 term)
    lap2_by_degree = np.array([-l * (l + 1) +2 for l in range(2 * lmax + 1)])
    
    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ 3/ (rhobar*(2*l+1)) for l in range(2 * lmax + 1)])

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
    
    
    # Field RHS 2a: lap2 * Te*H*alpha grid
    TeHa_grid = T_e_parent_grid * topo_grid * alpha_grid
    TeHa_clm = pysh.SHGrid.from_array(TeHa_grid).expand()
    TeHa_clm = pysh.SHCoeffs.from_array(TeHa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    # Perform multiplication with laplacian2, by multiplying it with 
    # the TeHa coefficients for the degrees l only
    TeHa_lap = TeHa_clm.copy()
    for l in range(TeHa_lap.coeffs.shape[1]):
        TeHa_lap.coeffs[:, l, :] *= lap2_by_degree[l]


    # (same H', G'; here each product also carries alpha and the Laplacian is +2)
    Hp = pysh.SHGrid.from_array(topo_grid).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]): 
        Hp.coeffs[:, l, :] *= 1.0/RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=3*lmax).data
    
    Gp = pysh.SHGrid.from_array(geoid_grid).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]): 
        Gp.coeffs[:, l, :] *= 1/( rhobar2l1[l] * RTcR_l2[l] )
    Gp_grid = Gp.expand(lmax=3*lmax).data
     
    d_dc1 = pysh.SHGrid.from_array(TeTc_grid * Hp_grid * alpha_grid).expand()   # max*H'*alpha
    d_dc1 = pysh.SHCoeffs.from_array(d_dc1.coeffs[:, :2*lmax+1, :2*lmax+1])
    d_dc2 = pysh.SHGrid.from_array(TeTc_grid * Gp_grid * alpha_grid).expand()   # max*G'*alpha
    d_dc2 = pysh.SHCoeffs.from_array(d_dc2.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(d_dc1.coeffs.shape[1]):       # Laplacian+2 on the inner product (as in 2a)
        d_dc1.coeffs[:, l, :] *= lap2_by_degree[l]
        d_dc2.coeffs[:, l, :] *= lap2_by_degree[l]
    
    
    
    
    # drho_lm terms
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']  
    g_M = _dl['g_M']
    B_1   = _dl['B_1']
    Cp    = _dl['Cp']  
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2*lmax+1)])
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)
 
    
    P_frac = -0.5 * nu/(1-nu) * g_M * MTeMt * TeMt0
    P_frac = pysh.SHGrid.from_array(P_frac).expand()
    P_frac = pysh.SHCoeffs.from_array(P_frac.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(P_frac.coeffs.shape[1]):
        P_frac.coeffs[:, l, :] *= 1/B_1[l]
    
    # field drho1 term:   lapl+2 * a_lm*P_frac*Cp*H_lm
    # =====================================================================
    # drho_lm RHS fields for eq 2 (solve_for == 'drho_lm')
    # ---------------------------------------------------------------------
    # eq-2 omega operator:  (1-nu)*scaler2 * Delta' ( alpha * [content] ),
    # content = P_hat * drhom, with the SAME per-degree drhom weights as eq 1
    #   drhom_H,lm = -Cbar_l*rho_l*H_lm/B_1_l ,  drhom_G,lm = G_lm/B_1_l .
    # Same three structural fixes as eq 1: the P_hat product is SPATIAL (not
    # coefficient-wise), 1/B_1 acts at the H/G degree (not P_hat's), and the
    # output Delta' acts on alpha*(P_hat*drhom) -- i.e. AFTER both products,
    # not on the weighted H/G beforehand.
    # =====================================================================
    Phat_g2 =  MTeMt * TeMt0

    def _drhom_grid2(src_clm, wts):
        c = pysh.SHCoeffs.from_array(np.array(src_clm.coeffs[:, :2*lmax+1, :2*lmax+1]))
        for l in range(c.coeffs.shape[1]):
            c.coeffs[:, l, :] *= wts[l]
        return c.expand(lmax=3*lmax).data

    def _eq2_field(prod_grid):
        c = pysh.SHGrid.from_array(alpha_grid * prod_grid).expand()
        c = pysh.SHCoeffs.from_array(c.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(c.coeffs.shape[1]):
            c.coeffs[:, l, :] *= lap2_by_degree[l]
        return c

    wH_d2 = np.array([-Cp[l]  / B_1[l] for l in range(2*lmax+1)])
    wG_d2 = np.array([ 1.0 / B_1[l]           for l in range(2*lmax+1)])
    _topo_c2  = pysh.SHGrid.from_array(topo_grid).expand()
    _geoid_c2 = pysh.SHGrid.from_array(geoid_grid).expand()
    d_drho1 = _eq2_field(Phat_g2 * _drhom_grid2(_topo_c2,  wH_d2))
    d_drho2 = _eq2_field(Phat_g2 * _drhom_grid2(_geoid_c2, wG_d2))



    # ------ PREFACTORS OF THE EQ2 RHS OMEGA TERMS ------
    Kalousova_scaler2 = Re**2/R
    factor2a_omega = -1.0* nu * rho_l * g0 * Kalousova_scaler2  # *(Laplacian+2)
    
    factorRHS_omega2_dc1 = -nu*g_m*rho_l * Kalousova_scaler2
    factorRHS_omega2_dc2 = nu*g_m * Kalousova_scaler2

    factorRHS_omega2_drho1 = (0.5*nu*g_M*rho_l) * Kalousova_scaler2   # * Delta'(alpha*P_hat*drhom_H)
    factorRHS_omega2_drho2 = (0.5*nu*g_M) * Kalousova_scaler2   # * Delta'(alpha*P_hat*drhom_G)
    


    Omega_RHS2_coeffs = ( 
                           factor2a_omega       * TeHa_lap.coeffs[:, :lmax+1, :lmax+1]
                        + (factorRHS_omega2_dc1 * d_dc1.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega2_dc2 * d_dc2.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega2_drho1 * d_drho1.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        + (factorRHS_omega2_drho2 * d_drho2.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        )
    
    
    # Then transform to an 'unstructured' vector (structure same as that of y in
    # solve_beuthe) 
    Omega_RHS2_unstr = pysh.shio.SHCilmToVector(Omega_RHS2_coeffs)
    
    return Omega_RHS2_unstr 
    
    

def Omega_eq2_LHS(T_e_parent, a_clm, lmax, R, T_e_0, Re, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the second equation.
    
    A number of Te and alpha products occur in the LHS terms. These can be
    simplified, since alpha = 1/(E*Te), thus reducing to 1/E for the product.
    """
        
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    

    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid = T_e_parent.expand(lmax=3*lmax).data
    a_grid = a_clm.expand(lmax=3*lmax).data
    # gTe FIELD (variable-Te fix): gravity at the LOCAL shell-base depth,
    # mantle branch only -- every gTe-carrying term also carries max(Te-Tc,0),
    # which is zero exactly where the density branch would switch. Monopole at
    # constant Te => benchmark preserved.
    RTeR_grid = (R - T_e_parent_grid) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid <= T_c, rho_c, rho_m)
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2
    
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
    Tcind_grid_2 = np.where(T_e_parent_grid > T_c, T_c, 0.0)
    Tcinda_clm   = pysh.SHGrid.from_array(Tcind_grid_2 * a_grid.data).expand()
    Tcinda_clm   = pysh.SHCoeffs.from_array(Tcinda_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 2c: max(Te-Tc,0) * alpha
    gTeTeTca_grid = gTe_grid * TeTc_grid * a_grid  # gTe grid folded into here for variable Te
    gTeTeTca_clm  = pysh.SHGrid.from_array(gTeTeTca_grid).expand()
    gTeTeTca_clm  = pysh.SHCoeffs.from_array(gTeTeTca_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    

    # ------- drho_lm VARIABLES AND FIELDS -------
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']
    g_M = _dl['g_M']
    
    # Te - Mt field
    TeMt_grid  = T_e_parent_grid - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)

    # 2d's field is max*alpha (NOT gTe*max*alpha -- gTe belongs to 2c only)
    TeTca_grid = pysh.SHGrid.from_array(TeTc_grid * a_grid).expand()
    TeTca_clm = pysh.SHCoeffs.from_array(TeTca_grid.coeffs[:, :2*lmax+1, :2*lmax+1])

    # drho branch: omega's drhom term  P_hat*drhom/R  contributes the
    # w-coupling  P_hat * Dw  (Dw per-degree, P_hat a field) -> supplied to
    # solve_beuthe as a separate (field, diagonal) pair, since the operand
    # weight cannot be folded into a single convolution field.
    Phata_clm = pysh.SHGrid.from_array( MTeMt * TeMt0 * a_grid ).expand()
    Phata_clm = pysh.SHCoeffs.from_array(Phata_clm.coeffs[:, :2*lmax+1, :2*lmax+1])


    # ------ PREFACTORS OF THE EQ2 LHS OMEGA TERMS ------
    Kalousova_scaler2 = Re**2/R
    factorLHS_omega2a = -drhol*g0*nu * Kalousova_scaler2         # *Laplacian+2!
    factorLHS_omega2b = (1-nu)*rho_c*g_m * Kalousova_scaler2 # *Laplacian+2! (Tc in field)
    
    # 2c changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    factorLHS_omega2c = (1-nu)*rho_m * Kalousova_scaler2 # *Laplacian+2! # gTe in field 
    
    # the fourth prefactor term for crustal thickness variations
    # 2d -- omega's  v1v*drho*g_m*max*(dc - w)/R  term, BRANCH-DEPENDENT:
    #  dc branch  : the (dc-w) combination must be substituted as a WHOLE.
    #               drho*dc carries a hidden +drho*w that cancels the -drho*w
    #               exactly, leaving only +v1v*g_m*drhol*max*phi^-(l+2)*w
    #               (handled in solve_beuthe). Hence 0 here.
    #  drho branch: dc = 0, so nothing cancels and the -w half STANDS ALONE:
    #               -v1v*drho*g_m*max*w  ->  factor -nu*drho*g_m*scaler2
    #               (field max*alpha). This is exactly the historical
    #               commented-out value: right here, wrong in the dc branch.
    factorLHS_omega2d = (0.0 if solve_for == 'dc_lm'
                            else -nu*drho*g_m*Kalousova_scaler2)
    
    factorLHS_omega2e = (1-nu) *-0.5 * nu/(1-nu) * g_M * Kalousova_scaler2

    # Transform into SHtools vectorformat again
    Omega_LHS_2a_unstr      = factorLHS_omega2a * pysh.shio.SHCilmToVector(Tea_clm.coeffs)
    Omega_LHS_2b_unstr      = factorLHS_omega2b * pysh.shio.SHCilmToVector(Tcinda_clm.coeffs)
    Omega_LHS_2c_unstr      = factorLHS_omega2c * pysh.shio.SHCilmToVector(gTeTeTca_clm.coeffs)
    Omega_LHS_2d_unstr      = factorLHS_omega2d * pysh.shio.SHCilmToVector(TeTca_clm.coeffs)
    Omega_LHS_2_Phata_unstr = factorLHS_omega2e * pysh.shio.SHCilmToVector(Phata_clm.coeffs)
    # raw (unfactored) max*alpha vector: the eq-2 dc coupling in solve_beuthe
    # convolves this then applies its own output/operand weights, so it needs
    # the bare field, not the 2d-factored one.
    maxa_raw_unstr          = pysh.shio.SHCilmToVector(TeTca_clm.coeffs)
    


    # NOTE: the old Omega_LHS_2e_drho_unstr is gone. It multiplied alpha,
    # P_frac and the per-degree weights COEFFICIENT-WISE into one vector;
    # (P*a)_lm != P_lm*a_lm, and the per-degree Dw weight is a DIAGONAL on the
    # operand, not a field. It is replaced by the (field, diagonal) pair
    # Omega_LHS_2_Phata_unstr + g2['Dw'], combined in solve_beuthe.
    return (Omega_LHS_2_Phata_unstr,
            Omega_LHS_2a_unstr, 
            Omega_LHS_2b_unstr, 
            Omega_LHS_2c_unstr,
            Omega_LHS_2d_unstr,
            maxa_raw_unstr)



def q_lm(topo_clm, geoid_clm, lmax, R, T_e_0, Re, g0, mass):
    """
    Compute the loading terms q_lm of the first equation. The equation is
    
    q_lm = g0 * rho_l * (H_lm - G_lm) + g_m * drho * (w_lm - dc_lm - Gc_lm)

    where dc_lm and Gc_lm are rewritten in terms of H_lm, G_lm and w_lm, and 
    the w_lm terms are moved from this equation to the Eq1 LHS to be included
    in the Gaunt computations.
    """

    topo_clm_copyq = topo_clm.copy()
    geoid_clm_copyq = geoid_clm.copy()
    topo_clm_copyq.coeffs[0,0,0] = 0
    geoid_clm_copyq.coeffs[0,0,0] = 0
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2   
    
    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(l+1) for degrees l
    RTcR_l1 = np.array([((R-T_c)/R)**(l) for l in range(2 * lmax + 1)])
    # --->> CHANGED FROM (l+1) TO (l) TO ALIGN WITH DSP

    # (R-Tc)/R^(-l+1) for degrees l
    RTcR_negl1 = np.array([((R-T_c)/R)**(-l-1) for l in range(2 * lmax + 1)])
    # --->> CHANGED FROM (-l+1) TO (-l-1) TO ALIGN WITH DSP

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ (rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])

    # Perform the multiplications with degree-dependent terms
    field_topo_dc1  = topo_clm_copyq.coeffs.copy()                
    field_topo_dc2  = topo_clm_copyq.coeffs.copy()   
    field_topo_dc3  = topo_clm_copyq.coeffs.copy() 
    field_geoid_dc4 = geoid_clm_copyq.coeffs.copy() 
    field_geoid_dc5 = geoid_clm_copyq.coeffs.copy()              
    
    for l in range(field_topo_dc1.shape[1]):
        field_topo_dc1[:, l, :] *= (1/RTcR_l2[l])
        field_topo_dc2[:, l, :] *= (RTcR_l1[l] / rhobar2l1[l])
        field_topo_dc3[:, l, :] *= (RTcR_negl1[l] / rhobar2l1[l])
        field_geoid_dc4[:, l, :] *= (rhobar2l1[l] / RTcR_l2[l])
        field_geoid_dc5[:, l, :] *= RTcR_negl1[l]
    
    
    
    # drho_lm terms
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M'] 
    g_M = _dl['g_M']
    B_1   = _dl['B_1']
    B_2 = _dl['B_2']
    Cp    = _dl['Cp']
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2*lmax+1)])
     

    field_topo_drho1  = topo_clm_copyq.coeffs.copy()
    field_topo_drho2  = topo_clm_copyq.coeffs.copy()
    field_topo_drho3  = topo_clm_copyq.coeffs.copy()

    field_geoid_drho4 = geoid_clm_copyq.coeffs.copy()
    field_geoid_drho5 = geoid_clm_copyq.coeffs.copy()
    
    for l in range(field_topo_drho1.shape[1]):
        field_topo_drho1[:, l, :] *= Cp[l] * RTcR_l1[l]
        field_topo_drho2[:, l, :] *= Cp[l] * B_2[l] / B_1[l]
        field_topo_drho3[:, l, :] *= Cp[l] / B_1[l]

        field_geoid_drho4[:, l, :] *= 1 / B_1[l]
        field_geoid_drho5[:, l, :] *= B_2[l] / B_1[l]



    # Loading terms per each SH field multiplication later & Kalousova scaler
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    q_topo_term1  = g0*rho_l            # No SH field
    q_geoid_term1 = -g0*rho_l           # No SH field
    
    q_topo_dc1    = -g_m*rho_l          # / RTcRl2
    q_topo_dc2    = -g0*drho*rho_l      # * RTcRl1/rhobar2l1
    q_topo_dc3    = g0*drho*rho_l       # * RTcR_nl1/rhobar2l1
    q_geoid_dc4   = g_m                 # *rhobar2l1 / RTcRl2
    q_geoid_dc5   = -g0*drho            # * RTcR_nl1

    q_topo_drho1  = -g0 * drho * rho_l  # * Cp * RTcR_l1 * topo
    q_topo_drho2  =  g0 * drho * rho_l  # * Cp * B_2/B_1 * topo
    q_topo_drho3  = -g_M * M * rho_l    # * Cp / B_1 * topo
    q_geoid_drho4 =  g_M * M            # * 1 / B_1 * geoid
    q_geoid_drho5 = -g0 * drho          # * B_2 / B_1 * geoid
    
    
    
    
    
    
    # Make coeffs array of size lmax+1 for the RHS
    q_coeffs = -Re**4 * Kalousova_scaler1 * (
                  q_topo_term1 
                   * topo_clm_copyq.coeffs[:, :lmax+1, :lmax+1]
                + q_geoid_term1
                   * geoid_clm_copyq.coeffs[:, :lmax+1, :lmax+1] 
                
                + (q_topo_dc1 
                   * field_topo_dc1[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_topo_dc2
                   * field_topo_dc2[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_topo_dc3 
                   * field_topo_dc3[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_geoid_dc4
                   * field_geoid_dc4[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_geoid_dc5
                   * field_geoid_dc5[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                
                + (q_topo_drho1
                   * field_topo_drho1[:, :lmax+1, :lmax+1]  if solve_for == 'drho_lm' else 0)
                + (q_topo_drho2
                   * field_topo_drho2[:, :lmax+1, :lmax+1]  if solve_for == 'drho_lm' else 0)
                + (q_topo_drho3
                   * field_topo_drho3[:, :lmax+1, :lmax+1]  if solve_for == 'drho_lm' else 0)
                + (q_geoid_drho4
                   * field_geoid_drho4[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                + (q_geoid_drho5
                   * field_geoid_drho5[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
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
                         gidx, gaunt_bare, starts, seg_len, ci, cj, mode_map, N,
                         fdc1_w=0.0, fdc2_w=0.0, Pw=None, Tcind_unstr=None, gTemax_unstr=None):
    """ Correct  Te^2 * Delta'( X * w )  operators, returned as a (generally
        NON-symmetric) dense N x N matrix to ADD into A_tilde group-2.
 
        drhol EXTENSION (rho_l != rho_c): the dc-elimination of Banerdt
        eq (1) leaves a residual w-coupling in omega,
            + v1v * g_m * drhol * max(Te-Tc,0) * phi^(-(l'+2)) * w / R
        (l' = OPERAND degree), reinstating the previously-zeroed LHS dc
        terms with the correct density (drhol, not drho), sign, and the
        operand-degree weight Pw = diag(phi^(-(l+2))). Both halves of the
        [c1 + c2*Delta] omega bracket receive it:
            fdc1_w * (C_max @ Pw)              (c1-half, pairs with 1b/1c)
            fdc2_w * (Te^2 Delta (max Pw w))   (c2-half, pairs with 1e/1f)
        These vanish identically for drhol = 0."""
    C_Te  = build_conv_matrix(Te_unstr,  gidx, gaunt_bare, starts, seg_len, ci, cj, N)
    C_Te2 = build_conv_matrix(Te2_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
    C_max = build_conv_matrix(max_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
 
    dl   = np.array([-l*(l+1) for l, _ in mode_map])      # Delta (no +2 in eq.1)
    Dlap = np.diag(dl)
 
    Te2_Lap = C_Te2 @ Dlap                                # Te^2 . Delta  (reused)
    M_1d = f1d * (Te2_Lap @ C_Te)                         # Te^2 Delta (Te  w)
    # Te<Tc FIX: the crust-column 1e term carries the indicator field
    # Tcind = Tc*1[Te>Tc] (Tc no longer folded into f1e), mirroring DSP's
    # (Tc if Tc < Te else 0) branch and valid for variable Te dipping below Tc.
    if Tcind_unstr is not None:
        C_Tci = build_conv_matrix(Tcind_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
        M_1e = f1e * (Te2_Lap @ C_Tci)                    # Te^2 Delta (Tcind w)
    else:
        M_1e = f1e * (Te2_Lap)                            # legacy: Tc inside f1e
    if gTemax_unstr is not None:                          # gTe FIELD in 1f
        C_gmax = build_conv_matrix(gTemax_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
        M_1f = f1f * (Te2_Lap @ C_gmax)                   # Te^2 Delta (gTe max w)
    else:
        M_1f = f1f * (Te2_Lap @ C_max)                    # legacy: gTe scalar in f1f
    M = M_1d + M_1e + M_1f                                # NOT symmetric -- keep full
    if (fdc1_w != 0.0 or fdc2_w != 0.0) and Pw is not None:
        CmaxP = C_max @ Pw
        M = M + fdc1_w * CmaxP + fdc2_w * (Te2_Lap @ CmaxP)
    return M


# %% BEUTHE MODEL SOLVER

def solve_beuthe(topo_clm, geoid_clm, T_e_parent, D_clm, a_clm, plan, lmax, R,
                 T_e_0, g0, mass,
                 D_eta_clm=None, a_eta_clm=None, eta_clm=None):
    """
    Solves the Beuthe system of equations w and F with the thin-shell 
    approximation factor eta.
    """
    mode_map = make_mode_map(lmax)
    N = len(mode_map)
    Re   = R - T_e_0/2
    scaler_A = 1.0/(E*T_e_0**3)
    scaler_B = Re
 
    Dlm = pysh.shio.SHCilmToVector(D_eta_clm.coeffs)
    alm = pysh.shio.SHCilmToVector(a_eta_clm.coeffs)
 
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
    (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr, g2) = (
                Omega_eq1_LHS(T_e_parent, lmax=lmax, 
                              R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
    # group 1: terms 1a,1c, dc1 -- NO output Laplacian
    field_ac = (Omega_LHS_1a_unstr[gidx] + Omega_LHS_1b_unstr[gidx]
                + Omega_LHS_1c_unstr[gidx]) * plan['term_gaunt_bare']
    cell_ac = np.add.reduceat(field_ac, starts)
    cell_ac[seg_len == 0] = 0.0
    
    
    cellA_tilde = cell_ac 
    cellA_tilde[seg_len == 0] = 0.0
 
 
    # Calculate the Omega LHS terms for equation 2
    (Omega_LHS_2_Phata_unstr,
     Omega_LHS_2a_unstr, Omega_LHS_2b_unstr, Omega_LHS_2c_unstr,
     Omega_LHS_2d_dc_unstr, maxa_raw_unstr) = (
        Omega_eq2_LHS(T_e_parent, a_eta_clm, lmax=lmax, 
                      R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
    # terms 2a, 2b, 2c (+2d, zero) -- carry Delta' at the OUTPUT degree.
    # b_tilde FIX (non-symmetric pathway): previously assembled per cell with
    # lap_out on the ci side and scattered SYMMETRICALLY -- exact at constant
    # Te (monopole omega-fields leave only diagonal cells) but WRONG for
    # laterally varying Te: the (j,i) orientation must carry Delta'(l_j),
    # not Delta'(l_i). Rebuilt in the omega_on block as
    #     b_tilde = diag(Delta'_out) @ C_conv(2a+2b+2c fields),
    # carrying the output weight on the correct side for both orientations.
    # NOTE: 2d must NOT be gated here -- it is exactly the other way round.
    # factorLHS_omega2d_dc is 0 in the dc branch (the (dc-w) cancellation) and
    # NONZERO in the drho branch (dc = 0, so the -w half stands alone). The
    # factor already carries the branch; gating on 'dc_lm' zeroed the only
    # branch where the term is alive.
    fields_2sum = (Omega_LHS_2a_unstr + Omega_LHS_2b_unstr + Omega_LHS_2c_unstr
                   + Omega_LHS_2d_dc_unstr)
 
 
 
    # ---- scatter per-cell values into dense blocks (loop over CELLS) ------
    # (b_tilde no longer scattered here: built as a non-symmetric matrix
    #  product in the omega_on block -- see b_tilde FIX above.)
    A = np.zeros((N, N))
    A_tilde = np.zeros((N, N))
    B = np.zeros((N, N))
    for c in range(ci.size):
        i, j = int(ci[c]), int(cj[c])
        vA = cellA[c]
        vA_tilde = cellA_tilde[c]   # 1b now a Tcind FIELD inside field_ac (Te<Tc fix)
        vB = cellB[c]
        
        A[i, j] = vA  
        A_tilde[i, j] = vA_tilde 
        B[i, j] = vB
        if i != j:
            A[j, i] = vA 
            A_tilde[j, i] = vA_tilde
            B[j, i] = vB     # operators are symmetric
 
 
    # ---- coupling blocks a_l, b_l --------------------------------------
    d_l2 = np.array([-l*(l+1)+2 for l,_ in mode_map], dtype=np.float64)
 
    # ETA_FULL: Beuthe eqs (58)/(66) couplings  R^3*A(eta; F)  and
    # -(1/R)*A(eta; w)  built from the existing plan. The pure-A weight
    #   W_Aonly = -br/4,
    #   br = (dl-dlp)^2 + 2(dl+dlp) - 8  +  dL^2 + 2dL - 2dL(dl+dlp)
    # splits into per-TERM (dL-dependent) and per-CELL (dl,dlp) pieces,
    # so three reduceat passes over term_gaunt_bare suffice:
    #   g0 = conv(eta, .; 1),  g1 = conv(eta, .; dL),  g2 = conv(eta, .; dL^2)
    #   A(eta;.)[cell] = -1/4 * [ ((dl-dlp)^2 + 2(dl+dlp) - 8)*g0
    #                             + (2 - 2(dl+dlp))*g1 + g2 ]
    # Monopole-eta check: dL=2, dl=dlp  ->  eta0*dl = eta0*Delta'  (the
    # exact constant-Te fix). Same symmetric (i,j)<->(j,i) scatter as
    # cellA (W_Aonly symmetric in dl<->dlp).
    eta_lm  = pysh.shio.SHCilmToVector(eta_clm.coeffs)
    term_dL = (-term_L*(term_L + 1) + 2).astype(np.float64)
    ebare   = eta_lm[gidx] * plan['term_gaunt_bare']
    cell_g0 = np.add.reduceat(ebare,            starts)
    cell_g1 = np.add.reduceat(ebare*term_dL,    starts)
    cell_g2 = np.add.reduceat(ebare*term_dL**2, starts)
    for arr in (cell_g0, cell_g1, cell_g2):
        arr[seg_len == 0] = 0.0
    dl_i = d_l2[ci.astype(np.int64)]
    dl_j = d_l2[cj.astype(np.int64)]
    Ssum = dl_i + dl_j
    W_eta_cell = -0.25*(((dl_i - dl_j)**2 + 2.0*Ssum - 8.0)*cell_g0
                        + (2.0 - 2.0*Ssum)*cell_g1
                        + cell_g2)
 
    a = np.zeros((N, N))
    b = np.zeros((N, N))
    fac_a = (Re/T_e_0)**3 / E
    for c in range(ci.size):
        i, j = int(ci[c]), int(cj[c])
        va =  fac_a * W_eta_cell[c]
        vb = -W_eta_cell[c]
        a[i, j] = va
        b[i, j] = vb
        if i != j:
            a[j, i] = va
            b[j, i] = vb

 
    # ------------------------------------------------------------------
    # Group-2 LHS operator  Te^2 . Delta( X . w ):
    #     1d: X = Te          1e: X = Tcind        1f: X = gTe*max
    #     dc pair: X = max . Pw   (Pw = phi^-(l'+2) on the OPERAND degree)
    # ALL factors and inner field vectors now come from Omega_eq1_LHS's
    # g2 dict -- there is no second copy here. (Two copies of the same
    # physics is what produced the historical f_dc2 inconsistency.)
    # The returned matrix is generally NON-symmetric, as expected for an
    # ordered  conv . Lap . conv  operator; it is added straight into A.
    # NOTE: build_A_tilde_group2 forms dense N x N convolution matrices
    # and matmuls (~O(N^3)); a few seconds at lmax~45.
    # ------------------------------------------------------------------
    # STEP 6: the only per-degree diagonal solve_beuthe still needs is Pw
    # (phi^-(l'+2) on the operand degree). Build it once from g2's RTcR.
    Pw_diag = np.diag(np.array([g2['RTcR']**(-(l+2)) for l, _ in mode_map]))
    _use_dc = (solve_for == 'dc_lm')

    # dc branch: the reinstated drhol dc coupling (zero iff rho_l==rho_c).
    # drho branch: dc = 0, so that pair does not exist -- instead the -w
    # half of omega's (dc-w) term STANDS ALONE (fdrho_w1/w2, field = max,
    # no operand weight -> Pw = I). Its factors are exactly the historical
    # commented-out dc1/dc2 values: correct here, wrong in the dc branch.
    A_tilde_group2 = build_A_tilde_group2(
        g2['Te_unstr'], g2['Te2_unstr'], g2['max_unstr'],
        g2['f1d'], g2['f1e'], g2['f1f'],
        gidx, plan['term_gaunt_bare'], starts, seg_len, ci, cj, mode_map, N,
        fdc1_w=(g2['fdc1_w'] if _use_dc else g2['fdrho_w1']),
        fdc2_w=(g2['fdc2_w'] if _use_dc else g2['fdrho_w2']),
        Pw=(Pw_diag if _use_dc else np.eye(N)),
        Tcind_unstr=g2['Tcind_unstr'], gTemax_unstr=g2['gTemax_unstr'])

    A = A + A_tilde + A_tilde_group2

    # ---- q's w-coupling: LHS diagonal, branch-dependent ---------------
    # dc branch  : Lam_q = qH with rho_l -> drhol   (zero iff rho_l==rho_c)
    # drho branch: Lam_q_drho (built in Omega_eq1_LHS) -- nonzero even at
    #              rho_l == rho_c, since the g_m*drho term survives.
    # Both enter via  -Re^4*K1*q  =>  A_diag += Re^4*K1*Lam_q.
    # STEP 3: both Lam_q arrays are precomputed in Omega_eq1_LHS. solve_beuthe
    # only applies the branch-appropriate one to the diagonal.
    Lam_q_arr = g2['Lam_q_dc'] if _use_dc else g2['Lam_q_drho']
    for idx, (l_m, _) in enumerate(mode_map):
        A[idx, idx] += Re**4 * scaler_A * Lam_q_arr[l_m]

    # ---- drho branch: omega's drhom term  P_hat * drhom / R -----------
    # conv(P_hat) @ diag(Dw): P_hat is a genuine Te-dependent field ->
    # convolution; Dw is a per-degree scalar -> diagonal on the OPERAND.
    if not _use_dc:
        Dw_diag = np.diag(np.array([g2['Dw'][l] for l, _ in mode_map]))
        C_Phat  = build_conv_matrix(g2['Phat_unstr'], gidx,
                                    plan['term_gaunt_bare'],
                                    starts, seg_len, ci, cj, N)
        C_Te2   = build_conv_matrix(g2['Te2_unstr'], gidx,
                                    plan['term_gaunt_bare'],
                                    starts, seg_len, ci, cj, N)
        Lap_d   = np.diag(np.array([-l*(l+1) for l, _ in mode_map],
                                   dtype=np.float64))
        PhatDw  = C_Phat @ Dw_diag
        A = A + g2['fdrho_om1'] * PhatDw
        A = A + g2['fdrho_om2'] * (C_Te2 @ Lap_d @ PhatDw)

    # b_tilde: non-symmetric matrix form (see b_tilde FIX above):
    # diag(Delta'_out) @ conv(2a + 2b + 2c fields). Reduces exactly to
    # the former symmetric-scatter result for monopole omega-fields
    # (constant Te), and is correct for laterally varying Te.
    C_2abc  = build_conv_matrix(fields_2sum, gidx, plan['term_gaunt_bare'],
                                starts, seg_len, ci, cj, N)
    b_tilde = np.diag(d_l2) @ C_2abc
 
    # (The dc-branch Lam_q diagonal is applied ONCE, in the branch-gated
    #  block above. A second, identical copy used to live here -- it made
    #  A carry 2*Lam_q + fdc instead of Lam_q + fdc, i.e. exactly one
    #  extra Lam_q. Invisible at rho_l == rho_c, where Lam_q = 0.)

    # drhol EXTENSION -- eq-2 omega dc coupling (zero iff rho_l == rho_c):
    # the same + v1v*g_m*drhol*max*P_l'*w/R content must enter b_tilde,
    # pattern of 2b/2c: field = max(Te-Tc,0)*(eta*alpha), output Delta',
    # operand weight Pw. Built as an explicit (non-symmetric) matrix
    # product to carry the output/operand weights on the correct sides.
    # STEP 5: eq-2 dc coupling. The max*alpha field and the fdc_2d factor are
    # now supplied by Omega_eq2_LHS / g2 -- solve_beuthe only forms the
    # ordered matrix product  Delta'_out @ conv(max*alpha) @ Pw(operand).
    if _use_dc and g2['fdc_2d'] != 0.0:
        C_maxa = build_conv_matrix(maxa_raw_unstr, gidx, plan['term_gaunt_bare'],
                                   starts, seg_len, ci, cj, N)
        Dlp_out = np.diag(d_l2)                 # Delta' at OUTPUT degree
        b_tilde = b_tilde + g2['fdc_2d'] * (Dlp_out @ C_maxa @ Pw_diag)
 
    # drho branch: omega's drhom w-coupling in eq 2:
    #   (1-nu)*scaler2 * Delta'_out @ conv(P_hat*alpha) @ diag(Dw)
    # P_hat*alpha is a genuine field (convolution); Dw is per-degree
    # (diagonal on the OPERAND). Non-symmetric by construction.
    if not _use_dc:
        Dw_diag_2 = np.diag(np.array([g2['Dw'][l] for l, _ in mode_map]))
        C_Phata   = build_conv_matrix(Omega_LHS_2_Phata_unstr, gidx,
                                      plan['term_gaunt_bare'],
                                      starts, seg_len, ci, cj, N)
        b_tilde = b_tilde + np.diag(d_l2) @ C_Phata @ Dw_diag_2

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
    q_lm_unstr = q_lm(topo_clm, geoid_clm, 
                      lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                      g0=g0, mass=mass)
    Omega_RHS1_unstr = Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, 
                                     lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                     g0=g0, mass=mass)
    Omega_RHS2_unstr = Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_eta_clm, 
                                     lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                     g0=g0, mass=mass)
    
    def elem(l,m,v):
        offset = 0 if m==0 else (m if m>0 else l+abs(m))
        return v[l*l+offset]
    
    q = np.array([elem(l,m,q_lm_unstr) for l,m in mode_map])
    Omega_RHS1 = np.array([elem(l,m,Omega_RHS1_unstr) for l,m in mode_map])
    Omega_RHS2 = np.array([elem(l,m,Omega_RHS2_unstr) for l,m in mode_map])
 
 
    y1 = q + Omega_RHS1
    y2 = Omega_RHS2

    
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
    topo_p, geoid_p, T_e_parent, R, g0, mass = load_inputs(lmax, strain=strain)
    T_e_0 = T_e_parent.coeffs[0,0,0]
    D_clm, a_clm, D_eta_clm, a_eta_clm, eta_clm = derive_D_a(T_e_parent, lmax)

    topo_clm  = pysh.SHCoeffs.from_array(topo_p.coeffs[:, :lmax+1, :lmax+1],
                                         normalization='4pi')
    geoid_clm = pysh.SHCoeffs.from_array(geoid_p.coeffs[:, :lmax+1, :lmax+1],
                                         normalization='4pi')
    plan  = build_or_load_gaunt(lmax, nu)        
            
    T_e_use, D_use, a_use, topo_use, geoid_use = (
        T_e_parent, D_clm, a_clm, topo_clm, geoid_clm)
    D_eta_use, a_eta_use, eta_use = D_eta_clm, a_eta_clm, eta_clm
        
    print('Start solving of system')
    t = time.perf_counter()
    w, F, q = solve_beuthe(topo_use, geoid_use, T_e_use, D_use, a_use, plan, 
                     lmax, R, T_e_0, g0, mass,
                     D_eta_clm=D_eta_use, a_eta_clm=a_eta_use,
                     eta_clm=eta_use)
    print(f'Finished solving of system in {(time.perf_counter()-t):.1f}s\n')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,9))
    w.plot_spectrum(ax=ax1, show=False, 
                    legend=(f'lmax={lmax}'))
    F.plot_spectrum(ax=ax2, show=False, 
                    legend=(f'lmax={lmax}'))
    
    ax1.set_title('M4 - Power spectra of displacement w')
    ax1.legend()
    ax1.set_ylim(1e-2)
    ax2.set_title('M4 - Power spectra of stress function F')
    ax2.legend()
    plt.tight_layout()
    plt.show() 
    plt.close()


    print(f'\nTotal model runtime: {(time.perf_counter() - t_begin):.1f}s')