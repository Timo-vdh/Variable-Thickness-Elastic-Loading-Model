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
    - Crustal root variations

Model 3 does not include:
    - Toroidal loading (V=0)
    - Mantle density variations

Following Beuthe's model requires implementation of the differential operator 
A(a;b). Beuthe (2008) does not give a spectral method for this, but in Beuthe
(2010) this spectral notation is made. Kalousova et al. (2012) describe the 
system of equations 75 and 76 in full spectral notation. This system of
equations is solved in Model 1, and extended here for the inclusion of 
tangential loading potential Omega and crustal root variations dc_lm.

This model includes the thin-shell approximation factor eta that Beuthe and
Kalousova neglect in their final equations (Beuthe does include it in 
equations 58 and 66). 

"""

import numpy as np
import pyshtools as pysh
import os
import time
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm as cmc
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(1, 'C:/Users/Timov/Displacement_strain_planet/Displacement_strain_planet')
from Displacement_strain_planet import Plt_tecto_Mars

# %% INPUTS

nu    = 0.25
E     = 100.0e9
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
cmap2 = cmc.broc
cmap3 = cmc.roma_r
os.makedirs(CACHE_DIR, exist_ok=True)

Te_input = 268.12e3

omega_On = True
strain = 0      # Set which Te map is used, strain-14, strain-17, or
                 # strain-0 (returns constant Te map with Te=average of Te-14)

ETA_FULL = True  # Implement Beuthe's UNSIMPLIFIED flexure equations (58)/(66),
                 # retaining eta = xi/(1+xi) = 1/(1+Te^2/(12*Re^2)) instead of
                 # the thin-shell limit eta->1 of eqs (74)-(78) (= Kalousova
                 # eqs 13-14). Removes the beta = Te^2/(12Re^2+Te^2) benchmark
                 # floor against DSP exactly at constant Te (verified 1e-13),
                 # and via the eta(theta,phi) field at variable Te. Changes:
                 # (i) coefficient fields D -> eta*D, alpha -> eta*alpha in
                 # the A/B operator convolutions; (ii) the off-diagonal
                 # couplings R^3*Delta'F and -(1/R)*Delta'w become the field
                 # operators R^3*A(eta;F) and -(1/R)*A(eta;w) (Beuthe's A
                 # operator, eq 33), built from the SAME Gaunt plan via
                 # term_gaunt_bare with the pure-A weight W_Aonly = -br/4;
                 # (iii) Kalousova_scaler2 loses its eta0 (the eta migrates
                 # into the F-coupling block; c1/c2 omega prefactors keep
                 # corr1 = eta0*Re/R unchanged). Set False for the legacy
                 # eta-truncated system (Kalousova 13-14).


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
    rhobar2l1 = np.array([ rhobar*(2*l+1) for l in range(2 * lmax + 1)])
    
    
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
    

    # ------- THE PREFACTORS OF THE RHS OMEGA TERMS -------
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    factor1a_omega = -2.0*Re**3 *rho_l *g0 *nu/(1.0-nu) * Kalousova_scaler1 * corr1
    factor1b_omega = rho_l * g0 * (Re/12.0) * nu/(1.0-nu) * Kalousova_scaler1 * corr1
    # Term b to be multiplied with Laplacian
    
    factorRHS_omega1_dc1 = 2*Re**3*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc2 = -Re/12*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc3 = -2*Re**3*( nu/(1-nu)*g_m / 3) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc4 = Re/12 * ( nu/(1-nu)*g_m / 3 ) * Kalousova_scaler1 * corr1
    
    
    # ------- ASSEMBLY -------
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
   
def Omega_eq1_LHS(T_e_parent, lmax, R, T_e_0, Re, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the first equation.
    """
        
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


    # Field 1e: Te^2 * laplacian 
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

    
    # Calculate the six prefactor Omega-terms of matrix A
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    
    # 1a & 1d change to take difference in rho_l and rho_c and the surface gravity
    # 1c and 1f changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    # factorLHS_omega1a = -2*Re**3*rho_c*g_m*nu/(1-nu) * Kalousova_scaler1
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
    factorLHS_omega1b = 2*Re**3*rho_c*g_m * Kalousova_scaler1 *corr1   # (Tc now in the field)
    Tcind_grid_1 = np.where(T_e_parent_grid > T_c, T_c, 0.0)
    Tcind_clm_1  = pysh.SHGrid.from_array(Tcind_grid_1).expand()
    Tcind_clm_1  = pysh.SHCoeffs.from_array(Tcind_clm_1.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # factorLHS_omega1c = 2*Re**3*g_m*rho_m * Kalousova_scaler1
    factorLHS_omega1c = 2*Re**3*gTe*rho_m * Kalousova_scaler1 *corr1

    # =====================================================================
    # DEAD CODE WARNING (1d/1e/1f): the group-2 Laplacian-half operators are
    # NOT assembled from these factors -- the LIVE versions are f1d/f1e/f1f
    # inside solve_beuthe (build_A_tilde_group2 block). These copies are kept
    # numerically synchronized (incl. corr1) only to avoid divergence if they
    # are ever re-enabled. Edit the solve_beuthe versions, not these.
    # =====================================================================
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
    Omega_LHS_1b_unstr = factorLHS_omega1b * pysh.shio.SHCilmToVector(Tcind_clm_1.coeffs)
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
    # ETA_FULL: with the eta-retained flexure/coupling blocks, the exact
    # constant-Te omega decomposition requires total eq-2 scale Y = Re^2
    # (the eta migrated into the A(eta;F) coupling)
    Kalousova_scaler2 = Re**2/R
    factor2a_omega = -1.0* nu * rho_l * g0 * Kalousova_scaler2
    # Term to be multiplied with Laplacian+2
    
    factorRHS_omega2_dc1 = nu*g_m*rho_l * Kalousova_scaler2
    factorRHS_omega2_dc2 = -nu*g_m/3 * Kalousova_scaler2
    
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
        Gp.coeffs[:, l, :] *= rhobar2l1[l]/RTcR_l2[l]
    Gp_grid = Gp.expand(lmax=3*lmax).data
     
    d_dc1 = pysh.SHGrid.from_array(TeTc_grid * Hp_grid * alpha_grid).expand()   # max*H'*alpha
    d_dc1 = pysh.SHCoeffs.from_array(d_dc1.coeffs[:, :2*lmax+1, :2*lmax+1])
    d_dc2 = pysh.SHGrid.from_array(TeTc_grid * Gp_grid * alpha_grid).expand()   # max*G'*alpha
    d_dc2 = pysh.SHCoeffs.from_array(d_dc2.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(d_dc1.coeffs.shape[1]):       # Laplacian+2 on the inner product (as in 2a)
        d_dc1.coeffs[:, l, :] *= lap2_by_degree[l]
        d_dc2.coeffs[:, l, :] *= lap2_by_degree[l]

    
    Omega_RHS2_coeffs = ( factor2a_omega       * TeHa_lap.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega2_dc1 * d_dc1.coeffs[:, :lmax+1, :lmax+1]
                        + -factorRHS_omega2_dc2 * d_dc2.coeffs[:, :lmax+1, :lmax+1] )
    
    
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

    
    # Calculate the three prefactor Omega-terms of matrix A
    # ETA_FULL: with the eta-retained flexure/coupling blocks, the exact
    # constant-Te omega decomposition requires total eq-2 scale Y = Re^2
    # (the eta migrated into the A(eta;F) coupling); legacy eta-truncated
    # system requires Y = eta0*Re^2. c1/c2 (corr1) unchanged in both.
    Kalousova_scaler2 = Re**2/R
    # factorLHS_omega2a = -rho_c*g_m*nu * Kalousova_scaler2          # *Laplacian+2!
    # 2a change to take difference in rho_l and rho_c and the surface gravity
    factorLHS_omega2a = -drhol*g0*nu * Kalousova_scaler2         # *Laplacian+2!
    # Te<Tc FIX: Tc -> Tcind*alpha field (DSP: rhoc*gmoho*(Tc if Tc<Te else 0))
    factorLHS_omega2b = (1-nu)*rho_c*g_m * Kalousova_scaler2 # *Laplacian+2! (Tc in field)
    # factorLHS_omega2c = (1-nu)*rho_m*g_m * Kalousova_scaler2 # *Laplacian+2!
    
    # 2c changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    factorLHS_omega2c = (1-nu)*rho_m*gTe * Kalousova_scaler2 # *Laplacian+2! 
    
    # the fourth prefactor term for crustal thickness variations
    # factorLHS_omega2d_dc = -nu*drho*g_m*Kalousova_scaler2
    factorLHS_omega2d_dc = 0

    Tcind_grid_2 = np.where(T_e_parent_grid > T_c, T_c, 0.0)
    Tcinda_clm   = pysh.SHGrid.from_array(Tcind_grid_2 * a_grid.data).expand()
    Tcinda_clm   = pysh.SHCoeffs.from_array(Tcinda_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    # Transform into SHtools vectorformat again
    Omega_LHS_2a_unstr = factorLHS_omega2a * pysh.shio.SHCilmToVector(Tea_clm.coeffs)
    Omega_LHS_2b_unstr = factorLHS_omega2b * pysh.shio.SHCilmToVector(Tcinda_clm.coeffs)
    Omega_LHS_2c_unstr = factorLHS_omega2c * pysh.shio.SHCilmToVector(TeTca_clm.coeffs)
    
    Omega_LHS_2d_dc_unstr = factorLHS_omega2d_dc * pysh.shio.SHCilmToVector(TeTca_clm.coeffs)


    return Omega_LHS_2a_unstr, Omega_LHS_2b_unstr, Omega_LHS_2c_unstr, Omega_LHS_2d_dc_unstr



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

    # (R-Tc)/R^(-l+1) for degrees l
    RTcR_negl1 = np.array([((R-T_c)/R)**(-l-1) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ rhobar*(2*l+1) for l in range(2 * lmax + 1)])


    # Loading terms per each SH field multiplication later & Kalousova scaler
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    q_topo_term1 = g0*rho_l          # No SH field
    q_topo_term2 = -g_m*rho_l        # / RTcRl2
    q_topo_term3 = -3*g0*drho*rho_l  # * RTcRl1/rhobar2l1
    q_topo_term4 = 3*g0*drho*rho_l   # * RTcR_nl1/rhobar2l1
    
    q_geoid_term1 = -g0*rho_l        # No SH field
    q_geoid_term2 = g_m/3            # *rhobar2l1 / RTcRl2
    q_geoid_term3 = -g0*drho         # * RTcR_nl1


    # Perform the multiplications with degree-dependent terms
    ReTcRe_l2_Hlap = topo_clm_copyq.coeffs.copy()                 # topo term 2 
    RTcR_l1_over_rhobar2l1_Hlap = topo_clm_copyq.coeffs.copy()    # topo term 3
    RTcR_negl1_over_rhobar2l1_Hlap = topo_clm_copyq.coeffs.copy() # topo term 4
    rhobar2l1_over_ReTcRe_l2_Glap = geoid_clm_copyq.coeffs.copy() # geoid term 2
    RTcR_negl1_Glap = geoid_clm_copyq.coeffs.copy()               # geoid term 3
    
    for l in range(ReTcRe_l2_Hlap.shape[1]):
        ReTcRe_l2_Hlap[:, l, :] *= (1/RTcR_l2[l])
        RTcR_l1_over_rhobar2l1_Hlap[:, l, :] *= (RTcR_l1[l] / rhobar2l1[l])
        RTcR_negl1_over_rhobar2l1_Hlap[:, l, :] *= (RTcR_negl1[l]/rhobar2l1[l])
        rhobar2l1_over_ReTcRe_l2_Glap[:, l, :] *= (rhobar2l1[l] / RTcR_l2[l])
        RTcR_negl1_Glap[:, l, :] *= RTcR_negl1[l]
    
    
    # Make coeffs array of size lmax+1 for the RHS
    q_coeffs = -Re**4 * Kalousova_scaler1 * (
                  q_topo_term1 
                   * topo_clm_copyq.coeffs[:, :lmax+1, :lmax+1]
                + q_topo_term2 
                   * ReTcRe_l2_Hlap[:, :lmax+1, :lmax+1] 
                + q_topo_term3
                   * RTcR_l1_over_rhobar2l1_Hlap[:, :lmax+1, :lmax+1] 
                + q_topo_term4 
                   * RTcR_negl1_over_rhobar2l1_Hlap[:, :lmax+1, :lmax+1] 
                + q_geoid_term1
                   * geoid_clm_copyq.coeffs[:, :lmax+1, :lmax+1] 
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
                         gidx, gaunt_bare, starts, seg_len, ci, cj, mode_map, N,
                         fdc1_w=0.0, fdc2_w=0.0, Pw=None, Tcind_unstr=None):
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
    M_1f = f1f * (Te2_Lap @ C_max)                        # Te^2 Delta (max w)
    M = M_1d + M_1e + M_1f                                # NOT symmetric -- keep full
    if (fdc1_w != 0.0 or fdc2_w != 0.0) and Pw is not None:
        CmaxP = C_max @ Pw
        M = M + fdc1_w * CmaxP + fdc2_w * (Te2_Lap @ CmaxP)
    return M



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

    # FIX (operator ordering): apply the per-degree dc-elimination weights to
    # H and G FIRST, then multiply by the max(Te-Tc,0) grid -- this is the 
    # ordering consistent with the per-degree elimination and with 
    # Omega_eq1_RHS in the solver (weight-then-multiply).
    # Previously the weights were applied to the coefficients of the PRODUCT
    # (TeTc*H), which differs for laterally varying Te.
    Hp_coeffs = pysh.SHGrid.from_array(topo_grid).expand()
    for l in range(Hp_coeffs.coeffs.shape[1]):
        Hp_coeffs.coeffs[:, l, :] *= 1/RTcR**(l+2)
    Hp_grid = Hp_coeffs.expand(lmax=lmax).data
    TeTcHp_grid = TeTc_grid * Hp_grid

    Gp_coeffs = pysh.SHGrid.from_array(geoid_grid).expand()
    for l in range(Gp_coeffs.coeffs.shape[1]):
        Gp_coeffs.coeffs[:, l, :] *= rhobar*(2*l+1)/(3 * RTcR**(l+2))
    Gp_grid = Gp_coeffs.expand(lmax=lmax).data
    TeTcGp_grid = TeTc_grid * Gp_grid
    
    
    Tcind_grid_o = np.where(T_e_parent_grid > T_c, T_c, 0.0)   # Te<Tc FIX

    # Compute Re*Omega_lm as the term Omega_lm (required in conversion between
    # Banerdt and Beuthe's formulations).  w-coefficient corrected to match the
    # solve: surface -> drhol*g0 (vanishes for rho_l=rho_c), mantle -> gTe.
    term_1 = nu/(1-nu)*rho_l*g0*TeH_grid
    term_2 = + nu/(1-nu)*g_m*rho_l * TeTcHp_grid
    term_3 = -drhol*g0*nu/(1-nu)*T_e_parent_grid *w_grid
    
    term_4 = rho_c*g_m*Tcind_grid_o *w_grid
    term_5 = rho_m*gTe*TeTc_grid *w_grid
    # drhol EXTENSION (zero if rho_l == rho_c): residual w-piece of the
    # (dc-w) substitution, + v1v*g_m*drhol*max(Te-Tc,0)*P_l*w  with the
    # weight applied to w FIRST (weight-then-multiply, as in the solver).
    wp_coeffs = pysh.SHGrid.from_array(w_grid).expand()
    for l in range(wp_coeffs.coeffs.shape[1]):
        wp_coeffs.coeffs[:, l, :] *= 1/RTcR**(l+2)
    wp_grid = wp_coeffs.expand(lmax=lmax).data
    term_6 = + nu/(1-nu)*g_m*drhol * TeTc_grid * wp_grid
    term_7 = - nu/(1-nu)*g_m * TeTcGp_grid
    
    
    # The sum of terms divided by R is omega in the DSP/Banerdt definition
    # (every term of DSP eq 5 carries 1/R). But the consumers of this output
    # (cons_disp_S, stress/strain evaluation, and DSP's own A_lm formula,
    # Beuthe 2008 eq 89) expect Beuthe's OMEGA = Re * omega -- see the DSP
    # comment "Note that omega (Beuthe) = Re * omega". Returning plain omega
    # made the Omega contribution ~Re (3.3e6x) too small, i.e. effectively
    # absent, which flipped the sign of S and corrupted all stress/strain
    # fields by factors of -0.4x to -5.6x. Hence the factor (Re/R) here.
    Re = R - T_e_0/2
    Omega_grid_data = (  term_1 
                       + term_2 
                       + term_3 
                       + term_4 
                       + term_5 
                       + term_6
                       + term_7) * (Re / R)
        
    Omega_grid = pysh.SHGrid.from_array(Omega_grid_data)
    Omega_clm = Omega_grid.expand()
    
    return Omega_clm


def compute_dc(w_clm, topo_clm, geoid_clm, R, T_e_0, lmax):
    """
    Compute the crustal root variations ('bottom loads') dc_lm using the 
    rewritten equation of Gc_lm with drho_lm=0. 
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R

    topo_clm_copy = topo_clm.copy()
    topo_clm_copy.coeffs[0,0,0] = 0

    geoid_clm_copy = geoid_clm.copy()
    geoid_clm_copy.coeffs[0,0,0] = 0
    for l in range(geoid_clm_copy.coeffs.shape[1]):
       geoid_clm_copy.coeffs[:, l, :] *= rhobar*(2*l+1)/3
    
    dc_clm = 1/drho * (rho_l*topo_clm_copy + drhol*w_clm - geoid_clm_copy)
        
    for l in range(dc_clm.coeffs.shape[1]):
       dc_clm.coeffs[:, l, :] *= 1/(RTcR**(l+2)) 
        
    dc_clm = dc_clm + w_clm
    
    return dc_clm


def compute_Gc(w_clm, dc_clm, topo_clm, g0, R, T_e_0, lmax):
    """
    Geoid at the crust-mantle interface, following the CURRENT DSP convention
    (thinshell.py eq 2):
        Gc = (3/(rhobar(2l+1))) * (g0/g_m) * [ (rho_l*H + drhol*w)*phi^l
                                               + drho*(w-dc)*phi ]
    Calculated using the factors:
        - phi^l (instead of phi^(l+1)) 
        - phi^1 (instead of phi^3)
        - (g0/g_m) prefactor

    """
    rhobar = mass * 3.0 / (4.0*np.pi) / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2

    # FIX (degree-0 bookkeeping): strip the reference-radius monopole locally.
    topo_clm_copy = topo_clm.copy()
    topo_clm_copy.coeffs[0,0,0] = 0

    wmdc = w_clm - dc_clm                        # (w - dc)

    H_term = (rho_l*topo_clm_copy + drhol*w_clm) # (rho_l*H + drhol*w) * phi^l
    for l in range(H_term.coeffs.shape[1]):
        H_term.coeffs[:, l, :] *= RTcR**l

    wmdc_term = wmdc.copy()                      # drho * (w-dc) * phi^1
    wmdc_term.coeffs *= drho * RTcR              # degree-independent -> scalar

    Gc_clm = H_term + wmdc_term                  # bracket
    for l in range(Gc_clm.coeffs.shape[1]):      # times (g0/g_m)*3/(rhobar(2l+1))
        Gc_clm.coeffs[:, l, :] *= (g0/g_m) * 3.0/(rhobar*(2*l+1))

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



def stress_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
    """
    Stresses in the DSP/Banerdt convention (Banerdt 1986 eqs A12-A14, as in
    DSP's compute_strains): plane-stress Hooke's law applied to membrane +
    bending strains built from the tangential potential S (== DSP's A_lm)
    and w, with 1/R kernels and the thin-shell top-fiber factor
    eps_f = (Te/2 - depth)/(1 + (Te/2 - depth)/R).
    Returns stresses in MPa (matching DSP).

    NOTE: this replaces the previous Beuthe eq-(73) stress-function form
    (kept below as stress_fields_beuthe73), which evaluates the top-fiber
    stress with exact z/(Re+z) curvature factors and 1/Re kernels. The two
    differ by O(Te/R) factors (~4-7% for Te=268 km) -- for benchmarking
    against DSP the convention must match DSP.
    """
    O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    S_grid  = S_sol.expand(lmax=lmax)
    w_grid  = w_sol.expand(lmax=lmax)
    Te_grid = T_e_parent.expand(lmax=lmax)

    # membrane strains (Banerdt A16-A18 with S in place of A)
    eps_t    = 1/R * (O1S - S_grid.data + w_grid.data)
    eps_p    = 1/R * (O2S - S_grid.data + w_grid.data)
    omega_sh = 1/R * (2*O3S)                       # engineering shear
    # bending strains (A19-A21); note O1w = d2w/dth2 + w, so
    # kappa_t = -(d2w/dth2)/R^2 - w/R^2 = -O1w/R^2 (and analogously kappa_p)
    kappa_t = -1/R**2 * O1w
    kappa_p = -1/R**2 * O2w
    tau     = -2/R**2 * O3w

    zeta  = Te_grid.data/2.0 - depth
    eps_f = zeta / (1.0 + zeta/R)
    DpsiTeR = E/(1.0 - nu**2)

    sigma_tt = (eps_t + nu*eps_p + eps_f*(kappa_t + nu*kappa_p)) * DpsiTeR / 1e6
    sigma_pp = (eps_p + nu*eps_t + eps_f*(kappa_p + nu*kappa_t)) * DpsiTeR / 1e6
    sigma_tp = (omega_sh + eps_f*tau) * 0.5 * DpsiTeR * (1.0 - nu) / 1e6

    return (pysh.SHGrid.from_array(sigma_tt),
            pysh.SHGrid.from_array(sigma_pp),
            pysh.SHGrid.from_array(sigma_tp))


def stress_fields_beuthe73(w_sol, F_sol, Omega_grid, T_e_parent, R, T_e_0, lmax):
    """
    Beuthe eqs (73). MISSING THE TOROIDAL STRESS FUNCTION H TERMS!
    Kept for reference; expects Omega_grid = Beuthe's Omega = Re*omega
    (i.e. the output of the corrected compute_Omega). Differs from the DSP
    convention by O(Te/R) factors (1/Re kernels, exact top-fiber z/(Re+z)).
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


def strain_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
    """
    Total strains in the DSP/Banerdt convention (membrane + top-fiber
    bending), matching DSP's tot_theta / tot_phi / tot_thetaphi:
        tot = eps + eps_f*kappa,  eps_f = (Te/2-depth)/(1+(Te/2-depth)/R)
    MISSING THE TOROIDAL DISPLACEMENT POTENTIAL T TERMS!

    FIXES vs previous version:
    (1) the returned fields previously OVERWROTE the totals with the
        membrane-only strains (tot_eps_tt = SHGrid(eps_t)), silently
        discarding the entire bending contribution -- which is comparable
        to the membrane strains at Te ~ 268 km;
    (2) kernels switched 1/Re -> 1/R and the bending factor to
        zeta/(1+zeta/R), matching the DSP benchmark convention.
    """
    # Return diff operator applied S and w terms, in grid.data format
    O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    
    S_grid  = S_sol.expand(lmax=lmax)
    w_grid  = w_sol.expand(lmax=lmax)
    Te_grid = T_e_parent.expand(lmax=lmax)

    eps_t    = 1/R * (O1S - S_grid.data + w_grid.data)    
    eps_p    = 1/R * (O2S - S_grid.data + w_grid.data)
    gamma_tp = 1/R * (2*O3S)
    
    kappa_t = -1/R**2 * O1w
    kappa_p = -1/R**2 * O2w
    tau     = -2/R**2 * O3w
    
    zeta = Te_grid.data/2.0 - depth
    tot_strain_pref = zeta / (1.0 + zeta/R)
    
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

    Re = R - T_e_0/2
    # FIX: xi from Re, not R -- DSP's A_lm formula (Beuthe 2008 eq 89) uses
    # eps = 12*Re^2/Te^2 and beta = 1/(1+eps) built with Re. Verified per
    # degree: with xi(Re) and Omega = Re*omega, S matches DSP's A_lm to
    # <0.25% (formulation floor); xi(R) leaves up to 0.7%.
    # NOTE: this function expects Omega_grid = Beuthe's Omega = Re*omega,
    # i.e. the output of the corrected compute_Omega.
    xi = 12*Re**2/Te_grid.data**2
    eta = xi/(1+xi)
    
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
                 T_e_0, g0, mass, rhs_override=None, omega_on=True,
                 D_eta_clm=None, a_eta_clm=None, eta_clm=None):
    """
    ETA_FULL mode (all three eta kwargs provided): implements Beuthe's
    UNSIMPLIFIED variable-thickness equations (58)/(66):
      Delta'(eta*D Delta' w) - (1-nu)A(eta*D; w) + R^3 A(eta; F) = RHS1
      Delta'(eta*a Delta' F) - (1+nu)A(eta*a; F) - (1/R)A(eta; w) = RHS2
    via: (i) D_eta/a_eta fields in the existing A/B convolutions;
    (ii) A(eta; .) coupling blocks from the same Gaunt plan with the pure-A
    weight W_Aonly = -br/4 (see W_numeric_Aonly), replacing the diagonal
    Delta' couplings of the eta-truncated system (Kalousova eqs 13-14).
    The PLAIN a_clm still feeds Omega_eq2_LHS (exact constant-Te omega
    decomposition requires eq-2 scale Y = Re^2 = plain-alpha *
    Kalousova_scaler2(Re^2/R)). Legacy behaviour when kwargs are None.
    """
    mode_map = make_mode_map(lmax)
    N = len(mode_map)
    Re   = R - T_e_0/2
    buoy = (Re/T_e_0)**3 * (Re/E) * g0 * (rho_m-rho_c)
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
    (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr, 
     Omega_LHS_1d_unstr, Omega_LHS_1e_unstr, Omega_LHS_1f_unstr,
     Omega_LHS_dc1_unstr, Omega_LHS_dc2_unstr) = (
                Omega_eq1_LHS(T_e_parent, lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
    # group 1: terms 1a,1c, dc1 -- NO output Laplacian
    field_ac = (Omega_LHS_1a_unstr[gidx] + Omega_LHS_1b_unstr[gidx]
                + Omega_LHS_1c_unstr[gidx]
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
        Omega_eq2_LHS(T_e_parent, a_eta_clm, lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
    # terms 2a, 2b, 2c (+2d, zero) -- carry Delta' at the OUTPUT degree.
    # b_tilde FIX (non-symmetric pathway): previously assembled per cell with
    # lap_out on the ci side and scattered SYMMETRICALLY -- exact at constant
    # Te (monopole omega-fields leave only diagonal cells) but WRONG for
    # laterally varying Te: the (j,i) orientation must carry Delta'(l_j),
    # not Delta'(l_i). Rebuilt in the omega_on block as
    #     b_tilde = diag(Delta'_out) @ C_conv(2a+2b+2c fields),
    # carrying the output weight on the correct side for both orientations.
    fields_2sum = (Omega_LHS_2a_unstr + Omega_LHS_2b_unstr
                   + Omega_LHS_2c_unstr + Omega_LHS_2d_dc_unstr)
 
 
 
    # ---- scatter per-cell values into dense blocks (loop over CELLS) ------
    # (b_tilde no longer scattered here: built as a non-symmetric matrix
    #  product in the omega_on block -- see b_tilde FIX above.)
    A = np.zeros((N, N))
    A_tilde = np.zeros((N, N))
    B = np.zeros((N, N))
    for c in range(ci.size):
        i, j = int(ci[c]), int(cj[c])
        vA = cellA[c] + (buoy if i == j and omega_on == False else 0.0)
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
        # Te<Tc FIX: Tc moved out of f1e into the indicator field Tcind
        # (DSP: rhoc*gmoho*(Tc if Tc < Te else 0)); see build_A_tilde_group2.
        f1e   = -Re/12 * g_m_loc * rho_c             * scaler_A * corr1   # crust column: g_m (matches 1e)
        Tcind_grid_loc = np.where(Te_grid_loc > T_c, T_c, 0.0)
        Tcind_clm_loc  = pysh.SHGrid.from_array(Tcind_grid_loc).expand()
        Tcind_clm_loc  = pysh.SHCoeffs.from_array(Tcind_clm_loc.coeffs[:, :2*lmax+1, :2*lmax+1])
        Tcind_unstr    = pysh.shio.SHCilmToVector(Tcind_clm_loc.coeffs)
        f1f   = -Re/12 * gTe * rho_m                 * scaler_A * corr1   # mantle: gTe (matches 1f)
        # drhol EXTENSION -- reinstated LHS dc coupling (zero iff rho_l == rho_c):
        # eq (1) with drhol leaves  drho*(w-dc) = P_l*[G/K_l - rho_l*H] - drhol*P_l*w,
        # P_l = phi^(-(l+2)), so omega's (dc-w) term contributes
        #   + v1v * g_m * drhol * max(Te-Tc,0) * P_l' * w / R    (l' = operand degree)
        # to C_w. Sign/pattern pairs with 1b/1c (c1-half, +2Re^3) and 1e/1f
        # (c2-half, -Re/12). The OLD commented value (+Re/12, drho) had the
        # wrong sign, wrong density, and no P weight.
        fdc1_w =  2*Re**3 * nu/(1-nu) * g_m_loc * drhol * scaler_A * corr1
        fdc2_w = -Re/12   * nu/(1-nu) * g_m_loc * drhol * scaler_A * corr1
        Pw_diag = np.diag(np.array([RTcR_loc**(-(l+2)) for l, _ in mode_map]))
        A_tilde_group2 = build_A_tilde_group2(
            Te_unstr, Te2_unstr, max_unstr,
            f1d, f1e, f1f,
            gidx, plan['term_gaunt_bare'], starts, seg_len, ci, cj, mode_map, N,
            fdc1_w=fdc1_w, fdc2_w=fdc2_w, Pw=Pw_diag, Tcind_unstr=Tcind_unstr)
 
        A = A + A_tilde + A_tilde_group2
 
        # b_tilde: non-symmetric matrix form (see b_tilde FIX above):
        # diag(Delta'_out) @ conv(2a + 2b + 2c fields). Reduces exactly to
        # the former symmetric-scatter result for monopole omega-fields
        # (constant Te), and is correct for laterally varying Te.
        C_2abc  = build_conv_matrix(fields_2sum, gidx, plan['term_gaunt_bare'],
                                    starts, seg_len, ci, cj, N)
        b_tilde = np.diag(d_l2) @ C_2abc
 
        # drhol EXTENSION -- q's w-coupling (LHS diagonal; zero iff rho_l==rho_c):
        # substituting the drhol terms of eqs (1)-(3) leaves
        #   q = q(H,G) + Lam_q(l)*w,
        #   Lam_q(l) = qH(l) with rho_l -> drhol   (eqs 1-3 pair rho_l*H + drhol*w):
        #   Lam_q = g0*drhol - g_m*drhol*P_l
        #           - 3*g0*drho*drhol/(rhobar*(2l+1)) * (phi^l - phi^(-(l+1)))
        # Enters the flexure eq as -Re^4*K1*q  =>  A_diag += Re^4*K1*Lam_q.
        # Pure per-degree scalars (no Te field): diagonal, no Gaunt needed.
        if drhol != 0.0:
            for idx, (l_m, _) in enumerate(mode_map):
                P_l  = RTcR_loc**(-(l_m + 2))
                Lam_q = ( g0*drhol - g_m_loc*drhol*P_l
                          - 3*g0*drho*drhol/(rhobar_loc*(2*l_m + 1))
                            * (RTcR_loc**l_m - RTcR_loc**(-(l_m + 1))) )
                A[idx, idx] += Re**4 * scaler_A * Lam_q
 
        # drhol EXTENSION -- eq-2 omega dc coupling (zero iff rho_l == rho_c):
        # the same + v1v*g_m*drhol*max*P_l'*w/R content must enter b_tilde,
        # pattern of 2b/2c: field = max(Te-Tc,0)*(eta*alpha), output Delta',
        # operand weight Pw. Built as an explicit (non-symmetric) matrix
        # product to carry the output/operand weights on the correct sides.
        if drhol != 0.0:
            a_field_clm = a_eta_clm 
            a_grid_loc  = a_field_clm.expand(lmax=3*lmax).data
            maxa_clm = pysh.SHGrid.from_array(TeTc_grid_loc * a_grid_loc).expand()
            maxa_clm = pysh.SHCoeffs.from_array(maxa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
            maxa_unstr = pysh.shio.SHCilmToVector(maxa_clm.coeffs)
            C_maxa = build_conv_matrix(maxa_unstr, gidx, plan['term_gaunt_bare'],
                                       starts, seg_len, ci, cj, N)
            scaler2_loc = (Re**2/R) 
            fdc_2d = (1-nu) * nu/(1-nu) * g_m_loc * drhol * scaler2_loc
            Dlp_out = np.diag(d_l2)                 # Delta' at OUTPUT degree
            b_tilde = b_tilde + fdc_2d * (Dlp_out @ C_maxa @ Pw_diag)
 
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
    D_clm, a_clm, D_eta_clm, a_eta_clm, eta_clm = derive_D_a(T_e_parent, LMAX_REF)

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
                
                al_r, be_r, ga_r = rotate_angles
                D_eta_use  = D_eta_clm.rotate(al_r, be_r, ga_r)
                a_eta_use  = a_eta_clm.rotate(al_r, be_r, ga_r)
                eta_use    = eta_clm.rotate(al_r, be_r, ga_r)

            else:
                T_e_use, D_use, a_use, topo_use, geoid_use = (
                    T_e_parent, D_clm, a_clm, topo_clm, geoid_clm)
                D_eta_use, a_eta_use, eta_use = D_eta_clm, a_eta_clm, eta_clm
                
            print('Start solving of system')
            t = time.perf_counter()
            w, F, q = solve_beuthe(topo_use, geoid_use, T_e_use, D_use, a_use, plan, 
                             lmax_run, R, T_e_0, g0, mass, omega_on=omega_On,
                             D_eta_clm=D_eta_use, a_eta_clm=a_eta_use,
                             eta_clm=eta_use)
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


    # Residual vs reference, plot only if there are more than one lmax runs
    if len(LMAX_RUNS)>1:
        S_ref = solutions_w[LMAX_REF, 0].spectrum()
        fig2, ax2 = plt.subplots(figsize=(8,5))
        for lmax_run in LMAX_RUNS[:-1]:
            do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
            for rotation in ([0, 1] if do_rotation_check else [0]):
                linestyle = 'solid' if rotation == 0 else 'dashed'
                S = solutions_w[lmax_run, rotation].spectrum()
                l = np.arange(2, lmax_run+1)
                ax2.plot(l, np.abs(S[2:]/S_ref[2:lmax_run+1]-1.0)*100, '.-', 
                         label=(f'lmax={lmax_run} vs {LMAX_REF}'
                                + (f', rotated {rotate_angles}' 
                                   if rotation else '')), 
                         linestyle=linestyle)
        ax2.set_xlabel('degree l'); ax2.set_ylabel(r'$|S_l/S_l^{ref}-1|$*100%')
        ax2.legend(); ax2.grid(True)
        ax2.set_title(f'M3 - Residual vs lmax_ref={LMAX_REF}')
        plt.tight_layout(); 
        if SaveFigs:
            plt2_title = (f'M3 - Residuals w power, lmax_run={LMAX_RUNS}, '
                          f'lmaxTe={lmax_Te_fit}'
                          + (f', rotated {rotate_angles}' if rotation else '') 
                          + '.png')
            FigPath2 = os.path.join(SavePath, plt2_title)
            plt.savefig(FigPath2, dpi=200)
        plt.show(); plt.close()


# %% PLOTS - INPUTS TOPOGRAPHY, Te, D AND alpha


    # Plot inputs topography, Te, D and alpha
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(13,8))
    do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    if do_rotation_check:
        T_e_use_clm, D_use_clm, a_use_clm, topo_use_clm, _ = rotate_inputs(
            rotate_angles, T_e_parent, D_clm, a_clm, 
            topo_clm, geoid_clm)
    else:
        T_e_use_clm, D_use_clm, a_use_clm, topo_use_clm = T_e_parent, D_clm, a_clm, topo_clm
    
    args_plot = dict(tick_interval=[45, 30])

    topography_km = topo_use_clm.expand(lmax=3*LMAX_REF)
    topography_km.data = (topography_km.data - R)/1e3
    topo_min, topo_max = topography_km.data.min(), topography_km.data.max()
    cmap_limits_topo_diff =[topo_min, 10]
    topography_km.plot(ax=ax0, 
                       cmap=cmc.navia,
                       cmap_limits = cmap_limits_topo_diff,
                       grid=True,
                       colorbar='right', 
                       cb_label='Topographic height [km]',
                       **args_plot)
    ax0.set_title(f'M3 - MOLA topography map, exp. to lmax={LMAX_REF}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    
    T_e_parent_km = T_e_use_clm.expand(lmax=3*lmax_Te_fit)
    T_e_parent_km.data = T_e_parent_km.data/1e3
    T_e_parent_km.plot(ax=ax1, 
                       ticks = 'wSne',
                       ylabel=None,
                       grid=True,
                       cmap=cmc.lajolla, 
                       colorbar='right', 
                       cb_label=r'$T_e \ [km]$',
                       **args_plot)
    ax1.set_title(f'M3 - Te input map (Plesa et al. 2018), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    D_use_clm.expand(lmax=3*lmax_Te_fit).plot(ax=ax2, 
                                        cmap=cmc.lajolla, 
                                        colorbar='right', 
                                        cb_label=r'$D \ [N\cdot m]$',
                                        **args_plot)  
    ax2.set_title(f'M3 - Flexural rigidity D (Te-derived), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    a_use_clm.expand(lmax=3*lmax_Te_fit).plot(ax=ax3, 
                                        cmap=cmc.lajolla, 
                                        colorbar='right', 
                                        cb_label=r'$\alpha \ [m/N$]',
                                        **args_plot) 
    ax3.set_title(f'M3 - Parameter alpha (Te-derived), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    plt.suptitle('M3 - Input maps topography & Te, and derived parameters D and $\\alpha$')
    plt.tight_layout()
    if SaveFigs:
        plt3_title = (f'M3 - Inputs Te, D and alpha, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath3 = os.path.join(SavePath, plt3_title)
        plt.savefig(FigPath3, dpi=200)
    plt.show(); plt.close()
        
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

    


# %% PLOTS - RESIDUALS MAPS BETWEEN DSP AND M3
    
    grid_expansion = 3*LMAX_REF
    
    w_fine = pysh.SHGrid.from_array(
            solutions_w[LMAX_REF, 0].expand(lmax=grid_expansion).data/1e3)
    w_clm = solutions_w[LMAX_REF, 0]
    dc_clm = compute_dc(w_clm, topo_clm, geoid_clm, R, T_e_0, LMAX_REF)
    dc_clm_zeroed = dc_clm.copy()
    dc_clm_zeroed.coeffs[0,0,0] = 0
    dc_grid = dc_clm_zeroed.expand(lmax=grid_expansion)/1e3
    
    topo_grid = topo_clm.expand(lmax=grid_expansion)/1e3 - R/1e3
    
    T_c_grid = topo_grid.data + dc_grid.data - w_fine.data + T_c*np.ones((2*(grid_expansion+1)+1, 4*(grid_expansion+1)+1))/1e3
    T_c_grid = pysh.SHGrid.from_array(T_c_grid)
    
    # Load in DSP results
    w_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_w_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    dc_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_dc_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    Tc_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Tc_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    
    
    # # PLOTTING
    # args_expand = dict(lmax=grid_expansion, lmax_calc=LMAX_REF)
    # args_plot = dict(tick_interval=[45, 30])
    # args_titles = dict(fontsize=13, fontweight='bold')
    # fig, ((ax1, ax2, ax3),
    #       (ax4, ax5, ax6),
    #       (ax7, ax8, ax9),) = plt.subplots(3,3, figsize=(16,10), 
    #                                        width_ratios=[1,1,1], 
    #                                        height_ratios=[1,1,1],
    #                                        )
    # grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    # grid_w_DSP.plot(ax=ax1, 
    #            cmap_limits=[-7, 3], 
    #            cmap=cmap3, 
    #            # colorbar='right', 
    #            # cb_label='w [km]',
    #            ticks = 'Wsen',
    #            xlabel = None,
    #            **args_plot
    #            )
    # ax1.set_title('DSP - Radial displacement w', **args_titles)
    # w_fine.plot(ax=ax2, 
    #            cmap_limits=[-7, 3], 
    #             cmap=cmap3, 
    #             colorbar='right', 
    #             cb_label='w [km]',
    #             ticks = 'wsen',
    #             xlabel = None,
    #             ylabel = None,
    #             **args_plot
    #             )
    # ax2.set_title('M3 - Radial displacement w', **args_titles)
    # w_diff_DSPM3 = grid_w_DSP.copy()
    # w_diff_DSPM3.data = grid_w_DSP.data - w_fine.data
    # w_diff_DSPM3.plot(ax=ax3, 
    #                   cmap=cmap2, 
    #                   colorbar='right', 
    #                   cb_label='w [km]',
    #                   ticks = 'wsen',
    #                   xlabel = None,
    #                   ylabel = None,
    #                   **args_plot
    #                   )
    # ax1.contour(grid_w_DSP.data>0, 
    #            levels=[0.99], 
    #            extent=(0,360,-90,90), 
    #            colors='k', 
    #            origin='upper')
    # ax2.contour(w_fine.data>0, 
    #            levels=[0.99], 
    #            extent=(0,360,-90,90), 
    #            colors='k', 
    #            origin='upper')
    # ax3.set_title('Radial displacement w \nresidual DSP - M3', **args_titles)


    # grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
    # grid_dc_DSP.plot(ax=ax4, 
    #             cmap=cmap3, 
    #             cmap_limits=[-50, 30], 
    #             # colorbar='right', 
    #             # cb_label='dc [km]',
    #             ticks = 'Wsen',
    #             xlabel = None,
    #             **args_plot
    #             )
    # ax4.set_title('DSP - Crustal root variations', **args_titles)
    # dc_grid.plot(ax=ax5, 
    #              cmap=cmap3, 
    #              cmap_limits=[-50, 30], 
    #              colorbar='right', 
    #              cb_label='$\\delta c$ [km]',
    #              ticks = 'wsen',
    #              xlabel = None,
    #              ylabel = None,
    #              **args_plot
    #               )
    # ax5.set_title('M3 - Crustal root variations', **args_titles)

    # dc_diff_DSPM3 = grid_dc_DSP.copy()
    # dc_diff_DSPM3.data = grid_dc_DSP.data - dc_grid.data
    # dc_diff_DSPM3.plot(ax=ax6, 
    #                    cmap=cmap2, 
    #                    colorbar='right', 
    #                    cb_label='$\\delta c$ [km]',
    #                    ticks = 'wsen',
    #                    xlabel = None,
    #                    ylabel = None,
    #                    **args_plot
    #                     )
    # ax6.set_title('Crustal root variations \nresidual DSP - M3', **args_titles)

    
    # grid_Tc_DSP = pysh.SHCoeffs.from_array(Tc_DSP.coeffs / 1e3).expand(**args_expand)
    # grid_Tc_DSP.plot(ax=ax7, 
    #             cmap=cmap3, 
    #             cmap_limits=[0, 110], 
    #             # colorbar='right', 
    #             # cb_label='Tc [km]',
    #             ticks = 'WSen',
    #             **args_plot
    #             )
    # ax7.set_title('DSP - Crustal thickness', **args_titles)
    # T_c_grid.plot(ax=ax8, 
    #               cmap=cmap3, 
    #               cmap_limits=[0, 110], 
    #               colorbar='right', 
    #               cb_label='$T_c$ [km]',
    #               ticks = 'wSen',
    #               ylabel = None,
    #               **args_plot
    #                )
    # ax8.set_title('M3 - Crustal thickness', **args_titles)

    # Tc_diff_DSPM3 = grid_Tc_DSP.copy()
    # Tc_diff_DSPM3.data = grid_Tc_DSP.data - T_c_grid.data
    # Tc_diff_DSPM3.plot(ax=ax9, 
    #                    cmap=cmap2, 
    #                    colorbar='right', 
    #                    cb_label='$T_c$ [km]',
    #                    ticks = 'wSen',
    #                    ylabel = None,
    #                    **args_plot
    #                    )
    # ax9.set_title('Crustal thickness \nresidual DSP - M3', **args_titles)

    # # ax7.set_visible(False);  ax8.set_visible(False);  ax9.set_visible(False)


    # plt.suptitle(f'Residual checks DSP and M3. lmax={LMAX_REF}, \n'
    #              f'DSP constant $T_e$={Te_input/1e3} km, '
    #              + (f'M3 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
    #              + ('M3 $T_e$=Plesa Strain14 Map' if strain==14 else '')
    #              + ('M3 $T_e$=Plesa Strain17 Map' if strain==17 else '')
    #              + f'\nDSP & M3 constant $T_c$={T_c/1e3} km, '
    #              f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, $\\rho_m$ = {rho_m} kg/m$^3$',
    #              y=0.98, fontsize=15)
    # plt.tight_layout()
    # plt.show()
    # if SaveFigs:
    #     plt_savetitle = ('Residual_checks_DSP_M3_lmax={LMAX_REF}_'
    #             + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
    #             + ('Te_M3=PlesaStrain14Map_'
    #                'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
    #             + ('Te_M3=PlesaStrain17Map_'
    #                'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
    #             + 'Tc={T_c/1e3}km'
    #             + '.png')
    #     FigPath = os.path.join(SavePath, plt_savetitle)
    #     plt.savefig(FigPath, dpi=100)
    # plt.close()
    
    
    
    
    
    
    
# %% DSP-M3 RESIDUAL PLOTS BETTER LAYOUT
    
# # PLOTTING
#     import matplotlib.gridspec as gridspec
#     import matplotlib.colors as mcolors
#     import matplotlib.cm as cm

#     args_expand = dict(lmax=grid_expansion, lmax_calc=LMAX_REF)
#     args_plot = dict(tick_interval=[45, 30])

#     # 1. Increase overall figure height to accommodate larger plots and clear spacing
#     fig = plt.figure(figsize=(16, 14))
    
#     # 2. Outer grid controls the 3 main data rows. 
#     # Increase hspace here to add massive spacing BETWEEN your rows.
#     outer_gs = gridspec.GridSpec(3, 1, hspace=-0.15) 

#     # --- ROW 1: Radial Displacement w ---
#     # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
#     # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
#     inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
#                                                  height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
#     ax1 = fig.add_subplot(inner_gs1[0, 0:2])
#     ax2 = fig.add_subplot(inner_gs1[0, 2:4])
#     ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
#     # Shared colorbar spans underneath columns 0 and 1
#     cax_w_shared = fig.add_subplot(inner_gs1[1, 1:3])
#     # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
#     cax_w_diff   = fig.add_subplot(inner_gs1[1, 4:6])

#     grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
#     grid_w_DSP.plot(ax=ax1, cmap_limits=[-7, 3], cmap=cmap3, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
#     ax1.set_title('DSP - Radial displacement w', fontweight="bold")
    
#     w_fine.plot(ax=ax2, cmap_limits=[-7, 3], cmap=cmap3, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
#     ax2.set_title('M3 - Radial displacement w', fontweight="bold")
    
#     w_diff_DSPM3 = grid_w_DSP.copy()
#     w_diff_DSPM3.data = grid_w_DSP.data - w_fine.data
#     w_min, w_max = w_diff_DSPM3.data.min(), w_diff_DSPM3.data.max()
#     w_diff_DSPM3.plot(ax=ax3, cmap=cmap2, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
#     ax3.set_title('Radial displacement w residual DSP - M3', fontweight="bold")

#     norm_w = mcolors.Normalize(vmin=-7, vmax=3)
#     cb1 = fig.colorbar(cm.ScalarMappable(norm=norm_w, cmap=cmap3), cax=cax_w_shared, orientation='horizontal')
#     cb1.set_label('w [km]', fontweight="bold")

#     norm_w_diff = mcolors.Normalize(vmin=w_min, vmax=w_max)
#     cb2 = fig.colorbar(cm.ScalarMappable(norm=norm_w_diff, cmap=cmap2), cax=cax_w_diff, orientation='horizontal')
#     cb2.set_label('w [km]', fontweight="bold")

#     ax1.contour(grid_w_DSP.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')
#     ax2.contour(w_fine.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')


#     # --- ROW 2: Crustal Root Variations ---
#     inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
#                                                  height_ratios=[1, 0.03], hspace=-0.45, wspace=0.15)
#     ax4 = fig.add_subplot(inner_gs2[0, 0:2])
#     ax5 = fig.add_subplot(inner_gs2[0, 2:4])
#     ax6 = fig.add_subplot(inner_gs2[0, 4:6])
    
#     cax_dc_shared = fig.add_subplot(inner_gs2[1, 1:3])
#     cax_dc_diff   = fig.add_subplot(inner_gs2[1, 4:6])

#     grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
#     grid_dc_DSP.plot(ax=ax4, cmap=cmap3, cmap_limits=[-50, 30], colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
#     ax4.set_title('DSP - Crustal root variations', fontweight="bold")
    
#     dc_grid.plot(ax=ax5, cmap=cmap3, cmap_limits=[-50, 30], colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
#     ax5.set_title('M3 - Crustal root variations', fontweight="bold")

#     dc_diff_DSPM3 = grid_dc_DSP.copy()
#     dc_diff_DSPM3.data = grid_dc_DSP.data - dc_grid.data
#     dc_min, dc_max = dc_diff_DSPM3.data.min(), dc_diff_DSPM3.data.max()
#     dc_diff_DSPM3.plot(ax=ax6, cmap=cmap2, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
#     ax6.set_title('Crustal root variations residual DSP - M3', fontweight="bold")

#     norm_dc = mcolors.Normalize(vmin=-50, vmax=30)
#     cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_dc, cmap=cmap3), cax=cax_dc_shared, orientation='horizontal')
#     cb3.set_label('$\\delta c$ [km]', fontweight="bold")

#     norm_dc_diff = mcolors.Normalize(vmin=dc_min, vmax=dc_max)
#     cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_dc_diff, cmap=cmap2), cax=cax_dc_diff, orientation='horizontal')
#     cb4.set_label('$\\delta c$ [km]', fontweight="bold")


#     # --- ROW 3: Crustal Thickness ---
#     inner_gs3 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[2], 
#                                                  height_ratios=[1, 0.03], hspace=-0.35, wspace=0.15)
#     ax7 = fig.add_subplot(inner_gs3[0, 0:2])
#     ax8 = fig.add_subplot(inner_gs3[0, 2:4])
#     ax9 = fig.add_subplot(inner_gs3[0, 4:6])
    
#     cax_tc_shared = fig.add_subplot(inner_gs3[1, 1:3])
#     cax_tc_diff   = fig.add_subplot(inner_gs3[1, 4:6])

#     grid_Tc_DSP = pysh.SHCoeffs.from_array(Tc_DSP.coeffs / 1e3).expand(**args_expand)
#     grid_Tc_DSP.plot(ax=ax7, cmap=cmap3, cmap_limits=[0, 110], colorbar=None, ticks='WSen', **args_plot)
#     ax7.set_title('DSP - Crustal thickness', fontweight="bold")
    
#     T_c_grid.plot(ax=ax8, cmap=cmap3, cmap_limits=[0, 110], colorbar=None, ticks='wSen', ylabel=None, **args_plot)
#     ax8.set_title('M3 - Crustal thickness', fontweight="bold")

#     Tc_diff_DSPM3 = grid_Tc_DSP.copy()
#     Tc_diff_DSPM3.data = grid_Tc_DSP.data - T_c_grid.data
#     tc_min, tc_max = Tc_diff_DSPM3.data.min(), Tc_diff_DSPM3.data.max()
#     Tc_diff_DSPM3.plot(ax=ax9, cmap=cmap2, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
#     ax9.set_title('Crustal thickness residual DSP - M3', fontweight="bold")
        
#     norm_tc = mcolors.Normalize(vmin=0, vmax=110)
#     cb5 = fig.colorbar(cm.ScalarMappable(norm=norm_tc, cmap=cmap3), cax=cax_tc_shared, orientation='horizontal')
#     cb5.set_label('$T_c$ [km]', fontweight="bold")

#     norm_tc_diff = mcolors.Normalize(vmin=tc_min, vmax=tc_max)
#     cb6 = fig.colorbar(cm.ScalarMappable(norm=norm_tc_diff, cmap=cmap2), cax=cax_tc_diff, orientation='horizontal')
#     cb6.set_label('$T_c$ [km]', fontweight="bold")


#     # --- GLOBAL SUPTITLE AND OUTPUT ---
#     plt.suptitle(f'Residual checks DSP and M3. lmax={LMAX_REF}, \n'
#                  f'DSP constant $T_e$={Te_input/1e3} km, '
#                  + (f'M3 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
#                  + ('M3 $T_e$=Plesa Strain14 Map' if strain==14 else '')
#                  + ('M3 $T_e$=Plesa Strain17 Map' if strain==17 else '')
#                  + f'\nDSP & M3 constant $T_c$={T_c/1e3} km, '
#                  f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, $\\rho_m$ = {rho_m} kg/m$^3$',
#                  y=0.85, fontsize=15)
                
#     if SaveFigs:
#         plt_savetitle = ('Residual_checks_DSP_M3_lmax={LMAX_REF}_'
#                 + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
#                 + ('Te_M3=PlesaStrain14Map_'
#                    'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
#                 + ('Te_M3=PlesaStrain17Map_'
#                    'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
#                 + 'Tc={T_c/1e3}km'
#                 + '.png')
#         FigPath = os.path.join(SavePath, plt_savetitle)
#         plt.savefig(FigPath, dpi=100, bbox_inches='tight')
#     plt.show()
#     plt.close()




# %% DSP-M3 RESIDUAL PLOTS BETTER LAYOUT - WITHOUT CRUSTAL THICKNESS
    
# PLOTTING
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    grid_expansion = 3*LMAX_REF
    args_expand = dict(lmax=grid_expansion, lmax_calc=LMAX_REF)
    args_plot = dict(tick_interval=[45, 30], grid=True)

    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    outer_gs = gridspec.GridSpec(2, 1, hspace=-0.15) 



    # --- ROW 1: Radial Displacement w ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
                                                 height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
    ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_w_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_w_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_w = [-23,3]
    
    grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    w_diff_DSPM3 = grid_w_DSP.copy()
    w_diff_DSPM3.data = grid_w_DSP.data - w_fine.data
    w_min, w_max = w_diff_DSPM3.data.min(), w_diff_DSPM3.data.max()
    cmap_limits_w_diff =[-max(abs(w_min), abs(w_max)), max(abs(w_min), abs(w_max))]

    grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    grid_w_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_w, 
                    cmap=cmap3, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Radial displacement w', fontweight="bold")
    
    w_fine.plot(ax=ax2, 
                cmap_limits=cmap_limits_w, 
                cmap=cmap3, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M3 - Radial displacement w', fontweight="bold")
    
    w_diff_DSPM3 = grid_w_DSP.copy()
    w_diff_DSPM3.data = grid_w_DSP.data - w_fine.data
    w_diff_DSPM3.plot(ax=ax3, cmap=cmap2,
                      cmap_limits = cmap_limits_w_diff,
                      colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Radial displacement w residual DSP - M3', fontweight="bold")

    norm_w = mcolors.Normalize(vmin=cmap_limits_w[0], vmax=cmap_limits_w[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_w, 
                    cmap=cmap3), 
                       cax=cax_w_shared, orientation='horizontal')
    cb1.set_label('w [km]', fontweight="bold")
   
    norm_w_diff = mcolors.Normalize(vmin=cmap_limits_w_diff[0], vmax=cmap_limits_w_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(norm=norm_w_diff, cmap=cmap2), cax=cax_w_diff, orientation='horizontal')
    cb2.set_label('w [km]', fontweight="bold")

    ax1.contour(grid_w_DSP.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')
    ax2.contour(w_fine.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')




    # --- ROW 2: Crustal Root Variations ---
    inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                 height_ratios=[1, 0.03], hspace=-0.35, wspace=0.15)
    ax4 = fig.add_subplot(inner_gs2[0, 0:2])
    ax5 = fig.add_subplot(inner_gs2[0, 2:4])
    ax6 = fig.add_subplot(inner_gs2[0, 4:6])
    
    cax_dc_shared = fig.add_subplot(inner_gs2[1, 1:3])
    cax_dc_diff   = fig.add_subplot(inner_gs2[1, 4:6])

    cmap_limits_dc = [-50, 30]
    grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
    dc_diff_DSPM3 = grid_dc_DSP.copy()
    dc_diff_DSPM3.data = grid_dc_DSP.data - dc_grid.data
    dc_min, dc_max = dc_diff_DSPM3.data.min(), dc_diff_DSPM3.data.max()
    cmap_limits_dc_diff =[-max(abs(dc_min), abs(dc_max)), max(abs(dc_min), abs(dc_max))]

    grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
    grid_dc_DSP.plot(ax=ax4, cmap=cmap3, cmap_limits=[-50, 30], colorbar=None, ticks='WSen', **args_plot)
    ax4.set_title('DSP - Crustal root variations', fontweight="bold")
    
    dc_grid.plot(ax=ax5, cmap=cmap3, cmap_limits=cmap_limits_dc, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax5.set_title('M3 - Crustal root variations', fontweight="bold")

    dc_diff_DSPM3 = grid_dc_DSP.copy()
    dc_diff_DSPM3.data = grid_dc_DSP.data - dc_grid.data
    dc_diff_DSPM3.plot(ax=ax6, cmap=cmap2, 
                       cmap_limits=cmap_limits_dc_diff, 
                       colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax6.set_title('Crustal root variations residual DSP - M3', fontweight="bold")

    norm_dc = mcolors.Normalize(vmin=cmap_limits_dc[0], vmax=cmap_limits_dc[1])
    cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_dc, cmap=cmap3), cax=cax_dc_shared, orientation='horizontal')
    cb3.set_label('$\\delta c$ [km]', fontweight="bold")

    norm_dc_diff = mcolors.Normalize(vmin=cmap_limits_dc_diff[0], vmax=cmap_limits_dc_diff[1])
    cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_dc_diff, cmap=cmap2), cax=cax_dc_diff, orientation='horizontal')
    cb4.set_label('$\\delta c$ [km]', fontweight="bold")



    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle(f'Residual checks DSP and M3. lmax={LMAX_REF}, \n'
                 f'DSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M3 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M3 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M3 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + f'\nDSP & M3 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=0.83, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = ('Residual_checks_DSP_M3_lmax={LMAX_REF}_'
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M3=PlesaStrain14Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M3=PlesaStrain17Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + 'Tc={T_c/1e3}km'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    
    
# %% w-POWER SPECTRA COMPARISONS BETWEEN DSP AND M3
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,9))

    # solutions_w[lmax_run, rotation].plot_spectrum(ax=ax1, show=False, 
    #                 legend=('M3 coeffs'), plot_dict={'linestyle': linestyle})
    # w_coeffs_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_w_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # w_coeffs_DSP.coeffs[0,0,0] = 0
    # w_coeffs_DSP.plot_spectrum(ax=ax1, show=False, 
    #                 legend=('DSP coeffs'), plot_dict={'linestyle': '--'})

    # w_spectrum_diff = solutions_w[lmax_run, rotation].spectrum() - w_coeffs_DSP.spectrum()    
    # l = np.arange(0,(LMAX_REF+1))
    # ax2.plot(l, w_spectrum_diff, 
    #          label=('M3 - DSP'), 
    #          linestyle=linestyle)
    # plt.tight_layout()
    # plt.grid()
    # # ax1.set_ylim(1e-5)
    # ax2.set_xlim(0,44)
    # plt.show()
    

    
# %% PLOTS - STRESS AND STRAIN FIELDS

    lmax_stress_strain = 3*LMAX_REF

    w_clm = solutions_w[LMAX_REF, 0].expand(lmax=lmax_stress_strain).expand()
    F_clm = solutions_F[LMAX_REF, 0].expand(lmax=lmax_stress_strain).expand()

    # compute_Omega now returns Beuthe's Omega = Re*omega (required by
    # cons_disp_S)
    Omega_coeffs = compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, g0, R, T_e_0, lmax_stress_strain)
    Omega_grid = Omega_coeffs.expand(lmax=lmax_stress_strain)

    # S first (needed by the DSP-convention stress_fields), then stresses
    S_clm = cons_disp_S(w_clm, F_clm, Omega_grid, T_e_parent, a_clm, R, T_e_0, lmax_stress_strain)
    S_clm.coeffs[0,0,0] = 0
    w_clm.coeffs[0,0,0] = 0
    sigma_tt, sigma_pp, sigma_tp = stress_fields(S_clm, w_clm, T_e_parent, lmax_stress_strain, R, T_e_0)
    eps_tt, eps_pp, eps_tp = strain_fields(S_clm, w_clm, T_e_parent, lmax_stress_strain, R, T_e_0)

    eps_tt.data = eps_tt.data*1e3; eps_pp.data = eps_pp.data*1e3; eps_tp.data = eps_tp.data*1e3
    sigma_tt.data = sigma_tt.data*1e3; sigma_pp.data = sigma_pp.data*1e3; sigma_tp.data = sigma_tp.data*1e3

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



    (   min_strain,
        max_strain,
        sum_strain,
        principal_angle_strain,
    ) = Principal_strainstress_angle(-eps_tt.data, -eps_pp.data, -eps_tp.data)
    
    args_plot = dict(
        tick_interval=[45, 30],
        colorbar="bottom",
        cmap=cmc.vik,
        grid=True,
        # cb_tick_interval=1,
    )
    fig, ((ax1, ax2), 
          (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10), dpi=100)
    
    
    pysh.SHGrid.from_array(min_strain * 1e3).plot(
        ax=ax1,
        ticks="WSne",
        cb_label="Minimum principal horizontal strain ($\\times 10^{-3}$)",
        cmap_limits=[-4, 4],
        xlabel=None,
        **args_plot,
    )
    pysh.SHGrid.from_array(max_strain * 1e3).plot(
        ax=ax2,
        cb_label="Maximum principal horizontal strain ($\\times 10^{-3}$)",
        ticks="WSne",
        cmap_limits=[-4, 4],
        xlabel=None,
        **args_plot,
    )
    pysh.SHGrid.from_array(sum_strain * 1e3).plot(
        ax=ax3,
        cb_label="Sum of principal horizontal strains ($\\times 10^{-3}$)",
        cmap_limits=[-3, 3],
        ticks="WSne",
        xlabel=None,
        **args_plot,
    )
    pysh.SHGrid.from_array(principal_angle_strain).plot(
        ax=ax4,
        cb_label="Principal strain angle (°)",
        ticks="WSne",
        cmap_limits=[-90, 90],
        tick_interval=[45, 30],
        colorbar="bottom",
        cmap=cmc.vikO,
    )
    

    
    # Plot strain direction
    skip_i = int(LMAX_REF / 2)
    skip = (slice(None, None, skip_i), slice(None, None, skip_i))
    grid_long, grid_lat = np.meshgrid(
        pysh.SHGrid.from_array(principal_angle_strain).lons(),
        pysh.SHGrid.from_array(principal_angle_strain).lats(),
    )
    ones = np.ones(np.shape(principal_angle_strain))
    ax4.quiver(
        grid_long[skip],
        grid_lat[skip],
        ones[skip],
        ones[skip],
        scale=5e1,
        angles=principal_angle_strain[skip],
        color="g",
    )
    plt.suptitle('M3 - Principal strains', y=1.0)
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
## %% PLOT RESIDUAL STRAINS AND ANGLES BETWEEN DSP AND M3
    
    # sigma_tt, sigma_pp, sigma_tp = stress_fields(S_clm, w_clm, T_e_parent, lmax_stress_strain, R, T_e_0)
    # eps_tt, eps_pp, eps_tp = strain_fields(S_clm, w_clm, T_e_parent, lmax_stress_strain, R, T_e_0)

    # eps_tt.data = eps_tt.data*1e3; eps_pp.data = eps_pp.data*1e3; eps_tp.data = eps_tp.data*1e3
    # sigma_tt.data = sigma_tt.data*1e3; sigma_pp.data = sigma_pp.data*1e3; sigma_tp.data = sigma_tp.data*1e3
    
    # (   min_strain,
    #     max_strain,
    #     sum_strain,
    #     principal_angle_strain,
    # ) = Principal_strainstress_angle(-eps_tt.data, -eps_pp.data, -eps_tp.data)
    
    sum_strain_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_sumstrain_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    princ_angle_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_princ_angle_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # stress_theta_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_stress_theta_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # stress_phi_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_stress_phi_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # stress_thetaphi_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_stress_thetaphi_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')

    sum_strain = pysh.SHGrid.from_array(sum_strain * 1e3)
    sum_strain_residual = sum_strain_DSP.data - sum_strain.data
    princ_angle_residual = princ_angle_DSP.data - pysh.SHGrid.from_array(principal_angle_strain).data
    # stress_theta_residual = stress_theta_DSP.data - sigma_tt.data
    # stress_phi_residual = stress_phi_DSP.data - sigma_pp.data
    # stress_thetaphi_residual = stress_thetaphi_DSP.data - sigma_tp.data
    
    
    args_expand = dict(lmax=grid_expansion, lmax_calc=LMAX_REF)
    args_plot = dict(tick_interval=[45, 30], grid=True)

    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    outer_gs = gridspec.GridSpec(2, 1, hspace=-0.15) 



    # --- ROW 1: Sum principal strain ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
                                                 height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
    ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_strain_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_strain_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_strain = [-3,3]
    strain_min, strain_max = sum_strain_residual.min(), sum_strain_residual.max()
    cmap_limits_strain_diff =[-max(abs(strain_min), abs(strain_max)), max(abs(strain_min), abs(strain_max))]

    # grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    sum_strain_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_strain, 
                    cmap=cmap1, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Sum principal strain', fontweight="bold")
    
    sum_strain.plot(ax=ax2, 
                cmap_limits=cmap_limits_strain, 
                cmap=cmap1, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M3 - Sum principal strain', fontweight="bold")
    
    pysh.SHGrid.from_array(sum_strain_residual).plot(ax=ax3, cmap=cmap2, 
                                                     cmap_limits=cmap_limits_strain_diff,
                                                     colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Sum principal strain residual DSP - M3', fontweight="bold")

    norm_strain = mcolors.Normalize(vmin=cmap_limits_strain[0], vmax=cmap_limits_strain[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_strain, 
                    cmap=cmap1), 
                       cax=cax_strain_shared, orientation='horizontal')
    cb1.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")


    norm_strain_diff = mcolors.Normalize(vmin=cmap_limits_strain_diff[0], vmax=cmap_limits_strain_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(norm=norm_strain_diff, cmap=cmap2), cax=cax_strain_diff, orientation='horizontal')
    cb2.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")




    # --- ROW 2: Principal angle ---
    inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                 height_ratios=[1, 0.03], hspace=-0.35, wspace=0.15)
    ax4 = fig.add_subplot(inner_gs2[0, 0:2])
    ax5 = fig.add_subplot(inner_gs2[0, 2:4])
    ax6 = fig.add_subplot(inner_gs2[0, 4:6])
    
    cax_angle_shared = fig.add_subplot(inner_gs2[1, 1:3])
    cax_angle_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
    cmap_limits_angle = [-90,90]

    princ_angle_DSP.plot(ax=ax4, cmap=cmap1, cmap_limits=cmap_limits_angle, colorbar=None, ticks='WSen', **args_plot)
    ax4.set_title('DSP - Principal angle', fontweight="bold")
    
    pysh.SHGrid.from_array(principal_angle_strain).plot(ax=ax5, cmap=cmap1, cmap_limits=cmap_limits_angle, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax5.set_title('M3 - Principal angle', fontweight="bold")


    angle_min, angle_max = princ_angle_residual.min(), princ_angle_residual.max()
    pysh.SHGrid.from_array(princ_angle_residual).plot(ax=ax6, cmap=cmap2, cmap_limits=cmap_limits_angle, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax6.set_title('Principal angle residual DSP - M3', fontweight="bold")

    norm_angle = mcolors.Normalize(vmin=cmap_limits_angle[0], vmax=cmap_limits_angle[1])
    cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_angle, cmap=cmap1), cax=cax_angle_shared, orientation='horizontal')
    cb3.set_label('Principal angle [°]', fontweight="bold")

    norm_angle_diff = mcolors.Normalize(vmin=cmap_limits_angle[0], vmax=cmap_limits_angle[1])
    cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_angle_diff, cmap=cmap2), cax=cax_angle_diff, orientation='horizontal')
    cb4.set_label('Principal angle [°]', fontweight="bold")

    # Plot strain direction
    skip_i = int(LMAX_REF / 2)
    skip = (slice(None, None, skip_i), slice(None, None, skip_i))
    grid_long, grid_lat = np.meshgrid(
        pysh.SHGrid.from_array(principal_angle_strain).lons(),
        pysh.SHGrid.from_array(principal_angle_strain).lats(),
    )
    ones = np.ones(np.shape(principal_angle_strain))
    ax4.quiver(
        grid_long[skip],
        grid_lat[skip],
        ones[skip],
        ones[skip],
        scale=5e1,
        angles=princ_angle_DSP.data[skip],
        color="g",
    )
    ax5.quiver(
        grid_long[skip],
        grid_lat[skip],
        ones[skip],
        ones[skip],
        scale=5e1,
        angles=principal_angle_strain[skip],
        color="g",
    )
    # ax6.quiver(
    #     grid_long[skip],
    #     grid_lat[skip],
    #     ones[skip],
    #     ones[skip],
    #     scale=5e1,
    #     angles=princ_angle_residual[skip],
    #     color="orange",
    # )


    # # Add extensional tectonic features from Knapmeyer et al. (2006)
    # tecto_path = f"{os.getcwd()}/data"
    # Plt_tecto_Mars(tecto_path, ax=[ax4,ax5], compression=True, extension=False)



    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle(f'Residual strains DSP and M3. lmax={LMAX_REF}, \n'
                 f'DSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M3 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M3 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M3 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + f'\nDSP & M3 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=0.83, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = ('Residual_strains_DSP_M3_lmax={LMAX_REF}_'
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M3=PlesaStrain14Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M3=PlesaStrain17Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + 'Tc={T_c/1e3}km'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

        
    
    
    
    
# %% TRANSFER FUNCTION IMPULSE TEST (constant Te, requires strain=0 inputs loaded)
 
    # T_e_0_t = T_e_parent.coeffs[0,0,0]
    # lmax_t  = 20
    
    # def mono(val, lmax):
    #     c = pysh.SHCoeffs.from_zeros(lmax=lmax, normalization='4pi')
    #     c.coeffs[0,0,0] = val
    #     return c
    
    # def impulse(Rval, lmax, amp=None):
    #     c = mono(Rval, lmax)
    #     if amp is not None:
    #         for l in range(2, lmax+1):
    #             c.coeffs[0, l, 0] = amp
    #     return c
    
    # Te_mono = mono(T_e_0_t, 2*lmax_t)
    # D_mono  = mono(E*T_e_0_t**3/(12*(1-nu**2)), 2*lmax_t)
    # a_mono  = mono(1.0/(E*T_e_0_t), 2*lmax_t)
    # if ETA_FULL:
    #     Re_i   = R - T_e_0_t/2
    #     eta0_i = 1.0/(1.0 + T_e_0_t**2/(12.0*Re_i**2))
    #     D_eta_mono   = mono(eta0_i*E*T_e_0_t**3/(12*(1-nu**2)), 2*lmax_t)
    #     a_eta_mono   = mono(eta0_i/(E*T_e_0_t), 2*lmax_t)
    #     eta_mono     = mono(eta0_i, 2*lmax_t)
    # else:
    #     D_eta_mono = a_eta_mono = eta_mono = None
    # plan_t  = build_or_load_gaunt(lmax_t, nu)
    

    # # H channel: 1 m of topo at every (l,0); geoid = monopole only
    # w_H,_,_ = solve_beuthe(impulse(R, lmax_t, amp=1.0), impulse(R, lmax_t),
    #                        Te_mono, D_mono, a_mono, plan_t, lmax_t, R,
    #                        T_e_0_t, g0, mass, omega_on=True,
    #                        D_eta_clm=D_eta_mono, a_eta_clm=a_eta_mono,
    #                        eta_clm=eta_mono)
    # # G channel: 1 m of geoid at every (l,0); topo = monopole only
    # w_G,_,_ = solve_beuthe(impulse(R, lmax_t), impulse(R, lmax_t, amp=1.0),
    #                        Te_mono, D_mono, a_mono, plan_t, lmax_t, R,
    #                        T_e_0_t, g0, mass, omega_on=True,
    #                        D_eta_clm=D_eta_mono, a_eta_clm=a_eta_mono,
    #                        eta_clm=eta_mono)
    
    # TH_code = np.array([w_H.coeffs[0,l,0] for l in range(2, lmax_t+1)])
    # TG_code = np.array([w_G.coeffs[0,l,0] for l in range(2, lmax_t+1)])
    
    
    # # DSP reference transfer (Banerdt 5-eq system, dc-solved config, nmax=1)
    # rhobar_t = mass*3/(4*np.pi*R**3)
    # Re_t   = R - T_e_0_t/2
    # phi_t  = (R - T_c)/R
    # g_m_t  = g0*(1+(phi_t**3-1)*rho_c/rhobar_t)/phi_t**2
    # RTeR_t = (R - T_e_0_t)/R
    # gTe_t  = g0*(1+(RTeR_t**3-1)*rho_m/rhobar_t)/RTeR_t**2
    # alph_t = 1/(E*T_e_0_t); D0_t = E*T_e_0_t**3/(12*(1-nu**2))
    # eps_t  = 12*Re_t**2/T_e_0_t**2
    # beta_t = 1/(1+eps_t); eta_t = eps_t/(1+eps_t)
    # v1v_t  = nu/(1-nu); mx = T_e_0_t - T_c
    
    # def dsp_w(l, H, G):
    #     Dl=-l*(l+1.); Dlp=Dl+2.; Kl=3./(rhobar_t*(2*l+1.))
    #     A=np.zeros((5,5)); rhs=np.zeros(5)          # x=[w,Gc,q,omega,dc]
    #     A[0,0]= Kl*drho*phi_t**(l+2) + Kl*drhol
    #     A[0,4]=-Kl*drho*phi_t**(l+2)
    #     rhs[0]= G - Kl*rho_l*H
        
    #     A[1,0]= Kl*(g0/g_m_t)*drho*phi_t + Kl*(g0/g_m_t)*drhol*phi_t**l
    #     A[1,4]=-Kl*(g0/g_m_t)*drho*phi_t
    #     A[1,1]=-1
    #     rhs[1]=-Kl*(g0/g_m_t)*rho_l*phi_t**l*H
        
    #     A[2,0]= g_m_t*drho + g0*drhol
    #     A[2,4]=-g_m_t*drho
    #     A[2,1]=-g_m_t*drho
    #     A[2,2]=-1
    #     rhs[2]=-g0*rho_l*(H-G)
        
    #     A[3,0]= eta_t*D0_t*Dl*Dlp**2 + (Re_t**2/alph_t)*Dlp
    #     A[3,2]= Re_t**4*(Dlp-1-nu)
    #     A[3,3]=-Re_t**4*(beta_t*Dlp-1-nu)*Dl
        
    #     A[4,3]=-1
    #     A[4,0]=(rho_c*g_m_t*T_c + rho_m*gTe_t*mx)/R - v1v_t*drho*g_m_t*mx/R - drhol*g0*v1v_t*T_e_0_t/R
    #     A[4,4]= v1v_t*drho*g_m_t*mx/R
    #     rhs[4]=-v1v_t*rho_l*g0*T_e_0_t*H/R
        
    #     return np.linalg.solve(A, rhs)[0]
    
    # for i, l in enumerate(range(2, lmax_t+1)):
    #     thd, tgd = dsp_w(l,1,0), dsp_w(l,0,1)
    #     print(f"l={l:2d}  TH_code/TH_DSP-1 = {TH_code[i]/thd-1:+.3e}   "
    #           f"TG_code/TG_DSP-1 = {TG_code[i]/tgd-1:+.3e}")

 
 
 

    
    print(f'\nTotal model runtime: {(time.perf_counter() - t_begin):.1f}s')
    