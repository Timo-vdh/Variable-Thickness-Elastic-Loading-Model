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
    - Mantle density variations (no iterating corrections!)

Model 4 does not include:
    - Toroidal loading (V=0 & T=0)
    - Iterations for redistributions due to internal density variations
    - Iterations for finite amplitude corrections

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
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm as cmc
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(1, 'C:/Users/Timov/Displacement_strain_planet/Displacement_strain_planet')
from Displacement_strain_planet import Plt_tecto_Mars
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
T_c = 60.0e3                 # Arbitrary crustal thickness value, TBC
Te_input = 268.12e3

# Top and bottom depth of density variations drho_lm
Mt = 0
Mb = T_c


LMAX_RUNS  = [45]        # last entry is the reference resolution
LMAX_REF = max(LMAX_RUNS)
grid_expansion_res = LMAX_REF * 3

rotate_angles = (0.0, 0.0, 0.0)
lmax_Te_fit = LMAX_RUNS[-1]
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


SaveFigs = False
SavePath = "Plots/M4VarD_SPEC_FinalPlots"        # If on own laptop
# SavePath = "/home/vand_t1/Documents/Figures_M4"  # If on DLR computer
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
    subfolder_Te_maps = "Elastic_Thickness_Input_Maps"
    Te_filename = "grl58258-sup-0002-data_set_1.dat"
    Te_file_path = os.path.join(subfolder_Te_maps, Te_filename)
    df = pd.read_csv(Te_file_path, sep=r'\s+', comment='#',
                     header=None,
                     names=['longitude','latitude','crustal_thickness_km',
                            'heat_flow_mW_M4','Te_1e-14_km','Te_1e-17_km',
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
    Function first expands the parent Te map to a finer grid of 
    grid_expansion_res, which is then used to compute D and alpha coefficients. 
    D and alpha are then truncated to 2*lmax+1 because the coupling coefficients 
    contain degrees up to the sum of two input degrees (the sum over LM goes 
    from l-l' to l+l', i.e. 2*l).

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
    selection rule for orders (m1+m2+M4=0). Drastically reduced number of 
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
    
    path = os.path.join(CACHE_DIR, f"gaunt_plan_v4_lmax{lmax}_nu{nu:.4f}.npz")
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
    # corr1 = eta0*Re/R
    corr1 = Re/R        # ETA FIELD FIX
    
    # Laplacian array for degrees
    lap_by_degree = np.array([-l * (l + 1) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(l+2) for degrees l
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ (rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])
    
    
    # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------    
    T_e_parent_grid_eq1RHS = T_e_parent.expand(lmax=grid_expansion_res).data
    topo_grid_eq1RHS = topo_clm.expand(lmax=grid_expansion_res).data - R
    geoid_grid_eq1RHS = geoid_clm.expand(lmax=grid_expansion_res).data - R
    Te2_grid = T_e_parent_grid_eq1RHS**2    
    
    # max(Te - Tc, 0) field
    TeTc_grid = T_e_parent_grid_eq1RHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
     
    # pre-weighted topo  H' = H / phi^(l+2)
    Hp = pysh.SHGrid.from_array(topo_grid_eq1RHS).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]):
        Hp.coeffs[:, l, :] *= 1.0 / RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=grid_expansion_res).data
    
    # pre-weighted geoid  G' = rhobar(2l+1)/phi^(l+2) * G
    Gp = pysh.SHGrid.from_array(geoid_grid_eq1RHS).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]):
        Gp.coeffs[:, l, :] *= rhobar2l1[l] / RTcR_l2[l]
    Gp_grid = Gp.expand(lmax=grid_expansion_res).data
    
    
    # ------- THE FIELDS FOR EACH TERM -------
    # Field RHS 1a: Te*H grid
    TeH_grid = T_e_parent_grid_eq1RHS * topo_grid_eq1RHS
    TeH_clm = pysh.SHGrid.from_array(TeH_grid).expand()
    TeH_clm = pysh.SHCoeffs.from_array(TeH_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field RHS 1b: Te**2 * Laplacian(Te * topo)
    # (Laplacian on the INNER product)
    TeH_lap = TeH_clm.copy()
    for l in range(TeH_lap.coeffs.shape[1]):
        TeH_lap.coeffs[:, l, :] *= lap_by_degree[l]
    TeH_lap_grid = TeH_lap.expand(lmax=grid_expansion_res)
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
    dc2_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=grid_expansion_res).data).expand()
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
    dc4_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=grid_expansion_res).data).expand()
    dc4_clm = pysh.SHCoeffs.from_array(dc4_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    # ------- drho_lm VARIABLES AND FIELDS -------
    #  definitions of g_M, B_1 and B_2, and for the two fixes it encodes.)
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']
    g_M   = _dl['g_M']
    B_1   = _dl['B_1']
    Cp    = _dl['Cp']
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid_eq1RHS - Mt
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
        return weighted_coeffs.expand(lmax=grid_expansion_res).data
    
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
        lap_g = Te2_grid.data * lap.expand(lmax=grid_expansion_res).data
        lap_clm = pysh.SHGrid.from_array(lap_g).expand()
        lap_clm = pysh.SHCoeffs.from_array(lap_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
        return p_clm, lap_clm
    
    # degree-weights for two H-field terms
    wH_drho = np.array([-Cp[l] / B_1[l] for l in range(2*lmax+1)])
    # degree-weights for two G-field terms
    wG_drho = np.array([ 1.0 / B_1[l]   for l in range(2*lmax+1)])
    
    _topo_clm  = pysh.SHGrid.from_array(topo_grid_eq1RHS).expand()
    _geoid_clm = pysh.SHGrid.from_array(geoid_grid_eq1RHS).expand()
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
    # corr1 = eta0*Re/R
    corr1 = Re/R            # ETA FIELD FIX
    
    # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------   
    T_e_parent_grid_eq1LHS = T_e_parent.expand(lmax=grid_expansion_res).data
    
    # Field Te
    Te_grid = T_e_parent_grid_eq1LHS 
    Te_clm = pysh.SHGrid.from_array(Te_grid).expand()
    Te_clm = pysh.SHCoeffs.from_array(Te_clm.coeffs[:, :2*lmax+1, :2*lmax+1])  
    
    # Field Te^2 
    Te2_grid = T_e_parent_grid_eq1LHS**2 
    Te2_clm = pysh.SHGrid.from_array(Te2_grid).expand()
    Te2_clm = pysh.SHCoeffs.from_array(Te2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field max(Te-Tc,0)
    TeTc_grid = T_e_parent_grid_eq1LHS - T_c
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
    RTeR_grid = (R - T_e_parent_grid_eq1LHS) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid_eq1LHS <= T_c, rho_c, rho_m)
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2    
    
    # Field Tc if Tc < Te else 0
    Tcind_grid_1 = np.where(T_e_parent_grid_eq1LHS > T_c, T_c, 0.0)
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
    TeMt_grid  = T_e_parent_grid_eq1LHS - Mt
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
    T_e_parent_grid_eq2RHS = T_e_parent.expand(lmax=grid_expansion_res).data
    topo_grid_eq2RHS = topo_clm.expand(lmax=grid_expansion_res).data - R
    geoid_grid_eq2RHS = geoid_clm.expand(lmax=grid_expansion_res).data - R
    alpha_grid_eq2RHS = a_clm.expand(lmax=grid_expansion_res).data
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid_eq2RHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
    
    
    # Field RHS 2a: lap2 * Te*H*alpha grid
    TeHa_grid = T_e_parent_grid_eq2RHS * topo_grid_eq2RHS * alpha_grid_eq2RHS
    TeHa_clm = pysh.SHGrid.from_array(TeHa_grid).expand()
    TeHa_clm = pysh.SHCoeffs.from_array(TeHa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    # Perform multiplication with laplacian2, by multiplying it with 
    # the TeHa coefficients for the degrees l only
    TeHa_lap = TeHa_clm.copy()
    for l in range(TeHa_lap.coeffs.shape[1]):
        TeHa_lap.coeffs[:, l, :] *= lap2_by_degree[l]


    # (same H', G'; here each product also carries alpha and the Laplacian is +2)
    Hp = pysh.SHGrid.from_array(topo_grid_eq2RHS).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]): 
        Hp.coeffs[:, l, :] *= 1.0/RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=grid_expansion_res).data
    
    Gp = pysh.SHGrid.from_array(geoid_grid_eq2RHS).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]): 
        Gp.coeffs[:, l, :] *= 1/( rhobar2l1[l] * RTcR_l2[l] )
    Gp_grid = Gp.expand(lmax=grid_expansion_res).data
     
    d_dc1 = pysh.SHGrid.from_array(TeTc_grid * Hp_grid * alpha_grid_eq2RHS).expand()   # max*H'*alpha
    d_dc1 = pysh.SHCoeffs.from_array(d_dc1.coeffs[:, :2*lmax+1, :2*lmax+1])
    d_dc2 = pysh.SHGrid.from_array(TeTc_grid * Gp_grid * alpha_grid_eq2RHS).expand()   # max*G'*alpha
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
    TeMt_grid  = T_e_parent_grid_eq2RHS - Mt
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
        return c.expand(lmax=grid_expansion_res).data

    def _eq2_field(prod_grid):
        c = pysh.SHGrid.from_array(alpha_grid_eq2RHS * prod_grid).expand()
        c = pysh.SHCoeffs.from_array(c.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(c.coeffs.shape[1]):
            c.coeffs[:, l, :] *= lap2_by_degree[l]
        return c

    wH_d2 = np.array([-Cp[l]  / B_1[l] for l in range(2*lmax+1)])
    wG_d2 = np.array([ 1.0 / B_1[l]           for l in range(2*lmax+1)])
    _topo_c2  = pysh.SHGrid.from_array(topo_grid_eq2RHS).expand()
    _geoid_c2 = pysh.SHGrid.from_array(geoid_grid_eq2RHS).expand()
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
    T_e_parent_grid_eq2LHS = T_e_parent.expand(lmax=grid_expansion_res).data
    a_grid_eq2LHS = a_clm.expand(lmax=grid_expansion_res).data
    # gTe FIELD (variable-Te fix): gravity at the LOCAL shell-base depth,
    # mantle branch only -- every gTe-carrying term also carries max(Te-Tc,0),
    # which is zero exactly where the density branch would switch. Monopole at
    # constant Te => benchmark preserved.
    RTeR_grid = (R - T_e_parent_grid_eq2LHS) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid_eq2LHS <= T_c, rho_c, rho_m)
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid_eq2LHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data


    # Field 2a: Te * alpha
    Tea_grid = T_e_parent_grid_eq2LHS * a_grid_eq2LHS
    Tea_clm = pysh.SHGrid.from_array(Tea_grid).expand()
    Tea_clm = pysh.SHCoeffs.from_array(Tea_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 2b: alpha
    a_clm_copy = a_clm.copy()
    a_clm_copy.coeffs = a_clm_copy.coeffs[:, :2*lmax+1, :2*lmax+1]
    Tcind_grid_2 = np.where(T_e_parent_grid_eq2LHS > T_c, T_c, 0.0)
    Tcinda_clm   = pysh.SHGrid.from_array(Tcind_grid_2 * a_grid_eq2LHS.data).expand()
    Tcinda_clm   = pysh.SHCoeffs.from_array(Tcinda_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 2c: max(Te-Tc,0) * alpha
    gTeTeTca_grid = gTe_grid * TeTc_grid * a_grid_eq2LHS  # gTe grid folded into here for variable Te
    gTeTeTca_clm  = pysh.SHGrid.from_array(gTeTeTca_grid).expand()
    gTeTeTca_clm  = pysh.SHCoeffs.from_array(gTeTeTca_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    

    # ------- drho_lm VARIABLES AND FIELDS -------
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']
    g_M = _dl['g_M']
    
    # Te - Mt field
    TeMt_grid  = T_e_parent_grid_eq2LHS - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)

    # 2d's field is max*alpha (NOT gTe*max*alpha -- gTe belongs to 2c only)
    TeTca_grid = pysh.SHGrid.from_array(TeTc_grid * a_grid_eq2LHS).expand()
    TeTca_clm = pysh.SHCoeffs.from_array(TeTca_grid.coeffs[:, :2*lmax+1, :2*lmax+1])

    # drho branch: omega's drhom term  P_hat*drhom/R  contributes the
    # w-coupling  P_hat * Dw  (Dw per-degree, P_hat a field) -> supplied to
    # solve_beuthe as a separate (field, diagonal) pair, since the operand
    # weight cannot be folded into a single convolution field.
    Phata_clm = pysh.SHGrid.from_array( MTeMt * TeMt0 * a_grid_eq2LHS ).expand()
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

    # ---- degree 1: DSP enforces the COM constraint Gc_1 = 0 (thinshell.py eq(2),
    # "Force the degree-1 geoid to zero"), so the Gc-elimination terms must be
    # dropped here. Harmless to the solve (solve_beuthe zeroes the l=0,1 rows of rhs),
    # but does affect compute_omega and the stress-strain calculations
    H1 = topo_clm_copyq.coeffs[:, 1, :2]
    G1 = geoid_clm_copyq.coeffs[:, 1, :2]

    if solve_for == 'dc_lm':
        dc1      = (rho_l * H1 - rhobar * G1) / (drho * RTcR**3)   # eq(1), w_1 = 0
        q_phys_1 = g0 * rho_l * (H1 - G1) - g_m * drho * dc1        # eq(3), Gc_1 = w_1 = 0
        q_coeffs[:, 1, :2] = -Re**4 * Kalousova_scaler1 * q_phys_1
        
    if solve_for == 'drho_lm':
        q_phys_1 =  (g0*rho_l*(H1-G1)   + q_topo_drho3 * field_topo_drho3[:, 1, :2]
                        + q_geoid_drho4 * field_geoid_drho4[:, 1, :2])        
        q_coeffs[:, 1, :2] = -Re**4 * Kalousova_scaler1 * q_phys_1
        
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



# %% FINAL OMEGA, dc, drho AND Gc EQUATIONS (COMPUTED AFTER w_lm IS KNOWN)

def compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, q_clm, g0, R, T_e_0, lmax_calc, lmax_grid):
    """
    Equation for tangential loading potential Omega, following the definition
    as given in Broquet & Andrews-Hanna (2022), which is derived from Banerdt
    (1986). 
    
    In this M4, this equation has been rewritten into w-terms in order
    to maintain a 2Nx2N block matrix system, neglecting effects of crustal 
    thickness variations dc and mantle density variations dm. The solution for 
    Omega itself can therefore be obtained using the result for w_lm.
    """
    
    R_e = R - T_e_0/2
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    # Grids
    w_grid_copyOmega = w_clm.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    T_e_parent_grid_copyOmega = T_e_parent.copy().expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    topo_grid_copyOmega = topo_clm.copy().expand(lmax=lmax_grid, lmax_calc=lmax_calc).data - R
    geoid_grid_copyOmega = geoid_clm.copy().expand(lmax=lmax_grid, lmax_calc=lmax_calc).data - R
    # Te - Tc field
    TeTc_grid = T_e_parent_grid_copyOmega - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
 
 
    # gravity at the elastic base (depth Te) for the mantle column term
    RTeR_grid = (R - T_e_parent_grid_copyOmega) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid_copyOmega <= T_c, rho_c, rho_m)
    
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2    
    
    TeH_grid = T_e_parent_grid_copyOmega * topo_grid_copyOmega
 
    # FIX (operator ordering): apply the per-degree dc-elimination weights to
    # H and G FIRST, then multiply by the max(Te-Tc,0) grid -- this is the 
    # ordering consistent with the per-degree elimination and with 
    # Omega_eq1_RHS in the solver (weight-then-multiply).
    # Previously the weights were applied to the coefficients of the PRODUCT
    # (TeTc*H), which differs for laterally varying Te.
    Hp_coeffs = pysh.SHGrid.from_array(topo_grid_copyOmega).expand()
    Hp_coeffs = truncate(Hp_coeffs, lmax=lmax_calc)
    for l in range(Hp_coeffs.coeffs.shape[1]):
        Hp_coeffs.coeffs[:, l, :] *= 1/RTcR**(l+2)
    Hp_grid = Hp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    TeTcHp_grid = TeTc_grid * Hp_grid
 
    Gp_coeffs = pysh.SHGrid.from_array(geoid_grid_copyOmega).expand()
    Gp_coeffs = truncate(Gp_coeffs, lmax=lmax_calc)
    for l in range(Gp_coeffs.coeffs.shape[1]):
        Gp_coeffs.coeffs[:, l, :] *= rhobar*(2*l+1)/(3 * RTcR**(l+2))
    Gp_grid = Gp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    TeTcGp_grid = TeTc_grid * Gp_grid
    
    
    Tcind_grid_o = np.where(T_e_parent_grid_copyOmega > T_c, T_c, 0.0)   # Te<Tc FIX
 
    # Compute Re*Omega_lm as the term Omega_lm (required in conversion between
    # Banerdt and Beuthe's formulations).  w-coefficient corrected to match the
    # solve: surface -> drhol*g0 (vanishes for rho_l=rho_c), mantle -> gTe.
    term_1 = nu/(1-nu)*rho_l*g0*TeH_grid
    term_2 = + nu/(1-nu)*g_m*rho_l * TeTcHp_grid
    term_3 = -drhol*g0*nu/(1-nu)*T_e_parent_grid_copyOmega *w_grid_copyOmega
    
    term_4 = rho_c*g_m*Tcind_grid_o *w_grid_copyOmega
    term_5 = rho_m*gTe_grid*TeTc_grid *w_grid_copyOmega  # gTe field instead of scalar
    
    # drhol EXTENSION (zero if rho_l == rho_c): residual w-piece of the
    # (dc-w) substitution, + v1v*g_m*drhol*max(Te-Tc,0)*P_l*w  with the
    # weight applied to w FIRST (weight-then-multiply, as in the solver).
    wp_coeffs = pysh.SHGrid.from_array(w_grid_copyOmega).expand()
    wp_coeffs = truncate(wp_coeffs, lmax=lmax_calc)
    for l in range(wp_coeffs.coeffs.shape[1]):
        wp_coeffs.coeffs[:, l, :] *= 1/RTcR**(l+2)
    wp_grid = wp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    term_6 = + nu/(1-nu)*g_m*drhol * TeTc_grid * wp_grid
    term_7 = - nu/(1-nu)*g_m * TeTcGp_grid
 
    # =====================================================================
    # BRANCH GATING. This function was previously dc-branch ONLY -- it had no
    # solve_for anywhere -- so every stress/strain field produced in the
    # drho_lm configuration was wrong. (w was unaffected: compute_Omega is
    # post-processing and never feeds the solver.)
    #
    # Terms 2, 6, 7 are artefacts of the dc-ELIMINATION: they carry the
    # phi^-(l+2)-weighted H' and G' that replace dc. With dc = 0 they do not
    # exist. The drho_lm branch has instead, straight from DSP eq (5):
    #   term_8: the -w half of  v1v*drho*g_m*max*(dc - w)/R  now STANDS ALONE
    #           (in the dc branch the +drho*w hidden in drho*dc cancels it,
    #           leaving only the small drhol piece that is term_6);
    #   term_9: + P_hat * drho_m, with drho_m from compute_drho(w, H, G) --
    #           no elimination is needed here because w is already known.
    # =====================================================================
    if solve_for == 'dc_lm':
        term_8 = 0.0
        term_9 = 0.0
    else:
        term_2 = 0.0          # dc-elimination artefacts: absent when dc = 0
        term_6 = 0.0
        term_7 = 0.0
        term_8 = - nu/(1-nu)*drho*g_m * TeTc_grid * w_grid_copyOmega
        _dl_o   = drho_layer(lmax_grid, R, g0, mass)
        TeMt_o  = T_e_parent_grid_copyOmega - Mt
        TeMt0_o = np.where(TeMt_o > 0.0, TeMt_o, 0.0)         # max(Te-Mt, 0)
        Phat_o  = (-0.5 * nu/(1-nu) * _dl_o['g_M'] * TeMt0_o
                   * np.minimum(TeMt_o, _dl_o['M']))          # min(M, Te-Mt)
        drho_m_grid = compute_drho(w_clm, topo_clm, geoid_clm, R, 
                            lmax_calc=lmax_calc, 
                            lmax_grid=lmax_grid).expand(lmax=lmax_grid).data
        term_9 = Phat_o * drho_m_grid
    
    
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
                       + term_7
                       + term_8
                       + term_9) * (Re / R)
        
    Omega_grid = pysh.SHGrid.from_array(Omega_grid_data)
    Omega_clm = Omega_grid.expand()
    Omega_clm = truncate(Omega_clm, lmax=lmax_calc)
    
    # Correctly set the degree 1 Omega coefficients
    Omega_clm.coeffs[:, 1, :2] = (E * T_e_0**3 / (2.0 * R_e**3)) * q_clm.coeffs[:, 1, :2]    # TODO: CHeck if this is the correct term that needs fixing to align DSP and M4
    return Omega_clm

def compute_dc(w_clm, topo_clm, geoid_clm, R, lmax_calc, lmax_grid):
    """
    Compute the crustal root variations ('bottom loads') dc_lm using the 
    rewritten equation of Gc_lm with drho_lm=0. 
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    L_comp = min(w_clm.lmax, topo_clm.lmax, geoid_clm.lmax, lmax_calc)

    topo_clm_copydc = topo_clm.copy()
    topo_clm_copydc.coeffs[0,0,0] = 0
    topo_clm_copydc = truncate(topo_clm_copydc, L_comp)

    geoid_clm_copydc = geoid_clm.copy()
    geoid_clm_copydc.coeffs[0,0,0] = 0
    geoid_clm_copydc = truncate(geoid_clm_copydc, L_comp)

    w_clm_copydc = w_clm.copy()
    w_clm_copydc.coeffs[0,0,0] = 0
    w_clm_copydc = truncate(w_clm_copydc, L_comp)
    
    for l in range(geoid_clm_copydc.coeffs.shape[1]):
       geoid_clm_copydc.coeffs[:, l, :] *= rhobar*(2*l+1)/3
    
    dc_clm = 1/drho * (rho_l*topo_clm_copydc + drhol*w_clm_copydc - geoid_clm_copydc)
        
    for l in range(dc_clm.coeffs.shape[1]):
       dc_clm.coeffs[:, l, :] *= 1/(RTcR**(l+2)) 
        
    dc_clm = dc_clm + w_clm_copydc
    
    return dc_clm


def compute_drho(w_clm, topo_clm, geoid_clm, R, lmax_calc, lmax_grid):
    """
    Compute the mantle density variations ('bottom loads') drho_lm using the 
    rewritten equation of Gc_lm with dc_lm=0. 
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    L_comp = min(w_clm.lmax, topo_clm.lmax, geoid_clm.lmax, lmax_calc)

    topo_clm_copydrho = topo_clm.copy()
    topo_clm_copydrho.coeffs[0,0,0] = 0
    topo_clm_copydrho = truncate(topo_clm_copydrho, L_comp)

    geoid_clm_copydrho = geoid_clm.copy()
    geoid_clm_copydrho.coeffs[0,0,0] = 0
    geoid_clm_copydrho = truncate(geoid_clm_copydrho, L_comp)

    w_clm_copydrho = w_clm.copy()
    w_clm_copydrho.coeffs[0,0,0] = 0
    w_clm_copydrho = truncate(w_clm_copydrho, L_comp)
 
    # ------- drho_lm VARIABLES AND FIELDS -------
    RMt = R - Mt
    RMb = R - Mb
    RMtR = RMt / R
    RMbR = RMb / R
    
    RMtR_l3   = np.array([RMtR**(l+3) for l in range(2 * lmax_grid + 1)])
    RMbR_l3   = np.array([RMbR**(l+3) for l in range(2 * lmax_grid + 1)])
    Rl3       = np.array([R/(l+3) for l in range(2 * lmax_grid + 1)])
    Cp        = np.array([3/(rhobar*(2*l+1)) for l in range(2 * lmax_grid + 1)])
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax_grid + 1)])
    
    B_1 = Cp * Rl3 * ( RMtR_l3 - RMbR_l3 )
 
    # FIXES (this function produced a sign-INVERTED drho map while w was
    # unaffected -- it is pure post-processing and does not feed the solver):
    # Banerdt/DSP eq (1) with dc = 0 reads
    #     G = Cbar*( rho_l*H + drhol*w + drho*phi^(l+2)*w ) + B_1*drhom
    # so, solving for drhom,
    #     drhom = [ G - Cbar*rho_l*H - Cbar*drhol*w - Cbar*drho*phi^(l+2)*w ] / B_1
    #   (1) the H term must be NEGATIVE and carry rho_l (the surface load
    #       density), not rho_c -- it was +rho_c;
    #   (2) the w (moho) term must be NEGATIVE -- it was +;
    #   (3) the drhol*w term was missing entirely (zero iff rho_l == rho_c,
    #       but required for distinct load densities);
    #   (4) the G term was already correct (+G/B_1) -- which is why the map
    #       came out only ROUGHLY inverted rather than exactly.
    topo_term  = topo_clm_copydrho * (-rho_l)
    w_term_1   = w_clm_copydrho * (-drho)          # moho term, weight phi^(l+2)
    w_term_2   = w_clm_copydrho * (-drhol)         # load-density term, no weight
    geoid_term = geoid_clm_copydrho.copy()
    for l in range(topo_term.coeffs.shape[1]):
        topo_term.coeffs[:, l, :]  *= Cp[l] / B_1[l]
        w_term_1.coeffs[:, l, :]   *= Cp[l] * RTcR_l2[l] / B_1[l]
        w_term_2.coeffs[:, l, :]   *= Cp[l] / B_1[l]
        geoid_term.coeffs[:, l, :] *= 1/B_1[l]
 
    drho_clm = topo_term + w_term_1 + w_term_2 + geoid_term
 
    return drho_clm



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

    Unused in current code; no direct application/reference parameter.
    """
    rhobar = mass * 3.0 / (4.0*np.pi) / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2

    # FIX (degree-0 bookkeeping): strip the reference-radius monopole locally.
    topo_clm_copyGc = topo_clm.copy()
    topo_clm_copyGc.coeffs[0,0,0] = 0

    w_clm_copyGc = w_clm.copy()
    dc_clm_copyGc = dc_clm.copy()

    wmdc = w_clm_copyGc - dc_clm_copyGc                        # (w - dc)

    H_term = (rho_l*topo_clm_copyGc + drhol*w_clm_copyGc) # (rho_l*H + drhol*w) * phi^l
    for l in range(H_term.coeffs.shape[1]):
        H_term.coeffs[:, l, :] *= RTcR**l

    wmdc_term = wmdc.copy()                      # drho * (w-dc) * phi^1
    wmdc_term.coeffs *= drho * RTcR              # degree-independent -> scalar

    Gc_clm = H_term + wmdc_term                  # bracket
    for l in range(Gc_clm.coeffs.shape[1]):      # times (g0/g_m)*3/(rhobar(2l+1))
        Gc_clm.coeffs[:, l, :] *= (g0/g_m) * 3.0/(rhobar*(2*l+1))

    Gc_grid = Gc_clm.expand(lmax=lmax)
    return Gc_grid, Gc_clm


# %% STRESS AND STRAIN FIELDS - OLD, MISALIGNED WITH DSP


# def O1(SH_function, lmax):
#     """ Beuthe (2008)'s differential operator O_1 in 2D spherical geometry. """
#     SH_function_grid = SH_function.expand(lmax=lmax)
#     dtheta_grid = SH_function.gradient(lmax=lmax).theta
#     dtheta_sh = dtheta_grid.expand()
#     dtheta2_grid = dtheta_sh.gradient(lmax=lmax).theta
#     return dtheta2_grid.data + SH_function_grid.data

# def O2(SH_function, lmax):
#     """ Beuthe (2008)'s differential operator O_2 in 2D spherical geometry. """
#     theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1)+1, endpoint=True))

#     cot_theta = np.divide( 1.0, np.tan(theta_range), 
#                           out=np.zeros_like(np.tan(theta_range)), 
#                           where=np.tan(theta_range) != 0)
#     cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

#     sin_theta = np.sin(theta_range)
#     sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

#     csc_theta = np.divide( 1.0, np.sin(theta_range), 
#                           out=np.zeros_like(np.sin(theta_range)), 
#                           where=np.sin(theta_range) != 0)
#     csc_theta_grid = np.tile(csc_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

#     SH_function_grid = SH_function.expand(lmax=lmax)
#     dtheta_grid = SH_function.gradient(lmax=lmax).theta

#     dphi_grid = SH_function.gradient(lmax=lmax).phi
#     dphi_grid.data *= sin_theta_grid
#     dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
#     dphi2_grid = dphi_sh.gradient(lmax=lmax).phi
#     dphi2_grid.data *= csc_theta_grid

#     return dphi2_grid.data + cot_theta_grid * dtheta_grid.data + SH_function_grid.data

# def O3(SH_function, lmax):
#     """ Beuthe (2008)'s differential operator O_3 in 2D spherical geometry. """
#     theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1)+1, endpoint=True))

#     cot_theta = np.divide( 1.0, np.tan(theta_range), 
#                           out=np.zeros_like(np.tan(theta_range)), 
#                           where=np.tan(theta_range) != 0)
#     cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

#     sin_theta = np.sin(theta_range)
#     sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

#     csc_theta = np.divide( 1.0, np.sin(theta_range), 
#                           out=np.zeros_like(np.sin(theta_range)), 
#                           where=np.sin(theta_range) != 0)
#     csc_theta_grid = np.tile(csc_theta.reshape(-1, 1), (1, 4*(lmax+1)+1))

#     dphi_grid = SH_function.gradient(lmax=lmax).phi
#     dphi_grid.data *= sin_theta_grid

#     dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
#     dthetaphi_grid = dphi_sh.gradient(lmax=lmax).theta

#     return (csc_theta_grid * dthetaphi_grid.data 
#             - cot_theta_grid * csc_theta_grid * dphi_grid.data)



# def stress_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
#     """
#     Stresses in the DSP/Banerdt convention (Banerdt 1986 eqs A12-A14, as in
#     DSP's compute_strains): plane-stress Hooke's law applied to membrane +
#     bending strains built from the tangential potential S (== DSP's A_lm)
#     and w, with 1/R kernels and the thin-shell top-fiber factor
#     eps_f = (Te/2 - depth)/(1 + (Te/2 - depth)/R).
#     Returns stresses in MPa (matching DSP).

#     NOTE: this replaces the previous Beuthe eq-(73) stress-function form
#     (kept below as stress_fields_beuthe73), which evaluates the top-fiber
#     stress with exact z/(Re+z) curvature factors and 1/Re kernels. The two
#     differ by O(Te/R) factors (~4-7% for Te=268 km) -- for benchmarking
#     against DSP the convention must match DSP.
#     """
#     O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
#     O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
#     S_grid  = S_sol.expand(lmax=lmax)
#     w_grid  = w_sol.expand(lmax=lmax)
#     Te_grid = T_e_parent.expand(lmax=lmax)

#     # membrane strains (Banerdt A16-A18 with S in place of A)
#     eps_t    = 1/R * (O1S - S_grid.data + w_grid.data)
#     eps_p    = 1/R * (O2S - S_grid.data + w_grid.data)
#     omega_sh = 1/R * (2*O3S)                       # engineering shear
#     # bending strains (A19-A21); note O1w = d2w/dth2 + w, so
#     # kappa_t = -(d2w/dth2)/R^2 - w/R^2 = -O1w/R^2 (and analogously kappa_p)
#     kappa_t = -1/R**2 * O1w
#     kappa_p = -1/R**2 * O2w
#     tau     = -2/R**2 * O3w

#     zeta  = Te_grid.data/2.0 - depth
#     eps_f = zeta / (1.0 + zeta/R)
#     DpsiTeR = E/(1.0 - nu**2)

#     sigma_tt = (eps_t + nu*eps_p + eps_f*(kappa_t + nu*kappa_p)) * DpsiTeR / 1e6
#     sigma_pp = (eps_p + nu*eps_t + eps_f*(kappa_p + nu*kappa_t)) * DpsiTeR / 1e6
#     sigma_tp = (omega_sh + eps_f*tau) * 0.5 * DpsiTeR * (1.0 - nu) / 1e6

#     return (pysh.SHGrid.from_array(sigma_tt),
#             pysh.SHGrid.from_array(sigma_pp),
#             pysh.SHGrid.from_array(sigma_tp))


# def stress_fields_beuthe73(w_sol, F_sol, Omega_grid, T_e_parent, R, T_e_0, lmax):
#     """
#     Beuthe eqs (73). *** DO NOT USE FOR PRODUCTION YET -- SEE BELOW. ***
 
#     Expects Omega_grid = Beuthe's Omega = Re*omega (the output of the
#     corrected compute_Omega).
 
#     STATUS (measured, not assumed). Compared against stress_fields() -- the
#     Banerdt/DSP Hooke form -- on the same (w, F, S, Omega) at three Te:
 
#         Te/R      rms ratio sigma_tt   rms ratio sigma_pp
#         0.0791          1.149                1.513
#         0.0181          1.195                1.234
#         0.0059          0.737                0.842
 
#     Both are FIRST-ORDER thin-shell forms of the SAME theory, so any genuine
#     convention difference is O(Te/R) and MUST vanish as Te/R -> 0. These
#     ratios do not converge to 1 -- they get worse. That is the signature of an
#     ERROR, not a convention offset. (An earlier note in this file claimed a
#     "4-7% convention difference"; that claim was wrong and is retracted.)
 
#     What was checked and is FINE:
#       * the bending structure. Banerdt uses (O1w + nu*O2w); this uses
#         (Delta'w - (1-nu)*O2w). Since O1w + O2w = Delta'w these are identical.
#       * the bending prefactor, to O(Te/R): Banerdt has
#         -E*zeta/((1-nu^2)*R*(R+zeta)); the -zeta/(Re+zeta) piece here matches
#         it with R -> Re. The extra +beta/Re piece is ~1.4% of it.
 
#     Known suspects, in order:
#       1. xi is built here as 12*R^2/Te^2 but as 12*Re^2/Te^2 everywhere else
#         (cons_disp_S, the solver, the eta fields). Inconsistent -- but only a
#         ~5% effect, so it cannot be the whole story.
#       2. the membrane term  eta/Te * (O2F + Omega): the eta prefactor and the
#         absence of any 1/Re^2-type kernel on O2F need checking against the
#         paper. This is where the residual 20-50% must live.
 
#     Re-deriving Beuthe eq (73) requires the paper and is a paper-level task
#     (standing division of labour). Until that is done, stress_fields() -- the
#     Banerdt/DSP form -- is the one to use: it is a direct Hooke evaluation
#     from S and w, both of which are now validated to ~1e-12 against the
#     reference (see the impulse test and compute_Omega's check).
#     """
#     # Laplacian array for degrees
#     lap2_by_degree = np.array([(-l * (l + 1) + 2) for l in range(2 * lmax + 1)])
#     w_lap2 = w_sol.copy()
#     for l in range(w_lap2.coeffs.shape[1]):
#         w_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
#     w_lap2_grid = w_lap2.expand(lmax=lmax)
    
#     Te_grid = T_e_parent.expand(lmax=lmax)
    
#     O1F = O1(F_sol, lmax); O2F = O2(F_sol, lmax); O3F = O3(F_sol, lmax)
#     O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax) 
    
#     xi = 12*R**2/Te_grid.data**2
#     eta = xi/(1+xi)
#     zeta = Te_grid.data/2
#     Re = R - T_e_0/2
    
#     sigma_tt = (eta/Te_grid.data * (O2F + Omega_grid.data) 
#                 + E/(Re*(1-nu**2))*(eta/xi - zeta/(Re+zeta)) * (w_lap2_grid.data - (1-nu)*O2w)
#                 )
#     sigma_pp = (eta/Te_grid.data * (O1F + Omega_grid.data) 
#                 + E/(Re*(1-nu**2))*(eta/xi - zeta/(Re+zeta)) * (w_lap2_grid.data - (1-nu)*O1w)
#                 )
#     sigma_tp = (eta/Te_grid.data * -O3F 
#                 + E/(Re*(1+nu))*(eta/xi - zeta/(Re+zeta)) *O3w
#                 )
    
#     sigma_tt = pysh.SHGrid.from_array(sigma_tt)
#     sigma_pp = pysh.SHGrid.from_array(sigma_pp)
#     sigma_tp = pysh.SHGrid.from_array(sigma_tp)
    
#     return sigma_tt, sigma_pp, sigma_tp


# def strain_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
#     """
#     Total strains in the DSP/Banerdt convention (membrane + top-fiber
#     bending), matching DSP's tot_theta / tot_phi / tot_thetaphi:
#         tot = eps + eps_f*kappa,  eps_f = (Te/2-depth)/(1+(Te/2-depth)/R)
    
#     MISSING THE TOROIDAL DISPLACEMENT POTENTIAL T TERMS!
#     """
#     # Return diff operator applied S and w terms, in grid.data format
#     O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
#     O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    
#     S_grid  = S_sol.expand(lmax=lmax)
#     w_grid  = w_sol.expand(lmax=lmax)
#     Te_grid = T_e_parent.expand(lmax=lmax)

#     eps_t    = 1/R * (O1S - S_grid.data + w_grid.data)    
#     eps_p    = 1/R * (O2S - S_grid.data + w_grid.data)
#     gamma_tp = 1/R * (2*O3S)
    
#     kappa_t = -1/R**2 * O1w
#     kappa_p = -1/R**2 * O2w
#     tau     = -2/R**2 * O3w
    
#     zeta = Te_grid.data/2.0 - depth
#     tot_strain_pref = zeta / (1.0 + zeta/R)
    
#     tot_eps_tt = eps_t    + tot_strain_pref*kappa_t
#     tot_eps_pp = eps_p    + tot_strain_pref*kappa_p
#     tot_eps_tp = (gamma_tp + tot_strain_pref*tau)/2.0
    
#     tot_eps_tt = pysh.SHGrid.from_array(tot_eps_tt)
#     tot_eps_pp = pysh.SHGrid.from_array(tot_eps_pp)
#     tot_eps_tp = pysh.SHGrid.from_array(tot_eps_tp)
    
#     return tot_eps_tt, tot_eps_pp, tot_eps_tp


# def cons_disp_S(w_sol, F_sol, Omega_grid, T_e_parent, a_clm, R, T_e_0, lmax_calc, lmax_grid):
#     """ 
#     Beuthe (2008)'s consoidal/poloidal tangential displacement potential S_lm 
#     (A_lm in DSP/Banerdt (1986)). Used in computations of strain.
#     """
    
#     lap_by_degree = np.array([(-l * (l + 1)) for l in range(2 * lmax_grid + 1)])
#     lap2_by_degree = np.array([(-l * (l + 1) + 2) for l in range(2 * lmax_grid + 1)])
#     F_lap2 = F_sol.copy()
#     w_lap2 = w_sol.copy()
#     for l in range(F_lap2.coeffs.shape[1]):
#         F_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
#         w_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
#     F_lap2_grid = F_lap2.expand(lmax=lmax_grid)
#     w_lap2_grid = w_lap2.expand(lmax=lmax_grid)
    
#     w_grid = w_sol.expand(lmax=lmax_grid)
#     a_grid = a_clm.expand(lmax=lmax_grid)
#     Te_grid = T_e_parent.expand(lmax=lmax_grid)

#     Re = R - T_e_0/2
#     xi = 12*Re**2/Te_grid.data**2
#     eta = xi/(1+xi)
    
#     lapl_S_grid = (Re*eta*a_grid.data*(1-nu)*(F_lap2_grid.data + 2*Omega_grid.data) 
#               + eta/xi * w_lap2_grid.data  
#               - 2*w_grid.data)
    
#     lapl_S_lm = pysh.SHGrid.from_array(lapl_S_grid).expand()
     
#     S_lm = lapl_S_lm.copy()
#     for l in range(1, S_lm.coeffs.shape[1]):
#         S_lm.coeffs[:, l, :] /= lap_by_degree[l]
#     S_lm.coeffs[0, 0, 0] = 0.0  
    
#     S_lm = truncate(S_lm, lmax=lmax_calc)
        
#     return S_lm


# def Principal_strainstress_angle(s_theta, s_phi, s_theta_phi):
#     """
#     Calculate principal strains, stresses, and
#     their principal angles.

#     Returns
#     -------
#     min_strain : array, size same as input arrays
#         Array with the minimum principal horizontal strain or stress.
#     max_strain : array, size same as input arrays
#         Array with the maximum principal horizontal strain or stress.
#     sum_strain : array, size same as input arrays
#         Array with the sum of the principal horizontal strain or stress.
#     principal_angle : array, size same as input arrays
#         Array with the principal strain or stress direction in degrees.

#     Parameters
#     ----------
#     s_theta : array, float, size(nlat, nlon)
#         Array of the colatitude component of the stress or strain field.
#     s_phi : array, float, size(nlat, nlon)
#         Array of the longitude component of the stress or strain field.
#     s_theta_phi : array, float, size(nlat, nlon)
#         Array of the colatitude and longitude component of the stress or strain field.
#     """

#     min_strain = 0.5 * (
#         (s_theta + s_phi) - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
#     )
#     max_strain = 0.5 * (
#         (s_theta + s_phi) + np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
#     )
#     sum_strain = min_strain + max_strain
#     principal_angle = 0.5 * np.arctan2(2 * s_theta_phi, s_theta - s_phi) * 180.0 / np.pi

#     return min_strain, max_strain, sum_strain, principal_angle



# %% STRESS AND STRAIN FIELDS - WITH CHANGES TO ALIGN WITH DSP!

kw_exp_grad = {"extend": False, "lmax_calc": LMAX_REF, "lmax": grid_expansion_res, "grid": "DH2"}
kw_exp_S = {"lmax_calc": LMAX_REF, "lmax": grid_expansion_res, "grid": "DH2"}


def O1(SH_function, lmax):
    """ Beuthe (2008)'s differential operator O_1 in 2D spherical geometry. """
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1), endpoint=False))

    cot_theta = np.divide( 1.0, np.tan(theta_range), 
                          out=np.zeros_like(np.tan(theta_range)), 
                          where=np.tan(theta_range) != 0)
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    sin_theta = np.sin(theta_range)
    sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    csc2_theta = np.divide( 1.0, (np.sin(theta_range))**2, 
                          out=np.zeros_like((np.sin(theta_range))**2), 
                          where=(np.sin(theta_range))**2 != 0)
    csc2_theta_grid = np.tile(csc2_theta.reshape(-1, 1), (1, 4*(lmax+1)))
    
    dtheta_grid = SH_function.gradient(**kw_exp_grad).theta    
    
    dphi_grid = SH_function.gradient(**kw_exp_grad).phi
    dphi_grid.data *= sin_theta_grid
    dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
    dphi2_grid = dphi_sh.gradient(**kw_exp_grad).phi
    dphi2_grid.data *= sin_theta_grid    
    
    lmax_func = SH_function.lmax
    SH_function_grid = SH_function.expand(**kw_exp_grad)

    # Laplacian identity for d2_theta            
    lapla_a = pysh.SHCoeffs.from_zeros(lmax_func)
    for l in range(lmax_func + 1):
        lapla_a.coeffs[:, l, : l + 1] = -l * (l + 1)
    
    # print("SH_function info", SH_function.info)
    # print("lapla_a info", lapla_a.info)
    
    SH_function_dtheta2 = (
                        (SH_function * lapla_a).expand(**kw_exp_grad).data 
                        - dtheta_grid.data*cot_theta_grid 
                        - dphi2_grid.data*csc2_theta_grid
                        )
    
    return SH_function_dtheta2 + SH_function_grid.data

def O2(SH_function, lmax):
    """ Beuthe (2008)'s differential operator O_2 in 2D spherical geometry. """
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1), endpoint=False))

    cot_theta = np.divide( 1.0, np.tan(theta_range), 
                          out=np.zeros_like(np.tan(theta_range)), 
                          where=np.tan(theta_range) != 0)
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    sin_theta = np.sin(theta_range)
    sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    csc_theta = np.divide( 1.0, np.sin(theta_range), 
                          out=np.zeros_like(np.sin(theta_range)), 
                          where=np.sin(theta_range) != 0)
    csc_theta_grid = np.tile(csc_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    SH_function_grid = SH_function.expand(**kw_exp_grad)
    dtheta_grid = SH_function.gradient(**kw_exp_grad).theta

    dphi_grid = SH_function.gradient(**kw_exp_grad).phi
    dphi_grid.data *= sin_theta_grid
    dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
    dphi2_grid = dphi_sh.gradient(**kw_exp_grad).phi
    dphi2_grid.data *= csc_theta_grid

    return dphi2_grid.data + cot_theta_grid * dtheta_grid.data + SH_function_grid.data

def O3(SH_function, lmax):
    """ Beuthe (2008)'s differential operator O_3 in 2D spherical geometry. """
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1), endpoint=False))

    cot_theta = np.divide( 1.0, np.tan(theta_range), 
                          out=np.zeros_like(np.tan(theta_range)), 
                          where=np.tan(theta_range) != 0)
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    sin_theta = np.sin(theta_range)
    sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    csc_theta = np.divide( 1.0, np.sin(theta_range), 
                          out=np.zeros_like(np.sin(theta_range)), 
                          where=np.sin(theta_range) != 0)
    csc_theta_grid = np.tile(csc_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    dphi_grid = SH_function.gradient(**kw_exp_grad).phi
    dphi_grid.data *= sin_theta_grid

    dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
    dthetaphi_grid = dphi_sh.gradient(**kw_exp_grad).theta

    return (csc_theta_grid * dthetaphi_grid.data 
            - cot_theta_grid * csc_theta_grid * dphi_grid.data)



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
    S_grid  = S_sol.expand(**kw_exp_grad)
    w_grid  = w_sol.expand(**kw_exp_grad)
    Te_grid = T_e_parent.expand(**kw_exp_grad)

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
    Beuthe eqs (73). *** DO NOT USE FOR PRODUCTION YET -- SEE BELOW. ***
 
    Expects Omega_grid = Beuthe's Omega = Re*omega (the output of the
    corrected compute_Omega).
 
    STATUS (measured, not assumed). Compared against stress_fields() -- the
    Banerdt/DSP Hooke form -- on the same (w, F, S, Omega) at three Te:
 
        Te/R      rms ratio sigma_tt   rms ratio sigma_pp
        0.0791          1.149                1.513
        0.0181          1.195                1.234
        0.0059          0.737                0.842
 
    Both are FIRST-ORDER thin-shell forms of the SAME theory, so any genuine
    convention difference is O(Te/R) and MUST vanish as Te/R -> 0. These
    ratios do not converge to 1 -- they get worse. That is the signature of an
    ERROR, not a convention offset. (An earlier note in this file claimed a
    "4-7% convention difference"; that claim was wrong and is retracted.)
 
    What was checked and is FINE:
      * the bending structure. Banerdt uses (O1w + nu*O2w); this uses
        (Delta'w - (1-nu)*O2w). Since O1w + O2w = Delta'w these are identical.
      * the bending prefactor, to O(Te/R): Banerdt has
        -E*zeta/((1-nu^2)*R*(R+zeta)); the -zeta/(Re+zeta) piece here matches
        it with R -> Re. The extra +beta/Re piece is ~1.4% of it.
 
    Known suspects, in order:
      1. xi is built here as 12*R^2/Te^2 but as 12*Re^2/Te^2 everywhere else
        (cons_disp_S, the solver, the eta fields). Inconsistent -- but only a
        ~5% effect, so it cannot be the whole story.
      2. the membrane term  eta/Te * (O2F + Omega): the eta prefactor and the
        absence of any 1/Re^2-type kernel on O2F need checking against the
        paper. This is where the residual 20-50% must live.
 
    Re-deriving Beuthe eq (73) requires the paper and is a paper-level task
    (standing division of labour). Until that is done, stress_fields() -- the
    Banerdt/DSP form -- is the one to use: it is a direct Hooke evaluation
    from S and w, both of which are now validated to ~1e-12 against the
    reference (see the impulse test and compute_Omega's check).
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
    """
    # Return diff operator applied S and w terms, in grid.data format
    O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    
    S_grid  = S_sol.expand(**kw_exp_grad)
    w_grid  = w_sol.expand(**kw_exp_grad)
    Te_grid = T_e_parent.expand(**kw_exp_grad)

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


def cons_disp_S(w_sol, F_sol, Omega_sol, T_e_parent, a_clm, R, T_e_0, lmax_calc, lmax_grid):
    """ 
    Beuthe (2008)'s consoidal/poloidal tangential displacement potential S_lm 
    (A_lm in DSP/Banerdt (1986)). Used in computations of strain.
    """
    
    lap_by_degree = np.array([(-l * (l + 1)) for l in range(2 * lmax_grid + 1)])
    lap2_by_degree = np.array([(-l * (l + 1) + 2) for l in range(2 * lmax_grid + 1)])
    F_lap2 = F_sol.copy()
    w_lap2 = w_sol.copy()
    for l in range(F_lap2.coeffs.shape[1]):
        F_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
        w_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
    F_lap2_grid = F_lap2.expand(**kw_exp_S)
    w_lap2_grid = w_lap2.expand(**kw_exp_S)
    
    w_grid = w_sol.expand(**kw_exp_S)
    a_grid = a_clm.expand(**kw_exp_S)
    Te_grid = T_e_parent.expand(**kw_exp_S)
    Omega_grid = Omega_sol.expand(**kw_exp_S)

    Re = R - T_e_0/2
    xi = 12*Re**2/Te_grid.data**2
    eta = xi/(1+xi)
    
    lapl_S_grid = (Re*eta*a_grid.data*(1-nu)*(F_lap2_grid.data + 2*Omega_grid.data) 
              + eta/xi * w_lap2_grid.data  
              - 2*w_grid.data)
    
    lapl_S_lm = pysh.SHGrid.from_array(lapl_S_grid).expand()
     
    S_lm = lapl_S_lm.copy()
    for l in range(1, S_lm.coeffs.shape[1]):
        S_lm.coeffs[:, l, :] /= lap_by_degree[l]
    S_lm.coeffs[0, 0, 0] = 0.0  
    
    S_lm = truncate(S_lm, lmax=lmax_calc)
        
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
                 T_e_0, g0, mass,
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
     g2) = (
                Omega_eq1_LHS(T_e_parent, lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
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
        # Omega_eq2_LHS(T_e_parent, a_eta_clm, lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
        Omega_eq2_LHS(T_e_parent, a_clm, lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))    # ETA FIELD FIX
    
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

    # A = A + A_tilde + A_tilde_group2


    # ETA FIELD FIX
    # ---- ETA-FIELD (Path 1) -------------------------------------------
    # Beuthe writes  eta * [ the whole Omega operator ], so eta is applied
    # ONCE here to the assembled Omega blocks rather than being threaded
    # into each individual field (which is ambiguous for terms whose c1 and
    # c2 halves share a field). eta is evaluated at the LOCAL Te.
    #   eta * (A_omega @ w)  =  (conv(eta) @ A_omega) @ w
    # At constant Te, eta_grid is a monopole equal to eta0, so
    # C_eta = eta0 * I and every constant-Te benchmark is preserved exactly.
    _Te_grid_eta = T_e_parent.expand(lmax=3*lmax).data
    _Re_grid_eta = R - _Te_grid_eta/2.0
    eta_grid_sb  = 1.0/(1.0 + _Te_grid_eta**2/(12.0*_Re_grid_eta**2))
    eta_clm_sb   = pysh.SHGrid.from_array(eta_grid_sb).expand()
    eta_unstr    = pysh.shio.SHCilmToVector(
                     pysh.SHCoeffs.from_array(
                       eta_clm_sb.coeffs[:, :2*lmax+1, :2*lmax+1]).coeffs)
    C_eta = build_conv_matrix(eta_unstr, gidx, plan['term_gaunt_bare'],
                              starts, seg_len, ci, cj, N)

    # eq-1 Omega LHS block = A_tilde + A_tilde_group2  ->  eta * (that)
    A = A + C_eta @ (A_tilde + A_tilde_group2)


    # ---- q's w-coupling: LHS diagonal, branch-dependent ---------------
    # NOTE: Lam_q is NOT an Omega term -- it comes from q -- so it is added
    # AFTER the C_eta multiplication above and must NOT be wrapped by C_eta.
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
        # A = A + g2['fdrho_om1'] * PhatDw
        # A = A + g2['fdrho_om2'] * (C_Te2 @ Lap_d @ PhatDw)

        # ETA FIELD FIX : these are eq-1 Omega LHS terms -> carry conv(eta).
        A = A + C_eta @ (g2['fdrho_om1'] * PhatDw)
        A = A + C_eta @ (g2['fdrho_om2'] * (C_Te2 @ Lap_d @ PhatDw))


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

    # b = b + b_tilde
    
    # ETA FIELD FIX : b_tilde is the assembled eq-2 Omega LHS operator (2a/2b/2c
    # plus the dc and drho couplings above) -> apply conv(eta) once.
    b = b + C_eta @ b_tilde        # was:  b = b + b_tilde
        
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
    # Omega_RHS2_unstr = Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_eta_clm, 
    #                                  lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
    #                                  g0=g0, mass=mass)
    Omega_RHS2_unstr = Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_clm,       # ETA FIELD FIX 
                                     lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                     g0=g0, mass=mass)

    def elem(l,m,v):
        off = 0 if m==0 else (m if m>0 else l+abs(m))
        return v[l*l+off]
    
    q = np.array([elem(l,m,q_lm_unstr) for l,m in mode_map])
    Omega_RHS1 = np.array([elem(l,m,Omega_RHS1_unstr) for l,m in mode_map])
    Omega_RHS2 = np.array([elem(l,m,Omega_RHS2_unstr) for l,m in mode_map])
    # ETA FIELD FIX : the Omega RHS vectors are the same Omega operator acting on
    # the KNOWN fields (H, G), so eta multiplies them too. In spectral form
    # that is the same convolution: eta*Omega_RHS = C_eta @ Omega_RHS.
    # (q is NOT an Omega term and is deliberately left untouched.)
    Omega_RHS1 = C_eta @ Omega_RHS1
    Omega_RHS2 = C_eta @ Omega_RHS2
 
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
    selftest_gaunt()
    topo_p, geoid_p, T_e_parent, R, g0, mass = load_inputs(LMAX_REF, strain=strain)
    T_e_0 = T_e_parent.coeffs[0,0,0]
    print(f'T_e_0 = {T_e_0/1e3:.2f} km')
    D_clm, a_clm, D_eta_clm, a_eta_clm, eta_clm = derive_D_a(T_e_parent, LMAX_REF)

    solutions_w = {}
    solutions_F = {}
    solutions_q = {}
    for lmax_run in LMAX_RUNS:
        topo_clm  = truncate(topo_p,  lmax_run)
        geoid_clm = truncate(geoid_p, lmax_run)
        plan  = build_or_load_gaunt(lmax_run, nu)        
                
        do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
        for rotation in ([0, 1] if do_rotation_check else [0]):
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
                             lmax=lmax_run, R=R, T_e_0=T_e_0, g0=g0, mass=mass,
                             D_eta_clm=D_eta_use, a_eta_clm=a_eta_use,
                             eta_clm=eta_use)
            print(f'Finished solving of system in {(time.perf_counter()-t):.1f}s\n')
            solutions_w[lmax_run, rotation] = w
            solutions_F[lmax_run, rotation] = F
            solutions_q[lmax_run, rotation] = q

    
# %% PLOTS - POWER SPECTRUM w + RESIDUAL RATIO IF ROTATION IS APPLIED
    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,9))
    for rotated in (range(2) if do_rotation_check else range(1)):
        linestyle = 'solid' if rotated == 0 else 'dashed'
        solutions_w[LMAX_REF, rotated].plot_spectrum(ax=ax1, show=False, 
                        legend=(f'lmax={LMAX_REF}'+ 
                                (f', rotated {rotate_angles}' 
                                 if rotated else '')), 
                        plot_dict={'linestyle': linestyle})
    if do_rotation_check:
        w_unrotated = solutions_w[LMAX_REF, 0].spectrum()
        w_rotated   = solutions_w[LMAX_REF, 1].spectrum()
        l = np.arange(2, LMAX_REF+1)
        ax2.plot(l, np.abs((w_unrotated[2:]/w_rotated[2:])-1)*100, 'k')
        ax2.set_xlabel('Spherical harmonic degree') 
        ax2.set_ylabel('Ratio $(S_{ww} \ / \ S_{ww-rot}) \\cdot$ 100%')
        ax2.grid(True)
        ax2.set_title('M4 - Ratio unrotated over rotated power spectra of w')
    else:
        ax2.set_visible(False)

    ax1.set_title('M4 - Power spectra of w ' 
                  + ('(Plesa Te Map)' if strain != 0 else '' ))
    ax1.legend()
    ax1.set_ylim(1e-2)
    plt.tight_layout()
    if SaveFigs:
        plt1_title = (f'M4 - Power spectra w, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath1 = os.path.join(SavePath, plt1_title)
        plt.savefig(FigPath1, dpi=200)    
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
        ax2.set_xlabel('degree l'); 
        ax2.set_ylabel(r'$|S_l/S_l^{ref}-1|$*100%')
        ax2.legend(); 
        ax2.grid(True)
        ax2.set_title(f'M4 - Residual vs lmax_ref={LMAX_REF}')
        plt.tight_layout(); 
        if SaveFigs:
            plt2_title = (f'M4 - Residuals w power, lmax_run={LMAX_RUNS}, '
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

    topography_km = topo_use_clm.expand(lmax=grid_expansion_res)
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
    ax0.set_title(f'M4 - MOLA topography map, exp. to lmax={LMAX_REF}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    
    T_e_parent_km = T_e_use_clm.expand(lmax=grid_expansion_res)
    T_e_parent_km.data = T_e_parent_km.data/1e3
    T_e_parent_km.plot(ax=ax1, 
                       ticks = 'wSne',
                       ylabel=None,
                       grid=True,
                       cmap=cmc.lajolla, 
                       colorbar='right', 
                       cb_label=r'$T_e \ [km]$',
                       **args_plot)
    ax1.set_title(f'M4 - Te input map (Plesa et al. 2018), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    D_use_clm.expand(lmax=grid_expansion_res).plot(ax=ax2, 
                                        cmap=cmc.lajolla, 
                                        colorbar='right', 
                                        cb_label=r'$D \ [N\cdot m]$',
                                        **args_plot)  
    ax2.set_title(f'M4 - Flexural rigidity D (Te-derived), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    a_use_clm.expand(lmax=grid_expansion_res).plot(ax=ax3, 
                                        cmap=cmc.lajolla, 
                                        colorbar='right', 
                                        cb_label=r'$\alpha \ [m/N$]',
                                        **args_plot) 
    ax3.set_title(f'M4 - Parameter alpha (Te-derived), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    plt.suptitle('M4 - Input maps topography & Te, and derived parameters D and $\\alpha$')
    plt.tight_layout()
    if SaveFigs:
        plt3_title = (f'M4 - Inputs Te, D and alpha, lmax_run={LMAX_RUNS}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath3 = os.path.join(SavePath, plt3_title)
        plt.savefig(FigPath3, dpi=200)
    plt.show(); plt.close()
        
# %% PLOTS - 2D DEFLECTION MAP + RESIDUAL BETWEEN LAST TWO LMAX RUNS

    # Only plot this when performing multiple lmax runs
    if len(LMAX_RUNS)>1:
        do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
        if do_rotation_check:
            w_fine = pysh.SHGrid.from_array(
                    solutions_w[LMAX_REF, 1].expand(lmax=grid_expansion_res).data/1e3)
            if len(LMAX_RUNS)>1:
                lo = LMAX_RUNS[-2] 
                d = (solutions_w[LMAX_REF, 1].coeffs[:, :lo+1, :lo+1] 
                     - solutions_w[lo, 1].coeffs[:, :lo+1, :lo+1])
                w_diff = pysh.SHCoeffs.from_array(d).expand(lmax=grid_expansion_res)
        else:
            w_fine = pysh.SHGrid.from_array(
                    solutions_w[LMAX_REF, 0].expand(lmax=grid_expansion_res).data/1e3)        
            if len(LMAX_RUNS)>1:
                lo = LMAX_RUNS[-2] 
                d = (solutions_w[LMAX_REF, 0].coeffs[:, :lo+1, :lo+1] 
                     - solutions_w[lo, 0].coeffs[:, :lo+1, :lo+1])
                w_diff = pysh.SHCoeffs.from_array(d).expand(lmax=grid_expansion_res)
                
        if len(LMAX_RUNS)>1:
            fig3, (a1,a2) = plt.subplots(2,1, figsize=(12,10))
            w_fine.plot(ax=a1, cmap=cmap1, colorbar='right', cb_label='w [km]',
                        # cmap_limits=[-24,11]
                        )
            a1.set_title(f'M4 - Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
                         + (f', rot={rotate_angles}' if rotation else ''))
    
            a1.contour(w_fine.data>0, 
                       levels=[0.99], 
                       extent=(0,360,-90,90), 
                       colors='k', 
                       origin='upper')
            w_diff.plot(ax=a2, cmap=cmap1, colorbar='right', cb_label='w diff [m]', 
                        # cmap_limits=[-320,200]
                        )
            a2.set_title(f'M4 - Residual w: lmax={LMAX_REF} minus lmax={lo}'
                         + (f', rot={rotate_angles}' if rotation else ''))
            
        else:
            fig3, a1 = plt.subplots(figsize=(12,10))
            w_fine.plot(ax=a1, cmap=cmap1, colorbar='right', cb_label='w [km]')
            a1.set_title(f'M4 - Transverse displacement w Beuthe-model (lmax={LMAX_REF})'
                         + (f', rot={rotate_angles}' if rotation else ''))
    
            a1.contour(w_fine.data>0, 
                       levels=[0.99], 
                       extent=(0,360,-90,90), 
                       colors='k', 
                       origin='upper')
        
        plt.tight_layout()
        if SaveFigs:
            plt4_title = (f'M4 - Displacement w 2D map, lmax_run={LMAX_RUNS}, '
                          f'lmaxTe={lmax_Te_fit}'
                          + (f', rotated {rotate_angles}' if rotation else '') 
                          + '.png')
            FigPath4 = os.path.join(SavePath, plt4_title)
            plt.savefig(FigPath4, dpi=200)
            print(f"Saved Figures to subfolder: {SavePath}")
        plt.show(); plt.close()
    
    

# %% PLOTS - DSP-M4 RESIDUAL PLOTS
        
    # Set whether to include crustal thickness in the plots
    show_Tc = True         
    args_expand = dict(lmax=grid_expansion_res, lmax_calc=LMAX_REF)
    args_plot = dict(tick_interval=[45, 30], grid=True)

    
    w_fine = pysh.SHGrid.from_array(
            solutions_w[LMAX_REF, 0].expand(**args_expand).data/1e3)
    w_clm = solutions_w[LMAX_REF, 0]
    dc_clm = compute_dc(w_clm, topo_clm, geoid_clm, R=R, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    dc_clm_zeroed = dc_clm.copy()
    dc_clm_zeroed.coeffs[0,0,0] = 0
    dc_grid = dc_clm_zeroed.expand(**args_expand)/1e3
    
    drho_clm = compute_drho(w_clm, topo_clm, geoid_clm, R=R, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    drho_clm_zeroed = drho_clm.copy()
    drho_clm_zeroed.coeffs[0,0,0] = 0
    drho_grid = drho_clm_zeroed.expand(**args_expand)
    
    topo_grid = topo_clm.expand(**args_expand)/1e3 - R/1e3
    
    T_c_grid = (topo_grid.data 
                + (dc_grid.data if solve_for=='dc_lm' else 0)
                - w_fine.data 
                + T_c*np.ones((2*(grid_expansion_res+1)+1, 4*(grid_expansion_res+1)+1))/1e3)
    T_c_grid = pysh.SHGrid.from_array(T_c_grid)
    
    # Load in DSP results
    w_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_w_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    dc_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_dc_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    drho_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_drho_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    Tc_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_Tc_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    
    
    # Compute residuals between DSP and M4 spatially
    grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    w_diff_DSPM4 = grid_w_DSP.copy()
    w_diff_DSPM4.data = grid_w_DSP.data - w_fine.data
    
    grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
    dc_diff_DSPM4 = grid_dc_DSP.copy()
    dc_diff_DSPM4.data = grid_dc_DSP.data - dc_grid.data
    
    grid_drho_DSP = pysh.SHCoeffs.from_array(drho_DSP.coeffs).expand(**args_expand)
    drho_diff_DSPM4 = grid_drho_DSP.copy()
    drho_diff_DSPM4.data = grid_drho_DSP.data - drho_grid.data
    
    grid_Tc_DSP = pysh.SHCoeffs.from_array(Tc_DSP.coeffs / 1e3).expand(**args_expand)
    Tc_diff_DSPM4 = grid_Tc_DSP.copy()
    Tc_diff_DSPM4.data = grid_Tc_DSP.data - T_c_grid.data
    
    # Compute residuals between DSP and M4 spectrally
    w_diff_DSPM4 = w_DSP - solutions_w[LMAX_REF,0]
    w_diff_DSPM4 = w_diff_DSPM4.expand(**args_expand)
    w_diff_DSPM4.data = w_diff_DSPM4.data / 1e3
    
    dc_diff_DSPM4 = dc_DSP - dc_clm_zeroed
    dc_diff_DSPM4 = dc_diff_DSPM4.expand(**args_expand)
    dc_diff_DSPM4.data = dc_diff_DSPM4.data / 1e3





    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    if show_Tc:
        h_space_outer = 0.3
        h_space_inner1 = 0.3
        h_space_inner2 = 0.3
        y_suptitle = 1.03
        rows=3
        cb_height = 0.06
        xticks1 = 'Wsen'
        xticks2 = 'wsen'
        xlabel = None
    else:
        h_space_outer = -0.15
        h_space_inner1 = -0.5
        h_space_inner2 = -0.35
        y_suptitle = 0.86
        rows=2
        cb_height = 0.03
        xticks1 = 'WSen'
        xticks2 = 'wSen'
        xlabel = 'Longitude'


    outer_gs = gridspec.GridSpec(rows, 1, hspace=h_space_outer)


    # --- ROW 1: Radial Displacement w ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
                                                 height_ratios=[1, cb_height], hspace=h_space_inner1, wspace=0.15)
    ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_w_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_w_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_w = [w_fine.data.min(), w_fine.data.max()]
    
    w_min, w_max = w_diff_DSPM4.data.min(), w_diff_DSPM4.data.max()
    cmap_limits_w_diff =[-max(abs(w_min), abs(w_max)), max(abs(w_min), abs(w_max))]

    grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    grid_w_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_w, 
                    cmap=cmap3, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Radial displacement w', fontweight="bold")
    
    w_fine.plot(ax=ax2, 
                cmap_limits=cmap_limits_w, 
                cmap=cmap3, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M4 - Radial displacement w', fontweight="bold")
    

    w_diff_DSPM4.plot(ax=ax3, cmap=cmap2,
                      cmap_limits = cmap_limits_w_diff,
                      colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Radial displacement w residual DSP - M4', fontweight="bold")

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



    if solve_for == 'dc_lm':
        # --- ROW 2: Crustal Root Variations ---
        inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                     height_ratios=[1, cb_height], hspace=h_space_inner2, wspace=0.15)
        ax4 = fig.add_subplot(inner_gs2[0, 0:2])
        ax5 = fig.add_subplot(inner_gs2[0, 2:4])
        ax6 = fig.add_subplot(inner_gs2[0, 4:6])
        
        cax_dc_shared = fig.add_subplot(inner_gs2[1, 1:3])
        cax_dc_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
        cmap_limits_dc = [-50, 30]

        dc_min, dc_max = dc_diff_DSPM4.data.min(), dc_diff_DSPM4.data.max()
        cmap_limits_dc_diff =[-max(abs(dc_min), abs(dc_max)), max(abs(dc_min), abs(dc_max))]
    
        grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
        grid_dc_DSP.plot(ax=ax4, cmap=cmap3, cmap_limits=[-50, 30], colorbar=None, ticks=xticks1, xlabel=xlabel, **args_plot)
        ax4.set_title('DSP - Crustal root variations', fontweight="bold")
        
        dc_grid.plot(ax=ax5, cmap=cmap3, cmap_limits=cmap_limits_dc, colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax5.set_title('M4 - Crustal root variations', fontweight="bold")
    
        dc_diff_DSPM4.plot(ax=ax6, cmap=cmap2, 
                           cmap_limits=cmap_limits_dc_diff, 
                           colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax6.set_title('Crustal root variations residual DSP - M4', fontweight="bold")
    
        norm_dc = mcolors.Normalize(vmin=cmap_limits_dc[0], vmax=cmap_limits_dc[1])
        cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_dc, cmap=cmap3), cax=cax_dc_shared, orientation='horizontal')
        cb3.set_label('$\\delta c$ [km]', fontweight="bold")
    
        norm_dc_diff = mcolors.Normalize(vmin=cmap_limits_dc_diff[0], vmax=cmap_limits_dc_diff[1])
        cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_dc_diff, cmap=cmap2), cax=cax_dc_diff, orientation='horizontal')
        cb4.set_label('$\\delta c$ [km]', fontweight="bold")


    if solve_for == 'drho_lm':
        # --- ROW 2: Mantle density Variations ---
        inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                     height_ratios=[1, cb_height], hspace=h_space_inner2, wspace=0.15)
        ax4 = fig.add_subplot(inner_gs2[0, 0:2])
        ax5 = fig.add_subplot(inner_gs2[0, 2:4])
        ax6 = fig.add_subplot(inner_gs2[0, 4:6])
        
        cax_drho_shared = fig.add_subplot(inner_gs2[1, 1:3])
        cax_drho_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
        cmap_limits_drho = [-500, 500]
        
        drho_min, drho_max = drho_diff_DSPM4.data.min(), drho_diff_DSPM4.data.max()
        cmap_limits_drho_diff =[-max(abs(drho_min), abs(drho_max)), max(abs(drho_min), abs(drho_max))]
    
        grid_drho_DSP.plot(ax=ax4, cmap=cmap1, cmap_limits=cmap_limits_drho, colorbar=None, ticks=xticks1, xlabel=xlabel, **args_plot)
        ax4.set_title('DSP - Mantle density variations', fontweight="bold")
        
        drho_grid.plot(ax=ax5, cmap=cmap1, cmap_limits=cmap_limits_drho, colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax5.set_title('M4 - Mantle density variations', fontweight="bold")
    
        drho_diff_DSPM4.plot(ax=ax6, cmap=cmap2, 
                           cmap_limits=cmap_limits_drho_diff, 
                           colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax6.set_title('Mantle density variations residual DSP - M4', fontweight="bold")
    
        norm_drho = mcolors.Normalize(vmin=cmap_limits_drho[0], vmax=cmap_limits_drho[1])
        cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_drho, cmap=cmap1), cax=cax_drho_shared, orientation='horizontal')
        cb3.set_label('$\\delta \\rho$ [kg/m$^3$]', fontweight="bold")
    
        norm_drho_diff = mcolors.Normalize(vmin=cmap_limits_drho_diff[0], vmax=cmap_limits_drho_diff[1])
        cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_drho_diff, cmap=cmap2), cax=cax_drho_diff, orientation='horizontal')
        cb4.set_label('$\\delta \\rho$ [kg/m$^3$]', fontweight="bold")

    # --- ROW 3: Crustal Thickness ---
    if show_Tc:
        inner_gs3 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[2], 
                                                     height_ratios=[1, cb_height], hspace=0.3, wspace=0.15)
        ax7 = fig.add_subplot(inner_gs3[0, 0:2])
        ax8 = fig.add_subplot(inner_gs3[0, 2:4])
        ax9 = fig.add_subplot(inner_gs3[0, 4:6])
        
        cax_tc_shared = fig.add_subplot(inner_gs3[1, 1:3])
        cax_tc_diff   = fig.add_subplot(inner_gs3[1, 4:6])
    
        Tc_min, Tc_max = Tc_diff_DSPM4.data.min(), Tc_diff_DSPM4.data.max()
        cmap_limits_Tc_diff =[-max(abs(Tc_min), abs(Tc_max)), max(abs(Tc_min), abs(Tc_max))]

        grid_Tc_DSP.plot(ax=ax7, cmap=cmap3, cmap_limits=[0, 110], colorbar=None, xlabel=None, ticks='WSen', **args_plot)
        ax7.set_title('DSP - Crustal thickness', fontweight="bold")
        
        T_c_grid.plot(ax=ax8, cmap=cmap3, cmap_limits=[0, 110], colorbar=None, ticks='wSen', xlabel=None, ylabel=None, **args_plot)
        ax8.set_title('M4 - Crustal thickness', fontweight="bold")
    
        tc_min, tc_max = Tc_diff_DSPM4.data.min(), Tc_diff_DSPM4.data.max()
        Tc_diff_DSPM4.plot(ax=ax9, cmap=cmap2, cmap_limits=cmap_limits_Tc_diff, colorbar=None, ticks='wSen', xlabel=None, ylabel=None, **args_plot)
        ax9.set_title('Crustal thickness residual DSP - M4', fontweight="bold")
            
        norm_tc = mcolors.Normalize(vmin=0, vmax=110)
        cb5 = fig.colorbar(cm.ScalarMappable(norm=norm_tc, cmap=cmap3), cax=cax_tc_shared, orientation='horizontal')
        cb5.set_label('$T_c$ [km]', fontweight="bold")
    
        norm_tc_diff = mcolors.Normalize(vmin=cmap_limits_Tc_diff[0], vmax=cmap_limits_Tc_diff[1])
        cb6 = fig.colorbar(cm.ScalarMappable(norm=norm_tc_diff, cmap=cmap2), cax=cax_tc_diff, orientation='horizontal')
        cb6.set_label('$T_c$ [km]', fontweight="bold")




    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle('Residual checks DSP and M4\n'
                 + ('Solving for $\\delta \\rho_{lm}$, $\\delta c_{lm}$=0' if solve_for == 'drho_lm' else '')
                 + ('Solving for $\\delta c_{lm}$, $\\delta \\rho_{lm}$=0' if solve_for == 'dc_lm' else '')
                 + f' --- lmax={LMAX_REF}'
                 + f'\nDSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M4 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M4 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M4 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + f'\nDSP & M4 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=y_suptitle, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = (f'Residual_checks_DSP_M4_lmax={LMAX_REF}_'
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M4=PlesaStrain14Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M4=PlesaStrain17Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + f'Tc={T_c/1e3}km'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    
# %% PLOTS - w-POWER SPECTRA COMPARISONS BETWEEN DSP AND M4
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,9))

    solutions_w[lmax_run, rotation].plot_spectrum(ax=ax1, show=False, 
                    legend=('M4 coeffs'))
    w_coeffs_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_w_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    w_coeffs_DSP.coeffs[0,0,0] = 0
    w_coeffs_DSP.plot_spectrum(ax=ax1, show=False, 
                    legend=('DSP coeffs'), plot_dict={'linestyle': '--'})
    ax1.set_title('M4 & DSP w-coeffs')

    w_spectrum_diff = (w_coeffs_DSP- solutions_w[LMAX_REF, 0]).spectrum() 
    l = np.arange(0,(LMAX_REF+1))
    ax2.plot(l, w_spectrum_diff, 
             label=('M4 - DSP'))
    ax2.set_title('|DSP-w| residual')
    plt.tight_layout()
    plt.grid()
    ax1.set_ylim(1e-2)
    ax2.set_xlim(0,44)
    if SaveFigs:
        plt_savetitle = (f'Comparison_w_DSP_M4_lmax={LMAX_REF}_'
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M4=PlesaStrain14Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M4=PlesaStrain17Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + f'Tc={T_c/1e3}km'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    
    
# %% PLOTS - w-POWER SPECTRA 2D COMPARISONS BETWEEN DSP AND M4
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7,9))

    solutions_w[LMAX_REF, 0].plot_spectrum2d(ax=ax1, show=False, cmap_limits=[1e-6,1e6])
    ax1.set_title('M4 w-coeffs')
    
    w_coeffs_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_w_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    w_coeffs_DSP.coeffs[0,0,0] = 0
    w_coeffs_DSP.plot_spectrum2d(ax=ax2, show=False, cmap_limits=[1e-6,1e6])
    ax2.set_title('DSP w-coeffs')

    w_diff = w_coeffs_DSP - solutions_w[LMAX_REF, 0]
    w_diff.plot_spectrum2d(ax=ax3, show=False )
    ax3.set_title('DSP-w residual')
    
    plt.tight_layout()
    plt.grid()
    plt.show()

    
# %% PLOTS - STRESS AND STRAIN FIELDS


    w_clm = solutions_w[LMAX_REF, 0].expand(lmax=grid_expansion_res).expand()
    F_clm = solutions_F[LMAX_REF, 0].expand(lmax=grid_expansion_res).expand()
    q_clm = solutions_q[LMAX_REF, 0].expand(lmax=grid_expansion_res).expand()

    # compute_Omega now returns Beuthe's Omega = Re*omega (required by
    # cons_disp_S)
    Omega_coeffs = compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, q_clm, g0=g0, R=R, T_e_0=T_e_0, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    Omega_grid = Omega_coeffs.expand(lmax=grid_expansion_res)

    # S first (needed by the DSP-convention stress_fields), then stresses
    S_clm = cons_disp_S(w_clm, F_clm, Omega_coeffs, T_e_parent, a_clm, R=R, T_e_0=T_e_0, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    S_clm.coeffs[0,0,0] = 0
    w_clm.coeffs[0,0,0] = 0
    
    # TEST TO ALIGN DSP AND M4:
    # There is a difference between DSP A_lm and M4 S_lm in degree 1 terms only.
    # Caused by laplacian difference in Beuthe formulation between constant and
    # variable Te (is hypothesis now). For now compare two by setting degree-1
    # terms to zero in both.
    # S_clm.coeffs[:,1,:] = 0    
    
    
    sigma_tt, sigma_pp, sigma_tp = stress_fields(S_clm, w_clm, T_e_parent, lmax=grid_expansion_res, R=R, T_e_0=T_e_0)
    eps_tt, eps_pp, eps_tp = strain_fields(S_clm, w_clm, T_e_parent, lmax=grid_expansion_res, R=R, T_e_0=T_e_0)

    eps_tt.data = eps_tt.data*1e3; eps_pp.data = eps_pp.data*1e3; eps_tp.data = eps_tp.data*1e3
    sigma_tt.data = sigma_tt.data*1e3/1e5; sigma_pp.data = sigma_pp.data*1e3/1e5; sigma_tp.data = sigma_tp.data*1e3/1e5

    fig, ((ax1, ax4), 
          (ax2, ax5),
          (ax3, ax6)) = plt.subplots(3, 2, figsize=(12,11), dpi=100)
    sigma_tt.plot(ax=ax1, cmap=cmap1, cmap_limits=[-2.0, 2.0], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field $\\sigma_{\\theta \\theta} \ (\\times 10^{5}$)')
    sigma_pp.plot(ax=ax2, cmap=cmap1, cmap_limits=[-2.0, 2.0],  tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field $\\sigma_{\\phi \\phi} \ (\\times 10^{5})$')
    sigma_tp.plot(ax=ax3, cmap=cmap1, cmap_limits=[-0.2, 0.2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Stress field $\\sigma_{\\theta \\phi} \ (\\times 10^{5})$')
    
    eps_tt.plot(  ax=ax4, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field $\\hat \\epsilon_{\\theta \\theta} \ (\\times 10^{-3}$)')
    eps_pp.plot(  ax=ax5, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field $\\hat \\epsilon_{\\phi \\phi} \ (\\times 10^{-3}$)')
    eps_tp.plot(  ax=ax6, cmap=cmap1, cmap_limits=[-2, 2], tick_interval=[45, 30], colorbar = 'bottom', cb_label='Strain field $\\hat \\epsilon{\\theta \\phi} \ (\\times 10^{-3}$)')

    plt.suptitle('M4 - Stress & strain fields')
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
        # grid=True,
        # cb_tick_interval=1,
    )
    fig, ((ax1, ax2), 
          (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10), dpi=100)
    
    
    pysh.SHGrid.from_array(min_strain * 1e3).plot(
        ax=ax1,
        ticks="WSne",
        cb_label="Minimum principal horizontal strain ($\\times 10^{-3}$)",
        cmap_limits=[-4, 4],
        # xlabel=None,
        **args_plot,
    )
    pysh.SHGrid.from_array(max_strain * 1e3).plot(
        ax=ax2,
        cb_label="Maximum principal horizontal strain ($\\times 10^{-3}$)",
        ticks="wSnE",
        cmap_limits=[-4, 4],
        ylabel=None,
        **args_plot,
    )
    pysh.SHGrid.from_array(sum_strain * 1e3).plot(
        ax=ax3,
        cb_label="Sum of principal horizontal strains ($\\times 10^{-3}$)",
        cmap_limits=[-3, 3],
        ticks="WSne",
        # xlabel=None,
        **args_plot,
    )
    pysh.SHGrid.from_array(principal_angle_strain).plot(
        ax=ax4,
        cb_label="Principal angle (°)",
        ticks="wSnE",
        cmap_limits=[-90, 90],
        ylabel=None,
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
    plt.suptitle('M4 - Principal strains', y=0.9)
    # plt.tight_layout()
    plt.show()
    
    
    
    ## %% PLOT RESIDUAL STRAINS AND ANGLES BETWEEN DSP AND M4
    sum_strain_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_sum_strain_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    princ_angle_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_principal_angle_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # stress_theta_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_stress_theta_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # stress_phi_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_stress_phi_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    # stress_thetaphi_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_stress_theta_phi_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    tot_thetaphi_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_strain_theta_phi_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')


    sum_strain = pysh.SHGrid.from_array(sum_strain * 1e3)
    sum_strain_residual = sum_strain_DSP.data - sum_strain.data
    
    princ_angle_residual = princ_angle_DSP.data - pysh.SHGrid.from_array(principal_angle_strain).data
    princ_angle_residual = ((princ_angle_residual + 90) % 180) - 90
    
    
    args_expand = dict(lmax=grid_expansion_res, lmax_calc=LMAX_REF)
    args_plot = dict(tick_interval=[45, 30], grid=True)

    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    outer_gs = gridspec.GridSpec(2, 1, hspace=-0.15) 



    # compare thetaphi strains to check definition residuals
    tot_thetaphi_M4 = (eps_tp * 1e3)
    tot_thetaphi_residual = tot_thetaphi_DSP.data - tot_thetaphi_M4.data





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
    # cmap_limits_strain_diff =[(strain_min), (strain_max)]
    
    # grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    sum_strain_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_strain, 
                    cmap=cmap1, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Sum principal strain', fontweight="bold")
    
    sum_strain.plot(ax=ax2, 
                cmap_limits=cmap_limits_strain, 
                cmap=cmap1, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M4 - Sum principal strain', fontweight="bold")
    
    pysh.SHGrid.from_array(sum_strain_residual).plot(ax=ax3, cmap=cmap2, 
                                                     cmap_limits=cmap_limits_strain_diff,
                                                     colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Sum principal strain residual DSP - M4', fontweight="bold")

    norm_strain = mcolors.Normalize(vmin=cmap_limits_strain[0], vmax=cmap_limits_strain[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_strain, 
                    cmap=cmap1), 
                        cax=cax_strain_shared, orientation='horizontal')
    cb1.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")


    norm_strain_diff = mcolors.Normalize(vmin=cmap_limits_strain_diff[0], vmax=cmap_limits_strain_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_strain_diff, 
                    cmap=cmap2), 
                        cax=cax_strain_diff, orientation='horizontal')
    cb2.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")


    # # --- ROW 1: strain theta phi ---
    # # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    # inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
    #                                              height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
    # ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    # ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    # ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # # Shared colorbar spans underneath columns 0 and 1
    # cax_strain_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    # cax_strain_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    # cmap_limits_strain = [-1,1]
    # strain_min, strain_max = tot_thetaphi_residual.min(), tot_thetaphi_residual.max()
    # cmap_limits_strain_diff =[-max(abs(strain_min), abs(strain_max)), max(abs(strain_min), abs(strain_max))]
    # # cmap_limits_strain_diff =[(strain_min), (strain_max)]
    
    # # grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    # tot_thetaphi_DSP.plot(ax=ax1, 
    #                 cmap_limits=cmap_limits_strain, 
    #                 cmap=cmap1, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    # ax1.set_title('DSP - tot strain thetaphi', fontweight="bold")
    
    # tot_thetaphi_M4.plot(ax=ax2, 
    #             cmap_limits=cmap_limits_strain, 
    #             cmap=cmap1, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    # ax2.set_title('M4 - tot strain thetaphi', fontweight="bold")
    
    # pysh.SHGrid.from_array(tot_thetaphi_residual).plot(ax=ax3, cmap=cmap2, 
    #                                                  cmap_limits=cmap_limits_strain_diff,
    #                                                  colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    # ax3.set_title('tot strain thetaphi residual DSP - M4', fontweight="bold")

    # norm_strain = mcolors.Normalize(vmin=cmap_limits_strain[0], vmax=cmap_limits_strain[1])
    # cb1 = fig.colorbar(cm.ScalarMappable(
    #                 norm=norm_strain, 
    #                 cmap=cmap1), 
    #                     cax=cax_strain_shared, orientation='horizontal')
    # cb1.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")


    # norm_strain_diff = mcolors.Normalize(vmin=cmap_limits_strain_diff[0], vmax=cmap_limits_strain_diff[1])
    # cb2 = fig.colorbar(cm.ScalarMappable(
    #                 norm=norm_strain_diff, 
    #                 cmap=cmap2), 
    #                     cax=cax_strain_diff, orientation='horizontal')
    # cb2.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")




    # --- ROW 2: Principal angle ---
    inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                 height_ratios=[1, 0.03], hspace=-0.35, wspace=0.15)
    ax4 = fig.add_subplot(inner_gs2[0, 0:2])
    ax5 = fig.add_subplot(inner_gs2[0, 2:4])
    ax6 = fig.add_subplot(inner_gs2[0, 4:6])
    
    cax_angle_shared = fig.add_subplot(inner_gs2[1, 1:3])
    cax_angle_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
    cmap_limits_angle = [-90,90]
    angle_min, angle_max = princ_angle_residual.min(), princ_angle_residual.max()
    cmap_limits_angle_diff =[-max(abs(angle_min), abs(angle_max)), max(abs(angle_min), abs(angle_max))]

    princ_angle_DSP.plot(ax=ax4, cmap=cmap1, cmap_limits=cmap_limits_angle, colorbar=None, ticks='WSen', **args_plot)
    ax4.set_title('DSP - Principal angle', fontweight="bold")
    
    pysh.SHGrid.from_array(principal_angle_strain).plot(ax=ax5, cmap=cmap1, cmap_limits=cmap_limits_angle, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax5.set_title('M4 - Principal angle', fontweight="bold")


    angle_min, angle_max = princ_angle_residual.min(), princ_angle_residual.max()
    pysh.SHGrid.from_array(princ_angle_residual).plot(ax=ax6, cmap=cmap2, cmap_limits=cmap_limits_angle_diff, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax6.set_title('Principal angle residual DSP - M4', fontweight="bold")

    norm_angle = mcolors.Normalize(vmin=cmap_limits_angle[0], vmax=cmap_limits_angle[1])
    cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_angle, cmap=cmap1), cax=cax_angle_shared, orientation='horizontal')
    cb3.set_label('Principal angle [°]', fontweight="bold")

    norm_angle_diff = mcolors.Normalize(vmin=cmap_limits_angle_diff[0], vmax=cmap_limits_angle_diff[1])
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
    # tecto_path = f"{os.getcwd()}/Tectonics_data"
    # Plt_tecto_Mars(tecto_path, ax=[ax4,ax5], compression=True, extension=False)



    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle('Residual strains DSP and M4\n'
                 + ('Solving for $\\delta \\rho_{lm}$, $\\delta c_{lm}$=0' if solve_for == 'drho_lm' else '')
                 + ('Solving for $\\delta c_{lm}$, $\\delta \\rho_{lm}$=0' if solve_for == 'dc_lm' else '')
                 + f' --- lmax={LMAX_REF}'
                 + f'\nDSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M4 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M4 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M4 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + f'\nDSP & M4 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=0.86, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = ('Residual_strains_DSP_M4_lmax={LMAX_REF}_'
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M4=PlesaStrain14Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M4=PlesaStrain17Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + 'Tc={T_c/1e3}km'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

# %% DSP A_lm vs M4 S_lm
        

    Omega_lm_M4 = compute_Omega(solutions_w[LMAX_REF,0], T_e_parent, topo_clm, geoid_clm, solutions_q[LMAX_REF,0], g0, R=R, T_e_0=T_e_0, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    Omega_lm_M4_DSPconv = Omega_lm_M4.copy()
    Omega_lm_M4_DSPconv.coeffs = Omega_lm_M4_DSPconv.coeffs/(R-T_e_0/2)
    Omega_grid_M4 = pysh.SHCoeffs.from_array(Omega_lm_M4_DSPconv.coeffs).expand(**args_expand)


    Omega_lm_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_omega_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    Omega_grid_DSP = pysh.SHCoeffs.from_array(Omega_lm_DSP.coeffs).expand(**args_expand)

    Omega_diff_DSPM4 = Omega_grid_DSP.copy()
    Omega_diff_DSPM4.data = Omega_grid_DSP.data - Omega_grid_M4.data


    S_lm = cons_disp_S(solutions_w[LMAX_REF,0], solutions_F[LMAX_REF,0], Omega_lm_M4, T_e_parent, a_clm, R=R, T_e_0=T_e_0, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    A_lm = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_SolveFor{solve_for}_A_lmax={LMAX_REF}_Tc={T_c}_Te={Te_input}_rhol={rho_l}_rhoc={rho_c}_rhom={rho_m}')
    
    
    S_grid = pysh.SHGrid.from_array(S_lm.expand(**args_expand).data/1e3)
    A_grid = pysh.SHCoeffs.from_array(A_lm.coeffs / 1e3).expand(**args_expand)

    A_S_diff_DSPM4 = A_grid.copy()
    A_S_diff_DSPM4.data = A_grid.data - S_grid.data


    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    outer_gs = gridspec.GridSpec(2, 1, hspace=-0.15) 


    # --- ROW 1: Consoidal load potential Omega ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
                                                 height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
    ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_Omega_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_Omega_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_Omega = [Omega_grid_M4.data.min(), Omega_grid_M4.data.max()]
    
    Omega_min, Omega_max = Omega_diff_DSPM4.data.min(), Omega_diff_DSPM4.data.max()
    cmap_limits_Omega_diff =[-max(abs(Omega_min), abs(Omega_max)), max(abs(Omega_min), abs(Omega_max))]

    Omega_grid_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_Omega, 
                    cmap=cmap3, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Consoidal load potential Omega_lm', fontweight="bold")
    
    Omega_grid_M4.plot(ax=ax2, 
                cmap_limits=cmap_limits_Omega, 
                cmap=cmap3, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M4 - Consoidal load potential Omega_lm', fontweight="bold")
    

    Omega_diff_DSPM4.plot(ax=ax3, cmap=cmap2,
                      cmap_limits = cmap_limits_Omega_diff,
                      colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Consoidal load potential residual Omega DSP - M4', fontweight="bold")

    norm_Omega = mcolors.Normalize(vmin=cmap_limits_Omega[0], vmax=cmap_limits_Omega[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_Omega, 
                    cmap=cmap3), 
                       cax=cax_Omega_shared, orientation='horizontal')
    cb1.set_label('$\\Omega$ [N/km]', fontweight="bold")
   
    norm_Omega_diff = mcolors.Normalize(vmin=cmap_limits_Omega_diff[0], vmax=cmap_limits_Omega_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(norm=norm_Omega_diff, cmap=cmap2), 
                       cax=cax_Omega_diff, orientation='horizontal')
    cb2.set_label('$\\Omega$ [N/km]', fontweight="bold")

    ax1.contour(Omega_grid_DSP.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')
    ax2.contour(Omega_grid_M4.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')



    # --- ROW 2: Tangential displacement A and S ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                 height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
    ax4 = fig.add_subplot(inner_gs1[0, 0:2])
    ax5 = fig.add_subplot(inner_gs1[0, 2:4])
    ax6 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_A_S_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_A_S_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_S = [S_grid.data.min(), S_grid.data.max()]
    
    A_S_min, A_S_max = A_S_diff_DSPM4.data.min(), A_S_diff_DSPM4.data.max()
    cmap_limits_A_S_diff =[-max(abs(A_S_min), abs(A_S_max)), max(abs(A_S_min), abs(A_S_max))]

    A_grid.plot(ax=ax4, 
                    cmap_limits=cmap_limits_S, 
                    cmap=cmap3, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax4.set_title('DSP - Consoidal displacement A_lm', fontweight="bold")
    
    S_grid.plot(ax=ax5, 
                cmap_limits=cmap_limits_S, 
                cmap=cmap3, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax5.set_title('M4 - Consoidal displacement S_lm', fontweight="bold")
    

    A_S_diff_DSPM4.plot(ax=ax6, cmap=cmap2,
                      cmap_limits = cmap_limits_A_S_diff,
                      colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax6.set_title('Consoidal displacement residual A-S DSP - M4', fontweight="bold")

    norm_A_S = mcolors.Normalize(vmin=cmap_limits_S[0], vmax=cmap_limits_S[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_A_S, 
                    cmap=cmap3), 
                       cax=cax_A_S_shared, orientation='horizontal')
    cb1.set_label('A & S [km]', fontweight="bold")
   
    norm_A_S_diff = mcolors.Normalize(vmin=cmap_limits_A_S_diff[0], vmax=cmap_limits_A_S_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(norm=norm_A_S_diff, cmap=cmap2), 
                       cax=cax_A_S_diff, orientation='horizontal')
    cb2.set_label('A & S [km]', fontweight="bold")

    ax4.contour(A_grid.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')
    ax5.contour(S_grid.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')




    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle('Residual A_lm DSP and S_lm M4\n'
                 + ('Solving for $\\delta \\rho_{lm}$, $\\delta c_{lm}$=0' if solve_for == 'drho_lm' else '')
                 + ('Solving for $\\delta c_{lm}$, $\\delta \\rho_{lm}$=0' if solve_for == 'dc_lm' else '')
                 + f' --- lmax={LMAX_REF}'
                 + f'\nDSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M4 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M4 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M4 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + f'\nDSP & M4 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=0.86, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = ('Residual_checks_DSP_M4_lmax={LMAX_REF}_'
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M4=PlesaStrain14Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M4=PlesaStrain17Map_'
                   'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + 'Tc={T_c/1e3}km'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


# %% PLOTS - RESIDUALS OMEGA AND A/S DSP-M4

    xlim = LMAX_REF-1

    fig, ((ax1, ax2), 
          (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,9))

    Omega_lm_M4_DSPconv.plot_spectrum(ax=ax1, show=False, 
                    legend=('M4 Omega coeffs'))
    Omega_lm_DSP.plot_spectrum(ax=ax1, show=False, 
                    legend=('DSP Omega coeffs'), plot_dict={'linestyle': '--'})
    ax1.set_title('M4 & DSP Omega-coeffs')
    ax1.set_xlim(0,xlim)    
    ax1.set_ylim(1e5)    

    Omega_spectrum_diff = (Omega_lm_DSP - Omega_lm_M4_DSPconv).spectrum() 
    l = np.arange(0,(LMAX_REF+1))
    ax2.plot(l, Omega_spectrum_diff, 
             label=('DSP - M4'))
    ax2.set_title('|DSP-M4| Omega residual')
    ax2.set_xlim(0,xlim)    
    ax2.grid(True)

    S_lm.plot_spectrum(ax=ax3, show=False, 
                    legend=('M4 S_lm coeffs'))
    A_lm.plot_spectrum(ax=ax3, show=False, 
                    legend=('DSP A_lm coeffs'), plot_dict={'linestyle': '--'})
    ax3.set_title('M4 & DSP S-A -coeffs')
    ax3.set_xlim(0,xlim)    
    ax3.set_ylim(1e-5)    

    A_S_spectrum_diff =(A_lm - S_lm).spectrum()
    l = np.arange(0,(LMAX_REF+1))
    ax4.plot(l, A_S_spectrum_diff, 
             label=('DSP - M4'))
    ax4.set_title('|DSP-M4| A-S residual')
    ax4.set_xlim(0,xlim)    

    plt.tight_layout()
    plt.grid()
    plt.show()
    
    
    
    
# %% TRANSFER FUNCTION IMPULSE TEST (constant Te; self-contained)


    # ---------------------------------------------------------------------
    # Validates the FULL production solver against an independent per-degree
    # transcription of the Banerdt/DSP 5-equation system -- for BOTH closures
    # and any rho_l. No external files, no Mars data: pure monopoles plus
    # unit impulses, so it exercises the plan, the eta fields, the omega
    # wiring, q, the matrix build and the solve, end to end.
    #
    # WHY IT WORKS: at constant Te every operator is degree-diagonal, so
    #   w(l,m) = T_H(l)*H(l,m) + T_G(l)*G(l,m).
    # Injecting 1 m at every (l,0) in ONE solve therefore reads out the whole
    # transfer function T_H(l); swapping the topo/geoid roles reads T_G(l).
    #
    # IMPORTANT: the reference below follows `solve_for`. Comparing the
    # drho_lm solver against a dc_lm reference (or vice versa) compares two
    # different physical problems and yields O(1) "errors" that are not errors.
    #
    # Expected: |T_code/T_ref - 1| ~ 1e-9 or below at every degree.
    #           Run this after EVERY structural edit.
    # ---------------------------------------------------------------------
    T_e_0_t = T_e_parent.coeffs[0,0,0]
    lmax_t  = 20

    def mono(val, lmax):
        c = pysh.SHCoeffs.from_zeros(lmax=lmax, normalization='4pi')
        c.coeffs[0,0,0] = val
        return c

    def impulse(Rval, lmax, amp=None):
        c = mono(Rval, lmax)
        if amp is not None:
            for l in range(2, lmax+1):
                c.coeffs[0, l, 0] = amp
        return c

    Te_mono = mono(T_e_0_t, 2*lmax_t)
    D_mono  = mono(E*T_e_0_t**3/(12*(1-nu**2)), 2*lmax_t)
    a_mono  = mono(1.0/(E*T_e_0_t), 2*lmax_t)
    # eta-retained (Beuthe eqs 58/66) monopoles. NOTE: the ETA_FULL flag no
    # longer exists in the config -- the model is always eta-full -- so these
    # are built unconditionally, mirroring derive_D_a's strain==0 branch.
    Re_i   = R - T_e_0_t/2
    eta0_i = 1.0/(1.0 + T_e_0_t**2/(12.0*Re_i**2))
    D_eta_mono = mono(eta0_i*E*T_e_0_t**3/(12*(1-nu**2)), 2*lmax_t)
    a_eta_mono = mono(eta0_i/(E*T_e_0_t), 2*lmax_t)
    eta_mono   = mono(eta0_i, 2*lmax_t)
    plan_t = build_or_load_gaunt(lmax_t, nu)

    _sb = dict(D_eta_clm=D_eta_mono,
               a_eta_clm=a_eta_mono, eta_clm=eta_mono)
    w_H,_,_ = solve_beuthe(impulse(R, lmax_t, amp=1.0), impulse(R, lmax_t),
                           Te_mono, D_mono, a_mono, plan_t, lmax_t, R,
                           T_e_0_t, g0, mass, **_sb)
    w_G,_,_ = solve_beuthe(impulse(R, lmax_t), impulse(R, lmax_t, amp=1.0),
                           Te_mono, D_mono, a_mono, plan_t, lmax_t, R,
                           T_e_0_t, g0, mass, **_sb)
    TH_code = np.array([w_H.coeffs[0,l,0] for l in range(2, lmax_t+1)])
    TG_code = np.array([w_G.coeffs[0,l,0] for l in range(2, lmax_t+1)])

    # ---- independent reference: Banerdt/DSP 5x5 per degree ---------------
    # Unknowns [w, Gc, q, omega, X], X = dc_lm or drhom_lm per `solve_for`.
    # Transcribed from upstream DSP (B1986_nmax.py, `Eqns`).
    rhobar_t = mass*3/(4*np.pi*R**3)
    Re_t     = R - T_e_0_t/2
    phi_t    = (R - T_c)/R
    Rc_t     = R - T_c
    g_m_t    = g0*(1+(phi_t**3-1)*rho_c/rhobar_t)/phi_t**2
    RTeR_t   = (R - T_e_0_t)/R
    rho_Te_t = rho_c if T_e_0_t <= T_c else rho_m
    gTe_t    = g0*(1+(RTeR_t**3-1)*rho_Te_t/rhobar_t)/RTeR_t**2
    alph_t   = 1/(E*T_e_0_t); D0_t = E*T_e_0_t**3/(12*(1-nu**2))
    eps_t    = 12*Re_t**2/T_e_0_t**2
    beta_t   = 1/(1+eps_t); eta_t = eps_t/(1+eps_t)
    v1v_t    = nu/(1-nu)
    Tcind_t  = T_c if T_c < T_e_0_t else 0.0
    mx_t     = max(T_e_0_t - T_c, 0.0)
    # density-anomaly layer (only used by the drho_lm branch)
    M_t_     = Mb - Mt
    Rtop_t   = R - Mt
    Rbase_t  = R - Mb
    Rmid_t   = R - (Mt + Mb)/2.0
    rho_d_t  = rho_c if Mt <= T_c else rho_m
    gdrho_t  = g0*(1+((Rmid_t/R)**3-1)*rho_d_t/rhobar_t)/(Rmid_t/R)**2
    Phat_t   = (-0.5*v1v_t*gdrho_t*(T_e_0_t - Mt)
                * (min(M_t_, T_e_0_t - Mt) if Mt < T_e_0_t else 0.0))

    def ref_w(l, H, G):
        Lap = -l*(l+1.); Lp = Lap + 2.
        Cb  = 3./(rhobar_t*(2*l+1.)); Rl3 = R/(l+3.)
        A = np.zeros((5,5)); rhs = np.zeros(5)
        iw, iGc, iq, iom, iX = 0, 1, 2, 3, 4
        # eq (1) G
        A[0,iw] = Cb*(drhol + drho*phi_t**(l+2))
        rhs[0]  = G - Cb*rho_l*H
        # eq (2) Gc
        A[1,iGc] = -1.
        A[1,iw]  = Cb*(g0/g_m_t)*(drhol*phi_t**l + drho*phi_t)
        rhs[1]   = -Cb*(g0/g_m_t)*rho_l*phi_t**l*H
        # eq (3) q
        A[2,iq]  = -1.
        A[2,iGc] = -g_m_t*drho
        A[2,iw]  = g0*drhol + g_m_t*drho
        rhs[2]   = -g0*rho_l*(H - G)
        # eq (4) w
        A[3,iw]  = eta_t*D0_t*Lap*Lp**2 + (Re_t**2/alph_t)*Lp
        A[3,iq]  = Re_t**4*(Lp - 1. - nu)
        A[3,iom] = -Re_t**4*(beta_t*Lp - 1. - nu)*Lap
        # eq (5) omega
        A[4,iom] = -1.
        A[4,iw]  = (-(drhol*g0*v1v_t*T_e_0_t - rho_c*g_m_t*Tcind_t
                      - rho_m*gTe_t*mx_t)/R - v1v_t*drho*g_m_t*mx_t/R)
        rhs[4]   = -v1v_t*rho_l*g0*T_e_0_t*H/R
        # ---- the X column: the ONLY branch-dependent part ----
        if solve_for == 'dc_lm':
            A[0,iX] = -Cb*drho*phi_t**(l+2)
            A[1,iX] = -Cb*(g0/g_m_t)*drho*phi_t
            A[2,iX] = -g_m_t*drho
            A[4,iX] =  v1v_t*drho*g_m_t*mx_t/R
        else:                                   # 'drho_lm'  (dc = 0)
            RtbRl3 = (Rtop_t/R)**(l+3.) - (Rbase_t/R)**(l+3.)
            RtRCl  = ((Rtop_t/Rc_t)**l if Rtop_t <= Rc_t
                      else (Rc_t/Rtop_t)**(l+1.)) * Rtop_t**3/(Rc_t*R**2)
            RbRCl  = ((Rbase_t/Rc_t)**l if Rbase_t <= Rc_t
                      else (Rc_t/Rbase_t)**(l+1.)) * Rbase_t**3/(Rc_t*R**2)
            A[0,iX] = Cb*Rl3*RtbRl3
            A[1,iX] = Cb*(g0/g_m_t)*Rl3*(RtRCl - RbRCl)
            A[2,iX] = gdrho_t*M_t_
            A[4,iX] = Phat_t/R
        return np.linalg.solve(A, rhs)[iw]

    print(f"\nIMPULSE TEST  solve_for={solve_for}  Te={T_e_0_t:.1f}  "
          f"rho_l={rho_l}  rho_c={rho_c}  rho_m={rho_m}")
    errH = errG = 0.0
    for i, l in enumerate(range(2, lmax_t+1)):
        thd, tgd = ref_w(l,1,0), ref_w(l,0,1)
        eH, eG = TH_code[i]/thd - 1, TG_code[i]/tgd - 1
        errH, errG = max(errH, abs(eH)), max(errG, abs(eG))
        print(f"l={l:2d}  TH_code/TH_ref-1 = {eH:+.3e}   TG_code/TG_ref-1 = {eG:+.3e}")
    print(f"  max|TH err| = {errH:.3e}   max|TG err| = {errG:.3e}"
          f"   -> {'PASS' if max(errH,errG) < 1e-8 else 'FAIL'}")


# %%
    

    print(f'\nTotal model runtime: {(time.perf_counter() - t_begin):.1f}s')