# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:37:18 2026

@author: vand_t1
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from palettable import scientific as scm
from cmcrameri import cm
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time
import os
import pandas as pd
import pickle 

start = time.time()

#######################################
"""
Model for the variable thickness deformations of Beuthe (2008).
Current model works with Beuthe's equations 75 and 76, 
neglecting tangential loading.

This requires implementation of the differential operator A(a;b).
Beuthe does not give a spectral method for this, but Kalousova et al. (2012)
do. The approach worked on in this code: 
    - Applying Kalousova et al. (2012)'s approach in spectral domain directly.
""" 


### INPUTS FOR DISPLACEMENT EQUATION ###
# theta : colatitude, 0 to pi [rad]
# phi : longitude, 0 to 2pi [rad]
# R : Reference radius, mid-plane of the shell [m]
# nu : Poisson's ratio [-]
# E : Young's Modulus [N/m^2]
# T_e : Elastic Thickness [m]
# g0 : Vertical gravitational acceleration at the surface [m/s^2]
# g_m : Vertical gravitational acceleration at the crust-mantle boundary [m/s^2]
# Omega : Tangential force potential [N ?]


### OUTPUTS ###
# w : Transverse displacement
# F : Stress function


### EQUATIONS ###
# D : Flexural rigidity [Nm]
# D = E*T_e^3/(12*(1-nu^2))

# K : Extensional rigidity [N/m]
# K = E*T_e/(1-nu^2)

# q_lm : Lithospheric loading in spherical harmonics
# q_lm = w_lm * (drho_l*g0)

# alpha = 1/(E*T_e)
# xi = R**2*K/D
# eta = xi/(1+xi)

### OPERATORS ###
# Delta w = d^2/dtheta^2 (w) + cot(theta)*d/dtheta (w) 
#            + csc^2(theta) d^2/dtheta^2 (w)
# (= nabla^2)

# Delta_p w = (Delta + 2) w

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## INPUTS
nu = 0.25       # Poisson's ratio
E = 100.0e9     # Young's Modulus
rho_c = 2900.   # Density of the crustal manterial
rho_m = 3500.   # Density of the mantle material
rho_l = rho_c
drho = rho_m - rho_c


# Set all lmax runs
LMAX_RUNS = [20, 25, 30, 30]

# Set whether rotation of inputs is applied or not - Verification method
rotate_angles = (0.0, 0.0, 0.0)

# Set whether output figures are saved or not
Save_Figs = False

# Set color maps
cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cm.davos

# Set Te max resolution and tapering cut & width degrees
lmax_Te_fit = 60    # LSQ fit resolution (top ~5 degrees never trusted)
l_cut       = 40    # working bandlimit of the Te field
taper_width = 10    # fade-out width in degrees

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# %% BASIC FUNCTION DEFINITIONS

def taper(clm, l_cut, width):
    """Smoothly fade coefficients to zero between l_cut-width and l_cut."""
    out = clm.copy()
    for l in range(out.lmax + 1):
        if l <= l_cut - width:
            f = 1.0
        elif l >= l_cut:
            f = 0.0
        else:
            f = 0.5 * (1 + np.cos(np.pi * (l - (l_cut - width)) / width))
        out.coeffs[:, l, :] *= f
    return out

def truncate(clm, lmax):
    return pysh.SHCoeffs.from_array(
        clm.coeffs[:, :lmax+1, :lmax+1].copy(), normalization='4pi')

def make_mode_map(lmax):
    # Build a flat sequence list matching the grid ordering approach
    mode_map = []

    for l_idx in range(lmax + 1):
        for m_idx in range(-l_idx, l_idx + 1):
            mode_map.append((l_idx, m_idx))
    return mode_map

# %% INPUT FUNCTIONS

def load_Temap(lmax_Te_fit):
    # Load in an elastic thickness input map
    Te_file_path =  "grl58258-sup-0002-data_set_1.dat"
    
    df = pd.read_csv(
        Te_file_path, 
        sep='\s+',
        comment='#',
        header=None,
        names=['longitude', 'latitude', 'crustal_thickness_km',
               'heat_flow_mW_m2', 'Te_1e-14_km', 'Te_1e-17_km',
               'T_150km_K', 'depth_1370K_km'],
        usecols=['longitude', 'latitude', 'Te_1e-14_km', 'Te_1e-17_km']
    )
    
    # Extract data as arrays
    lon      = df['longitude'].values
    lat      = df['latitude'].values
    Te_14    = (df['Te_1e-14_km'].values)*1e3   # Convert to m
    # Te_17    = (df['Te_1e-17_km'].values)*1e3   # Convert to m, currently unused

    # Create a parent Te grid that is resolved to high resolution
    print(f"Loading in Te map at high degree resolution (lmax={lmax_Te_fit})...")
    T_e_fit = pysh.SHCoeffs.from_least_squares(Te_14, lat, lon, lmax=lmax_Te_fit)

    return T_e_fit, Te_14


def load_inputs(lmax, lmax_Te_fit, l_cut, taper_width):
    ## Load in topography and gravity data
    pot_clm = pysh.datasets.Mars.GMM3(lmax=lmax)
    topo_clm = pysh.datasets.Mars.MOLA_shape(lmax=lmax)
    
    R = topo_clm.coeffs[0, 0, 0]  # Mean planetary radius
    pot_clm = pot_clm.change_ref(r0=R)  # Downward continue to Mean planetary radius
    
    # Compute the geoid as approximated in Banerdt's formulation
    geoid_clm = pot_clm * R
    
    # Remove 100% of C20
    percent_C20 = 0.0
    topo_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo_clm.coeffs[0, 2, 0]
    geoid_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid_clm.coeffs[0, 2, 0]
    
    # CONSTANTS
    G = pysh.constants.G.value  # Gravitational constant
    gm = pot_clm.gm  # GM given in the gravity model file
    mass = gm / G  # Mass of the planet
    g0 = gm / R**2  # Mean gravitational attraction of the planet
    
    # Load in the parent Temap, taper to l_cut with taper width
    T_e_fit, _ = load_Temap(lmax_Te_fit)
    
    print(f"Tapering Te map to lower degree l_cut={l_cut}")
    T_e_parent = taper(T_e_fit, l_cut, taper_width)
    print("Map loaded.")
    
    return topo_clm, geoid_clm, T_e_parent, R, g0, mass


def derive_D_a(T_e_parent, lmax):
    T_e_grid_hires = T_e_parent.expand(lmax=max(3*l_cut, 2*lmax))
    D_clm = pysh.SHGrid.from_array(E * T_e_grid_hires.data**3 / (12.0 * (1.0 - nu**2))).expand()
    D_clm = pysh.SHCoeffs.from_array(D_clm.coeffs[:,  :2*lmax+1, :2*lmax+1])

    a_clm = pysh.SHGrid.from_array(1.0 / (E * T_e_grid_hires.data)).expand()
    a_clm = pysh.SHCoeffs.from_array(a_clm.coeffs[:,  :2*lmax+1, :2*lmax+1])

    return D_clm, a_clm

# %% INPUT ROTATIONS

def rotate_inputs(rotate_angles, T_e_parent, D_clm, a_clm, topo_clm, geoid_clm):
    print(f"Rotating inputs with angles {rotate_angles}...")
    alpha, beta, gamma = rotate_angles
    T_e_parent = T_e_parent.rotate(alpha, beta, gamma)
    D_clm = D_clm.rotate(alpha, beta, gamma)
    a_clm = a_clm.rotate(alpha, beta, gamma)
    topo_clm = topo_clm.rotate(alpha, beta, gamma)
    geoid_clm = geoid_clm.rotate(alpha, beta, gamma)

    return T_e_parent, D_clm, a_clm, topo_clm, geoid_clm

# %% BEUTHE/KALOUSOVA MATRIX FUNCTIONS

# Map out how the pyshstools array is structured
def find_custom_element(l_param, m_param, xlm_unstr):
    # Find the starting index of degree l in the shtools array (which is l^2)
    block_start = l_param**2
    if m_param == 0:
        offset = 0
    elif m_param > 0:
        offset = m_param
    else:
        offset = l_param + abs(m_param)
    return xlm_unstr[block_start + offset]

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

# Make/identify gaunt cache directory to save or load gaunt coefficient tables
CACHE_DIR = "gaunt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def build_or_load_gaunt(lmax, nu, l_cutoff=None):
    """
    The l_cutoff is an added variable to test whether the coupling terms
    have a significant effect above a certain lmax. For example, when analysing
    to lmax=45, the power difference that occurs between lmax=40 may be largely
    caused by the zonal coupling terms of l0 only. In other words, the total
    power per degree might consist for 90% of the power of l0-terms only, and 
    the remaining 10% of all lm values for that degree l.
    
    l_cutoff is set to the degree from which onward the Gaunt coefficients 
    would be set to the value of l0 only, up to the final lmax. Not specifying
    it means the full Gaunt coefficients will be calculated.
    """
    
    if l_cutoff == None:
        l_cutoff = lmax
    
    if l_cutoff == lmax:
        plan_path = os.path.join(CACHE_DIR, f"gaunt_plan_v3_lmax{lmax}_nu{nu:.4f}.pkl")
    else:
        plan_path = os.path.join(CACHE_DIR, f"gaunt_plan_v3_lmax{lmax}_nu{nu:.4f}_lcut={l_cutoff}.pkl")

    
    mode_map = make_mode_map(lmax)
    
    # Load a Gaunt plan if it exists in the cache
    if os.path.exists(plan_path):
        print(f"Loading Gaunt plan from cache: {plan_path}")
        t_load = time.perf_counter()
        with open(plan_path, 'rb') as fh:
            cached = pickle.load(fh)
        assembly_plan = cached['plan']
        print(f"  Loaded {len(assembly_plan):,} entries in "
              f"{time.perf_counter()-t_load:.2f}s")
    
    # # If not existing yet, calculate all the coefficients and save them to cache
    # else:
    #     print(f"Building Gaunt plan (first run for this lmax (={lmax}) — will be cached)...")
    #     t_build = time.perf_counter()
    #     assembly_plan = []
    
    #     for i, (l_val, m_val) in enumerate(mode_map):
    #         for j, (l_prime, m_prime) in enumerate(mode_map[i:], start=i):
    #             L_entries = []
    #             min_L = abs(l_val - l_prime)
    #             max_L = l_val + l_prime
    
    #             for L in range(min_L, max_L + 1):
    #                 if (l_val + l_prime + L) % 2 != 0:
    #                     continue
    #                 w_coef_A = W_numeric_A(l_val, l_prime, L, nu)
    #                 w_coef_B = W_numeric_B(l_val, l_prime, L, nu)
    #                 if w_coef_A == 0.0 and w_coef_B == 0.0:
    #                     continue
    
    #                 # Evaluate Gaunt for all M at once, keep only nonzero (nz) values
    #                 M_vals = np.arange(-L, L + 1)
    #                 q_vals = np.array([get_real_gaunt(l_val, m_val, L, M, l_prime, m_prime)
    #                                    for M in M_vals])
    #                 nz_mask = np.abs(q_vals) > 1e-15
    #                 if not np.any(nz_mask):
    #                     continue
    
    #                 # Store (M_offset_into_slice, w_coef_A*q, w_coef_B*q) for the nonzero M
    #                 M_offsets = np.where(nz_mask)[0].astype(np.int16)
    #                 wAq = (w_coef_A * q_vals[nz_mask]).astype(np.float64)
    #                 wBq = (w_coef_B * q_vals[nz_mask]).astype(np.float64)
    #                 L_entries.append((L, M_offsets, wAq, wBq))
    
    #             if L_entries:
    #                 assembly_plan.append((i, j, L_entries))
    
    #     build_time = time.perf_counter() - t_build
    #     print(f"  Built {len(assembly_plan):,} entries in {build_time:.1f}s — saving...")
    #     with open(plan_path, 'wb') as fh:
    #         pickle.dump({'lmax': lmax, 'nu': nu, 'plan': assembly_plan}, fh,
    #                     protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"  Saved to {plan_path}  "
    #           f"({os.path.getsize(plan_path)/1e6:.1f} MB)")

    # If not existing yet, calculate all the coefficients and save them to cache
    else:
        if l_cutoff != lmax:
            print(f"Building Gaunt plan (first run for this lmax (={lmax} with l_cutoff={l_cutoff}) — will be cached)...")
        else:
            print(f"Building Gaunt plan (first run for this lmax (={lmax}) — will be cached)...")

        
        t_build = time.perf_counter()
        assembly_plan = []
    
        for i, (l_val, m_val) in enumerate(mode_map):
            if l_val <= l_cutoff:
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
                        wAq = (w_coef_A * q_vals[nz_mask]).astype(np.float64)
                        wBq = (w_coef_B * q_vals[nz_mask]).astype(np.float64)
                        L_entries.append((L, M_offsets, wAq, wBq))
        
                    if L_entries:
                        assembly_plan.append((i, j, L_entries))
        
            else: # between l_cutoff and lmax set all non-zonal terms to zero
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
        
                        q0 = get_real_gaunt(l_val, m_val, L, 0, l_prime, m_prime)
                        if abs(q0) > 1e-15:          # nonzero only when m_val == m_prime
                            L_entries.append((L,
                                np.array([L], dtype=np.int16),          # offset of M=0
                                np.array([w_coef_A * q0]),
                                np.array([w_coef_B * q0])))
        
                    if L_entries:
                        assembly_plan.append((i, j, L_entries))
        
        build_time = time.perf_counter() - t_build
        print(f"  Built {len(assembly_plan):,} entries in {build_time:.1f}s — saving...")
        with open(plan_path, 'wb') as fh:
            pickle.dump({'lmax': lmax, 'nu': nu, 'plan': assembly_plan}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved to {plan_path}  "
              f"({os.path.getsize(plan_path)/1e6:.1f} MB)")

    return assembly_plan



# %% TANGENTIAL LOAD EQUATION

def Omega_clm(T_e_parent, topo_clm, w_clm, dc_clm, drho_m_clm):
    
    c = 50e3
    M = 100e3
    
    base_drho=50e3,
    top_drho=0
    
    v1v = nu/(1-nu)
    Te_clm = T_e_parent
    drhol = rho_c - rho_l
    
    RCR = (R-c)/R
    RTeR = (R - Te_clm) / R
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    R_base_drho = R - base_drho
    R_top_drho = R - top_drho
    R_drho_mid = (R_top_drho + R_top_drho) / 2.0 # TODO: Check if correct?
    
    if Te_clm <= c:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_c / rhobar) / RTeR**2
    else:
        gTe = g0 * (1.0 + (RTeR**3 - 1.0) * rho_m / rhobar) / RTeR**2

    if top_drho <= c:
        gdrho = (
            g0
            * (1.0 + ((R_drho_mid / R) ** 3 - 1.0) * rho_c / rhobar)
            / (R_drho_mid / R) ** 2
        )
    else:
        gdrho = (
            g0
            * (1.0 + ((R_drho_mid / R) ** 3 - 1) * rho_m / rhobar)
            / (R_drho_mid / R) ** 2
        )
    
    # Gravity at moho depth
    gmoho = g0 * (1.0 + (RCR**3 - 1.0) * rho_c / rhobar) / RCR**2
    
    Omega_clm = (
                v1v * rho_l * g0 * Te_clm * topo_clm / R
                - (
                    drhol * g0 * v1v * Te_clm
                    - rho_c * gmoho * (c if c < Te_clm else 0)
                    # If crust-mantle interface below Te, no tangential load associated
                    - rho_m * gTe * np.max([Te_clm - c, 0])
                    # If crust-mantle interface below Te, no tangential load associated
                )
                * w_clm
                / R
                + v1v * drho * gmoho * np.max([Te_clm - c, 0]) * (dc_clm - w_clm) / R
                - 0.5
                * v1v
                * drho_m_clm
                * gdrho
                * (Te_clm - top_drho)
                * (np.min([M, Te_clm - top_drho]) if top_drho < Te_clm else 0)
                # If mantle load below Te, no tangential load associated
                / R
                )
    
    return Omega_clm




# %% GEOID EQUATION

def G_clm(lmax, topo_clm, w_clm, dc_clm, drho_clm):
    
    l = np.arange(lmax+1, dtype=float)
    
    base_drho = 50e3,   # base of mantle density anomaly
    top_drho = 0        # top depth of mantle density anomaly
    c = 50e3            # mean crustal thickness
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    phi = (R - c)/R
    Mt = R - top_drho   # Radius of top depth mantle density anomaly
    Mb = R - base_drho  # Radius of bottom depth mantle density anomaly
    
    
    G_clm = (
        3/(rhobar*(2*l + 1)) 
        * (rho_c*topo_clm
           + drho * (w_clm - dc_clm) * phi**(l+2)
           + drho_clm * R/(l+3) * (((R - Mt)/R)**(l+3) - ((R-Mb)/R)**(l+3)) )
        )
    
    return G_clm




# %% SOLVE BEUTHE/KALOUSOVA MODEL

def solve_beuthe(topo_clm, geoid_clm, D_clm, a_clm, assembly_plan, lmax,
                 R, T_e_0, g0):
    mode_map = make_mode_map(lmax)
    N_modes = len(mode_map)
    
    # Calculate the buoyancy term used in Matrix A, 
    # and the two scaling factors of the two matrices
    Re = R - T_e_0 / 2
    buoy = (Re / T_e_0)**3 * (Re / E) * g0 * (rho_m - rho_c)
    scaler_A = 1.0 / (E * T_e_0**3)
    scaler_B = Re
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Convert the D and alpha coefficients into a vector for the computations
    Dlm_unstr = pysh.shio.SHCilmToVector(D_clm.coeffs)
    alm_unstr = pysh.shio.SHCilmToVector(a_clm.coeffs)

    # Pre-extract D / alpha coefficient slices per degree L
    # One array of length (2L+1) per L, indexed M = -L … +L.
    # Avoids repeated find_custom_element() calls inside the fill loop.
    D_slices = {}
    a_slices = {}
    for L in range(2*lmax + 1):
        block = L * L
        idx_list = [0 if M == 0 else (M if M > 0 else L + abs(M))
                    for M in range(-L, L + 1)]
        flat_idx = np.array([block + off for off in idx_list], dtype=np.int32)
        D_slices[L] = Dlm_unstr[flat_idx]
        a_slices[L] = alm_unstr[flat_idx]
        
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("Initializing sparse matrices...")
    diag_a = np.zeros(N_modes, dtype=np.float64)
    diag_b = np.zeros(N_modes, dtype=np.float64)
    
    for i, (l_val, m_val) in enumerate(mode_map):
        d_l = -l_val * (l_val + 1) + 2
        diag_a[i] = ((Re / T_e_0)**3 / E) * d_l
        diag_b[i] = -1.0 * d_l
    
    matrix_a_l_sparse = sparse.diags(diag_a, format="lil")
    matrix_b_l_sparse = sparse.diags(diag_b, format="lil")
    
    matrix_A_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)
    matrix_B_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)    

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("Filling matrices A and B...")
    for i, j, L_entries in assembly_plan:
        cell_A = 0.0
        cell_B = 0.0
        for L, M_offsets, wAq, wBq in L_entries:
            D_sel = D_slices[L][M_offsets]
            a_sel = a_slices[L][M_offsets]
            cell_A += float(np.dot(D_sel, wAq))   # BLAS dot — no Python loop
            cell_B += float(np.dot(a_sel, wBq))

        val_A = cell_A * scaler_A + (buoy if i == j else 0.0)
        val_B = cell_B * scaler_B

        if val_A != 0.0:
            matrix_A_sparse[i, j] = val_A
            if i != j:
                matrix_A_sparse[j, i] = val_A      
        if val_B != 0.0:
            matrix_B_sparse[i, j] = val_B
            if i != j:
                matrix_B_sparse[j, i] = val_B      


    print("Combining sub-matrices into a sparse 2N x 2N architecture...")
    M_system_sparse = sparse.bmat([
        [matrix_A_sparse,     matrix_a_l_sparse],
        [matrix_b_l_sparse,   matrix_B_sparse]
    ], format="lil")
    
    print("Setting degree 0 and 1 of large matrix to zero...")
    for idx, (l_val, _) in enumerate(mode_map):
        if l_val == 0 or l_val == 1:
            M_system_sparse[idx, :] = 0.0
            M_system_sparse[idx, idx] = 1.0
            M_system_sparse[idx + N_modes, :] = 0.0
            M_system_sparse[idx + N_modes, idx + N_modes] = 1.0
    
    # Convert to CSR (compressed sparse row) format for faster calculations
    M_system_csr = M_system_sparse.tocsr()
    
    
    
    
    # Calculate the RHS components
    print(f"Solving structural displacement vector for lmax={lmax}")
    factors_y_lm = (Re / T_e_0)**3 * (rho_c * g0 * Re) / E
       
    # True topographic loading case, negative to match displacement
    y_lm_topo = -factors_y_lm * (topo_clm.coeffs - geoid_clm.coeffs)
    y_lm_unstr = pysh.shio.SHCilmToVector(y_lm_topo)
    y_lm_str = np.array([find_custom_element(l_v, m_v, y_lm_unstr) for l_v, m_v in mode_map])
    
    
    # Term in second equation due to tangential loading potential
    r_lm_2 = -nu*(-l*(l+1)+2)*a_clm*rho_c* g0*T_e_clm*topo_clm
    
    
    
    
    rhs_dense = np.concatenate([y_lm_str, np.zeros(N_modes)])
    
    print("Setting degree 0 and 1 of rhs vector to zero...")
    for idx, (l_val, m_val) in enumerate(mode_map):
        if l_val == 0 or l_val == 1:
            rhs_dense[idx] = 0.0
            rhs_dense[idx + N_modes] = 0.0
    
    # Run linear solver
    sol_vector = spla.spsolve(M_system_csr, rhs_dense)
    w_sol = sol_vector[:N_modes]
    F_sol = sol_vector[N_modes:]
    
    # Map flat 1D solution back into 3D SH shape
    w_coeffs_np = np.zeros((2, lmax + 1, lmax + 1))
    for idx, (l_val, m_val) in enumerate(mode_map):
        if m_val >= 0:
            w_coeffs_np[0, l_val, m_val] = float(w_sol[idx])
        else:
            w_coeffs_np[1, l_val, abs(m_val)] = float(w_sol[idx])
     
    # Finally, transform the Beuthe solution vector into pysh coefficient and grid format
    w_sol_clm_beuthe = pysh.SHCoeffs.from_array(w_coeffs_np, normalization='4pi')

    return w_sol_clm_beuthe









# %% SET FIGURE-SAVING DIRECTORY
home_dir = os.path.expanduser("~")
save_path = os.path.join(home_dir,
                         "Variable-Thickness-Elastic-Loading-Model",
                         "Variable-Thickness-Elastic-Loading-Model",
                         "Plots",
                         "FixedCodeResults")
os.makedirs(save_path, exist_ok=True)


# %% RUN CONVERGENCE LOOP OVER MULTIPLE LMAX

LMAX_REF  = max(LMAX_RUNS)

# ---- load everything ONCE at the highest resolution -------------------
topo_parent, geoid_parent, T_e_parent, R, g0, mass = load_inputs(LMAX_REF, lmax_Te_fit, l_cut, taper_width)
T_e_0 = T_e_parent.coeffs[0, 0, 0]

solutions = {}
fig, ax = plt.subplots(figsize=(10, 7))

for k, lmax_run in enumerate(LMAX_RUNS):
    # fresh truncated copies per run — parents are never modified
    topo_clm  = truncate(topo_parent, lmax_run)
    geoid_clm = truncate(geoid_parent, lmax_run)
    D_clm, a_clm = derive_D_a(T_e_parent, lmax_run)
    
    if k == 2:
        l_cutoff = lmax_run-5
        assembly_plan = build_or_load_gaunt(lmax_run, nu, l_cutoff)
    else:
        l_cutoff = 0
        assembly_plan = build_or_load_gaunt(lmax_run, nu)

    do_rotation_check = any(a != 0.0 for a in rotate_angles)
    for rotation in ([0, 1] if do_rotation_check else [0]):
        linestyle = 'solid' if rotation == 0 else 'dashed'
        if rotation == 1:
            _, D_clm, a_clm, topo_clm, geoid_clm = rotate_inputs(
                rotate_angles, T_e_parent, D_clm, a_clm, topo_clm, geoid_clm)
        w_clm = solve_beuthe(topo_clm, geoid_clm, D_clm, a_clm,
                             assembly_plan, lmax_run, R, T_e_0, g0)
        solutions[lmax_run, rotation, l_cutoff] = w_clm
        label = (f'lmax={lmax_run}' 
                 + (f', rotated {rotate_angles}' if rotation else '')
                 + (f', coupling cut at l={l_cutoff}' if k == 2 else ''))
        if ax is None:
            fig, ax = w_clm.plot_spectrum(show=False, legend=label)
        else:
            w_clm.plot_spectrum(ax=ax, show=False, legend=label,
                                plot_dict={'linestyle': linestyle})

ax.legend()
ax.set_title(f'Power spectra of w lmax comparisons (TSA-B, Plesa Te map, lmax_Te={lmax_Te_fit} cut at l_cut={l_cut})')
plt.tight_layout()
plt.savefig(f'{save_path}/'
            f'Power_spectra_w_comparisons_lmax={lmax_run}_lcutTest', dpi = 200)
plt.show()



# %% CALCULATE & PLOT RESIDUALS BETWEEN LMAX RUNS

# S_ref = solutions[LMAX_REF, 0].spectrum()
# fig2, ax2 = plt.subplots(figsize=(8, 5))
# for lmax_run in LMAX_RUNS[:-1]:
#     do_rotation_check = any(a != 0.0 for a in rotate_angles)
#     for rotation in ([0, 1] if do_rotation_check else [0]):
#         linestyle = 'solid' if rotation == 0 else 'dashed'
#         S = solutions[lmax_run, rotation].spectrum()
#         l = np.arange(2, lmax_run + 1)
#         ax2.semilogy(l, np.abs(S[2:] / S_ref[2:lmax_run+1] - 1.0),
#                      marker='.', 
#                      label=f'lmax={lmax_run} vs {LMAX_REF}'+ 
#                      (f', rotated {rotate_angles}' if rotation else ''),
#                      linestyle=linestyle)
# ax2.set_xlabel('degree l')
# ax2.set_ylabel(r'$|S_l / S_l^{\rm ref} - 1|$')
# ax2.legend(); ax2.grid(True)
# ax2.set_title(f'Residual between lmax_run and lmax_ref={LMAX_REF}')
# plt.tight_layout()
# plt.savefig(f'{save_path}/'
#             f'Power_spectra_w_comparisons_residuals_lmax={lmax_run}_lcutTest', dpi = 200)
# plt.show()

S_orig = solutions[LMAX_RUNS[-1], 0, 0].spectrum()
S_pert = solutions[LMAX_RUNS[-1], 0, LMAX_RUNS[-1]-5].spectrum()
fig2, ax2 = plt.subplots(figsize=(8, 5))

l = np.arange(2, lmax_run + 1)
ax2.plot(l, np.abs(S_pert[2:] / S_orig[2:] - 1.0)*100,
             marker='.', 
             label=f'lmax={lmax_run} cut at l={lmax_run-5} vs {LMAX_REF}'+ 
             (f', rotated {rotate_angles}' if rotation else ''),
             linestyle=linestyle)
ax2.set_xlabel('degree l')
ax2.set_ylabel(r'$|S_l / S_l^{\rm ref} - 1|$, [%]')
ax2.legend(); ax2.grid(True)
ax2.set_title(f'Residual between lmax_run and lmax_ref={LMAX_REF}')
plt.tight_layout()
plt.savefig(f'{save_path}/'
            f'Power_spectra_w_comparisons_residuals_lmax={lmax_run}_lcutTest_linearYaxis', dpi = 200)
plt.show()

# %% PERFORM CHECKS ON TE COEFFS AND ALPHA DECAY


# T_e_parent60, _     = load_Temap(lmax_Te_fit=60)
# T_e_parent70, Te_14 = load_Temap(lmax_Te_fit=70)
# T_e_parent80, _     = load_Temap(lmax_Te_fit=80)

# l_common = 60   # compare over the band all three fits share
# d80_60 = truncate(T_e_parent80, l_common) - truncate(T_e_parent60, l_common)
# d70_60 = truncate(T_e_parent70,   l_common) - truncate(T_e_parent60, l_common)
# d80_70 = truncate(T_e_parent80, l_common) - truncate(T_e_parent70, l_common)

# S60 = T_e_parent60.spectrum()
# rel8060 = np.sqrt(d80_60.spectrum() / S60)   # per-degree relative amplitude
# rel7060 = np.sqrt(d70_60.spectrum() / S60)
# rel8070 = np.sqrt(d80_70.spectrum() / S60)

# fig, ax = plt.subplots(figsize=(8, 4))
# l = np.arange(l_common + 1)
# ax.semilogy(l[2:], rel8060[2:], '.-', label='|fit80 − fit60| / fit60')
# ax.semilogy(l[2:], rel7060[2:], '.-', label='|fit70 − fit60| / fit60')
# ax.semilogy(l[2:], rel8070[2:], '.-', label='|fit80 − fit70| / fit60')
# ax.set_xlabel('degree l'); ax.set_ylabel('relative coefficient difference')
# ax.legend(); ax.grid(True)
# plt.title('Coefficient difference of Te between different Te lmax resolution')
# plt.savefig(f'{save_path}/'
#             f'Coefficient difference of Te between different Te lmax resolution',
#             dpi = 200)
# plt.show()


# # 2. Is the synthesized field still physical?
# g = T_e_parent.expand(lmax=3*l_cut)
# print('Te min/max on hires grid [km]:',
#       g.data.min()/1e3, g.data.max()/1e3)
# print('Te data min/max [km]:', Te_14.min()/1e3, Te_14.max()/1e3)

# # 3. Does alpha's spectrum decay over the range the solver consumes?
# a_full = pysh.SHGrid.from_array(1.0/(E*g.data)).expand()
# a_full.plot_spectrum(
#     show=False, 
#     plot_dict = dict(title='Power spectrum decay of alpha per harmonic degree')
#     )

# # %% PLOT INPUT MAPS

# print("Plotting input Te map, flexural rigidity D and parameter alpha")
# fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
# T_e_parent.expand(lmax=LMAX_RUNS[-1]).plot(ax=ax1, cmap=cmap2, colorbar='right', cb_label= 'Synthetic T_e map, m')
# D_clm.expand(lmax=LMAX_RUNS[-1]).plot(ax=ax2, cmap=cmap2, colorbar='right', cb_label= 'Synthetic D map')
# a_clm.expand(lmax=LMAX_RUNS[-1]).plot(ax=ax3, cmap=cmap2, colorbar='right', cb_label= 'Synthetic alpha map') 
# plt.tight_layout()
# plt.savefig(f'{save_path}/'
#             f'Input maps Te, D and a', dpi = 200)
# plt.show()



# %% PLOT 2D DEFLECTION MAP FOR HIGHEST LMAX

# Transform meters to kilometers
w_grid_fine = pysh.SHGrid.from_array(solutions[LMAX_RUNS[-1],0, 0].expand().data/1e3)

sol_lmax_maxmin1 = solutions[LMAX_RUNS[-2],0, LMAX_RUNS[-1]-5].coeffs[:, :LMAX_RUNS[-2]+1, :LMAX_RUNS[-2]+1]
sol_lmax_max = solutions[LMAX_RUNS[-1],0, 0].coeffs[:, :LMAX_RUNS[-2]+1, :LMAX_RUNS[-2]+1]
w_diff = pysh.SHCoeffs.from_array(sol_lmax_max - sol_lmax_maxmin1).expand()

print(f"Plotting 2D deflection map for lmax={LMAX_RUNS[-1]}")
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 10))
w_grid_fine.plot(ax=ax1, cmap=cmap1, colorbar='right', cb_label='w [km]')
ax1.set_title(f'TSA-B  Beuthe model solution (Te = PlesaMap, lmax={LMAX_RUNS[-1]}, TeCut={l_cut})')
ax1.contour(w_grid_fine.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper")
w_diff.plot(ax=ax2, cmap=cmap1, colorbar='right', cb_label='w difference [m]', cmap_limits=[-776, 590])
ax2.set_title(f'Difference in w of TSA-B between lmax={LMAX_RUNS[-1]} and lmax={LMAX_RUNS[-2]} cut at l={LMAX_RUNS[-1]-5}')
plt.tight_layout()
plt.savefig(f'{save_path}/'
            f'2D deflection map w, lmax={LMAX_RUNS[-1]}, cut at l={LMAX_RUNS[-1]-5}', dpi = 200)
# ax1.grid()
# ax2.grid()
plt.show()


end = time.time()
print("\n--- Entire Model Run Complete ---")
print("Total runtime:", round(end - start, 1), "seconds")

