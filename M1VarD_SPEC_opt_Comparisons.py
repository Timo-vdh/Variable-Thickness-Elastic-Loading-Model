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

# Set maximum spherical harmonic degree to perform all calculations
lmax = 25

# Initialize a coefficient shape for cosine-sine format as used in pyshtools
shape = (2, lmax + 1, lmax + 1)

# Set whether rotation of inputs is applied or not - Verification method
Rotated = True
Rotated_SaveFig = False
# Set whether output figures are saved or not
Save_Figs = False

# Set color maps
cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cm.davos


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load in an elastic thickness input map
Te_inputs_subfolder = "Elastic Thickness Input Maps/"
Te_file_path = f"{Te_inputs_subfolder}grl58258-sup-0002-data set 1.dat"
T_e_type = 'Input_TeMap_Plesa2018'

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

# # Check whether data format is collected correctly
# print(df.head())
# print(df.shape)

# Extract data as arrays
lon      = df['longitude'].values
lat      = df['latitude'].values
Te_14    = (df['Te_1e-14_km'].values)*1e3   # Convert to m
Te_17    = (df['Te_1e-17_km'].values)*1e3   # Convert to m

T_e_clm = pysh.SHCoeffs.from_least_squares(Te_14, lat, lon, lmax)
T_e_grid = T_e_clm.expand()

# %%

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
############################################
### CREATING A SYNTHETIC VARIABLE Te MAP ###
############################################
""" 
Three different types of Te input maps:
    1. Constant Te map
    2. Random map with a small variation to the constant thickness
       (used to check whether small values of orders m impact the final 
        solution a lot.)
    3. Harmonic distribution of Te similar to Kalousova model I 
    4. Harmonic distribution of Te similar to Kalousova model III
"""
# Make a colatitude range for the harmonic T_e functions to be created over
theta_range = np.linspace(0, 180, 2*(lmax+1)+1)


# # 1. Making a constant T_e map
# T_e_type  = 'Constant_TeMap'
# T_e_mean = 150e3
# T_e_array = T_e_mean * np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])


# # 2. Make a Random Te map 
# # (Random model with only small variations to check small variations from constant case)
# seed = 1; l_corner = lmax+1; beta = 3.0
# power = np.zeros(lmax + 1)
# for li in range(2, lmax+1):
#     if li <= l_corner:
#         power[li] = 1e-8
#     else:
#         power[li] = (l_corner / li) ** beta

# # Make a random coefficient map
# T_e_type = 'Random_TeMap'
# T_e_clm = pysh.SHCoeffs.from_random(power, lmax=lmax, seed=seed)
# T_e_array = T_e_clm.expand().to_array()*1e3 + 150e3


# # 3. Make a harmonically varying Te map (same as Kalousova)
# # Make T_e distribution - Model I of Kalousova
# T_e_type = 'Harmonic_TeMap_MI'
# T_e_I = []
# start_trans = 80
# stop_trans = 100
# phi = np.pi * (theta_range - start_trans)/(100 - start_trans)
# transition_T_e_I = 125e3 + 75e3*np.cos(phi)

# for i, theta in enumerate(theta_range):
#     if theta <= start_trans:
#         T_e_I.append(200e3)
#     elif theta >= 100:
#         T_e_I.append(50e3)
#     else:
#         T_e_I.append(transition_T_e_I[i])
# T_e_I = np.array(T_e_I)
# T_e_array = np.tile(T_e_I.reshape(-1, 1), (1, 4*(lmax+1)+1))


# # 4.a Make harmonic T_e distribution - Model III of Kalousova - latitudinal
# T_e_type = 'Harmonic_TeMap_MIIIa'
# T_e_III = 100e3 + 50e3*np.cos(10*np.radians(theta_range))
# T_e_array = np.tile(T_e_III.reshape(-1, 1), (1, 4*(lmax+1)+1))


# # 4.b Make harmonic T_e distribution - Model III of Kalousova - longitudinal
# phi_range = np.linspace(0, 360, 4*(lmax+1)+1)
# T_e_type = 'Harmonic_TeMap_MIIIb'
# T_e_IIIa = 100e3 + 50e3*np.cos(10*np.radians(theta_range))
# T_e_IIIb = 100e3 + 50e3*np.cos(15*np.radians(phi_range))
# T_e_arraya = np.tile(T_e_IIIa.reshape(-1, 1), (1, 4*(lmax+1)+1))
# T_e_arrayb = np.tile(T_e_IIIb, (2*(lmax+1)+1, 1))
# T_e_array = T_e_arraya + T_e_arrayb


# # Convert to pyshtools classes
# T_e_grid = pysh.SHGrid.from_array(T_e_array)
# T_e_clm = T_e_grid.expand() 




D_grid = pysh.SHGrid.from_array(E * T_e_grid.data**3 / (12.0 * (1.0 - nu**2)))
D_clm = D_grid.expand()

a_grid = pysh.SHGrid.from_array(1.0 / (E * T_e_grid.data))
a_clm = a_grid.expand()



if Rotated == True:
    T_e_clm.expand().plot()
    T_e_clm = T_e_clm.rotate(0, 90, 0)
    T_e_clm.expand().plot()
    
    D_clm.expand().plot()
    D_clm = D_clm.rotate(0, 90, 0)
    D_clm.expand().plot()
    
    a_clm.expand().plot()
    a_clm = a_clm.rotate(0, 90, 0)
    a_clm.expand().plot()

    topo_clm = topo_clm.rotate(0, 90, 0)
    geoid_clm = geoid_clm.rotate(0, 90, 0)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # Plot the original and rotated topography and Te maps
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

# if Rotated == True:
#     topo_grid_org.plot(ax=ax1, cmap=cmap1, colorbar='right', cb_label='w [m]')
# else:
#     topo_clm.expand().plot(ax=ax1, cmap=cmap1, colorbar='right', cb_label='w [m]')
# ax1.set_title('Topography original')
# if Rotated == True:
#     topo_grid.plot(ax=ax2, cmap=cmap1, colorbar='right', cb_label='w [m]')
#     ax2.set_title('Topography rotated 90 deg')
# else:
#     topo_clm.expand().plot(ax=ax2, cmap=cmap1, colorbar='right', cb_label='w [m]')
#     ax2.set_title('Topography (no rotation applied)')
# if Rotated == True:
#     T_e_grid_org.plot(ax=ax3, cmap=cmap2, colorbar='right', cb_label='w [m]')
# else:
#     T_e_grid.plot(ax=ax3, cmap=cmap2, colorbar='right', cb_label='w [m]')
# ax3.set_title('Te original')
# T_e_grid.plot(ax=ax4, cmap=cmap2, colorbar='right', cb_label='w [m]')
# if Rotated == True:
#     ax4.set_title('Te rotated 90 deg')
# else:
#     ax4.set_title('Te (no rotation applied)')
    
# plt.tight_layout()
# if Save_Figs == True and Rotated_SaveFig == False: 
#     plt.savefig(f'Plots/M1VarD_SPEC_opt results/InputTopoTe_{T_e_type}_lmax{lmax}.png', dpi=200)
# elif Save_Figs == True and Rotated_SaveFig == True:
#     plt.savefig(f'Plots/M1VarD_SPEC_opt Rotations results/InputTopoTe_Rot={Rotated}_{T_e_type}_lmax{lmax}.png', dpi=200)
# plt.show()

# %% BEUTHE FUNCTIONS

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
####################################################
### DEFINE FUNCTIONS FOR THE MATRIX CALCULATIONS ###
####################################################

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
        return [(m,  1.0/np.sqrt(2)+0j),
                (-m, (-1)**m / np.sqrt(2)+0j)]
    else:
        absm = abs(m)
        return [(-absm,  1j / np.sqrt(2)),
                ( absm, -(-1)**absm * 1j / np.sqrt(2))]


_sqrt4pi = np.sqrt(4.0 * np.pi)


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


# %% TURCOTTE FUNCTIONS

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#############################################################
### DEFINE TURCOTTE EQUATION FOR VARIABLE THICKNESS SHELL ###
#############################################################

# Simplified form of constant thickness thin shell approximation applied to a 
# shell of variable thickness, following Kalousova et al. (2012) definition
def C_l_functional(l_val, nu, E, T_e_local, Re, rho_m, rho_c, g0):
    tau   = E * T_e_local / (Re**2 * (rho_m - rho_c) * g0)
    sigma = tau / (12*(1 - nu**2)) * (T_e_local / Re)**2
    d_b1  = (l_val**3*(l_val+1)**3
             - 4*l_val**2*(l_val+1)**2
             + 4*l_val*(l_val+1))
    num   = l_val*(l_val+1) - (1 - nu)
    denom = sigma*d_b1 + tau*(l_val*(l_val+1) - 2) + l_val*(l_val+1) - (1 - nu)
    return num / denom


# Pre-calculate the density-ratio term
rho_term = -rho_c / (rho_m - rho_c)

# %% BEUTHE SYSTEM SOLVER INITIALIZATIONS

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#########################################################################
### STRUCTURAL EXECUTION LOOP FOR SPECTRAL CALCULATIONS BEUTHE METHOD ###
#########################################################################

# Build a flat sequence list matching the grid ordering approach
mode_map = []

# 2D MODE MAP
for l_idx in range(lmax + 1):
    for m_idx in range(-l_idx, l_idx + 1):
        mode_map.append((l_idx, m_idx))

N_modes = len(mode_map)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Run the spectral calculations
print(f"\n--- Starting Pure Numeric Matrix Generation (lmax = {lmax}) ---")

# Set the T_e average value and reference radius Re based on average T_e
T_e_0 = T_e_clm.coeffs[0,0,0]
Re = R - T_e_0/2
    
# Precalculate the buoyancy term used in Matrix A, and the two scaling factors 
# of the two matrices
buoy = (Re / T_e_0)**3 * (Re / E) * g0 * (rho_m - rho_c)
scaler_A = 1.0 / (E * T_e_0**3)
scaler_B = Re



Dlm_unstr = pysh.shio.SHCilmToVector(D_clm.coeffs)
alm_unstr = pysh.shio.SHCilmToVector(a_clm.coeffs)

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
# Pre-extract D / alpha coefficient slices per degree L
# One array of length (2L+1) per L, indexed M = -L … +L.
# Avoids repeated find_custom_element() calls inside the fill loop.
D_slices = {}
a_slices = {}
for L in range(lmax + 1):
    block = L * L
    idx_list = [0 if M == 0 else (M if M > 0 else L + abs(M))
                for M in range(-L, L + 1)]
    flat_idx = np.array([block + off for off in idx_list], dtype=np.int32)
    D_slices[L] = Dlm_unstr[flat_idx]
    a_slices[L] = alm_unstr[flat_idx]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# %% GAUNT COEFFICIENT TABLES PRE-CALCULATION OR LOADING
"""
The Gaunt coefficient calculations in the main Beuthe loop take by far the 
longest time to calculate, which explodes when increasing lmax.
Since these values only depend on lmax (specifically l, l', L, m, m' M, which
are defined up to lmax) and the Poisson's ratio nu, these values can be 
precalculated, stored in a subfolder of this repository, and loaded for 
specific lmax and Poisson's ratios.

This section precomputes the non-zero coefficients if they do not exist yet,
or loads them from the cache directory otherwise. 

Note: the first time computing these coefficients for a certain lmax takes 
very long (~12 hours for lmax = 50).
"""

# Make/identify gaunt cache directory to save or load gaunt coefficient tables
CACHE_DIR = "gaunt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
plan_path = os.path.join(CACHE_DIR, f"gaunt_plan_lmax{lmax}_nu{nu:.4f}.pkl")

# Load a Gaunt plan if it exists in the cache
if os.path.exists(plan_path):
    print(f"Loading Gaunt plan from cache: {plan_path}")
    t_load = time.perf_counter()
    with open(plan_path, 'rb') as fh:
        cached = pickle.load(fh)
    assembly_plan = cached['plan']
    print(f"  Loaded {len(assembly_plan):,} entries in "
          f"{time.perf_counter()-t_load:.2f}s")

# If not existing yet, calculate all the coefficients and save them to cache
else:
    print(f"Building Gaunt plan (first run for this lmax (={lmax}) — will be cached)...")
    t_build = time.perf_counter()
    assembly_plan = []

    for i, (l_val, m_val) in enumerate(mode_map):
        for j, (l_prime, m_prime) in enumerate(mode_map[i:], start=i):
            L_entries = []
            min_L = abs(l_val - l_prime)
            max_L = min(l_val + l_prime, lmax)

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

    build_time = time.perf_counter() - t_build
    print(f"  Built {len(assembly_plan):,} entries in {build_time:.1f}s — saving...")
    with open(plan_path, 'wb') as fh:
        pickle.dump({'lmax': lmax, 'nu': nu, 'plan': assembly_plan}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {plan_path}  "
          f"({os.path.getsize(plan_path)/1e6:.1f} MB)")

print("Assembling coupling matrices (plan + BLAS dot)...")
t_fill = time.perf_counter()



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
            matrix_A_sparse[j, i] = val_A      # OPT-3: symmetry copy
    if val_B != 0.0:
        matrix_B_sparse[i, j] = val_B
        if i != j:
            matrix_B_sparse[j, i] = val_B      # OPT-3: symmetry copy

print(f"  Matrix fill: {time.perf_counter()-t_fill:.2f}s")

matrix_A_dense = matrix_A_sparse.todense()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
### CREATE AND SOLVE FULL BEUTHE MATRIX SYSTEM OF SUB-MATRICES ###
##################################################################

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

rhs_dense = np.concatenate([y_lm_str, np.zeros(N_modes)])

print("Setting degree 0 and 1 of rhs vector to zero...")
for idx, (l_val, m_val) in enumerate(mode_map):
    if l_val == 0 or l_val == 1:
        rhs_dense[idx] = 0.0
        rhs_dense[idx + N_modes] = 0.0

# Run linear solver
sol_vector = spla.spsolve(M_system_csr, rhs_dense)
w_sol = sol_vector[:N_modes]
# F_sol = sol_vector[N_modes:]

# Map flat 1D solution back into 3D SH shape
w_coeffs_np = np.zeros((2, lmax + 1, lmax + 1))
for idx, (l_val, m_val) in enumerate(mode_map):
    if m_val >= 0:
        w_coeffs_np[0, l_val, m_val] = float(w_sol[idx])
    else:
        w_coeffs_np[1, l_val, abs(m_val)] = float(w_sol[idx])
 
# Finally, transform the Beuthe solution vector into pysh coefficient and grid format
w_sol_clm_beuthe = pysh.SHCoeffs.from_array(w_coeffs_np, normalization='4pi')
w_sol_grid_beuthe = w_sol_clm_beuthe.expand()


# %% TURCOTTE CONSTANT TE MODEL SOLVER
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ###################################################
# ### SOLVE THE TURCOTTE CONSTANT Te DISPLACEMENT ###
# ###################################################
# print("Computing TSA-T (constant Te) …")
# w_coeffs_turcotte = np.zeros((2, lmax+1, lmax+1))

# for l_val in range(2, lmax+1):
#     C_l = C_l_functional(l_val, nu, E, T_e_0, Re, rho_m, rho_c, g0)
#     w_coeffs_turcotte[:, l_val, :l_val+1] = (
#         rho_term * C_l
#         * (topo_clm.coeffs[:, l_val, :l_val+1]
#            - geoid_clm.coeffs[:, l_val, :l_val+1])
#     )

# w_sol_clm_turcotte  = pysh.SHCoeffs.from_array(w_coeffs_turcotte, normalization='4pi')
# w_sol_grid_turcotte = w_sol_clm_turcotte.expand()


# %% TURCOTTE VARIABLE TE MODEL SOLVER


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
###################################################
### SOLVE THE TURCOTTE VARIABLE Te DISPLACEMENT ###
###################################################
# Kalousova eq. 20:
#   w(Ω) = rho_term · Σ_l  C_l(Te(Ω))  ·  h_l(Ω)
#
# where:
#   h_l(Ω) = Σ_m (topo-geoid)_lm · Y_lm(Ω)   [degree-l band-pass of the load]
#   C_l(Te(Ω)) = C_l evaluated with the LOCAL elastic thickness at Ω
#
# For each grid point the local Te value is used to compute C_l, so the
# admittance varies spatially.  This captures first-order lateral variations
# without the full spectral coupling of TSA-B.
#
# Implementation:
#   Step 1. Pre-compute the load SHCoeffs: load_clm = topo_clm - geoid_clm
#   Step 2. For each degree l, band-pass filter load_clm to get degree-l
#           contribution as a spatial grid: h_l_grid[nlat,nlon]
#   Step 3. Evaluate the Te grid to get Te(theta,phi)
#   Step 4. Accumulate: w_varTe_grid += rho_term * C_l(Te_local) * h_l_grid
#           where C_l(Te_local) is evaluated POINTWISE at each grid cell.
#
# Note: for constant Te this exactly reproduces TSA-T.
# ─────────────────────────────────────────────────────────────────────────────
print("Computing TSA-Tv (variable Te, Kalousova eq. 20) …")

load_coeffs = topo_clm.coeffs - geoid_clm.coeffs        # SHCoeffs object, 4pi norm
load_clm = pysh.SHCoeffs.from_array(load_coeffs)

# Get the grid dimensions from the load expansion
load_grid_full = load_clm.expand()
nlat, nlon = load_grid_full.data.shape

# Te grid on the same (nlat, nlon) Driscoll-Healy grid
T_e_grid_data = T_e_clm.expand(lmax=lmax).data
                    
# Pre-compute C_l(Te) at every grid point for every degree l.
# Shape: (lmax+1, nlat, nlon)  -- C_l varies spatially for variable-Te case.
# For large grids/lmax this can be memory-intensive; compute on-the-fly instead.

# Vectorised: compute C_l for every Te value in the grid at once.
tau_grid   = E * T_e_grid_data / (Re**2 * (rho_m - rho_c) * g0)
sigma_grid = tau_grid / (12*(1 - nu**2)) * (T_e_grid_data / Re)**2


# w_varTe_data = np.zeros((nlat, nlon))
w_varTe_coeffs = np.zeros((2, lmax+1, lmax+1))

for l_val in range(2, lmax+1):
    # --- degree-l band of the load ---
    # Zero all coefficients except degree l
    load_l_coeffs = np.zeros((2, lmax+1, lmax+1))
    load_l_coeffs[:, l_val, :l_val+1] = load_clm.coeffs[:, l_val, :l_val+1]
    load_l_clm  = pysh.SHCoeffs.from_array(load_l_coeffs, normalization='4pi')
    h_l_grid    = load_l_clm.expand().data   # shape (nlat, nlon)

    # --- pointwise C_l using the local Te ---
    d_b1       = (l_val**3*(l_val+1)**3
                  - 4*l_val**2*(l_val+1)**2
                  + 4*l_val*(l_val+1))
    num_C      = l_val*(l_val+1) - (1 - nu)
    denom_C    = (sigma_grid * d_b1
                  + tau_grid * (l_val*(l_val+1) - 2)
                  + l_val*(l_val+1) - (1 - nu))
    C_l_grid   = num_C / denom_C                # shape (nlat, nlon)

    # --- accumulate ---
    # w_varTe_data += rho_term * C_l_grid * h_l_grid

    # Expand the per-degree product into SH and keep only l >= 2
    product      = rho_term * C_l_grid * h_l_grid
    product_clm  = pysh.SHGrid.from_array(product).expand(normalization='4pi')
    w_varTe_coeffs[:, 2:, :] += product_clm.coeffs[:, 2:, :]

# w_sol_grid_turcotteV = pysh.SHGrid.from_array(w_varTe_data)
# w_sol_clm_turcotteV = w_sol_grid_turcotteV.expand()

# Single synthesis at the end — no round-trip on the accumulated field
w_sol_clm_turcotteV  = pysh.SHCoeffs.from_array(w_varTe_coeffs, normalization='4pi')
w_sol_grid_turcotteV = w_sol_clm_turcotteV.expand()


# %%

# Save the current Beuthe solution vector 
w_sol_clm_beuthe.to_file(f'w_beuthe_Rot={Rotated}_lmax={lmax}_TeType={T_e_type}')

# # Load a Beuthe solution vector to overlay in 1D power spectrum plot
w_sol_clm_loaded1 = pysh.SHCoeffs.from_file('w_beuthe_Rot=False_lmax=25_TeType=Input_TeMap_Plesa2018')



# %% PLOTTING
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#########################################################
### PLOT 2D MAPS AND POWER SPECTRA OF DISPLACEMENTS W ###
#########################################################
os.makedirs('Plots/M1VarD_SPEC_opt results/', exist_ok=True)
os.makedirs('Plots/M1VarD_SPEC_opt Rotations results/', exist_ok=True)


print("Plotting input Te map, flexural rigidity D and parameter alpha")
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
T_e_clm.expand(lmax=lmax).plot(ax=ax1, cmap=cmap2, colorbar='right', cb_label= 'Synthetic T_e map, m')
D_clm.expand(lmax=lmax).plot(ax=ax2, cmap=cmap2, colorbar='right', cb_label= 'Synthetic D map')
a_clm.expand(lmax=lmax).plot(ax=ax3, cmap=cmap2, colorbar='right', cb_label= 'Synthetic alpha map') 
plt.tight_layout()
if Save_Figs == True and Rotated_SaveFig == False: 
    plt.savefig(f'Plots/M1VarD_SPEC_opt results/InputTe_{T_e_type}_lmax{lmax}.png', dpi=200)
elif Save_Figs == True and Rotated_SaveFig == True:
    plt.savefig(f'Plots/M1VarD_SPEC_opt Rotations results/InputTe_Rot={Rotated}_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


print("Plotting 1D power spectra ratios of w - variable")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
w_sol_clm_beuthe.plot_spectrum(show=False, ax=ax1, legend=f'Current w_sol Beuthe (Rot={Rotated}, lmax={lmax})')
# w_sol_clm_turcotteV.plot_spectrum(show=False, ax=ax1, legend=f'Current w_sol TurcotteV (Rot={Rotated}, lmax={lmax})')
ax1.legend()
ax1.set_title(f"Power Spectrum of displacement w Beuthe (Te = {T_e_type}) ")
(w_sol_clm_beuthe-w_sol_clm_turcotteV).plot_spectrum(show=False, ax=ax2, yscale='lin')
ax2.set_title("Residual of Power Spectra of displacement w (B-Tv)")
plt.tight_layout()
if Save_Figs == True and Rotated_SaveFig == False: 
    plt.savefig(f'Plots/M1VarD_SPEC_opt results/PowerSpectra1D_Var_{T_e_type}_lmax{lmax}.png', dpi=200)
elif Save_Figs == True and Rotated_SaveFig == True:
    plt.savefig(f'Plots/M1VarD_SPEC_opt Rotations results/PowerSpectra1D_Var_Rot={Rotated}_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


print("Plotting 2D power spectra of w")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
w_sol_clm_beuthe.plot_spectrum2d(show=False, ax=ax1, cmap_rlimits=(1e-7, 1))
ax1.set_title(f"Power Spectrum of displacement w Beuthe (Te = {T_e_type})")
w_sol_clm_turcotteV.plot_spectrum2d(show=False, ax=ax2, cmap_rlimits=(1e-7, 1))
ax2.set_title(f"Power Spectrum of displacement w Turcotte Variable (Te = {T_e_type})")
(w_sol_clm_beuthe - w_sol_clm_turcotteV).plot_spectrum2d(show=False, ax=ax3, cmap_rlimits=(1e-7, 1))
ax3.set_title("Residual Power Spectrum of displacement w (B-Tv)")
plt.tight_layout()
if Save_Figs == True and Rotated_SaveFig == False: 
    plt.savefig(f'Plots/M1VarD_SPEC_opt results/PowerSpectra2D_Var_{T_e_type}_lmax{lmax}.png', dpi=200)
elif Save_Figs == True and Rotated_SaveFig == True:
    plt.savefig(f'Plots/M1VarD_SPEC_opt Rotations results/PowerSpectra2D_Var_Rot={Rotated}_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


# Transform meters to kilometers
w_sol_grid_beuthe.data = w_sol_grid_beuthe.data/1e3
w_sol_grid_turcotteV.data = w_sol_grid_turcotteV.data/1e3

print("Plotting 2D deflection maps")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))
ax2.set_visible(False)
w_sol_grid_beuthe.plot(ax=ax1, cmap=cmap1, colorbar='right', cb_label='w [km]')
ax1.set_title(f'TSA-B  Beuthe model solution (Te = {T_e_type})')
ax1.contour(w_sol_grid_beuthe.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper")

w_sol_grid_turcotteV.plot(ax=ax3, cmap=cmap1, colorbar='right', cb_label='w [km]')
ax3.set_title(f'TSA-T  Turcotte variable (Te = {T_e_type})')
ax3.contour(w_sol_grid_turcotteV.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper")

# Transform kilometers back to meters
w_sol_grid_beuthe.data = w_sol_grid_beuthe.data*1e3
w_sol_grid_turcotteV.data = w_sol_grid_turcotteV.data*1e3

diff_grid_BTv = w_sol_grid_beuthe - w_sol_grid_turcotteV
diff_grid_BTv.plot(ax=ax4, cmap=cmap1, colorbar='right', cb_label='Misfit [m]')
ax4.set_title('Residual TSA-B − TSA-Tv')

plt.tight_layout()
if Save_Figs == True and Rotated_SaveFig == False: 
    plt.savefig(f'Plots/M1VarD_SPEC_opt results/DeflectionMap2D_{T_e_type}_lmax{lmax}.png', dpi=200)
elif Save_Figs == True and Rotated_SaveFig == True:
    plt.savefig(f'Plots/M1VarD_SPEC_opt Rotations results/DeflectionMap2D_Rot={Rotated}_{T_e_type}_lmax{lmax}.png', dpi=200)

plt.show()






end = time.time()
print("\n--- Entire Model Run Complete ---")
print("Total runtime:", round(end - start, 1), "seconds")

