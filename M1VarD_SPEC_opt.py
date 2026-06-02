# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:37:18 2026

@author: vand_t1
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from palettable import scientific as scm
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time
import os

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
rho_l = rho_c   # Density of the topographic load

dc = 0          # Crustal thickness variations delta c (used in Banerdt)
dp = 0          # Crustal density variations delta rho (used in Banerdt)
M = 0           # Thickness of density anomaly in mantle

# Set maximum spherical harmonic degree to perform all calculations
lmax = 15 

# Set whether output figures are saved or not
Save_Figs = False

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Load in topography and gravity data
pot_clm = pysh.datasets.Mars.GMM3(lmax=lmax)
topo_clm = pysh.datasets.Mars.MOLA_shape(lmax=lmax)

R = topo_clm.coeffs[0, 0, 0]  # Mean planetary radius
pot_clm = pot_clm.change_ref(r0=R)  # Downward continue to Mean planetary radius

# Compute the geoid as approximated in Banerdt's formulation
geoid_clm = pot_clm * R

# Constants
G = pysh.constants.G.value  # Gravitational constant
gm = pot_clm.gm  # GM given in the gravity model file
mass = gm / G  # Mass of the planet
g0 = gm / R**2  # Mean gravitational attraction of the planet

# Remove 100% of C20
percent_C20 = 0.0
topo_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo_clm.coeffs[0, 2, 0]
geoid_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid_clm.coeffs[0, 2, 0]

# Set color map
mycmap = scm.diverging.Vik_20.mpl_colormap

shape = (2, lmax + 1, lmax + 1)

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
# T_e_mean = np.mean(T_e_array)


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
# T_e_mean = np.mean(T_e_array)


# 4. Make harmonic T_e distribution - Model III of Kalousova
T_e_type = 'Harmonic_TeMap_MIII'
T_e_III = 100e3 + 50e3*np.cos(10*np.radians(theta_range))
T_e_array = np.tile(T_e_III.reshape(-1, 1), (1, 4*(lmax+1)+1))
T_e_mean = np.mean(T_e_array)



# Convert to pyshtools classes
T_e_grid = pysh.SHGrid.from_array(T_e_array)
T_e_coeffs = T_e_grid.expand()

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
    
    factor = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * np.pi))  # Is the convention right here????
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


_sqrt4pi = np.sqrt(4.0 * np.pi)   # BUG 1 + BUG 3 normalisation factor


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
T_e_0 = np.mean(T_e_array)
Re = R - T_e_0/2
    
# Precalculate the buoyancy term used in Matrix A, and the two scaling factors 
# of the two matrices
buoy = (Re / T_e_0)**3 * (Re / E) * g0 * (rho_m - rho_c)
scaler_A = 1.0 / (E * T_e_0**3)
scaler_B = Re

# Dynamically compute Flexural Rigidity D and Alpha maps for this specific profile
D_array = E * T_e_array**3 / (12 * (1 - nu**2))
a_array = 1.0 / (E * T_e_array)

D_grid = pysh.SHGrid.from_array(D_array)
a_grid = pysh.SHGrid.from_array(a_array)

D_clm = D_grid.expand(normalization='4pi')
a_clm = a_grid.expand(normalization='4pi')

Dlm_unstr = pysh.shio.SHCilmToVector(D_clm.coeffs)
alm_unstr = pysh.shio.SHCilmToVector(a_clm.coeffs)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("Initializing sparse matrix buffers...")
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

print("Assembling coupling combinations across spectral elements...")

for i, (l_val, m_val) in enumerate(mode_map):
    for j, (l_prime, m_prime) in enumerate(mode_map):
        cell_sum_A = 0.0
        cell_sum_B = 0.0
        
        min_L = abs(l_val - l_prime)
        max_L = min(l_val + l_prime, lmax)
        
        for L in range(min_L, max_L + 1):
            if (l_val + l_prime + L) % 2 != 0:
                continue
            
            w_coef_A = W_numeric_A(l_val, l_prime, L, nu)
            w_coef_B = W_numeric_B(l_val, l_prime, L, nu)
            if w_coef_A == 0.0 or w_coef_B == 0:     
                continue
            
            for M in range(-L, L + 1):
                q_val = get_real_gaunt(l_val, m_val, L, M, l_prime, m_prime)
                if q_val == 0.0:
                    continue
                
                D_val = float(find_custom_element(L, M, Dlm_unstr))
                a_val = float(find_custom_element(L, M, alm_unstr))

                cell_sum_A += w_coef_A * D_val * q_val 
                cell_sum_B += w_coef_B * a_val * q_val
        
                # print(f"\nl,m={l_val, m_val}, l',m'={l_prime, m_prime}, L,M={L,M}")
                # print(f'w_coef_A = {w_coef_A}')
                # print(f'D_val = {D_val}')
                # print(f'q_val = {q_val}')
                
        val_A = cell_sum_A * scaler_A
        val_B = cell_sum_B * scaler_B
        
        if l_val == l_prime and m_val == m_prime:
            val_A += buoy
            
        if val_A != 0.0:
            matrix_A_sparse[i, j] = val_A
        if val_B != 0.0:
            matrix_B_sparse[i, j] = val_B


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


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
###################################################
### SOLVE THE TURCOTTE CONSTANT Te DISPLACEMENT ###
###################################################
print("Computing TSA-T (constant Te) …")
w_coeffs_turcotte = np.zeros((2, lmax+1, lmax+1))

for l_val in range(2, lmax+1):
    C_l = C_l_functional(l_val, nu, E, T_e_0, Re, rho_m, rho_c, g0)
    w_coeffs_turcotte[:, l_val, :l_val+1] = (
        rho_term * C_l
        * (topo_clm.coeffs[:, l_val, :l_val+1]
           - geoid_clm.coeffs[:, l_val, :l_val+1])
    )

w_sol_clm_turcotte  = pysh.SHCoeffs.from_array(w_coeffs_turcotte, normalization='4pi')
w_sol_grid_turcotte = w_sol_clm_turcotte.expand()


# %%


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
Te_grid_data = T_e_grid.data   # shape (nlat, nlon)

# Pre-compute C_l(Te) at every grid point for every degree l.
# Shape: (lmax+1, nlat, nlon)  -- C_l varies spatially for variable-Te case.
# For large grids/lmax this can be memory-intensive; compute on-the-fly instead.

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
    # Vectorised: compute C_l for every Te value in the grid at once.
    tau_grid   = E * Te_grid_data / (Re**2 * (rho_m - rho_c) * g0)
    sigma_grid = tau_grid / (12*(1 - nu**2)) * (Te_grid_data / Re)**2
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
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#########################################################
### PLOT 2D MAPS AND POWER SPECTRA OF DISPLACEMENTS W ###
#########################################################
os.makedirs('Plots/M1VarD_SPEC_opt results/', exist_ok=True)

print("Plotting input Te map, flexural rigidity D and parameter alpha")
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
T_e_grid.plot(ax=ax1, cmap=mycmap, colorbar='right', cb_label= 'Synthetic T_e map, m')
D_grid.plot(ax=ax2, cmap=mycmap, colorbar='right', cb_label= 'Synthetic D map')
a_grid.plot(ax=ax3, cmap=mycmap, colorbar='right', cb_label= 'Synthetic alpha map') 
plt.tight_layout()
if Save_Figs: plt.savefig(f'Plots/M1VarD_SPEC_opt results/InputTe_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


print("Plotting 1D power spectra ratios of w - constant")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
w_sol_clm_beuthe.plot_spectrum(show=False, ax=ax1)
ax1.set_title(f"Power Spectrum of displacement w Beuthe (Te = {T_e_type}) ")
w_sol_clm_turcotte.plot_spectrum(show=False, ax=ax2)
ax2.set_title(f"Power Spectrum of displacement w Turcotte Constant (Te = {T_e_type})")
(w_sol_clm_beuthe/w_sol_clm_turcotte).plot_spectrum(show=False, ax=ax3, yscale='lin')
ax3.set_title("Ratio of Power Spectra of displacement w (B-T)")
plt.tight_layout()
if Save_Figs: plt.savefig(f'Plots/M1VarD_SPEC_opt results/PowerSpectra1D_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


print("Plotting 1D power spectra ratios of w - variable")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
w_sol_clm_beuthe.plot_spectrum(show=False, ax=ax1)
ax1.set_title(f"Power Spectrum of displacement w Beuthe (Te = {T_e_type}) ")
w_sol_clm_turcotteV.plot_spectrum(show=False, ax=ax2)
ax2.set_title(f"Power Spectrum of displacement w Turcotte Variable (Te = {T_e_type})")
(w_sol_clm_beuthe/w_sol_clm_turcotteV).plot_spectrum(show=False, ax=ax3, yscale='lin')
ax3.set_title("Ratio of Power Spectra of displacement w (B-Tv)")
plt.tight_layout()
if Save_Figs: plt.savefig(f'Plots/M1VarD_SPEC_opt results/PowerSpectra1D_Var_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


print("Plotting 2D power spectra of w")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
w_sol_clm_beuthe.plot_spectrum2d(show=False, ax=ax1, cmap_rlimits=(1e-7, 1))
ax1.set_title(f"Power Spectrum of displacement w Beuthe (Te = {T_e_type})")
w_sol_clm_turcotte.plot_spectrum2d(show=False, ax=ax2, cmap_rlimits=(1e-7, 1))
ax2.set_title(f"Power Spectrum of displacement w Turcotte Constant (Te = {T_e_type})")
(w_sol_clm_beuthe - w_sol_clm_turcotte).plot_spectrum2d(show=False, ax=ax3, cmap_rlimits=(1e-7, 1))
ax3.set_title("Residual Power Spectrum of displacement w (B-T)")
plt.tight_layout()
if Save_Figs: plt.savefig(f'Plots/M1VarD_SPEC_opt results/PowerSpectra2D_{T_e_type}_lmax{lmax}.png', dpi=200)
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
if Save_Figs: plt.savefig(f'Plots/M1VarD_SPEC_opt results/PowerSpectra2D_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


# Transform meters to kilometers
w_sol_grid_beuthe.data = w_sol_grid_beuthe.data/1e3
w_sol_grid_turcotte.data = w_sol_grid_turcotte.data/1e3
w_sol_grid_turcotteV.data = w_sol_grid_turcotteV.data/1e3




print("Plotting 2D deflection maps")
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(13, 10))
ax2.set_visible(False)
w_sol_grid_beuthe.plot(ax=ax1, cmap=mycmap, colorbar='right', cb_label='w [km]')
ax1.set_title(f'TSA-B  Beuthe model solution (Te = {T_e_type})')
ax1.contour(w_sol_grid_beuthe.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper")

w_sol_grid_turcotte.plot(ax=ax3, cmap=mycmap, colorbar='right', cb_label='w [km]')
ax3.set_title(f'TSA-T  Turcotte constant (Te = {T_e_mean/1e3:.0f} km)')
ax3.contour(w_sol_grid_turcotte.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper")

w_sol_grid_turcotteV.plot(ax=ax5, cmap=mycmap, colorbar='right', cb_label='w [km]')
ax5.set_title(f'TSA-T  Turcotte variable (Te = {T_e_type})')
ax5.contour(w_sol_grid_turcotteV.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper")

# Transform kilometers back to meters
w_sol_grid_beuthe.data = w_sol_grid_beuthe.data*1e3
w_sol_grid_turcotte.data = w_sol_grid_turcotte.data*1e3
w_sol_grid_turcotteV.data = w_sol_grid_turcotteV.data*1e3

diff_grid_BT = w_sol_grid_beuthe - w_sol_grid_turcotte
diff_grid_BT.plot(ax=ax4, cmap=mycmap, colorbar='right', cb_label='Misfit [m]')
ax4.set_title('Residual TSA-B − TSA-T')

diff_grid_BTv = w_sol_grid_beuthe - w_sol_grid_turcotteV
diff_grid_BTv.plot(ax=ax6, cmap=mycmap, colorbar='right', cb_label='Misfit [m]')
ax6.set_title('Residual TSA-B − TSA-Tv')

plt.tight_layout()
if Save_Figs: plt.savefig(f'Plots/M1VarD_SPEC_opt results/DeflectionMap2D_{T_e_type}_lmax{lmax}.png', dpi=200)
plt.show()


end = time.time()
print("\n--- Entire Model Run Complete ---")
print("Total runtime:", round(end - start, 1), "seconds")

matrix_A_dense = (matrix_A_sparse.todense())