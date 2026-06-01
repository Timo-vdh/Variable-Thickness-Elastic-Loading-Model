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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Load in topography and gravity data
pot_clm = pysh.datasets.Mars.GMM3(lmax=lmax)
topo_clm = pysh.datasets.Mars.MOLA_shape(lmax=lmax)

R = topo_clm.coeffs[0, 0, 0]  # Mean planetary radius
pot_clm = pot_clm.change_ref(r0=R)  # Downward continue to Mean
# planetary radius

# Compute the geoid as approximated in Banerdt's formulation
geoid_clm = pot_clm * R

# Constants
G = pysh.constants.G.value  # Gravitational constant
gm = pot_clm.gm  # GM given in the gravity
# model file
mass = gm / G  # Mass of the planet
g0 = gm / R**2  # Mean gravitational
# attraction of the planet

# Remove 100% of C20
percent_C20 = 0.0
topo_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo_clm.coeffs[0, 2, 0]
geoid_clm.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid_clm.coeffs[0, 2, 0]

# Set color map
mycmap = scm.diverging.Vik_20.mpl_colormap


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### Finding first estimate for displacement w after derivation of 
# constant thickness model using Beuthe eq (88) ###
l = np.arange(lmax + 1, dtype=float)
Lapl = -l*(l+1)

shape = (2, lmax + 1, lmax + 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
############################################
### CREATING A SYNTHETIC VARIABLE Te MAP ###
############################################

# Initialize randomizer
seed = 1
l_corner = 10
beta = 3.0
power = np.zeros(lmax + 1)
for li in range(2, lmax+1):
    if li <= l_corner:
        power[li] = 20.0
    else:
        power[li] = (l_corner / li) ** beta

# Make a random coefficient map
T_e_type = 'Random_TeMap'
T_e_clm = pysh.SHCoeffs.from_random(power, lmax=lmax, seed=seed)
T_e_array = T_e_clm.expand().to_array()*1e3 + 150e3
T_e_grid = pysh.SHGrid.from_array(T_e_array)

# # Making a constant T_e map
# T_e_type = 'Constant_TeMap'
# T_e = 150e3
# const_T_e_grid = np.ones(shape)*T_e

# T_e_coeffs = pysh.SHCoeffs.from_array(const_T_e_grid).convert(normalization = '4pi')
# T_e_grid = T_e_coeffs.expand()
# T_e_array = T_e * np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
# T_e_grid = pysh.SHGrid.from_array(T_e_array)

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#############################################################
### DEFINE TURCOTTE EQUATION FOR VARIABLE THICKNESS SHELL ###
#############################################################

# Simplified form of constant thickness thin shell approximation applied to a 
# shell of variable thickness, following Kalousova et al. (2012) definition

# First set up terms of C_l
def tau(E, T_e_local, Re, rho_m, rho_c, g0):
    return E * T_e_local / (Re**2 * (rho_m - rho_c) * g0)
def sigma(tau_val, nu, T_e_local, Re):
    return tau_val / (12 * (1 - nu**2)) * (T_e_local / Re)**2

# Then definition of C_l itself
def C_l_functional(l_val, nu, E, T_e_local, Re, rho_m, rho_c, g0): 
    tau1 = tau(E, T_e_local, Re, rho_m, rho_c, g0)
    sigma1 = sigma(tau1, nu, T_e_local, Re)
    
    numerator = l_val * (l_val + 1) - (1 - nu)
    denominator_b1 = (l_val**3 * (l_val + 1)**3 
                      - 4 * l_val**2 * (l_val + 1)**2 
                      + 4 * l_val * (l_val + 1))
    return numerator / (sigma1 * denominator_b1 + tau1 * (l_val * (l_val + 1) - 2) + l_val * (l_val + 1) - (1 - nu))

# Pre-calculate the density-ratio term
rho_term = -rho_c / (rho_m - rho_c)

# Quick and simple calculation of constant thickness deflection
wlm_turcotte = np.zeros(shape)
T_e_constant = 150e3
for degree in range(2, lmax + 1):  # Ignore degree 0 from calculations
    C_l_local = C_l_functional(degree, nu, E, T_e_constant, R-T_e_constant/2, rho_m, rho_c, g0)
    wlm_turcotte[: , degree , : degree+1] = (rho_term * C_l_local
                                    * topo_clm.coeffs[: , degree , : degree+1])
wlm_turcotte_coeffs = pysh.SHCoeffs.from_array(wlm_turcotte/1e3)
wlm_turcotte_grid = wlm_turcotte_coeffs.expand()
fig1, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
wlm_turcotte_grid.plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              cb_label= f'Displacement at ref. surface [km] - T_e={int(T_e_constant/1e3)} km'
              )

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
###########################################################
### STRUCTURAL EXECUTION LOOP FOR SPECTRAL CALCULATIONS ###
###########################################################

# Make a colatitude range for the T_e functions to be created over
theta_range = np.linspace(0, 180, 2*(lmax+1)+1)

# The single load degrees analyzed in Figure 6-8 of Kalousova et al. (2012)
target_load_degrees = [2, 5, 8, 15]

# Build a flat sequence list matching the grid ordering approach
mode_map = []

# 2D MODE MAP
for l_idx in range(lmax + 1):
    for m_idx in range(-l_idx, l_idx + 1):
        mode_map.append((l_idx, m_idx))

N_modes = len(mode_map)

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

Dlm_str = np.array([find_custom_element(l_v, m_v, Dlm_unstr) for l_v, m_v in mode_map])
alm_str = np.array([find_custom_element(l_v, m_v, alm_unstr) for l_v, m_v in mode_map])

print("Initializing sparse matrix buffers...")
matrix_A_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)
matrix_B_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)

diag_a = np.zeros(N_modes, dtype=np.float64)
diag_b = np.zeros(N_modes, dtype=np.float64)

for i, (l_val, m_val) in enumerate(mode_map):
    d_l = -l_val * (l_val + 1) + 2
    diag_a[i] = ((Re / T_e_0)**3 / E) * d_l
    diag_b[i] = -1.0 * d_l

matrix_a_l_sparse = sparse.diags(diag_a, format="lil")
matrix_b_l_sparse = sparse.diags(diag_b, format="lil")

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
            if w_coef_A == 0.0:
                continue
            
            for M in range(-L, L + 1):
                q_val = get_numeric_gaunt(l_val, L, l_prime, m_val, M, m_prime)
                if q_val == 0.0:
                    continue
                
                L_idx = L * (L + 1) + M
                D_val = float(find_custom_element(L, M, Dlm_unstr))
                a_val = float(find_custom_element(L, M, alm_unstr))

                print(f'w_coef_A = {w_coef_A}')
                print(f'D_val = {D_val}')
                print(f'q_val = {q_val}')
                cell_sum_A += w_coef_A * D_val * q_val 
                cell_sum_B += w_coef_B * a_val * q_val
        
        cell_sum_A = cell_sum_A * scaler_A
        cell_sum_B = cell_sum_B * scaler_B
        
        if l_val == l_prime and m_val == m_prime:
            cell_sum_A += buoy
            
        if cell_sum_A != 0.0:
            matrix_A_sparse[i, j] = cell_sum_A
        if cell_sum_B != 0.0:
            matrix_B_sparse[i, j] = cell_sum_B


print("Combining sub-matrices into a sparse 2N x 2N architecture...")
M_system_sparse = sparse.bmat([
    [matrix_A_sparse,     matrix_a_l_sparse],
    [matrix_b_l_sparse,   matrix_B_sparse]
], format="lil")

print("Setting degree 0 and 1 to zero...")
for idx, (l_val, m_val) in enumerate(mode_map):
    if l_val == 0 or l_val == 1:
        M_system_sparse[idx, :] = 0.0
        M_system_sparse[idx, idx] = 1.0
        M_system_sparse[idx + N_modes, :] = 0.0
        M_system_sparse[idx + N_modes, idx + N_modes] = 1.0

# Convert to CSR (compressed sparse row) format for faster calculations
M_system_csr = M_system_sparse.tocsr()

# Resolve responses for each individual loading case specified in the paper
print(f"Solving structural displacement vector for lmax={lmax}")
factors_y_lm = (Re / T_e_0)**3 * (rho_c * g0 * Re) / E
   
# True topographic loading case
y_lm_topo = factors_y_lm * (topo_clm.coeffs - geoid_clm.coeffs)
y_lm_unstr = pysh.shio.SHCilmToVector(y_lm_topo)

y_lm_str = np.array([find_custom_element(l_v, m_v, y_lm_unstr) for l_v, m_v in mode_map])

rhs_dense = np.concatenate([y_lm_str, np.zeros(N_modes)])

# Impose boundary vector zeros on low degrees
for idx, (l_val, m_val) in enumerate(mode_map):
    if l_val == 0 or l_val == 1:
        rhs_dense[idx] = 0.0
        rhs_dense[idx + N_modes] = 0.0

# Run linear algebra solver
sol_vector = spla.spsolve(M_system_csr, rhs_dense)
w_sol = sol_vector[:N_modes]

# Map flat 1D solution back into 3D SH footprint matrix array
w_coeffs_np = np.zeros((2, lmax + 1, lmax + 1))
for idx, (l_val, m_val) in enumerate(mode_map):
    if m_val >= 0:
        w_coeffs_np[0, l_val, m_val] = float(w_sol[idx])
    else:
        w_coeffs_np[1, l_val, abs(m_val)] = float(w_sol[idx])
 

# Multiplication by -1 depending on the convention of positive displacement
w_sol_clm_beuthe = pysh.SHCoeffs.from_array(w_coeffs_np, normalization='4pi')
w_sol_grid_beuthe = w_sol_clm_beuthe.expand()

displacement_profile_beuthe = w_sol_grid_beuthe.data[:, 0]  # Isolated zonal line profile

# Plot the 1D and 2D power spectra of the displacement coefficients
w_sol_clm_beuthe.plot_spectrum()
w_sol_clm_beuthe.plot_spectrum2d()



# PLOT SPATIAL 2D COMPONENT FIELDS ---
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 12))

w_sol_grid_beuthe.plot(ax=ax1, cmap=mycmap, colorbar='right', cb_label='w [m]')
ax1.set_title(f'Spectral Model Variable Thickness (TSA-B), lmax={lmax}, Te = {T_e_type}')


plt.tight_layout()
# os.makedirs('Plots/M1VarD_SPEC_opt results/', exist_ok=True)
# plt.savefig(f'Plots/M1VarD_SPEC_opt results/TestDispMars_{T_e_type}_lmax{lmax}_1D={AXISYMMETRIC}.png', dpi=200)
plt.show()

end = time.time()
print("\n--- Entire Test System Run Complete ---")
print("Total runtime:", round(end - start, 1), "seconds")




# Plot the T_e, D and alpha maps
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
T_e_grid.plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic T_e map, m'
              )
D_grid.plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic D map'
              )
a_grid.plot(ax=ax3,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic alpha map'
              ) 



