# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:37:18 2026

@author: Timov
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from palettable import scientific as scm
import time

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
nu = 0.25
E = 100.0e9
rho_c = 2900.
rho_m = 3500.
rho_l = rho_c

lmax = 100  # Maximum spherical harmonic degree to perform all calculations


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Loading in topography and gravity data

pot_clm = pysh.datasets.Mars.GMM3(lmax=120)
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


# inputs
T_e = 150e3
Re = float(R-T_e/2)


# if 10e3 < T_e < 500e3: # Prevent issues when testing limit cases
#     Re = R-T_e/2
# else:
#     Re = R-150e3/2
    
D = E*T_e**3/(12*(1-nu**2))
dc = 0      # Crustal thickness variations delta c (used in Banerdt)
dp = 0      # Crustal density variations delta rho (used in Banerdt)
M = 0       # Thickness of density anomaly in mantle

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
        power[li] = 1.0
    else:
        power[li] = (l_corner / li) ** beta

# Make a random coefficient map
T_e_clm = pysh.SHCoeffs.from_random(power, lmax=lmax, seed=seed)
T_e_array = T_e_clm.expand().to_array()*1e3 + 150e3
T_e_grid = pysh.SHGrid.from_array(T_e_array)



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Calculate the variable Flexural rigidity D and parameter alpha (a) 
## and convert to spherical harmonics

# D = E*T_e^3/(12*(1-nu^2))
# alpha = 1/(E*T_e_lm)

Dlm = np.zeros(shape)
alm = np.zeros(shape)

# T_e_coeffs = input T_e map from GAIA, in SH coefficients
# T_e_array = T_e_coeffs.expand().to_array()

# First calculate in array form, then convert result to SHGrid and then to SHCoeffs
D_array = E*T_e_array**3 / (12*(1-nu**2))
a_array = 1/(E*T_e_array)

D_grid = pysh.SHGrid.from_array(D_array)        # Grid format
a_grid = pysh.SHGrid.from_array(a_array)        # Grid format

D_clm = D_grid.expand()                         # Coeffs format
a_clm = a_grid.expand()                         # Coeffs format

# # Plot the resulting synthetic T_e, D and alpha maps
# fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
# T_e_grid.plot(ax=ax1,
#               cmap=mycmap,
#               colorbar='right',
#               # cb_tick_interval=2,
#               # cmap_limits=[-4, 4],
#               cb_label= 'Synthetic T_e map, m'
#               )
# D_grid.plot(ax=ax2,
#               cmap=mycmap,
#               colorbar='right',
#               # cb_tick_interval=2,
#               # cmap_limits=[-4, 4],
#               cb_label= 'Synthetic D map'
#               )
# a_grid.plot(ax=ax3,
#               cmap=mycmap,
#               colorbar='right',
#               # cb_tick_interval=2,
#               # cmap_limits=[-4, 4],
#               cb_label= 'Synthetic alpha map'
#               )



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################
### BUILDING THE MATRICES OF KALOUSOVA APPENDIX A ###
################## EQS. A18 - A22 ###################
#####################################################
# Create a flat list of all valid (l, m) configurations from 0 to lmax
# Note: m ranges from -l to +l to capture both cosmic phases/orientations
mode_map = []
for l in range(lmax + 1):
    for m in range(-l, l + 1):
        mode_map.append((l, m))

N_modes = len(mode_map)  # SIZE OF THE MATRICES IS (lmax+1)**2 BY (lmax+1)**2

# Calculate topographic loading coefficients y_lm
factors_y_lm = (Re/T_e)**3 * (rho_c * g0 * Re) / E
y_lm = np.zeros(shape)
for degree in range(0, lmax + 1):  # Ignore degree 0 from calculations
    y_lm[: , degree , : degree+1] = (factors_y_lm * topo_clm.coeffs[: , degree , : degree+1])

# Set SH coefficients of D, alpha and y in the correct format (D00, D1-1, D10, D11, D2-2, ...)
Dlm_unstr = pysh.shio.SHCilmToVector(D_clm.coeffs)       # This has format (D00, D10, D11, D1-1, D20, D21, ..)
Dlm_str = []
alm_unstr = pysh.shio.SHCilmToVector(a_clm.coeffs)
alm_str = []
y_lm_unstr = pysh.shio.SHCilmToVector(y_lm)
y_lm_str = []

# Map out how the pyshstools array is structured
def find_custom_element(l, m, xlm_unstr):
    # Find the starting index of degree l in the shtools array (which is l^2)
    block_start = l**2
    if m == 0:
        offset = 0
    elif m > 0:
        offset = m
    else:
        offset = l + abs(m)
    return xlm_unstr[block_start + offset]

# Re-populate into the mathematical sequence (D00, D1-1, D10, D11, D2-2, ...)
for l in range(lmax+1):
    for m in range(-l, l + 1): # Strictly from -l to +l
        element_D = find_custom_element(l, m, Dlm_unstr)
        element_a = find_custom_element(l, m, alm_unstr)
        element_y = find_custom_element(l, m, y_lm_unstr)
        
        Dlm_str.append(element_D)
        alm_str.append(element_a)
        y_lm_str.append(element_y)
         
Dlm_str = np.array(Dlm_str)
alm_str = np.array(alm_str)
y_lm_str = np.array(y_lm_str)


import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# =============================================================================
# OPTIMIZED HIGH-SPEED REPLACEMENT SECTION
# =============================================================================
print(f"--- Starting Pure Numeric Matrix Generation (lmax = {lmax}) ---")

# 1. Map out structural constants directly as float64 representations
nu_num = 0.25
E_num = 100.0e9
T_e_num = 150e3
Re_num = float(R - T_e_num / 2)
buoy_num = (Re_num / T_e_num)**3 * (Re_num / E_num) * g0 * (rho_m - rho_c)

scaler_A = 1.0 / (E_num * T_e_num**3)
scaler_B = Re_num

# 2. Fast numeric W-coefficient evaluation matching your system equation
def W_numeric(l, l_prime, L, nu=0.25):
    d_l = -l * (l + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L = -L * (L + 1) + 2
    
    term1 = d_l * d_lp
    bracket = (d_l**2 + d_lp**2 + d_L**2 + 
               2*(d_l + d_lp + d_L) - 
               2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return term1 + 0.25 * (1.0 - nu) * bracket

# 3. Fast numerical evaluation of Gaunt Coefficients using spherical_functions
def get_numeric_gaunt(l1, l2, l3, m1, m2, m3):
    """
    Computes the Gaunt integral of three real spherical harmonics.
    Uses pyshtools.utils.Wigner3j vector evaluation to find the targets.
    
    Mapping documentation parameters to physical loops:
    l1 = target j loop
    l2 = L loop (fixed j2)
    l3 = l_prime loop (fixed j3)
    """
    if (l1 + l2 + l3) % 2 != 0:
        return 0.0
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return 0.0
    if m1 + m2 + m3 != 0:
        return 0.0

    # 1. Evaluate the vector of symbols for m components
    w3j_m_array, jmin_m, jmax_m = pysh.utils.Wigner3j(l2, l3, m1, m2, m3)
    # Check if our current target degree l1 exists within the calculated array bounds
    if not (jmin_m <= l1 <= jmax_m):
        return 0.0
    # Pull the exact value out using the index offset
    w3j_m = w3j_m_array[l1 - jmin_m]

    # 2. Evaluate the vector of symbols for the zero-magnitudes (m=0)
    w3j_0_array, jmin_0, jmax_0 = pysh.utils.Wigner3j(l2, l3, 0, 0, 0)
    if not (jmin_0 <= l1 <= jmax_0):
        return 0.0
    w3j_0 = w3j_0_array[l1 - jmin_0]
    
    # Geometric scale factor for real spherical harmonic integration
    factor = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * np.pi))
    return factor * w3j_m * w3j_0

# 4. Initialize Sparse dynamic List-of-Lists containers
print("Initializing sparse matrix buffers...")
matrix_A_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)
matrix_B_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)

diag_a = np.zeros(N_modes, dtype=np.float64)
diag_b = np.zeros(N_modes, dtype=np.float64)

for i, (l, m) in enumerate(mode_map):
    d_l = -l * (l + 1) + 2
    diag_a[i] = ((Re_num / T_e_num)**3 / E_num) * d_l
    diag_b[i] = -1.0 * d_l

matrix_a_l_sparse = sparse.diags(diag_a, format="lil")
matrix_b_l_sparse = sparse.diags(diag_b, format="lil")

# 5. Populate System Matrix Fields
print("Assembling coupling combinations across spectral elements...")
for i, (l, m) in enumerate(mode_map):
    for j, (l_prime, m_prime) in enumerate(mode_map):
        
        cell_sum_A = 0.0
        cell_sum_B = 0.0
        
        min_L = abs(l - l_prime)
        max_L = min(l + l_prime, lmax)
        
        for L in range(min_L, max_L + 1):
            if (l + l_prime + L) % 2 != 0:
                continue
            
            w_coef = W_numeric(l, l_prime, L, nu_num)
            if w_coef == 0.0:
                continue
            
            for M in range(-L, L + 1):
                # Calculate quick numerical Gaunt alignment
                q_val = get_numeric_gaunt(l, L, l_prime, m, M, m_prime)
                if q_val == 0.0:
                    continue
                
                # Fetch your variable thickness / topography input coefficients directly by index
                # Ensure your input string lists or maps match the (L, M) sequence indexing
                L_idx = L * (L + 1) + M
                D_val = float(Dlm_str[L_idx])
                a_val = float(alm_str[L_idx])
                
                cell_sum_A += w_coef * D_val * q_val
                cell_sum_B += w_coef * a_val * q_val
        
        # Apply scaling terms
        val_A = cell_sum_A * scaler_A
        val_B = cell_sum_B * scaler_B
        
        # Add core buoyancy constraint along main diagonal
        if l == l_prime and m == m_prime:
            val_A += buoy_num
            
        if val_A != 0.0:
            matrix_A_sparse[i, j] = val_A
        if val_B != 0.0:
            matrix_B_sparse[i, j] = val_B

# 6. Global 2N x 2N Structural Stacking
print("Combining sub-blocks into a sparse 2N x 2N architecture...")
M_system_sparse = sparse.bmat([
    [matrix_A_sparse,     matrix_a_l_sparse],
    [matrix_b_l_sparse,   matrix_B_sparse]
], format="lil")

# Turn the symbolic target vector into a clean numeric NumPy array
rhs_dense = np.concatenate([np.array(y_lm_str, dtype=np.float64), np.zeros(N_modes)])

# 7. Apply Regularization (Pin l=0 and l=1 modes directly inside sparse array)
print("Applying boundary constraints to low-degree modes...")
for idx, (l, m) in enumerate(mode_map):
    if l == 0 or l == 1:
        # Zero-out displacement mode equations (w)
        M_system_sparse[idx, :] = 0.0
        M_system_sparse[idx, idx] = 1.0
        rhs_dense[idx] = 0.0
        
        # Zero-out stress mode equations (F)
        f_idx = idx + N_modes
        M_system_sparse[f_idx, :] = 0.0
        M_system_sparse[f_idx, f_idx] = 1.0
        rhs_dense[f_idx] = 0.0

# 8. Compress and Solve via Linear Solvers
print("Converting to CSR format and solving...")
M_system_csr = M_system_sparse.tocsr()

# Solves the entire system in seconds using optimized SuperLU operations
sol_vector = spla.spsolve(M_system_csr, rhs_dense)

# Unpack solutions back to pure numeric arrays
w_sol = sol_vector[:N_modes]
F_sol = sol_vector[N_modes:]

print("--- System Solved Successfully via Sparse Math Engines ---")

# =============================================================================
# 6. PRINT RESULTS
# =============================================================================
print(f"Deflection w_0_0 amplitude value: {w_sol[0]}")
print(f"Stress function F_0_0 amplitude value: {F_sol[0]}")





# CONVERT SOLUTIONS INTO PYSHTOOLS COEFFICIENTS AND THEN INTO GRIDS
# 1. Initialize empty numpy coefficient arrays matching the shape (2, lmax+1, lmax+1)
w_coeffs_np = np.zeros((2, lmax + 1, lmax + 1))
F_coeffs_np = np.zeros((2, lmax + 1, lmax + 1))

# 2. Map the flat solutions back using the pyshtools indexing rule
for idx, (l, m) in enumerate(mode_map):
    # Extract the numeric float value from the SymPy matrix elements
    w_val = float(w_sol[idx])
    F_val = float(F_sol[idx])
    
    if m >= 0:
        # Positive m goes to the Cosine block (index 0)
        w_coeffs_np[0, l, m] = w_val
        F_coeffs_np[0, l, m] = F_val
    else:
        # Negative m goes to the Sine block (index 1) using its absolute value
        m_abs = abs(m)
        w_coeffs_np[1, l, m_abs] = w_val
        F_coeffs_np[1, l, m_abs] = F_val

# 3. Wrap the raw numpy arrays back into formal pyshtools SHCoeffs classes
# Normalization must match the 'ortho' (fully normalized) standard used by the Gaunt integrals
w_sol_clm = pysh.SHCoeffs.from_array(w_coeffs_np, normalization='ortho')
F_sol_clm = pysh.SHCoeffs.from_array(F_coeffs_np, normalization='ortho')

print("--- Conversion to SHCoeffs format complete ---")


# 4. Expand the spectral coefficients out to spatial grids
w_sol_grid = w_sol_clm.expand()
F_sol_grid = F_sol_clm.expand()


# 5. Generate the Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

w_sol_grid.plot(
    ax=ax1,
    cmap=mycmap,
    colorbar='right',
    cb_label='Transverse Displacement w [m]'
)
ax1.set_title(f'Deflection Solution Field (lmax={lmax})')

F_sol_grid.plot(
    ax=ax2,
    cmap=mycmap,
    colorbar='right',
    cb_label='Stress Function F'
)
ax2.set_title(f'Elastic Stress Function Field (lmax={lmax})')

plt.tight_layout()
plt.show()

end = time.time()
print("total runtime:", end - start)