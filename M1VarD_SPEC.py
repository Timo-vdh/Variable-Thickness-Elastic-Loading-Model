# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:37:18 2026

@author: Timov
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from palettable import scientific as scm

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

lmax = 10  # Maximum spherical harmonic degree to perform all calculations


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
import sympy as sp
from sympy.physics.wigner import gaunt

# Set all inputs as symbols (at least for now)
# nu, E, T_e, Re, g0, rho_m, rho_c = sp.symbols('nu E T_e Re g0 rho_m rho_c')

# Pre-calculate the buoyancy term, as it is independent of degree and order directly
buoy = (Re/T_e)**3 * Re/E * g0 * (rho_m-rho_c)

# Set the constant scaling factors of the two matrices
scaler_A = 1 / (E * T_e**3)
scaler_B = Re

# Print out the expressions for different l, l_prime, L
# W_sym is the symbolic expression of the large term in square brackets for A_l'm'^lm
# This is the same term for the B matrix
def W_sym(l, l_prime, L, nu=sp.symbols('nu')):
    d_l = -l*(l+1)+2
    d_lp = -l_prime*(l_prime+1)+2       # Coefficients of the unknown parameter
    d_L = -L*(L+1)+2                    # Coefficients of the input parameters
    
    term1 = d_l * d_lp
    bracket = (d_l**2 + d_lp**2 + d_L**2 + 
               2*(d_l + d_lp + d_L) - 
               2*(d_l*d_lp + d_l*d_L + d_lp*d_L)
               -8)
    expr = term1 + sp.Rational(1,4) * (1 - nu) * bracket
    return sp.simplify(expr)

# # Evaluate the value of W for all combinations of l, l' and L
# combinations = []
# lmax = 3
# for l in range(lmax+1):
#     for lp in range(lmax+1):
#         for L in range(2*lmax+1): # L can go from |l-lp| to l+lp
#             if abs(l - lp) <= L <= (l + lp) and (l + lp + L) % 2 == 0:
#                 combinations.append(((l, lp, L), W_sym(l, lp, L, 0.25)))

# for c, expr in combinations:
#     print(f"l={c[0]}, l'={c[1]}, L={c[2]} -> W = {expr}")




#############################
### SETUP & INDEX MAPPING ###
#############################

# Create a flat list of all valid (l, m) configurations from 0 to lmax
# Note: m ranges from -l to +l to capture both cosmic phases/orientations
mode_map = []
for l in range(lmax + 1):
    for m in range(-l, l + 1):
        mode_map.append((l, m))

N_modes = len(mode_map)  # SIZE OF THE MATRICES IS (lmax+1)**2 BY (lmax+1)**2
matrix_A = sp.Matrix.zeros(N_modes, N_modes)
matrix_B = sp.Matrix.zeros(N_modes, N_modes)


# Mock Rigidity Coefficients D_LM as symbolic variables for tracking
# In the final code, replace these with actual numerical values.
D = {}
a = {}
for L in range(lmax + 1):
    for M in range(-L, L + 1):
        D[(L, M)] = sp.Symbol(f'D_{L}_{M}')
        a[(L, M)] = sp.Symbol(f'alpha_{L}_{M}')


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


##########################
### PARAMETERS a AND b ###
##########################

# 1. Build the diagonal operator matrices a_l and b_l
# These act as purely diagonal scalars multiplying the vectors element-by-element
diag_a = np.zeros(N_modes)
diag_b = np.zeros(N_modes)

for i, (l, m) in enumerate(mode_map):
    d_l = -l * (l + 1) + 2
    
    # Operator a^l (multiplies F_lm)
    diag_a[i] = ((Re / T_e)**3 / E) * d_l
    
    # Operator b^l (multiplies w_lm, which is just -1 * d_l)
    diag_b[i] = -1.0 * d_l

# Convert vectors to diagonal matrices
matrix_a_l = np.diag(diag_a)
matrix_b_l = np.diag(diag_b)


##########################################
### MASTER SUMMATION LOOP MATRIX A & B ###
##########################################

# Loop over every row (i) and column (j) of the matrix
for i, (l, m) in enumerate(mode_map):
    for j, (l_prime, m_prime) in enumerate(mode_map):
        
        cell_sum_A = 0
        cell_sum_B = 0
        
        # Triangle Rule Selection Filter for L
        min_L = abs(l - l_prime)
        max_L = min(l + l_prime, lmax)  # Boundary cap at data resolution limit
        
        for L in range(min_L, max_L + 1):
            # Parity Filter Check
            if (l + l_prime + L) % 2 != 0:
                continue
            
            # Calculate square bracket term for all values of L (independent of M)
            w_coef = W_sym(l, l_prime, L, 0.25)
            
            # Optimization: If W evaluates to 0 (like any l=1 mode), skip the inner M loop
            if w_coef == 0:
                continue
            
            # Loop over all valid azimuthal orientations M for this thickness scale
            for M in range(-L, L + 1):
                
                # --- SYMPY GAUNT COEFFICIENT CALL ---
                # gaunt(l1, l2, l3, m1, m2, m3) computes the integral of Y_l1_m1 * Y_l2_m2 * Y_l3_m3
                # For our equation: target row is (l, m), data is (L, M), unknown col is (l_prime, m_prime)
                # We use the complex conjugate notation logic matching standard physics expansions:
                q_val = gaunt(l, L, l_prime, m, M, m_prime)
                
                # Optimization: If q_val evaluates to 0, skip the multiplication
                if q_val == 0:
                    continue
                
                # Select coefficient of D and alpha for the multiplication of this combination of L and M
                # according to the indexing set above this section
                D_val = Dlm_str[L*(L+1) + M]
                a_val = alm_str[L*(L+1) + M]
                
                ### Multiply the structural terms together and add to the cell accumulator
                # IF EVALUATING NUMERICALLY, USE BELOW EQUATION FOR NUMERIC COEFFICIENTS OF D AND ALPHA
                cell_sum_A += w_coef * D_val * q_val
                cell_sum_B += w_coef * a_val * q_val
                
                # IF EVALUATING SYMBOLICALLY, USE BELOW EQUATION FOR SYMBOLIC COEFFICIENTS OF D AND ALPHA
                # cell_sum_A += w_coef * D[(L,M)] * q_val
                # cell_sum_B += w_coef * a[(L,M)] * q_val
        
        cell_sum_A = cell_sum_A * scaler_A
        cell_sum_B = cell_sum_B * scaler_B

        
        # Add the buoyancy term to the A matrix value when l=l' (Kronecker operator)
        if l == l_prime and m == m_prime:
            cell_sum_A += buoy
        
        
        matrix_A[i, j] = sp.simplify(cell_sum_A)
        matrix_B[i, j] = sp.simplify(cell_sum_B)




###############################
### PRINTING SAMPLE RESULTS ###
###############################

print("--- Matrix Generations Complete ---")
print(f"A-Matrix Shape: {matrix_A.shape}")
print(f"B-Matrix Shape: {matrix_B.shape}")
print(f"y_lm Shape: {y_lm_str.shape}\n")

# Inspecting the very first cell: Row (0,0) Col (0,0) -> index 0, 0
print(f"Row (0,0), Col (0,0) Entry Matrix A:\n{(matrix_A[0, 0])}")
print(f"Row (0,0), Col (0,0) Entry Matrix B:\n{(matrix_B[0, 0])}\n")

# Inspecting Row (0,0) Col (2,0) -> index 0, 6
# (Matches Mode 1 to Mode 7 layout)
print(f"Row (0,0), Col (2,0) Entry Matrix A:\n{matrix_A[0, 6]}")
print(f"Row (0,0), Col (2,0) Entry Matrix B:\n{matrix_B[0, 6]}\n")




#################################################
### BUILDING SYSTEM OF EQUATIONS OF KALOUSOVA ###
################# EQS. A16 & A17#################
#################################################

# =============================================================================
# 5. FIXED SYSTEM ASSEMBLY & SOLUTION
# =============================================================================

# 1. Create vectors of SymPy symbols for the unknowns matching your mode_map
w_symbols = []
F_symbols = []
for (l, m) in mode_map:
    w_symbols.append(sp.Symbol(f'w_{l}_{m}'))
    F_symbols.append(sp.Symbol(f'F_{l}_{m}'))

vec_w = sp.Matrix(w_symbols)
vec_F = sp.Matrix(F_symbols)
vec_y = sp.Matrix(y_lm_str) 

# 2. Construct the diagonal operator matrices a_l and b_l symbolically
matrix_a_l = sp.Matrix.zeros(N_modes, N_modes)
matrix_b_l = sp.Matrix.zeros(N_modes, N_modes)

for i, (l, m) in enumerate(mode_map):
    d_l = -l * (l + 1) + 2
    matrix_a_l[i, i] = (Re / T_e)**3 / E * d_l
    matrix_b_l[i, i] = -1 * d_l

# 3. Assemble the Full Block System Matrix and Right-Hand Side Vector
M_top = matrix_A.row_join(matrix_a_l)     
M_bottom = matrix_b_l.row_join(matrix_B)  
M_system = M_top.col_join(M_bottom)       

X_unknowns = vec_w.col_join(vec_F)
RHS_vector = vec_y.col_join(sp.Matrix.zeros(N_modes, 1))

print("--- Preparing Mapping Dict for Numeric Fast-Solve ---")

# 4. Map ALL symbolic placeholders to their concrete numerical counter-parts
subs_dict = {}
for idx, (l, m) in enumerate(mode_map):
    subs_dict[D[(l, m)]] = Dlm_str[idx]
    subs_dict[a[(l, m)]] = alm_str[idx]

subs_dict[nu] = 0.25
subs_dict[E] = 100.0e9
subs_dict[T_e] = 150e3
subs_dict[Re] = float(R - 150e3/2)
subs_dict[g0] = g0
subs_dict[rho_m] = 3500.
subs_dict[rho_c] = 2900.

# Collapse system to pure float decimals
print("Collapsing system to pure numeric representations...")
M_numeric = M_system.subs(subs_dict).evalf()
RHS_numeric = RHS_vector.subs(subs_dict).evalf()


# =============================================================================
# REGULARIZATION: PINNING L=0 AND L=1 SINGULARITIES
# =============================================================================
print("Regularizing degree 0 and 1 rigid body translations...")

for idx, (l, m) in enumerate(mode_map):
    if l == 0 or l == 1:
        # --- Pin Deflection (w_lm) ---
        # Clear out the top row equation block for this mode
        for col in range(M_numeric.cols):
            M_numeric[idx, col] = 0.0
        M_numeric[idx, idx] = 1.0  # Set diagonal to 1
        RHS_numeric[idx, 0] = 0.0  # Force w_lm = 0
        
        # --- Pin Stress Function (F_lm) ---
        # The F block is shifted by +N_modes rows down in the matrix
        f_row_idx = idx + N_modes
        for col in range(M_numeric.cols):
            M_numeric[f_row_idx, col] = 0.0
        M_numeric[f_row_idx, f_row_idx] = 1.0  # Set diagonal to 1
        RHS_numeric[f_row_idx, 0] = 0.0       # Force F_lm = 0


print("--- Solving System via LU Decomposition ---")
# The matrix is now regularized and non-singular, so it solves immediately
sol_matrix = M_numeric.LUsolve(RHS_numeric)

# Split the answer vector back into w and F fields
w_sol = sol_matrix[:N_modes, 0]
F_sol = sol_matrix[N_modes:, 0]

# =============================================================================
# 6. PRINT RESULTS
# =============================================================================
print("\n--- System Solved Successfully ---")
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
ax1.set_title('Deflection Solution Field (lmax=3)')

F_sol_grid.plot(
    ax=ax2,
    cmap=mycmap,
    colorbar='right',
    cb_label='Stress Function F [N]'
)
ax2.set_title('Elastic Stress Function Field (lmax=3)')

plt.tight_layout()
plt.show()

