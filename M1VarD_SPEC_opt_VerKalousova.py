# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:02:56 2026

@author: vand_t1
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import time
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import os

start = time.time()

#######################################
"""
Model for the variable thickness deformations of Beuthe (2008).
Current model works with Beuthe's equations 75 and 76, 
neglecting tangential loading.

This requires implementation of the differential operator A(a;b).
Beuthe does not give a spectral method for this, but Kalousova et al. (2012)
do. The approach worked on in this code relies on: 
    - Applying Kalousova et al. (2012)'s approach in spectral domain directly.

This approach as written in code is verified here by creating the same 
synthetic harmonic loading cases as done by Kalousova et al. (2012). If the 
same resulting graphs as in their paper are obtained, the spectral approach 
can afterwards be applied to the real planet Mars.
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
# (Delta w = nabla^2)
# Delta_p w = (Delta + 2) w


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## INPUTS OF KALOUSOVA ET AL. (2012)
nu = 0.25
E = 65.0e9
rho_c = 2900.
rho_m = 3400.
rho_l = rho_c
g0 = 3.8
R = 3395e3

# Set maximum spherical harmonic degree to perform all calculations
lmax = 100  
# Set whether analysing in 1D (Axisymmetric = True) or 2D geometry
# 1D is MUCH faster for high lmax analyses
AXISYMMETRIC = True

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Coefficients shape following pyshtools definition
shape = (2, lmax + 1, lmax + 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
######################################################
### CREATING Te DISTRIBUTION SAME WAY AS KALOUSOVA ###
######################################################

# Make a colatitude range for the harmonic T_e functions to be created over
theta_range = np.linspace(0, 180, 2*(lmax+1)+1)

# Make T_e distribution - Model I of Kalousova
T_e_I = []
start_trans = 80
stop_trans = 100
phi = np.pi * (theta_range - start_trans)/(100 - start_trans)
transition_T_e_I = 125e3 + 75e3*np.cos(phi)

for i, theta in enumerate(theta_range):
    if theta <= start_trans:
        T_e_I.append(200e3)
    elif theta >= 100:
        T_e_I.append(50e3)
    else:
        T_e_I.append(transition_T_e_I[i])
T_e_I = np.array(T_e_I)

# Make harmonic T_e distribution - Model II of Kalousova
T_e_II = 100e3 + 50e3*np.cos(2*np.radians(theta_range))

# Make harmonic T_e distribution - Model III of Kalousova
T_e_III = 100e3 + 50e3*np.cos(10*np.radians(theta_range))

plt.figure()
plt.plot(theta_range, 1e-3*T_e_I)
plt.plot(theta_range, 1e-3*T_e_II)
plt.plot(theta_range, 1e-3*T_e_III)
plt.ylim(0, 250)
plt.xlim(0, 180)
plt.grid()
plt.xlabel('theta (degrees)',)
plt.ylabel('T_e (km) ')
plt.legend(['model I', 'model II', 'model III'])
plt.show()

T_e_I_array = np.tile(T_e_I.reshape(-1, 1), (1, 4*(lmax+1)+1))
T_e_II_array = np.tile(T_e_II.reshape(-1, 1), (1, 4*(lmax+1)+1))
T_e_III_array = np.tile(T_e_III.reshape(-1, 1), (1, 4*(lmax+1)+1))

# Make grid and SHC from the arrays
T_e_I_grid = pysh.SHGrid.from_array(T_e_I_array)
T_e_I_clm = T_e_I_grid.expand()

T_e_II_grid = pysh.SHGrid.from_array(T_e_II_array)
T_e_II_clm = T_e_II_grid.expand()

T_e_III_grid = pysh.SHGrid.from_array(T_e_III_array)
T_e_III_clm = T_e_III_grid.expand()


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
    The W-term of Matrix B is the large term in square brackets of Kalousova 
    et al. (2012) equation A18. The only difference with W_numeric_A is in the 
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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################
### STRUCTURAL EXECUTION LOOP FOR VERIFICATION PLOTS ###
########################################################

# Mapping test models into an automated sequential dictionary
models_to_test = {
    "Model I": T_e_I_array,
    "Model II": T_e_II_array,
    "Model III": T_e_III_array
}

# The single load degrees analyzed in Figure 6-8 of Kalousova et al. (2012)
target_load_degrees = [2, 5, 8, 15]
A_amplitude = 1000.0  # Synthetic load amplitude (1000m)

# Build a flat sequence list matching the grid ordering approach
mode_map = []

# 1D MODE MAP (AXISYMMETRIC MODEL)
if AXISYMMETRIC == True:
    for l_idx in range(lmax + 1):
        mode_map.append((l_idx, 0))     # This skips all m-modes (so 1D axisymmetric)
# 2D MODE MAP
else: 
    for l_idx in range(lmax + 1):
        for m_idx in range(-l_idx, l_idx + 1):
            mode_map.append((l_idx, m_idx))

N_modes = len(mode_map)

# Run the spectral calculations sequentially for all models
for model_name, active_Te_array in models_to_test.items():
    print(f"\n--- Starting Pure Numeric Matrix Generation ({model_name}, lmax = {lmax}) ---")
    
    # Since average shell thickness is different for Model I compared to Model II and III,
    # Set the T_e average value and reference radius Re for each model
    T_e_0 = np.mean(active_Te_array)
    Re = R - T_e_0
    
    buoy = (Re / T_e_0)**3 * (Re / E) * g0 * (rho_m - rho_c)
    scaler_A = 1.0 / (E * T_e_0**3)
    scaler_B = Re

    # Dynamically compute Flexural Rigidity D and Alpha maps for this specific profile
    D_array = E * active_Te_array**3 / (12 * (1 - nu**2))
    a_array = 1.0 / (E * active_Te_array)

    D_grid = pysh.SHGrid.from_array(D_array)
    a_grid = pysh.SHGrid.from_array(a_array)

    D_clm = D_grid.expand(normalization='ortho')
    a_clm = a_grid.expand(normalization='ortho')

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
    if AXISYMMETRIC == False:                                   # Assess model in 2D
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
                        D_val = float(Dlm_str[L_idx])
                        a_val = float(alm_str[L_idx])
                        
                        cell_sum_A += w_coef_A * D_val * q_val
                        cell_sum_B += w_coef_B * a_val * q_val
                
                val_A = cell_sum_A * scaler_A
                val_B = cell_sum_B * scaler_B
                
                if l_val == l_prime and m_val == m_prime:
                    val_A += buoy
                    
                if val_A != 0.0:
                    matrix_A_sparse[i, j] = val_A
                if val_B != 0.0:
                    matrix_B_sparse[i, j] = val_B
    else:                                                     # Assess model in 1D
        for i, (l_val, _) in enumerate(mode_map):
            for j, (l_prime, _) in enumerate(mode_map):
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
                    
                    # AXISYMMETRIC SHORTCUT: M is strictly 0 because m_val = m_prime = 0
                    q_val = get_numeric_gaunt(l_val, L, l_prime, 0, 0, 0)
                    if q_val == 0.0:
                        continue
                    
                    # Safely extract the L-degree zonal component directly 
                    D_val = float(find_custom_element(L, 0, Dlm_unstr))
                    a_val = float(find_custom_element(L, 0, alm_unstr))
                    
                    cell_sum_A += w_coef_A * D_val * q_val
                    cell_sum_B += w_coef_B * a_val * q_val
                
                val_A = cell_sum_A * scaler_A
                val_B = cell_sum_B * scaler_B
                
                if l_val == l_prime:
                    val_A += buoy
                    
                if val_A != 0.0:
                    matrix_A_sparse[i, j] = val_A
                if val_B != 0.0:
                    matrix_B_sparse[i, j] = val_B        

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

    # Create subplot framework mimicking the layout structure of Kalousova Figure 6-8
    fig_verify, axes = plt.subplots(4, 1, figsize=(6, 16))
    fig_verify.suptitle(f"Radial Displacement Response Lines for {model_name}, lmax={lmax}", fontsize=12, fontweight='bold')

    # Resolve responses for each individual loading case specified in the paper
    for ax_idx, l_load in enumerate(target_load_degrees):
        print(f"Solving structural displacement vector for Synthetic Load at degree l = {l_load}...")
        
        # Build pure synthetic harmonic loading array matching Kalousova Eq. 21
        h_synthetic = np.zeros(shape)
        h_synthetic[0, l_load, 0] = A_amplitude  # Zonal input load (m=0)
        
        factors_y_lm = (Re / T_e_0)**3 * (rho_c * g0 * Re) / E
        y_lm_synthetic = factors_y_lm * h_synthetic
        
        y_lm_unstr_syn = pysh.shio.SHCilmToVector(y_lm_synthetic)
        y_lm_str_syn = np.array([find_custom_element(l_v, m_v, y_lm_unstr_syn) for l_v, m_v in mode_map])
        
        rhs_dense = np.concatenate([y_lm_str_syn, np.zeros(N_modes)])
        
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
        if AXISYMMETRIC == False:
            for idx, (l_val, m_val) in enumerate(mode_map):
                if m_val >= 0:
                    w_coeffs_np[0, l_val, m_val] = float(w_sol[idx])
                else:
                    w_coeffs_np[1, l_val, abs(m_val)] = float(w_sol[idx])
        else:
            for idx, (l_val, _) in enumerate(mode_map):
                w_coeffs_np[0, l_val, 0] = float(w_sol[idx])  # Zonal component only            

        # Multiplication by -1 depending on the convention of positive displacement
        w_sol_clm = pysh.SHCoeffs.from_array(-1*w_coeffs_np, normalization='ortho')
        w_sol_grid = w_sol_clm.expand()

        # Extract precise 1D Axisymmetric cross-sections matching the paper
        spatial_latitudes = w_sol_grid.lats()
        colatitudes_deg = 90.0 - spatial_latitudes  # Mapping latitude indices directly to colatitude
        displacement_profile_beuthe = w_sol_grid.data[:, 0]  # Isolated zonal line profile



        # 2. LOCAL SOLUTION VIA TURCOTTE APPROXIMATION (EQ. 20)
        # Generate spatial layout grid for each isolated target input harmonic load
        h_single_clm = pysh.SHCoeffs.from_array(h_synthetic, normalization='ortho')
        h_single_grid = h_single_clm.expand()
        h_spatial_profile = h_single_grid.data[:, 0] # Zonal spatial slice of input load
        
        # Extract 1D spatial layout thickness slice matching our zonal line
        T_e_spatial_profile = active_Te_array[:, 0]
        
        # Compute Turcotte local response point-by-point along spatial profile grid
        displacement_profile_turcotte = np.zeros_like(colatitudes_deg)
        for lat_idx in range(len(colatitudes_deg)):
            Te_local = T_e_spatial_profile[lat_idx]
            h_local = h_spatial_profile[lat_idx]
            
            C_l_local = C_l_functional(l_load, nu, E, Te_local, Re, rho_m, rho_c, g0)
            
            # Equation 20: w = rho_term * C_l * h
            displacement_profile_turcotte[lat_idx] = rho_term * C_l_local * h_local

        # Plot both profile configurations together on common subplot axis
        axes[ax_idx].plot(colatitudes_deg, displacement_profile_beuthe, color='black', label='TSA-B', linewidth=2)
        axes[ax_idx].plot(colatitudes_deg, displacement_profile_turcotte, color='red', linestyle='--', label='TSA-T', linewidth=1.5)
        
        axes[ax_idx].set_title(f"Load degree $\ell$ = {l_load}")
        axes[ax_idx].set_xlim(0, 180)
        axes[ax_idx].set_ylim(-8000, 8000)
        axes[ax_idx].set_xlabel(r"Colatitude $\theta$ (deg)")
        axes[ax_idx].set_ylabel("Displacement $w$ (m)")
        axes[ax_idx].legend(loc='lower right', fontsize=8)
        axes[ax_idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs('Plots/M1VarD_SPEC_opt_ver results/', exist_ok=True)
    plt.savefig(f'Plots/M1VarD_SPEC_opt_ver results/Kalousova_{model_name.replace(" ", "_")}_lmax{lmax}_1D={AXISYMMETRIC}.png', dpi=200)
    plt.show()

end = time.time()
print("\n--- Entire Verification System Run Complete ---")
print("Total runtime:", round(end - start, 1), "seconds")