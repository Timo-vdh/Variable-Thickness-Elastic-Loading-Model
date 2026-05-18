# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:26:07 2026

@author: Timov
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from cartopy import crs as ccrs
from palettable import scientific as scm
import scipy.sparse as sp
from sympy.physics.quantum.cg import CG


#######################################
# Test file to try and understand all required functions, 
# parameters, outputs, and processes required for the 
# constant thickness model as derived by Beuthe (2008),
# equation (88).

# Final equation for the transverse displacement w depends
# only on the load q and the consoidal component of the tangential
# load Omega (the equation with stress function F is eliminated)

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



### EQUATIONS ###
# D : Flexural rigidity [Nm]
# D = E*T_e^3/(12*(1-nu^2))

# K : Extensional rigidity [N/m]
# K = E*T_e/(1-nu^2)

# q_lm : Lithospheric loading in spherical harmonics
# q_lm = w_lm * (drho_l*g0)



### OPERATORS ###
# Delta w = d^2/dtheta^2 (w) + cot(theta)*d/dtheta (w) 
#            + csc^2(theta) d^2/dtheta^2 (w)

# Delta_p w = (Delta + 2) w



#######################################

# INPUTS
nu = 0.25
E = 100.0e9
rho_c = 2900.
rho_m = 3500.
rho_l = rho_c



# D = E*T_e**3/(12*(1-nu**2))
# K = E*T_e/(1-nu**2)

# xi = R**2*K/D
# eta = xi/(1+xi)
# alpha = 1/(K*(1-nu**2))

######################################

lmax = 40  # Maximum spherical harmonic degree to perform all
# calculations

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



#####################################
### Finding deformation after derivation of constant thickness model
# using Beuthe eq (88) ###
l = np.arange(lmax + 1, dtype=float)
Lapl= -l*(l+1)


# inputs
T_e = 150e3

if 10e3 < T_e < 500e3: # Prevent issues when testing limit cases
    Re = R-T_e/2
else:
    Re = R-150e3/2
    
D = E*T_e**3/(12*(1-nu**2))
dc = 0      # Crustal thickness variations delta c (used in Banerdt)
dp = 0      # Crustal density variations delta rho (used in Banerdt)
M = 0       # Thickness of density anomaly in mantle

shape = (2, lmax + 1, lmax + 1)

wlm = np.zeros(shape)

# Constant thickness calculation in order to obtain first guess for w
numerator = -Re**4*g0*(Lapl+1-nu)*rho_l
denominator = (
                D*Lapl**3 + 4*D*Lapl**2 
                + (4*D + Re**2*E*T_e)*Lapl 
                + 2*Re**2*E*T_e 
                + Re**4*g0*(rho_m-rho_c)*(Lapl+1-nu)
                )



factors = numerator / denominator
factors[0]=0


for degree in range(2, lmax + 1):  # Ignore degree 0 from calculations
    wlm[: , degree , : degree+1] = (factors[degree]
                                    * topo_clm.coeffs[: , degree , : degree+1])

wlm_coeffs = pysh.SHCoeffs.from_array(wlm/1e3)
wlm_grid = wlm_coeffs.expand()
topo_grid = (topo_clm/1e3).expand() - R/1e3
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))


topo_grid.plot(ax=ax1,
                cmap=mycmap,
                colorbar='right',
                cb_label='Elevation, km',
                cmap_limits=[-6, 10]
                )
wlm_grid.plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= f'Displacement at ref. surface [km] - T_e={int(T_e/1e3)} km'
              )

r_Airy = -rho_c/(rho_m-rho_c) * topo_grid
r_Airy.plot(ax=ax3,
            cmap=mycmap,
            colorbar='right',
            cb_label='Airy, km',
            # cmap_limits=[-50,30]
            )

# geoid_grid = geoid_clm.expand()
# fig2, ax2 = geoid_grid.plot()



# Add zero elevation/displacement contour
ax1.contour(
    topo_grid.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper"
)
ax2.contour(
    wlm_grid.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper"
)

# Plot value for multiplication factor with topography per spherical harmonic degree
plt.figure()
plt.semilogx(l[2:], factors[2:])
plt.ylabel('Multiplication factor with H'), plt.xlabel('Spherical harmonic degree')
plt.grid()
plt.show()








######################
### SOLVING A IN SPATIAL DOMAIN ###
# Here the gradients of the parameters D, w, alpha and F are calculated
# in order to solve the differential operators A(D;w) and A(alpha;F).
# A first guess is needed for all parameters in order to calculate the gradients.
# For D
# For w this is the constant thickness deflection
# For F this is set to 0


# First derivative of the transverse displacement w
dw = wlm_coeffs.gradient()
# Expand the first gradient grids to sh coeffs
dw_theta_coeffs = dw.theta.expand()     #dw/dtheta
dw_phi_coeffs = dw.phi.expand()         #dw/dphi

# Calculate second derivative and directly expand second gradient grids into sh coeffs
dw2_theta_coeffs = (dw_theta_coeffs.gradient()).theta.expand()  #d2w/dtheta2
dw_thetaphi_coeffs = (dw_theta_coeffs.gradient()).phi.expand()  #d2w/dthetadphi
dw2_phi_coeffs = (dw_phi_coeffs.gradient()).phi.expand()        #d2w/dphi2


# # Plotting the first gradients in theta and phi direction
# dw.plot_theta(title='dw/dtheta') 
# dw.plot_phi(title='dw/dphi')

# # Plotting the second gradients in theta and phi direction
# dw_theta_coeffs.gradient().plot_theta(title='d2w/dtheta2')
# dw_theta_coeffs.gradient().plot_phi(title='d2w/dthetadphi')
# dw_phi_coeffs.gradient().plot_phi(title='d2w/dphi2')




### SOLVING A IN SPECTRAL DOMAIN ###
# Here the operator A is directly solved in the spectral domain using
# the equations derived by Kalousova et al. (2012). 

# First need to expand all grids into spherical harmonic coefficients
# D(theta,phi)      --> D_lm            [D=ET_e**3/(12(1-nu**2))]
# alpha(theta,phi)  --> alpha_lm        [alpha=1/(ET_e]
# w(theta,phi)      --> w_lm
# F(theta,phi)      --> F_lm

# Initial guess for w using constant thickness
wlm_coeffs1 = wlm_coeffs
# Initial guess for F is all zero
F_coeffs1 = pysh.SHCoeffs.from_array(np.zeros(shape)) 

# Initial guess for alpha and D follow from T_e coeffs, 
# using the constant thickness value
T_e1 = 150e3
T_e1_array = np.ones([lmax+1,lmax+1])*T_e1
T_e_grid1 = pysh.SHGrid.from_array(T_e1_array)
T_e_coeffs1 = pysh.SHCoeffs.from_array(np.ones(shape)*T_e1)


alpha_array1 = 1/(E*T_e1_array)
alpha_grid1 = pysh.SHGrid.from_array(alpha_array1)
alpha_coeffs1 = alpha_grid1.expand()

D_array1 = E*T_e1_array**3/(12*(1-nu**2))
D_grid1 = pysh.SHGrid.from_array(D_array1)
D_coeffs1 = D_grid1.expand()










n_coeffs = (lmax + 1)**2

def get_idx(l, m):
    """Maps (l, m) to a unique 1D index. 
    m > 0 for cosine, m < 0 for sine."""
    return l**2 + l + m

# Precompute delta and delta_prime arrays
l_arr = np.arange(lmax + 1)
delta = l_arr * (l_arr + 1)
delta_p = 2 - delta

def interaction_polynomial(l, l_prime, L):
    d_l = l * (l + 1)
    d_lp = l_prime * (l_prime + 1)
    d_L = L * (L + 1)
    
    term1 = d_l**2 + d_lp**2 + d_L**2
    term2 = 2 * (d_l + d_lp + d_L)
    term3 = 2 * (d_l * d_lp + d_l * d_L + d_lp * d_L)
    
    return 0.25 * (term1 + term2 - term3 - 8)



# Initialize Block Matrices
# System size is 2 * n_coeffs (for w and F)
LHS = np.zeros((2 * n_coeffs, 2 * n_coeffs))
RHS = np.zeros(2 * n_coeffs)

# Flatten your topography and geoid into the RHS
topo_coeffs = topo_clm.coeffs
for l in range(lmax + 1):
    for m in range(-l, l + 1):
        idx = get_idx(l, m)
        # Load q_lm (Equation A16 RHS)
        # q = -g0 * rho_l * (H_lm - Geoid_lm)
        h_val = topo_coeffs[0, l, m] if m >= 0 else topo_coeffs[1, l, abs(m)]
        RHS[idx] = -Re**4 * g0 * rho_l * h_val

# Populate Matrix (Triple Loop: Row l, Col l', Property L)
# Note: In production, use selection rules to skip zeros!
for l in range(lmax + 1):
    for m in range(-l, l + 1):
        row = get_idx(l, m)
        
        # diagonal-only coupling terms (a and b matrices)
        # R^3 * Delta' * F term in Eq 75
        LHS[row, n_coeffs + row] = Re**3 * delta_p[l]
        # -1/R * Delta' * w term in Eq 76
        LHS[n_coeffs + row, row] = -(1.0 / Re) * delta_p[l]
        
        # Buoyancy term (moves to LHS)
        # + Re^4 * g0 * (rho_m - rho_c) * (Delta' * w)
        LHS[row, row] += Re**4 * g0 * (rho_m - rho_c) * delta_p[l]

        for lp in range(lmax + 1):
            for mp in range(-lp, lp + 1):
                col = get_idx(lp, mp)
                
                # The property degree L must satisfy |l-lp| <= L <= l+lp
                for L in range(abs(l - lp), l + lp + 1):
                    # For a variable thickness, we look for property coeff D_LM
                    # where M = m - mp
                    M = m - mp
                    if abs(M) > L: continue
                    
                    # Fetch D_LM and Alpha_LM from your expanded coefficients
                    D_LM = D_coeffs1.coeffs[0, L, M] if M >= 0 else D_coeffs1.coeffs[1, L, abs(M)]
                    A_LM = alpha_coeffs1.coeffs[0, L, M] if M >= 0 else alpha_coeffs1.coeffs[1, L, abs(M)]
                    
                    # Compute Gaunt Coefficient Q (simplified placeholder)
                    # You need a library call here for Q_lm_lpmp_LM
                    Q = 1.0 # Replace with actual Gaunt/Clebsch-Gordan coupling
                    
                    # # CG(j1, m1, j2, m2, j3, m3).doit()
                    # val = float(CG(l1, m1, l2, m2, l, m).doit())
                    
                    # Matrix A (Bending)
                    # term: delta_p[l]*delta_p[lp] - (1-nu)*I
                    val_A = D_LM * Q * (delta_p[l] * delta_p[lp] - (1 - nu) * interaction_polynomial(l, lp, L))
                    LHS[row, col] += val_A
                    
                    # Matrix B (Stretching)
                    # term: delta_p[l]*delta_p[lp] - (1+nu)*I
                    val_B = A_LM * Q * (delta_p[l] * delta_p[lp] - (1 + nu) * interaction_polynomial(l, lp, L))
                    LHS[n_coeffs + row, n_coeffs + col] += val_B
                    
                    
                    
# Solve the system
X = np.linalg.solve(LHS, RHS)

# Split the solution back into w and F
w_vec = X[:n_coeffs]
F_vec = X[n_coeffs:]

# Map back to SHCoeffs object
w_final_coeffs = np.zeros(shape)
F_final_coeffs = np.zeros(shape)

for l in range(lmax + 1):
    for m in range(-l, l + 1):
        idx = get_idx(l, m)
        if m >= 0:
            w_final_coeffs[0, l, m] = w_vec[idx]
            F_final_coeffs[0, l, m] = F_vec[idx]
        else:
            w_final_coeffs[1, l, abs(m)] = w_vec[idx]
            F_final_coeffs[1, l, abs(m)] = F_vec[idx]

# Expand to grid for plotting
w_final_grid = pysh.SHCoeffs.from_array(w_final_coeffs).expand()

w_final_grid.plot(cmap=mycmap)

