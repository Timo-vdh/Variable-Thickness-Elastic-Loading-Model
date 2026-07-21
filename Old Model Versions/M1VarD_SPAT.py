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
from sympy import linsolve, symbols, Matrix

#######################################
"""
Model for the variable thickness deformations of Beuthe (2008).
Current model works with Beuthe's equations 75 and 76, 
neglecting tangential loading.

This requires implementation of the differential operator A(a;b).
Beuthe does not give a spectral method for this, but Kalousova et al. (2012)
do. Two approaches are worked on in this code: 
    1) a transformation of this operator to the spatial domain, calculating 
       derivatives there, then transforming back to spectral domain and 
       solving for displacement w and stress function F in spectral domain.
    2) Applying Kalousova et al.'s approach in spectral domain directly.
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


#######################################

# INPUTS
nu = 0.25
E = 100.0e9
rho_c = 2900.
rho_m = 3500.
rho_l = rho_c



######################################

lmax = 120  # Maximum spherical harmonic degree to perform all calculations

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

wlm1 = np.zeros(shape)

# Constant thickness calculation in order to obtain first guess for w
numerator_w = -Re**4*g0*(Lapl+1-nu)*rho_l
denominator_w = (
                D*Lapl**3 + 4*D*Lapl**2 
                + (4*D + Re**2*E*T_e)*Lapl 
                + 2*Re**2*E*T_e 
                + Re**4*g0*(rho_m-rho_c)*(Lapl+1-nu)
                )



factors_w = numerator_w / denominator_w
factors_w[:2]=0


for degree in range(2, lmax + 1):  # Ignore degree 0 from calculations
    wlm1[: , degree , : degree+1] = (factors_w[degree]
                                    * topo_clm.coeffs[: , degree , : degree+1])

wlm1_coeffs = pysh.SHCoeffs.from_array(wlm1)
wlm1_grid = wlm1_coeffs.expand()
topo_grid = (topo_clm).expand() - R
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

topo_grid_plot = (topo_clm/1e3).expand() - R/1e3
topo_grid_plot.plot(ax=ax1,
                cmap=mycmap,
                colorbar='right',
                cb_label='Elevation, km',
                cmap_limits=[-6, 10]
                )
wlm1_grid.plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= f'Displacement at ref. surface [m] - T_e={int(T_e/1e3)} km'
              )

r_Airy = -rho_c/(rho_m-rho_c) * topo_grid
r_Airy.plot(ax=ax3,
            cmap=mycmap,
            colorbar='right',
            cb_label='Airy, m',
            # cmap_limits=[-50,30]
            )

# geoid_grid = geoid_clm.expand()
# fig2, ax2 = geoid_grid.plot()



# Add zero elevation/displacement contour
ax1.contour(
    topo_grid.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper"
)
ax2.contour(
    wlm1_grid.data > 0, levels=[0.99], extent=(0, 360, -90, 90), colors="k", origin="upper"
)

# # Plot value for multiplication factor with topography per spherical harmonic degree
# plt.figure()
# plt.semilogx(l[2:], factors_w[2:])
# plt.ylabel('Multiplication factor with H'), plt.xlabel('Spherical harmonic degree')
# plt.grid()
# plt.show()



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Solve for first estimate of F using Beuthe eq (87)
numerator_F = (E*T_e/Re)
denominator_F = (Lapl+2) - (1+nu)

factors_F = numerator_F / denominator_F
factors_F[:2] = 0

Flm1 = np.zeros(shape)


for degree in range(2, lmax + 1):  # Ignore degree 0 from calculations
    Flm1[: , degree , : degree+1] = (factors_F[degree]
                                    * wlm1_coeffs.coeffs[: , degree , : degree+1])

Flm1_coeffs = pysh.SHCoeffs.from_array(Flm1)
Flm1_grid = Flm1_coeffs.expand()


plt.figure()        
Flm1_grid.plot(cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= 'Stress function F with constant D'
              )


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
############################################
### CREATING A SYNTHETIC VARIABLE Te MAP ###
############################################

# # Initialize randomizer
# seed = 1
# l_corner = 10
# beta = 3.0
# power = np.zeros(lmax + 1)
# for li in range(2, lmax+1):
#     if li <= l_corner:
#         power[li] = 1.0
#     else:
#         power[li] = (l_corner / li) ** beta

# # Make a random coefficient map
# T_e_coeffs = pysh.SHCoeffs.from_random(power, lmax=lmax, seed=seed)
# T_e_array = T_e_coeffs.expand().to_array() + 150
# T_e_grid = pysh.SHGrid.from_array(T_e_array)



# Making a constant T_e map
const_T_e_grid = np.ones(shape)*T_e

T_e_coeffs = pysh.SHCoeffs.from_array(const_T_e_grid)
T_e_grid = T_e_coeffs.expand()
T_e_array = T_e * np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
T_e_grid = pysh.SHGrid.from_array(T_e_array)

T_e_clm = T_e_grid.expand()
T_e_array2 = pysh.expand.MakeGridDH(T_e_clm.coeffs, lmax=lmax, sampling = 2)
T_e_grid2 = pysh.SHGrid.from_array(T_e_array2)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Calculate the variable Flexural rigidity D and parameter alpha (a) 
# and convert to spherical harmonics
# D = E*T_e^3/(12*(1-nu^2))
# alpha = 1/(E*T_e_lm)

Dlm = np.zeros(shape)
alm = np.zeros(shape)

# T_e_coeffs = input T_e map from GAIA, in SH coefficients
# T_e_array = T_e_coeffs.expand().to_array()

# First calculate in array form, then convert result to SHGrid and then to SHCoeffs
D_array = E*(T_e_array)**3 / (12*(1-nu**2))
a_array = 1/(E*T_e_array)

D_grid = pysh.SHGrid.from_array(D_array)
a_grid = pysh.SHGrid.from_array(a_array)

Dlm_coeffs = D_grid.expand()
alm_coeffs = a_grid.expand()
 

# Plot the resulting synthetic T_e, D and alpha maps
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
T_e_grid.plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[149999, 150001],
              cb_label= 'Synthetic T_e map, km'
              )
# T_e_grid2.plot(ax=ax2,
#               cmap=mycmap,
#               colorbar='right',
#               # cb_tick_interval=2,
#               cmap_limits=[149999, 150001],
#               cb_label= 'Synthetic T_e map, km'
#               )
D_grid.plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic D map'
              )
a_grid.plot(ax=ax3,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic alpha map'
              )

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

###################################
### SOLVING A IN SPATIAL DOMAIN ###
###################################
# Here the gradients of the parameters D, w, alpha and F are calculated
# in order to solve the differential operators A(D;w) and A(alpha;F).
# A first guess is needed for all parameters in order to calculate the gradients.
# For D and alpha this is the input elastic thickness map (remains unchanged throughout)
# For w this is the constant thickness deflection (Beuthe eq 86)
# For F this is the constant thickness stress function (Beuthe eq 87) 

# Make some functions to get the derivatives in SHCoeffs and array forms
def get_derivatives(par_coeffs):
    """
    This function is used to get the first and second spatial derivatives
    of any parameter, using its spherical harmonic coefficients. 
    These derivatives are required to solve the differential operator A(a;b)
    in the spatial domain, which is required to solve for a variable thickness
    shell following equations (75) and (76) from Beuthe (for Omega=0).
    
    Input:
        - Spherical harmonic coefficients of a variable 'par'
        
    Output:
        - dpar1/dtheta     --> First derivative of variable 'par' wrt theta
        - dpar1/dphi       --> First derivative of variable 'par' wrt phi
        - dpar2/dtheta2    --> Second derivative of variable 'par' wrt theta twice
        - dpar2/dphi2      --> Second derivative of variable 'par' wrt phi twice
        - dpar2/dthetadphi --> Second derivative of variable 'par' wrt theta and phi
    """

    # Calculate first derivatives
    dpar1 = par_coeffs.gradient()                             # Gradient file
    dpar1_theta = dpar1.theta                                 #dpar1/dtheta, grid
    dpar1_phi = dpar1.phi                                     #dpar1/dphi, grid
    
    # Calculate second derivatives
    dpar2_theta2 = dpar1.theta.expand().gradient().theta      #dpar2/dtheta2, grid
    dpar2_thetaphi = dpar1.theta.expand().gradient().phi      #dpar2/dthetadphi, grid
    dpar2_phi2 = dpar1.phi.expand().gradient().phi            #dpar/dphi2, grid
    
    # # For verification, below derivative should be approximately equal to
    # # dpar2_thetaphi
    # dpar2_phitheta = dpar1_phi_coeffs.gradient().theta      #d2w/dphidtheta, grid
    
    # Return only the coefficients of the derivatives
    return (
            dpar1_theta, dpar1_phi, 
            dpar2_theta2, dpar2_phi2, dpar2_thetaphi
            )
    
def derivative_arrays(par_coeffs):
    """
    Computes pure partial derivatives on the spatial grid, 
    bypassing the 1/sin(theta) singular divisions at the poles.
    
    This function makes arrays of the derivatives to use in calculations
    Input:
        - Spherical harmonic coefficients of a variable 'par'
    Output:
        - par_array             --> Array of undifferentiated par
        - dpar1_theta_array     --> Array of dpar1/dtheta
        - dpar1_phi_array       --> Array of dpar1/dphi
        - dpar2_theta2_array    --> Array of dpar2/dtheta2
        - dpar2_phi2_array      --> Array of dpar12dphi2
        - dpar2_thetaphi_array  --> Array of dpar2/dthetaphi
    """
    
    grid = par_coeffs.expand()
    data = grid.data
    nlat, nlon = data.shape
    
    # Grid spacing in radians
    lats = np.radians(grid.lats())
    lons = np.radians(grid.lons())
    
    # We want colatitude grid steps
    thetas = np.pi/2.0 - lats
    dtheta = np.abs(thetas[1] - thetas[0])
    dphi = np.abs(lons[1] - lons[0])
    
    # 1. First derivatives via central differences (edge-padded to preserve shape)
    dpar1_theta_array = np.gradient(data, dtheta, axis=0)
    dpar1_phi_array = np.gradient(data, dphi, axis=1)
    
    # 2. Second derivatives
    dpar2_theta2_array = np.gradient(dpar1_theta_array, dtheta, axis=0)
    dpar2_phi2_array = np.gradient(dpar1_phi_array, dphi, axis=1)
    dpar2_thetaphi_array = np.gradient(dpar1_theta_array, dphi, axis=1)
    
    return (
        data, 
        dpar1_theta_array, dpar1_phi_array, dpar2_theta2_array, 
        dpar2_phi2_array, dpar2_thetaphi_array
    )

def plot_derivatives(par_coeffs, par):
    """
    This function plots the first and second derivatives of a variable
    
    Input:
        - Spherical harmonic coefficients of a variable 'par'
    
    Output:
        - Plots of dpar1/dtheta, dpar1/dphi, 
          dpar2/dtheta2, dpar2/dphi2, dpar2/dthetaphi
    """

    # Call derivative function
    dpar1, dpar1_theta_coeffs, dpar1_phi_coeffs, _, _, _ = get_derivatives(par_coeffs)

    # Plotting the first gradients in theta and phi direction
    dpar1.plot_theta(title=f'd{par}/dtheta') 
    dpar1.plot_phi(title=f'd{par}/dphi')

    # Plotting the second gradients in theta and phi direction
    dpar1_theta_coeffs.gradient().plot_theta(title=f'd{par}2/dtheta2')
    dpar1_phi_coeffs.gradient().plot_phi(title=f'd{par}2/dphi2')
    dpar1_theta_coeffs.gradient().plot_phi(title=f'd{par}2/dthetadphi')
    # dpar1_phi_coeffs.gradient().plot_theta(title='d2/dphidtheta')
    

""" Derivatives of w in SH coeffs and in arrays """
(w_array1, dw1_theta_array, dw1_phi_array, dw2_theta2_array, 
        dw2_phi2_array, dw2_thetaphi_array) = derivative_arrays(wlm1_coeffs)
# plot_derivatives(wlm1_coeffs, 'w')        # Plot derivatives of w


""" Derivatives of F in SH coeffs and in arrays """
(F_array1, dF1_theta_array, dF1_phi_array, dF2_theta2_array, 
        dF2_phi2_array, dF2_thetaphi_array) = derivative_arrays(Flm1_coeffs)
# plot_derivatives(Flm_coeffs, 'F')        # Plot derivatives of F


""" Derivatives of D in SH coeffs and in arrays """
(D_array1, dD1_theta_array, dD1_phi_array, dD2_theta2_array, 
        dD2_phi2_array, dD2_thetaphi_array) = derivative_arrays(Dlm_coeffs)
# plot_derivatives(Dlm_coeffs, 'D')        # Plot derivatives of D


""" Derivatives of alpha in SH coeffs and in arrays """
(a_array1, da1_theta_array, da1_phi_array, da2_theta2_array, 
        da2_phi2_array, da2_thetaphi_array) = derivative_arrays(alm_coeffs)
# plot_derivatives(alm_coeffs, 'a')        # Plot derivatives of a


# --- DOMAIN SCALING FACTORS ---
# Use the constant baseline values as your characteristic reference scales
D_scale = float(E * T_e**3 / (12 * (1 - nu**2)))  # ~2.81e25
a_scale = float(1.0 / (E * T_e))                  # ~6.66e-17

# Non-dimensionalize the arrays prior to calculating Operator A
D_array = D_array / D_scale
dD1_theta_array = dD1_theta_array / D_scale
dD1_phi_array   = dD1_phi_array / D_scale
dD2_theta2_array = dD2_theta2_array / D_scale
dD2_phi2_array   = dD2_phi2_array / D_scale
dD2_thetaphi_array = dD2_thetaphi_array / D_scale

a_array = a_array / a_scale
da1_theta_array = da1_theta_array / a_scale
da1_phi_array   = da1_phi_array / a_scale
da2_theta2_array = da2_theta2_array / a_scale
da2_phi2_array   = da2_phi2_array / a_scale
da2_thetaphi_array = da2_thetaphi_array / a_scale


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### OPERATOR A (VECTORIZED) ###
# Create grids for theta values
# pyshtools grids are usually (lmax+1, 2*lmax+1) or (2*lmax+1, 4*lmax+1)
# Make sure your rad_angle_theta matches your grid's latitude array
ref_grid = wlm1_coeffs.expand()
lats_deg  = ref_grid.lats()                       # geographic lat, degrees
lons_deg  = ref_grid.lons()                       # longitude, degrees
nlat      = len(lats_deg)
nlon      = len(lons_deg)
 
# Colatitude in radians: theta = pi/2 - lat_rad
theta_1d  = np.radians(90.0 - lats_deg)           # shape (nlat,), 0 to pi
# Broadcast to full grid shape (nlat, nlon)
theta_grid = np.tile(theta_1d[:, np.newaxis], (1, nlon))  # (nlat, nlon)
 
# Safe trig functions: avoid singularity exactly at poles
_sin = np.sin(theta_grid)
_sin_safe = np.where(np.abs(_sin) < 1e-50, 1e-50, _sin)
 
COT  = np.cos(theta_grid) / _sin_safe             # cot(theta)
CSC2 = 1.0 / _sin_safe**2                         # csc^2(theta)


A_Dw = (
    (dD2_theta2_array + D_array) * (CSC2 * dw2_phi2_array + COT * dw1_theta_array + wlm1_grid.data)
    + (CSC2 * dD2_phi2_array + COT * dD1_theta_array + D_array) * (dw2_theta2_array + wlm1_grid.data) 
    - 2 * CSC2 * (dD2_thetaphi_array - COT * dD1_phi_array) * (dw2_thetaphi_array - COT * dw1_phi_array)
)
A_aF = (
    (da2_theta2_array + a_array) * (CSC2 * dF2_phi2_array + COT * dF1_theta_array + Flm1_grid.data)
    + (CSC2 * da2_phi2_array + COT * da1_theta_array + a_array) * (dF2_theta2_array + Flm1_grid.data) 
    - 2 * CSC2 * (da2_thetaphi_array - COT * da1_phi_array) * (dF2_thetaphi_array - COT * dF1_phi_array)
)


# Now the term with differential operator A is in SH coefficients
A_Dw_lm_grid = pysh.SHGrid.from_array(A_Dw)
A_aF_lm_grid = pysh.SHGrid.from_array(A_aF)

A_Dw_lm = A_Dw_lm_grid.expand() * D_scale
A_aF_lm = A_aF_lm_grid.expand() * a_scale

# Plot the power spectrum of the operator A_Dw
fig, ax = A_Dw_lm.plot_spectrum2d(show=False)
ax.set_title("Power Spectrum of Operator A(D;w)")
ax.grid(True, which="both", ls="--", alpha=0.5)
# plt.ylim(1e-7,1)
plt.show()

fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
A_Dw_lm.expand().plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-4, 4],
              cb_label= 'A_Dw'
              )
A_aF_lm.expand().plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-4, 4],
              cb_label= 'A_aF'
              )


# If D and a are constant, the operator should reduce to A(D;w) = D*(nabla**2+2)*wlm
# Checking this:
A_Dw_const = np.zeros(shape)

for degree in range(2, lmax + 1):   #Ignore degree 0 from calculations
    A_Dw_const[: , degree , : degree+1] = (E*T_e**3/(12*(1-nu**2)) * 
                                           (Lapl[degree]+2)*wlm1_coeffs.coeffs[:, degree, : degree+1])

A_Dw_const = pysh.SHCoeffs.from_array(A_Dw_const)
A_Dw_const_grid = A_Dw_const.expand()

# This should be fully zero for constant D
check_eqA = A_Dw_lm_grid.data - A_Dw_const_grid.data/(E*T_e**3/(12*(1-nu**2)))

# Plot equation A(D;w) Grid result vs simplified A(D;w) Grid result
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
A_Dw_lm.expand().plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-2e31, 5e31],
              cb_label= 'A_Dw from eq'
              )
A_Dw_const_grid.plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cmap_limits=[-2e31, 5e31],
              cb_label= 'A_Dw with constant D'
              )

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### SOLVING FOR w AND F ITERATIVELY ###

wlm2 = np.zeros(shape)
Dlm_coeffs_p = np.zeros(shape)

for degree in range(2, lmax + 1):#Ignore degree 0 from calculations
    for degree_prime in range(2, lmax + 1):#Ignore degree 0 from calculations
        
        Dlm_coeffs_p[: , degree , : degree+1] += (Lapl[degree_prime]+2) * Dlm_coeffs.coeffs[: , degree , : degree+1]
    
    wlm2[: , degree , : degree+1]=(
                                    ((1-nu)*A_Dw_lm.coeffs[:,degree,:degree+1]
                                     - Re**3 * (Lapl[degree]+2) * Flm1_coeffs.coeffs[: , degree , : degree+1]
                                     - Re**4 * g0 * rho_l * topo_clm.coeffs[: , degree , : degree+1])
                                    /
                                    ( (Lapl[degree]+2) * Dlm_coeffs_p[: , degree , : degree+1]       # Think this step is wrong with Dlm
                                     + Re**4 * g0 * (rho_m - rho_c) )
                                    )

wlm2_coeffs = pysh.SHCoeffs.from_array(wlm2)
wlm2_grid = wlm2_coeffs.expand()

fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
wlm1_grid.plot(ax=ax1,
               cmap=mycmap,
               colorbar='right',
               # cb_tick_interval=2,
               cmap_limits=[-20e3, 20e3],
               cb_label= f'Displacement 1 at ref. surface [km] - T_e={int(T_e/1e3)} km'
               )
wlm2_grid.plot(ax=ax2,
               cmap=mycmap,
               colorbar='right',
               # cb_tick_interval=2,
               cmap_limits=[-20e3, 20e3],
               cb_label= f'Displacement 2 at ref. surface [km] - T_e={int(T_e/1e3)} km'
               )



Flm2 = np.zeros(shape)

for degree in range(2, lmax + 1):#Ignore degree 0 from calculations
    Flm2[: , degree , : degree+1]=(
                                    ((1+nu)*A_aF_lm.coeffs[:,degree,:degree+1]
                                     + 1/Re * (Lapl[degree]+2) * wlm2_coeffs.coeffs[: , degree , : degree+1])
                                    /
                                    ( (Lapl[degree]+2)**2 * alm_coeffs.coeffs[: , degree , : degree+1])       # Think this step is wrong with alm
                                    )

Flm2_coeffs = pysh.SHCoeffs.from_array(Flm2)
Flm2_grid = Flm2_coeffs.expand()

fig6, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
Flm1_grid.plot(ax=ax1,
               cmap=mycmap,
               colorbar='right',
               cb_label= f'Stress function 1 - T_e={int(T_e/1e3)} km'
               )
Flm2_grid.plot(ax=ax2,
               cmap=mycmap,
               colorbar='right',
               cb_label= f'Stress function 2 - T_e={int(T_e/1e3)} km'
               )

