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
# from sympy import cot
from math import pi
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

lmax = 100  # Maximum spherical harmonic degree to perform all calculations

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

wlm = np.zeros(shape)

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
    wlm[: , degree , : degree+1] = (factors_w[degree]
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
plt.semilogx(l[2:], factors_w[2:])
plt.ylabel('Multiplication factor with H'), plt.xlabel('Spherical harmonic degree')
plt.grid()
plt.show()



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Solve for first estimate of F using Beuthe eq (87)
numerator_F = (E*T_e/Re)
denominator_F = (Lapl+2) - (1+nu)

factors_F = numerator_F / denominator_F
factors_F[:2] = 0

Flm = np.zeros(shape)


for degree in range(2, lmax + 1):  # Ignore degree 0 from calculations
    Flm[: , degree , : degree+1] = (factors_F[degree]
                                    * wlm_coeffs.coeffs[: , degree , : degree+1])

Flm_coeffs = pysh.SHCoeffs.from_array(Flm)
Flm_grid = Flm_coeffs.expand()


plt.figure()        
Flm_grid.plot(cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= 'Stress function F with constant D'
              )





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################
### SOLVE w AND F SIMULTANEOUSLY IN SYSTEM ###
##############################################
K = E*T_e/(1-nu**2)
xi = Re**2*K/D
eta = xi/(1+xi)

# eqs 86 and 87 of Beuthe
# HOW DO I DO THIS?






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
T_e_coeffs = pysh.SHCoeffs.from_random(power, lmax=lmax, seed=seed)
T_e_array = T_e_coeffs.expand().to_array() + 150
T_e_grid = pysh.SHGrid.from_array(T_e_array)


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
D_array = E*(T_e_array*1e3)**3 / (12*(1-nu**2))
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
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic T_e map, km'
              )
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
    dpar1 = par_coeffs.gradient()
    dpar1_theta_coeffs = dpar1.theta.expand()               #dpar1/dtheta
    dpar1_phi_coeffs = dpar1.phi.expand()                   #dpar1/dphi
    
    # Calculate second derivatives
    dpar2_theta2 = dpar1_theta_coeffs.gradient().theta      #dpar2/dtheta2
    dpar2_thetaphi = dpar1_theta_coeffs.gradient().phi      #dpar2/dthetadphi
    dpar2_phi2 = dpar1_phi_coeffs.gradient().phi            #dpar/dphi2
    
    # # For verification, below derivative should be approximately equal to
    # # dpar2_thetaphi
    # dpar2_phitheta = dpar1_phi_coeffs.gradient().theta      #d2w/dphidtheta
    
    # Return only the coefficients of the derivatives
    return (dpar1, dpar1_theta_coeffs, dpar1_phi_coeffs, 
            dpar2_theta2, dpar2_phi2, dpar2_thetaphi)
    
def derivative_arrays(par_coeffs):
    """
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
    
    # Call derivative function
    (dpar1, dpar1_theta_coeffs, dpar1_phi_coeffs, 
    dpar2_theta2, dpar2_phi2, dpar2_thetaphi) = get_derivatives(par_coeffs)
    
    # Make arrays for computations
    par_array = par_coeffs.expand().to_array()
    dpar1_theta_array = dpar1.theta.to_array()
    dpar1_phi_array = dpar1.phi.to_array()
    dpar2_theta2_array = dpar2_theta2.to_array()
    dpar2_phi2_array = dpar2_phi2.to_array()
    dpar2_thetaphi_array = dpar2_thetaphi.to_array()


    return (par_array, dpar1_theta_array, dpar1_phi_array, dpar2_theta2_array, 
            dpar2_phi2_array, dpar2_thetaphi_array)

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
(dw1, dw1_theta_coeffs, dw1_phi_coeffs, 
 dw2_theta2, dw2_phi2, dw2_thetaphi) = get_derivatives(wlm_coeffs)

(w_array, dw1_theta_array, dw1_phi_array, dw2_theta2_array, 
        dw2_phi2_array, dw2_thetaphi_array) = derivative_arrays(wlm_coeffs)

# plot_derivatives(wlm_coeffs, 'w')        # Plot derivatives of w


""" Derivatives of F in SH coeffs and in arrays """
(dF1, dF1_theta_coeffs, dF1_phi_coeffs, 
 dF2_theta2, dF2_phi2, dF2_thetaphi) = get_derivatives(Flm_coeffs)

(F_array, dF1_theta_array, dF1_phi_array, dF2_theta2_array, 
        dF2_phi2_array, dF2_thetaphi_array) = derivative_arrays(Flm_coeffs)

# plot_derivatives(Flm_coeffs, 'F')        # Plot derivatives of F


""" Derivatives of D in SH coeffs and in arrays """
(dD1, dD1_theta_coeffs, dD1_phi_coeffs, 
 dD2_theta2, dD2_phi2, dD2_thetaphi) = get_derivatives(Dlm_coeffs)

(D_array, dD1_theta_array, dD1_phi_array, dD2_theta2_array, 
        dD2_phi2_array, dD2_thetaphi_array) = derivative_arrays(Dlm_coeffs)

# plot_derivatives(Dlm_coeffs, 'D')        # Plot derivatives of D


""" Derivatives of alpha in SH coeffs and in arrays """
(da1, da1_theta_coeffs, da1_phi_coeffs, 
 da2_theta2, da2_phi2, da2_thetaphi) = get_derivatives(alm_coeffs)

(a_array, da1_theta_array, da1_phi_array, da2_theta2_array, 
        da2_phi2_array, da2_thetaphi_array) = derivative_arrays(alm_coeffs)

# plot_derivatives(alm_coeffs, 'a')        # Plot derivatives of a








### OPERATOR A (VECTORIZED) ###
# Create grids for theta values
# pyshtools grids are usually (lmax+1, 2*lmax+1) or (2*lmax+1, 4*lmax+1)
# Make sure your rad_angle_theta matches your grid's latitude array
thetas = wlm_coeffs.expand().lats() * (np.pi / 180) # Convert to radians
theta_grid, _ = np.meshgrid(thetas, np.zeros(dw2_phi2_array.shape[1]))
theta_grid = theta_grid.T # Shape must match your data arrays

def cot(x): return np.where(np.isclose(np.sin(x), 0), 0, 1.0 / np.tan(x))
def csc2(x): return np.where(np.isclose(np.sin(x), 0), 1.0, 1.0 / (np.sin(x)**2))

# Vectorized Operator A
C2 = csc2(theta_grid)
CT = cot(theta_grid)

A_Dw = (
    (dD2_theta2_array + D_array) * (C2 * dw2_phi2_array + CT * dw1_theta_array + w_array)
    + (C2 * dD2_phi2_array + CT * dD1_theta_array + D_array) * (dw2_theta2_array + w_array) 
    - 2 * C2 * (dD2_thetaphi_array - CT * dD1_phi_array) * (dw2_thetaphi_array - CT * dw1_phi_array)
)
A_aF = (
    (da2_theta2_array + a_array) * (C2 * dF2_phi2_array + CT * dF1_theta_array + F_array)
    + (C2 * da2_phi2_array + CT * da1_theta_array + a_array) * (dF2_theta2_array + F_array) 
    - 2 * C2 * (da2_thetaphi_array - CT * da1_phi_array) * (dF2_thetaphi_array - CT * dF1_phi_array)
)


# Now the term with differential operator A is in SH coefficients
A_Dw_lm = pysh.SHGrid.from_array((1-nu)*A_Dw).expand()
A_aF_lm = pysh.SHGrid.from_array((1+nu)*A_aF).expand()

w, F = symbols(' w, F ')


for degree in range(2, lmax+1):
    
    # Writing the two equations of Beuthe's conclusion now in SH coeffs:
    # eq1
    term1_eq1 = ( (Lapl[degree]+2)*Dlm_coeffs.coeffs[: , degree , : degree+1]*(Lapl[degree]+2) 
                + Re**4 * g0 * (rho_m-rho_c) )                                                      # w-terms
    term2_eq1 = Re**3 * (Lapl[degree]+2)                                                            # F-terms
    term3_eq1 = ( -Re**4 *g0 * rho_l * topo_clm.coeffs[: , degree , : degree+1] 
                + A_Dw_lm.coeffs[: , degree , : degree+1] )                                         # RHS
    
    #eq2
    term1_eq2 = -1/Re * (Lapl[degree]+2)                                                            # w-terms
    term2_eq2 = (Lapl[degree]+2)*alm_coeffs.coeffs[: , degree , : degree+1]*(Lapl[degree]+2)        # F-terms
    term3_eq2 = A_aF_lm.coeffs[: , degree , : degree+1]                                             # RHS
    
    
    A = np.array([[term1_eq1, term2_eq1], [term1_eq2, term2_eq2]])
    b = np.array([[term3_eq1], [term3_eq2]])
    
    system = (A,b)
    
    linsolve(system, [w, F])
