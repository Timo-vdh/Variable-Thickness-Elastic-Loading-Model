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

w_array = wlm_coeffs.expand().to_array()


# First derivative of the transverse displacement w
dw = wlm_coeffs.gradient()
# Expand the first gradient grids to sh coeffs in order to take gradient again
dw_theta_coeffs = dw.theta.expand()     #dw/dtheta
dw_phi_coeffs = dw.phi.expand()         #dw/dphi

# Calculate second derivatives
dw2_theta2 = (dw_theta_coeffs.gradient()).theta       #d2w/dtheta2
dw2_thetaphi = (dw_theta_coeffs.gradient()).phi       #d2w/dthetadphi
dw2_phi2 = (dw_phi_coeffs.gradient()).phi             #d2w/dphi2
dw2_phitheta = (dw_phi_coeffs.gradient()).theta      #d2w/dphidtheta

# Make arrays for computations
dw_theta_array = dw.theta.to_array()
dw_phi_array = dw.phi.to_array()
dw2_theta2_array = dw2_theta2.to_array()
dw2_phi2_array = dw2_phi2.to_array()
dw2_thetaphi_array = dw2_thetaphi.to_array()

# Plotting the first gradients in theta and phi direction
dw.plot_theta(title='dw/dtheta') 
dw.plot_phi(title='dw/dphi')

# Plotting the second gradients in theta and phi direction
dw_theta_coeffs.gradient().plot_theta(title='d2w/dtheta2')
dw_phi_coeffs.gradient().plot_phi(title='d2w/dphi2')
dw_theta_coeffs.gradient().plot_phi(title='d2w/dthetadphi')
dw_phi_coeffs.gradient().plot_theta(title='d2w/dphidtheta')


dD2_theta2_array = np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
dD2_phi2_array = np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
dD_theta_array = np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
dD2_thetaphi_array = np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
D_array = np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
# First derivative of Rigidity D 



    

# ### OPERATOR A ###
# A_Dw = np.zeros([2*(lmax+1)+1, 4*(lmax+1)+1])

# rad_angle_theta = pi/(2*lmax)
# rad_angle_phi = pi/(2*lmax)

# for theta in range(2*(lmax+1)+1):
#     for phi in range(4*(lmax+1)+1):
#         if theta != 0:
#           A_Dw[theta,phi] = (
#                         (dD2_theta2_array[theta,phi] + D_array[theta,phi]) 
#                         * (csc2(theta*rad_angle_theta) * dw2_phi2_array[theta,phi] 
#                         + cot(theta*rad_angle_theta) * dw_theta_array[theta,phi]
#                         + w_array[theta,phi])
                        
#                         + (csc2(theta*rad_angle_theta) * dD2_phi2_array[theta,phi] 
#                         + cot(theta*rad_angle_theta) * dD_theta_array[theta,phi]
#                         + D_array[theta,phi])
                        
#                         - 2*csc2(theta*rad_angle_theta)
#                         * (dD2_thetaphi_array[theta,phi] 
#                            - cot(theta*rad_angle_theta)*dD_theta_array[theta,phi])
#                         * (dw2_thetaphi_array[theta,phi] 
#                            - cot(theta*rad_angle_theta)*dw_theta_array[theta,phi])
#                         )
#         else:
#             # Prevent error at the pole
#             A_Dw[theta,phi] = 1

# print(rad_angle_theta*2*(lmax+1)+1)



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
    (dD2_theta2_array + D_array) * (C2 * dw2_phi2_array + CT * dw_theta_array + w_array)
    + (C2 * dD2_phi2_array + CT * dD_theta_array + D_array) # Add indexing if necessary
    - 2 * C2 * (dD2_thetaphi_array - CT * dD_theta_array) * (dw2_thetaphi_array - CT * dw_theta_array)
)

A_Dw_lm = pysh.SHGrid.from_array((1-nu)*A_Dw).expand()


