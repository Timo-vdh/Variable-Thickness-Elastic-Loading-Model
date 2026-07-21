# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:37:18 2026

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
do. The approach worked on in this code: 
    - Determining operator A by finding the derivatives of the Legendre Polynomials
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

lmax = 3  # Maximum spherical harmonic degree to perform all calculations


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

# Plot the resulting synthetic T_e, D and alpha maps
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
T_e_grid.plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              # cmap_limits=[-4, 4],
              cb_label= 'Synthetic T_e map, m'
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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
###############################################
### FINDING LEGENDRE POLYNOMIAL DERIVATIVES ###
###############################################
from pathlib import Path 
from pyshtools.expand import SHGLQ

base_dir = Path.cwd()
save_folder = base_dir / "PolynomialDerivatives"

# Latitude to colatitude:
ref_grid = T_e_clm.expand()         # reference grid of latitudes and longitudes
lats_deg = ref_grid.lats()          # Latitudes in degrees
lons_deg = ref_grid.lons()          # Longitudes in degrees

colats_rad = np.radians(90 - lats_deg)
lons_rad = np.radians(lons_deg)

# theta = np.radians(90 - lats_deg)   # Colatitude
# phi = np.radians(lons_deg)
lmax = lmax

pi = np.pi

def SH_deriv(theta, phi, lmax):
    """
    Compute spherical harmonic derivatives at a given
    location (first and second order).

    Returns
    -------
    Y_lm_d1_theta_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d1_phi_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_theta_a : array, size(2,lmax+1,lmax+1)
        Array with the second derivative
        of Legendre polynomials with respect to colatitude.
    Y_lm_d2_phi_a : array, size(2,lmax+1,lmax+1)
        Array with the second derivative
        of Legendre polynomials with respect to longitude.
    Y_lm_d2_thetaphi_a : array, size(2,lmax+1,lmax+1)
        Array with the first derivative
        of Legendre polynomials with respect to colatitude and longitude.
    y_lm : array, size(2,lmax+1,lmax+1)
        Array of spherical harmonic functions.

    Parameters
    ----------
    theta : float
        Colatitude in radian.
    phi : float
        Longitude in radian.
    lmax : int
        Maximum spherical harmonic degree to compute for the derivatives.
    """
    shape = (2, lmax + 1, lmax + 1)
    Y_lm_d1_theta_a = np.zeros(shape)
    Y_lm_d1_phi_a = np.zeros(shape)
    Y_lm_d2_phi_a = np.zeros(shape)
    Y_lm_d2_thetaphi_a = np.zeros(shape)
    Y_lm_d2_theta_a = np.zeros(shape)
    y_lm = np.zeros(shape)

    cost = np.cos(theta)
    sint = np.sin(theta)
    if theta == 0 or theta == pi:
        dp_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        p_theta = np.zeros((int((lmax + 1) * (lmax + 2) / 2)))
        costsint = 0.0
        sintt = 0.0
    else:
        p_theta, dp_theta = pysh.legendre.PlmBar_d1(lmax, cost)
        dp_theta *= -sint  # Derivative with respect to
        # theta.
        costsint = cost / sint
        sintt = 1.0 / sint**2
    for l in range(lmax + 1):
        lapla = float(-l * (l + 1))
        for m in range(-l, l + 1):
            m_abs = np.abs(m)
            index = int(l * (l + 1) / 2 + m_abs)
            cosmphi = np.cos(m_abs * phi)
            sinmphi = np.sin(m_abs * phi)
            if m >= 0:
                msinmphi = -m * sinmphi  # First cos(m*phi)
                # derivative.
                m2cosphi = -(m**2) * cosmphi  # Second cos(m*phi)
                # derivative.
                Y_lm_d1_theta_a[0, l, m] = dp_theta[index] * cosmphi
                Y_lm_d1_phi_a[0, l, m] = p_theta[index] * msinmphi
                Y_lm_d2_phi_a[0, l, m] = p_theta[index] * m2cosphi
                Y_lm_d2_thetaphi_a[0, l, m] = dp_theta[index] * msinmphi
                y_lm[0, l, m] = p_theta[index] * cosmphi
            else:
                mcosmphi = m_abs * cosmphi
                m2sinphi = -(m_abs**2) * sinmphi
                Y_lm_d1_theta_a[1, l, m_abs] = dp_theta[index] * sinmphi
                Y_lm_d1_phi_a[1, l, m_abs] = p_theta[index] * mcosmphi
                Y_lm_d2_phi_a[1, l, m_abs] = p_theta[index] * m2sinphi
                Y_lm_d2_thetaphi_a[1, l, m_abs] = dp_theta[index] * mcosmphi
                y_lm[1, l, m_abs] = p_theta[index] * sinmphi

        if theta == 0 or theta == pi:
            Y_lm_d2_theta_a[:, l, : l + 1] = 0.0  # Not defined.
        else:
            # Make use of the Laplacian identity to estimate
            # last derivative.
            Y_lm_d2_theta_a[:, l, : l + 1] = (
                lapla * y_lm[:, l, : l + 1]
                - Y_lm_d1_theta_a[:, l, : l + 1] * costsint
                - sintt * Y_lm_d2_phi_a[:, l, : l + 1]
            )

    return (
        Y_lm_d1_theta_a,
        Y_lm_d1_phi_a,
        Y_lm_d2_theta_a,
        Y_lm_d2_phi_a,
        Y_lm_d2_thetaphi_a,
        y_lm,
    )


""""""" A CHECK TO SEE IF THE SH_deriv FUNCTION IS UNDERSTOOD """""""
""""""" AND PROVIDES THE SAME GRID AS THE INPUT GRID WITH y_lm """""""
(
    Y_lm_d1_theta_a,
    Y_lm_d1_phi_a,
    Y_lm_d2_theta_a,
    Y_lm_d2_phi_a,
    Y_lm_d2_thetaphi_a,
    y_lm,
) = SH_deriv(pi/2, pi, lmax)


# y_lm at theta = pi/2, phi = pi
D_check = D_clm.coeffs * y_lm

# The sum of all the terms in D_check should be equal to the value of D_array at 
# theta = pi/2 and phi = pi

# Array of nlat, nlon, mode, l, m
y_lm_arr = np.zeros(((2*lmax+2)+1, 2*(2*lmax+2)+1, 2, lmax+1, lmax+1))

# Array of nlat, nlon
D_lm_arr = np.zeros(((2*lmax+2)+1, 2*(2*lmax+2)+1))

for i, lat in enumerate(colats_rad):
    for j, lon in enumerate(lons_rad):
        # print(i, lat, j, lon)
        (Y_lm_d1_theta_a,
         Y_lm_d1_phi_a, 
         Y_lm_d2_theta_a, 
         Y_lm_d2_phi_a, 
         Y_lm_d2_thetaphi_a,
         y_lm) = SH_deriv(lat, lon, lmax)
        
        y_lm_arr[i,j] = y_lm
        D_lm_arr[i,j] = np.sum(y_lm * D_clm.coeffs)
        
        
print(np.sum(y_lm_arr[4,9] * D_clm.coeffs))
print(D_array[4,9])

D_lm_arr_grid = pysh.SHGrid.from_array(D_lm_arr)
D_array_grid  = pysh.SHGrid.from_array(D_array) 

fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
D_lm_arr_grid.plot(ax=ax1,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              cmap_limits=[3.6e25, 2.6e25],
              cb_label= 'SH calculated D map'
              )
D_array_grid.plot(ax=ax2,
              cmap=mycmap,
              colorbar='right',
              # cb_tick_interval=2,
              cmap_limits=[3.6e25, 2.6e25],
              cb_label= 'Input D map'
              )



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#########################################################
### BUILDING OPERATOR A USING SH FUNCTION DERIVATIVES ###
#########################################################

# # Define geometric operators
# def CSC2(theta): return 1/(np.sin(theta)**2)
# def COT(theta): return 1/np.tan(theta)

# # Defining some operators for simplification of expressions
# L1 = Y_lm_d2_theta_a + y_lm
# L2 = CSC2(theta) * Y_lm_d2_phi_a + COT(theta) * Y_lm_d1_theta_a + y_lm
# L3 = Y_lm_d2_thetaphi_a - COT(theta) * Y_lm_d1_phi_a

# # Defining the base term of operator A that is only dependent on 
# # theta, phi, and (derivatives of) y_lm
# A_base = L1 * L2 - CSC2(theta) * L3**2



# At each location (theta, phi) this operator A_base has to be calculated and
# multiplied with the coefficients 





