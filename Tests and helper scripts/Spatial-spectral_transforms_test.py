# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:39:14 2026

@author: Timov
"""
import numpy as np
import pyshtools as pysh
import os
import time
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
nu      = 0.25
E       = 100.0e9
rho_l = 2900.0
rho_c = 2900.0 
rho_m = 3500.0
drho = rho_m - rho_c
drhol = rho_c - rho_l
T_c = 65e3                 # Arbitrary crustal thickness value, TBC


lmax  = 45        # last entry is the reference resolution
rotate_angles = (0.0, 0.0, 0.0)
lmax_Te_fit = 45
CACHE_DIR  = "gaunt_cache"
cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cm.davos
cmap3 = cm.roma_r
os.makedirs(CACHE_DIR, exist_ok=True)

omega_On = True
strain = 14      # Set which Te map is used, strain-14, strain-17, or
                 # strain-0 (returns constant Te map with Te=average of Te-14)


def load_Temap(lmax_Te_fit, strain=14):
    """
    Load in a Te map from a data file stored in the same directory as this 
    script. Return the LSQ-derived coefficients of the Te data array, using
    the data file's latitude and longitude values.
    
    Currently loads in the Plesa et al. (2018) data set 1 file,
    specifically the Te map from a strain rate of 1e-14 1/s. 
    A 1e-17 1/s strain rate Te map is also available
    """
    subfolder_Te_maps = "Elastic Thickness Input Maps"
    Te_filename = "grl58258-sup-0002-data_set_1.dat"
    Te_file_path = os.path.join(subfolder_Te_maps, Te_filename)
    df = pd.read_csv(Te_file_path, sep=r'\s+', comment='#',
                     header=None,
                     names=['longitude','latitude','crustal_thickness_km',
                            'heat_flow_mW_M3','Te_1e-14_km','Te_1e-17_km',
                            'T_150km_K','depth_1370km_km'],
                     usecols=['longitude','latitude','Te_1e-14_km','Te_1e-17_km'])
    Te_14 = df['Te_1e-14_km'].values*1e3
    Te_17 = df['Te_1e-17_km'].values*1e3
    
    print(f'Computing Te SHCoeffs from Te map up to lmax={lmax_Te_fit}')
    if strain == 14:
        print(f'For input strain=14, Te_mean = {np.mean(Te_14)/1e3:.2f} km')
        return pysh.SHCoeffs.from_least_squares(Te_14, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_14
    elif strain == 17:
        print(f'For input strain=17, Te_mean = {np.mean(Te_17)/1e3:.2f} km')
        return pysh.SHCoeffs.from_least_squares(Te_17, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_17
    elif strain == 0:
        Te_constant = np.mean(Te_14)
        print(f'For input strain=0, Te_mean = {np.mean(Te_14)/1e3:.2f} km')
        Te_constant_array = Te_constant * np.ones([64800])
        return pysh.SHCoeffs.from_least_squares(Te_constant_array, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_17
        # return pysh.SHGrid.from_array(np.mean(Te_14) * np.ones(
        #     [2*(lmax_Te_fit+1)+1, 4*(lmax_Te_fit+1)+1])).expand(), Te_constant
    else:
        print('ERROR: Input strain rate of 1e-{strain} 1/s is not available'
              ' for data files of Plesa et al. (2018). Please select strain'
              ' rate exponent of 14 or 17, or change the data file manually.')
        
def load_inputs(lmax, strain=14):
    """
    Load in the GMM3 potential and MOLA topography up to lmax. Use these to 
    obtain mean planetary radius R, geoid (pot*R) and g0. Also loads in Te map
    up to lmax_Te_fit (which )
    """
    pot  = pysh.datasets.Mars.GMM3(lmax=lmax)
    topo = pysh.datasets.Mars.MOLA_shape(lmax=lmax)
    R = topo.coeffs[0,0,0]
    pot = pot.change_ref(r0=R)
    geoid = pot*R
    gm = pot.gm; 
    g0 = gm/R**2
    G = pysh.constants.G.value  # Gravitational constant
    mass = gm / G  # Mass of the planet

    percent_C20 = 0.0
    print(f'\nSetting C20 of topo and geoid to {percent_C20}% of original value')
    topo.coeffs[0, 2, 0] = (percent_C20 / 100.0) * topo.coeffs[0, 2, 0]
    geoid.coeffs[0, 2, 0] = (percent_C20 / 100.0) * geoid.coeffs[0, 2, 0]
    
    # print(f'Loading Te map at lmax={lmax_Te_fit}')
    T_e_parent,_ = load_Temap(lmax_Te_fit, strain)
    # print('Te map loaded in')
    
    # # 1. Making a constant T_e map
    # T_e_parent_mean = T_e_parent.coeffs[0,0,0]
    # print(f'Constant Te map as input, Te={T_e_parent_mean/1e3} km (mean of Plesa Te map')
    # T_e_array = T_e_parent_mean * np.ones([2*(lmax_Te_fit+1)+1, 4*(lmax_Te_fit+1)+1])
    # T_e_parent = pysh.SHGrid.from_array(T_e_array).expand()
    
    return topo, geoid, T_e_parent, R, g0, mass

def truncate(clm, lmax):
    """
    Truncate any SHCoeffs object coefficients up to degree lmax.
    """
    return pysh.SHCoeffs.from_array(clm.coeffs[:, :lmax+1, :lmax+1].copy(),
                                    normalization='4pi')



topo, geoid, T_e_parent, R, g0, mass = load_inputs(lmax, strain)

topo_clm  = truncate(topo,  lmax)
geoid_clm = truncate(geoid, lmax)

T_e_use, topo_use, geoid_use = T_e_parent, topo_clm, geoid_clm

T_e_grid   = T_e_use.expand(lmax=lmax)
T_e_grid.data = T_e_grid.data
topo_grid  = topo_use.expand(lmax=lmax) - R
geoid_grid = geoid_use.expand(lmax=lmax) - R

lap_by_degree = np.array([-l * (l + 1) for l in range(2 * lmax + 1)])
Te_new = T_e_grid
Te_spec = T_e_use
for iteration in range(100):
    Te_new *= T_e_grid
    Te_error = T_e_use.expand().data - Te_new.data
    print(f'geoid error at iter:{iteration} = ', Te_error)
    
    
    Te_spat = Te_spec.expand()
    Te_spec = Te_spat.expand()
    for l in range(Te_spec.coeffs.shape[1]):
        Te_spec.coeffs[:, l, :] *= lap_by_degree[l]

for l in range(T_e_use.coeffs.shape[1]):
    T_e_use.coeffs[:, l, :] *= lap_by_degree[l]

Te_spec_res = T_e_use.copy()
Te_spec_res.coeffs = T_e_use.coeffs - Te_spec.coeffs



fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,11))
T_e_use.expand().plot(ax=ax1, colorbar='right')
ax1.set_title('original Te^3')

Te_spec.expand().plot(ax=ax2, colorbar='right')
ax2.set_title(f'iterated Te^3 expanded {iteration} times')

Te_spec_res.expand().plot(ax=ax3, colorbar='right')
ax3.set_title(f'residual after {iteration} iterations')
plt.tight_layout()
plt.show()