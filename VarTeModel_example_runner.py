# -*- coding: utf-8 -*-
"""
Example runner for the Beuthe (2008) variable-Te model.

Given the observed topography and geoid, this example solves for the flexure w 
and the crustal root dc that together explain the geoid. 

Different Te inputs can be given by the user, which can be a constant Te,
two-step Te based on a certain input file, or a specific Te map from a data file.
"""

import os
import numpy as np
import pandas as pd
import pyshtools as pysh
from VarTeModel import VarTeModel
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
from palettable import scientific as scm
from Te_twoval_builder import (build_Te_indicator, Te_clm_from_indicator)

# ============================== INPUTS ===================================
lmax   = 20
T_c    = 40.0e3        # reference crustal thickness [m]; dc varies about it
rho_l  = 2900.0        # surface load density   [kg/m3]
rho_c  = 2900.0        # crust density          [kg/m3]
rho_m  = 3500.0        # mantle density         [kg/m3]
E      = 100.0e9       # Young's modulus        [Pa]
nu     = 0.25          # Poisson's ratio
Mt     = 0.0e3         # top of internal density anomaly depth 
Mb     = 100.0e3       # bottom of internal density anomaly depth 

solve_for = 'dc_lm'     # set which field is solved for, 'dc_lm' or 'drho_lm'

# ------------------------ elastic thickness input ------------------------
Te_source   = 'map'            # 'constant' | 'twoval' | 'map' 

# in case Te_source = 'constant'
Te_constant = 100.0e3          # [m], used when Te_source = 'constant'

# in case Te_source = 'twoval', set the low and high value
Te_twoval_big = 100.0e3        # [m] 
Te_twoval_sml = 50.0e3         # [m] 

# in case Te_source = 'map'
Te_folder   = 'Elastic_Thickness_Input_Maps'
Te_filename = 'grl58258-sup-0002-data_set_1.dat'   # From Plesa et al. (2018)
Te_column   = 4                # 0-based: column 4 = strain 1e-14, 5 = 1e-17


# ------------------ downward-continuation filter -------------------------
filter_type = 'Ma'             # None | 'Ma' minimum amplitude | 'Mc' curvature
filter_half = 50               # degree at which the filter equals 0.5

# ------------------------ numerics ---------------------------------------
iterate   = True
nmax      = 5        # 1 = mass sheet (linear, one solve); > 1 = finite amplitude
delta_max = 1e-8     # convergence threshold [m] on the tracked Moho relief
iter_max  = 300      # Maximum number of iterations

damp      = True     # None = on for dc_lm (recommended); True/False to force

grid_res  = 3*lmax   # grid bandwidth for the spatial products (>= 2*lmax)

# =========================================================================


# --------------------------- load the data -------------------------------
pot       = pysh.datasets.Mars.GMM3(lmax=lmax)
topo_clm  = pysh.datasets.Mars.MOLA_shape(lmax=lmax)
R         = topo_clm.coeffs[0, 0, 0]
pot       = pot.change_ref(r0=R)
geoid_clm = pot*R
mass      = pot.gm / pysh.constants.G.value
g0        = pot.gm / R**2

# Set C20 of topo and geoid to {percent_C20}% of its original value'
percent_C20 = 0.0      # scale C20 of topo and geoid
topo_clm.coeffs[0, 2, 0]  *= percent_C20 / 100.0
geoid_clm.coeffs[0, 2, 0] *= percent_C20 / 100.0


def load_Te_map(lmax, folder, filename, column):
    """Least-squares SH fit of a lon/lat/Te table. Te is read in km."""
    raw = pd.read_csv(os.path.join(folder, filename), sep=r'\s+',
                      comment='#', header=None).values
    n_first = len(np.unique(raw[:, 0]))
    if n_first == 360:
        lon_i, lat_i = 0, 1
    elif n_first == 180:
        lon_i, lat_i = 1, 0
    else:
        raise ValueError(f'{filename}: first column has {n_first} unique values, '
                         'expected 360 (longitude) or 180 (latitude)')
    Te = raw[:, column] * 1e3        # km -> m
    print(f'  {filename}: mean Te = {np.mean(Te)/1e3:.2f} km')
    return pysh.SHCoeffs.from_least_squares(Te, raw[:, lat_i], raw[:, lon_i],
                                            lmax=lmax)

# Build the Te coefficients depending on input source
if Te_source == 'constant':
    Te_clm = pysh.SHCoeffs.from_zeros(lmax)
    Te_clm.coeffs[0, 0, 0] = Te_constant
    print(f'Constant Te = {Te_constant/1e3:.2f} km')
elif Te_source == 'twoval':    
    Te_src = load_Te_map(lmax, Te_folder, 
                         Te_filename, 
                         Te_column)
    binary_mask, meta = build_Te_indicator(Te_src, 
                                 grid_res, lmax,
                                 lat_range=(-90., 90.), 
                                 lon_range=(0., 360.),
                                 sigma_deg=2.0, 
                                 mask_val=75.0e3, 
                                 largest_blob=True, 
                                 fill_holes=True)
    # Print cos-weighted area fraction of the small-Te region
    print(meta['area_frac_sharp'])  
    
    Te_clm, _ = Te_clm_from_indicator(binary_mask, Te_twoval_big, Te_twoval_sml,
                                     lmax)
elif Te_source == 'map':
    Te_clm = load_Te_map(lmax, Te_folder, Te_filename, Te_column)



# Initialize the model
model = VarTeModel(
    R=R, g0=g0, mass=mass, lmax=lmax, solve_for=solve_for,
    T_c=T_c, rho_l=rho_l, rho_c=rho_c, rho_m=rho_m, E=E, nu=nu, Mt=Mt, Mb=Mb,
    grid_expansion_res=grid_res, nmax=nmax, 
    delta_max=delta_max,
    iterate=iterate, iter_max=iter_max, damp=damp,
    filter_type=filter_type, filter_half=filter_half)

# Run the model
output = model.solve(topo_clm, geoid_clm, Te_clm)

# Compute principal stresses (compression positive)
(   min_stress,
    max_stress,
    sum_stress,
    principal_angle_stress,
) = model.compute_stress()


# Plot the Te map, flexure, crustal root variations, 
# Make Te coeffs into grid and plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10, 8), dpi=100)
args_plot = dict(tick_interval=[60,45], colorbar='bottom')
args_expand = dict(lmax=grid_res, lmax_calc=lmax)

Te_grid = Te_clm.expand(**args_expand)/1e3
Te_grid.plot(ax=ax1, cmap=cmc.lajolla_r, 
             cb_label='Elastic thickness $T_e$ [km]',
             ticks='WSne',
             **args_plot)

grid_w = model.w_lm.expand(**args_expand)/1e3
grid_w.plot(ax=ax2, cmap=cmc.roma_r, 
            cb_label='Upward displacement $w$ [km]',
             ticks='wSnE',
             ylabel=None,
            **args_plot)

grid_dc = model.dc_lm.expand(**args_expand)/1e3
grid_dc.plot(ax=ax3, cmap=cmc.navia, 
             cb_label='Crustal root variations $\delta c$ [km]',
             ticks='WSne',
             **args_plot)

sum_stress_grid = pysh.SHGrid.from_array(sum_stress)
sum_stress_grid.plot(ax=ax4, cmap=scm.diverging.Vik_20.mpl_colormap, 
                     cb_label='Sum principal horizontal stress $\sigma$ [MPa]',
                     ticks='wSnE',
                     ylabel=None,
                     **args_plot) 

plt.tight_layout()
plt.show()


