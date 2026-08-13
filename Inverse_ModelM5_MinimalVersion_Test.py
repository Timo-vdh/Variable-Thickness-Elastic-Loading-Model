# -*- coding: utf-8 -*-
"""
Beuthe (2008) variable-thickness flexure solver — Model 5 (M5)

Model for the variable thickness deformations of a thin elastic spherical shell,
including consoidal term of the tangential loading (the surface gradient of 
a scalar potential Omega).
Current model (M5) works with:
    - Beuthe (2008)'s equations 75 and 76 for the vertical displacement w and the 
      stress function F. 
    - Banerdt (1986)/Broquet & Andrews-Hanna (2023) equation for tangential
      loading potential Omega (with zero dc and zero drho).
    - Geoid self-consistency solving
    - Crustal root variations
    - Internal density variations
    - Iterations for redistributions due to internal density variations
    - Iterations for finite amplitude corrections
    
Model 5 does not include:
    - Toroidal loading (V=0 & T=0)

Following Beuthe's model requires implementation of the differential operator 
A(a;b). Beuthe (2008) does not give a spectral method for this, but in Beuthe
(2010) this spectral notation is made. Kalousova et al. (2012) describe the 
system of equations 75 and 76 in full spectral notation. This system of
equations is solved in Model 1, and extended here for the inclusion of 
tangential loading potential Omega_lm, crustal root variations dc_lm and 
internal density variations drho_lm.

This model includes the thin-shell approximation factor eta that Beuthe and
Kalousova neglect in their final equations (Beuthe does include it in 
equations 58 and 66). 

"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import pyshtools as pysh
import os
import time
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm as cmc
import pandas as pd
import sys
sys.path.insert(1, 'C:/Users/Timov/Displacement_strain_planet/Displacement_strain_planet')
from Displacement_strain_planet import corr_nmax_drho, DownContFilter, Plt_tecto_Mars
from pyshtools.expand import MakeGridDH, SHExpandDH
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from Gaunt_utils_v5 import (build_or_load_gaunt, selftest_gaunt,
                         selftest_term_weights, mode_degrees,
                         build_gidx, cell_sums, cell_sums_AB,
                         cell_sums_eta_dL,
                         build_conv_matrix)

# %% INPUTS

nu    = 0.25
E     = 100.0e9
rho_l = 2900.0
rho_c = 2900.0 
rho_m = 3500.0
drho  = rho_m - rho_c
drhol = rho_c - rho_l
T_c   = 60.0e3             # Arbitrary crustal thickness value, TBC
Te_input = 100.0e3

# Top and bottom depth of density variations drho_lm
Mt = 0.0e3
Mb = T_c

# lmax settings
LMAX_RUNS  = [45]        # last entry is the reference resolution
LMAX_REF = max(LMAX_RUNS)
grid_expansion_res = LMAX_REF * 3

rotate_angles = (0.0, 0.0, 0.0)
lmax_Te_fit = 60

# Set which Te map is used, strain-14, strain-17, or strain-0
# (0 returns constant Te map with Te=Te_input)
strain = 0

# Select whether solving for crustal root or internal density variations
# solve_for = 'drho_lm'
solve_for = 'dc_lm'


# Make custom Te map that is based on two values (for now only Tharsis map)
strain = 'twoval'
Te_twoval_name = 'Te_twoval_tharsis'
Te_twoval_big = 130e3
Te_twoval_sml = 70e3


# ========================== iteration config =============================
nmax       = 5
iterate    = True
delta_max  = 1e-6      # convergence threshold on the tracked grid
delta_out  = 1e12      # divergence threshold
iter_max   = 300

damp = (False if solve_for=='drho_lm' else True)

# DOWNWARD-CONTINUATION FILTER (Wieczorek & Phillips 1998)
# filter_type : 'Ma' minimum amplitude, 'Mc' minimum curvature, None off
# filter_half : the degree at which the filter equals 0.5
filter_on = True
filter_type = ("Ma" if filter_on else None)
filter_half = (50 if filter_on else None)
# ==========================================================================

# Set some colormaps used in plotting
cmap1 = scm.diverging.Vik_20.mpl_colormap
cmap2 = cmc.broc
cmap3 = cmc.roma_r

# Set saving & loading directories for Gaunt and plots
CACHE_DIR  = "gaunt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SaveFigs = False
SavePath = "Plots/M5VarD_SPEC_FinalPlots"          # If on own laptop
# SavePath = "/home/vand_t1/Documents/Figures_M5"  # If on DLR computer
os.makedirs(SavePath, exist_ok=True)


# %% LOAD IN DSP SOLUTIONS

# Load in DSP results
w_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_w_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')
dc_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_dc_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')
drho_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_drho_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')
Tc_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_Tc_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')
w_coeffs_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_w_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')

Omega_lm_DSP = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_Om_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')
A_lm = pysh.SHCoeffs.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_A_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')

sum_strain_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_strn_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')
princ_angle_DSP = pysh.SHGrid.from_file(f'DSP_result_files/DSP_Slv{solve_for}_it{iterate}_n{nmax}_ft{filter_type}_fh{filter_half}_angle_l{LMAX_REF}_Tc{T_c}_Te{Te_input}_pl{rho_l}_pc{rho_c}_pm{rho_m}_Mt{Mt/1e3}_Mb{Mb/1e3}_dmax{delta_max}_lgrid{grid_expansion_res}')



# %% TWO-VALUE TE MAP BUILDER & PLOTTER

if strain == 'twoval':
    from Te_twoval_builder import (build_Te_indicator, save_Te_indicator,
                                   load_Te_indicator, Te_clm_from_indicator, 
                                   plot_Te_map)
    
    # --- pick your w source (coeffs file, grid file, or an in-memory object) ---
    w_src = w_DSP
    
    f, meta = build_Te_indicator(w_src, grid_expansion_res, LMAX_REF,
                                 lat_range=(-50., 70.), lon_range=(210., 290.),
                                 sigma_deg=3.5, largest_blob=True, fill_holes=True)
    print(meta['area_frac_sharp'])                 # cos-weighted area fraction of the small-Te region
    save_Te_indicator(f, meta, 'Te_twoval_tharsis')
    plot_Te_map(f, Te_twoval_big, Te_twoval_sml, w_src, grid_expansion_res, LMAX_REF, lmax_fit=lmax_Te_fit)


# %% BASIC FUNCTION DEFINITIONS

def make_mode_map(lmax):
    """
    Flatten all combinations of l,m into a flat array based on input lmax.
    """
    return [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]

def truncate(clm, lmax):
    """
    Truncate any SHCoeffs object coefficients up to degree lmax.
    """
    return pysh.SHCoeffs.from_array(clm.coeffs[:, :lmax+1, :lmax+1].copy(),
                                    normalization='4pi')


# %% INPUT LOADERS

def load_Temap(lmax_Te_fit, strain=14):
    """
    Load in a Te map from a data file stored in the same directory as this 
    script. Return the LSQ-derived coefficients of the Te data array, using
    the data file's latitude and longitude values.
    
    Currently loads in the Plesa et al. (2018) data set 1 file,
    specifically the Te map from a strain rate of 1e-14 1/s. 
    A 1e-17 1/s strain rate Te map is also available
    """
    subfolder_Te_maps = "Elastic_Thickness_Input_Maps"
    Te_filename = "grl58258-sup-0002-data_set_1.dat"
    Te_file_path = os.path.join(subfolder_Te_maps, Te_filename)
    df = pd.read_csv(Te_file_path, sep=r'\s+', comment='#',
                     header=None,
                     names=['longitude','latitude','crustal_thickness_km',
                            'heat_flow_mW_M5','Te_1e-14_km','Te_1e-17_km',
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
        Te_constant_array = Te_input * np.ones([64800])
        return pysh.SHCoeffs.from_least_squares(Te_constant_array, df['latitude'].values, 
                                            df['longitude'].values, 
                                            lmax=lmax_Te_fit), Te_14
    elif strain == 'twoval':
        f, _ = load_Te_indicator(Te_twoval_name)
        clm, Te_grid = Te_clm_from_indicator(f, Te_twoval_big, Te_twoval_sml,
                                             lmax_Te_fit)
        return clm, Te_grid.ravel()
    else:
        print('ERROR: Input strain rate of 1e-{strain} 1/s is not available'
              ' for data files of Plesa et al. (2018). Please select strain'
              ' rate exponent of 0, 14 or 17, or change the data file manually.')
        
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
        
    print(f'Loading Te map at lmax={lmax_Te_fit}')
    T_e_parent,_ = load_Temap(lmax_Te_fit, strain)
    
    # Ensure all coefficients apart from 0,0 are truly zero in constant Te case
    if strain == 0:
        T_e_0 = T_e_parent.coeffs[0,0,0]
        T_e_parent.coeffs[:,:,:] = 0
        T_e_parent.coeffs[0,0,0] = T_e_0
    print('Te map loaded in')
    
    return topo, geoid, T_e_parent, R, g0, mass

def derive_D_a(T_e_parent, lmax):
    """
    Compute the flexural rigidity D and parameter alpha using the parent Te.
    Function first expands the parent Te map to a finer grid of 
    grid_expansion_res, which is then used to compute D and alpha coefficients. 
    D and alpha are then truncated to 2*lmax+1 because the coupling coefficients 
    contain degrees up to the sum of two input degrees (the sum over LM goes 
    from l-l' to l+l', i.e. 2*l).

    Extended to follow Beuthe eqs 58/66, unsimplified forms. 
    These include:
        - The eta-weighted fields (eta*D, eta*alpha) used by the A/B 
          operators 
        - The eta field itself used by the A(eta;F) and A(eta;w) coupling blocks.
    The products are formed on the spatial grids.
    
    The plain a_clm and eta_clm are still returned too. The Omega_RHS1 and 2 
    apply the bare alpha field for each term and perform the convolution
    with the eta_clm after the full RHS terms are built. S applies the bare a_clm.

    eta convention: eta = 1/(1 + Te^2/(12*Re^2)) with Re = R - T_e_0/2 built
    from the reference (mean) thickness, matching DSP's eps/beta/eta
    definitions.
    """

    grid = T_e_parent.expand(lmax=grid_expansion_res)
    print('Computing D and alpha using Te grid expanded to '
          'lmax=grid_expansion_res')
    Te0_loc = T_e_parent.coeffs[0,0,0]
    Re_loc  = R - Te0_loc/2
    if strain == 0:
        D = E*T_e_parent.coeffs[0,0,0]**3/(12*(1-nu**2))
        D_coef = grid.expand().copy()
        D_coef.coeffs[:,:,:] = 0
        D_coef.coeffs[0,0,0] = D
        D = pysh.SHCoeffs.from_array(D_coef.coeffs[:, :2*lmax+1, :2*lmax+1])
        
        a = 1.0/(E*T_e_parent.coeffs[0,0,0])
        a_coef = grid.expand().copy()
        a_coef.coeffs[:,:,:] = 0
        a_coef.coeffs[0,0,0] = a
        a = pysh.SHCoeffs.from_array(a_coef.coeffs[:, :2*lmax+1, :2*lmax+1])

        eta0_loc = 1.0/(1.0 + Te0_loc**2/(12.0*Re_loc**2))
        D_eta = pysh.SHCoeffs.from_array(D.coeffs.copy());  D_eta.coeffs *= eta0_loc
        a_eta = pysh.SHCoeffs.from_array(a.coeffs.copy());  a_eta.coeffs *= eta0_loc
        eta_coef = grid.expand().copy()
        eta_coef.coeffs[:,:,:] = 0
        eta_coef.coeffs[0,0,0] = eta0_loc
        eta_clm = pysh.SHCoeffs.from_array(eta_coef.coeffs[:, :2*lmax+1, :2*lmax+1])

    else:
        D = pysh.SHGrid.from_array(E*grid.data**3/(12*(1-nu**2))).expand()
        D = pysh.SHCoeffs.from_array(D.coeffs[:, :2*lmax+1, :2*lmax+1])
        a = pysh.SHGrid.from_array(1.0/(E*grid.data)).expand()
        a = pysh.SHCoeffs.from_array(a.coeffs[:, :2*lmax+1, :2*lmax+1])

        eta_grid = 1.0/(1.0 + grid.data**2/(12.0*Re_loc**2))
        D_eta = pysh.SHGrid.from_array(eta_grid*E*grid.data**3/(12*(1-nu**2))).expand()
        D_eta = pysh.SHCoeffs.from_array(D_eta.coeffs[:, :2*lmax+1, :2*lmax+1])
        a_eta = pysh.SHGrid.from_array(eta_grid/(E*grid.data)).expand()
        a_eta = pysh.SHCoeffs.from_array(a_eta.coeffs[:, :2*lmax+1, :2*lmax+1])
        eta_clm = pysh.SHGrid.from_array(eta_grid).expand()
        eta_clm = pysh.SHCoeffs.from_array(eta_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    print('D and alpha computed \n')
    return D, a, D_eta, a_eta, eta_clm

def rotate_inputs(rot_angles, T_e_parent, D_clm, a_clm, topo_clm, geoid_clm):
    """
    Rotate the input topography, geoid, Te map and the resulting D and alpha 
    maps using the user-input rotation angles based on the rotation convention
    as used by pyshtools.
    """
    print(f"Rotating inputs with angles {rot_angles}...")
    alpha, beta, gamma = rot_angles
    T_e_parent = T_e_parent.rotate(alpha, beta, gamma)
    D_clm = D_clm.rotate(alpha, beta, gamma)
    a_clm = a_clm.rotate(alpha, beta, gamma)
    topo_clm = topo_clm.rotate(alpha, beta, gamma)
    geoid_clm = geoid_clm.rotate(alpha, beta, gamma)

    return T_e_parent, D_clm, a_clm, topo_clm, geoid_clm

# %% drho_lm HELPER FUNCTION

def drho_layer(lmax, R, g0, mass):
    """
    Helper function used for precalculating recurring constants and 
    degree-dependent arrays used for the solve_for=drho_lm model runs. 
    Some functions and naming:

      - rhobar: mean density of the planet
      - M     : layer thickness (Mb - Mt)
      - g_M   : gravity at mid-depth of the density anomaly layer (DSP's `gdrho`)
      - Cp    : 3 / (rhobar * (2l+1))
      - B_1   : Cp * R/(l+3) * [(Rt/R)^(l+3) - (Rb/R)^(l+3)]
      - B_2   : Cp * R/(l+3) * [RtRCl - RbRCl]
               (a (g0/g_m) prefactor is applied OUTSIDE, by the caller, exactly 
               as in DSP eq (2) )
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    M      = Mb - Mt
    R_top  = R - Mt
    R_base = R - Mb
    R_c    = R - T_c
    R_mid  = (R_top + R_base) / 2.0
    rho_d  = rho_c if Mt <= T_c else rho_m
    g_M    = g0 * (1.0 + ((R_mid/R)**3 - 1.0) * rho_d / rhobar) / (R_mid/R)**2

    degs = np.arange(2*lmax + 1, dtype=float)
    Cp   = 3.0 / (rhobar * (2.0*degs + 1.0))
    Rl3  = R / (degs + 3.0)

    # [(R-Mt) / (R-Tc)] ^ l
    RtRCl = ((R_top/R_c)**degs if R_top <= R_c else (R_c/R_top)**(degs + 1.0))
    # [(R-Mb) / (R-Tc)] ^ l
    RbRCl = ((R_base/R_c)**degs if R_base <= R_c else (R_c/R_base)**(degs + 1.0))
    
    # {[(R-Mt) / (R-Tc)] ^ l} * {(R-Mt)^3 / [(R-Tc) * R^2]}
    RtRCl = RtRCl * R_top**3  / (R_c * R**2)
    # {[(R-Mb) / (R-Tc)] ^ l} * {(R-Mb)^3 / [(R-Tc) * R^2]}
    RbRCl = RbRCl * R_base**3 / (R_c * R**2)
    
    B_1 = Cp * Rl3 * ((R_top/R)**(degs + 3.0) - (R_base/R)**(degs + 3.0))
    B_2 = Cp * Rl3 * (RtRCl - RbRCl)

    # -------------------- DOWNWARD-CONTINUATION FILTER ---------------------
    # B_1 and B_2 are the drhom coefficients of eqs (1) and (2):
    #     eq (1): rhobconst * drhom_lm * Rl3 * RtbRl3   / DCfilter_drhom
    #     eq (2): rhobconst * drhom_lm * Rl3 * (RtRCl-RbRCl) / DCfilter_drhomc
    # (thinshell.py 1087 and 1111), and rhobconst == Cp. So dividing here
    # reaches EVERY consumer of drho_layer at once -- q_lm, Omega_eq1_RHS,
    # Omega_eq2_RHS and the omega assembly -- with no edit to any of them.
    # That matters: the OMEGA-TERMS cell is the reference against which
    # omega_corr_to_rhs's prefactors were derived, and it must stay untouched.
    #
    # The remaining consumer, compute_drho, rebuilds B_1 locally and is
    # filtered separately. If you ever unify the two, filter in one place.
    _, _, DCfilter_drhom, DCfilter_drhomc = DCfilters(len(degs) - 1)
    B_1 = B_1 / DCfilter_drhom
    B_2 = B_2 / DCfilter_drhomc

    return dict(M=M, g_M=g_M, B_1=B_1, B_2=B_2, Cp=Cp, Rl3=Rl3, rhobar=rhobar)


# %% FINITE AMPLITUDE (FA) CORRECTION FUNCTIONS

_GRID_CACHE = {}

def _cached_grid(key, lmax, build):
    """
    The topography, geoid, and Te fields are cached for every (lmax, rotation)
    run, such that they do not get recomputed every iteration, since they don't
    change. 
    
    Any caller that changes topo, the geoid or Te MUST call
    `_GRID_CACHE.clear()` first. Two places do:
      - each (lmax_run, rotation) pass of the main loop (the rotated run
        feeds rotated topo/Te, and without the clear it would silently reuse
        the unrotated grids)
      - every self-test, which feeds synthetic fields.
    """
    k = (key, lmax, grid_expansion_res)
    if k not in _GRID_CACHE:
        _GRID_CACHE[k] = build()
    return _GRID_CACHE[k]

def interface_geoid_correction(interface, R, mass, lmax, lmaxgrid, nmax,
                               topo_clm=None, w_clm=None, dc_clm=None,
                               drho_clm=None):
    """
    The geoid correction for any lithospheric interface, covering four cases:
        - drho_clm is None, nmax=1  ->  pure mass sheet with no density 
                                          variations, correction = 0
                                          (drho = 1.0, rho_grid = ones)
        - drho_clm is None, nmax >1 ->  pure finite amplitude correction 
                                          (drho = 1.0, rho_grid = ones)
        - drho_clm given, nmax=1    ->  pure lateral density correction
        - drho_clm given, nmax >1   ->  finite amplitude + lateral density

    The function is intended for the three possible density interfaces:
    | interface | relief dr_lm | degree 0 |
    |-----------|--------------|----------|
    | 'H'       | topo         | R        |
    | 'w'       | w            | R        |
    | 'wdc'     | w - dc       | R - T_c  |

    The degree-0 values are not cosmetic: `corr_nmax_drho` feeds `shape_grid`
    to `CilmPlusRhoHDH`, which takes its own mean radius D from that grid. Get
    the degree-0 term wrong and D is wrong.

    Each density interface (/relief) has its own specific density corrections:
    | interface | drho scalar     | rho_grid passed                  |
    |-----------|-----------------|----------------------------------|
    | 'H'       | rho_l           | drho_lm + rho_l                  |
    | 'w'       | rho_c - rho_l,  | -drho_lm                         |
    |           | or 1 if equal   |                                  |
    | 'wdc'     | rho_m - rho_c   | rho_m - (drho_lm + rho_c)        |
    |           |                 |   if Mb <= T_c (anomaly in crust)|
    |           |                 | (drho_lm + rho_m) - rho_c        |
    |           |                 |   otherwise   (anomaly in mantle)|

    In the pure finite-amplitude path both sides of the substraction are at 
    unit density (drho = 1.0 & rho_grid = ones), so the result is per unit 
    density with nothing to undo, and `corr_nmax_drho` performs no division. 
    
    In the lateral density path both sides carry real densities, and the 
    division by `drho_Thinshell` puts the result back on the same per-unit 
    footing. 
    Either way, at the point where the correction is added, the real contrast 
    is multiplied back in, which is what `geoid_corrections` does. 

    The density interface is determined by (rho_grid - drho), the departure of 
    the local density from the scalar the thin-shell equation already assumes:
        'H'          : drho_lm + (rho_l - rho_l) = drho_lm
        'w'          : -drho_lm - 1     (the constant is 1 kg/m3, negligible)
        'wdc' mantle : (drho_lm + rho_m - rho_c) - (rho_m - rho_c) = drho_lm

    Degree 0 of the correction output has to be zeroed explicitly here, even
    though this does not happen in DSP. Not an issue there because DSP iterates
    from l=1 to lmax, while M5 starts from l=0. Inside DSP's `corr_nmax_drho` 
    the mass-sheet side uses dr_lm  with absolute radii, while CilmPlusRhoHDH 
    works internally with h = r - D. Identical for l >= 1, but at l = 0 they 
    differ by the whole of R, giving a spurious term of order -2600 m.

    Returns metres of geoid, per unit of the density contrast depending on the 
    interface.
    """
    args_grid = dict(lmax=lmaxgrid, norm=1, sampling=2, extend=False)

    # -------------------- the interface reliefs ------------------------
    # Set the relief interface in coeffs, ensure correct degree-0 and make grid
    if interface == 'H':
        dr_lm = truncate(topo_clm, lmax).coeffs.copy()
        dr_lm[0, 0, 0] = R
        # topo is fixed within a pass, so its grid is worth caching. w and
        # w - dc change every iteration and are not.
        shape_grid = _cached_grid('shapeH', lmax,
                                  lambda: MakeGridDH(dr_lm, **args_grid))
        # Use lambda here to only trigger the MakeGridDH if it is not in the cache.
        # Saves time, because now it is not recomputed every iteration.

    elif interface == 'w':
        dr_lm = truncate(w_clm, lmax).coeffs.copy()
        dr_lm[0, 0, 0] = R
        shape_grid = MakeGridDH(dr_lm, **args_grid)
    elif interface == 'wdc':
        dr_lm = (truncate(w_clm, lmax).coeffs - truncate(dc_clm, lmax).coeffs)
        dr_lm[0, 0, 0] = R - T_c
        shape_grid = MakeGridDH(dr_lm, **args_grid)
    else:
        raise ValueError(f"interface must be 'H', 'w' or 'wdc', "
                         f"got {interface!r}")

    # ---- densities: unit for pure FA, real for the density path --------
    if drho_clm is None:
        # No lateral density variations, so unit density calculations
        drho_scalar = 1.0
        rho_grid    = np.ones_like(shape_grid)
    else:
        # Make grid of the lateral density variations
        drho_grid = MakeGridDH(truncate(drho_clm, lmax).coeffs, **args_grid)
        # Set the scalar rho and rho_grid for each interface
        if interface == 'H':
            drho_scalar = rho_l
            rho_grid    = drho_grid + rho_l
        elif interface == 'w':
            drho_scalar = (rho_c - rho_l) if rho_c != rho_l else 1.0
            rho_grid    = -drho_grid
        else:
            drho_scalar = drho
            rho_grid    = (rho_m - (drho_grid + rho_c) if Mb <= T_c
                           else (drho_grid + rho_m) - rho_c)

    # Calculate the SH coefficients of the FA-MS correction of this interface
    # using DSP's corr_nmax_drho function
    SH_correction = corr_nmax_drho(dr_lm=dr_lm, drho=drho_scalar, shape_grid=shape_grid,
                         rho_grid=rho_grid, lmax=lmax, mass=mass, nmax=nmax,
                         R=R, drho_Thinshell=drho_scalar,
                         density_var=drho_clm is not None)
    SH_correction[:, 0, 0] = 0.0
    
    return SH_correction


def geoid_corrections(H_corr, w_corr, wdc_corr, R, lmax):
    """
    Assemble the G_lm and Gc_lm corrections from all three interface
    reliefs. Transcribed from DSP:

      G_lm eq:   rhol*H_corr + drhol*w_corr + drho*wdc_corr*RCRl2
      Gc_lm eq: (rhol*H_corr + drhol*w_corr)*RCRl + drho*wdc_corr*RCR

    with RCRl = phi**l and RCRl2 = phi**(l+2), phi = (R - T_c)/R.
    Note DSP eq (2) carries the moho term with a single power of phi, not phi**3.
    Following DSP here, unsure which is the correct formulation.

    Degree 1 of corrGc is zeroed: DSP multiplies eq (2) by zero completely at
    l = 1 when Gc is an unknown, corrections included. DSP's eq (1)'s guard
    tests G_lm, which is an input here, so the G_lm corrections stay live at 
    degree 1.
    (Keep in mind, M5 is only written for the specific inputs of H_lm and
    G_lm. G_lm cannot be used as an output here yet.)
    """
    phi     = (R - T_c) / R
    
    # Here the interface corrections (defined per unit density) are multiplied
    # with the scalar densities of each corresponding interface.
    surface = rho_l * H_corr[:, :lmax+1, :lmax+1] 
    flexure = drhol * w_corr[:, :lmax+1, :lmax+1]
    moho    = drho  * wdc_corr[:, :lmax+1, :lmax+1]
    
    # Initialize the correction terms as SH coeffs
    corrG   = np.zeros_like(surface)
    corrGc  = np.zeros_like(surface)
    
    # Fill the correction coeffs (if an interface is inactive, the corr above
    # is zero anyways)
    for l in range(lmax + 1):
        corrG[:, l, :]  = (surface[:, l, :] 
                           + flexure[:, l, :] 
                           + moho[:, l, :] * phi**(l + 2) )
        corrGc[:, l, :] = (surface[:, l, :] * phi**l
                           + flexure[:, l, :] * phi**l
                           + moho[:, l, :] * phi )

    # Set the degree-1 term of the Gc correction to 0 explicitly
    corrGc[:, 1, :] = 0.0
        
    return corrG, corrGc


def geoid_mass_sheet(geoid_clm, corrG, lmax):
    """
    Fold the G_lm correction into the geoid that is handed to the solver.

    q_lm, Omega_eq1_RHS and Omega_eq2_RHS each invert DSP's eq (1) in closed 
    form to remove the interface unknown. With the correction, eq (1) says
        G = MassSheet + corrG
    so the mass-sheet part those three functions need is G - corrG. Substituting
    it once in solve_beuthe reaches all three without editing any of them.
    
    Degree 0 is preserved: the file relies on G[0,0,0] = R throughout (the
    Omega builders form the undulation as `geoid_clm.expand().data - R`).
    """
    correction_G = corrG.copy()
    correction_G[:, 0, 0] = 0.0
    geoid_MS = truncate(geoid_clm, lmax).copy()
    geoid_MS.coeffs -= correction_G
    
    return geoid_MS


def q_correction_from_geoid(corrG, corrGc, R, g0, mass):
    """
    The two repairs that handing geoid_MS to q_lm makes necessary.

    q_lm builds DSP eq (3), which contains
        - g0*rho_l*G      with the observed geoid (FA), not geoid_MS. 
                          Passing geoid_MS injects a spurious +g0*rho_l*corrG.
                          The first term cancels it.
        - g_m*drho*Gc     with Gc eliminated through the uncorrected  eq (2), so
                          the second term supplies what q_lm cannot know.
    Returns a physical load correction in Pa.
    """
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    phi    = (R - T_c) / R
    g_m    = g0 * (1.0 + (phi**3 - 1.0) * rho_c / rhobar) / phi**2
    
    return -g0*rho_l*corrG - g_m*drho*corrGc


# %% DOWNWARD CONTINUATION FILTERS FOR FINITE AMPLITUDE CORRECTIONS

def DCfilters(lmax):
    """
    Build the downward-continuation filters of Wieczorek & Phillips (1998), 
    using DSP's DownContFilter function.

    There are four filters possible, depending on whether its intended for the
    G_lm or the Gc_lm correction and on solving for dc_lm or drho_lm:
    
    if solve_for == 'dc_lm':
        DCfilter_mohoD  = DownContFilter(*param_filt, R,    R-Tc, **kw_filt)
        DCfilter_mohoDc = DownContFilter(*param_filt, R-Tc, R-Tc, **kw_filt)
    else:
        DCfilter_drhom  = DownContFilter(*param_filt, R,    R-Mb, **kw_filt)
        DCfilter_drhomc = DownContFilter(*param_filt, R-Tc, R-Mb, **kw_filt)

    Within a pair the two entries differ only in `R_ref`: G_lm eq is written at
    the surface R, Gc_lm eq at the moho R-Tc. `D_relief` is the depth being
    continued to: moho boundary for the dc-terms, bottom of the density anomaly
    for the drho-terms.
    
    The filters are all returned at once, with each an array of degrees 0 to lmax.

    Two special filter cases, both handled inside DownContFilter itself:
      - DCfilter_mohoDc has R_ref == D_relief, so (R_ref/D_relief)**l = 1
        Still a filter, just without the growth with degree.
      - If base_drho < c, DCfilter_drhomc has D_relief > R_ref, i.e. the density 
        anomaly base lies above the moho. In this case DSP's DownContFilter 
        returns ones, as the filter is upward there. 
    The second case does not need filtering because (R_ref/D_relief)**l with 
    D_relief > R_ref will always be <1, i.e. attenuating and thus filtering itself.


    HOW THEY ARE APPLIED. DSP divides the interface term of eqs (1) and (2) by
    them (thinshell.py 1086-1087, 1105-1111). Because DownContFilter returns
    1/(1 + ...) <= 1, dividing in the FORWARD equation means the RECOVERED
    interface is MULTIPLIED by the filter -- damped at high degree, which is
    the point of downward-continuation stabilisation.

    In this file eqs (1) and (2) have already been inverted to eliminate the
    interface unknown, so the divisions appear as:
        every factor 1/P2   (P2 = phi**(l+2))          ->  * DCfilter_mohoD
        every factor phi/P2 (i.e. phi**-(l+1))         ->  * (DCfilter_mohoD
                                                            / DCfilter_mohoDc)
        every factor phi**l (the eq-(2) surface term)  ->  unchanged
        B_1                                            ->  / DCfilter_drhom
        B_2                                            ->  / DCfilter_drhomc
    verified symbolically against this file's own unfiltered coefficients.
    The finite-amplitude and lateral-density correction terms are not
    filtered, matching DSP, which comments those divisions out at 1091/1118.
    """
    
    degrees = np.arange(lmax + 1, dtype=float)
    ones = np.ones_like(degrees)
    if filter_type is None:
        return ones, ones, ones, ones

    param_filt = (degrees, filter_half)
    kw_filt = dict(filter_type=filter_type, quiet=False)
    R_c = R - T_c
    R_base_drho = R - Mb

    DCfilter_mohoD = DCfilter_mohoDc = DCfilter_drhom = DCfilter_drhomc = ones
    if solve_for == 'dc_lm':
        # The two filters for the crustal root variations, first one for G_lm
        # eq and second one for the Gc_lm eq
        DCfilter_mohoD  = DownContFilter(*param_filt, R, R_c, **kw_filt)
        DCfilter_mohoDc = DownContFilter(*param_filt, R_c, R_c, **kw_filt)
    else:
        # The two filters for the crustal root variations, first one for G_lm
        # eq and second one for the Gc_lm eq 
        DCfilter_drhom  = DownContFilter(*param_filt, R, R_base_drho, **kw_filt)
        DCfilter_drhomc = DownContFilter(*param_filt, R_c, R_base_drho, **kw_filt)

    return DCfilter_mohoD, DCfilter_mohoDc, DCfilter_drhom, DCfilter_drhomc


# %% LATERAL-DENSITY CORRECTIONS

def density_gates():
    """
    DSP's three density gates, transcribed (thinshell.py 1587-1601):
        density_var_H  : top_drho == 0
        density_var_w  : base_drho < c and top_drho == 0 and rhol == rhoc
        density_var_dc : c in (base_drho, top_drho)

    Which configurations isolate which interface:
        Mt = 0,   Mb < T_c,  rho_l != rho_c  -> H only
        Mt = T_c, Mb > T_c                   -> dc only
        Mt = 0,   Mb < T_c,  rho_l == rho_c  -> H and w together
        Mt = 0,   Mb = T_c or 0,             -> H and dc together
    w cannot be isolated: it requires top_drho == 0, which forces H on. And no
    configuration fires all three: w needs Mt == 0 and Mb < T_c, while dc then
    reduces to T_c == Mb, which contradicts Mb < T_c.
    """
    gate_H  = (Mt == 0)
    gate_dc = (T_c in (Mb, Mt))
    gate_w  = (Mb < T_c and Mt == 0 and rho_l == rho_c)
    return gate_H, gate_dc, gate_w


def drho_q_omega_correction(interface, drho_clm, topo_clm, geoid_clm, w_clm,
                            T_e_parent, R, lmax, g0):
    """
    The q and omega corrections for the drho_lm case.
    DSP's `drho_q_corr` and `drho_omega_corr` for one interface, in one place.

    The corrections differ per density interface:
        
    drho_q_corr for each interface (in Pa):
    | interface | drho_q_corr               |
    |-----------|---------------------------|
    | 'H'       | g0 * ( drho * (H-G) )_lm  | (GRID MULTIPLICATION)
    | 'w'       | g0 * ( -drho * w )_lm     | (GRID MULTIPLICATION)
    | 'dc'      | 0                         |

    drho_omega_corr for each interface (in Pa):
    | interface | drho_omega_corr                     |
    |-----------|-------------------------------------|
    | 'H'       | v1v * g0 * Te/R * ( drho * H )_lm   | (GRID MULTIPLICATION)
    | 'w'       | v1v * g0 * Te/R * ( -drho * w )_lm  | (GRID MULTIPLICATION)
    | 'dc'      | 0                                   |

    'drho' here is the pure density variations field drho_lm that is solved for.

    The sign of the drho-field is determined by a condition:
        `grid1 = -rho_grid_var if Mt==0 else rho_grid_var`. 
    - For w this is always a negative sign by selection criteria.
    - For dc it can be either one.
    - H does not have this condition, it is always positive.

    Returns (q_corr, omega_corr) as SH coeffs in Pa. Zero arrays for 'dc'.
    """
    args_grid = dict(sampling=2, lmax=grid_expansion_res, extend=False,
                lmax_calc=lmax)
    
    zero   = np.zeros((2, lmax+1, lmax+1))
    d_grid = MakeGridDH(truncate(drho_clm, lmax).coeffs, **args_grid)
    
    if interface == 'dc':
        # Always returns zero for now, because either drho_lm or dc_lm is set
        # to zero, since they can't be solve simultaneously (yet), and the 
        # field is the multiplication of drho_lm and dc_lm
        return zero, zero.copy()
    
    elif interface == 'H':
        # THE q FIELD
        # Build H-G
        HG_coeffs = (truncate(topo_clm, lmax).coeffs
              - truncate(geoid_clm, lmax).coeffs)
        HG_coeffs[0, 0, 0] = 0.0      # H and G share the same R, making sure that deg0 is zero then
        
        # H-field is constant, so cache the grid
        q_field = _cached_grid('HG', lmax, 
                               lambda: MakeGridDH(HG_coeffs, **args_grid))
        
        # THE OMEGA FIELD
        H_coeffs = truncate(topo_clm, lmax).coeffs.copy()
        H_coeffs[0, 0, 0] = 0
        
        # H-field is constant, so cache the grid
        om_field = _cached_grid('Hrelief', lmax, 
                                lambda: MakeGridDH(H_coeffs, **args_grid))
        
    elif interface == 'w':
        # THE q FIELD = THE OMEGA FIELD FOR w
        w_lm = truncate(w_clm, lmax).coeffs.copy()
        w_lm[0, 0, 0] = 0.0                   # RELIEF (thinshell.py 1872)
        q_field = om_field = MakeGridDH(w_lm, **args_grid)
        
        # Following DSP, negative drho_grid here for the w-interface
        d_grid  = -d_grid
        
    else:
        raise ValueError(f"interface must be 'H', 'w' or 'dc', got {interface!r}")

    Te_grid = _cached_grid('Te', lmax, 
                lambda: MakeGridDH( truncate(T_e_parent, lmax).coeffs, **args_grid))
    # Use lambda here to only trigger the MakeGridDH if it is not in the cache.
    # Saves time, because now it is not recomputed every iteration.

    q_corr  = g0 * SHExpandDH(d_grid*q_field, lmax_calc=lmax, sampling=2)
    om_corr = ((nu/(1.0 - nu)) * g0 / R 
               * SHExpandDH( (d_grid*Te_grid)*om_field, lmax_calc=lmax, sampling=2))

    return q_corr, om_corr



# %% OMEGA CORRECTION HOOK

def omega_corr_to_rhs(omega_corr_phys, T_e_parent, a_clm, lmax, R, T_e_0, Re):
    """
    Map the omega correction into additive terms on the two Omega right-hand 
    sides.

    The omega equation as from DSP is:
        omega_lm = <the H, w, dc, drhom content>  +  drho_omega_corr .
    M5 does not solve eq (5): it substitutes omega's content into the w and F
    eqs of Beuthe (2008). Because the omega equation is substituted and not 
    inverted *like the G_lm and Gc_lm equations), the correction is simply an
    added term to the equation. This is why the drho_omega_corr is entered here
    and not in the geoid_corrections.

    THE OPERATOR. Both RHS builders apply the SAME linear operator to whatever
    omega content they are given; only the field differs. Inverting the
    prefactors that Omega_eq1_RHS / Omega_eq2_RHS already carry on their
    known-good H terms gives, for an arbitrary omega,

        RHS1 += K1 * (Re/R) * R * [ -2*Re**3 * omega  +  (Re/12) * Te**2 * Lap(omega) ]
        RHS2 += -(1 - nu) * Re**2 * Delta'( alpha * omega )

    with K1 = 1/(E*Te0**3). This was verified symbolically against all of
    factor1a_omega, factor1b_omega and factor2a_omega by substituting the H-part
    of DSP eq (5),
        omega_H = v1v * rho_l * g0 * Te * H / R ,
    and confirming the three differences are identically zero -- see
    `selftest_omega_corr_rhs`, which repeats the check NUMERICALLY against the
    production builders themselves.

    Structure of calculation, in the order the operators demand (the trap that 
    sank the first drho attempt, handover section 2 of Omega_eq1_RHS's own 
    comments):
      * Te**2 multiplies the laplacian of omega, so Lap acts first, on the
        omega coefficients, and the Te**2 product is formed on the spatial grid
        afterwards.
      * alpha multiplies omega as a product of functions, formed on the grid;
        (alpha*omega)_lm != alpha_lm * omega_lm.
      * Delta' = -l(l+1) + 2 acts on the product alpha*omega, at the output
        degree.
    """
    #   eq 1 uses Lap  = -l(l+1)        (lap_by_degree)
    #   eq 2 uses Lap' = -l(l+1) + 2    (lap2_by_degree)
    lap_by_degree  = np.array([-l*(l + 1)     for l in range(2*lmax + 1)])
    lap2_by_degree = np.array([-l*(l + 1) + 2 for l in range(2*lmax + 1)])

    # Load in the cached, field multiplications that are constant throughout
    Te_grid  = _cached_grid('om_Te',  lmax,
                            lambda: T_e_parent.expand(lmax=grid_expansion_res).data)
    Te2_grid = _cached_grid('om_Te2', lmax, lambda: Te_grid**2)
    a_grid   = _cached_grid('om_a',   lmax,
                            lambda: a_clm.expand(lmax=grid_expansion_res).data)

    # omega as coefficients, on the 2*lmax+1 working band the builders use.
    om = np.zeros((2, 2*lmax + 1, 2*lmax + 1))
    band = min(2*lmax + 1, omega_corr_phys.shape[1])
    om[:, :band, :band] = omega_corr_phys[:, :band, :band]
    om_clm = pysh.SHCoeffs.from_array(om)

    # FIELD: eq-1 c2 half:  Te^2 * Lap(omega).  Lap first, then the grid product
    om_lap = om_clm.copy()
    for l in range(2*lmax + 1):
        om_lap.coeffs[:, l, :] *= lap_by_degree[l]
    Te2_lap_om = pysh.SHGrid.from_array(
        Te2_grid * om_lap.expand(lmax=grid_expansion_res).data).expand()
    Te2_lap_om = pysh.SHCoeffs.from_array(Te2_lap_om.coeffs[:, :2*lmax + 1, :2*lmax + 1])

    # FIELD: eq-2:  Delta'( alpha * omega ).  Product on the grid, Delta' after
    a_om = pysh.SHGrid.from_array(
        a_grid * om_clm.expand(lmax=grid_expansion_res).data).expand()
    a_om = pysh.SHCoeffs.from_array(a_om.coeffs[:, :2*lmax + 1, :2*lmax + 1])
    for l in range(2*lmax + 1):
        a_om.coeffs[:, l, :] *= lap2_by_degree[l]

    # Prefactors
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    corr1 = Re/R
    factor_eq1a = -2.0*Re**3 * Kalousova_scaler1 * corr1 * R
    factor_eq1b = (Re/12.0)  * Kalousova_scaler1 * corr1 * R
    factor_eq2  = -(1.0 - nu) * Re**2

    rhs1_add = (factor_eq1a * om_clm.coeffs[:, :lmax+1, :lmax+1]
                + factor_eq1b * Te2_lap_om.coeffs[:, :lmax+1, :lmax+1])
    rhs2_add = factor_eq2 * a_om.coeffs[:, :lmax+1, :lmax+1]

    return (pysh.shio.SHCilmToVector(rhs1_add),
            pysh.shio.SHCilmToVector(rhs2_add))


# %% OMEGA-TERMS EQUATIONS (SUBSTITUTIONS INTO w-F EQUATIONS)

def Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, lmax, R, T_e_0, Re, g0, mass):
    """
    Full set of terms for the Omega parameters of the first equation in the
    system of two equations. 
    
    At current stage, the SH function products are done by expansion
    into the spatial domain, performing the multiplication there and then
    transforming back to spatial domain. It should be possible to perform this
    product using the Gaunt coefficients too, which may be implemented at a 
    next stage.
    """
        
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    # Thin-shell correction factor
    corr1 = Re/R        # ETA FIELD FIX
    
    # Laplacian array for degrees
    lap_by_degree = np.array([-l * (l + 1) for l in range(2 * lmax + 1)])

    # (R-Tc)/R^(l+2) for degrees l
    # DC FILTER (moho, dc_lm branch): RTcR_l2 carries the eq-(1) moho
    # continuation, so dividing it by DCfilter_mohoD makes every 1/RTcR_l2 below pick
    # up x DCfilter_mohoD. See DCfilters.
    RTcR_l2 = (np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])
               / DCfilters(2 * lmax)[0])   # [0] = DCfilter_mohoD

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ (rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])
    
    
    # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------    
    T_e_parent_grid_eq1RHS = T_e_parent.expand(lmax=grid_expansion_res).data
    topo_grid_eq1RHS = topo_clm.expand(lmax=grid_expansion_res).data - R
    geoid_grid_eq1RHS = geoid_clm.expand(lmax=grid_expansion_res).data - R
    Te2_grid = T_e_parent_grid_eq1RHS**2    
    
    # max(Te - Tc, 0) field
    TeTc_grid = T_e_parent_grid_eq1RHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
     
    # pre-weighted topo  H' = H / phi^(l+2)
    Hp = pysh.SHGrid.from_array(topo_grid_eq1RHS).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]):
        Hp.coeffs[:, l, :] *= 1.0 / RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=grid_expansion_res).data
    
    # pre-weighted geoid  G' = rhobar(2l+1)/phi^(l+2) * G
    Gp = pysh.SHGrid.from_array(geoid_grid_eq1RHS).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]):
        Gp.coeffs[:, l, :] *= rhobar2l1[l] / RTcR_l2[l]
    Gp_grid = Gp.expand(lmax=grid_expansion_res).data
    
    
    # ------- THE FIELDS FOR EACH TERM -------
    # Field RHS 1a: Te*H grid
    TeH_grid = T_e_parent_grid_eq1RHS * topo_grid_eq1RHS
    TeH_clm = pysh.SHGrid.from_array(TeH_grid).expand()
    TeH_clm = pysh.SHCoeffs.from_array(TeH_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field RHS 1b: Te**2 * Laplacian(Te * topo)
    # (Laplacian on the INNER product)
    TeH_lap = TeH_clm.copy()
    for l in range(TeH_lap.coeffs.shape[1]):
        TeH_lap.coeffs[:, l, :] *= lap_by_degree[l]
    TeH_lap_grid = TeH_lap.expand(lmax=grid_expansion_res)
    TeH_lap_Te2_grid = TeH_lap_grid.data * Te2_grid.data
    TeH_lap_Te2_clm = pysh.SHGrid.from_array(TeH_lap_Te2_grid).expand()
    
    
    # Field dc1 :  max * H'    
    # (no Laplacian)
    dc1_clm = pysh.SHGrid.from_array(TeTc_grid * Hp_grid).expand()
    dc1_clm = pysh.SHCoeffs.from_array(dc1_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # Field dc2 :  Te^2 * Laplacian( max * H' )      
    # (Laplacian on the INNER product, as in 1b)
    tmp = dc1_clm.copy()
    for l in range(tmp.coeffs.shape[1]):
        tmp.coeffs[:, l, :] *= lap_by_degree[l]
    dc2_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=grid_expansion_res).data).expand()
    dc2_clm = pysh.SHCoeffs.from_array(dc2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
     
    # dc3 :  max * G'   
    # (no Laplacian)  
    dc3_clm = pysh.SHGrid.from_array(TeTc_grid * Gp_grid).expand()
    dc3_clm = pysh.SHCoeffs.from_array(dc3_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # dc4 :  Te^2 * Laplacian( max * G' )
    # (Laplacian on the INNER product, as in 1b)
    tmp = dc3_clm.copy()
    for l in range(tmp.coeffs.shape[1]):
        tmp.coeffs[:, l, :] *= lap_by_degree[l]
    dc4_clm = pysh.SHGrid.from_array(Te2_grid.data * tmp.expand(lmax=grid_expansion_res).data).expand()
    dc4_clm = pysh.SHCoeffs.from_array(dc4_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    
    # ------- drho_lm VARIABLES AND FIELDS -------
    #  definitions of g_M, B_1 and B_2, and for the two fixes it encodes.)
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']
    g_M   = _dl['g_M']
    B_1   = _dl['B_1']
    Cp    = _dl['Cp']
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid_eq1RHS - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)
    
    # =====================================================================
    # drho_lm RHS fields (omega's drhom part; solve_for == 'drho_lm')
    # ---------------------------------------------------------------------
    # omega gains  P_hat * drhom / R, and with dc = 0 the eq-(1) elimination
    # gives, PER DEGREE,
    #     drhom_H,lm = -Cbar_l * rho_l * H_lm / B_1_l
    #     drhom_G,lm = +           G_lm / B_1_l
    # so the omega content is the SPATIAL product
    #     P_hat(theta,phi) * drhom(theta,phi).
    #
    # STRUCTURAL FIXES vs the previous version:
    #  1. P_hat was multiplied into H/G COEFFICIENT-WISE
    #     (`H.coeffs[:,l,:] *= P_frac.coeffs[:,l,:]`). That is not a product
    #     of two functions: (P*H)_lm != P_lm * H_lm. The product must be
    #     formed on the GRID and re-expanded.
    #  2. 1/B_1 was folded into P_hat's OWN coefficients, i.e. applied at the
    #     FIELD degree. It belongs to drhom, i.e. to the degree of H/G --
    #     it is applied to the H/G coefficients here, before the grid product.
    #  3. The c2-half Laplacian was applied to the weighted H/G BEFORE
    #     multiplying by P_hat. The operator is  Te^2 * Delta( P_hat * drhom ),
    #     so Delta acts on the PRODUCT.
    # =====================================================================
    Phat_g = MTeMt * TeMt0        # pure grid field
    
    def _drhom_part(source_coeffs, weights):
        """ 
        Per-degree weight on the source coefficients. Return grid for calcs. 
        """
        weighted_coeffs = pysh.SHCoeffs.from_array(np.array(source_coeffs.coeffs[:, :2*lmax+1, :2*lmax+1]))
        for l in range(weighted_coeffs.coeffs.shape[1]):
            weighted_coeffs.coeffs[:, l, :] *= weights[l]
        return weighted_coeffs.expand(lmax=grid_expansion_res).data
    
    def _c1_c2(prod_grid):
        """
        Each Omega_eq1_RHS term has two halves due to two omega terms in the eq.
        Calculate each drho-term half and return as coeffs.
        c1-half = P_hat*drhom 
        c2-half = Te^2 * Delta(P_hat*drhom)
        """
        p_clm = pysh.SHGrid.from_array(prod_grid).expand()
        p_clm = pysh.SHCoeffs.from_array(p_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
        lap = pysh.SHCoeffs.from_array(np.array(p_clm.coeffs))
        for l in range(lap.coeffs.shape[1]):
            lap.coeffs[:, l, :] *= lap_by_degree[l]
        lap_g = Te2_grid.data * lap.expand(lmax=grid_expansion_res).data
        lap_clm = pysh.SHGrid.from_array(lap_g).expand()
        lap_clm = pysh.SHCoeffs.from_array(lap_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
        return p_clm, lap_clm
    
    # degree-weights for two H-field terms
    wH_drho = np.array([-Cp[l] / B_1[l] for l in range(2*lmax+1)])
    # degree-weights for two G-field terms
    wG_drho = np.array([ 1.0 / B_1[l]   for l in range(2*lmax+1)])
    
    topo_clm_eq1RHS  = pysh.SHGrid.from_array(topo_grid_eq1RHS).expand()
    geoid_clm_eq1RHS = pysh.SHGrid.from_array(geoid_grid_eq1RHS).expand()
    field_drho1, field_drho2 = _c1_c2(Phat_g * _drhom_part(topo_clm_eq1RHS,  wH_drho))
    field_drho3, field_drho4 = _c1_c2(Phat_g * _drhom_part(geoid_clm_eq1RHS, wG_drho))




    # ------- THE PREFACTORS OF THE EQ1 RHS OMEGA TERMS -------
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    factor1a_omega = -2.0*Re**3 *rho_l *g0 *nu/(1.0-nu) * Kalousova_scaler1 * corr1
    factor1b_omega = rho_l * g0 * (Re/12.0) * nu/(1.0-nu) * Kalousova_scaler1 * corr1
    # Term b to be multiplied with Laplacian
    
    # factors from dc_lm inclusion
    factorRHS_omega1_dc1 = -2*Re**3*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc2 = Re/12*( nu/(1-nu)*g_m*rho_l ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc3 = 2*Re**3*( nu/(1-nu)*g_m ) * Kalousova_scaler1 * corr1
    factorRHS_omega1_dc4 = -Re/12 * ( nu/(1-nu)*g_m ) * Kalousova_scaler1 * corr1
    
    # factors from drho_lm inclusion
    # RHS = -K1*R*(c1 + c2*Delta)*omega_content, and the drhom weights
    # (-Cbar*rho_l/B_1 for H, +1/B_1 for G) are now inside field_drho1..4,
    # so only the operator halves remain here:
    #   c1-half: -2Re^3*K1*corr1        c2-half: +Re/12*K1*corr1
    factorRHS_omega1_drho1 = -2*Re**3 * (-0.5*nu/(1-nu)*g_M*rho_l) * Kalousova_scaler1 * corr1  # * P_hat*drhom_H
    factorRHS_omega1_drho2 =  Re/12   * (-0.5*nu/(1-nu)*g_M*rho_l) * Kalousova_scaler1 * corr1  # * Te2*lapl(P_hat*drhom_H)
    factorRHS_omega1_drho3 = -2*Re**3 * (-0.5*nu/(1-nu)*g_M) * Kalousova_scaler1 * corr1  # * P_hat*drhom_G
    factorRHS_omega1_drho4 =  Re/12   * (-0.5*nu/(1-nu)*g_M) * Kalousova_scaler1 * corr1  # * Te2*lapl(P_hat*drhom_G)

    
    
    # ------- ASSEMBLY -------
    Omega_RHS1_coeffs = ( factor1a_omega       * TeH_clm.coeffs[:, :lmax+1, :lmax+1]
                        + factor1b_omega       * TeH_lap_Te2_clm.coeffs[:, :lmax+1, :lmax+1]
                        
                        + (factorRHS_omega1_dc1 * dc1_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega1_dc2 * dc2_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega1_dc3 * dc3_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega1_dc4 * dc4_clm.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        
                        + (factorRHS_omega1_drho1 * field_drho1.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        + (factorRHS_omega1_drho2 * field_drho2.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0) 
                        + (factorRHS_omega1_drho3 * field_drho3.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        + (factorRHS_omega1_drho4 * field_drho4.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        )
   
    # Then transform to an 'unstructured' vector (structure same as that of y in
    # solve_beuthe) 
    Omega_RHS1_unstr = pysh.shio.SHCilmToVector(Omega_RHS1_coeffs)
    
    return Omega_RHS1_unstr
   
def Omega_eq1_LHS(T_e_parent, lmax, R, T_e_0, Re, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the first equation.
    """
        
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    
    # Thin-shell approximations/correction
    corr1 = Re/R            # ETA FIELD FIX
    
    # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------   
    T_e_parent_grid_eq1LHS = T_e_parent.expand(lmax=grid_expansion_res).data
    
    # Field Te
    Te_grid = T_e_parent_grid_eq1LHS 
    Te_clm = pysh.SHGrid.from_array(Te_grid).expand()
    Te_clm = pysh.SHCoeffs.from_array(Te_clm.coeffs[:, :2*lmax+1, :2*lmax+1])  
    
    # Field Te^2 
    Te2_grid = T_e_parent_grid_eq1LHS**2 
    Te2_clm = pysh.SHGrid.from_array(Te2_grid).expand()
    Te2_clm = pysh.SHCoeffs.from_array(Te2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field max(Te-Tc,0)
    TeTc_grid = T_e_parent_grid_eq1LHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid = np.array(TeTc_grid.data)
    TeTc_grid[TeTc_grid < 0.0] = 0  
    TeTc_clm = pysh.SHGrid.from_array(TeTc_grid).expand()
    TeTc_clm = pysh.SHCoeffs.from_array(TeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field gTe (variable-Te fix)
    # gravity at the LOCAL shell-base depth,
    # mantle branch only -- every gTe-carrying term also carries max(Te-Tc,0),
    # which is zero exactly where the density branch would switch. Monopole at
    # constant Te => benchmark preserved.
    RTeR_grid = (R - T_e_parent_grid_eq1LHS) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid_eq1LHS <= T_c, rho_c, rho_m)
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2    
    
    # Field Tc if Tc < Te else 0
    Tcind_grid_1 = np.where(T_e_parent_grid_eq1LHS > T_c, T_c, 0.0)
    Tcind_clm_1  = pysh.SHGrid.from_array(Tcind_grid_1).expand()
    Tcind_clm_1  = pysh.SHCoeffs.from_array(Tcind_clm_1.coeffs[:, :2*lmax+1, :2*lmax+1])


    # ------- THE PREFACTORS OF THE EQ1 RHS OMEGA TERMS -------
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    
    # 1a & 1d change to take difference in rho_l and rho_c and the surface gravity
    # 1c and 1f changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    factorLHS_omega1a = -2*Re**3*drhol*g0*nu/(1-nu) * Kalousova_scaler1 *corr1
    
    
    # Te<Tc FIX (DSP eq 5: rhoc*gmoho*(Tc if Tc < Te else 0)): the crustal-
    # column term must vanish wherever the crust-mantle interface lies BELOW
    # the elastic shell. Implemented as the indicator field
    #     Tcind(theta,phi) = Tc * 1[Te > Tc],
    # which reduces to the constant Tc (or 0) monopole at constant Te, and
    # handles variable-Te maps that locally dip below Tc. NOTE: for maps
    # crossing Tc the indicator is discontinuous -> spectral ringing near
    # the Te = Tc contour is inherent; consider a smooth Te map or a
    # tapered indicator if that contour matters.
    factorLHS_omega1b = 2*Re**3*rho_c*g_m * Kalousova_scaler1 *corr1  # Tc in field
    factorLHS_omega1c = 2*Re**3*rho_m * Kalousova_scaler1 *corr1      # gTe in field

    # =====================================================================
    # GROUP-2 factors (1d/1e/1f) -- these are now the LIVE definitions,
    # handed to solve_beuthe in the g2 dict below. There is no second copy.
    # They are FACTORS ONLY: group 2 is the ordered operator
    #     Te^2 . Delta( X . w ),
    # so the factor and the inner field X must stay separate -- a single
    # pre-multiplied vector (Te^2 * X) cannot express it, because Te^2 sits
    # OUTSIDE the Laplacian and X INSIDE. Hence the dict, not *_unstr.
    #   1d: X = Te        
    #   1e: X = Tcind (Tc dropped from the factor)
    #   1f: X = gTe*max   (gTe dropped from the factor)
    # =====================================================================
    factorLHS_omega1d = Re/12*drhol*g0*nu/(1-nu) * Kalousova_scaler1 *corr1   # *Laplacian!
    factorLHS_omega1e = -Re/12*g_m*rho_c * Kalousova_scaler1  *corr1          # *Laplacian! (Tc -> Tcind field)
    factorLHS_omega1f = -Re/12*rho_m * Kalousova_scaler1  *corr1              # *Laplacian! (gTe -> gTemax field)


    factor_fdc1_w =  2*Re**3 * nu/(1-nu) * g_m * drhol * Kalousova_scaler1 * corr1
    factor_fdc2_w = -Re/12   * nu/(1-nu) * g_m * drhol * Kalousova_scaler1 * corr1

    # Wrap gTe in the factor fields for 1c and 1f
    gTeTeTc_clm = pysh.SHGrid.from_array(gTe_grid * TeTc_grid).expand()
    gTeTeTc_clm = pysh.SHCoeffs.from_array(gTeTeTc_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    # REMOVED: gTeTe2TeTc_clm = gTe*Te^2*max  was a PRE-MULTIPLIED single
    # field for term 1f. That cannot express group 2: the operator is
    # Te^2 . Delta( gTe*max . w ), i.e. Te^2 OUTSIDE the Laplacian and
    # gTe*max INSIDE. Convolving with one merged field (Te^2*gTe*max) is a
    # different operator. 1f is now built in build_A_tilde_group2 from the
    # separate Te2_unstr (outer) and gTemax_unstr (inner) vectors in g2.


    # ------- drho_lm VARIABLES AND FIELDS -------
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']  
    g_M = _dl['g_M']
    B_1   = _dl['B_1']
    B_2 = _dl['B_2']
    Cp    = _dl['Cp']
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid_eq1LHS - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)
 

    RTcR_l2 = np.array([RTcR**(l+2) for l in range(2*lmax+1)])
    RTcR_l  = np.array([RTcR**l     for l in range(2*lmax+1)])
    Dw_arr = -Cp * (drhol + drho*RTcR_l2) / B_1

    Lam_q_drho_arr = ( g0*drhol 
                       + g_m*drho
                       - g0*drho*Cp*drhol*RTcR_l
                       - g0*drho**2*Cp*RTcR
                       + (g_M*M - g0*drho*B_2) * Dw_arr )

    # STEP 1: dc-branch q w-coupling, precomputed here (was inline in
    # solve_beuthe). Lam_q_dc = q_H with rho_l -> drhol; zero iff rho_l==rho_c.
    # DC FILTER: negl2 IS 1/P2 -> x F1 ; nl1 IS phi/P2 -> x F1/F2.
    # (RTcR_l2 above belongs to the DRHO branch's Dw_arr and is NOT filtered
    #  here -- in that branch DSP sets DCfilter_mohoD = ones.)
    DCfilter_mohoD, DCfilter_mohoDc, _, _ = DCfilters(2*lmax)
    RTcR_nl1     = (np.array([RTcR**(-(l+1)) for l in range(2*lmax+1)])
                    * DCfilter_mohoD / DCfilter_mohoDc)
    RTcR_negl2   = (np.array([RTcR**(-(l+2)) for l in range(2*lmax+1)])
                    * DCfilter_mohoD)
    Lam_q_dc_arr = ( g0*drhol 
                     - g_m*drhol*RTcR_negl2
                     - g0*drho*drhol*Cp*(RTcR_l - RTcR_nl1) )

    # P_hat as a pure GRID field (no 1/B_1 baked in -- that lives in Dw_arr)
    Phat_g   = TeMt0 * MTeMt
    Phat_clm = pysh.SHGrid.from_array(Phat_g).expand()
    Phat_clm = pysh.SHCoeffs.from_array(Phat_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    fdrho_w1 = -2*Re**3 * nu/(1-nu) * g_m * drho * Kalousova_scaler1 * corr1
    fdrho_w2 =  Re/12   * nu/(1-nu) * g_m * drho * Kalousova_scaler1 * corr1
    fdrho_om1 = 2*Re**3 * -0.5 * nu/(1-nu) * g_M * Kalousova_scaler1 * corr1
    fdrho_om2 = -Re/12  * -0.5 * nu/(1-nu) * g_M * Kalousova_scaler1 * corr1

    # ---- group-1 LHS vectors (restored: these sat inside the replaced span)
    Omega_LHS_1a_unstr = factorLHS_omega1a * pysh.shio.SHCilmToVector(Te_clm.coeffs)
    Omega_LHS_1b_unstr = factorLHS_omega1b * pysh.shio.SHCilmToVector(Tcind_clm_1.coeffs)
    Omega_LHS_1c_unstr = factorLHS_omega1c * pysh.shio.SHCilmToVector(gTeTeTc_clm.coeffs)

    # ---- group-2 ingredients: FACTORS + separate inner field vectors ------
    gTemax_clm = pysh.SHGrid.from_array(gTe_grid * TeTc_grid).expand()
    gTemax_clm = pysh.SHCoeffs.from_array(gTemax_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    g2 = dict(
        f1d          = factorLHS_omega1d,
        f1e          = factorLHS_omega1e,
        f1f          = factorLHS_omega1f,
        fdc1_w       = factor_fdc1_w,
        fdc2_w       = factor_fdc2_w,
        Te_unstr     = pysh.shio.SHCilmToVector(Te_clm.coeffs),      # X = Te   (1d)
        Te2_unstr    = pysh.shio.SHCilmToVector(Te2_clm.coeffs),     # outer Te^2
        max_unstr    = pysh.shio.SHCilmToVector(TeTc_clm.coeffs),    # X = max  (dc pair)
        Tcind_unstr  = pysh.shio.SHCilmToVector(Tcind_clm_1.coeffs), # X = Tcind (1e)
        gTemax_unstr = pysh.shio.SHCilmToVector(gTemax_clm.coeffs),  # X = gTe*max (1f)
        # shared scalars/grids -- single source of truth for solve_beuthe
        g_m          = g_m,
        rhobar       = rhobar,
        RTcR         = RTcR,
        TeTc_grid    = TeTc_grid,
        # ---- drho_lm branch ingredients (unused when solve_for=='dc_lm') --
        w_corr           = Dw_arr,             # per-degree: drhom's w-coefficient
        Lam_q_drho   = Lam_q_drho_arr,     # per-degree: q's w-coupling (diagonal)
        Lam_q_dc     = Lam_q_dc_arr,       # per-degree: q's w-coupling (dc branch)
        # eq-2 dc coupling factor (scalar); field/weights assembled in solver
        fdc_2d       = nu * g_m * drhol * (Re**2/R),
        Phat_unstr   = pysh.shio.SHCilmToVector(Phat_clm.coeffs),  # genuine field
        fdrho_w1     = fdrho_w1,           # -w half of (dc-w), c1 (field: max)
        fdrho_w2     = fdrho_w2,           # -w half of (dc-w), c2 (field: max)
        fdrho_om1    = fdrho_om1,          # omega/drhom c1 scale
        fdrho_om2    = fdrho_om2,          # omega/drhom c2 scale
    )

    return (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr,
            g2)

 
    
def Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_clm, lmax, R, T_e_0, Re, g0, mass):
    """
    Full set of terms for the Omega parameters of the second equation in the
    system of two equations.
    
    At current stage, the SH function products are done by expansion
    into the spatial domain, performing the multiplication there and then
    transforming back to spatial domain. It should be possible to perform this
    product using the Gaunt coefficients too, which may be implemented at a 
    next stage.
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    

    # Laplacian array for degrees (incl +2 term)
    lap2_by_degree = np.array([-l * (l + 1) +2 for l in range(2 * lmax + 1)])
    
    # (R-Tc)/R^(l+2) for degrees l
    # DC FILTER (moho, dc_lm branch): RTcR_l2 carries the eq-(1) moho
    # continuation, so dividing it by DCfilter_mohoD makes every 1/RTcR_l2 below pick
    # up x DCfilter_mohoD. See DCfilters.
    RTcR_l2 = (np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])
               / DCfilters(2 * lmax)[0])   # [0] = DCfilter_mohoD

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ 3/ (rhobar*(2*l+1)) for l in range(2 * lmax + 1)])

    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid_eq2RHS = T_e_parent.expand(lmax=grid_expansion_res).data
    topo_grid_eq2RHS = topo_clm.expand(lmax=grid_expansion_res).data - R
    geoid_grid_eq2RHS = geoid_clm.expand(lmax=grid_expansion_res).data - R
    alpha_grid_eq2RHS = a_clm.expand(lmax=grid_expansion_res).data
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid_eq2RHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
    
    
    # Field RHS 2a: lap2 * Te*H*alpha grid
    TeHa_grid = T_e_parent_grid_eq2RHS * topo_grid_eq2RHS * alpha_grid_eq2RHS
    TeHa_clm = pysh.SHGrid.from_array(TeHa_grid).expand()
    TeHa_clm = pysh.SHCoeffs.from_array(TeHa_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    # Perform multiplication with laplacian2, by multiplying it with 
    # the TeHa coefficients for the degrees l only
    TeHa_lap = TeHa_clm.copy()
    for l in range(TeHa_lap.coeffs.shape[1]):
        TeHa_lap.coeffs[:, l, :] *= lap2_by_degree[l]


    # (same H', G'; here each product also carries alpha and the Laplacian is +2)
    Hp = pysh.SHGrid.from_array(topo_grid_eq2RHS).expand()
    Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Hp.coeffs.shape[1]): 
        Hp.coeffs[:, l, :] *= 1.0/RTcR_l2[l]
    Hp_grid = Hp.expand(lmax=grid_expansion_res).data
    
    Gp = pysh.SHGrid.from_array(geoid_grid_eq2RHS).expand()
    Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(Gp.coeffs.shape[1]): 
        Gp.coeffs[:, l, :] *= 1/( rhobar2l1[l] * RTcR_l2[l] )
    Gp_grid = Gp.expand(lmax=grid_expansion_res).data
     
    d_dc1 = pysh.SHGrid.from_array(TeTc_grid * Hp_grid * alpha_grid_eq2RHS).expand()   # max*H'*alpha
    d_dc1 = pysh.SHCoeffs.from_array(d_dc1.coeffs[:, :2*lmax+1, :2*lmax+1])
    d_dc2 = pysh.SHGrid.from_array(TeTc_grid * Gp_grid * alpha_grid_eq2RHS).expand()   # max*G'*alpha
    d_dc2 = pysh.SHCoeffs.from_array(d_dc2.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(d_dc1.coeffs.shape[1]):       # Laplacian+2 on the inner product (as in 2a)
        d_dc1.coeffs[:, l, :] *= lap2_by_degree[l]
        d_dc2.coeffs[:, l, :] *= lap2_by_degree[l]
    
    
    
    
    # drho_lm terms
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']  
    g_M = _dl['g_M']
    B_1   = _dl['B_1']
    Cp    = _dl['Cp']  
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2*lmax+1)])
    # Te-dependent layer fields (kept local: they need T_e_parent_grid)
    TeMt_grid  = T_e_parent_grid_eq2RHS - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)
 
    
    P_frac = -0.5 * nu/(1-nu) * g_M * MTeMt * TeMt0
    P_frac = pysh.SHGrid.from_array(P_frac).expand()
    P_frac = pysh.SHCoeffs.from_array(P_frac.coeffs[:, :2*lmax+1, :2*lmax+1])
    for l in range(P_frac.coeffs.shape[1]):
        P_frac.coeffs[:, l, :] *= 1/B_1[l]
    
    # field drho1 term:   lapl+2 * a_lm*P_frac*Cp*H_lm
    # =====================================================================
    # drho_lm RHS fields for eq 2 (solve_for == 'drho_lm')
    # ---------------------------------------------------------------------
    # eq-2 omega operator:  (1-nu)*scaler2 * Delta' ( alpha * [content] ),
    # content = P_hat * drhom, with the SAME per-degree drhom weights as eq 1
    #   drhom_H,lm = -Cbar_l*rho_l*H_lm/B_1_l ,  drhom_G,lm = G_lm/B_1_l .
    # Same three structural fixes as eq 1: the P_hat product is SPATIAL (not
    # coefficient-wise), 1/B_1 acts at the H/G degree (not P_hat's), and the
    # output Delta' acts on alpha*(P_hat*drhom) -- i.e. AFTER both products,
    # not on the weighted H/G beforehand.
    # =====================================================================
    Phat_g2 =  MTeMt * TeMt0

    def _drhom_grid2(src_clm, wts):
        c = pysh.SHCoeffs.from_array(np.array(src_clm.coeffs[:, :2*lmax+1, :2*lmax+1]))
        for l in range(c.coeffs.shape[1]):
            c.coeffs[:, l, :] *= wts[l]
        return c.expand(lmax=grid_expansion_res).data

    def _eq2_field(prod_grid):
        c = pysh.SHGrid.from_array(alpha_grid_eq2RHS * prod_grid).expand()
        c = pysh.SHCoeffs.from_array(c.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(c.coeffs.shape[1]):
            c.coeffs[:, l, :] *= lap2_by_degree[l]
        return c

    wH_d2 = np.array([-Cp[l]  / B_1[l] for l in range(2*lmax+1)])
    wG_d2 = np.array([ 1.0 / B_1[l]           for l in range(2*lmax+1)])
    _topo_c2  = pysh.SHGrid.from_array(topo_grid_eq2RHS).expand()
    _geoid_c2 = pysh.SHGrid.from_array(geoid_grid_eq2RHS).expand()
    d_drho1 = _eq2_field(Phat_g2 * _drhom_grid2(_topo_c2,  wH_d2))
    d_drho2 = _eq2_field(Phat_g2 * _drhom_grid2(_geoid_c2, wG_d2))



    # ------ PREFACTORS OF THE EQ2 RHS OMEGA TERMS ------
    Kalousova_scaler2 = Re**2/R
    factor2a_omega = -1.0* nu * rho_l * g0 * Kalousova_scaler2  # *(Laplacian+2)
    
    factorRHS_omega2_dc1 = -nu*g_m*rho_l * Kalousova_scaler2
    factorRHS_omega2_dc2 = nu*g_m * Kalousova_scaler2

    factorRHS_omega2_drho1 = (0.5*nu*g_M*rho_l) * Kalousova_scaler2   # * Delta'(alpha*P_hat*drhom_H)
    factorRHS_omega2_drho2 = (0.5*nu*g_M) * Kalousova_scaler2   # * Delta'(alpha*P_hat*drhom_G)
    
    
    # ------ ASSEMBLY ------
    Omega_RHS2_coeffs = ( 
                           factor2a_omega       * TeHa_lap.coeffs[:, :lmax+1, :lmax+1]
                        + (factorRHS_omega2_dc1 * d_dc1.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega2_dc2 * d_dc2.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                        + (factorRHS_omega2_drho1 * d_drho1.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        + (factorRHS_omega2_drho2 * d_drho2.coeffs[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                        )
    
    
    # Then transform to an 'unstructured' vector (structure same as that of y in
    # solve_beuthe) 
    Omega_RHS2_unstr = pysh.shio.SHCilmToVector(Omega_RHS2_coeffs)
    
    return Omega_RHS2_unstr 
    
    

def Omega_eq2_LHS(T_e_parent, a_clm, lmax, R, T_e_0, Re, g0, mass):
    """ 
    Compute the spherical harmonic function field products and the prefactors
    for the LHS integration of the omega coefficients of the second equation.
    
    A number of Te and alpha products occur in the LHS terms. These can be
    simplified, since alpha = 1/(E*Te), thus reducing to 1/E for the product.
    """
        
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    

    # SH-MULTIPLIED FIELDS        
    T_e_parent_grid_eq2LHS = T_e_parent.expand(lmax=grid_expansion_res).data
    a_grid_eq2LHS = a_clm.expand(lmax=grid_expansion_res).data
    # gTe FIELD (variable-Te fix): gravity at the LOCAL shell-base depth,
    # mantle branch only -- every gTe-carrying term also carries max(Te-Tc,0),
    # which is zero exactly where the density branch would switch. Monopole at
    # constant Te => benchmark preserved.
    RTeR_grid = (R - T_e_parent_grid_eq2LHS) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid_eq2LHS <= T_c, rho_c, rho_m)
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2
    
    # Te - Tc field
    TeTc_grid = T_e_parent_grid_eq2LHS - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data


    # Field 2a: Te * alpha
    Tea_grid = T_e_parent_grid_eq2LHS * a_grid_eq2LHS
    Tea_clm = pysh.SHGrid.from_array(Tea_grid).expand()
    Tea_clm = pysh.SHCoeffs.from_array(Tea_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 2b: alpha
    a_clm_copy = a_clm.copy()
    a_clm_copy.coeffs = a_clm_copy.coeffs[:, :2*lmax+1, :2*lmax+1]
    Tcind_grid_2 = np.where(T_e_parent_grid_eq2LHS > T_c, T_c, 0.0)
    Tcinda_clm   = pysh.SHGrid.from_array(Tcind_grid_2 * a_grid_eq2LHS.data).expand()
    Tcinda_clm   = pysh.SHCoeffs.from_array(Tcinda_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
    
    # Field 2c: max(Te-Tc,0) * alpha
    gTeTeTca_grid = gTe_grid * TeTc_grid * a_grid_eq2LHS  # gTe grid folded into here for variable Te
    gTeTeTca_clm  = pysh.SHGrid.from_array(gTeTeTca_grid).expand()
    gTeTeTca_clm  = pysh.SHCoeffs.from_array(gTeTeTca_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

    

    # ------- drho_lm VARIABLES AND FIELDS -------
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M']
    g_M = _dl['g_M']
    
    # Te - Mt field
    TeMt_grid  = T_e_parent_grid_eq2LHS - Mt
    TeMt0      = np.array(TeMt_grid.data)
    TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
    MTeMt      = np.array(TeMt_grid.data)
    MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)

    # 2d's field is max*alpha (NOT gTe*max*alpha -- gTe belongs to 2c only)
    TeTca_grid = pysh.SHGrid.from_array(TeTc_grid * a_grid_eq2LHS).expand()
    TeTca_clm = pysh.SHCoeffs.from_array(TeTca_grid.coeffs[:, :2*lmax+1, :2*lmax+1])

    # drho branch: omega's drhom term  P_hat*drhom/R  contributes the
    # w-coupling  P_hat * w_corr  (w_corr per-degree, P_hat a field) -> supplied to
    # solve_beuthe as a separate (field, diagonal) pair, since the operand
    # weight cannot be folded into a single convolution field.
    Phata_clm = pysh.SHGrid.from_array( MTeMt * TeMt0 * a_grid_eq2LHS ).expand()
    Phata_clm = pysh.SHCoeffs.from_array(Phata_clm.coeffs[:, :2*lmax+1, :2*lmax+1])


    # ------ PREFACTORS OF THE EQ2 LHS OMEGA TERMS ------
    Kalousova_scaler2 = Re**2/R
    factorLHS_omega2a = -drhol*g0*nu * Kalousova_scaler2         # *Laplacian+2!
    factorLHS_omega2b = (1-nu)*rho_c*g_m * Kalousova_scaler2 # *Laplacian+2! (Tc in field)
    
    # 2c changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
    factorLHS_omega2c = (1-nu)*rho_m * Kalousova_scaler2 # *Laplacian+2! # gTe in field 
    
    # the fourth prefactor term for crustal thickness variations
    # 2d -- omega's  v1v*drho*g_m*max*(dc - w)/R  term, BRANCH-DEPENDENT:
    #  dc branch  : the (dc-w) combination must be substituted as a WHOLE.
    #               drho*dc carries a hidden +drho*w that cancels the -drho*w
    #               exactly, leaving only +v1v*g_m*drhol*max*phi^-(l+2)*w
    #               (handled in solve_beuthe). Hence 0 here.
    #  drho branch: dc = 0, so nothing cancels and the -w half STANDS ALONE:
    #               -v1v*drho*g_m*max*w  ->  factor -nu*drho*g_m*scaler2
    #               (field max*alpha). This is exactly the historical
    #               commented-out value: right here, wrong in the dc branch.
    factorLHS_omega2d = (0.0 if solve_for == 'dc_lm'
                            else -nu*drho*g_m*Kalousova_scaler2)
    
    factorLHS_omega2e = (1-nu) *-0.5 * nu/(1-nu) * g_M * Kalousova_scaler2

    # Transform into SHtools vectorformat again
    Omega_LHS_2a_unstr      = factorLHS_omega2a * pysh.shio.SHCilmToVector(Tea_clm.coeffs)
    Omega_LHS_2b_unstr      = factorLHS_omega2b * pysh.shio.SHCilmToVector(Tcinda_clm.coeffs)
    Omega_LHS_2c_unstr      = factorLHS_omega2c * pysh.shio.SHCilmToVector(gTeTeTca_clm.coeffs)
    Omega_LHS_2d_unstr      = factorLHS_omega2d * pysh.shio.SHCilmToVector(TeTca_clm.coeffs)
    Omega_LHS_2_Phata_unstr = factorLHS_omega2e * pysh.shio.SHCilmToVector(Phata_clm.coeffs)
    # raw (unfactored) max*alpha vector: the eq-2 dc coupling in solve_beuthe
    # convolves this then applies its own output/operand weights, so it needs
    # the bare field, not the 2d-factored one.
    maxa_raw_unstr          = pysh.shio.SHCilmToVector(TeTca_clm.coeffs)
    


    # NOTE: the old Omega_LHS_2e_drho_unstr is gone. It multiplied alpha,
    # P_frac and the per-degree weights COEFFICIENT-WISE into one vector;
    # (P*a)_lm != P_lm*a_lm, and the per-degree w_corr weight is a DIAGONAL on the
    # operand, not a field. It is replaced by the (field, diagonal) pair
    # Omega_LHS_2_Phata_unstr + g2['w_corr'], combined in solve_beuthe.
    return (Omega_LHS_2_Phata_unstr,
            Omega_LHS_2a_unstr, 
            Omega_LHS_2b_unstr, 
            Omega_LHS_2c_unstr,
            Omega_LHS_2d_unstr,
            maxa_raw_unstr)



def q_lm(topo_clm, geoid_clm, lmax, R, T_e_0, Re, g0, mass):
    """
    Compute the loading terms q_lm of the first equation. The equation is
    
    q_lm = g0 * rho_l * (H_lm - G_lm) + g_m * drho * (w_lm - dc_lm - Gc_lm)

    where dc_lm and Gc_lm are rewritten in terms of H_lm, G_lm and w_lm, and 
    the w_lm terms are moved from this equation to the Eq1 LHS to be included
    in the Gaunt computations.
    """

    topo_clm_copyq = topo_clm.copy()
    geoid_clm_copyq = geoid_clm.copy()
    topo_clm_copyq.coeffs[0,0,0] = 0
    geoid_clm_copyq.coeffs[0,0,0] = 0
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2   
    
    # (R-Tc)/R^(l+2) for degrees l
    # DC FILTER (moho, dc_lm branch): RTcR_l2 carries the eq-(1) moho
    # continuation, so dividing it by DCfilter_mohoD makes every 1/RTcR_l2 below pick
    # up x DCfilter_mohoD. See DCfilters.
    # NOTE: RTcR_l2 is REDEFINED further down for the drho_lm block, so any
    # dc-branch use BELOW that point must not rely on this name. The degree-1
    # block near the end of this function was doing exactly that, and the
    # filter silently did not reach degree 1. DCfilter_mohoD is kept under its
    # own name for that reason.
    DCfilter_mohoD = DCfilters(2 * lmax)[0]
    RTcR_l2 = (np.array([((R-T_c)/R)**(l+2) for l in range(2 * lmax + 1)])
               / DCfilter_mohoD)

    # (R-Tc)/R^(l+1) for degrees l
    RTcR_l1 = np.array([((R-T_c)/R)**(l) for l in range(2 * lmax + 1)])
    # --->> CHANGED FROM (l+1) TO (l) TO ALIGN WITH DSP

    # (R-Tc)/R^(-l+1) for degrees l
    # DC FILTER: phi**(-l-1) IS phi/P2, so it carries BOTH moho filters.
    # Building it from the already-filtered RTcR_l2 inherits F1 automatically;
    # dividing by F2 supplies the eq-(2) half.  -> x F1/F2, as derived in
    # DCfilters.
    RTcR_negl1 = RTcR / (RTcR_l2 * DCfilters(2 * lmax)[1])  # [1] = mohoDc
    # --->> CHANGED FROM (-l+1) TO (-l-1) TO ALIGN WITH DSP

    # 2l+1 for degrees l
    rhobar2l1 = np.array([ (rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])

    # Perform the multiplications with degree-dependent terms
    field_topo_dc1  = topo_clm_copyq.coeffs.copy()                
    field_topo_dc2  = topo_clm_copyq.coeffs.copy()   
    field_topo_dc3  = topo_clm_copyq.coeffs.copy() 
    field_geoid_dc4 = geoid_clm_copyq.coeffs.copy() 
    field_geoid_dc5 = geoid_clm_copyq.coeffs.copy()              
    
    for l in range(field_topo_dc1.shape[1]):
        field_topo_dc1[:, l, :] *= (1/RTcR_l2[l])
        field_topo_dc2[:, l, :] *= (RTcR_l1[l] / rhobar2l1[l])
        field_topo_dc3[:, l, :] *= (RTcR_negl1[l] / rhobar2l1[l])
        field_geoid_dc4[:, l, :] *= (rhobar2l1[l] / RTcR_l2[l])
        field_geoid_dc5[:, l, :] *= RTcR_negl1[l]
    
    
    
    # drho_lm terms
    _dl   = drho_layer(lmax, R, g0, mass)
    M     = _dl['M'] 
    g_M = _dl['g_M']
    B_1   = _dl['B_1']
    B_2 = _dl['B_2']
    Cp    = _dl['Cp']
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2*lmax+1)])
     

    field_topo_drho1  = topo_clm_copyq.coeffs.copy()
    field_topo_drho2  = topo_clm_copyq.coeffs.copy()
    field_topo_drho3  = topo_clm_copyq.coeffs.copy()

    field_geoid_drho4 = geoid_clm_copyq.coeffs.copy()
    field_geoid_drho5 = geoid_clm_copyq.coeffs.copy()
    
    for l in range(field_topo_drho1.shape[1]):
        field_topo_drho1[:, l, :] *= Cp[l] * RTcR_l1[l]
        field_topo_drho2[:, l, :] *= Cp[l] * B_2[l] / B_1[l]
        field_topo_drho3[:, l, :] *= Cp[l] / B_1[l]

        field_geoid_drho4[:, l, :] *= 1 / B_1[l]
        field_geoid_drho5[:, l, :] *= B_2[l] / B_1[l]



    # Loading terms per each SH field multiplication later & Kalousova scaler
    Kalousova_scaler1 = 1.0/(E*T_e_0**3)
    q_topo_term1  = g0*rho_l            # No SH field
    q_geoid_term1 = -g0*rho_l           # No SH field
    
    q_topo_dc1    = -g_m*rho_l          # / RTcRl2
    q_topo_dc2    = -g0*drho*rho_l      # * RTcRl1/rhobar2l1
    q_topo_dc3    = g0*drho*rho_l       # * RTcR_nl1/rhobar2l1
    q_geoid_dc4   = g_m                 # *rhobar2l1 / RTcRl2
    q_geoid_dc5   = -g0*drho            # * RTcR_nl1

    q_topo_drho1  = -g0 * drho * rho_l  # * Cp * RTcR_l1 * topo
    q_topo_drho2  =  g0 * drho * rho_l  # * Cp * B_2/B_1 * topo
    q_topo_drho3  = -g_M * M * rho_l    # * Cp / B_1 * topo
    q_geoid_drho4 =  g_M * M            # * 1 / B_1 * geoid
    q_geoid_drho5 = -g0 * drho          # * B_2 / B_1 * geoid
    
    
    # Make coeffs array of size lmax+1 for the RHS
    q_coeffs = -Re**4 * Kalousova_scaler1 * (
                  q_topo_term1 
                   * topo_clm_copyq.coeffs[:, :lmax+1, :lmax+1]
                + q_geoid_term1
                   * geoid_clm_copyq.coeffs[:, :lmax+1, :lmax+1] 
                
                + (q_topo_dc1 
                   * field_topo_dc1[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_topo_dc2
                   * field_topo_dc2[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_topo_dc3 
                   * field_topo_dc3[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_geoid_dc4
                   * field_geoid_dc4[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                + (q_geoid_dc5
                   * field_geoid_dc5[:, :lmax+1, :lmax+1] if solve_for == 'dc_lm' else 0)
                
                + (q_topo_drho1
                   * field_topo_drho1[:, :lmax+1, :lmax+1]  if solve_for == 'drho_lm' else 0)
                + (q_topo_drho2
                   * field_topo_drho2[:, :lmax+1, :lmax+1]  if solve_for == 'drho_lm' else 0)
                + (q_topo_drho3
                   * field_topo_drho3[:, :lmax+1, :lmax+1]  if solve_for == 'drho_lm' else 0)
                + (q_geoid_drho4
                   * field_geoid_drho4[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                + (q_geoid_drho5
                   * field_geoid_drho5[:, :lmax+1, :lmax+1] if solve_for == 'drho_lm' else 0)
                ) 

    # ---- degree 1: DSP enforces the COM constraint Gc_1 = 0 (thinshell.py eq(2),
    # "Force the degree-1 geoid to zero"), so the Gc-elimination terms must be
    # dropped here. Harmless to the solve (solve_beuthe zeroes the l=0,1 rows of rhs),
    # but does affect compute_omega and the stress-strain calculations
    H1 = topo_clm_copyq.coeffs[:, 1, :2]
    G1 = geoid_clm_copyq.coeffs[:, 1, :2]

    if solve_for == 'dc_lm':
        # DC FILTER: eq (1) at l = 1 with w_1 = 0, so dc1 picks up
        # x DCfilter_mohoD[1] like every other eq-(1) elimination.
        # RTcR**3 is written out rather than reusing RTcR_l2, because that
        # name has been REBOUND to the unfiltered drho-branch array above --
        # using it here left degree 1 unfiltered and produced a degree-1
        # Omega/A-S mismatch that appeared only in the dc_lm branch with the
        # filter on.
        dc1      = ((rho_l * H1 - rhobar * G1) * DCfilter_mohoD[1]
                    / (drho * RTcR**3))                          # eq(1), w_1 = 0
        q_phys_1 = g0 * rho_l * (H1 - G1) - g_m * drho * dc1        # eq(3), Gc_1 = w_1 = 0
        q_coeffs[:, 1, :2] = -Re**4 * Kalousova_scaler1 * q_phys_1
        
    if solve_for == 'drho_lm':
        q_phys_1 =  (g0*rho_l*(H1-G1)   + q_topo_drho3 * field_topo_drho3[:, 1, :2]
                        + q_geoid_drho4 * field_geoid_drho4[:, 1, :2])        
        q_coeffs[:, 1, :2] = -Re**4 * Kalousova_scaler1 * q_phys_1
        
    q_lm_unstr = pysh.shio.SHCilmToVector(q_coeffs)
        
    return q_lm_unstr


# %% A-TILDE DOUBLE CONVOLUTIONS FUNCTIONS

def build_A_tilde_group2(Te_unstr, Te2_unstr, max_unstr,
                         f1d, f1e, f1f,
                         gidx, gaunt_bare, starts, seg_len, ci, cj, mode_map, N,
                         fdc1_w=0.0, fdc2_w=0.0, Pw=None, Tcind_unstr=None, gTemax_unstr=None):
    """ 
    Correct  Te^2 * Delta'( X * w )  operators, returned as a (generally
    non-symmetric) dense N x N matrix to add into A_tilde group-2.
    This function is required for the laplacian multiplication of specific
    SH-convoluted terms in the LHS of equation 1, followed by another 
    convolution with Te^2. This group of terms is referred to as A-tilde 
    group 2.
 
    drhol EXTENSION (rho_l != rho_c): the dc-elimination of Banerdt
    eq (1) leaves a residual w-coupling in omega,
        + v1v * g_m * drhol * max(Te-Tc,0) * phi^(-(l'+2)) * w / R
    (l' = OPERAND degree), reinstating the previously-zeroed LHS dc
    terms with the correct density (drhol, not drho), sign, and the
    operand-degree weight Pw = diag(phi^(-(l+2))). Both halves of the
    [c1 + c2*Delta] omega bracket receive it:
        fdc1_w * (C_max @ Pw)              (c1-half, pairs with 1b/1c)
        fdc2_w * (Te^2 Delta (max Pw w))   (c2-half, pairs with 1e/1f)
    These vanish identically for drhol = 0.
    """
    C_Te  = build_conv_matrix(Te_unstr,  gidx, gaunt_bare, starts, seg_len, ci, cj, N)
    C_Te2 = build_conv_matrix(Te2_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
    C_max = build_conv_matrix(max_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
 
    dl   = np.array([-l*(l+1) for l, _ in mode_map])      # Delta (no +2 in eq.1)
    Dlap = np.diag(dl)
 
    Te2_Lap = C_Te2 @ Dlap                                # Te^2 . Delta  (reused)
    M_1d = f1d * (Te2_Lap @ C_Te)                         # Te^2 Delta (Te  w)
    # Te<Tc FIX: the crust-column 1e term carries the indicator field
    # Tcind = Tc*1[Te>Tc] (Tc no longer folded into f1e), mirroring DSP's
    # (Tc if Tc < Te else 0) branch and valid for variable Te dipping below Tc.
    if Tcind_unstr is not None:
        C_Tci = build_conv_matrix(Tcind_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
        M_1e = f1e * (Te2_Lap @ C_Tci)                    # Te^2 Delta (Tcind w)
    else:
        M_1e = f1e * (Te2_Lap)                            # legacy: Tc inside f1e
    if gTemax_unstr is not None:                          # gTe FIELD in 1f
        C_gmax = build_conv_matrix(gTemax_unstr, gidx, gaunt_bare, starts, seg_len, ci, cj, N)
        M_1f = f1f * (Te2_Lap @ C_gmax)                   # Te^2 Delta (gTe max w)
    else:
        M_1f = f1f * (Te2_Lap @ C_max)                    # legacy: gTe scalar in f1f
    M = M_1d + M_1e + M_1f                                # NOT symmetric -- keep full
    if (fdc1_w != 0.0 or fdc2_w != 0.0) and Pw is not None:
        CmaxP = C_max @ Pw
        M = M + fdc1_w * CmaxP + fdc2_w * (Te2_Lap @ CmaxP)
    return M



# %% FINAL OMEGA, dc AND drho EQUATIONS (COMPUTED AFTER w_lm IS KNOWN)

def compute_Omega(w_clm, T_e_parent, topo_clm, geoid_clm, q_clm, g0, R, T_e_0,
                  lmax_calc, lmax_grid, omega_corr_phys=None):
    """
    Equation for tangential loading potential Omega, following the definition
    as given in Broquet & Andrews-Hanna (2022), which is derived from Banerdt
    (1986). 
    
    In this M5, this equation has been rewritten into w-terms in order
    to maintain a 2Nx2N block matrix system, neglecting effects of crustal 
    thickness variations dc and mantle density variations dm. The solution for 
    Omega itself can therefore be obtained using the result for w_lm.
    """
    
    R_e = R - T_e_0/2
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    g_m = g0 * (1.0 + (RTcR**3 - 1.0) * rho_c / rhobar) / RTcR**2    
    # DC FILTER: the three 1/RTcR**(l+2) weightings below are eq-(1)
    # eliminations of dc (terms 2, 6 and 7, the dc-elimination
    # artefacts), so each picks up x F1.
    DCfilter_mohoD = DCfilters(lmax_calc)[0]
    
    # Grids
    w_grid_copyOmega = w_clm.expand(lmax=lmax_grid, 
                                    lmax_calc=lmax_calc).data
    T_e_parent_grid_copyOmega = T_e_parent.copy().expand(lmax=lmax_grid, 
                                                         lmax_calc=lmax_calc).data
    topo_grid_copyOmega = topo_clm.copy().expand(lmax=lmax_grid, 
                                                 lmax_calc=lmax_calc).data - R
    geoid_grid_copyOmega = geoid_clm.copy().expand(lmax=lmax_grid, 
                                                   lmax_calc=lmax_calc).data - R
    # Te - Tc field
    TeTc_grid = T_e_parent_grid_copyOmega - T_c
    # If a value is below 0, set to 0 to apply the 'max' call
    TeTc_grid_data = np.array(TeTc_grid.data)
    TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
    TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data
 
 
    # gravity at the elastic base (depth Te) for the mantle column term
    RTeR_grid = (R - T_e_parent_grid_copyOmega) / R
    # Create a dynamic rho grid based on the local thickness threshold
    rho_grid = np.where(T_e_parent_grid_copyOmega <= T_c, rho_c, rho_m)
    
    # Calculate the final gTe_grid using the dynamic rho_grid
    gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / rhobar) / RTeR_grid**2    
    
    TeH_grid = T_e_parent_grid_copyOmega * topo_grid_copyOmega
 
    # FIX (operator ordering): apply the per-degree dc-elimination weights to
    # H and G FIRST, then multiply by the max(Te-Tc,0) grid -- this is the 
    # ordering consistent with the per-degree elimination and with 
    # Omega_eq1_RHS in the solver (weight-then-multiply).
    # Previously the weights were applied to the coefficients of the PRODUCT
    # (TeTc*H), which differs for laterally varying Te.
    Hp_coeffs = pysh.SHGrid.from_array(topo_grid_copyOmega).expand()
    Hp_coeffs = truncate(Hp_coeffs, lmax=lmax_calc)
    for l in range(Hp_coeffs.coeffs.shape[1]):
        Hp_coeffs.coeffs[:, l, :] *= DCfilter_mohoD[l]/RTcR**(l+2)   # DC FILTER
    Hp_grid = Hp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    TeTcHp_grid = TeTc_grid * Hp_grid
 
    Gp_coeffs = pysh.SHGrid.from_array(geoid_grid_copyOmega).expand()
    Gp_coeffs = truncate(Gp_coeffs, lmax=lmax_calc)
    for l in range(Gp_coeffs.coeffs.shape[1]):
        Gp_coeffs.coeffs[:, l, :] *= (DCfilter_mohoD[l] * rhobar*(2*l+1)
                                      / (3 * RTcR**(l+2)))     # DC FILTER
    Gp_grid = Gp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    TeTcGp_grid = TeTc_grid * Gp_grid
    
    
    Tcind_grid_o = np.where(T_e_parent_grid_copyOmega > T_c, T_c, 0.0)   # Te<Tc FIX
 
    # Compute Re*Omega_lm as the term Omega_lm (required in conversion between
    # Banerdt and Beuthe's formulations).  w-coefficient corrected to match the
    # solve: surface -> drhol*g0 (vanishes for rho_l=rho_c), mantle -> gTe.
    term_1 = nu/(1-nu)*rho_l*g0*TeH_grid
    term_2 = + nu/(1-nu)*g_m*rho_l * TeTcHp_grid
    term_3 = -drhol*g0*nu/(1-nu)*T_e_parent_grid_copyOmega *w_grid_copyOmega
    
    term_4 = rho_c*g_m*Tcind_grid_o *w_grid_copyOmega
    term_5 = rho_m*gTe_grid*TeTc_grid *w_grid_copyOmega  # gTe field instead of scalar
    
    # drhol EXTENSION (zero if rho_l == rho_c): residual w-piece of the
    # (dc-w) substitution, + v1v*g_m*drhol*max(Te-Tc,0)*P_l*w  with the
    # weight applied to w FIRST (weight-then-multiply, as in the solver).
    wp_coeffs = pysh.SHGrid.from_array(w_grid_copyOmega).expand()
    wp_coeffs = truncate(wp_coeffs, lmax=lmax_calc)
    for l in range(wp_coeffs.coeffs.shape[1]):
        wp_coeffs.coeffs[:, l, :] *= DCfilter_mohoD[l]/RTcR**(l+2)   # DC FILTER
    wp_grid = wp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
    term_6 = + nu/(1-nu)*g_m*drhol * TeTc_grid * wp_grid
    term_7 = - nu/(1-nu)*g_m * TeTcGp_grid
 
    # Set terms depending on which internal variation is solved for.
    # Terms 2, 6, 7 are artefacts of the dc-ELIMINATION: they carry the
    # phi^-(l+2)-weighted H' and G' that replace dc. With dc = 0 they do not
    # exist. The drho_lm branch has instead, straight from DSP eq (5):
    #   term_8: the -w half of  v1v*drho*g_m*max*(dc - w)/R  now STANDS ALONE
    #           (in the dc branch the +drho*w hidden in drho*dc cancels it,
    #           leaving only the small drhol piece that is term_6);
    #   term_9: + P_hat * drho_m, with drho_m from compute_drho(w, H, G). 
    #           No elimination is needed here because w is already known.
    if solve_for == 'dc_lm':
        term_8 = 0.0
        term_9 = 0.0
    else:
        term_2 = 0.0          # dc-elimination artefacts: absent when dc = 0
        term_6 = 0.0
        term_7 = 0.0
        term_8 = - nu/(1-nu)*drho*g_m * TeTc_grid * w_grid_copyOmega
        _dl_o   = drho_layer(lmax_grid, R, g0, mass)
        TeMt_o  = T_e_parent_grid_copyOmega - Mt
        TeMt0_o = np.where(TeMt_o > 0.0, TeMt_o, 0.0)         # max(Te-Mt, 0)
        Phat_o  = (-0.5 * nu/(1-nu) * _dl_o['g_M'] * TeMt0_o
                   * np.minimum(TeMt_o, _dl_o['M']))          # min(M, Te-Mt)
        drho_m_grid = compute_drho(w_clm, topo_clm, geoid_clm, R, 
                            lmax_calc=lmax_calc, 
                            lmax_grid=lmax_grid).expand(lmax=lmax_grid).data
        term_9 = Phat_o * drho_m_grid
    
    
    # The sum of terms divided by R is omega in the DSP/Banerdt definition
    # (every term of DSP eq 5 carries 1/R). But the consumers of this output
    # (cons_disp_S, stress/strain evaluation, and DSP's own A_lm formula,
    # Beuthe 2008 eq 89) expect Beuthe's OMEGA = Re * omega -- see the DSP
    # comment "Note that omega (Beuthe) = Re * omega". Returning plain omega
    # made the Omega contribution ~Re (3.3e6x) too small, i.e. effectively
    # absent, which flipped the sign of S and corrupted all stress/strain
    # fields by factors of -0.4x to -5.6x. Hence the factor (Re/R) here.
    Re = R - T_e_0/2
    Omega_grid_data = (  term_1 
                       + term_2 
                       + term_3 
                       + term_4 
                       + term_5 
                       + term_6
                       + term_7
                       + term_8
                       + term_9) * (Re / R)
        
    Omega_grid = pysh.SHGrid.from_array(Omega_grid_data)
    Omega_clm = Omega_grid.expand()
    Omega_clm = truncate(Omega_clm, lmax=lmax_calc)
    
    # Correctly set the degree 1 Omega coefficients
    # ==== STEP 5c: drho_omega_corr, in BEUTHE's convention ================
    # THE UNIT TRAP, and the reason the first attempt at this changed
    # nothing visible. This function returns  Omega_Beuthe = Re * omega  --
    # the terms above are R*omega and the assembly multiplies by (Re/R).
    # `drho_omega_correction` returns DSP's `omega`, which already carries
    # the 1/R that every term of DSP eq (5) has. Adding the two directly is
    # therefore wrong by a factor Re = 3.26e6: the correction lands ~6.5
    # orders of magnitude too small and the plot is unchanged. It must be
    # scaled by Re here.
    #
    # WHY IT IS NEEDED AT ALL. DSP SOLVES for omega as one of its five
    # unknowns, so its stored omega_lm contains drho_omega_corr by
    # construction. M5 never forms omega as an unknown -- this function
    # re-evaluates the eq-(5) CONTENT from the converged w -- so the
    # correction has to be put back by hand. It is not a solver
    # discrepancy: w and drho already agree with DSP to ~1e-12 relative.
    # It propagates: cons_disp_S consumes Omega, so S_lm and every stress
    # and strain field built from it inherit the omission.
    if omega_corr_phys is not None:
        _n = min(Omega_clm.coeffs.shape[1], omega_corr_phys.shape[1])
        Omega_clm.coeffs[:, :_n, :_n] += Re * omega_corr_phys[:, :_n, :_n]

    # ==== STEP 5d: degree 0 ==============================================
    # DSP assembles eq (5) with `for l in range(1, lmax+1)`, so its omega_lm
    # has NO degree-0 term. M5 builds omega as a GRID and expands it, so the
    # mean of that grid lands in [0,0,0] -- a uniform ~5e4 N/km offset, which
    # is exactly the flat residual that survived the step-5c fix (residual
    # power spiked at l = 0 and was ~zero for every l >= 1).
    #
    # It is physically arbitrary, not a discrepancy: omega enters the physics
    # only through its surface gradient, and a constant has none. It is also
    # harmless downstream -- cons_disp_S divides by -l(l+1) and sets
    # S_lm[0,0,0] = 0 by hand, which is why S and every strain field already
    # matched to ~1e-12 while Omega showed a constant offset. Zeroed here so
    # the Omega comparison is like for like.
    Omega_clm.coeffs[0, 0, 0] = 0.0

    Omega_clm.coeffs[:, 1, :2] = (E * T_e_0**3 / (2.0 * R_e**3)) * q_clm.coeffs[:, 1, :2]    
    
    return Omega_clm


def compute_dc(w_clm, topo_clm, geoid_clm, R, lmax_calc, lmax_grid):
    """
    Compute the crustal root variations ('bottom loads') dc_lm using the 
    rewritten equation of Gc_lm with drho_lm=0. 
    """
    
    rhobar = mass * 3.0 / 4.0 / np.pi / R**3
    RTcR = (R-T_c)/R
    L_comp = min(w_clm.lmax, topo_clm.lmax, geoid_clm.lmax, lmax_calc)
    DCfilter_mohoD = DCfilters(L_comp)[0]   # DC FILTER, eq-(1) elimination

    # ------- PRECOMPUTE SH FIELDS -------
    topo_clm_copydc = topo_clm.copy()
    topo_clm_copydc.coeffs[0,0,0] = 0
    topo_clm_copydc = truncate(topo_clm_copydc, L_comp)

    geoid_clm_copydc = geoid_clm.copy()
    geoid_clm_copydc.coeffs[0,0,0] = 0
    geoid_clm_copydc = truncate(geoid_clm_copydc, L_comp)

    w_clm_copydc = w_clm.copy()
    w_clm_copydc.coeffs[0,0,0] = 0
    w_clm_copydc = truncate(w_clm_copydc, L_comp)
    
    # ------- ASSEMBLY -------
    for l in range(geoid_clm_copydc.coeffs.shape[1]):
       geoid_clm_copydc.coeffs[:, l, :] *= rhobar*(2*l+1)/3
    dc_clm = 1/drho * (rho_l*topo_clm_copydc + drhol*w_clm_copydc - geoid_clm_copydc)
        
    for l in range(dc_clm.coeffs.shape[1]):
       dc_clm.coeffs[:, l, :] *= (DCfilter_mohoD[l]
                                  / (RTcR**(l+2)))   # DC FILTER 
    dc_clm = dc_clm + w_clm_copydc
    
    return dc_clm


def compute_drho(w_clm, topo_clm, geoid_clm, R, lmax_calc, lmax_grid):
    """
    Compute the mantle density variations ('bottom loads') drho_lm using the 
    rewritten equation of Gc_lm with dc_lm=0. 
    """
    
    L_comp = min(w_clm.lmax, topo_clm.lmax, geoid_clm.lmax, lmax_calc)

    # ------- PRECOMPUTE SH FIELDS -------
    topo_clm_copydrho = topo_clm.copy()
    topo_clm_copydrho.coeffs[0,0,0] = 0
    topo_clm_copydrho = truncate(topo_clm_copydrho, L_comp)

    geoid_clm_copydrho = geoid_clm.copy()
    geoid_clm_copydrho.coeffs[0,0,0] = 0
    geoid_clm_copydrho = truncate(geoid_clm_copydrho, L_comp)

    w_clm_copydrho = w_clm.copy()
    w_clm_copydrho.coeffs[0,0,0] = 0
    w_clm_copydrho = truncate(w_clm_copydrho, L_comp)
 
    
    # ------- drho_lm VARIABLES AND FIELDS ------- 
    _dl   = drho_layer(lmax_grid, R, g0, mass) 
    Cp    = _dl['Cp']
    B_1 = _dl['B_1']
    RTcR_l2 = np.array([((R-T_c)/R)**(l+2) for l in range(2*lmax_grid+1)])
    
    # ------- THE FIELDS FOR EACH TERM -------
    topo_term  = topo_clm_copydrho * (-rho_l)
    w_term_1   = w_clm_copydrho * (-drho)          # moho term, weight phi^(l+2)
    w_term_2   = w_clm_copydrho * (-drhol)         # load-density term, no weight
    geoid_term = geoid_clm_copydrho.copy()
    for l in range(topo_term.coeffs.shape[1]):
        topo_term.coeffs[:, l, :]  *= Cp[l] / B_1[l]
        w_term_1.coeffs[:, l, :]   *= Cp[l] * RTcR_l2[l] / B_1[l]
        w_term_2.coeffs[:, l, :]   *= Cp[l] / B_1[l]
        geoid_term.coeffs[:, l, :] *= 1/B_1[l]
 
    # ------- ASSEMBLY -------
    drho_clm = topo_term + w_term_1 + w_term_2 + geoid_term
 
    return drho_clm


# %% STRESS AND STRAIN FIELDS - WITH CHANGES TO ALIGN WITH DSP!

kw_exp_grad = {"extend": False, "lmax_calc": LMAX_REF, "lmax": grid_expansion_res, "grid": "DH2"}
kw_exp_S = {"lmax_calc": LMAX_REF, "lmax": grid_expansion_res, "grid": "DH2"}


def O1(SH_function, lmax):
    """ Beuthe (2008)'s differential operator O_1 in 2D spherical geometry. """
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1), endpoint=False))

    cot_theta = np.divide( 1.0, np.tan(theta_range), 
                          out=np.zeros_like(np.tan(theta_range)), 
                          where=np.tan(theta_range) != 0)
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    sin_theta = np.sin(theta_range)
    sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    csc2_theta = np.divide( 1.0, (np.sin(theta_range))**2, 
                          out=np.zeros_like((np.sin(theta_range))**2), 
                          where=(np.sin(theta_range))**2 != 0)
    csc2_theta_grid = np.tile(csc2_theta.reshape(-1, 1), (1, 4*(lmax+1)))
    
    dtheta_grid = SH_function.gradient(**kw_exp_grad).theta    
    
    dphi_grid = SH_function.gradient(**kw_exp_grad).phi
    dphi_grid.data *= sin_theta_grid
    dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
    dphi2_grid = dphi_sh.gradient(**kw_exp_grad).phi
    dphi2_grid.data *= sin_theta_grid    
    
    lmax_func = SH_function.lmax
    SH_function_grid = SH_function.expand(**kw_exp_grad)

    # Laplacian identity for d2_theta            
    lapla_a = pysh.SHCoeffs.from_zeros(lmax_func)
    for l in range(lmax_func + 1):
        lapla_a.coeffs[:, l, : l + 1] = -l * (l + 1)

    
    SH_function_dtheta2 = (
                        (SH_function * lapla_a).expand(**kw_exp_grad).data 
                        - dtheta_grid.data*cot_theta_grid 
                        - dphi2_grid.data*csc2_theta_grid
                        )
    
    return SH_function_dtheta2 + SH_function_grid.data

def O2(SH_function, lmax):
    """ Beuthe (2008)'s differential operator O_2 in 2D spherical geometry. """
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1), endpoint=False))

    cot_theta = np.divide( 1.0, np.tan(theta_range), 
                          out=np.zeros_like(np.tan(theta_range)), 
                          where=np.tan(theta_range) != 0)
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    sin_theta = np.sin(theta_range)
    sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    csc_theta = np.divide( 1.0, np.sin(theta_range), 
                          out=np.zeros_like(np.sin(theta_range)), 
                          where=np.sin(theta_range) != 0)
    csc_theta_grid = np.tile(csc_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    SH_function_grid = SH_function.expand(**kw_exp_grad)
    dtheta_grid = SH_function.gradient(**kw_exp_grad).theta

    dphi_grid = SH_function.gradient(**kw_exp_grad).phi
    dphi_grid.data *= sin_theta_grid
    dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
    dphi2_grid = dphi_sh.gradient(**kw_exp_grad).phi
    dphi2_grid.data *= csc_theta_grid

    return dphi2_grid.data + cot_theta_grid * dtheta_grid.data + SH_function_grid.data

def O3(SH_function, lmax):
    """ Beuthe (2008)'s differential operator O_3 in 2D spherical geometry. """
    theta_range = np.radians(np.linspace(0, 180, 2*(lmax+1), endpoint=False))

    cot_theta = np.divide( 1.0, np.tan(theta_range), 
                          out=np.zeros_like(np.tan(theta_range)), 
                          where=np.tan(theta_range) != 0)
    cot_theta_grid = np.tile(cot_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    sin_theta = np.sin(theta_range)
    sin_theta_grid = np.tile(sin_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    csc_theta = np.divide( 1.0, np.sin(theta_range), 
                          out=np.zeros_like(np.sin(theta_range)), 
                          where=np.sin(theta_range) != 0)
    csc_theta_grid = np.tile(csc_theta.reshape(-1, 1), (1, 4*(lmax+1)))

    dphi_grid = SH_function.gradient(**kw_exp_grad).phi
    dphi_grid.data *= sin_theta_grid

    dphi_sh = dphi_grid.expand(lmax_calc=LMAX_REF)
    dthetaphi_grid = dphi_sh.gradient(**kw_exp_grad).theta

    return (csc_theta_grid * dthetaphi_grid.data 
            - cot_theta_grid * csc_theta_grid * dphi_grid.data)



def stress_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
    """
    Stresses in the DSP/Banerdt convention (Banerdt 1986 eqs A12-A14, as in
    DSP's compute_strains): plane-stress Hooke's law applied to membrane +
    bending strains built from the tangential potential S (== DSP's A_lm)
    and w, with 1/R kernels and the thin-shell top-fiber factor
    eps_f = (Te/2 - depth)/(1 + (Te/2 - depth)/R).
    Returns stresses in MPa (matching DSP).

    NOTE: this replaces the previous Beuthe eq-(73) stress-function form, 
    which evaluates the top-fiberstress with exact z/(Re+z) curvature factors 
    and 1/Re kernels. The two differ by O(Te/R) factors (~4-7% for Te=268 km).
    For benchmarking against DSP the convention must match DSP.
    """
    O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    S_grid  = S_sol.expand(**kw_exp_grad)
    w_grid  = w_sol.expand(**kw_exp_grad)
    Te_grid = T_e_parent.expand(**kw_exp_grad)

    # membrane strains (Banerdt A16-A18 with S in place of A)
    eps_t    = 1/R * (O1S - S_grid.data + w_grid.data)
    eps_p    = 1/R * (O2S - S_grid.data + w_grid.data)
    omega_sh = 1/R * (2*O3S)                       # engineering shear
    # bending strains (A19-A21); note O1w = d2w/dth2 + w, so
    # kappa_t = -(d2w/dth2)/R^2 - w/R^2 = -O1w/R^2 (and analogously kappa_p)
    kappa_t = -1/R**2 * O1w
    kappa_p = -1/R**2 * O2w
    tau     = -2/R**2 * O3w

    zeta  = Te_grid.data/2.0 - depth
    eps_f = zeta / (1.0 + zeta/R)
    DpsiTeR = E/(1.0 - nu**2)

    sigma_tt = (eps_t + nu*eps_p + eps_f*(kappa_t + nu*kappa_p)) * DpsiTeR / 1e6
    sigma_pp = (eps_p + nu*eps_t + eps_f*(kappa_p + nu*kappa_t)) * DpsiTeR / 1e6
    sigma_tp = (omega_sh + eps_f*tau) * 0.5 * DpsiTeR * (1.0 - nu) / 1e6

    return (pysh.SHGrid.from_array(sigma_tt),
            pysh.SHGrid.from_array(sigma_pp),
            pysh.SHGrid.from_array(sigma_tp))


def strain_fields(S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
    """
    Total strains in the DSP/Banerdt convention (membrane + top-fiber
    bending), matching DSP's tot_theta / tot_phi / tot_thetaphi:
        tot = eps + eps_f*kappa,  eps_f = (Te/2-depth)/(1+(Te/2-depth)/R)
    
    MISSING THE TOROIDAL DISPLACEMENT POTENTIAL T TERMS!
    """
    # Return diff operator applied S and w terms, in grid.data format
    O1S = O1(S_sol, lmax); O2S = O2(S_sol, lmax); O3S = O3(S_sol, lmax)
    O1w = O1(w_sol, lmax); O2w = O2(w_sol, lmax); O3w = O3(w_sol, lmax)
    
    S_grid  = S_sol.expand(**kw_exp_grad)
    w_grid  = w_sol.expand(**kw_exp_grad)
    Te_grid = T_e_parent.expand(**kw_exp_grad)

    eps_t    = 1/R * (O1S - S_grid.data + w_grid.data)    
    eps_p    = 1/R * (O2S - S_grid.data + w_grid.data)
    gamma_tp = 1/R * (2*O3S)
    
    kappa_t = -1/R**2 * O1w
    kappa_p = -1/R**2 * O2w
    tau     = -2/R**2 * O3w
    
    zeta = Te_grid.data/2.0 - depth
    tot_strain_pref = zeta / (1.0 + zeta/R)
    
    tot_eps_tt = eps_t    + tot_strain_pref*kappa_t
    tot_eps_pp = eps_p    + tot_strain_pref*kappa_p
    tot_eps_tp = (gamma_tp + tot_strain_pref*tau)/2.0
    
    tot_eps_tt = pysh.SHGrid.from_array(tot_eps_tt)
    tot_eps_pp = pysh.SHGrid.from_array(tot_eps_pp)
    tot_eps_tp = pysh.SHGrid.from_array(tot_eps_tp)
    
    return tot_eps_tt, tot_eps_pp, tot_eps_tp


def cons_disp_S(w_sol, F_sol, Omega_sol, T_e_parent, a_clm, R, T_e_0, lmax_calc, lmax_grid):
    """ 
    Beuthe (2008)'s consoidal/poloidal tangential displacement potential S_lm 
    (A_lm in DSP/Banerdt (1986)). Used in computations of strain.
    """
    
    lap_by_degree = np.array([(-l * (l + 1)) for l in range(2 * lmax_grid + 1)])
    lap2_by_degree = np.array([(-l * (l + 1) + 2) for l in range(2 * lmax_grid + 1)])
    F_lap2 = F_sol.copy()
    w_lap2 = w_sol.copy()
    for l in range(F_lap2.coeffs.shape[1]):
        F_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
        w_lap2.coeffs[:, l, :] *= lap2_by_degree[l]
    F_lap2_grid = F_lap2.expand(**kw_exp_S)
    w_lap2_grid = w_lap2.expand(**kw_exp_S)
    
    w_grid = w_sol.expand(**kw_exp_S)
    a_grid = a_clm.expand(**kw_exp_S)
    Te_grid = T_e_parent.expand(**kw_exp_S)
    Omega_grid = Omega_sol.expand(**kw_exp_S)

    Re = R - T_e_0/2
    xi = 12*Re**2/Te_grid.data**2
    eta = xi/(1+xi)
    
    lapl_S_grid = (Re*eta*a_grid.data*(1-nu)*(F_lap2_grid.data + 2*Omega_grid.data) 
              + eta/xi * w_lap2_grid.data  
              - 2*w_grid.data)
    
    lapl_S_lm = pysh.SHGrid.from_array(lapl_S_grid).expand()
     
    S_lm = lapl_S_lm.copy()
    for l in range(1, S_lm.coeffs.shape[1]):
        S_lm.coeffs[:, l, :] /= lap_by_degree[l]
    S_lm.coeffs[0, 0, 0] = 0.0  
    
    S_lm = truncate(S_lm, lmax=lmax_calc)
        
    return S_lm


def Principal_strainstress_angle(s_theta, s_phi, s_theta_phi):
    """
    Calculate principal strains, stresses, and
    their principal angles.

    Function definition taken over from displacement_strain_planet v0.5.0
    """

    min_strain = 0.5 * (
        (s_theta + s_phi) - np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    max_strain = 0.5 * (
        (s_theta + s_phi) + np.sqrt((s_theta - s_phi) ** 2 + 4 * s_theta_phi**2)
    )
    sum_strain = min_strain + max_strain
    principal_angle = 0.5 * np.arctan2(2 * s_theta_phi, s_theta - s_phi) * 180.0 / np.pi

    return min_strain, max_strain, sum_strain, principal_angle





# %% BEUTHE MODEL SOLVER

def solve_beuthe(topo_clm, geoid_clm, T_e_parent, D_clm, a_clm, plan, lmax, R,
                 T_e_0, g0, mass,
                 D_eta_clm=None, a_eta_clm=None, eta_clm=None,
                 q_corr_phys=None, omega_corr_phys=None, cache=None):
    """
    Implements Beuthe's UNSIMPLIFIED variable-thickness equations (58)/(66):
      Delta'(eta*D Delta' w) - (1-nu)A(eta*D; w) + R^3 A(eta; F) = RHS1
      Delta'(eta*a Delta' F) - (1+nu)A(eta*a; F) - (1/R)A(eta; w) = RHS2
    """
    
    mode_map = make_mode_map(lmax)
    N        = len(mode_map)
    Re       = R - T_e_0/2
    # Kalousova scalers
    scaler_A = 1.0/(E*T_e_0**3)
    scaler_B = Re
 
    # ============== Assemble the LHS matrix once, then cache =================
    # The LHS depends only on Te, D, alpha, eta, the Gaunt plan and lmax. None of
    # these change between iterations, only the RHS does. C_eta must exist on
    # both paths because the RHS below needs it.
    _cached = cache is not None and 'lu' in cache
    if _cached:
        lu, piv, C_eta = cache['lu'], cache['piv'], cache['C_eta']
    else:
        Dlm = pysh.shio.SHCilmToVector(D_eta_clm.coeffs)
        alm = pysh.shio.SHCilmToVector(a_eta_clm.coeffs)
 
        # ---- SoA fill: chunked gather + reduceat, no per-term Python loop
        gidx    = build_gidx(plan)                  # int32, chunked
        starts  = plan['cell_start'][:-1]
        seg_len = np.diff(plan['cell_start'])
        l_of    = mode_degrees(lmax)

        # A and B cell sums
        cellA, cellB = cell_sums_AB(Dlm, alm, gidx, plan, l_of, starts, seg_len, nu)
        cellA *= scaler_A
        cellB *= scaler_B
 
        ci, cj = plan['cell_i'], plan['cell_j']

        # Calculate the Omega LHS terms for equation 1
        (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr,
         g2) = (Omega_eq1_LHS(T_e_parent, lmax=lmax, 
                              R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
        # Sum the three fields first (they are length (lmax+1)**2 = 8281, not
        # n_terms), then do a single chunked gather instead of three.
        cell_1abc = cell_sums(Omega_LHS_1a_unstr + Omega_LHS_1b_unstr
                            + Omega_LHS_1c_unstr,
                            gidx, plan['term_gaunt_bare'], starts, seg_len)
    
        cellA_tilde = cell_1abc 
        cellA_tilde[seg_len == 0] = 0.0
 
        # Calculate the Omega LHS terms for equation 2
        (Omega_LHS_2_Phata_unstr,
         Omega_LHS_2a_unstr, 
         Omega_LHS_2b_unstr, 
         Omega_LHS_2c_unstr,
         Omega_LHS_2d_dc_unstr, 
         maxa_raw_unstr) = (Omega_eq2_LHS(T_e_parent, a_clm, lmax=lmax, 
                                          R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))
    
        # terms 2a, 2b, 2c (+2d, zero) -- carry Delta' at the OUTPUT degree.
        # b_tilde FIX (non-symmetric pathway): previously assembled per cell with
        # lap_out on the ci side and scattered SYMMETRICALLY -- exact at constant
        # Te (monopole omega-fields leave only diagonal cells) but WRONG for
        # laterally varying Te: the (j,i) orientation must carry Delta'(l_j),
        # not Delta'(l_i). Rebuilt in the omega_on block as
        #     b_tilde = diag(Delta'_out) @ C_conv(2a+2b+2c fields),
        # carrying the output weight on the correct side for both orientations.
        # NOTE: 2d must NOT be gated here -- it is exactly the other way round.
        # factorLHS_omega2d_dc is 0 in the dc branch (the (dc-w) cancellation) and
        # NONZERO in the drho branch (dc = 0, so the -w half stands alone). The
        # factor already carries the branch; gating on 'dc_lm' zeroed the only
        # branch where the term is alive.
        fields_2sum = (Omega_LHS_2a_unstr + Omega_LHS_2b_unstr + Omega_LHS_2c_unstr
                       + Omega_LHS_2d_dc_unstr)
 
 
 
        # ---- scatter per-cell values into dense blocks (loop over CELLS) ------
        # (b_tilde no longer scattered here: built as a non-symmetric matrix
        #  product in the omega_on block -- see b_tilde FIX above.)
        # Vectorised scatter. The old Python loop ran ci.size times -- 34
        # million iterations at lmax = 90. All three operators are symmetric,
        # and [i,j] / [j,i] receive the same value, so i == j needs no guard.
        A = np.zeros((N, N))
        A_tilde = np.zeros((N, N))
        B = np.zeros((N, N))
        
        A[ci, cj] = cellA          
        A[cj, ci] = cellA
        A_tilde[ci, cj] = cellA_tilde 
        A_tilde[cj, ci] = cellA_tilde
        B[ci, cj] = cellB        
        B[cj, ci] = cellB
 
 
        # ---- coupling blocks a_l, b_l --------------------------------------
        d_l2 = np.array([-l*(l+1)+2 for l,_ in mode_map], dtype=np.float64)
 
        # ETA_FULL: Beuthe eqs (58)/(66) couplings  R^3*A(eta; F)  and
        # -(1/R)*A(eta; w)  built from the existing plan. The pure-A weight
        #   W_Aonly = -br/4,
        #   br = (dl-dlp)^2 + 2(dl+dlp) - 8  +  dL^2 + 2dL - 2dL(dl+dlp)
        # splits into per-TERM (dL-dependent) and per-CELL (dl,dlp) pieces,
        # so three reduceat passes over term_gaunt_bare suffice:
        #   g0 = conv(eta, .; 1),  g1 = conv(eta, .; dL),  g2 = conv(eta, .; dL^2)
        #   A(eta;.)[cell] = -1/4 * [ ((dl-dlp)^2 + 2(dl+dlp) - 8)*g0
        #                             + (2 - 2(dl+dlp))*g1 + g2 ]
        # Monopole-eta check: dL=2, dl=dlp  ->  eta0*dl = eta0*Delta'  (the
        # exact constant-Te fix). Same symmetric (i,j)<->(j,i) scatter as
        # cellA (W_Aonly symmetric in dl<->dlp).
        # Chunked: the direct form built `ebare` plus `ebare*term_dL` and
        # `ebare*term_dL**2` -- five full-length arrays, 5 x 18.3 GB at
        # lmax = 90. term_dL is derived from plan['term_L'] per block.
        eta_lm = pysh.shio.SHCilmToVector(eta_clm.coeffs)
        cell_g0, cell_g1, cell_g2 = cell_sums_eta_dL(
            eta_lm, gidx, plan, starts, seg_len)
        dl_i = d_l2[ci.astype(np.int64)]
        dl_j = d_l2[cj.astype(np.int64)]
        Ssum = dl_i + dl_j
        W_eta_cell = -0.25*(((dl_i - dl_j)**2 + 2.0*Ssum - 8.0)*cell_g0
                            + (2.0 - 2.0*Ssum)*cell_g1
                            + cell_g2)
 
        # Vectorised scatter (was a 34-million-iteration Python loop at
        # lmax = 90). [i,j] and [j,i] get the same value, so i == j is fine.
        a = np.zeros((N, N))
        b = np.zeros((N, N))
        fac_a = (Re/T_e_0)**3 / E
        _va =  fac_a * W_eta_cell
        a[ci, cj] = _va;  a[cj, ci] = _va
        b[ci, cj] = -W_eta_cell;  b[cj, ci] = -W_eta_cell
 
    
        # ------------------------------------------------------------------
        # Group-2 LHS operator  Te^2 . Delta( X . w ):
        #     1d: X = Te          1e: X = Tcind        1f: X = gTe*max
        #     dc pair: X = max . Pw   (Pw = phi^-(l'+2) on the OPERAND degree)
        # ALL factors and inner field vectors now come from Omega_eq1_LHS's
        # g2 dict -- there is no second copy here. (Two copies of the same
        # physics is what produced the historical f_dc2 inconsistency.)
        # The returned matrix is generally NON-symmetric, as expected for an
        # ordered  conv . Lap . conv  operator; it is added straight into A.
        # NOTE: build_A_tilde_group2 forms dense N x N convolution matrices
        # and matmuls (~O(N^3)); a few seconds at lmax~45.
        # ------------------------------------------------------------------
        # STEP 6: the only per-degree diagonal solve_beuthe still needs is Pw
        # (phi^-(l'+2) on the operand degree). Build it once from g2's RTcR.
        # DC FILTER: phi**-(l+2) IS 1/P2, the eq-(1) elimination -> x F1.
        DCfilter_mohoD = DCfilters(lmax)[0]
        Pw_diag = np.diag(np.array([g2['RTcR']**(-(l+2)) * DCfilter_mohoD[l]
                                    for l, _ in mode_map]))
        _use_dc = (solve_for == 'dc_lm')

        # dc branch: the reinstated drhol dc coupling (zero iff rho_l==rho_c).
        # drho branch: dc = 0, so that pair does not exist -- instead the -w
        # half of omega's (dc-w) term STANDS ALONE (fdrho_w1/w2, field = max,
        # no operand weight -> Pw = I). Its factors are exactly the historical
        # commented-out dc1/dc2 values: correct here, wrong in the dc branch.
        A_tilde_group2 = build_A_tilde_group2(
            g2['Te_unstr'], g2['Te2_unstr'], g2['max_unstr'],
            g2['f1d'], g2['f1e'], g2['f1f'],
            gidx, plan['term_gaunt_bare'], starts, seg_len, ci, cj, mode_map, N,
            fdc1_w=(g2['fdc1_w'] if _use_dc else g2['fdrho_w1']),
            fdc2_w=(g2['fdc2_w'] if _use_dc else g2['fdrho_w2']),
            Pw=(Pw_diag if _use_dc else np.eye(N)),
            Tcind_unstr=g2['Tcind_unstr'], gTemax_unstr=g2['gTemax_unstr'])


        # ETA FIELD FIX
        # ---- ETA-FIELD (Path 1) -------------------------------------------
        # Beuthe writes  eta * [ the whole Omega operator ], so eta is applied
        # ONCE here to the assembled Omega blocks rather than being threaded
        # into each individual field (which is ambiguous for terms whose c1 and
        # c2 halves share a field). eta is evaluated at the LOCAL Te.
        #   eta * (A_omega @ w)  =  (conv(eta) @ A_omega) @ w
        # At constant Te, eta_grid is a monopole equal to eta0, so
        # C_eta = eta0 * I and every constant-Te benchmark is preserved exactly.
        _Te_grid_eta = T_e_parent.expand(lmax=3*lmax).data
        _Re_grid_eta = R - _Te_grid_eta/2.0
        eta_grid_sb  = 1.0/(1.0 + _Te_grid_eta**2/(12.0*_Re_grid_eta**2))
        eta_clm_sb   = pysh.SHGrid.from_array(eta_grid_sb).expand()
        eta_unstr    = pysh.shio.SHCilmToVector(
                         pysh.SHCoeffs.from_array(
                           eta_clm_sb.coeffs[:, :2*lmax+1, :2*lmax+1]).coeffs)
        C_eta = build_conv_matrix(eta_unstr, gidx, plan['term_gaunt_bare'],
                                  starts, seg_len, ci, cj, N)

        # eq-1 Omega LHS block = A_tilde + A_tilde_group2  ->  eta * (that)
        A = A + C_eta @ (A_tilde + A_tilde_group2)


        # ---- q's w-coupling: LHS diagonal, branch-dependent ---------------
        # NOTE: Lam_q is NOT an Omega term -- it comes from q -- so it is added
        # AFTER the C_eta multiplication above and must NOT be wrapped by C_eta.
        # dc branch  : Lam_q = qH with rho_l -> drhol   (zero iff rho_l==rho_c)
        # drho branch: Lam_q_drho (built in Omega_eq1_LHS) -- nonzero even at
        #              rho_l == rho_c, since the g_m*drho term survives.
        # Both enter via  -Re^4*K1*q  =>  A_diag += Re^4*K1*Lam_q.
        # STEP 3: both Lam_q arrays are precomputed in Omega_eq1_LHS. solve_beuthe
        # only applies the branch-appropriate one to the diagonal.
        Lam_q_arr = g2['Lam_q_dc'] if _use_dc else g2['Lam_q_drho']
        for idx, (l_m, _) in enumerate(mode_map):
            A[idx, idx] += Re**4 * scaler_A * Lam_q_arr[l_m]

        # ---- drho branch: omega's drhom term  P_hat * drhom / R -----------
        # conv(P_hat) @ diag(w_corr): P_hat is a genuine Te-dependent field ->
        # convolution; w_corr is a per-degree scalar -> diagonal on the OPERAND.
        if not _use_dc:
            Dw_diag = np.diag(np.array([g2['w_corr'][l] for l, _ in mode_map]))
            C_Phat  = build_conv_matrix(g2['Phat_unstr'], gidx,
                                        plan['term_gaunt_bare'],
                                        starts, seg_len, ci, cj, N)
            C_Te2   = build_conv_matrix(g2['Te2_unstr'], gidx,
                                        plan['term_gaunt_bare'],
                                        starts, seg_len, ci, cj, N)
            Lap_d   = np.diag(np.array([-l*(l+1) for l, _ in mode_map],
                                       dtype=np.float64))
            PhatDw  = C_Phat @ Dw_diag
            # A = A + g2['fdrho_om1'] * PhatDw
            # A = A + g2['fdrho_om2'] * (C_Te2 @ Lap_d @ PhatDw)

            # ETA FIELD FIX : these are eq-1 Omega LHS terms -> carry conv(eta).
            A = A + C_eta @ (g2['fdrho_om1'] * PhatDw)
            A = A + C_eta @ (g2['fdrho_om2'] * (C_Te2 @ Lap_d @ PhatDw))


        # b_tilde: non-symmetric matrix form (see b_tilde FIX above):
        # diag(Delta'_out) @ conv(2a + 2b + 2c fields). Reduces exactly to
        # the former symmetric-scatter result for monopole omega-fields
        # (constant Te), and is correct for laterally varying Te.
        C_2abc  = build_conv_matrix(fields_2sum, gidx, plan['term_gaunt_bare'],
                                    starts, seg_len, ci, cj, N)
        b_tilde = np.diag(d_l2) @ C_2abc
 
        # (The dc-branch Lam_q diagonal is applied ONCE, in the branch-gated
        #  block above. A second, identical copy used to live here -- it made
        #  A carry 2*Lam_q + fdc instead of Lam_q + fdc, i.e. exactly one
        #  extra Lam_q. Invisible at rho_l == rho_c, where Lam_q = 0.)

        # drhol EXTENSION -- eq-2 omega dc coupling (zero iff rho_l == rho_c):
        # the same + v1v*g_m*drhol*max*P_l'*w/R content must enter b_tilde,
        # pattern of 2b/2c: field = max(Te-Tc,0)*(eta*alpha), output Delta',
        # operand weight Pw. Built as an explicit (non-symmetric) matrix
        # product to carry the output/operand weights on the correct sides.
        # STEP 5: eq-2 dc coupling. The max*alpha field and the fdc_2d factor are
        # now supplied by Omega_eq2_LHS / g2 -- solve_beuthe only forms the
        # ordered matrix product  Delta'_out @ conv(max*alpha) @ Pw(operand).
        if _use_dc and g2['fdc_2d'] != 0.0:
            C_maxa = build_conv_matrix(maxa_raw_unstr, gidx, plan['term_gaunt_bare'],
                                       starts, seg_len, ci, cj, N)
            Dlp_out = np.diag(d_l2)                 # Delta' at OUTPUT degree
            b_tilde = b_tilde + g2['fdc_2d'] * (Dlp_out @ C_maxa @ Pw_diag)
 
        # drho branch: omega's drhom w-coupling in eq 2:
        #   (1-nu)*scaler2 * Delta'_out @ conv(P_hat*alpha) @ diag(w_corr)
        # P_hat*alpha is a genuine field (convolution); w_corr is per-degree
        # (diagonal on the OPERAND). Non-symmetric by construction.
        if not _use_dc:
            Dw_diag_2 = np.diag(np.array([g2['w_corr'][l] for l, _ in mode_map]))
            C_Phata   = build_conv_matrix(Omega_LHS_2_Phata_unstr, gidx,
                                          plan['term_gaunt_bare'],
                                          starts, seg_len, ci, cj, N)
            b_tilde = b_tilde + np.diag(d_l2) @ C_Phata @ Dw_diag_2

        # b = b + b_tilde
    
        # ETA FIELD FIX : b_tilde is the assembled eq-2 Omega LHS operator (2a/2b/2c
        # plus the dc and drho couplings above) -> apply conv(eta) once.
        b = b + C_eta @ b_tilde        # was:  b = b + b_tilde
        
        # assemble 2N x 2N dense system
        M = np.zeros((2*N, 2*N))
        M[:N, :N]   = A
        M[N:, N:]   = B
        M[:N, N:]   = a
        M[N:, :N]   = b
 
        # pin degree 0 and 1 (rigid-body / translation freedom)
        for idx,(l,_) in enumerate(mode_map):
            if l in (0,1):
                M[idx, :] = 0.0
                M[idx, idx] = 1.0
                M[idx+N, :] = 0.0
                M[idx+N, idx+N] = 1.0
 

        lu, piv = lu_factor(M)
        if cache is not None:
            cache['lu'], cache['piv'], cache['C_eta'] = lu, piv, C_eta
            # Keep the assembled operator for the coupling diagnostics. Not
            # used by the solve -- lu/piv already contain everything needed --
            # purely so the Te-coupling structure can be inspected afterwards.
            cache['M'], cache['N'], cache['mode_map'] = M, N, mode_map            

    # Build the RHS (loads), changes with every iteration due to correction 
    # terms and therefore not cached!
    q_lm_unstr = q_lm(topo_clm, geoid_clm, 
                      lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                      g0=g0, mass=mass)
    Omega_RHS1_unstr = Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, 
                                     lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                     g0=g0, mass=mass)
    Omega_RHS2_unstr = Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_clm,
                                     lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                     g0=g0, mass=mass)

    # =============== the omega correction hook ==================
    # drho_omega_corr is an additive term on DSP eq (5), and M5 keeps eq (5)
    # by substituting its content into BOTH Omega RHS builders -- so the
    # correction must enter BOTH, through the same operator the builders
    # apply to their own omega content (see omega_corr_to_rhs).
    #
    # PLACEMENT IS LOAD-BEARING: this addition happens BEFORE the
    # `C_eta @ Omega_RHS*` lines below, so the correction picks up the eta
    # field exactly as the rest of the Omega RHS does. Moving it after them
    # would drop eta from the correction alone -- invisible at constant Te
    # (where C_eta = eta0*I merely rescales) and wrong at variable Te.
    if omega_corr_phys is not None:
        _o1, _o2 = omega_corr_to_rhs(omega_corr_phys, T_e_parent, a_clm,
                                     lmax, R, T_e_0, Re)
        Omega_RHS1_unstr = Omega_RHS1_unstr + _o1
        Omega_RHS2_unstr = Omega_RHS2_unstr + _o2

    def elem(l,m,v):
        off = 0 if m==0 else (m if m>0 else l+abs(m))
        return v[l*l+off]
    
    q = np.array([elem(l,m,q_lm_unstr) for l,m in mode_map])
    Omega_RHS1 = np.array([elem(l,m,Omega_RHS1_unstr) for l,m in mode_map])
    Omega_RHS2 = np.array([elem(l,m,Omega_RHS2_unstr) for l,m in mode_map])
    
    # ETA FIELD FIX : the Omega RHS vectors are the same Omega operator acting on
    # the KNOWN fields (H, G), so eta multiplies them too. In spectral form
    # that is the same convolution: eta*Omega_RHS = C_eta @ Omega_RHS.
    # (q is NOT an Omega term and is deliberately left untouched.)
    Omega_RHS1 = C_eta @ Omega_RHS1
    Omega_RHS2 = C_eta @ Omega_RHS2
 
    # ============== the q correction hook =======================
    # q_lm() returns the SCALED vector q = -Re^4 * q_phys / (E*Te0^3), so a
    # physical correction in Pa carries the same constant. No degree factor.
    # This single hook is all the framework needs: every correction in the
    # thin-shell system is an additive term on the right-hand side.
    if q_corr_phys is not None:
        q_correction = pysh.shio.SHCilmToVector(-Re**4/(E*T_e_0**3) * q_corr_phys)
        q_correction = np.array([elem(l, m, q_correction) for l, m in mode_map])
        q = q + q_correction

    y1 = q + Omega_RHS1
    y2 = Omega_RHS2

    
    rhs = np.concatenate([y1, y2])
    for idx,(l,_) in enumerate(mode_map):
        if l in (0,1): rhs[idx] = 0.0; rhs[idx+N] = 0.0
 
    sol = lu_solve((lu, piv), rhs)
    w_sol = sol[:N]
    F_sol = sol[N:]
 
    w_coeffs = np.zeros((2, lmax+1, lmax+1))
    F_coeffs = np.zeros((2, lmax+1, lmax+1))
    q_coeffs = np.zeros((2, lmax+1, lmax+1))
    for idx,(l,m) in enumerate(mode_map):
        if m >= 0: 
            w_coeffs[0,l,m]     = w_sol[idx]
            F_coeffs[0,l,m]     = F_sol[idx]
            q_coeffs[0,l,m]     = q[idx]
        else:      
            w_coeffs[1,l,abs(m)] = w_sol[idx]
            F_coeffs[1,l,abs(m)] = F_sol[idx]
            q_coeffs[1,l,abs(m)] = q[idx]
    return (pysh.SHCoeffs.from_array(w_coeffs, normalization='4pi'), 
            pysh.SHCoeffs.from_array(F_coeffs, normalization='4pi'),
            pysh.SHCoeffs.from_array(q_coeffs, normalization='4pi'),)



# %% MAIN LOOP & PLOTTING

if __name__ == "__main__":
    t_begin = time.perf_counter()
    
    # Perform some selftests to see if the gaunt code runs correctly
    selftest_gaunt()
    selftest_term_weights()
    
    # Load in the observables and Te map
    topo_p, geoid_p, T_e_parent, R, g0, mass = load_inputs(LMAX_REF, strain=strain)
    T_e_0 = T_e_parent.coeffs[0,0,0]
    print(f'T_e_0 = {T_e_0/1e3:.2f} km')
    D_clm, a_clm, D_eta_clm, a_eta_clm, eta_clm = derive_D_a(T_e_parent, LMAX_REF)

    # Trigger a warning if Tc < min(Te), because this case causes a discontinuous
    # jump in the term rho_grid and Tcind_grid. This jump can cause some ringing
    if np.min(T_e_parent.expand().data) < T_c:
        print(f"! WARNING: The elastic thickness map is shallower than the "
              f"crustal thickness with Tc < min(Te) = {T_c} < "
              "{np.min(T_e_parent.expand().data)}, which causes a discontinuity "
              "in some Omega terms. This may cause some ringing in the results")

    # downward-continuation filter: report it, and refuse to run the branch
    # it is not wired into 
    if filter_type is not None:
        (DCfilter_mohoD, DCfilter_mohoDc,
         DCfilter_drhom, DCfilter_drhomc) = DCfilters(LMAX_REF)
        
        _F1, _F2 = ((DCfilter_mohoD, DCfilter_mohoDc) if solve_for == 'dc_lm'
                    else (DCfilter_drhom, DCfilter_drhomc))
        _iface   = ('moho R_c = R - c' if solve_for == 'dc_lm'
                    else 'R_base_drho = R - base_drho')
        _name = {'Ma': 'minimum amplitude', 'Mc': 'minimum curvature'}[filter_type]
        print(f"  Downward-continuation filter: {_name} ('{filter_type}'), "
              f"half = {filter_half}, interface = {_iface}")
        print(f"    eq (1)  F1: l=2 {_F1[2]:.4f}   "
              f"l={min(filter_half, LMAX_REF)} "
              f"{_F1[min(filter_half, LMAX_REF)]:.4f}   "
              f"l={LMAX_REF} {_F1[LMAX_REF]:.4f}")
        
        if np.allclose(_F2, 1.0):
            print("    eq (2)  F2: all ones -- D_relief lies ABOVE R_ref, so "
                  "eq (2) continues UPWARD and needs no filter")
        else:
            print(f"    eq (2)  F2: l=2 {_F2[2]:.4f}   l={LMAX_REF} "
                  f"{_F2[LMAX_REF]:.4f}")
        # HOW MUCH SHOULD THE ANSWER MOVE? The filter multiplies the RECOVERED
        # interface degree by degree, so this table is the expected size of the
        # filter-on / filter-off difference. It is the honest scale to judge a
        # run against: at half = 50 with lmax = 45 the filter is ~1 below l~20
        # and only reaches ~0.6 at the band edge, so most configurations should
        # show a SMALL change in the map -- a large one would be the surprise.
        # If a filter-on/off comparison moves the solution far more (or far
        # less) than this, check filter_half and the interface depth first:
        # the strength goes as (R/D_relief)**l, so a deeper interface is
        # filtered far harder.
        print("    expected |change| in the recovered field, per degree:")
        print("      " + "  ".join(f"l={l}: {100*(1-_F1[l]):4.1f}%"
                                   for l in (2, 10, 20, 30, LMAX_REF)
                                   if l <= LMAX_REF))
    else:
        print("  Downward-continuation filter: OFF (filter_type = None)")


    solutions_w = {}
    solutions_g_ms = {}   # converged mass-sheet geoid
    solutions_F = {}
    solutions_q = {}
    # Store the omega correction the solve used, required for final omega calc
    solutions_omega_corr = {}   
    
    for lmax_run in LMAX_RUNS:
        topo_clm  = truncate(topo_p,  lmax_run)
        geoid_clm = truncate(geoid_p, lmax_run)
        plan  = build_or_load_gaunt(lmax_run, nu)        
                
        do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
        for rotation in ([0, 1] if do_rotation_check else [0]):
            if rotation == 1:
                T_e_use, D_use, a_use, topo_use, geoid_use = rotate_inputs(
                    rotate_angles, T_e_parent, D_clm, a_clm, 
                    topo_clm, geoid_clm)
                
                alpha, beta, gamma = rotate_angles
                D_eta_use  = D_eta_clm.rotate(alpha, beta, gamma)
                a_eta_use  = a_eta_clm.rotate(alpha, beta, gamma)
                eta_use    = eta_clm.rotate(alpha, beta, gamma)

            else:
                T_e_use, D_use, a_use, topo_use, geoid_use = (
                    T_e_parent, D_clm, a_clm, topo_clm, geoid_clm)
                D_eta_use, a_eta_use, eta_use = D_eta_clm, a_eta_clm, eta_clm
                
            # ====================== The iterating loop =======================
            ### INITIALIZATION
            print('Start solving of system')
            # The constant-grid cache holds topo/geoid/Te grids for one input
            # set. A rotated pass feeds different fields, so it must start
            # clean: cached grids need to be cleared
            _GRID_CACHE.clear()
            t = time.perf_counter()
            cache = {}                      # matrix assembled once, reused
            
            ### INITALIZING CORRECTION TERMS
            # The omega correction starts at zero, so iteration 1 is
            # identical to M5 whatever else is switched on.
            omega_corr_phys = None

            # In the pure finite-amplitude case H_corr depends only on H, a known
            # input, so it has no feedback and no Picard gain. Once the surface 
            # density gate is live H_corr depends on drho, which depends on w, 
            # so it is rebuilt inside the loop and acquires a gain of its own.
            # The value built here is the pure-FA seed used on iteration 1 and 
            # whenever not solving for drho_lm.
            H_corr = interface_geoid_correction('H', R, mass, lmax_run,
                                            grid_expansion_res, nmax, 
                                            topo_clm=topo_use)
            
            # w_corr and wdc_corr are rebuilt each iteration from the solution; on the
            # first pass they are zero, so iteration 1 is the plain M5 solve.
            w_corr   = np.zeros_like(H_corr)
            wdc_corr = np.zeros_like(H_corr)
            
            # Geoid corrections are built from the three density interface
            # corrections:
            corrG, corrGc = geoid_corrections(H_corr, w_corr, wdc_corr, R, lmax_run)
            geoid_MS      = geoid_mass_sheet(geoid_use, corrG, lmax_run)
            q_corr_phys   = q_correction_from_geoid(corrG, corrGc, R, g0, mass)
            
            
            ### Some initializations for the iterations (comparisons, delta's)
            RMS = lambda a: np.sqrt(np.mean(np.asarray(a)[:, 2:lmax_run+1, :]**2))
            # To compare the input observed geoid with the calculated mass-sheet
            G_obs  = truncate(geoid_use, lmax_run).coeffs 
            w, F, q    = None, None, None 
            track_prev = None
            
            # init the comparison increment array and scalar delta
            increment_prev   = None
            delta_last = None

            # Store the previous iter arrays, for the damping
            hist_d = {1: None, 2: None} 
            
            for it in range(1, (iter_max if iterate else 1) + 1):
                w, F, q = solve_beuthe(topo_use, geoid_MS, T_e_use, D_use, a_use,
                                       plan, lmax=lmax_run, R=R, T_e_0=T_e_0,
                                       g0=g0, mass=mass, D_eta_clm=D_eta_use,
                                       a_eta_clm=a_eta_use, eta_clm=eta_use,
                                       q_corr_phys=q_corr_phys,
                                       omega_corr_phys=omega_corr_phys,
                                       cache=cache)
                if not iterate:
                    break

                # Copmute the recovered interface field dc_lm or drho_lm.
                # Needed by the delta-tracker and by the corrections. 
                # It uses the geoid_MS that this iteration's w was
                # actually solved with, because both compute_dc and
                # compute_drho invert eq (1), and eq (1) is the equation
                # geoid_MS encodes the correction into.
                dc_it = (compute_dc(w, topo_use, geoid_MS, R, lmax_run,
                                    grid_expansion_res)
                         if solve_for == 'dc_lm'
                         else pysh.SHCoeffs.from_zeros(lmax_run))
                drho_it = (compute_drho(w, topo_use, geoid_MS, R, lmax_run,
                                        grid_expansion_res)
                           if solve_for == 'drho_lm'
                           else pysh.SHCoeffs.from_zeros(lmax_run))


                # Convergence check. Tracks dc-w; the crust-mantle relief.
                # As of now always, when solving for drho_lm, so dc_lm=0, the 
                # delta is pure w. Maybe change to the delta in drho_lm.
                # Breaking here leaves geoid_MS and q_corr_phys as the ones this
                # w was solved with, so the stored pair is consistent with iter.
                track = (dc_it - w).expand(lmax=grid_expansion_res).data
                
                if track_prev is not None:
                    # calculate difference in tracker with prev iter
                    increment = track - track_prev
                    # Take delta as the absolute maximum of the increment
                    delta = np.abs(increment).max()

                    # Sign of the dominant eigenvalue. The gain gives |lambda|;
                    # this gives its sign, which decides whether under-relaxation
                    # can help when |lambda| > 1:
                    #   lambda < 0 -> oscillatory, relax with omega ~ 2/(1+|lambda|)
                    #   lambda > 0 -> monotonic, no omega > 0 helps
                    # Must lie in [-1, +1]; anything outside means this is broken.
                    if increment_prev is not None:
                        RSS_cur = np.sqrt(np.sum(increment**2))
                        RSS_prev = np.sqrt(np.sum(increment_prev**2))
                        inc_ratio = (np.sum(increment*increment_prev)/(RSS_cur*RSS_prev) 
                               if RSS_cur*RSS_prev > 0 else np.nan)
                        print(f'      increment correlation = {inc_ratio:+.4f}'
                              f'   (+1 => lambda>0, -1 => lambda<0)')
                    increment_prev = increment.copy() # Set the comparison increment

                    # Calculate the iteration gain by comparing delta and delta_last
                    iter_gain = f'{delta/delta_last:.6f}' if delta_last else '  ----  '
                    print(f'  iter {it}: delta = {delta:.4e} m       gain (delta/delta_last) = {iter_gain}')
                    delta_last = delta
                    if delta < delta_max:
                        print(f'  converged in {it} iterations')
                        break
                    if delta > delta_out:
                        print(f'  ! DIVERGING at iteration {it} - stopping')
                        break
                track_prev = track

                gate_H, gate_dc, gate_w = density_gates()
                # Iteration damping scheme, same as DSP, from Wieczorek et al. (2013)
                # Only damp every third iteration, taking the average of the previous two
                _damp_now = (damp and it % 3 == 0
                             and hist_d[1] is not None and hist_d[2] is not None)
 
                # init the drho-branch q and omega correction-term coefficients
                drho_q  = np.zeros_like(q.coeffs)
                drho_om = np.zeros_like(q.coeffs)
 
                if _damp_now:
                    drho_q, drho_om, H_corr, w_corr, wdc_corr = (
                        0.5*(hist_d[1][k] + hist_d[2][k]) for k in range(5))
                    print('      (DAMPING THIS ITER!: every third correction is the '
                          'average of the last two iterations)')
                else:
                    w_corr   = interface_geoid_correction('w', R, mass, lmax_run,
                                                          grid_expansion_res, nmax,
                                                          w_clm=w)
                    wdc_corr = interface_geoid_correction('wdc', R, mass, lmax_run,
                                                          grid_expansion_res, nmax,
                                                          w_clm=w, dc_clm=dc_it)
 
                    # ---- lateral density: only where a gate is live --------
                    if solve_for == 'drho_lm':
 
                        # q and omega: sum over the live interfaces. Note the
                        # OBSERVED geoid, not geoid_MS: DSP forms (H - G) from
                        # H_lm_o - G_lm_o where G is an input constraint, so
                        # this term is evaluated directly rather than through
                        # an inversion of eq (1).
                        for _interface, _live in (('H', gate_H), ('dc', gate_dc),
                                                  ('w', gate_w)):
                            if not _live:
                                continue
                            _q_part, _om_part = drho_q_omega_correction(
                                _interface, drho_it, topo_use, geoid_use, w,
                                T_e_use, R, lmax_run, g0)
                            drho_q  += _q_part
                            drho_om += _om_part
 
                        # Geoid corrections: each REPLACES its pure-FA
                        # counterpart where its gate is live, because
                        # corr_nmax_drho returns finite amplitude AND lateral
                        # density from one call. Summing them would subtract
                        # the mass sheet twice.
                        _geo = lambda face: interface_geoid_correction(
                                face, R, mass, lmax_run, grid_expansion_res,
                                nmax, topo_clm=topo_use, w_clm=w, dc_clm=dc_it,
                                drho_clm=drho_it)
                        if gate_H:
                            H_corr   = _geo('H')
                        if gate_dc:
                            wdc_corr = _geo('wdc')
                        if gate_w:
                            w_corr   = _geo('w')
 
                # ---- ONE history, ONE kernel rebuild, both branches --------
                # This is the part that must NOT sit inside the solve_for
                # conditional: without it the dc_lm branch never updates its
                # right-hand side and the loop has nothing to converge on.
                hist_d[2 if it % 2 == 0 else 1] = (
                    drho_q.copy(), drho_om.copy(),
                    H_corr.copy(), w_corr.copy(), wdc_corr.copy())
 
                corrG, corrGc   = geoid_corrections(H_corr, w_corr, wdc_corr,
                                                    R, lmax_run)
                geoid_MS        = geoid_mass_sheet(geoid_use, corrG, lmax_run)
                q_corr_phys     = q_correction_from_geoid(corrG, corrGc,
                                                          R, g0, mass) + drho_q
                # None rather than a zero array: solve_beuthe skips the omega
                # hook entirely when it is None, saving four full-resolution
                # transforms per iteration in the dc_lm branch.
                omega_corr_phys = drho_om if solve_for == 'drho_lm' else None

                print(f'      RMS(corrG)/RMS(G_obs) = {RMS(corrG)/RMS(G_obs):9.6f}'
                      '       RMS H_corr/w_corr/wdc_corr = '
                      f'{RMS(H_corr):.4f} / {RMS(w_corr):.4f} / {RMS(wdc_corr):.4f} m')
                
                if solve_for == 'drho_lm':
                    # Zero in dc_lm branch
                    _scaling_q    = -(R - T_e_0/2.0)**4/(E*T_e_0**3)
                    _ratio_q      = RMS(drho_q) / (RMS(q.coeffs)/np.abs(_scaling_q))
                    _omega_H_term = (nu/(1.0-nu))*rho_l*g0*T_e_0/R * RMS(
                                           truncate(topo_use, lmax_run).coeffs)
                    RMS_drho_it   = np.sqrt(np.mean(drho_it.coeffs[:, 2:, :]**2))
                    print(f'      [drho gates H/dc/w = {int(gate_H)}/{int(gate_dc)}/{int(gate_w)}] (1=On)'
                          f'    |q_corr|/|q| = {_ratio_q:.4f}'
                          f'    |om_corr|/|om_H| = {RMS(drho_om)/_omega_H_term:.4f}'
                          f'    RMS drho = {RMS_drho_it:.4f} kg/m3')


            print(f'Finished solving in {(time.perf_counter()-t):.1f}s ({it} iters)\n')
            
            solutions_w[lmax_run, rotation]    = w
            solutions_g_ms[lmax_run, rotation] = geoid_MS
            solutions_F[lmax_run, rotation]    = F
            solutions_q[lmax_run, rotation]    = q
            solutions_omega_corr[lmax_run, rotation] = omega_corr_phys



# %% PLOTS - INPUTS TOPOGRAPHY, Te, D AND alpha


    # Plot inputs topography, Te, D and alpha
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(13,8))
    do_rotation_check = any(angle != 0.0 for angle in rotate_angles)
    if do_rotation_check:
        T_e_use_clm, D_use_clm, a_use_clm, topo_use_clm, _ = rotate_inputs(
            rotate_angles, T_e_parent, D_clm, a_clm, 
            topo_clm, geoid_clm)
    else:
        T_e_use_clm, D_use_clm, a_use_clm, topo_use_clm = T_e_parent, D_clm, a_clm, topo_clm
    
    args_plot = dict(tick_interval=[45, 30])

    topography_km = topo_use_clm.expand(lmax=grid_expansion_res)
    topography_km.data = (topography_km.data - R)/1e3
    topo_min, topo_max = topography_km.data.min(), topography_km.data.max()
    cmap_limits_topo_diff =[topo_min, 10]
    topography_km.plot(ax=ax0, 
                       cmap=cmc.navia,
                       cmap_limits = cmap_limits_topo_diff,
                       grid=True,
                       colorbar='right', 
                       cb_label='Topographic height [km]',
                       **args_plot)
    ax0.set_title(f'M5 - MOLA topography map, exp. to lmax={LMAX_REF}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    
    T_e_parent_km = T_e_use_clm.expand(lmax=grid_expansion_res)
    T_e_parent_km.data = T_e_parent_km.data/1e3
    T_e_parent_km.plot(ax=ax1, 
                       ticks = 'wSne',
                       ylabel=None,
                       grid=True,
                       cmap=cmc.lajolla, 
                       colorbar='right', 
                       cb_label=r'$T_e \ [km]$',
                       **args_plot)
    ax1.set_title(f'M5 - Te input map (Plesa et al. 2018), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    
    D_use_clm.expand(lmax=grid_expansion_res).plot(ax=ax2, 
                                        cmap=cmc.lajolla, 
                                        colorbar='right', 
                                        cb_label=r'$D \ [N\cdot m]$',
                                        **args_plot)  
    ax2.set_title(f'M5 - Flexural rigidity D (Te-derived), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    a_use_clm.expand(lmax=grid_expansion_res).plot(ax=ax3, 
                                        cmap=cmc.lajolla, 
                                        colorbar='right', 
                                        cb_label=r'$\alpha \ [m/N$]',
                                        **args_plot) 
    ax3.set_title(f'M5 - Parameter alpha (Te-derived), exp. to lmax={lmax_Te_fit}'
                  + (f', rot={rotate_angles}' if rotation else ''))
    plt.suptitle('M5 - Input maps topography & Te, and derived parameters D and $\\alpha$')
    plt.tight_layout()
    if SaveFigs:
        plt3_title = (f'M5 - Inputs Te, D and alpha, lmax={LMAX_REF}, '
                      f'lmaxTe={lmax_Te_fit}'
                      + (f', rotated {rotate_angles}' if rotation else '') 
                      + '.png')
        FigPath3 = os.path.join(SavePath, plt3_title)
        plt.savefig(FigPath3, dpi=200)
    plt.show(); plt.close()
    
  
# %% PLOTS - DSP-M5 w_lm, dc_lm, drho_lm, Tc_lm RESIDUAL PLOTS
        
    # Set whether to include crustal thickness in the plots
    show_Tc = True         
    args_expand = dict(lmax=grid_expansion_res, lmax_calc=LMAX_REF)
    args_plot = dict(tick_interval=[45, 30], grid=True)

    
    w_fine = pysh.SHGrid.from_array(
            solutions_w[LMAX_REF, 0].expand(**args_expand).data/1e3)
    w_clm = solutions_w[LMAX_REF, 0]
    geoid_ref = solutions_g_ms[LMAX_REF, 0]
    dc_clm = compute_dc(w_clm, topo_clm, geoid_ref, R=R, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    dc_clm_zeroed = dc_clm.copy()
    dc_clm_zeroed.coeffs[0,0,0] = 0
    dc_grid = dc_clm_zeroed.expand(**args_expand)/1e3
    
    drho_clm = compute_drho(w_clm, topo_clm, geoid_ref, R=R, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    drho_clm_zeroed = drho_clm.copy()
    drho_clm_zeroed.coeffs[0,0,0] = 0
    drho_grid = drho_clm_zeroed.expand(**args_expand)
    
    topo_grid = topo_clm.expand(**args_expand)/1e3 - R/1e3
    
    T_c_grid = (topo_grid.data 
                + (dc_grid.data if solve_for=='dc_lm' else 0)
                - w_fine.data 
                + T_c*np.ones((2*(grid_expansion_res+1)+1, 4*(grid_expansion_res+1)+1))/1e3)
    T_c_grid = pysh.SHGrid.from_array(T_c_grid)
    




    # Compute residuals between DSP and M5 spatially
    grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    w_diff_DSPM5 = grid_w_DSP.copy()
    w_diff_DSPM5.data = grid_w_DSP.data - w_fine.data
    
    grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
    dc_diff_DSPM5 = grid_dc_DSP.copy()
    dc_diff_DSPM5.data = grid_dc_DSP.data - dc_grid.data
    
    grid_drho_DSP = pysh.SHCoeffs.from_array(drho_DSP.coeffs).expand(**args_expand)
    drho_diff_DSPM5 = grid_drho_DSP.copy()
    drho_diff_DSPM5.data = grid_drho_DSP.data - drho_grid.data
    
    grid_Tc_DSP = pysh.SHCoeffs.from_array(Tc_DSP.coeffs / 1e3).expand(**args_expand)
    Tc_diff_DSPM5 = grid_Tc_DSP.copy()
    Tc_diff_DSPM5.data = grid_Tc_DSP.data - T_c_grid.data
    
    # Compute residuals between DSP and M5 spectrally
    w_diff_DSPM5 = w_DSP - solutions_w[LMAX_REF,0]
    w_diff_DSPM5 = w_diff_DSPM5.expand(**args_expand)
    w_diff_DSPM5.data = w_diff_DSPM5.data / 1e3
    
    dc_diff_DSPM5 = dc_DSP - dc_clm_zeroed
    dc_diff_DSPM5 = dc_diff_DSPM5.expand(**args_expand)
    dc_diff_DSPM5.data = dc_diff_DSPM5.data / 1e3



    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    if show_Tc:
        h_space_outer = 0.3
        h_space_inner1 = 0.3
        h_space_inner2 = 0.3
        y_suptitle = 1.03
        rows=3
        cb_height = 0.06
        xticks1 = 'Wsen'
        xticks2 = 'wsen'
        xlabel = None
    else:
        h_space_outer = -0.15
        h_space_inner1 = -0.5
        h_space_inner2 = -0.35
        y_suptitle = 0.86
        rows=2
        cb_height = 0.03
        xticks1 = 'WSen'
        xticks2 = 'wSen'
        xlabel = 'Longitude'


    outer_gs = gridspec.GridSpec(rows, 1, hspace=h_space_outer)


    # --- ROW 1: Radial Displacement w ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
                                                 height_ratios=[1, cb_height], hspace=h_space_inner1, wspace=0.15)
    ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_w_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_w_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_w = [w_fine.data.min(), w_fine.data.max()]
    # cmap_limits_w = [-7, 3]
    
    w_min, w_max = w_diff_DSPM5.data.min(), w_diff_DSPM5.data.max()
    cmap_limits_w_diff =[-max(abs(w_min), abs(w_max)), max(abs(w_min), abs(w_max))]

    grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    grid_w_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_w, 
                    cmap=cmap3, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Radial displacement w', fontweight="bold")
    
    w_fine.plot(ax=ax2, 
                cmap_limits=cmap_limits_w, 
                cmap=cmap3, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M5 - Radial displacement w', fontweight="bold")
    

    w_diff_DSPM5.plot(ax=ax3, cmap=cmap2,
                      cmap_limits = cmap_limits_w_diff,
                      colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Radial displacement w residual DSP - M5', fontweight="bold")

    norm_w = mcolors.Normalize(vmin=cmap_limits_w[0], vmax=cmap_limits_w[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_w, 
                    cmap=cmap3), 
                       cax=cax_w_shared, orientation='horizontal')
    cb1.set_label('w [km]', fontweight="bold")
   
    norm_w_diff = mcolors.Normalize(vmin=cmap_limits_w_diff[0], vmax=cmap_limits_w_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(norm=norm_w_diff, cmap=cmap2), cax=cax_w_diff, orientation='horizontal')
    cb2.set_label('w [km]', fontweight="bold")

    ax1.contour(grid_w_DSP.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')
    ax2.contour(w_fine.data>0, levels=[0.99], extent=(0,360,-90,90), colors='k', origin='upper')



    if solve_for == 'dc_lm':
        # --- ROW 2: Crustal Root Variations ---
        inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                     height_ratios=[1, cb_height], hspace=h_space_inner2, wspace=0.15)
        ax4 = fig.add_subplot(inner_gs2[0, 0:2])
        ax5 = fig.add_subplot(inner_gs2[0, 2:4])
        ax6 = fig.add_subplot(inner_gs2[0, 4:6])
        
        cax_dc_shared = fig.add_subplot(inner_gs2[1, 1:3])
        cax_dc_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
        cmap_limits_dc = [-50, 30]

        dc_min, dc_max = dc_diff_DSPM5.data.min(), dc_diff_DSPM5.data.max()
        cmap_limits_dc_diff =[-max(abs(dc_min), abs(dc_max)), max(abs(dc_min), abs(dc_max))]
    
        grid_dc_DSP = pysh.SHCoeffs.from_array(dc_DSP.coeffs / 1e3).expand(**args_expand)
        grid_dc_DSP.plot(ax=ax4, cmap=cmap3, cmap_limits=[-50, 30], colorbar=None, ticks=xticks1, xlabel=xlabel, **args_plot)
        ax4.set_title('DSP - Crustal root variations', fontweight="bold")
        
        dc_grid.plot(ax=ax5, cmap=cmap3, cmap_limits=cmap_limits_dc, colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax5.set_title('M5 - Crustal root variations', fontweight="bold")
    
        dc_diff_DSPM5.plot(ax=ax6, cmap=cmap2, 
                           cmap_limits=cmap_limits_dc_diff, 
                           colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax6.set_title('Crustal root variations residual DSP - M5', fontweight="bold")
    
        norm_dc = mcolors.Normalize(vmin=cmap_limits_dc[0], vmax=cmap_limits_dc[1])
        cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_dc, cmap=cmap3), cax=cax_dc_shared, orientation='horizontal')
        cb3.set_label('$\\delta c$ [km]', fontweight="bold")
    
        norm_dc_diff = mcolors.Normalize(vmin=cmap_limits_dc_diff[0], vmax=cmap_limits_dc_diff[1])
        cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_dc_diff, cmap=cmap2), cax=cax_dc_diff, orientation='horizontal')
        cb4.set_label('$\\delta c$ [km]', fontweight="bold")


    if solve_for == 'drho_lm':
        # --- ROW 2: Internal density Variations ---
        inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                     height_ratios=[1, cb_height], hspace=h_space_inner2, wspace=0.15)
        ax4 = fig.add_subplot(inner_gs2[0, 0:2])
        ax5 = fig.add_subplot(inner_gs2[0, 2:4])
        ax6 = fig.add_subplot(inner_gs2[0, 4:6])
        
        cax_drho_shared = fig.add_subplot(inner_gs2[1, 1:3])
        cax_drho_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
        cmap_limits_drho = [-500, 500]
        
        drho_min, drho_max = drho_diff_DSPM5.data.min(), drho_diff_DSPM5.data.max()
        cmap_limits_drho_diff =[-max(abs(drho_min), abs(drho_max)), max(abs(drho_min), abs(drho_max))]
    
        grid_drho_DSP.plot(ax=ax4, cmap=cmap1, cmap_limits=cmap_limits_drho, colorbar=None, ticks=xticks1, xlabel=xlabel, **args_plot)
        ax4.set_title('DSP - Internal density variations', fontweight="bold")
        
        drho_grid.plot(ax=ax5, cmap=cmap1, cmap_limits=cmap_limits_drho, colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax5.set_title('M5 - Internal density variations', fontweight="bold")
    
        drho_diff_DSPM5.plot(ax=ax6, cmap=cmap2, 
                           cmap_limits=cmap_limits_drho_diff, 
                           colorbar=None, ticks=xticks2, xlabel=xlabel, ylabel=None, **args_plot)
        ax6.set_title('Internal density variations residual DSP - M5', fontweight="bold")
    
        norm_drho = mcolors.Normalize(vmin=cmap_limits_drho[0], vmax=cmap_limits_drho[1])
        cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_drho, cmap=cmap1), cax=cax_drho_shared, orientation='horizontal')
        cb3.set_label('$\\delta \\rho$ [kg/m$^3$]', fontweight="bold")
    
        norm_drho_diff = mcolors.Normalize(vmin=cmap_limits_drho_diff[0], vmax=cmap_limits_drho_diff[1])
        cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_drho_diff, cmap=cmap2), cax=cax_drho_diff, orientation='horizontal')
        cb4.set_label('$\\delta \\rho$ [kg/m$^3$]', fontweight="bold")

    # --- ROW 3: Crustal Thickness ---
    if show_Tc:
        inner_gs3 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[2], 
                                                     height_ratios=[1, cb_height], hspace=0.3, wspace=0.15)
        ax7 = fig.add_subplot(inner_gs3[0, 0:2])
        ax8 = fig.add_subplot(inner_gs3[0, 2:4])
        ax9 = fig.add_subplot(inner_gs3[0, 4:6])
        
        cax_tc_shared = fig.add_subplot(inner_gs3[1, 1:3])
        cax_tc_diff   = fig.add_subplot(inner_gs3[1, 4:6])
    
        Tc_min, Tc_max = Tc_diff_DSPM5.data.min(), Tc_diff_DSPM5.data.max()
        cmap_limits_Tc_diff =[-max(abs(Tc_min), abs(Tc_max)), max(abs(Tc_min), abs(Tc_max))]

        grid_Tc_DSP.plot(ax=ax7, cmap=cmap3, cmap_limits=[0, 110], colorbar=None, xlabel=None, ticks='WSen', **args_plot)
        ax7.set_title('DSP - Crustal thickness', fontweight="bold")
        
        T_c_grid.plot(ax=ax8, cmap=cmap3, cmap_limits=[0, 110], colorbar=None, ticks='wSen', xlabel=None, ylabel=None, **args_plot)
        ax8.set_title('M5 - Crustal thickness', fontweight="bold")
    
        tc_min, tc_max = Tc_diff_DSPM5.data.min(), Tc_diff_DSPM5.data.max()
        Tc_diff_DSPM5.plot(ax=ax9, cmap=cmap2, cmap_limits=cmap_limits_Tc_diff, colorbar=None, ticks='wSen', xlabel=None, ylabel=None, **args_plot)
        ax9.set_title('Crustal thickness residual DSP - M5', fontweight="bold")
            
        norm_tc = mcolors.Normalize(vmin=0, vmax=110)
        cb5 = fig.colorbar(cm.ScalarMappable(norm=norm_tc, cmap=cmap3), cax=cax_tc_shared, orientation='horizontal')
        cb5.set_label('$T_c$ [km]', fontweight="bold")
    
        norm_tc_diff = mcolors.Normalize(vmin=cmap_limits_Tc_diff[0], vmax=cmap_limits_Tc_diff[1])
        cb6 = fig.colorbar(cm.ScalarMappable(norm=norm_tc_diff, cmap=cmap2), cax=cax_tc_diff, orientation='horizontal')
        cb6.set_label('$T_c$ [km]', fontweight="bold")




    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle('Residual checks DSP and M5 - '
                 + (f'{filter_type} filtered at l={filter_half}\n' if filter_on else 'unfiltered fin amp\n')
                 + ('Solving for $\\delta \\rho_{lm}$, $\\delta c_{lm}$=0' if solve_for == 'drho_lm' else '')
                 + ('Solving for $\\delta c_{lm}$, $\\delta \\rho_{lm}$=0' if solve_for == 'dc_lm' else '')
                 + f' --- lmax={LMAX_REF}, nmax={nmax}, delta_max={delta_max}'
                 + f'\nDSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M5 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M5 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M5 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + (f'M5 $T_e$={Te_twoval_big/1e3:.0f}/{Te_twoval_sml/1e3:.0f} km two-value map' if strain=='twoval' else '')
                 + f'\nDSP & M5 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=y_suptitle, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = (f'Residuals_w_dcdrho_Tc_DSP_M5_lmax={LMAX_REF}_iter{iterate}_nmax={nmax}_delta_max={delta_max}_'
                + f'SolveFor{solve_for}_'
                + (f'Mb{Mb/1e3}_Mt{Mt/1e3}_' if solve_for == 'drho_lm' else '')
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + ('Te_M5=PlesaStrain14Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + ('Te_M5=PlesaStrain17Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + ('Te_M5={Te_twoval_big/1e3:.0f}/{Te_twoval_sml/1e3:.0f}twovalmap_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain=='twoval' else '')
                + f'Tc={T_c/1e3}km'
                + f'filter_on{filter_on}'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    
    
# %% PLOTS - STRESS AND STRAIN FIELDS


    w_clm = solutions_w[LMAX_REF, 0].expand(lmax=grid_expansion_res).expand()
    F_clm = solutions_F[LMAX_REF, 0].expand(lmax=grid_expansion_res).expand()
    q_clm = solutions_q[LMAX_REF, 0].expand(lmax=grid_expansion_res).expand()

    # compute_Omega now returns Beuthe's Omega = Re*omega (required by
    # cons_disp_S)
    # STEP 5c: the stress/strain fields were the OTHER consumer missing the
    # omega correction -- S_lm feeds every sigma and epsilon below.
    Omega_coeffs = compute_Omega(w_clm, T_e_parent, topo_clm, geoid_ref, q_clm,
                                 g0=g0, R=R, T_e_0=T_e_0, lmax_calc=LMAX_REF,
                                 lmax_grid=grid_expansion_res,
                                 omega_corr_phys=solutions_omega_corr.get((LMAX_REF, 0)))
    Omega_grid = Omega_coeffs.expand(lmax=grid_expansion_res)

    # S first (needed by the DSP-convention stress_fields), then stresses
    S_clm = cons_disp_S(w_clm, F_clm, Omega_coeffs, T_e_parent, a_clm, R=R, T_e_0=T_e_0, lmax_calc=LMAX_REF, lmax_grid=grid_expansion_res)
    S_clm.coeffs[0,0,0] = 0
    w_clm.coeffs[0,0,0] = 0
    
    sigma_tt, sigma_pp, sigma_tp = stress_fields(S_clm, w_clm, T_e_parent, lmax=grid_expansion_res, R=R, T_e_0=T_e_0)
    eps_tt, eps_pp, eps_tp = strain_fields(S_clm, w_clm, T_e_parent, lmax=grid_expansion_res, R=R, T_e_0=T_e_0)


    (   min_strain,
        max_strain,
        sum_strain,
        principal_angle_strain,
    ) = Principal_strainstress_angle(-eps_tt.data, -eps_pp.data, -eps_tp.data)    
    
    
    ## %% PLOT RESIDUAL STRAINS AND ANGLES BETWEEN DSP AND M5
    sum_strain = pysh.SHGrid.from_array(sum_strain * 1e3)
    sum_strain_residual = sum_strain_DSP.data - sum_strain.data
    
    princ_angle_residual = princ_angle_DSP.data - pysh.SHGrid.from_array(principal_angle_strain).data
    princ_angle_residual = ((princ_angle_residual + 90) % 180) - 90
    
    
    args_expand = dict(lmax=grid_expansion_res, lmax_calc=LMAX_REF)
    args_plot = dict(tick_interval=[45, 30], grid=True)

    # 1. Increase overall figure height to accommodate larger plots and clear spacing
    fig = plt.figure(figsize=(16, 10))
    
    # 2. Outer grid controls the 3 main data rows. 
    # Increase hspace here to add massive spacing BETWEEN your rows.
    outer_gs = gridspec.GridSpec(2, 1, hspace=-0.15) 



    # --- ROW 1: Sum principal strain ---
    # inner_gs creates a sub-layout for the 3 plots + colorbars in Row 1
    # height_ratios=[1, 0.05] places a thin colorbar strip tightly underneath
    inner_gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[0], 
                                                 height_ratios=[1, 0.03], hspace=-0.5, wspace=0.15)
    ax1 = fig.add_subplot(inner_gs1[0, 0:2])
    ax2 = fig.add_subplot(inner_gs1[0, 2:4])
    ax3 = fig.add_subplot(inner_gs1[0, 4:6])
    
    # Shared colorbar spans underneath columns 0 and 1
    cax_strain_shared = fig.add_subplot(inner_gs1[1, 1:3])
    # Residual colorbar spans exactly underneath column 2 (Perfect 1:1 width match)
    cax_strain_diff   = fig.add_subplot(inner_gs1[1, 4:6])
    
    cmap_limits_strain = [-6,6]
    strain_min, strain_max = sum_strain_residual.min(), sum_strain_residual.max()
    cmap_limits_strain_diff =[-max(abs(strain_min), abs(strain_max)), max(abs(strain_min), abs(strain_max))]
    # cmap_limits_strain_diff =[(strain_min), (strain_max)]
    
    # grid_w_DSP = pysh.SHCoeffs.from_array(w_DSP.coeffs / 1e3).expand(**args_expand)
    sum_strain_DSP.plot(ax=ax1, 
                    cmap_limits=cmap_limits_strain, 
                    cmap=cmap1, colorbar=None, ticks='Wsen', xlabel=None, **args_plot)
    ax1.set_title('DSP - Sum principal strain', fontweight="bold")
    
    sum_strain.plot(ax=ax2, 
                cmap_limits=cmap_limits_strain, 
                cmap=cmap1, colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax2.set_title('M5 - Sum principal strain', fontweight="bold")
    
    pysh.SHGrid.from_array(sum_strain_residual).plot(ax=ax3, cmap=cmap2, 
                                                     cmap_limits=cmap_limits_strain_diff,
                                                     colorbar=None, ticks='wsen', xlabel=None, ylabel=None, **args_plot)
    ax3.set_title('Sum principal strain residual DSP - M5', fontweight="bold")

    norm_strain = mcolors.Normalize(vmin=cmap_limits_strain[0], vmax=cmap_limits_strain[1])
    cb1 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_strain, 
                    cmap=cmap1), 
                        cax=cax_strain_shared, orientation='horizontal')
    cb1.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")


    norm_strain_diff = mcolors.Normalize(vmin=cmap_limits_strain_diff[0], vmax=cmap_limits_strain_diff[1])
    cb2 = fig.colorbar(cm.ScalarMappable(
                    norm=norm_strain_diff, 
                    cmap=cmap2), 
                        cax=cax_strain_diff, orientation='horizontal')
    cb2.set_label('Principal strain $\\epsilon$ ($\\times 10^{-3}$)', fontweight="bold")

    # --- ROW 2: Principal angle ---
    inner_gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer_gs[1], 
                                                 height_ratios=[1, 0.03], hspace=-0.35, wspace=0.15)
    ax4 = fig.add_subplot(inner_gs2[0, 0:2])
    ax5 = fig.add_subplot(inner_gs2[0, 2:4])
    ax6 = fig.add_subplot(inner_gs2[0, 4:6])
    
    cax_angle_shared = fig.add_subplot(inner_gs2[1, 1:3])
    cax_angle_diff   = fig.add_subplot(inner_gs2[1, 4:6])
    
    cmap_limits_angle = [-90,90]
    angle_min, angle_max = princ_angle_residual.min(), princ_angle_residual.max()
    cmap_limits_angle_diff =[-max(abs(angle_min), abs(angle_max)), max(abs(angle_min), abs(angle_max))]

    princ_angle_DSP.plot(ax=ax4, cmap=cmap1, cmap_limits=cmap_limits_angle, colorbar=None, ticks='WSen', **args_plot)
    ax4.set_title('DSP - Principal angle', fontweight="bold")
    
    pysh.SHGrid.from_array(principal_angle_strain).plot(ax=ax5, cmap=cmap1, cmap_limits=cmap_limits_angle, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax5.set_title('M5 - Principal angle', fontweight="bold")


    angle_min, angle_max = princ_angle_residual.min(), princ_angle_residual.max()
    pysh.SHGrid.from_array(princ_angle_residual).plot(ax=ax6, cmap=cmap2, cmap_limits=cmap_limits_angle_diff, colorbar=None, ticks='wSen', ylabel=None, **args_plot)
    ax6.set_title('Principal angle residual DSP - M5', fontweight="bold")

    norm_angle = mcolors.Normalize(vmin=cmap_limits_angle[0], vmax=cmap_limits_angle[1])
    cb3 = fig.colorbar(cm.ScalarMappable(norm=norm_angle, cmap=cmap1), cax=cax_angle_shared, orientation='horizontal')
    cb3.set_label('Principal angle [°]', fontweight="bold")

    norm_angle_diff = mcolors.Normalize(vmin=cmap_limits_angle_diff[0], vmax=cmap_limits_angle_diff[1])
    cb4 = fig.colorbar(cm.ScalarMappable(norm=norm_angle_diff, cmap=cmap2), cax=cax_angle_diff, orientation='horizontal')
    cb4.set_label('Principal angle [°]', fontweight="bold")

    # Plot strain direction
    skip_i = int(LMAX_REF / 2)
    skip = (slice(None, None, skip_i), slice(None, None, skip_i))
    grid_long, grid_lat = np.meshgrid(
        pysh.SHGrid.from_array(principal_angle_strain).lons(),
        pysh.SHGrid.from_array(principal_angle_strain).lats(),
    )
    ones = np.ones(np.shape(principal_angle_strain))
    ax4.quiver(
        grid_long[skip],
        grid_lat[skip],
        ones[skip],
        ones[skip],
        scale=5e1,
        angles=princ_angle_DSP.data[skip],
        color="g",
    )
    ax5.quiver(
        grid_long[skip],
        grid_lat[skip],
        ones[skip],
        ones[skip],
        scale=5e1,
        angles=principal_angle_strain[skip],
        color="g",
    )

    # --- GLOBAL SUPTITLE AND OUTPUT ---
    plt.suptitle('Residual strains DSP and M5 - '
                 + (f'{filter_type} filtered at l={filter_half}\n' if filter_on else 'unfiltered fin amp\n')
                 + ('Solving for $\\delta \\rho_{lm}$, $\\delta c_{lm}$=0' if solve_for == 'drho_lm' else '')
                 + ('Solving for $\\delta c_{lm}$, $\\delta \\rho_{lm}$=0' if solve_for == 'dc_lm' else '')
                 + f' --- lmax={LMAX_REF}, nmax={nmax}, delta_max={delta_max}'
                 + f'\nDSP constant $T_e$={Te_input/1e3} km, '
                 + (f'M5 constant $T_e$={Te_input/1e3} km' if strain==0 else '')
                 + ('M5 $T_e$=Plesa Strain14 Map' if strain==14 else '')
                 + ('M5 $T_e$=Plesa Strain17 Map' if strain==17 else '')
                 + (f'M5 $T_e$={Te_twoval_big/1e3:.0f}/{Te_twoval_sml/1e3:.0f} km two-value map' if strain=='twoval' else '')
                 + f'\nDSP & M5 constant $T_c$={T_c/1e3} km, '
                 f'$\\rho_c$ = {rho_c} kg/m$^3$, $\\rho_l$ = {rho_l} kg/m$^3$, '
                 f'$\\rho_m$ = {rho_m} kg/m$^3$',
                 y=0.86, fontsize=15)
                
    if SaveFigs:
        plt_savetitle = (f'Residuals_strains_DSP_M5_lmax={LMAX_REF}_iter{iterate}_nmax={nmax}_delta_max={delta_max}_'
                + f'SolveFor{solve_for}_'
                + (f'Mb{Mb/1e3}_Mt{Mt/1e3}_' if solve_for == 'drho_lm' else '')
                + (f'both_constant_Te={Te_input/1e3}km_' if strain==0 else '')
                + (f'Te_M5=PlesaStrain14Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==14 else '')
                + (f'Te_M5=PlesaStrain17Map_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain==17 else '')
                + ('Te_M5={Te_twoval_big/1e3:.0f}/{Te_twoval_sml/1e3:.0f}twovalmap_'
                   f'Te_DSP={Te_input/1e3}km_'  if strain=='twoval' else '')
                + f'Tc={T_c/1e3}km'
                + f'filter_on{filter_on}'
                + '.png')
        FigPath = os.path.join(SavePath, plt_savetitle)
        plt.savefig(FigPath, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()



# %% THARSIS STRAIN PLOT
    
    def crop_grid(grid, lon_range=(180., 330.), lat_range=(-75., 75.)):
        """(sub_array, extent) for a pyshtools SHGrid. Works for extend=True/False.
        Longitude range must be increasing (no 0/360 crossing)."""
        lats, lons = grid.lats(), grid.lons()
        jlat = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
        jlon = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
        sub  = grid.data[np.ix_(jlat, jlon)]
        dlat, dlon = abs(lats[1]-lats[0]), abs(lons[1]-lons[0])
        extent = (lons[jlon[0]]-dlon/2, lons[jlon[-1]]+dlon/2,
                  lats[jlat[-1]]-dlat/2, lats[jlat[0]]+dlat/2)
        return sub, extent

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,8))
    sub_DSP, ext_DSP = crop_grid(sum_strain_DSP)
    sub_M5,  ext_M5  = crop_grid(sum_strain)
    cropped_Te, ext_Te = crop_grid(T_e_parent_km)
    
    cmap_limits_strain_Tharsis =[-np.max(sub_M5), np.max(sub_M5)]

    im1 = ax1.imshow(sub_DSP, extent=ext_DSP, origin='upper', cmap=cmap1,
                    vmin=cmap_limits_strain_Tharsis[0], vmax=cmap_limits_strain_Tharsis[1])
    ax1.set_xticks(np.arange(180, 331, 30)); ax1.set_yticks(np.arange(-75, 76, 25))
    ax1.grid(True); ax1.set_ylabel('Latitude'); ax1.set_xlabel('longitude')
    ax1.set_title('DSP - Constant Te=100km', fontweight="bold")
    ax1.contour(cropped_Te < 100, levels=[0.99], extent=ext_Te, colors='k', origin='upper')
    
    im2 = ax2.imshow(sub_M5, extent=ext_M5, origin='upper', cmap=cmap1,
                    vmin=cmap_limits_strain_Tharsis[0], vmax=cmap_limits_strain_Tharsis[1])
    ax2.set_xticks(np.arange(180, 331, 30)); ax2.set_yticks(np.arange(-75, 76, 25))
    ax2.grid(True); ax2.set_ylabel('Latitude'); ax2.set_xlabel('longitude')
    if strain == 'twoval':
        ax2.set_title(f'M5 - 2-Step variable Te=[{Te_twoval_sml/1e3}, {Te_twoval_big/1e3}] km', fontweight="bold")
    elif strain == 14:
        ax2.set_title(f'M5 - Plesa Te map strain= $10^{-14}$, Te $\\approx${np.min(T_e_parent_km.data)} km', fontweight="bold")
    elif strain == 0:
        ax2.set_title(f'M5 - Constant Te={Te_input/1e3} km', fontweight="bold")
        
    ax2.contour(cropped_Te < 100, levels=[0.99], extent=ext_Te, colors='k', origin='upper')
    
    ax1.set_autoscale_on(False)      # add after im1 = ax1.imshow(...)
    ax2.set_autoscale_on(False)      # add after im2 = ax2.imshow(...)
    # Add extensional tectonic features from Knapmeyer et al. (2006)
    tecto_path = f"{os.getcwd()}/Tectonics_data"
    # Plt_tecto_Mars(tecto_path, ax=[ax1, ax2], compression=False, extension=True)

    
    
    fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', label='Sum strain ($\\times 10^{-3}$)')
    fig.suptitle('Sum horizontal strain Tharsis')
    plt.show()


# %%
    

    print(f'\nTotal model runtime: {(time.perf_counter() - t_begin):.1f}s')