# -*- coding: utf-8 -*-
"""
Beuthe (2008) variable-thickness thin-shell flexure model for a planetary
lithosphere, with three solve modes in one class.

    solve_for = 'dc_lm'    INVERSE MODEL. 
                             Observed topography H and geoid G are given,
                             flexure w and the crustal root dc are the 
                             corresponding results.
    solve_for = 'drho_lm'  INVERSE MODEL. 
                             Observed topography H and geoid G are given,
                             flexure w and the lateral density variation in the
                             layer [Mt, Mb] are the corresponding results.
    solve_for = 'G_lm'     FORWARD MODEL. 
                             H and a prescribed crustal root dc are given, 
                             flexure w and the geoid are the corresponding 
                             results.
                           
All three solve the same equation set. Only the split between known and
unknown terms changes.

At constant Te the system is diagonal in degree. A laterally varying Te turns
the coefficients into fields, so products become Gaunt convolutions and every
degree couples to every other. That results in a 2N x 2N system that is assembled
once, LU-factorised, and reused across the iterations. N = (lmax+1)^2

Finite amplitude of gravitational potential (nmax > 1), the downward-continuation 
filter (inverse modes only) and the lateral-density corrections (drho_lm) are 
all supported, and the iteration uses DSP's damping scheme.


Conventions:
    - topo, geoid : SHCoeffs in m, degree 0 = R
    - dc          : SHCoeffs in m, positive = thicker crust.
    - drho        : SHCoeffs in kg/m3, lateral density variation in [Mt, Mb].
    - Te          : SHCoeffs in m. A map with no l >= 1 power is treated exactly
                    as a constant-Te run.

This file carries the model only: no diagnostics, no self-tests, no data
loading. Old model versions include these.

"""

import os
import time
import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt
from palettable import scientific as scm
from cmcrameri import cm as cmc
from scipy.linalg import lu_factor, lu_solve
from pyshtools.expand import MakeGridDH, SHExpandDH
from Displacement_strain_planet.utils import corr_nmax_drho, DownContFilter

from Gaunt_utils_v5 import (build_or_load_gaunt, selftest_gaunt,
                            selftest_term_weights, mode_degrees,
                            build_gidx, cell_sums_AB,
                            cell_sums_eta_dL, build_conv_matrix)



class VarTeModel:
    """
    Beuthe (2008) flexure model with laterally varying Te, in three modes.

    Constant parameters:
       - R                  : planetary radius [m]
       - g0                 : surface gravity [m/s2]
       - mass               : mass [kg]
       - T_c                : crustal thickness [m]
       - rho_l              : surface load density [kg/m3]
       - rho_c              : crustal density [kg/m3]
       - rho_m              : mantle density [kg/m3]
       - E                  : Young's modulus [Pa]
       - nu                 : Poisson's ratio
       - lmax               : maximum spherical harmonic degree of the solution
       - grid_expansion_res : grid bandwidth for the spatial products. Must be at
                              least 2*lmax; 3*lmax (the default) leaves headroom.

    Iteration parameters:
       - nmax               : Finite amplitude correction power of topography.
                              1 = mass-sheet (linear, single solve). > 1 switches on
                              the finite-amplitude corrections, which iterate.
       - iterate            : run the correction loop at all (ignored when nmax = 1).
       - delta_max          : convergence threshold [m] on the tracked moho relief
                              (dc - w).
       - delta_out          : divergence threshold [m].
       - iter_max           : max number of iterations.
        
        
    The input crustal root field (FORWARD mode only.The inverse modes solve 
    for it or set it to zero):
        - DC_MODE           : None -> uncompensated (no root)
                             'airy'    -> dc = rho_l*H/drho
                             'partial' -> DC_FRACTION of Airy
        - DC_FRACTION       : the fraction used by 'partial'.
        - DC_TAPER          : None, 'linear' or 'smooth'. A degree taper on the
                              prescribed root, 1 at L_TAPER_ONE and 0 at L_TAPER_ZERO.
                             'linear' is a straight ramp, 'smooth' is a smoothstep,
                             which rings less in the map domain.
        - DC_FROM_FILE      : path to an inverse-model export, used instead of a
                              prescribed root. Its stored setup is checked against
                              this one, and a filtered export is refused (a filtered
                              root does not satisfy the geoid equation, so its 
                              forward geoid cannot reproduce the observed one).

    The density layer (INVERSE drho_lm mode):
        Mt                  : top depth of the layer the lateral density sits, in [m].
        Mb                  : bottom depth of the layer the lateral density sits, in [m].
    Together with rho_l vs rho_c these fix which interfaces the density acts on. 
    The live gates are printed each iteration by density_gates().


    Output:
        savefigs, savepath  : where the plot_* methods write.
        plot_dc_spectrum    : show the prescribed root's power spectrum and taper.
    """
    

    def __init__(self, 
                 solve_for,  # Which model to run, can be 'G_lm', 'dc_lm', 'drho_lm'
                 R, g0, mass, lmax,
                 T_c, rho_l, rho_c, rho_m, 
                 Mt=None, Mb=None,
                 E=100.0e9, nu=0.25,
                 grid_expansion_res=None, nmax=5, iterate=True,
                 delta_max=1e-6, delta_out=1e12, iter_max=300,
                 damp=None,
                 DC_MODE=None, DC_FRACTION=0.5, DC_FROM_FILE=None,
                 DC_TAPER=None, L_TAPER_ONE=1, L_TAPER_ZERO=30,
                 filter_type=None, filter_half=50,
                 savefigs=False, savepath='Plots', 
                 plot_dc_spectrum=False):

        # --- geometry and material --------------------------------------
        self.R     = R
        self.g0    = g0
        self.mass  = mass
        self.lmax  = lmax
        self.T_c   = T_c
        self.rho_l = rho_l
        self.rho_c = rho_c
        self.rho_m = rho_m
        self.drho  = rho_m - rho_c            # moho density contrast
        self.drhol = rho_c - rho_l           # surface load / crust contrast
        self.E     = E
        self.nu    = nu
        self.Mt = 0.0 if Mt is None else Mt
        self.Mb = T_c if Mb is None else Mb
        self.rhobar = mass * 3.0 / (4.0 * np.pi * R**3)

        # --- numerics ----------------------------------------------------
        self.grid_expansion_res = (3 * lmax if grid_expansion_res is None
                                   else grid_expansion_res)
        if self.grid_expansion_res < 2 * lmax:
            raise ValueError('grid_expansion_res must be at least 2*lmax, '
                             'or the spatial products alias.')
        self.nmax      = nmax
        self.iterate   = iterate
        self.delta_max = delta_max
        self.delta_out = delta_out
        self.iter_max  = iter_max
        
        if nmax > 1 and corr_nmax_drho is None:
            raise ImportError('nmax > 1 needs Displacement_strain_planet develop '
                              'branch for the finite-amplitude correction.')

        # --- the crustal root -------------------------------------------
        self.DC_MODE      = DC_MODE
        self.DC_FRACTION  =  DC_FRACTION
        self.DC_FROM_FILE = DC_FROM_FILE
        self.DC_TAPER     = DC_TAPER
        self.L_TAPER_ONE  = L_TAPER_ONE
        self.L_TAPER_ZERO = L_TAPER_ZERO

        # --- the downward-continuation filter belongs to the INVERSE
        #     problem (it regularises continuing the observed geoid down to
        #     the moho). The forward model continues the root's gravity up,
        #     which only attenuates, so there is nothing to regularise: leave
        #     filter_type = None unless you are reproducing a filtered
        #     inversion, in which case the root no longer satisfies eq (1)
        #     and the predicted geoid cannot match the observed one. --------
        self.filter_type = filter_type
        self.filter_half = filter_half


        # --- Build the Gaunt plan once
        self.plan  = build_or_load_gaunt(self.lmax, self.nu)        



        # --- which quantity the system is solved for ---------------------
        #   'G_lm'    forward: H and dc known -> w, G           (linear at nmax=1)
        #   'dc_lm'   inverse: H and G known  -> w, crustal root
        #   'drho_lm' inverse: H and G known  -> w, lateral density in [Mt, Mb]
        if solve_for not in ('G_lm', 'dc_lm', 'drho_lm'):
            raise ValueError("solve_for must be 'G_lm', 'dc_lm' or 'drho_lm', "
                             f"got {solve_for!r}")
        self.solve_for = solve_for
        # DSP's damping (Wieczorek et al. 2013): every third iteration takes the
        # average of the previous two corrections. On by default for dc_lm,
        # whose Picard iteration oscillates without it.
        self.damp = (solve_for == 'dc_lm') if damp is None else damp

        if solve_for == 'G_lm':
            if filter_type is not None:
                raise ValueError(
                    "solve_for='G_lm' is a FORWARD model: eq (1) is substituted, "
                    "not inverted, so there is no downward continuation to "
                    "filter. A live filter would amplify the predicted geoid. "
                    "Set filter_type=None.")
        else:
            if DC_MODE is not None or DC_FROM_FILE is not None:
                raise ValueError(
                    f"solve_for={solve_for!r} SOLVES for the internal structure, "
                    "so a prescribed root is meaningless here. Set DC_MODE=None "
                    "and DC_FROM_FILE=None.")
                # This is not entirely the case when solving for drho_lm, though 
                # it is not yet implemented in the current model to incorporate this

        # --- output ------------------------------------------------------
        self.savefigs = savefigs
        self.savepath =  savepath
        self.plot_dc_spectrum = plot_dc_spectrum
        if savefigs:
            os.makedirs(savepath, exist_ok=True)

        # grids are cached per (field, lmax); cleared at the start of solve()
        self._grid_cache = {}


    def make_mode_map(self, lmax):
        """
        Flatten all combinations of l,m into a flat array based on input lmax.
        """
        return [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]

    def truncate(self, clm, lmax):
        """
        Truncate any SHCoeffs object coefficients up to degree lmax.
        """
        return pysh.SHCoeffs.from_array(clm.coeffs[:, :lmax+1, :lmax+1].copy(),
                                        normalization='4pi')

    def derive_D_a(self, T_e_parent, lmax):
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

        grid = T_e_parent.expand(lmax=self.grid_expansion_res)
        print('Computing D and alpha using Te grid expanded to '
              'lmax=grid_expansion_res')
        Te0_loc = T_e_parent.coeffs[0,0,0]
        Re_loc  = self.R - Te0_loc/2
        # a Te map with no l >= 1 power is a constant-Te run: build D, alpha
        # and eta as exact monopoles instead of going through the grid
        is_constant_Te = not np.any(T_e_parent.coeffs[:, 1:, :])
        if is_constant_Te:
            D = self.E*T_e_parent.coeffs[0,0,0]**3/(12*(1-self.nu**2))
            D_coef = grid.expand().copy()
            D_coef.coeffs[:,:,:] = 0
            D_coef.coeffs[0,0,0] = D
            D = pysh.SHCoeffs.from_array(D_coef.coeffs[:, :2*lmax+1, :2*lmax+1])

            a = 1.0/(self.E*T_e_parent.coeffs[0,0,0])
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
            D = pysh.SHGrid.from_array(self.E*grid.data**3/(12*(1-self.nu**2))).expand()
            D = pysh.SHCoeffs.from_array(D.coeffs[:, :2*lmax+1, :2*lmax+1])
            a = pysh.SHGrid.from_array(1.0/(self.E*grid.data)).expand()
            a = pysh.SHCoeffs.from_array(a.coeffs[:, :2*lmax+1, :2*lmax+1])

            eta_grid = 1.0/(1.0 + grid.data**2/(12.0*Re_loc**2))
            D_eta = pysh.SHGrid.from_array(eta_grid*self.E*grid.data**3/(12*(1-self.nu**2))).expand()
            D_eta = pysh.SHCoeffs.from_array(D_eta.coeffs[:, :2*lmax+1, :2*lmax+1])
            a_eta = pysh.SHGrid.from_array(eta_grid/(self.E*grid.data)).expand()
            a_eta = pysh.SHCoeffs.from_array(a_eta.coeffs[:, :2*lmax+1, :2*lmax+1])
            eta_clm = pysh.SHGrid.from_array(eta_grid).expand()
            eta_clm = pysh.SHCoeffs.from_array(eta_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

        print('D and alpha computed \n')
        return D, a, D_eta, a_eta, eta_clm

    def build_dc_prescribed(self, topo_clm, lmax, force_airy=False):
        """
        Prescribed crustal-root map, or None for the uncompensated case.

        Positive dc > 0 means thicker crust. A thicker crust pushes the moho 
        down -> negative relief, positive dc. Airy balance rho_l*H = drho*r gives
        moho relief -r, hence dc = +rho_l*H/drho. Sanity: 3 km of topography with
        rho_l = 2900, drho = 600 gives a 14.5 km root.

        Degrees 0 and 1 forced to zero: dc is a variation about T_c (so zero mean
        by definition), and G_1 = 0 is the COM rule. A non-zero dc_1 would put
        degree-1 mass that the rule then refuses to represent in the geoid.
        """
        if self.DC_MODE is None and not force_airy:
            return None
        H = self.truncate(topo_clm, lmax).copy()
        H.coeffs[0, 0, 0] = 0.0

        # Airy definition for spherical geometry, for equal pressure at equal depth
        # (Hemingway & Matsuyama, 2017)
        phi    = (self.R - self.T_c) / self.R
        g_m    = self.g0 * (1.0 + (phi**3 - 1.0) * self.rho_c / self.rhobar) / phi**2
        dc = H.coeffs * (self.rho_l / self.drho)  * (self.g0 / g_m)

        # Partial Airy compensation
        if self.DC_MODE == 'partial' and not force_airy:
            dc *= self.DC_FRACTION
        elif self.DC_MODE not in ('airy', 'partial') and not force_airy:
            raise ValueError(f"DC_MODE must be None, 'airy' or 'partial', "
                             f"got {self.DC_MODE!r}")

        # Plot power spectrum of the input dc
        if self.plot_dc_spectrum:
            fig, ax1 = plt.subplots(figsize=(10,9))
            pysh.SHCoeffs.from_array(dc).plot_spectrum(ax=ax1, show=False,
                                                       legend='Input dc_lm power spectrum')
        # Add taper on the Airy compensation to reduce with degree
        if self.DC_TAPER is not None:
            _l = np.arange(dc.shape[1], dtype=float)
            # position along the transition band: 0 at L_TAPER_ONE, 1 at L_TAPER_ZERO
            _x = np.clip((_l - self.L_TAPER_ONE)
                         / (self.L_TAPER_ZERO - self.L_TAPER_ONE), 0.0, 1.0)
            if self.DC_TAPER == 'linear':
                _t = 1.0 - _x                     # straight ramp, kinks at both ends
            elif self.DC_TAPER == 'smooth':
                _t = 1.0 - _x*_x*(3.0 - 2.0*_x)   # smoothstep: zero slope at both ends
            else:
                raise ValueError(f"DC_TAPER must be None, 'linear' or 'smooth', "
                                 f"got {self.DC_TAPER!r}")
            dc *= _t[None, :, None]

            if self.plot_dc_spectrum:
                pysh.SHCoeffs.from_array(dc).plot_spectrum(ax=ax1, 
                                                           show=False,
                                                           legend='Input dc_lm power spectrum, tapered')
                ax1.set_title(f'Custom input dc power spectrum: \n mode={self.DC_MODE}, '
                            + (f'Fraction of Airy={self.DC_FRACTION}' if self.DC_MODE=='partial' else '')
                            + (f'\nTapered with {self.DC_TAPER} transition between '
                               f'{self.L_TAPER_ONE} and {self.L_TAPER_ZERO}' if self.DC_TAPER is not None else '')
                            , y=1.0)
    
                # Put the lines and legend labels into one plot
                ax2 = ax1.twinx()
                ax2.plot(_l, _t, '--r', label='Taper function')
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc=0)
                plt.show()

        dc[:, 0, 0] = 0.0
        dc[:, 1, :] = 0.
        return pysh.SHCoeffs.from_array(dc, normalization='4pi')

    def load_dc_from_file(self, path, lmax, expect, Te_clm=None):
        """
        Load the dc_lm solution exported by M5 and verify the two runs used the
        same configuration.

        `expect` is a dict of the M6 values to check against. A mismatch raises
        rather than warns: comparing a filtered inverse solution against an
        unfiltered forward model produces a large, plausible-looking, meaningless
        residual, and that is exactly the failure this test exists to avoid.

        Returns (dc_clm, G_obs_coeffs, w_inv_coeffs).
        """
        file = np.load(path)
        for k, v in expect.items():
            if k not in file:
                raise ValueError(f"{path}: missing metadata '{k}'")
            if not np.isclose(float(file[k]), float(v), rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"{path}: configuration mismatch in '{k}': "
                    f"loaded dc saved {float(file[k])}, current run has {float(v)}. "
                    f"The inverse->forward test is only exact when both runs use "
                    f"identical parameters.")
        if float(file['filt']) != -1:
            raise ValueError(
                f"{path} was produced with the downward-continuation filter on. "
                f"The forward model has no downward continuation, so the two are "
                f"off by exactly the filter. Re-run M5 with filter_on = False.")

        # The Te map is part of the operator. Scalars like T_e_0 cannot catch a
        # changed map (e.g. its degree-1 content removed in one script only).
        if Te_clm is not None:
            if 'Te' not in file:
                print(f"  ! {path} stores no Te map -- cannot verify that loaded "
                      f"file used the same Te as this run. "
                      "Re-export from inverse run!")
            else:
                _Tf = file['Te']
                _n  = min(_Tf.shape[1], Te_clm.coeffs.shape[1])
                _d  = np.max(np.abs(_Tf[:, :_n, :_n] - Te_clm.coeffs[:, :_n, :_n]))
                if _d > 1e-9 * abs(Te_clm.coeffs[0, 0, 0]):
                    raise ValueError(
                        f"{path}: Te map differs from this run's (max |dTe_lm| = "
                        f"{_d:.3e} m). The inverse->forward test is only exact "
                        f"with the identical Te map.")

        dc = file['dc'].copy()
        print(f'\nRMS dc degree-1={np.sqrt(np.mean(dc[:, 1, :]**2)):.4f} m')
        dc[:, 0, 0] = 0.0        # dc is a variation about T_c: zero mean
        # Degree 1 is kept! It is the root that balances H_1 (it is what makes
        # G_1 = 0 in inverse runs), and with laterally varying Te it reaches 
        # every degree of omega through max(Te-Tc,0), Te^2 and alpha. Zeroing 
        # it here was the dominant inverse->forward discrepancy.
        
        L = int(file['lmax'])
        print(f"  loaded {path}: RMS(dc) = "
              f"{np.sqrt(np.mean(dc[:, 2:L+1, :]**2)):.4f} m\n")
        return (pysh.SHCoeffs.from_array(dc, normalization='4pi'),
                file['geoid_obs'].copy(), file['w'].copy())

    def drho_layer(self, lmax, R, g0, mass):
        """
        Helper function used for precalculating recurring constants and 
        degree-dependent arrays used for the solve_for=drho_lm model runs. 
        Some functions and naming:

          - M     : layer thickness (Mb - Mt)
          - g_M   : gravity at mid-depth of the density anomaly layer (DSP's `gdrho`)
          - Cp    : 3 / (rhobar * (2l+1))
          - B_1   : Cp * R/(l+3) * [(Rt/R)^(l+3) - (Rb/R)^(l+3)]
          - B_2   : Cp * R/(l+3) * [RtRCl - RbRCl]
                   (a (g0/g_m) prefactor is applied outside by the function caller)
        """

        M      = self.Mb - self.Mt
        R_top  = R - self.Mt
        R_base = R - self.Mb
        R_c    = R - self.T_c
        R_mid  = (R_top + R_base) / 2.0
        rho_d  = self.rho_c if self.Mt <= self.T_c else self.rho_m
        g_M    = g0 * (1.0 + ((R_mid/R)**3 - 1.0) * rho_d / self.rhobar) / (R_mid/R)**2

        degs = np.arange(2*lmax + 1, dtype=float)
        Cp   = 3.0 / (self.rhobar * (2.0*degs + 1.0))
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
        # and rhobconst == Cp. So dividing here reaches every consumer of 
        # drho_layer at once (q_lm, Omega_eq1_RHS, Omega_eq2_RHS and the omega 
        # assembly) with no edit to any of them.
        _, _, DCfilter_drhom, DCfilter_drhomc = self.DCfilters(len(degs) - 1)
        B_1 = B_1 / DCfilter_drhom
        B_2 = B_2 / DCfilter_drhomc

        return dict(M=M, g_M=g_M, B_1=B_1, B_2=B_2, Cp=Cp, Rl3=Rl3)

    def _cached_grid(self, key, lmax, build):
        """
        The topography, geoid, and Te fields are cached for every unique run, 
        such that they do not get recomputed every iteration, since they don't
        change. 

        Any remaining cached grids are cleared at the start of every solve, 
        using `_GRID_CACHE.clear()`.
        """
        k = (key, lmax, self.grid_expansion_res)
        if k not in self._grid_cache:
            self._grid_cache[k] = build()
        return self._grid_cache[k]

    # ------------------------------------------------------------------
    #  Beuthe (2008) eq (58)/(66) eta placement helpers
    # ------------------------------------------------------------------
    def _eta_fields(self, T_e_parent):
        """
        eta and eta*Te^2 on the solver grid. Re is built from the MEAN Te, as in
        derive_D_a and cons_disp_S, so every eta in the model is the same field.
        Not cached: it is cheap, and caching would tie it to one Te map.
        """
        Te  = T_e_parent.expand(lmax=self.grid_expansion_res).data
        Re0 = self.R - T_e_parent.coeffs[0, 0, 0]/2
        eta = 1.0/(1.0 + Te**2/(12.0*Re0**2))
        return eta, eta*Te**2

    def _expand_band(self, grid, lmax):
        """Expand a grid and keep the 2*lmax+1 working band."""
        c = pysh.SHGrid.from_array(grid).expand()
        return pysh.SHCoeffs.from_array(c.coeffs[:, :2*lmax+1, :2*lmax+1])

    def _eq1_halves(self, X_grid, T_e_parent, lmax):
        """
        The two halves of the eq-1 Omega operator of Beuthe (2008) eq (58),

            R^3 [ Delta Omega - Delta'(eta Omega) ]
              = -2 R^3 eta Omega  +  (R/12) Delta( eta Te^2 Omega ),

        for an arbitrary Omega content X:
            c1 = eta * X                 (pairs with the -2 Re^3 factors)
            c2 = Delta( eta * Te^2 * X ) (pairs with the  Re/12 factors)

        eta sits INSIDE the Laplacian, and the Laplacian acts last, at the
        output degree. At constant Te this reduces exactly to the previous
        eta0 * [ X , Te0^2 Delta X ], so constant-Te benchmarks are unchanged.
        """
        eta_g, etaTe2_g = self._eta_fields(T_e_parent)
        c1 = self._expand_band(eta_g*X_grid, lmax)
        c2 = self._expand_band(etaTe2_g*X_grid, lmax)
        for l in range(c2.coeffs.shape[1]):
            c2.coeffs[:, l, :] *= -l*(l + 1)
        return c1, c2

    def density_gates(self, ):
        """
        The three possible density variation interface gates:
            density_var_H  : Mt == 0
            density_var_w  : Mb < T_c and Mt == 0 and rho_l == rho_c
            density_var_dc : T_c in (Mb, Mt)

        Which configurations isolate which interface:
            Mt = 0,   Mb < T_c,  rho_l != rho_c  -> H only
            Mt = T_c, Mb > T_c                   -> dc only
            Mt = 0,   Mb < T_c,  rho_l == rho_c  -> H and w together
            Mt = 0,   Mb = T_c or 0,             -> H and dc together
        w cannot be isolated: it requires top_drho == 0, which forces H on. And no
        configuration fires all three: w needs Mt == 0 and Mb < T_c, while dc then
        reduces to T_c == Mb, which contradicts Mb < T_c.
        """
        gate_H  = (self.Mt == 0)
        gate_dc = (self.T_c in (self.Mb, self.Mt))
        gate_w  = (self.Mb < self.T_c and self.Mt == 0 and self.rho_l == self.rho_c)
        return gate_H, gate_dc, gate_w

    def interface_geoid_correction(self, interface, R, mass, lmax, lmaxgrid, nmax,
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
            dr_lm = self.truncate(topo_clm, lmax).coeffs.copy()
            dr_lm[0, 0, 0] = R
            # topo is fixed within a pass, so its grid is worth caching. w and
            # w - dc change every iteration and are not.
            shape_grid = self._cached_grid('shapeH', lmax,
                                      lambda: MakeGridDH(dr_lm, **args_grid))
            # Use lambda here to only trigger the MakeGridDH if it is not in the cache.
            # Saves time, because now it is not recomputed every iteration.

        elif interface == 'w':
            dr_lm = self.truncate(w_clm, lmax).coeffs.copy()
            dr_lm[0, 0, 0] = R
            shape_grid = MakeGridDH(dr_lm, **args_grid)
        elif interface == 'wdc':
            dr_lm = (self.truncate(w_clm, lmax).coeffs - self.truncate(dc_clm, lmax).coeffs)
            dr_lm[0, 0, 0] = R - self.T_c
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
            drho_grid = MakeGridDH(self.truncate(drho_clm, lmax).coeffs, **args_grid)
            # Set the scalar rho and rho_grid for each interface
            if interface == 'H':
                drho_scalar = self.rho_l
                rho_grid    = drho_grid + self.rho_l
            elif interface == 'w':
                drho_scalar = (self.rho_c - self.rho_l) if self.rho_c != self.rho_l else 1.0
                rho_grid    = -drho_grid
            else:
                drho_scalar = self.drho
                rho_grid    = (self.rho_m - (drho_grid + self.rho_c) if self.Mb <= self.T_c
                               else (drho_grid + self.rho_m) - self.rho_c)

        # Calculate the SH coefficients of the FA-MS correction of this interface
        # using DSP's corr_nmax_drho function
        SH_correction = corr_nmax_drho(dr_lm=dr_lm, drho=drho_scalar, shape_grid=shape_grid,
                             rho_grid=rho_grid, lmax=lmax, mass=mass, nmax=nmax,
                             R=R, drho_Thinshell=drho_scalar,
                             density_var=drho_clm is not None)
        SH_correction[:, 0, 0] = 0.0

        return SH_correction

    def geoid_corrections(self, H_corr, w_corr, wdc_corr, R, lmax):
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
        phi     = (R - self.T_c) / R

        # Here the interface corrections (defined per unit density) are multiplied
        # with the scalar densities of each corresponding interface.
        surface = self.rho_l * H_corr[:, :lmax+1, :lmax+1] 
        flexure = self.drhol * w_corr[:, :lmax+1, :lmax+1]
        moho    = self.drho  * wdc_corr[:, :lmax+1, :lmax+1]

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

    def geoid_corrections_fwd(self, H_corr, w_corr, wdc_corr, R, lmax):
        """
        geoid_corrections with degree 1 removed from BOTH outputs.

        In M6, corrG keeps its degree-1 term because DSP's eq (1) guard tests
        G_lm, which is an INPUT there, so the guard is False and eq (1) stays
        live at l = 1. In M6, G_lm is an unknown, so thinshell.py 984 multiplies
        the whole of eq (1) -- corrections included -- by zero at l = 1, and the
        COM gauge sets G_1 = 0 outright. corrGc is already zeroed at l = 1 by
        geoid_corrections itself, for the matching reason on eq (2).

        Leaving corrG_1 live would inject a degree-1 term into both the predicted
        geoid and, through q_correction_from_geoid, into q_1 -- which is exactly
        the channel that reaches Omega_1.
        """
        corrG, corrGc = self.geoid_corrections(H_corr, w_corr, wdc_corr, R, lmax)
        corrG[:, 1, :]  = 0.0
        corrGc[:, 1, :] = 0.0          # already zero; explicit for symmetry
        return corrG, corrGc

    def geoid_mass_sheet(self, geoid_clm, corrG, lmax):
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
        geoid_MS = self.truncate(geoid_clm, lmax).copy()
        geoid_MS.coeffs -= correction_G

        return geoid_MS

    def q_correction_from_geoid(self, corrG, corrGc, R, g0, mass):
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
        phi    = (R - self.T_c) / R
        g_m    = g0 * (1.0 + (phi**3 - 1.0) * self.rho_c / self.rhobar) / phi**2

        return -g0*self.rho_l*corrG - g_m*self.drho*corrGc

    def DCfilters(self, lmax):
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
        D_relief > R_ref will always be <1, i.e. attenuating and thus filtering 
        itself.


        DSP divides the interface term of eqs (1) and (2) by them. Because 
        DownContFilter returns 1/(1 + ...) <= 1, dividing in the FORWARD equation 
        means the recovered interface is multiplied by the filter. Damped at 
        high degree, which is the point of downward-continuation stabilisation.

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
        if self.filter_type is None:
            return ones, ones, ones, ones

        param_filt = (degrees, self.filter_half)
        kw_filt = dict(filter_type=self.filter_type, quiet=False)
        R_c = self.R - self.T_c
        R_base_drho = self.R - self.Mb

        DCfilter_mohoD = DCfilter_mohoDc = DCfilter_drhom = DCfilter_drhomc = ones
        if self.solve_for == 'dc_lm':
            # The two filters for the crustal root variations, first one for G_lm
            # eq and second one for the Gc_lm eq
            DCfilter_mohoD  = DownContFilter(*param_filt, self.R, R_c, **kw_filt)
            DCfilter_mohoDc = DownContFilter(*param_filt, R_c, R_c, **kw_filt)
        else:
            # The two filters for the internal density variations, first one for G_lm
            # eq and second one for the Gc_lm eq 
            DCfilter_drhom  = DownContFilter(*param_filt, self.R, R_base_drho, **kw_filt)
            DCfilter_drhomc = DownContFilter(*param_filt, R_c, R_base_drho, **kw_filt)

        return DCfilter_mohoD, DCfilter_mohoDc, DCfilter_drhom, DCfilter_drhomc



    def drho_q_omega_correction(self, interface, drho_clm, topo_clm, geoid_clm, w_clm,
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
        args_grid = dict(sampling=2, lmax=self.grid_expansion_res, extend=False,
                    lmax_calc=lmax)

        zero   = np.zeros((2, lmax+1, lmax+1))
        d_grid = MakeGridDH(self.truncate(drho_clm, lmax).coeffs, **args_grid)

        if interface == 'dc':
            # Always returns zero for now, because either drho_lm or dc_lm is set
            # to zero, since they can't be solve simultaneously (yet), and the 
            # field is the multiplication of drho_lm and dc_lm
            return zero, zero.copy()

        elif interface == 'H':
            # THE q FIELD
            # Build H-G
            HG_coeffs = (self.truncate(topo_clm, lmax).coeffs
                  - self.truncate(geoid_clm, lmax).coeffs)
            HG_coeffs[0, 0, 0] = 0.0      # H and G share the same R, making sure that deg0 is zero then

            # H-field is constant, so cache the grid
            q_field = self._cached_grid('HG', lmax, 
                                   lambda: MakeGridDH(HG_coeffs, **args_grid))

            # THE OMEGA FIELD
            H_coeffs = self.truncate(topo_clm, lmax).coeffs.copy()
            H_coeffs[0, 0, 0] = 0

            # H-field is constant, so cache the grid
            om_field = self._cached_grid('Hrelief', lmax, 
                                    lambda: MakeGridDH(H_coeffs, **args_grid))

        elif interface == 'w':
            # THE q FIELD = THE OMEGA FIELD FOR w
            w_lm = self.truncate(w_clm, lmax).coeffs.copy()
            w_lm[0, 0, 0] = 0.0                   # RELIEF (thinshell.py 1872)
            q_field = om_field = MakeGridDH(w_lm, **args_grid)

            # Following DSP, negative drho_grid here for the w-interface
            d_grid  = -d_grid

        else:
            raise ValueError(f"interface must be 'H', 'w' or 'dc', got {interface!r}")

        Te_grid = self._cached_grid('Te', lmax, 
                    lambda: MakeGridDH( self.truncate(T_e_parent, lmax).coeffs, **args_grid))
        # Use lambda here to only trigger the MakeGridDH if it is not in the cache.
        # Saves time, because now it is not recomputed every iteration.

        q_corr  = g0 * SHExpandDH(d_grid*q_field, lmax_calc=lmax, sampling=2)
        om_corr = ((self.nu/(1.0 - self.nu)) * g0 / R 
                   * SHExpandDH( (d_grid*Te_grid)*om_field, lmax_calc=lmax, sampling=2))

        return q_corr, om_corr

    def omega_corr_to_rhs(self, omega_corr_phys, T_e_parent, a_clm, lmax, R, T_e_0, Re):
        """
        Map the omega correction into additive terms on the two Omega right-hand 
        sides.

        The omega equation as from DSP is:
            omega_lm = <the H, w, dc, drhom content>  +  drho_omega_corr .
        M6 does not solve eq (5): it substitutes omega's content into the w and F
        eqs of Beuthe (2008). Because the omega equation is substituted and not 
        inverted *like the G_lm and Gc_lm equations), the correction is simply an
        added term to the equation. This is why the drho_omega_corr is entered here
        and not in the geoid_corrections.

        THE OPERATOR. Both RHS builders apply the SAME linear operator to whatever
        omega content they are given; only the field differs. Inverting the
        prefactors that Omega_eq1_RHS / Omega_eq2_RHS already carry on their
        known-good H terms gives, for an arbitrary omega,

            RHS1 += K1 * (Re/R) * R * [ -2*Re**3 * eta*omega
                                        + (Re/12) * Lap(eta*Te**2*omega) ]
            RHS2 += -(1 - nu) * Re**2 * Delta'( (eta*alpha) * omega )

        (Beuthe 2008 eqs 58 and 66: eta sits inside the Laplacians. The caller
        passes a_clm = eta*alpha, i.e. a_eta_clm.)

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
          * eq 1: eta*Te**2 multiplies omega on the grid, the Laplacian acts on
            that product at the output degree (see _eq1_halves).
          * eq 2: (eta*alpha) multiplies omega as a product of functions, formed
            on the grid; (alpha*omega)_lm != alpha_lm * omega_lm.
          * Delta' = -l(l+1) + 2 acts on the product (eta*alpha)*omega, at the
            output degree.
        """
        #   eq 2 uses Lap' = -l(l+1) + 2    (lap2_by_degree)
        lap2_by_degree = np.array([-l*(l + 1) + 2 for l in range(2*lmax + 1)])

        # eta*alpha grid (a_clm is a_eta_clm), constant throughout the solve
        a_grid   = self._cached_grid('om_a',   lmax,
                                lambda: a_clm.expand(lmax=self.grid_expansion_res).data)

        # omega as coefficients, on the 2*lmax+1 working band the builders use.
        om = np.zeros((2, 2*lmax + 1, 2*lmax + 1))
        band = min(2*lmax + 1, omega_corr_phys.shape[1])
        om[:, :band, :band] = omega_corr_phys[:, :band, :band]
        om_clm = pysh.SHCoeffs.from_array(om)

        om_grid = om_clm.expand(lmax=self.grid_expansion_res).data

        # FIELDS: eq-1 halves  eta*omega  and  Lap(eta*Te^2*omega)
        om_c1, om_c2 = self._eq1_halves(om_grid, T_e_parent, lmax)

        # FIELD: eq-2:  Delta'( (eta*alpha) * omega ).  Product on the grid, Delta' after
        a_om = pysh.SHGrid.from_array(a_grid * om_grid).expand()
        a_om = pysh.SHCoeffs.from_array(a_om.coeffs[:, :2*lmax + 1, :2*lmax + 1])
        for l in range(2*lmax + 1):
            a_om.coeffs[:, l, :] *= lap2_by_degree[l]

        # Prefactors
        Kalousova_scaler1 = 1.0/(self.E*self.T_e_0**3)
        corr1 = Re/R
        factor_eq1a = -2.0*Re**3 * Kalousova_scaler1 * corr1 * R
        factor_eq1b = (Re/12.0)  * Kalousova_scaler1 * corr1 * R
        factor_eq2  = -(1.0 - self.nu) * Re**2

        rhs1_add = (factor_eq1a * om_c1.coeffs[:, :lmax+1, :lmax+1]
                    + factor_eq1b * om_c2.coeffs[:, :lmax+1, :lmax+1])
        rhs2_add = factor_eq2 * a_om.coeffs[:, :lmax+1, :lmax+1]

        return (pysh.shio.SHCilmToVector(rhs1_add),
                pysh.shio.SHCilmToVector(rhs2_add))

    def Omega_eq1_RHS(self, topo_clm, geoid_clm, T_e_parent, lmax, R, T_e_0, Re, g0, mass, 
                      dc_pre = None):
        """
        Full set of terms for the Omega parameters of the first equation in the
        system of two equations. 

        At current stage, the SH function products are done by expansion
        into the spatial domain, performing the multiplication there and then
        transforming back to spatial domain. It should be possible to perform this
        product using the Gaunt coefficients too, which may be implemented at a 
        next stage.
        """

        RTcR = (R-self.T_c)/R
        g_m = g0 * (1.0 + (RTcR**3 - 1.0) * self.rho_c / self.rhobar) / RTcR**2    

        # Thin-shell correction factor
        corr1 = Re/R        # ETA FIELD FIX


        # (R-Tc)/R^(l+2) for degrees l
        # DC FILTER (moho, dc_lm branch): RTcR_l2 carries the eq-(1) moho
        # continuation, so dividing it by DCfilter_mohoD makes every 1/RTcR_l2 below pick
        # up x DCfilter_mohoD. See DCfilters.
        RTcR_l2 = (np.array([((R-self.T_c)/R)**(l+2) for l in range(2 * lmax + 1)])
                   / self.DCfilters(2 * lmax)[0])   # [0] = DCfilter_mohoD

        # 2l+1 for degrees l
        rhobar2l1 = np.array([ (self.rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])


        # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------    
        T_e_parent_grid_eq1RHS = T_e_parent.expand(lmax=self.grid_expansion_res).data
        topo_grid_eq1RHS = topo_clm.expand(lmax=self.grid_expansion_res).data - R
        geoid_grid_eq1RHS = geoid_clm.expand(lmax=self.grid_expansion_res).data - R

        # max(Te - Tc, 0) field
        TeTc_grid = T_e_parent_grid_eq1RHS - self.T_c
        # If a value is below 0, set to 0 to apply the 'max' call
        TeTc_grid_data = np.array(TeTc_grid.data)
        TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
        TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data

        # pre-weighted topo  H' = H / phi^(l+2)
        Hp = pysh.SHGrid.from_array(topo_grid_eq1RHS).expand()
        Hp = pysh.SHCoeffs.from_array(Hp.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(Hp.coeffs.shape[1]):
            Hp.coeffs[:, l, :] *= 1.0 / RTcR_l2[l]
        Hp_grid = Hp.expand(lmax=self.grid_expansion_res).data

        # pre-weighted geoid  G' = rhobar(2l+1)/phi^(l+2) * G
        Gp = pysh.SHGrid.from_array(geoid_grid_eq1RHS).expand()
        Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(Gp.coeffs.shape[1]):
            Gp.coeffs[:, l, :] *= rhobar2l1[l] / RTcR_l2[l]
        Gp_grid = Gp.expand(lmax=self.grid_expansion_res).data


        # ------- THE FIELDS FOR EACH TERM -------
        # Beuthe (2008) eq (58): every Omega content X enters as
        #     c1 = eta*X                (with the -2 Re^3 factors)
        #     c2 = Delta(eta*Te^2*X)    (with the  Re/12  factors)
        # eta and Te^2 sit INSIDE the Laplacian, which acts last at the output
        # degree. See _eq1_halves. The eta wrap (C_eta) is therefore gone from
        # solve_beuthe.

        # Fields RHS 1a/1b:  content Te*H
        TeH_grid = T_e_parent_grid_eq1RHS * topo_grid_eq1RHS
        TeH_clm, TeH_lap_Te2_clm = self._eq1_halves(TeH_grid, T_e_parent, lmax)

        # Fields dc1/dc2:  content max*H'
        dc1_clm, dc2_clm = self._eq1_halves(TeTc_grid * Hp_grid, T_e_parent, lmax)

        # Fields dc3/dc4:  content max*G'
        dc3_clm, dc4_clm = self._eq1_halves(TeTc_grid * Gp_grid, T_e_parent, lmax)

        # ---- prescribed root (forward mode) ----
        if dc_pre is not None and self.solve_for == 'G_lm':
             dc_pre_grid = self.truncate(dc_pre, lmax).expand(
                               lmax=self.grid_expansion_res).data
             # dcp1/dcp2:  content max*dc
             dcp1_clm, dcp2_clm = self._eq1_halves(TeTc_grid * dc_pre_grid,
                                                   T_e_parent, lmax)


        # ------- drho_lm VARIABLES AND FIELDS -------
        #  definitions of g_M, B_1 and B_2, and for the two fixes it encodes.)
        _dl   = self.drho_layer(lmax, R, g0, mass)
        M     = _dl['M']
        g_M   = _dl['g_M']
        B_1   = _dl['B_1']
        Cp    = _dl['Cp']
        # Te-dependent layer fields (kept local: they need T_e_parent_grid)
        TeMt_grid  = T_e_parent_grid_eq1RHS - self.Mt
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
            return weighted_coeffs.expand(lmax=self.grid_expansion_res).data

        def _c1_c2(prod_grid):
            """
            Each Omega_eq1_RHS term has two halves due to two omega terms in the eq.
            Calculate each drho-term half and return as coeffs.
            c1-half = eta * P_hat*drhom
            c2-half = Delta( eta*Te^2 * P_hat*drhom )
            """
            return self._eq1_halves(prod_grid, T_e_parent, lmax)

        # degree-weights for two H-field terms
        wH_drho = np.array([-Cp[l] / B_1[l] for l in range(2*lmax+1)])
        # degree-weights for two G-field terms
        wG_drho = np.array([ 1.0 / B_1[l]   for l in range(2*lmax+1)])

        topo_clm_eq1RHS  = pysh.SHGrid.from_array(topo_grid_eq1RHS).expand()
        geoid_clm_eq1RHS = pysh.SHGrid.from_array(geoid_grid_eq1RHS).expand()
        field_drho1, field_drho2 = _c1_c2(Phat_g * _drhom_part(topo_clm_eq1RHS,  wH_drho))
        field_drho3, field_drho4 = _c1_c2(Phat_g * _drhom_part(geoid_clm_eq1RHS, wG_drho))




        # ------- THE PREFACTORS OF THE EQ1 RHS OMEGA TERMS -------
        Kalousova_scaler1 = 1.0/(self.E*self.T_e_0**3)
        factor1a_omega = -2.0*Re**3 *self.rho_l *g0 *self.nu/(1.0-self.nu) * Kalousova_scaler1 * corr1
        factor1b_omega = self.rho_l * g0 * (Re/12.0) * self.nu/(1.0-self.nu) * Kalousova_scaler1 * corr1
        # Term b to be multiplied with Laplacian

        # factors from dc_lm inclusion
        factorRHS_omega1_dc1 = -2*Re**3*( self.nu/(1-self.nu)*g_m*self.rho_l ) * Kalousova_scaler1 * corr1
        factorRHS_omega1_dc2 = Re/12*( self.nu/(1-self.nu)*g_m*self.rho_l ) * Kalousova_scaler1 * corr1
        factorRHS_omega1_dc3 = 2*Re**3*( self.nu/(1-self.nu)*g_m ) * Kalousova_scaler1 * corr1
        factorRHS_omega1_dc4 = -Re/12 * ( self.nu/(1-self.nu)*g_m ) * Kalousova_scaler1 * corr1

        # Prescribed dc: the "+dc half" of  v1v*drho*g_m*max*(dc - w)/R.
        # Identical expressions to fdrho_w1/fdrho_w2 (Omega_eq1_LHS 1348-1349),
        # which carry the "-w half" -- that they match is the cross-check.
        factorRHS_omega1_dcp1 = -2*Re**3*( self.nu/(1-self.nu)*g_m*self.drho ) * Kalousova_scaler1 * corr1
        factorRHS_omega1_dcp2 =  Re/12  *( self.nu/(1-self.nu)*g_m*self.drho ) * Kalousova_scaler1 * corr1

        # factors from drho_lm inclusion
        # RHS = -K1*R*(c1 + c2*Delta)*omega_content, and the drhom weights
        # (-Cbar*rho_l/B_1 for H, +1/B_1 for G) are now inside field_drho1..4,
        # so only the operator halves remain here:
        #   c1-half: -2Re^3*K1*corr1        c2-half: +Re/12*K1*corr1
        factorRHS_omega1_drho1 = -2*Re**3 * (-0.5*self.nu/(1-self.nu)*g_M*self.rho_l) * Kalousova_scaler1 * corr1  # * P_hat*drhom_H
        factorRHS_omega1_drho2 =  Re/12   * (-0.5*self.nu/(1-self.nu)*g_M*self.rho_l) * Kalousova_scaler1 * corr1  # * Te2*lapl(P_hat*drhom_H)
        factorRHS_omega1_drho3 = -2*Re**3 * (-0.5*self.nu/(1-self.nu)*g_M) * Kalousova_scaler1 * corr1  # * P_hat*drhom_G
        factorRHS_omega1_drho4 =  Re/12   * (-0.5*self.nu/(1-self.nu)*g_M) * Kalousova_scaler1 * corr1  # * Te2*lapl(P_hat*drhom_G)



        # ------- ASSEMBLY -------
        Omega_RHS1_coeffs = ( factor1a_omega       * TeH_clm.coeffs[:, :lmax+1, :lmax+1]
                            + factor1b_omega       * TeH_lap_Te2_clm.coeffs[:, :lmax+1, :lmax+1]

                            + (factorRHS_omega1_dc1 * dc1_clm.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                            + (factorRHS_omega1_dc2 * dc2_clm.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                            + (factorRHS_omega1_dc3 * dc3_clm.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                            + (factorRHS_omega1_dc4 * dc4_clm.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)

                            + (factorRHS_omega1_drho1 * field_drho1.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)
                            + (factorRHS_omega1_drho2 * field_drho2.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0) 
                            + (factorRHS_omega1_drho3 * field_drho3.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)
                            + (factorRHS_omega1_drho4 * field_drho4.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)

                            + (factorRHS_omega1_dcp1 * dcp1_clm.coeffs[:, :lmax+1, :lmax+1] if (dc_pre is not None and self.solve_for == 'G_lm') else 0)
                            + (factorRHS_omega1_dcp2 * dcp2_clm.coeffs[:, :lmax+1, :lmax+1] if (dc_pre is not None and self.solve_for == 'G_lm') else 0))

        # Then transform to an 'unstructured' vector (structure same as that of y in
        # solve_beuthe) 
        Omega_RHS1_unstr = pysh.shio.SHCilmToVector(Omega_RHS1_coeffs)

        return Omega_RHS1_unstr

    def Omega_eq1_LHS(self, T_e_parent, lmax, R, T_e_0, Re, g0, mass):
        """ 
        Compute the spherical harmonic function field products and the prefactors
        for the LHS integration of the omega coefficients of the first equation.
        """

        RTcR = (R-self.T_c)/R
        g_m = g0 * (1.0 + (RTcR**3 - 1.0) * self.rho_c / self.rhobar) / RTcR**2    

        # Thin-shell approximations/correction
        corr1 = Re/R            # ETA FIELD FIX

        # ------- PRECOMPUTED SH-MULTIPLIED FIELDS -------   
        T_e_parent_grid_eq1LHS = T_e_parent.expand(lmax=self.grid_expansion_res).data

        # Field Te
        Te_grid = T_e_parent_grid_eq1LHS 
        Te_clm = pysh.SHGrid.from_array(Te_grid).expand()
        Te_clm = pysh.SHCoeffs.from_array(Te_clm.coeffs[:, :2*lmax+1, :2*lmax+1])  

        # Field Te^2 
        Te2_grid = T_e_parent_grid_eq1LHS**2     
        Te2_clm = pysh.SHGrid.from_array(Te2_grid).expand()
        Te2_clm = pysh.SHCoeffs.from_array(Te2_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

        # Field max(Te-Tc,0)
        TeTc_grid = T_e_parent_grid_eq1LHS - self.T_c
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
        rho_grid = np.where(T_e_parent_grid_eq1LHS <= self.T_c, self.rho_c, self.rho_m)
        # Calculate the final gTe_grid using the dynamic rho_grid
        gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / self.rhobar) / RTeR_grid**2    

        # Field Tc if Tc < Te else 0
        Tcind_grid_1 = np.where(T_e_parent_grid_eq1LHS > self.T_c, self.T_c, 0.0)
        Tcind_clm_1  = pysh.SHGrid.from_array(Tcind_grid_1).expand()
        Tcind_clm_1  = pysh.SHCoeffs.from_array(Tcind_clm_1.coeffs[:, :2*lmax+1, :2*lmax+1])


        # ------- THE PREFACTORS OF THE EQ1 RHS OMEGA TERMS -------
        Kalousova_scaler1 = 1.0/(self.E*self.T_e_0**3)

        # 1a & 1d change to take difference in rho_l and rho_c and the surface gravity
        # 1c and 1f changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
        factorLHS_omega1a = -2*Re**3*self.drhol*g0*self.nu/(1-self.nu) * Kalousova_scaler1 *corr1


        # Te<Tc FIX (DSP eq 5: rhoc*gmoho*(Tc if Tc < Te else 0)): the crustal-
        # column term must vanish wherever the crust-mantle interface lies BELOW
        # the elastic shell. Implemented as the indicator field
        #     Tcind(theta,phi) = Tc * 1[Te > Tc],
        # which reduces to the constant Tc (or 0) monopole at constant Te, and
        # handles variable-Te maps that locally dip below Tc. NOTE: for maps
        # crossing Tc the indicator is discontinuous -> spectral ringing near
        # the Te = Tc contour is inherent; consider a smooth Te map or a
        # tapered indicator if that contour matters.
        factorLHS_omega1b = 2*Re**3*self.rho_c*g_m * Kalousova_scaler1 *corr1  # Tc in field
        factorLHS_omega1c = 2*Re**3*self.rho_m * Kalousova_scaler1 *corr1      # gTe in field

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
        factorLHS_omega1d = Re/12*self.drhol*g0*self.nu/(1-self.nu) * Kalousova_scaler1 *corr1   # *Laplacian!
        factorLHS_omega1e = -Re/12*g_m*self.rho_c * Kalousova_scaler1  *corr1          # *Laplacian! (Tc -> Tcind field)
        factorLHS_omega1f = -Re/12*self.rho_m * Kalousova_scaler1  *corr1              # *Laplacian! (gTe -> gTemax field)


        factor_fdc1_w =  2*Re**3 * self.nu/(1-self.nu) * g_m * self.drhol * Kalousova_scaler1 * corr1
        factor_fdc2_w = -Re/12   * self.nu/(1-self.nu) * g_m * self.drhol * Kalousova_scaler1 * corr1

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
        _dl   = self.drho_layer(lmax, R, g0, mass)
        M     = _dl['M']  
        g_M = _dl['g_M']
        B_1   = _dl['B_1']
        B_2 = _dl['B_2']
        Cp    = _dl['Cp']
        # Te-dependent layer fields (kept local: they need T_e_parent_grid)
        TeMt_grid  = T_e_parent_grid_eq1LHS - self.Mt
        TeMt0      = np.array(TeMt_grid.data)
        TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
        MTeMt      = np.array(TeMt_grid.data)
        MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)


        RTcR_l2 = np.array([RTcR**(l+2) for l in range(2*lmax+1)])
        RTcR_l  = np.array([RTcR**l     for l in range(2*lmax+1)])
        Dw_arr = -Cp * (self.drhol + self.drho*RTcR_l2) / B_1

        Lam_q_drho_arr = ( g0*self.drhol 
                           + g_m*self.drho
                           - g0*self.drho*Cp*self.drhol*RTcR_l
                           - g0*self.drho**2*Cp*RTcR
                           + (g_M*M - g0*self.drho*B_2) * Dw_arr )

        # STEP 1: dc-branch q w-coupling, precomputed here (was inline in
        # solve_beuthe). Lam_q_dc = q_H with rho_l -> drhol; zero iff rho_l==rho_c.
        # DC FILTER: negl2 IS 1/P2 -> x F1 ; nl1 IS phi/P2 -> x F1/F2.
        # (RTcR_l2 above belongs to the DRHO branch's Dw_arr and is NOT filtered
        #  here -- in that branch DSP sets DCfilter_mohoD = ones.)
        DCfilter_mohoD, DCfilter_mohoDc, _, _ = self.DCfilters(2*lmax)
        RTcR_nl1     = (np.array([RTcR**(-(l+1)) for l in range(2*lmax+1)])
                        * DCfilter_mohoD / DCfilter_mohoDc)
        RTcR_negl2   = (np.array([RTcR**(-(l+2)) for l in range(2*lmax+1)])
                        * DCfilter_mohoD)
        Lam_q_dc_arr = ( g0*self.drhol 
                         - g_m*self.drhol*RTcR_negl2
                         - g0*self.drho*self.drhol*Cp*(RTcR_l - RTcR_nl1) )

        # P_hat as a pure GRID field (no 1/B_1 baked in -- that lives in Dw_arr)
        Phat_g   = TeMt0 * MTeMt
        Phat_clm = pysh.SHGrid.from_array(Phat_g).expand()
        Phat_clm = pysh.SHCoeffs.from_array(Phat_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

        fdrho_w1 = -2*Re**3 * self.nu/(1-self.nu) * g_m * self.drho * Kalousova_scaler1 * corr1
        fdrho_w2 =  Re/12   * self.nu/(1-self.nu) * g_m * self.drho * Kalousova_scaler1 * corr1
        fdrho_om1 = 2*Re**3 * -0.5 * self.nu/(1-self.nu) * g_M * Kalousova_scaler1 * corr1
        fdrho_om2 = -Re/12  * -0.5 * self.nu/(1-self.nu) * g_M * Kalousova_scaler1 * corr1

        # ---- eta placement (Beuthe 2008 eq 58) ------------------------------
        # c1 halves carry the field  eta*X,  c2 halves the field  eta*Te^2*X,
        # with the Laplacian applied OUTSIDE (in build_A_tilde_group2). eta is
        # folded into the fields here, so solve_beuthe no longer wraps the
        # Omega blocks in conv(eta).
        eta_g, etaTe2_g = self._eta_fields(T_e_parent)
        vec = lambda g: pysh.shio.SHCilmToVector(self._expand_band(g, lmax).coeffs)

        # ---- group-1 LHS vectors (c1 halves):  eta * X
        Omega_LHS_1a_unstr = factorLHS_omega1a * vec(eta_g * Te_grid)
        Omega_LHS_1b_unstr = factorLHS_omega1b * vec(eta_g * Tcind_grid_1)
        Omega_LHS_1c_unstr = factorLHS_omega1c * vec(eta_g * gTe_grid * TeTc_grid)

        # ---- group-2 ingredients: FACTORS + separate inner field vectors ------
        gTemax_clm = pysh.SHGrid.from_array(gTe_grid * TeTc_grid).expand()
        gTemax_clm = pysh.SHCoeffs.from_array(gTemax_clm.coeffs[:, :2*lmax+1, :2*lmax+1])
        g2 = dict(
            f1d          = factorLHS_omega1d,
            f1e          = factorLHS_omega1e,
            f1f          = factorLHS_omega1f,
            fdc1_w       = factor_fdc1_w,
            fdc2_w       = factor_fdc2_w,
            # c2 fields eta*Te^2*X (Laplacian applied outside), c1 fields eta*X
            Te_unstr      = vec(etaTe2_g * Te_grid),                 # c2, X = Te (1d)
            max_unstr     = vec(etaTe2_g * TeTc_grid),               # c2, X = max (dc pair / -w half)
            max_c1_unstr  = vec(eta_g    * TeTc_grid),               # c1, X = max (dc pair / -w half)
            Tcind_unstr   = vec(etaTe2_g * Tcind_grid_1),            # c2, X = Tcind (1e)
            gTemax_unstr  = vec(etaTe2_g * gTe_grid * TeTc_grid),    # c2, X = gTe*max (1f)
            # shared scalars/grids -- single source of truth for solve_beuthe
            g_m          = g_m,
            RTcR         = RTcR,
            TeTc_grid    = TeTc_grid,
            # ---- drho_lm branch ingredients (unused when solve_for=='dc_lm') --
            w_corr           = Dw_arr,             # per-degree: drhom's w-coefficient
            Lam_q_drho   = Lam_q_drho_arr,     # per-degree: q's w-coupling (diagonal)
            Lam_q_dc     = Lam_q_dc_arr,       # per-degree: q's w-coupling (dc branch)
            # eq-2 dc coupling factor (scalar); field/weights assembled in solver
            fdc_2d       = self.nu * g_m * self.drhol * (Re**2/R),
            Phat_unstr    = vec(eta_g    * Phat_g),                  # c1, X = P_hat (drho)
            Phat_c2_unstr = vec(etaTe2_g * Phat_g),                  # c2, X = P_hat (drho)
            fdrho_w1     = fdrho_w1,           # -w half of (dc-w), c1 (field: max)
            fdrho_w2     = fdrho_w2,           # -w half of (dc-w), c2 (field: max)
            fdrho_om1    = fdrho_om1,          # omega/drhom c1 scale
            fdrho_om2    = fdrho_om2,          # omega/drhom c2 scale

            # ---- M6 forward branch: q's w-coupling with G and Gc substituted
            # FORWARD out of eqs (1)/(2) instead of eliminated. Terms 3 and 4 of
            # Lam_q_drho are reused verbatim; the -g0*rho_l*Cp*(...) piece is the
            # only new one. See lam_q_forward.
            Lam_q_fwd    = self.lam_q_forward(lmax, R, g0, mass),
        )

        return (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr,
                g2)

    def Omega_eq2_RHS(self, topo_clm, geoid_clm, T_e_parent, a_clm, lmax, R, T_e_0, Re, g0, mass, 
                      dc_pre=None):
        """
        Full set of terms for the Omega parameters of the second equation in the
        system of two equations.

        At current stage, the SH function products are done by expansion
        into the spatial domain, performing the multiplication there and then
        transforming back to spatial domain. It should be possible to perform this
        product using the Gaunt coefficients too, which may be implemented at a 
        next stage.
        """

        RTcR = (R-self.T_c)/R
        g_m = g0 * (1.0 + (RTcR**3 - 1.0) * self.rho_c / self.rhobar) / RTcR**2    

        # Laplacian array for degrees (incl +2 term)
        lap2_by_degree = np.array([-l * (l + 1) +2 for l in range(2 * lmax + 1)])

        # (R-Tc)/R^(l+2) for degrees l
        # DC FILTER (moho, dc_lm branch): RTcR_l2 carries the eq-(1) moho
        # continuation, so dividing it by DCfilter_mohoD makes every 1/RTcR_l2 below pick
        # up x DCfilter_mohoD. See DCfilters.
        RTcR_l2 = (np.array([((R-self.T_c)/R)**(l+2) for l in range(2 * lmax + 1)])
                   / self.DCfilters(2 * lmax)[0])   # [0] = DCfilter_mohoD

        # 2l+1 for degrees l
        rhobar2l1 = np.array([ 3/ (self.rhobar*(2*l+1)) for l in range(2 * lmax + 1)])

        # SH-MULTIPLIED FIELDS        
        T_e_parent_grid_eq2RHS = T_e_parent.expand(lmax=self.grid_expansion_res).data
        topo_grid_eq2RHS = topo_clm.expand(lmax=self.grid_expansion_res).data - R
        geoid_grid_eq2RHS = geoid_clm.expand(lmax=self.grid_expansion_res).data - R
        alpha_grid_eq2RHS = a_clm.expand(lmax=self.grid_expansion_res).data

        # Te - Tc field
        TeTc_grid = T_e_parent_grid_eq2RHS - self.T_c
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
        Hp_grid = Hp.expand(lmax=self.grid_expansion_res).data

        Gp = pysh.SHGrid.from_array(geoid_grid_eq2RHS).expand()
        Gp = pysh.SHCoeffs.from_array(Gp.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(Gp.coeffs.shape[1]): 
            Gp.coeffs[:, l, :] *= 1/( rhobar2l1[l] * RTcR_l2[l] )
        Gp_grid = Gp.expand(lmax=self.grid_expansion_res).data

        d_dc1 = pysh.SHGrid.from_array(TeTc_grid * Hp_grid * alpha_grid_eq2RHS).expand()   # max*H'*alpha
        d_dc1 = pysh.SHCoeffs.from_array(d_dc1.coeffs[:, :2*lmax+1, :2*lmax+1])
        d_dc2 = pysh.SHGrid.from_array(TeTc_grid * Gp_grid * alpha_grid_eq2RHS).expand()   # max*G'*alpha
        d_dc2 = pysh.SHCoeffs.from_array(d_dc2.coeffs[:, :2*lmax+1, :2*lmax+1])
        for l in range(d_dc1.coeffs.shape[1]):       # Laplacian+2 on the inner product (as in 2a)
            d_dc1.coeffs[:, l, :] *= lap2_by_degree[l]
            d_dc2.coeffs[:, l, :] *= lap2_by_degree[l]

        if dc_pre is not None and self.solve_for == 'G_lm':
            dc_pre_grid2 = self.truncate(dc_pre, lmax).expand(
                               lmax=self.grid_expansion_res).data
            d_dcp = pysh.SHGrid.from_array(
                        TeTc_grid * dc_pre_grid2 * alpha_grid_eq2RHS).expand()
            d_dcp = pysh.SHCoeffs.from_array(
                        d_dcp.coeffs[:, :2*lmax+1, :2*lmax+1])
            for l in range(d_dcp.coeffs.shape[1]):
                d_dcp.coeffs[:, l, :] *= lap2_by_degree[l]


        # drho_lm terms
        _dl   = self.drho_layer(lmax, R, g0, mass)
        M     = _dl['M']  
        g_M = _dl['g_M']
        B_1   = _dl['B_1']
        Cp    = _dl['Cp']  
        RTcR_l2 = np.array([((R-self.T_c)/R)**(l+2) for l in range(2*lmax+1)])
        # Te-dependent layer fields (kept local: they need T_e_parent_grid)
        TeMt_grid  = T_e_parent_grid_eq2RHS - self.Mt
        TeMt0      = np.array(TeMt_grid.data)
        TeMt0[TeMt0 < 0.0] = 0.0                       # max(Te - Mt, 0)
        MTeMt      = np.array(TeMt_grid.data)
        MTeMt[MTeMt > M] = M                           # min(Te - Mt, M)


        P_frac = -0.5 * self.nu/(1-self.nu) * g_M * MTeMt * TeMt0
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
            return c.expand(lmax=self.grid_expansion_res).data

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
        factor2a_omega = -1.0* self.nu * self.rho_l * g0 * Kalousova_scaler2  # *(Laplacian+2)

        factorRHS_omega2_dc1 = -self.nu*g_m*self.rho_l * Kalousova_scaler2
        factorRHS_omega2_dc2 = self.nu*g_m * Kalousova_scaler2

        factorRHS_omega2_drho1 = (0.5*self.nu*g_M*self.rho_l) * Kalousova_scaler2   # * Delta'(alpha*P_hat*drhom_H)
        factorRHS_omega2_drho2 = (0.5*self.nu*g_M) * Kalousova_scaler2   # * Delta'(alpha*P_hat*drhom_G)

        factorRHS_omega2_dcp = -self.nu*self.drho*g_m * Kalousova_scaler2

        # ------ ASSEMBLY ------
        Omega_RHS2_coeffs = ( 
                               factor2a_omega       * TeHa_lap.coeffs[:, :lmax+1, :lmax+1]
                            + (factorRHS_omega2_dc1 * d_dc1.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                            + (factorRHS_omega2_dc2 * d_dc2.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                            + (factorRHS_omega2_drho1 * d_drho1.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)
                            + (factorRHS_omega2_drho2 * d_drho2.coeffs[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)
                            + (factorRHS_omega2_dcp * d_dcp.coeffs[:, :lmax+1, :lmax+1] if (dc_pre is not None and self.solve_for == 'G_lm') else 0)
                            )

        # Then transform to an 'unstructured' vector (structure same as that of y in
        # solve_beuthe) 
        Omega_RHS2_unstr = pysh.shio.SHCilmToVector(Omega_RHS2_coeffs)

        return Omega_RHS2_unstr 

    def Omega_eq2_LHS(self, T_e_parent, a_clm, lmax, R, T_e_0, Re, g0, mass):
        """ 
        Compute the spherical harmonic function field products and the prefactors
        for the LHS integration of the omega coefficients of the second equation.

        A number of Te and alpha products occur in the LHS terms. These can be
        simplified, since alpha = 1/(E*Te), thus reducing to 1/E for the product.
        """

        RTcR = (R-self.T_c)/R
        g_m = g0 * (1.0 + (RTcR**3 - 1.0) * self.rho_c / self.rhobar) / RTcR**2    

        # SH-MULTIPLIED FIELDS        
        T_e_parent_grid_eq2LHS = T_e_parent.expand(lmax=self.grid_expansion_res).data
        a_grid_eq2LHS = a_clm.expand(lmax=self.grid_expansion_res).data

        # gTe FIELD (variable-Te fix): gravity at the LOCAL shell-base depth,
        # mantle branch only -- every gTe-carrying term also carries max(Te-Tc,0),
        # which is zero exactly where the density branch would switch. Monopole at
        # constant Te => benchmark preserved.
        RTeR_grid = (R - T_e_parent_grid_eq2LHS) / R
        # Create a dynamic rho grid based on the local thickness threshold
        rho_grid = np.where(T_e_parent_grid_eq2LHS <= self.T_c, self.rho_c, self.rho_m)
        # Calculate the final gTe_grid using the dynamic rho_grid
        gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / self.rhobar) / RTeR_grid**2    

        # Te - Tc field
        TeTc_grid = T_e_parent_grid_eq2LHS - self.T_c
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
        Tcind_grid_2 = np.where(T_e_parent_grid_eq2LHS > self.T_c, self.T_c, 0.0)    
        Tcinda_clm   = pysh.SHGrid.from_array(Tcind_grid_2 * a_grid_eq2LHS.data).expand()
        Tcinda_clm   = pysh.SHCoeffs.from_array(Tcinda_clm.coeffs[:, :2*lmax+1, :2*lmax+1])

        # Field 2c: max(Te-Tc,0) * alpha
        gTeTeTca_grid = gTe_grid * TeTc_grid * a_grid_eq2LHS  # gTe grid folded into here for variable Te
        gTeTeTca_clm  = pysh.SHGrid.from_array(gTeTeTca_grid).expand()
        gTeTeTca_clm  = pysh.SHCoeffs.from_array(gTeTeTca_clm.coeffs[:, :2*lmax+1, :2*lmax+1])



        # ------- drho_lm VARIABLES AND FIELDS -------
        _dl   = self.drho_layer(lmax, R, g0, mass)
        M     = _dl['M']
        g_M = _dl['g_M']

        # Te - Mt field
        TeMt_grid  = T_e_parent_grid_eq2LHS - self.Mt
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
        factorLHS_omega2a = -self.drhol*g0*self.nu * Kalousova_scaler2         # *Laplacian+2!
        factorLHS_omega2b = (1-self.nu)*self.rho_c*g_m * Kalousova_scaler2 # *Laplacian+2! (Tc in field)

        # 2c changed to comply with DSP eq 4 w-term of Omega --> g_m to gTe
        factorLHS_omega2c = (1-self.nu)*self.rho_m * Kalousova_scaler2 # *Laplacian+2! # gTe in field 

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
        factorLHS_omega2d = (0.0 if self.solve_for == 'dc_lm'
                                else -self.nu*self.drho*g_m*Kalousova_scaler2)

        factorLHS_omega2e = (1-self.nu) *-0.5 * self.nu/(1-self.nu) * g_M * Kalousova_scaler2

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

    def q_lm(self, topo_clm, geoid_clm, lmax, R, T_e_0, Re, g0, mass):
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

        RTcR = (R-self.T_c)/R
        g_m = g0 * (1.0 + (RTcR**3 - 1.0) * self.rho_c / self.rhobar) / RTcR**2   

        # (R-Tc)/R^(l+2) for degrees l
        # DC FILTER (moho, dc_lm branch): RTcR_l2 carries the eq-(1) moho
        # continuation, so dividing it by DCfilter_mohoD makes every 1/RTcR_l2 below pick
        # up x DCfilter_mohoD. See DCfilters.
        # NOTE: RTcR_l2 is REDEFINED further down for the drho_lm block, so any
        # dc-branch use BELOW that point must not rely on this name. The degree-1
        # block near the end of this function was doing exactly that, and the
        # filter silently did not reach degree 1. DCfilter_mohoD is kept under its
        # own name for that reason.
        DCfilter_mohoD = self.DCfilters(2 * lmax)[0]
        RTcR_l2 = (np.array([((R-self.T_c)/R)**(l+2) for l in range(2 * lmax + 1)])
                   / DCfilter_mohoD)

        # (R-Tc)/R^(l+1) for degrees l
        RTcR_l1 = np.array([((R-self.T_c)/R)**(l) for l in range(2 * lmax + 1)])
        # --->> CHANGED FROM (l+1) TO (l) TO ALIGN WITH DSP

        # (R-Tc)/R^(-l+1) for degrees l
        # DC FILTER: phi**(-l-1) IS phi/P2, so it carries BOTH moho filters.
        # Building it from the already-filtered RTcR_l2 inherits F1 automatically;
        # dividing by F2 supplies the eq-(2) half.  -> x F1/F2, as derived in
        # DCfilters.
        RTcR_negl1 = RTcR / (RTcR_l2 * self.DCfilters(2 * lmax)[1])  # [1] = mohoDc
        # --->> CHANGED FROM (-l+1) TO (-l-1) TO ALIGN WITH DSP

        # 2l+1 for degrees l
        rhobar2l1 = np.array([ (self.rhobar*(2*l+1))/3 for l in range(2 * lmax + 1)])

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
        _dl   = self.drho_layer(lmax, R, g0, mass)
        M     = _dl['M'] 
        g_M = _dl['g_M']
        B_1   = _dl['B_1']
        B_2 = _dl['B_2']
        Cp    = _dl['Cp']
        RTcR_l2 = np.array([((R-self.T_c)/R)**(l+2) for l in range(2*lmax+1)])


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
        Kalousova_scaler1 = 1.0/(self.E*self.T_e_0**3)
        q_topo_term1  = g0*self.rho_l            # No SH field
        q_geoid_term1 = -g0*self.rho_l           # No SH field

        q_topo_dc1    = -g_m*self.rho_l          # / RTcRl2
        q_topo_dc2    = -g0*self.drho*self.rho_l      # * RTcRl1/rhobar2l1
        q_topo_dc3    = g0*self.drho*self.rho_l       # * RTcR_nl1/rhobar2l1
        q_geoid_dc4   = g_m                 # *rhobar2l1 / RTcRl2
        q_geoid_dc5   = -g0*self.drho            # * RTcR_nl1

        q_topo_drho1  = -g0 * self.drho * self.rho_l  # * Cp * RTcR_l1 * topo
        q_topo_drho2  =  g0 * self.drho * self.rho_l  # * Cp * B_2/B_1 * topo
        q_topo_drho3  = -g_M * M * self.rho_l    # * Cp / B_1 * topo
        q_geoid_drho4 =  g_M * M            # * 1 / B_1 * geoid
        q_geoid_drho5 = -g0 * self.drho          # * B_2 / B_1 * geoid


        # Make coeffs array of size lmax+1 for the RHS
        q_coeffs = -Re**4 * Kalousova_scaler1 * (
                      q_topo_term1 
                       * topo_clm_copyq.coeffs[:, :lmax+1, :lmax+1]
                    + q_geoid_term1
                       * geoid_clm_copyq.coeffs[:, :lmax+1, :lmax+1] 

                    + (q_topo_dc1 
                       * field_topo_dc1[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                    + (q_topo_dc2
                       * field_topo_dc2[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                    + (q_topo_dc3 
                       * field_topo_dc3[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                    + (q_geoid_dc4
                       * field_geoid_dc4[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)
                    + (q_geoid_dc5
                       * field_geoid_dc5[:, :lmax+1, :lmax+1] if self.solve_for == 'dc_lm' else 0)

                    + (q_topo_drho1
                       * field_topo_drho1[:, :lmax+1, :lmax+1]  if self.solve_for == 'drho_lm' else 0)
                    + (q_topo_drho2
                       * field_topo_drho2[:, :lmax+1, :lmax+1]  if self.solve_for == 'drho_lm' else 0)
                    + (q_topo_drho3
                       * field_topo_drho3[:, :lmax+1, :lmax+1]  if self.solve_for == 'drho_lm' else 0)
                    + (q_geoid_drho4
                       * field_geoid_drho4[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)
                    + (q_geoid_drho5
                       * field_geoid_drho5[:, :lmax+1, :lmax+1] if self.solve_for == 'drho_lm' else 0)
                    ) 

        # ---- degree 1: DSP enforces the COM constraint Gc_1 = 0 (thinshell.py eq(2),
        # "Force the degree-1 geoid to zero"), so the Gc-elimination terms must be
        # dropped here. Harmless to the solve (solve_beuthe zeroes the l=0,1 rows of rhs),
        # but does affect compute_omega and the stress-strain calculations
        H1 = topo_clm_copyq.coeffs[:, 1, :2]
        G1 = geoid_clm_copyq.coeffs[:, 1, :2]

        if self.solve_for == 'dc_lm':
            # DC FILTER: eq (1) at l = 1 with w_1 = 0, so dc1 picks up
            # x DCfilter_mohoD[1] like every other eq-(1) elimination.
            # RTcR**3 is written out rather than reusing RTcR_l2, because that
            # name has been REBOUND to the unfiltered drho-branch array above --
            # using it here left degree 1 unfiltered and produced a degree-1
            # Omega/A-S mismatch that appeared only in the dc_lm branch with the
            # filter on.
            dc1      = ((self.rho_l * H1 - self.rhobar * G1) * DCfilter_mohoD[1]
                        / (self.drho * RTcR**3))                          # eq(1), w_1 = 0
            q_phys_1 = g0 * self.rho_l * (H1 - G1) - g_m * self.drho * dc1        # eq(3), Gc_1 = w_1 = 0
            q_coeffs[:, 1, :2] = -Re**4 * Kalousova_scaler1 * q_phys_1

        if self.solve_for == 'drho_lm':
            q_phys_1 =  (g0*self.rho_l*(H1-G1)   + q_topo_drho3 * field_topo_drho3[:, 1, :2]
                            + q_geoid_drho4 * field_geoid_drho4[:, 1, :2])        
            q_coeffs[:, 1, :2] = -Re**4 * Kalousova_scaler1 * q_phys_1

        q_lm_unstr = pysh.shio.SHCilmToVector(q_coeffs)

        return q_lm_unstr

    def q_lm_fwd(self, topo_clm, lmax, R, T_e_0, Re, g0, mass, dc_pre=None):
        """
        Forward-mode loading term. DSP eq (3) with dc_lm = 0, drho_lm = 0, and
        both G_lm and Gc_lm substituted forward from eqs (1) and (2).

            q_phys = g0*rho_l*(H - G) + g0*drhol*w + g_m*drho*(w - Gc)

        Only the H part is returned here. The w part is a per-degree diagonal and
        belongs on the LHS -- see `lam_q_forward` -- exactly as `Lam_q_dc` /
        `Lam_q_drho` do in the existing branches.

        Returns the SCALED vector -Re^4/(E*T_e_0^3) * q_phys, same convention as
        q_lm, so solve_beuthe's q-correction hook needs no change.

        DEGREE 1 IS NOT THE GENERAL FORMULA. DSP zeroes eq (1) at l = 1 whenever
        G_lm is an unknown (thinshell.py 984) and eq (2) whenever Gc_lm is
        (line 1018), so in forward mode BOTH vanish and the COM gauge sets
        G_1 = Gc_1 = 0. Eq (3) at l = 1 therefore reduces to the bare

            q_phys_1 = g0*rho_l*H_1

        which at the target config is 10810.76*H_1 against 1221.68*H_1 from the
        general formula -- a factor 8.85. It does not reach w (solve_beuthe pins
        the l = 0,1 rows of the RHS) but it DOES reach Omega_1 through
        compute_Omega's degree-1 override, and thence every stress and strain
        field. Same failure mode as the degree-1 filter bug of handover Part 4.

        dc_pre : prescribed crustal-root map, or None for dc = 0. Its weight
        collects the direct -g_m*drho*dc of eq (3) plus the feedback through
        the dc terms that eq (1) and eq (2) put into G and Gc.
        """

        topo_q = topo_clm.copy()
        topo_q.coeffs[0, 0, 0] = 0.0            # relief, not shape

        phi    = (R - self.T_c) / R
        g_m    = g0 * (1.0 + (phi**3 - 1.0) * self.rho_c / self.rhobar) / phi**2

        degs  = np.arange(lmax + 1, dtype=float)
        Cp    = 3.0 / (self.rhobar * (2.0*degs + 1.0))       # DSP's rhobconst
        phil  = phi**degs                               # phi^l   (eq 2 surface)
        phil2 = phi**(degs + 2.0)                       # phi^(l+2) (eq 1 moho)

        q_H = g0 * self.rho_l * (1.0 - Cp*self.rho_l - self.drho*Cp*phil)

        # dc weight:  direct term of eq (3), plus  -g0*rho_l*(dG/ddc)
        #                                     and  -g_m*drho*(dGc/ddc)
        q_dc = -g_m*self.drho + g0*Cp*self.drho*(self.rho_l*phil2 + self.drho*phi)

        q_coeffs = np.zeros((2, lmax+1, lmax+1))
        for l in range(lmax + 1):
            q_coeffs[:, l, :] = q_H[l] * topo_q.coeffs[:, l, :lmax+1]

        if dc_pre is not None:
            dc_t = self.truncate(dc_pre, lmax).coeffs
            for l in range(lmax + 1):
                q_coeffs[:, l, :] += q_dc[l] * dc_t[:, l, :lmax+1]

        # degree 1: eq (3) with w_1 = Gc_1 = 0 (the COM convention q_lm uses in
        # the dc branch) and G_1 from eq (1), the value compute_geoid returns.
        # For a root that balances H_1 (every dc an inverse returns) G_1 = 0 and
        # this is  g0*rho_l*H_1 - g_m*drho*dc_1,  i.e. q_lm's own degree 1.
        # Reaches only Omega_1 (compute_Omega) and the stress/strain fields.
        H1  = topo_q.coeffs[:, 1, :2]
        dc1 = (self.truncate(dc_pre, lmax).coeffs[:, 1, :2] if dc_pre is not None
               else np.zeros_like(H1))
        G1  = Cp[1] * (self.rho_l*H1 - self.drho*dc1*phil2[1])
        q_coeffs[:, 1, :2] = g0*self.rho_l*(H1 - G1) - g_m*self.drho*dc1

        q_coeffs *= -Re**4 / (self.E * self.T_e_0**3)

        return pysh.shio.SHCilmToVector(q_coeffs)

    def lam_q_forward(self, lmax, R, g0, mass):
        """
        q's w-coupling in forward mode: a per-degree diagonal for the eq-1 LHS.

        Consumed by solve_beuthe exactly like Lam_q_dc / Lam_q_drho:
            A[idx, idx] += Re**4 * scaler_A * Lam_q_arr[l]
        (the sign follows from q entering the RHS as -Re^4*K1*q_phys, so moving
        the w part across flips it -- unchanged from the existing branches).

        Built to 2*lmax like the other degree arrays in this file so it can be
        indexed by mode_map degrees without a second bound.
        """
        phi    = (R - self.T_c) / R
        g_m    = g0 * (1.0 + (phi**3 - 1.0) * self.rho_c / self.rhobar) / phi**2

        degs  = np.arange(2*lmax + 1, dtype=float)
        Cp    = 3.0 / (self.rhobar * (2.0*degs + 1.0))
        phil  = phi**degs                 # phi^l      (eq 2, surface continuation)
        phil2 = phi**(degs + 2.0)         # phi^(l+2)  (eq 1, moho continuation)

        return ( g0*self.drhol + g_m*self.drho
                 - g0*self.rho_l*Cp*(self.drhol + self.drho*phil2)      # from  -g0*rho_l*G
                 - g0*self.drho *Cp*(self.drhol*phil + self.drho*phi) ) # from  -g_m*drho*Gc

    def build_A_tilde_group2(self, Te_unstr, max_unstr, max_c1_unstr,
                             f1d, f1e, f1f,
                             gidx, gaunt_bare, starts, seg_len, ci, cj, mode_map, N,
                             fdc1_w=0.0, fdc2_w=0.0, Pw=None,
                             Tcind_unstr=None, gTemax_unstr=None):
        """
        The c2 halves of the eq-1 Omega operator of Beuthe (2008) eq (58),

            (R/12) Delta( eta*Te^2 * X * w ),

        returned as a (generally non-symmetric) dense N x N matrix that is added
        straight into A. The fields passed in ALREADY carry eta*Te^2 (built in
        Omega_eq1_LHS), so the operator is simply  Delta_out @ conv(field):
            1d: X = Te      1e: X = Tcind      1f: X = gTe*max

        The dc / -w pair (X = max, operand weight Pw) also carries its c1 half,
        eta*max*Pw*w, through max_c1_unstr:
            fdc1_w * conv(eta*max)            @ Pw    (c1 half)
            fdc2_w * Delta @ conv(eta*Te^2*max) @ Pw  (c2 half)
        dc branch: Pw = diag(phi^-(l'+2)), factors carry drhol (zero iff
        rho_l == rho_c). drho / forward branches: Pw = I, factors are the
        standalone -w half of v1v*drho*g_m*max*(dc - w)/R.

        At constant Te every field is a monopole and this reduces exactly to the
        previous  eta0 * Te0^2 * Delta  form.
        """
        conv = lambda v: build_conv_matrix(v, gidx, gaunt_bare, starts,
                                           seg_len, ci, cj, N)
        Dlap = np.diag(np.array([-l*(l+1) for l, _ in mode_map], dtype=np.float64))

        M = Dlap @ (f1d*conv(Te_unstr) + f1e*conv(Tcind_unstr)
                    + f1f*conv(gTemax_unstr))             # NOT symmetric -- keep full
        if (fdc1_w != 0.0 or fdc2_w != 0.0) and Pw is not None:
            M = M + (fdc1_w*conv(max_c1_unstr)
                     + fdc2_w*(Dlap @ conv(max_unstr))) @ Pw
        return M

    def compute_Omega(self, w_clm, T_e_parent, topo_clm, geoid_clm, q_clm, g0, R, T_e_0,
                      lmax_calc, lmax_grid, omega_corr_phys=None, dc_pre=None):
        """
        Equation for tangential loading potential Omega, following the definition
        as given in Broquet & Andrews-Hanna (2022), which is derived from Banerdt
        (1986). 

        In this M6, this equation has been rewritten into w-terms in order
        to maintain a 2Nx2N block matrix system, neglecting effects of crustal 
        thickness variations dc and mantle density variations dm. The solution for 
        Omega itself can therefore be obtained using the result for w_lm.
        """

        R_e = R - self.T_e_0/2
        RTcR = (R-self.T_c)/R
        g_m = g0 * (1.0 + (RTcR**3 - 1.0) * self.rho_c / self.rhobar) / RTcR**2    
        # DC FILTER: the three 1/RTcR**(l+2) weightings below are eq-(1)
        # eliminations of dc (terms 2, 6 and 7, the dc-elimination
        # artefacts), so each picks up x F1.
        DCfilter_mohoD = self.DCfilters(lmax_calc)[0]

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
        TeTc_grid = T_e_parent_grid_copyOmega - self.T_c
        # If a value is below 0, set to 0 to apply the 'max' call
        TeTc_grid_data = np.array(TeTc_grid.data)
        TeTc_grid_data[TeTc_grid_data < 0.0] = 0  
        TeTc_grid = pysh.SHGrid.from_array(TeTc_grid_data).data    

        # gravity at the elastic base (depth Te) for the mantle column term
        RTeR_grid = (R - T_e_parent_grid_copyOmega) / R
        # Create a dynamic rho grid based on the local thickness threshold
        rho_grid = np.where(T_e_parent_grid_copyOmega <= self.T_c, self.rho_c, self.rho_m)

        # Calculate the final gTe_grid using the dynamic rho_grid
        gTe_grid = g0 * (1.0 + (RTeR_grid**3 - 1.0) * rho_grid / self.rhobar) / RTeR_grid**2        

        TeH_grid = T_e_parent_grid_copyOmega * topo_grid_copyOmega

        # FIX (operator ordering): apply the per-degree dc-elimination weights to
        # H and G FIRST, then multiply by the max(Te-Tc,0) grid -- this is the 
        # ordering consistent with the per-degree elimination and with 
        # Omega_eq1_RHS in the solver (weight-then-multiply).
        # Previously the weights were applied to the coefficients of the PRODUCT
        # (TeTc*H), which differs for laterally varying Te.
        Hp_coeffs = pysh.SHGrid.from_array(topo_grid_copyOmega).expand()
        Hp_coeffs = self.truncate(Hp_coeffs, lmax=lmax_calc)
        for l in range(Hp_coeffs.coeffs.shape[1]):
            Hp_coeffs.coeffs[:, l, :] *= DCfilter_mohoD[l]/RTcR**(l+2)   # DC FILTER
        Hp_grid = Hp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
        TeTcHp_grid = TeTc_grid * Hp_grid

        Gp_coeffs = pysh.SHGrid.from_array(geoid_grid_copyOmega).expand()
        Gp_coeffs = self.truncate(Gp_coeffs, lmax=lmax_calc)
        for l in range(Gp_coeffs.coeffs.shape[1]):
            Gp_coeffs.coeffs[:, l, :] *= (DCfilter_mohoD[l] * self.rhobar*(2*l+1)
                                          / (3 * RTcR**(l+2)))     # DC FILTER
        Gp_grid = Gp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
        TeTcGp_grid = TeTc_grid * Gp_grid


        Tcind_grid_o = np.where(T_e_parent_grid_copyOmega > self.T_c, self.T_c, 0.0)   # Te<Tc FIX

        # Compute Re*Omega_lm as the term Omega_lm (required in conversion between
        # Banerdt and Beuthe's formulations).  w-coefficient corrected to match the
        # solve: surface -> drhol*g0 (vanishes for rho_l=rho_c), mantle -> gTe.
        term_1 = self.nu/(1-self.nu)*self.rho_l*g0*TeH_grid
        term_2 = + self.nu/(1-self.nu)*g_m*self.rho_l * TeTcHp_grid
        term_3 = -self.drhol*g0*self.nu/(1-self.nu)*T_e_parent_grid_copyOmega *w_grid_copyOmega

        term_4 = self.rho_c*g_m*Tcind_grid_o *w_grid_copyOmega
        term_5 = self.rho_m*gTe_grid*TeTc_grid *w_grid_copyOmega  # gTe field instead of scalar

        # drhol EXTENSION (zero if rho_l == rho_c): residual w-piece of the
        # (dc-w) substitution, + v1v*g_m*drhol*max(Te-Tc,0)*P_l*w  with the
        # weight applied to w FIRST (weight-then-multiply, as in the solver).
        wp_coeffs = pysh.SHGrid.from_array(w_grid_copyOmega).expand()
        wp_coeffs = self.truncate(wp_coeffs, lmax=lmax_calc)
        for l in range(wp_coeffs.coeffs.shape[1]):
            wp_coeffs.coeffs[:, l, :] *= DCfilter_mohoD[l]/RTcR**(l+2)   # DC FILTER
        wp_grid = wp_coeffs.expand(lmax=lmax_grid, lmax_calc=lmax_calc).data
        term_6 = + self.nu/(1-self.nu)*g_m*self.drhol * TeTc_grid * wp_grid
        term_7 = - self.nu/(1-self.nu)*g_m * TeTcGp_grid

        # Set terms depending on which internal variation is solved for.
        # Terms 2, 6, 7 are artefacts of the dc-ELIMINATION: they carry the
        # phi^-(l+2)-weighted H' and G' that replace dc. With dc = 0 they do not
        # exist. The drho_lm branch has instead, straight from DSP eq (5):
        #   term_8: the -w half of  v1v*drho*g_m*max*(dc - w)/R  now STANDS ALONE
        #           (in the dc branch the +drho*w hidden in drho*dc cancels it,
        #           leaving only the small drhol piece that is term_6);
        #   term_9: + P_hat * drho_m, with drho_m from compute_drho(w, H, G). 
        #           No elimination is needed here because w is already known.
        if self.solve_for == 'dc_lm':
            term_8 = 0.0
            term_9 = 0.0
        elif self.solve_for == 'G_lm':
            # M6: dc = 0 AND drho = 0. term_8 (the standalone -w half) survives;
            # every elimination artefact and the drhom term do not.
            term_2 = 0.0
            term_6 = 0.0
            term_7 = 0.0
            # v1v*drho*g_m*max*(dc - w):  both halves now, since dc is known.
            if dc_pre is not None:
                _dcg = self.truncate(dc_pre, lmax_calc).expand(
                           lmax=lmax_grid).data
                term_8 = self.nu/(1-self.nu)*self.drho*g_m * TeTc_grid * (_dcg - w_grid_copyOmega)
            else:
                term_8 = - self.nu/(1-self.nu)*self.drho*g_m * TeTc_grid * w_grid_copyOmega
            term_9 = 0.0
        else:
            term_2 = 0.0          # dc-elimination artefacts: absent when dc = 0
            term_6 = 0.0
            term_7 = 0.0
            term_8 = - self.nu/(1-self.nu)*self.drho*g_m * TeTc_grid * w_grid_copyOmega
            _dl_o   = self.drho_layer(lmax_grid, R, g0, self.mass)
            TeMt_o  = T_e_parent_grid_copyOmega - self.Mt
            TeMt0_o = np.where(TeMt_o > 0.0, TeMt_o, 0.0)         # max(Te-Mt, 0)
            Phat_o  = (-0.5 * self.nu/(1-self.nu) * _dl_o['g_M'] * TeMt0_o
                       * np.minimum(TeMt_o, _dl_o['M']))          # min(M, Te-Mt)
            drho_m_grid = self.compute_drho(w_clm, topo_clm, geoid_clm, R, 
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
        Re = R - self.T_e_0/2
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
        Omega_clm = self.truncate(Omega_clm, lmax=lmax_calc)

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
        # construction. M6 never forms omega as an unknown -- this function
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
        # has NO degree-0 term. M6 builds omega as a GRID and expands it, so the
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

        Omega_clm.coeffs[:, 1, :2] = (self.E * self.T_e_0**3 / (2.0 * R_e**3)) * q_clm.coeffs[:, 1, :2]    

        return Omega_clm

    def compute_dc(self, w_clm, topo_clm, geoid_clm, R, lmax_calc, lmax_grid):
        """
        Compute the crustal root variations ('bottom loads') dc_lm using the 
        rewritten equation of Gc_lm with drho_lm=0. 
        """

        RTcR = (R-self.T_c)/R
        L_comp = min(w_clm.lmax, topo_clm.lmax, geoid_clm.lmax, lmax_calc)
        DCfilter_mohoD = self.DCfilters(L_comp)[0]   # DC FILTER, eq-(1) elimination

        # ------- PRECOMPUTE SH FIELDS -------
        topo_clm_copydc = topo_clm.copy()
        topo_clm_copydc.coeffs[0,0,0] = 0
        topo_clm_copydc = self.truncate(topo_clm_copydc, L_comp)

        geoid_clm_copydc = geoid_clm.copy()
        geoid_clm_copydc.coeffs[0,0,0] = 0
        geoid_clm_copydc = self.truncate(geoid_clm_copydc, L_comp)

        w_clm_copydc = w_clm.copy()
        w_clm_copydc.coeffs[0,0,0] = 0
        w_clm_copydc = self.truncate(w_clm_copydc, L_comp)

        # ------- ASSEMBLY -------
        for l in range(geoid_clm_copydc.coeffs.shape[1]):
           geoid_clm_copydc.coeffs[:, l, :] *= self.rhobar*(2*l+1)/3
        dc_clm = 1/self.drho * (self.rho_l*topo_clm_copydc + self.drhol*w_clm_copydc - geoid_clm_copydc)

        for l in range(dc_clm.coeffs.shape[1]):
           dc_clm.coeffs[:, l, :] *= (DCfilter_mohoD[l]
                                      / (RTcR**(l+2)))   # DC FILTER 
        dc_clm = dc_clm + w_clm_copydc

        return dc_clm

    def compute_drho(self, w_clm, topo_clm, geoid_clm, R, lmax_calc, lmax_grid):
        """
        Compute the mantle density variations ('bottom loads') drho_lm using the 
        rewritten equation of Gc_lm with dc_lm=0. 
        """

        L_comp = min(w_clm.lmax, topo_clm.lmax, geoid_clm.lmax, lmax_calc)

        # ------- PRECOMPUTE SH FIELDS -------
        topo_clm_copydrho = topo_clm.copy()
        topo_clm_copydrho.coeffs[0,0,0] = 0
        topo_clm_copydrho = self.truncate(topo_clm_copydrho, L_comp)

        geoid_clm_copydrho = geoid_clm.copy()
        geoid_clm_copydrho.coeffs[0,0,0] = 0
        geoid_clm_copydrho = self.truncate(geoid_clm_copydrho, L_comp)

        w_clm_copydrho = w_clm.copy()
        w_clm_copydrho.coeffs[0,0,0] = 0
        w_clm_copydrho = self.truncate(w_clm_copydrho, L_comp)


        # ------- drho_lm VARIABLES AND FIELDS ------- 
        _dl   = self.drho_layer(lmax_grid, R, self.g0, self.mass) 
        Cp    = _dl['Cp']
        B_1 = _dl['B_1']
        RTcR_l2 = np.array([((R-self.T_c)/R)**(l+2) for l in range(2*lmax_grid+1)])

        # ------- THE FIELDS FOR EACH TERM -------
        topo_term  = topo_clm_copydrho * (-self.rho_l)
        w_term_1   = w_clm_copydrho * (-self.drho)          # moho term, weight phi^(l+2)
        w_term_2   = w_clm_copydrho * (-self.drhol)         # load-density term, no weight
        geoid_term = geoid_clm_copydrho.copy()
        for l in range(topo_term.coeffs.shape[1]):
            topo_term.coeffs[:, l, :]  *= Cp[l] / B_1[l]
            w_term_1.coeffs[:, l, :]   *= Cp[l] * RTcR_l2[l] / B_1[l]
            w_term_2.coeffs[:, l, :]   *= Cp[l] / B_1[l]
            geoid_term.coeffs[:, l, :] *= 1/B_1[l]

        # ------- ASSEMBLY -------
        drho_clm = topo_term + w_term_1 + w_term_2 + geoid_term

        return drho_clm

    def compute_geoid(self, w_clm, topo_clm, R, lmax_calc, lmax_grid, g0, mass,
                      corrG=None, corrGc=None, dc_pre=None):
        """
        The M6 OUTPUT: DSP eqs (1) and (2) evaluated FORWARD with dc = drho = 0.

        Replaces compute_dc / compute_drho as the loop's diagnostic. Note there
        is no division anywhere -- that is the whole structural difference from
        the two inverse branches, and the reason no filter appears here.

        Degree-0 convention follows the rest of the file: G_lm[0,0,0] = R, because
        every Omega builder forms the undulation as `geoid_clm.expand().data - R`
        (handover Part 9 trap 3). Degree 1 is set to zero: COM gauge.

        Returns (G_clm, Gc_clm). Gc is not consumed by the solve -- it is
        eliminated analytically inside q -- but it is a genuine model output and
        the round-trip test wants it.
        """
        phi    = (R - self.T_c) / R
        g_m    = g0 * (1.0 + (phi**3 - 1.0) * self.rho_c / self.rhobar) / phi**2

        L = min(w_clm.lmax, topo_clm.lmax, lmax_calc)

        H = self.truncate(topo_clm, L).copy();  H.coeffs[0, 0, 0] = 0.0
        w = self.truncate(w_clm,    L).copy();  w.coeffs[0, 0, 0] = 0.0

        degs  = np.arange(L + 1, dtype=float)
        Cp    = 3.0 / (self.rhobar * (2.0*degs + 1.0))
        phil  = phi**degs
        phil2 = phi**(degs + 2.0)

        dc_c = (self.truncate(dc_pre, L).coeffs if dc_pre is not None
                else np.zeros_like(H.coeffs))

        G_c  = np.zeros_like(H.coeffs)
        Gc_c = np.zeros_like(H.coeffs)
        for l in range(L + 1):
            # eq (1): the moho term is drho*(w - dc), not drho*w
            G_c[:, l, :]  = Cp[l] * ( self.rho_l*H.coeffs[:, l, :]
                                      + self.drhol*w.coeffs[:, l, :]
                                      + self.drho *(w.coeffs[:, l, :]
                                               - dc_c[:, l, :])*phil2[l] )
            # eq (2): same substitution at the moho continuation
            Gc_c[:, l, :] = Cp[l] * (g0/g_m) * (
                              (self.rho_l*H.coeffs[:, l, :]
                               + self.drhol*w.coeffs[:, l, :]) * phil[l]
                              + self.drho*(w.coeffs[:, l, :] - dc_c[:, l, :])*phi )

        if corrG is not None:
            G_c  += corrG[:, :L+1, :L+1]
        if corrGc is not None:
            Gc_c += corrGc[:, :L+1, :L+1]

        # G_1 is NOT forced to zero: eq (1) returns G_1 = 0 by itself whenever the
        # root balances H_1 (every dc an inverse returns does). For any other dc
        # the non-zero G_1 is the model's real degree-1 mass imbalance; forcing
        # it to 0 hands an inverse a geoid no internal structure here produces.
        Gc_c[:, 1, :] = 0.0
        G_c[0, 0, 0]  = R              # file-wide shape convention
        Gc_c[0, 0, 0] = R - self.T_c

        return (pysh.SHCoeffs.from_array(G_c,  normalization='4pi'),
                pysh.SHCoeffs.from_array(Gc_c, normalization='4pi'))

    
    
    def O1(self, SH_function, lmax):
        """ Beuthe (2008)'s differential operator O_1 in 2D spherical geometry. """
        kw_exp_grad = {"extend": False, "lmax_calc": lmax, "lmax": self.grid_expansion_res, "grid": "DH2"}

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
        dphi_sh = dphi_grid.expand(lmax_calc=lmax)
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
    
    def O2(self, SH_function, lmax):
        """ Beuthe (2008)'s differential operator O_2 in 2D spherical geometry. """
        kw_exp_grad = {"extend": False, "lmax_calc": lmax, "lmax": self.grid_expansion_res, "grid": "DH2"}

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
        dphi_sh = dphi_grid.expand(lmax_calc=lmax)
        dphi2_grid = dphi_sh.gradient(**kw_exp_grad).phi
        dphi2_grid.data *= csc_theta_grid
    
        return dphi2_grid.data + cot_theta_grid * dtheta_grid.data + SH_function_grid.data
    
    def O3(self, SH_function, lmax):
        """ Beuthe (2008)'s differential operator O_3 in 2D spherical geometry. """
        kw_exp_grad = {"extend": False, "lmax_calc": lmax, "lmax": self.grid_expansion_res, "grid": "DH2"}
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
    
        dphi_sh = dphi_grid.expand(lmax_calc=lmax)
        dthetaphi_grid = dphi_sh.gradient(**kw_exp_grad).theta
    
        return (csc_theta_grid * dthetaphi_grid.data 
                - cot_theta_grid * csc_theta_grid * dphi_grid.data)
    
    
    
    def stress_fields(self, S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
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
        kw_exp_grad = {"extend": False, "lmax_calc": lmax, "lmax": self.grid_expansion_res, "grid": "DH2"}

        O1S = self.O1(S_sol, lmax); O2S = self.O2(S_sol, lmax); O3S = self.O3(S_sol, lmax)
        O1w = self.O1(w_sol, lmax); O2w = self.O2(w_sol, lmax); O3w = self.O3(w_sol, lmax)
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
        DpsiTeR = self.E/(1.0 - self.nu**2)
    
        sigma_tt = (eps_t + self.nu*eps_p + eps_f*(kappa_t + self.nu*kappa_p)) * DpsiTeR / 1e6
        sigma_pp = (eps_p + self.nu*eps_t + eps_f*(kappa_p + self.nu*kappa_t)) * DpsiTeR / 1e6
        sigma_tp = (omega_sh + eps_f*tau) * 0.5 * DpsiTeR * (1.0 - self.nu) / 1e6
    
        return (pysh.SHGrid.from_array(sigma_tt),
                pysh.SHGrid.from_array(sigma_pp),
                pysh.SHGrid.from_array(sigma_tp))
    
    
    def strain_fields(self, S_sol, w_sol, T_e_parent, lmax, R, T_e_0, depth=0.0):
        """
        Total strains in the DSP/Banerdt convention (membrane + top-fiber
        bending), matching DSP's tot_theta / tot_phi / tot_thetaphi:
            tot = eps + eps_f*kappa,  eps_f = (Te/2-depth)/(1+(Te/2-depth)/R)
        
        MISSING THE TOROIDAL DISPLACEMENT POTENTIAL T TERMS!
        """
        kw_exp_grad = {"extend": False, "lmax_calc": lmax, "lmax": self.grid_expansion_res, "grid": "DH2"}

        # Return diff operator applied S and w terms, in grid.data format
        O1S = self.O1(S_sol, lmax); O2S = self.O2(S_sol, lmax); O3S = self.O3(S_sol, lmax)
        O1w = self.O1(w_sol, lmax); O2w = self.O2(w_sol, lmax); O3w = self.O3(w_sol, lmax)
        
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
    
    
    def cons_disp_S(self, w_sol, F_sol, Omega_sol, T_e_parent, a_clm, R, T_e_0, lmax_calc, lmax_grid):
        """ 
        Beuthe (2008)'s consoidal/poloidal tangential displacement potential S_lm 
        (A_lm in DSP/Banerdt (1986)). Used in computations of strain.
        """
        kw_exp_S = {"lmax_calc": lmax_calc, "lmax": lmax_grid, "grid": "DH2"}
        
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
    
        Re = R - self.T_e_0/2
        xi = 12*Re**2/Te_grid.data**2
        eta = xi/(1+xi)
        
        lapl_S_grid = (Re*eta*a_grid.data*(1-self.nu)*(F_lap2_grid.data + 2*Omega_grid.data) 
                  + eta/xi * w_lap2_grid.data  
                  - 2*w_grid.data)
        
        lapl_S_lm = pysh.SHGrid.from_array(lapl_S_grid).expand()
         
        S_lm = lapl_S_lm.copy()
        for l in range(1, S_lm.coeffs.shape[1]):
            S_lm.coeffs[:, l, :] /= lap_by_degree[l]
        S_lm.coeffs[0, 0, 0] = 0.0  
        
        S_lm = self.truncate(S_lm, lmax=lmax_calc)
            
        return S_lm
    
    
    def Principal_strainstress_angle(self, s_theta, s_phi, s_theta_phi):
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


    def compute_stress(self):
        """ 
        Combine the directional and principal stress calculations into one 
        function, returning the principal stresses and angle.
        """
        # S first (needed by the DSP-convention stress_fields), then stresses
        S_lm = self.cons_disp_S(self.w_lm, self.F_lm, self.Omega_lm, self.Te_lm, self.a_lm, R=self.R, T_e_0=self.T_e_0, lmax_calc=self.lmax, lmax_grid=self.grid_expansion_res)
        S_lm.coeffs[0,0,0] = 0
        self.w_lm.coeffs[0,0,0] = 0
        
        sigma_tt, sigma_pp, sigma_tp = self.stress_fields(S_lm, self.w_lm, self.Te_lm, lmax=self.grid_expansion_res, R=self.R, T_e_0=self.T_e_0)

        (   min_stress,
            max_stress,
            sum_stress,
            principal_angle_stress,
        ) = self.Principal_strainstress_angle(-sigma_tt.data, -sigma_pp.data, -sigma_tp.data)  
        
        return min_stress, max_stress, sum_stress, principal_angle_stress



    def solve_beuthe(self, topo_clm, geoid_clm, T_e_parent, D_clm, a_clm, plan, lmax, R,
                     T_e_0, g0, mass,
                     D_eta_clm=None, a_eta_clm=None, eta_clm=None,
                     q_corr_phys=None, omega_corr_phys=None, cache=None, 
                     dc_pre=None):
        """
        Implements Beuthe's UNSIMPLIFIED variable-thickness equations (58)/(66):
          Delta'(eta*D Delta' w) - (1-nu)A(eta*D; w) + R^3 A(eta; F) = RHS1
          Delta'(eta*a Delta' F) - (1+nu)A(eta*a; F) - (1/R)A(eta; w) = RHS2
        """

        mode_map = self.make_mode_map(lmax)
        N        = len(mode_map)
        Re       = R - T_e_0/2
        # Kalousova scalers
        scaler_A = 1.0/(self.E*T_e_0**3)
        scaler_B = Re

        # ============== Assemble the LHS matrix once, then cache =================
        # The LHS depends only on Te, D, alpha, eta, the Gaunt plan and lmax. None of
        # these change between iterations, only the RHS does.
        _cached = cache is not None and 'lu' in cache
        if _cached:
            lu, piv = cache['lu'], cache['piv']
        else:
            Dlm = pysh.shio.SHCilmToVector(D_eta_clm.coeffs)
            alm = pysh.shio.SHCilmToVector(a_eta_clm.coeffs)

            # ---- SoA fill: chunked gather + reduceat, no per-term Python loop
            gidx    = build_gidx(plan)                  # int32, chunked
            starts  = plan['cell_start'][:-1]
            seg_len = np.diff(plan['cell_start'])
            l_of    = mode_degrees(lmax)

            # A and B cell sums
            cellA, cellB = cell_sums_AB(Dlm, alm, gidx, plan, l_of, starts, seg_len, self.nu)
            cellA *= scaler_A
            cellB *= scaler_B

            ci, cj = plan['cell_i'], plan['cell_j']

            # Calculate the Omega LHS terms for equation 1
            (Omega_LHS_1a_unstr, Omega_LHS_1b_unstr, Omega_LHS_1c_unstr,
             g2) = (self.Omega_eq1_LHS(T_e_parent, lmax=lmax, 
                                  R=R, T_e_0=T_e_0, Re=Re, g0=g0, mass=mass))

            # Sum the three fields first (they are length (lmax+1)**2 = 8281, not
            # n_terms), then ONE plain convolution. Built through build_conv_matrix
            # so it carries the degree-1 cells the plan omits (Gaunt_utils_v5,
            # degree1_supplement); every other cell is unchanged.
            A_tilde = build_conv_matrix(Omega_LHS_1a_unstr + Omega_LHS_1b_unstr
                                        + Omega_LHS_1c_unstr,
                                        gidx, plan['term_gaunt_bare'], starts,
                                        seg_len, ci, cj, N)

            # Calculate the Omega LHS terms for equation 2
            (Omega_LHS_2_Phata_unstr,
             Omega_LHS_2a_unstr, 
             Omega_LHS_2b_unstr, 
             Omega_LHS_2c_unstr,
             Omega_LHS_2d_dc_unstr, 
             maxa_raw_unstr) = (self.Omega_eq2_LHS(T_e_parent, a_eta_clm, lmax=lmax, 
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
            B = np.zeros((N, N))

            A[ci, cj] = cellA          
            A[cj, ci] = cellA
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
            fac_a = (Re/T_e_0)**3 / self.E
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
            DCfilter_mohoD = self.DCfilters(lmax)[0]
            Pw_diag = np.diag(np.array([g2['RTcR']**(-(l+2)) * DCfilter_mohoD[l]
                                        for l, _ in mode_map]))
            _use_dc = (self.solve_for == 'dc_lm')
            _fwd    = (self.solve_for == 'G_lm')     # M6: G is an OUTPUT

            # dc branch: the reinstated drhol dc coupling (zero iff rho_l==rho_c).
            # drho branch: dc = 0, so that pair does not exist -- instead the -w
            # half of omega's (dc-w) term STANDS ALONE (fdrho_w1/w2, field = max,
            # no operand weight -> Pw = I). Its factors are exactly the historical
            # commented-out dc1/dc2 values: correct here, wrong in the dc branch.
            A_tilde_group2 = self.build_A_tilde_group2(
                g2['Te_unstr'], g2['max_unstr'], g2['max_c1_unstr'],
                g2['f1d'], g2['f1e'], g2['f1f'],
                gidx, plan['term_gaunt_bare'], starts, seg_len, ci, cj, mode_map, N,
                fdc1_w=(g2['fdc1_w'] if _use_dc else g2['fdrho_w1']),
                fdc2_w=(g2['fdc2_w'] if _use_dc else g2['fdrho_w2']),
                Pw=(Pw_diag if _use_dc else np.eye(N)),
                Tcind_unstr=g2['Tcind_unstr'], gTemax_unstr=g2['gTemax_unstr'])


            # ---- eq-1 Omega LHS block ----------------------------------------
            # eta (Beuthe 2008 eq 58) is folded into the Omega fields themselves
            # (Omega_eq1_LHS): c1 halves carry eta*X, c2 halves Delta(eta*Te^2*X).
            # No conv(eta) wrap is applied any more -- eta*[Te^2 Delta X] and
            # Delta(eta Te^2 X) differ by the gradients of eta*Te^2 at variable Te.
            # eta uses Re from the MEAN Te, consistent with derive_D_a.
            A = A + A_tilde + A_tilde_group2


            # ---- q's w-coupling: LHS diagonal, branch-dependent ---------------
            # NOTE: Lam_q is NOT an Omega term -- it comes from q -- so it is added
            # as a plain diagonal and must NOT carry eta.
            # dc branch  : Lam_q = qH with rho_l -> drhol   (zero iff rho_l==rho_c)
            # drho branch: Lam_q_drho (built in Omega_eq1_LHS) -- nonzero even at
            #              rho_l == rho_c, since the g_m*drho term survives.
            # Both enter via  -Re^4*K1*q  =>  A_diag += Re^4*K1*Lam_q.
            # STEP 3: both Lam_q arrays are precomputed in Omega_eq1_LHS. solve_beuthe
            # only applies the branch-appropriate one to the diagonal.

            # solve_for G_lm: Lam_q_drho carries
            #     + (g_M*M - g0*drho*B_2) * Dw_arr
            # which is the delta-rho ELIMINATION term. With drho_lm identically zero that
            # term is not merely unnecessary, it is wrong: it encodes "whatever density
            # anomaly reproduces the observed geoid", and there is no observed geoid in
            # forward mode.

            Lam_q_arr = (g2['Lam_q_dc']  if _use_dc else
                         g2['Lam_q_fwd'] if _fwd    else
                         g2['Lam_q_drho'])        

            for idx, (l_m, _) in enumerate(mode_map):
                A[idx, idx] += Re**4 * scaler_A * Lam_q_arr[l_m]

            # ---- drho branch: omega's drhom term  P_hat * drhom / R -----------
            # conv(P_hat) @ diag(w_corr): P_hat is a genuine Te-dependent field ->
            # convolution; w_corr is a per-degree scalar -> diagonal on the OPERAND.
            if self.solve_for == 'drho_lm':
                Dw_diag = np.diag(np.array([g2['w_corr'][l] for l, _ in mode_map]))
                C_Phat  = build_conv_matrix(g2['Phat_unstr'], gidx,
                                            plan['term_gaunt_bare'],
                                            starts, seg_len, ci, cj, N)
                C_Phat_c2 = build_conv_matrix(g2['Phat_c2_unstr'], gidx,
                                              plan['term_gaunt_bare'],
                                              starts, seg_len, ci, cj, N)
                Lap_d   = np.diag(np.array([-l*(l+1) for l, _ in mode_map],
                                           dtype=np.float64))

                # eq-1 Omega LHS terms, eta folded into the fields:
                #   c1:  conv(eta*P_hat) @ Dw
                #   c2:  Delta @ conv(eta*Te^2*P_hat) @ Dw
                A = A + g2['fdrho_om1'] * (C_Phat @ Dw_diag)
                A = A + g2['fdrho_om2'] * (Lap_d @ C_Phat_c2 @ Dw_diag)


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
                # b_tilde = b_tilde + g2['fdc_2d'] * (Dlp_out @ C_maxa)   # Pw_diag removed

            # drho branch: omega's drhom w-coupling in eq 2:
            #   (1-nu)*scaler2 * Delta'_out @ conv(P_hat*alpha) @ diag(w_corr)
            # P_hat*alpha is a genuine field (convolution); w_corr is per-degree
            # (diagonal on the OPERAND). Non-symmetric by construction.
            if self.solve_for == 'drho_lm':
                Dw_diag_2 = np.diag(np.array([g2['w_corr'][l] for l, _ in mode_map]))
                C_Phata   = build_conv_matrix(Omega_LHS_2_Phata_unstr, gidx,
                                              plan['term_gaunt_bare'],
                                              starts, seg_len, ci, cj, N)
                b_tilde = b_tilde + np.diag(d_l2) @ C_Phata @ Dw_diag_2

            # ETA FIELD FIX : b_tilde is the assembled eq-2 Omega LHS operator (2a/2b/2c
            # plus the dc and drho couplings above) -> apply conv(eta) once.
            # eq-2 Omega LHS: eta already sits inside via the eta*alpha fields
            # (Omega_eq2_LHS is fed a_eta_clm), i.e. Delta'(eta*alpha*X*w).
            b = b + b_tilde

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
                cache['lu'], cache['piv'] = lu, piv
                # Keep the assembled operator for the coupling diagnostics. Not
                # used by the solve -- lu/piv already contain everything needed --
                # purely so the Te-coupling structure can be inspected afterwards.
                cache['M'], cache['N'], cache['mode_map'] = M, N, mode_map            

        # Build the RHS (loads), changes with every iteration due to correction 
        # terms and therefore not cached!
        if self.solve_for == 'G_lm':
            # Use the forward q function
            q_lm_unstr = self.q_lm_fwd(topo_clm,
                                  lmax=lmax, R=R, T_e_0=T_e_0, Re=Re,
                                  g0=g0, mass=mass, dc_pre=dc_pre)
        else:
            # Use the inversion q function
            q_lm_unstr = self.q_lm(topo_clm, geoid_clm,
                              lmax=lmax, R=R, T_e_0=T_e_0, Re=Re,
                              g0=g0, mass=mass)

        Omega_RHS1_unstr = self.Omega_eq1_RHS(topo_clm, geoid_clm, T_e_parent, 
                                         lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                         g0=g0, mass=mass, dc_pre=dc_pre)
        Omega_RHS2_unstr = self.Omega_eq2_RHS(topo_clm, geoid_clm, T_e_parent, a_eta_clm,
                                         lmax=lmax, R=R, T_e_0=T_e_0, Re=Re, 
                                         g0=g0, mass=mass, dc_pre=dc_pre)

        # =============== the omega correction hook ==================
        # drho_omega_corr is an additive term on DSP eq (5), and M6 keeps eq (5)
        # by substituting its content into BOTH Omega RHS builders -- so the
        # correction must enter BOTH, through the same operator the builders
        # apply to their own omega content (see omega_corr_to_rhs).
        #
        # eta is applied inside omega_corr_to_rhs exactly as the builders apply
        # it to their own content (eq 1 via _eq1_halves, eq 2 via eta*alpha).
        if omega_corr_phys is not None:
            _o1, _o2 = self.omega_corr_to_rhs(omega_corr_phys, T_e_parent, a_eta_clm,
                                         lmax, R, T_e_0, Re)
            Omega_RHS1_unstr = Omega_RHS1_unstr + _o1
            Omega_RHS2_unstr = Omega_RHS2_unstr + _o2

        def elem(l,m,v):
            off = 0 if m==0 else (m if m>0 else l+abs(m))
            return v[l*l+off]

        q = np.array([elem(l,m,q_lm_unstr) for l,m in mode_map])
        Omega_RHS1 = np.array([elem(l,m,Omega_RHS1_unstr) for l,m in mode_map])
        Omega_RHS2 = np.array([elem(l,m,Omega_RHS2_unstr) for l,m in mode_map])

        # eta is already inside the Omega RHS fields (Beuthe 2008 eqs 58/66):
        # no conv(eta) wrap. q is not an Omega term and never carried eta.

        # ============== the q correction hook =======================
        # q_lm() returns the SCALED vector q = -Re^4 * q_phys / (E*Te0^3), so a
        # physical correction in Pa carries the same constant. No degree factor.
        # This single hook is all the framework needs: every correction in the
        # thin-shell system is an additive term on the right-hand side.
        if q_corr_phys is not None:
            q_correction = pysh.shio.SHCilmToVector(-Re**4/(self.E*T_e_0**3) * q_corr_phys)
            q_correction = np.array([elem(l, m, q_correction) for l, m in mode_map])
            q = q + q_correction

        y1 = q + Omega_RHS1
        y2 = Omega_RHS2


        rhs = np.concatenate([y1, y2])
        for idx,(l,_) in enumerate(mode_map):
            if l in (0,1): rhs[idx] = 0.0; rhs[idx+N] = 0.0


        if cache is not None:
            # Diagnostic capture only -- nothing downstream reads these.
            cache['rhs'] = rhs.copy()
            cache['q_vec'] = q.copy()
            cache['O1'] = Omega_RHS1.copy()
            cache['O2'] = Omega_RHS2.copy()

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

    def solve(self, topo_clm, geoid_clm, Te_clm):
        """
        Run the forward model.

        Parameters
        ----------
        topo_clm  : SHCoeffs, shape in m (degree 0 = R).
        geoid_clm : SHCoeffs, observed geoid in m (degree 0 = R). Returned for
                    comparison; the forward solution does not depend on it.
        Te_clm    : SHCoeffs, elastic thickness map in m.

        Returns
        -------
        dict with 'w' (flexure), 'G' (geoid), 'Gc' (geoid at the shell base),
        'topo', 'geoid' and 'dc' (the root that was used, or None).
        """
        topo_p, geoid_p, T_e_parent = topo_clm, geoid_clm, Te_clm


        g0 = self.g0
        R = self.R
        T_c = self.T_c
        rho_m = self.rho_m
        rho_c = self.rho_c
        rho_l = self.rho_l
        lmax = self.lmax        
        grid_expansion_res = self.grid_expansion_res
        nu = self.nu
        mass = self.mass


        # Perform some selftests to see if the gaunt code runs correctly
        selftest_gaunt()
        selftest_term_weights()

        # Load in the observables and Te map
        self.T_e_0 = T_e_parent.coeffs[0,0,0]
        print(f'T_e_0 = {self.T_e_0/1e3:.2f} km')
        D_clm, a_clm, D_eta_clm, a_eta_clm, eta_clm = self.derive_D_a(T_e_parent, lmax)



        # Tc > min(Te) anywhere makes max(Te-Tc,0) and Tc*1[Te>Tc] non-band-limited,
        # so their expansion depends on the grid. Test on the grid the SOLVER uses
        # (grid_expansion_res), not the coarser default of .expand().
        _Te_solver = T_e_parent.expand(lmax=grid_expansion_res).data
        _frac = np.mean(_Te_solver < T_c)
        if _frac > 0:
            print(f"\n! WARNING: Te < T_c on {100*_frac:.2f}% of the solver grid "
                  f"(min(Te) = {_Te_solver.min()/1e3:.1f} km, T_c = {T_c/1e3:.1f} km). "
                  f"The Te = Tc switch is then discontinuous: the solution carries a "
                  f"grid-dependent error and is not exactly rotation-covariant.\n")


        solutions_w = {}
        solutions_G  = {}
        solutions_Gc = {}

        dc_pre = None

        topo_clm  = self.truncate(topo_p,  lmax)
        geoid_clm = self.truncate(geoid_p, lmax)


        T_e_use, D_use, a_use, topo_use, geoid_use = (
            T_e_parent, D_clm, a_clm, topo_clm, geoid_clm)
        D_eta_use, a_eta_use, eta_use = D_eta_clm, a_eta_clm, eta_clm


        if self.solve_for != 'G_lm':
            dc_pre = None                     # the inverse solves for it
        elif self.DC_FROM_FILE is not None:
            dc_pre, _G_obs_M5, _w_inv_M5 = self.load_dc_from_file(
                self.DC_FROM_FILE, lmax,
                expect=dict(lmax=lmax, nmax=self.nmax, T_c=T_c,
                            rho_l=rho_l, rho_c=rho_c, rho_m=rho_m,
                            Mt=self.Mt, Mb=self.Mb, grid_res=grid_expansion_res,
                            delta_max=self.delta_max))
        else:
            dc_pre = self.build_dc_prescribed(topo_use, lmax)

        # ====================== The iterating loop =======================
        ### INITIALIZATION
        print('Start solving of system')
        # The constant-grid cache holds topo/geoid/Te grids for one input
        # set. A rotated pass feeds different fields, so it must start
        # clean: cached grids need to be cleared
        self._grid_cache.clear()
        t = time.perf_counter()
        cache = {}                      # matrix assembled once, reused

        ### INITIALIZING CORRECTION TERMS
        # The omega correction starts at zero, so iteration 1 is
        # identical to M6 whatever else is switched on.
        omega_corr_phys = None

        # In the pure finite-amplitude case H_corr depends only on H, a known
        # input, so it has no feedback and no Picard gain. Once the surface 
        # density gate is live H_corr depends on drho, which depends on w, 
        # so it is rebuilt inside the loop and acquires a gain of its own.
        # The value built here is the pure-FA seed used on iteration 1 and 
        # whenever not solving for drho_lm.
        H_corr = self.interface_geoid_correction('H', R, mass, lmax,
                                        grid_expansion_res, self.nmax,
                                        topo_clm=topo_use)

        # w_corr and wdc_corr are rebuilt each iteration from the solution; on the
        # first pass they are zero, so iteration 1 is the plain M6 solve.
        w_corr   = np.zeros_like(H_corr)
        wdc_corr = np.zeros_like(H_corr)

        # Geoid corrections are built from the three density interface
        # corrections:
        # corrG_1 live in forward mode too (G_1 comes out of eq (1))
        _geoid_correct_func = self.geoid_corrections
        corrG, corrGc = _geoid_correct_func(H_corr, w_corr, wdc_corr, R, lmax)
        geoid_MS      = self.geoid_mass_sheet(geoid_use, corrG, lmax)
        q_corr_phys   = self.q_correction_from_geoid(corrG, corrGc, R, g0, mass)


        ### Some initializations for the iterations (comparisons, delta's)
        w, F, q    = None, None, None 
        track_prev = None

        # init the comparison increment array and scalar delta
        increment_prev   = None
        delta_last = None

        # Store the previous iter arrays, for the damping
        hist_d = {1: None, 2: None} 

        for it in range(1, (self.iter_max if self.iterate else 1) + 1):
            w, F, q = self.solve_beuthe(topo_use, geoid_MS, T_e_use, D_use, a_use,
                                   self.plan, lmax=lmax, R=R, T_e_0=self.T_e_0,
                                   g0=g0, mass=mass, D_eta_clm=D_eta_use,
                                   a_eta_clm=a_eta_use, eta_clm=eta_use,
                                   q_corr_phys=q_corr_phys,
                                   omega_corr_phys=omega_corr_phys,
                                   cache=cache, dc_pre=dc_pre)
            if not self.iterate:
                break

            # Compute the recovered interface field dc_lm or drho_lm.
            # Needed by the delta-tracker and by the corrections. 
            # It uses the geoid_MS that this iteration's w was
            # actually solved with, because both compute_dc and
            # compute_drho invert eq (1), and eq (1) is the equation
            # geoid_MS encodes the correction into.

            # forward: the moho relief for the wdc finite-amplitude correction
            # is w - dc_pre; with no prescribed root it is just w
            # The interface reliefs the finite-amplitude corrections need.
            # Both compute_dc and compute_drho invert eq (1), so they are fed
            # geoid_MS -- the geoid this w was actually solved with -- not the
            # raw observation.
            if self.solve_for == 'dc_lm':
                dc_it = self.compute_dc(w, topo_use, geoid_MS, R, lmax,
                                        grid_expansion_res)
            elif self.solve_for == 'G_lm' and dc_pre is not None:
                dc_it = self.truncate(dc_pre, lmax).copy()   # relief = w - dc_pre
            else:
                dc_it = pysh.SHCoeffs.from_zeros(lmax)
            
            if self.solve_for == 'drho_lm':
                drho_it = self.compute_drho(w, topo_use, geoid_MS, R, lmax,
                                         grid_expansion_res)
            else:
                drho_it = pysh.SHCoeffs.from_zeros(lmax)


            # Convergence check. Tracks dc-w; the crust-mantle relief.
            # As of now always, when solving for drho_lm, so dc_lm=0, the 
            # delta is pure w. Maybe change to the delta in drho_lm.
            # Breaking here leaves geoid_MS and q_corr_phys as the ones this
            # w was solved with, so the stored pair is consistent with iter.
            # Change for the forward G_lm case?
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
                if delta < self.delta_max:
                    print(f'  converged in {it} iterations')
                    break
                if delta > self.delta_out:
                    print(f'  ! DIVERGING at iteration {it} - stopping')
                    break
            track_prev = track

            gate_H, gate_dc, gate_w = self.density_gates()
            # damping: every third iteration replaces every correction with the
            # average of the previous two (DSP / Wieczorek et al. 2013)
            _damp_now = (self.damp and it % 3 == 0
                         and hist_d[1] is not None and hist_d[2] is not None)

            drho_q  = np.zeros_like(q.coeffs)
            drho_om = np.zeros_like(q.coeffs)

            if _damp_now:
                drho_q, drho_om, H_corr, w_corr, wdc_corr = (
                    0.5*(hist_d[1][k] + hist_d[2][k]) for k in range(5))
                print('      (DAMPING THIS ITER: every correction is the '
                      'average of the last two iterations)')
            else:
                w_corr   = self.interface_geoid_correction('w', R, mass, lmax,
                                                      grid_expansion_res, self.nmax,
                                                      w_clm=w)
                wdc_corr = self.interface_geoid_correction('wdc', R, mass, lmax,
                                                      grid_expansion_res, self.nmax,
                                                      w_clm=w, dc_clm=dc_it)

                if self.solve_for == 'drho_lm':
                    # q and omega corrections, summed over the live interfaces.
                    # Note the OBSERVED geoid, not geoid_MS: DSP forms (H - G)
                    # from the input constraint directly, not via eq (1).
                    for _interface, _live in (('H', gate_H), ('dc', gate_dc),
                                              ('w', gate_w)):
                        if not _live:
                            continue
                        _q_part, _om_part = self.drho_q_omega_correction(
                            _interface, drho_it, topo_use, geoid_use, w,
                            T_e_use, R, lmax, g0)
                        drho_q  += _q_part
                        drho_om += _om_part

                    # Each density-aware geoid correction REPLACES its pure-FA
                    # counterpart where its gate is live: corr_nmax_drho returns
                    # finite amplitude AND lateral density from one call, so
                    # summing them would subtract the mass sheet twice.
                    _geo = lambda face: self.interface_geoid_correction(
                            face, R, mass, lmax, grid_expansion_res,
                            self.nmax, topo_clm=topo_use, w_clm=w, dc_clm=dc_it,
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

            corrG, corrGc = _geoid_correct_func(H_corr, w_corr, wdc_corr, R, lmax)
            geoid_MS        = self.geoid_mass_sheet(geoid_use, corrG, lmax)
            q_corr_phys     = self.q_correction_from_geoid(corrG, corrGc,
                                                      R, g0, mass) + drho_q
            # None rather than a zero array: solve_beuthe skips the omega
            # hook entirely when it is None, saving four full-resolution
            # transforms per iteration in the dc_lm branch.
            omega_corr_phys = drho_om if self.solve_for == 'drho_lm' else None

            # print(f'      RMS(corrG)/RMS(G_obs) = {RMS(corrG)/RMS(G_obs):9.6f}'
            #       '       RMS H_corr/w_corr/wdc_corr = '
            #       f'{RMS(H_corr):.4f} / {RMS(w_corr):.4f} / {RMS(wdc_corr):.4f} m/(kg/m^3)\n')

        print(f'Finished solving in {(time.perf_counter()-t):.1f}s ({it} iters)\n')

        # ---- the recovered internal structure ---------------------------
        # geoid_MS is the geoid this w was solved with, so eq (1) inverts
        # consistently; degree 0 is dropped (dc varies about T_c).
        if self.solve_for == 'dc_lm':
            dc_out = self.compute_dc(w, topo_use, geoid_MS, R, lmax,
                                     grid_expansion_res)
            dc_out.coeffs[0, 0, 0] = 0.0
            drho_out = None
        elif self.solve_for == 'drho_lm':
            drho_out = self.compute_drho(w, topo_use, geoid_MS, R, lmax,
                                         grid_expansion_res)
            drho_out.coeffs[0, 0, 0] = 0.0
            dc_out = None
        else:
            dc_out, drho_out = dc_pre, None

        # The geoid this solution produces. In the inverse modes the root is
        # the one just recovered, so eq (1) must be evaluated with it -- with
        # dc = 0 it would return the geoid of a root-free crust, which is not
        # the solution. Fed its own root, the dc_lm branch reproduces the
        # observed geoid to rounding, which is a free consistency check.
        # The drho_lm branch has no density term in eq (1) as written here, so
        # its geoid is reported as None rather than as something misleading.
        solutions_w    = w
        if self.solve_for == 'drho_lm':
            solutions_G = solutions_Gc = None
        else:
            _dc_for_G = dc_out if self.solve_for == 'dc_lm' else dc_pre
            if self.solve_for == 'dc_lm':
                # rebuild the FA corrections around the recovered root
                _Hc  = self.interface_geoid_correction('H', R, mass, lmax,
                            grid_expansion_res, self.nmax, topo_clm=topo_use)
                _wc  = self.interface_geoid_correction('w', R, mass, lmax,
                            grid_expansion_res, self.nmax, w_clm=w)
                _wdc = self.interface_geoid_correction('wdc', R, mass, lmax,
                            grid_expansion_res, self.nmax, w_clm=w,
                            dc_clm=_dc_for_G)
                corrG, corrGc = _geoid_correct_func(_Hc, _wc, _wdc, R, lmax)
            
            solutions_G, solutions_Gc = self.compute_geoid(
                w, topo_use, R, lmax_calc=lmax,
                lmax_grid=grid_expansion_res, g0=g0, mass=mass,
                corrG=corrG, corrGc=corrGc, dc_pre=_dc_for_G)


        self.Te_lm = T_e_parent
        self.w_lm  = solutions_w
        self.F_lm  = F
        self.G_lm  = (solutions_G if self.solve_for=='G_lm' else geoid_use)
        self.Gc_lm = solutions_Gc
        self.q_lm = q
        self.dc_lm = dc_out
        self.drho_lm = drho_out
        self.H_lm = topo_use
        self.Omega_lm = self.compute_Omega(self.w_lm, self.Te_lm, self.H_lm, 
                                           self.G_lm, self.q_lm, self.g0, self.R, self.T_e_0, 
                                           lmax_calc=self.lmax, lmax_grid=self.grid_expansion_res)
        self.a_lm = a_clm
        self.D_lm = D_clm
        self.geoid = geoid_use
        self.geoid_ms = geoid_MS
        self.iters = it
        

        return dict(w=solutions_w, G=solutions_G, Gc=solutions_Gc, F=F, q=q,
                    topo=topo_use, geoid=geoid_use, geoid_MS=geoid_MS,
                    dc=dc_out, drho=drho_out, iters=it)


    def export_dc(self, out, Te_clm, path):
        """
        Save a dc_lm solution in the format load_dc_from_file reads, so an
        inverse run can be fed straight back through the forward model.

        The stored setup is checked on load, and a filtered solution is
        refused there: a filtered root no longer satisfies eq (1), so its
        forward geoid misses the observation by exactly what the filter took
        out. Use filter_type=None for any root you intend to feed forward.
        """
        if self.solve_for != 'dc_lm' or out.get('dc') is None:
            raise ValueError('export_dc needs the result of a dc_lm run.')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path,
                 dc=out['dc'].coeffs, w=out['w'].coeffs,
                 geoid_obs=out['geoid'].coeffs, topo=out['topo'].coeffs,
                 Te=Te_clm.coeffs,
                 lmax=self.lmax, nmax=self.nmax, T_c=self.T_c,
                 rho_l=self.rho_l, rho_c=self.rho_c, rho_m=self.rho_m,
                 Mt=self.Mt, Mb=self.Mb, grid_res=self.grid_expansion_res,
                 delta_max=self.delta_max,
                 filt=(-1 if self.filter_type is None else self.filter_half))
        print(f'  saved {path}   RMS(dc) = '
              f'{np.sqrt(np.mean(out["dc"].coeffs[:, 2:, :]**2)):.4f} m')

    # ==================================================================
    #  Plotting. Each takes the dict returned by solve(); nothing here
    #  feeds back into the model.
    # ==================================================================

    def report_geoid(self, out):
        """Print the modelled-vs-observed geoid misfit, globally and per degree."""
        if out['G'] is None:
            print('\n  (no modelled geoid: the drho_lm branch fits the observed '
                  'geoid by construction and eq (1) here carries no density term)')
            return None, None
        lmax = self.lmax
        G_pred = out['G'].coeffs.copy()
        G_obs = out['geoid'].coeffs.copy()
        # degree 0 is the shape convention (both hold R); strip it from both or
        # the "residual" is the planetary radius
        G_pred[0, 0, 0] = 0.0
        G_obs[0, 0, 0] = 0.0
        rms = lambda a: np.sqrt(np.mean(a[:, 2:lmax+1, :]**2))
        print(f"\n{'='*60}\nMODELLED vs OBSERVED GEOID\n{'='*60}")
        print(f"  RMS(G_obs)  = {rms(G_obs):.4f} m")
        print(f"  RMS(G_pred) = {rms(G_pred):.4f} m")
        print(f"  RMS(diff)   = {rms(G_pred - G_obs):.4f} m"
              f"   ({rms(G_pred - G_obs)/rms(G_obs):.4e} relative)")
        print(f"  degree 1    : max|G_pred - G_obs| = "
              f"{np.abs(G_pred[:, 1, :] - G_obs[:, 1, :]).max():.3e} m")
        print('\n  per-degree geoid residual:')
        for l in [2, 3, 4, 5, 8, 10, 15, 20, 30, lmax]:
            if l > lmax:
                continue
            n = np.sqrt(np.mean((G_pred - G_obs)[:, l, :l+1]**2))
            d = np.sqrt(np.mean(G_obs[:, l, :l+1]**2))
            print(f'    l = {l:2d} : {n:.4e} m   ({n/max(d, 1e-30):.3e} rel)')
        return G_pred, G_obs

    def plot_geoid(self, out, cmap=scm.diverging.Vik_20.mpl_colormap, cmap_diff=cmc.broc):
        """Modelled geoid, observed geoid, and their difference."""
        if out['G'] is None:
            print('  (plot_geoid skipped: no modelled geoid in the drho_lm branch)')
            return
        G_pred, G_obs = out['G'].coeffs.copy(), out['geoid'].coeffs.copy()
        G_pred[0, 0, 0] = 0.0
        G_pred[:, 1, :] = 0.0
        G_pred[0, 2, 0] = 0.0
        G_obs[0, 0, 0] = 0.0
        res = self.grid_expansion_res
        G_diff = pysh.SHCoeffs.from_array(G_pred - G_obs).expand(lmax=res)
        lim_d = np.abs(G_diff.data).max()
        lim_a = np.abs(pysh.SHCoeffs.from_array(G_obs).expand(lmax=res).data).max()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 13))
        for ax, arr, lab, ttl, cm_, lim in (
                (ax1, G_pred, '$G_{pred}$ [m]', 'Forward modelled geoid', cmap, lim_a),
                (ax2, G_obs,  '$G_{obs}$ [m]',  'Observed geoid',         cmap, lim_a),
                (ax3, None,   '$G_{pred} - G_{obs}$ [m]',
                 'Modelled - observed geoid residual', cmap_diff, lim_d)):
            grid = G_diff if arr is None else pysh.SHCoeffs.from_array(arr).expand(lmax=res)
            grid.plot(ax=ax, cmap=cm_, cmap_limits=[-lim, lim], colorbar='bottom',
                      cb_label=lab, show=False)
            ax.set_title(ttl, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'geoid_residual')
        plt.show()

    def plot_fields(self, out, Te_clm, cmap1=cmc.navia, cmap2=cmc.lajolla, 
                    cmap3=cmc.roma_r):
        """Input topography and Te, the prescribed root, and the flexure."""
        res, R = self.grid_expansion_res, self.R
        fig, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(13, 8))
        kw = dict(tick_interval=[90, 45], grid=True, colorbar='right')
        topo_km = out['topo'].expand(lmax=res)
        cmap_limits_topo =[-8, 10]
        topo_km.data = (topo_km.data - R)/1e3
        topo_km.plot(ax=a0, cmap=cmap1, cmap_limits=cmap_limits_topo, 
                     cb_label='Topography [km]', **kw)
        a0.set_title(f'Topography, lmax = {self.lmax}')
        Te_km = Te_clm.expand(lmax=res)
        Te_km.data = np.round(Te_km.data/1e3, 2)
        Te_km.plot(ax=a1, cmap=cmap2, cb_label=r'$T_e$ [km]', **kw)
        a1.set_title('Elastic thickness input map')
        if out['dc'] is not None:
            pysh.SHGrid.from_array(out['dc'].expand(lmax=res).data/1e3).plot(
                ax=a2, cmap=cmap3, cb_label=r'$\delta c$ [km]', **kw)
            a2.set_title('Prescribed crustal root')
        else:
            a2.set_visible(False)
        w_grid = out['w'].expand(lmax=res).data/1e3
        pysh.SHGrid.from_array(w_grid).plot(ax=a3, cmap=cmap3,
                                            cb_label=r'$w$ [km]', **kw)
        a3.set_title('Flexure (output)')
        a3.contour(w_grid > 0, levels=[0.99], extent=(0, 360, -90, 90),
                   colors='k', origin='upper')
        plt.tight_layout()
        self._save(fig, 'input_output_fields')
        plt.suptitle('Input topography, Te, and dc_lm + output flexure w_lm')
        plt.tight_layout()
        plt.show()

    def plot_admittance(self, out):
        """Global admittance and correlation, modelled against observed."""
        if out['G'] is None:
            print('  (plot_admittance skipped: no modelled geoid in the '
                  'drho_lm branch)')
            return
        lmax, R, g0 = self.lmax, self.R, self.g0
        topo = out['topo'].coeffs
        Z_mod, _, g_mod = pysh.spectralanalysis.SHAdmitCorr(out['G'].coeffs, topo, lmax)
        Z_obs, _, g_obs = pysh.spectralanalysis.SHAdmitCorr(out['geoid'].coeffs, topo, lmax)
        l = np.arange(lmax + 1)
        to_mgal_km = (g0 / R) * (l + 1) * 1e8       # geoid m/m -> mGal/km
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 10))
        a1.plot(l[2:], (Z_mod*to_mgal_km)[2:], label='Modelled')
        a1.plot(l[2:], (Z_obs*to_mgal_km)[2:], '--', label='Observed')
        a1.set(xlabel='Spherical harmonic degree', ylabel='Z [mGal km$^{-1}$]',
               title='Admittance')
        a2=a1.twinx()
        a2.plot(l[2:], g_mod[2:], label='Modelled')
        a2.plot(l[2:], g_obs[2:], '--', label='Observed')
        a2.set(xlabel='Spherical harmonic degree', ylabel='Correlation',
               title='Correlation')
        # The modelled geoid is built entirely from the topography, so its
        # correlation is 1 by construction (variable Te and the finite-amplitude
        # terms move it only slightly). A lower observed correlation means
        # gravity that no topography-driven model can produce.
        for ax in (a1, a2):
            ax.grid(True)
            ax.legend(loc='lower right')
        plt.tight_layout()
        self._save(fig, 'admittance_correlation')
        plt.show()

    def _save(self, fig, name):
        if self.savefigs:
            fig.savefig(os.path.join(self.savepath, f'{name}_lmax{self.lmax}.png'),
                        dpi=200)