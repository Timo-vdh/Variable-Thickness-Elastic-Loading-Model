# -*- coding: utf-8 -*-
"""
Spectral variable-thickness thin-shell deflection model.
Implements Beuthe (2008) / Kalousova et al. (2012) for Mars.

BUGS FIXED vs. the original code
─────────────────────────────────
BUG 1 – Missing sqrt(4π) in the Gaunt coupling sums  [AXISYMMETRIC + 2-D]
  The Gaunt function get_numeric_gaunt() returns
      G = sqrt[(2l+1)(2l₁+1)(2l₂+1)/(4π)] · W3j(0,0,0) · W3j(m,m₁,m₂)
  which is the integral of three ORTHO-normalised harmonics.
  All SH coefficient arrays (D_lm, α_lm, y_lm) live in pyshtools' 4π
  normalisation.  The correct spectral-product formula for 4π-normalised
  real SH picks up an extra factor of sqrt(4π):
      coupling = sqrt(4π) · Σ G(l_out, L, l'; …) · D_{LM} · w_{l'm'}
  This factor was missing from both cell_sum_A and cell_sum_B.

BUG 2 – Wrong sign on the RHS loading vector  [AXISYMMETRIC + 2-D]
  With Bug 1 fixed the Beuthe and Turcotte magnitudes agree perfectly but
  every value has the opposite sign (≈ −200 % error).  Positive topography
  must produce a downward (negative) deflection, so the load must be negated:
      y_lm = −factors_y_lm · (topo − geoid)

BUG 3 – 2-D Gaunt coupling uses wrong formula for m ≠ 0  [2-D only]
  The original code applies the complex-SH selection rule m+M+m'=0 directly
  to the real (cos/sin) signed-m indices.  For m=0 the two representations
  coincide (which is why the axisymmetric tests reproduce Kalousova's figures
  correctly), but for m≠0 the coupling is wrong.  Example: the product
      Y_{2,+1}^real × Y_{2,+2}^real
  has non-zero output at cos(2,1), cos(4,1), cos(4,3); the complex Gaunt with
  signed-m real indices misses all three and instead predicts only sin(4,3).

  The fix replaces each single Gaunt call with a short sum over the complex-SH
  decomposition of each real SH:
      Y_{l,0}^real = Y_{l,0}^cmplx
      Y_{l,+m}^real = [Y_{l,m}^cmplx + (−1)^m Y_{l,−m}^cmplx] / √2   (cos)
      Y_{l,−m}^real = i[Y_{l,−m}^cmplx − (−1)^m Y_{l,+m}^cmplx] / √2  (sin)

  The resulting real Gaunt coefficient H is real-valued and was verified
  numerically against direct grid computation for a range of (l,m) triples.
  The coupling factor becomes:
      coupling_real(l_out,m_out; L,M; l',m')
          = sqrt(4π) · Σ_{cmplx_ijk} c_i · c_j · c_k
                        · G_code(l_out, L, l'; cm_i, cM_j, cm_k)

  This is a pure real computation; no complex SH conversion of D or α is
  needed, and the back-mapping of w_sol to SHCoeffs is straightforward.

NOTE: All other scalers (scaler_A, scaler_B, buoy, diag_a, diag_b,
factors_y_lm) are retained exactly as in the original code.  They are
self-consistent with the Kalousova (2012) dimensionless formulation and
require no changes.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from palettable import scientific as scm
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time

start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
#  INPUTS
# ─────────────────────────────────────────────────────────────────────────────
nu    = 0.25
E     = 100.0e9
rho_c = 2900.
rho_m = 3500.

lmax         = 30
AXISYMMETRIC = False   # True → 1-D (m = 0 only, much faster)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
pot_clm  = pysh.datasets.Mars.GMM3(lmax=lmax)
topo_clm = pysh.datasets.Mars.MOLA_shape(lmax=lmax)

R         = topo_clm.coeffs[0, 0, 0]
pot_clm   = pot_clm.change_ref(r0=R)
geoid_clm = pot_clm * R

G    = pysh.constants.G.value
gm   = pot_clm.gm
g0   = gm / R**2

percent_C20 = 0.0
topo_clm.coeffs[0, 2, 0]   *= percent_C20 / 100.
geoid_clm.coeffs[0, 2, 0]  *= percent_C20 / 100.

mycmap = scm.diverging.Vik_20.mpl_colormap

# ─────────────────────────────────────────────────────────────────────────────
#  ELASTIC THICKNESS MAP  (constant here)
# ─────────────────────────────────────────────────────────────────────────────
T_e_type  = 'Constant_TeMap'
T_e       = 150e3
T_e_array = T_e * np.ones([2*(lmax+1)+1, 4*(lmax+1)+1])
T_e_grid  = pysh.SHGrid.from_array(T_e_array)
T_e_clm   = T_e_grid.expand()


# # Initialize randomizer
# seed = 1
# l_corner = 10
# beta = 3.0
# power = np.zeros(lmax + 1)
# for li in range(2, lmax+1):
#     if li <= l_corner:
#         power[li] = 50.0
#     else:
#         power[li] = (l_corner / li) ** beta

# # Make a random coefficient map
# T_e_type = 'Random_TeMap'
# T_e_clm = pysh.SHCoeffs.from_random(power, lmax=lmax, seed=seed)
# T_e_array = T_e_clm.expand().to_array()*1e3 + 150e3
# T_e_grid = pysh.SHGrid.from_array(T_e_array)


    
# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def find_custom_element(l_param, m_param, xlm_unstr):
    """Extract element (l, m) from a SHCilmToVector flat array (real SH)."""
    block_start = l_param**2
    if m_param == 0:
        offset = 0
    elif m_param > 0:
        offset = m_param
    else:
        offset = l_param + abs(m_param)
    return xlm_unstr[block_start + offset]


def W_numeric_A(l_deg, l_prime, L, nu_val=0.25):
    """W-coefficient for Matrix A (Kalousova 2012 eq. A18, (1−ν) variant)."""
    d_l  = -l_deg   * (l_deg   + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L  = -L       * (L       + 1) + 2
    t1   = d_l * d_lp
    br   = (d_l**2 + d_lp**2 + d_L**2
            + 2*(d_l + d_lp + d_L)
            - 2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return t1 + 0.25*(1.0 - nu_val)*br


def W_numeric_B(l_deg, l_prime, L, nu_val=0.25):
    """W-coefficient for Matrix B (Kalousova 2012 eq. A18, (1+ν) variant)."""
    d_l  = -l_deg   * (l_deg   + 1) + 2
    d_lp = -l_prime * (l_prime + 1) + 2
    d_L  = -L       * (L       + 1) + 2
    t1   = d_l * d_lp
    br   = (d_l**2 + d_lp**2 + d_L**2
            + 2*(d_l + d_lp + d_L)
            - 2*(d_l*d_lp + d_l*d_L + d_lp*d_L) - 8)
    return t1 + 0.25*(1.0 + nu_val)*br


def get_numeric_gaunt(l1, l2, l3, m1, m2, m3):
    """
    Gaunt coefficient for three complex (or zonal real) SH.
        G = sqrt[(2l₁+1)(2l₂+1)(2l₃+1)/(4π)] · W3j(0,0,0) · W3j(m₁,m₂,m₃)
    Selection rules: parity (l₁+l₂+l₃ even), triangle, m₁+m₂+m₃=0.
    Returns a real scalar.
    """
    if (l1 + l2 + l3) % 2 != 0:
        return 0.0
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return 0.0
    if m1 + m2 + m3 != 0:
        return 0.0
    w3j_m_arr, jmin_m, jmax_m = pysh.utils.Wigner3j(l2, l3, m1, m2, m3)
    if not (jmin_m <= l1 <= jmax_m):
        return 0.0
    w3j_m = w3j_m_arr[l1 - jmin_m]
    w3j_0_arr, jmin_0, jmax_0 = pysh.utils.Wigner3j(l2, l3, 0, 0, 0)
    if not (jmin_0 <= l1 <= jmax_0):
        return 0.0
    w3j_0 = w3j_0_arr[l1 - jmin_0]
    return (np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1) / (4.0*np.pi))
            * w3j_m * w3j_0)


# ── BUG 3 FIX: real-SH Gaunt via complex decomposition ──────────────────────

def _decompose_real_sh(l, m):
    """
    Express Y_{l,m}^{4π real} as a linear combination of Y_{l,cm}^{4π complex}.
    Returns a list of (complex_m_index, coefficient) pairs.
    Convention (pyshtools):
      m = 0  →  Y_{l,0}^real = Y_{l,0}^cmplx
      m > 0  →  Y_{l,+m}^real = (Y_{l,m}^c + (−1)^m Y_{l,−m}^c) / √2  [cosine]
      m < 0  →  Y_{l,m}^real  = i(Y_{l,m}^c − (−1)^|m| Y_{l,−m}^c) / √2  [sine]
    """
    if m == 0:
        return [(0, 1.0+0j)]
    elif m > 0:
        return [(m,  1.0/np.sqrt(2)+0j),
                (-m, (-1)**m / np.sqrt(2)+0j)]
    else:
        absm = abs(m)
        return [(-absm,  1j / np.sqrt(2)),
                ( absm, -(-1)**absm * 1j / np.sqrt(2))]


_sqrt4pi = np.sqrt(4.0 * np.pi)   # BUG 1 + BUG 3 normalisation factor


def get_real_gaunt(l_out, m_out, L, M, l_prime, m_prime):
    """
    Coupling coefficient for REAL SH (signed-m) in the 4π-normalised product
    formula:
        (A·B)_{l_out,m_out} = Σ_{L,M,l',m'} A_{L,M} · B_{l',m'} · H

    H = sqrt(4π) · Σ_{complex_decompositions} c_out · c_L · c_p
                   · G_code(l_out, L, l'; cm_out, cM, cm_p)

    This is the general real-SH Gaunt coefficient, valid for ALL signed-m
    values.  For m=0 it reduces to sqrt(4π)·G_code (i.e. the Bug-1 fix for
    the axisymmetric case).  Verified numerically against direct grid
    integration for a broad set of (l,m) triples.

    Returns a real scalar (imaginary part cancels exactly for real SH).
    """
    terms_out   = _decompose_real_sh(l_out, m_out)
    terms_L     = _decompose_real_sh(L, M)
    terms_prime = _decompose_real_sh(l_prime, m_prime)

    total = 0.0 + 0j
    for cm_out, c_out in terms_out:
        for cM, cL in terms_L:
            for cm_p, c_p in terms_prime:
                g = get_numeric_gaunt(l_out, L, l_prime,
                                      cm_out, cM, cm_p)
                total += c_out * cL * c_p * g

    return total.real * _sqrt4pi  # imaginary part is exactly 0 for real SH


# ─────────────────────────────────────────────────────────────────────────────
#  TURCOTTE (TSA-T) REFERENCE FORMULA
# ─────────────────────────────────────────────────────────────────────────────

def C_l_functional(l_val, nu, E, T_e_local, Re, rho_m, rho_c, g0):
    tau   = E * T_e_local / (Re**2 * (rho_m - rho_c) * g0)
    sigma = tau / (12*(1 - nu**2)) * (T_e_local / Re)**2
    d_b1  = (l_val**3*(l_val+1)**3
             - 4*l_val**2*(l_val+1)**2
             + 4*l_val*(l_val+1))
    num   = l_val*(l_val+1) - (1 - nu)
    denom = sigma*d_b1 + tau*(l_val*(l_val+1) - 2) + l_val*(l_val+1) - (1 - nu)
    return num / denom


rho_term = -rho_c / (rho_m - rho_c)

# ─────────────────────────────────────────────────────────────────────────────
#  MODE MAP AND SYSTEM SETUP
# ─────────────────────────────────────────────────────────────────────────────
mode_map = []
if AXISYMMETRIC:
    for l in range(lmax+1):
        mode_map.append((l, 0))
else:
    for l in range(lmax+1):
        for m in range(-l, l+1):
            mode_map.append((l, m))
N_modes = len(mode_map)

print(f"\n--- Spectral Matrix Assembly (lmax={lmax}, AXISYMMETRIC={AXISYMMETRIC}) ---")

T_e_0 = np.mean(T_e_array)
Re    = R - T_e_0 / 2.0

# Kalousova dimensionless scalers (unchanged from original — all correct)
buoy     = (Re/T_e_0)**3 * (Re/E) * g0 * (rho_m - rho_c)
scaler_A = 1.0 / (E * T_e_0**3)
scaler_B = Re

D_array = E * T_e_array**3 / (12*(1 - nu**2))
a_array = 1.0 / (E * T_e_array)

D_grid = pysh.SHGrid.from_array(D_array)
a_grid = pysh.SHGrid.from_array(a_array)

D_clm_4pi = D_grid.expand(normalization='4pi')
a_clm_4pi = a_grid.expand(normalization='4pi')

Dlm_unstr = pysh.shio.SHCilmToVector(D_clm_4pi.coeffs)
alm_unstr = pysh.shio.SHCilmToVector(a_clm_4pi.coeffs)

# Pre-extract coefficient vectors aligned to mode_map
Dlm_str = np.array([find_custom_element(l, m, Dlm_unstr) for l, m in mode_map])
alm_str = np.array([find_custom_element(l, m, alm_unstr) for l, m in mode_map])

# ─────────────────────────────────────────────────────────────────────────────
#  DIAGONAL COUPLING BLOCKS  (unchanged scalers)
# ─────────────────────────────────────────────────────────────────────────────
print("Initialising sparse matrix buffers …")
matrix_A_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)
matrix_B_sparse = sparse.lil_matrix((N_modes, N_modes), dtype=np.float64)

diag_a = np.zeros(N_modes)
diag_b = np.zeros(N_modes)
for i, (l_val, m_val) in enumerate(mode_map):
    d_l       = -l_val*(l_val+1) + 2
    diag_a[i] = ((Re/T_e_0)**3 / E) * d_l
    diag_b[i] = -1.0 * d_l

matrix_a_l_sparse = sparse.diags(diag_a, format="lil")
matrix_b_l_sparse = sparse.diags(diag_b, format="lil")

# ─────────────────────────────────────────────────────────────────────────────
#  MATRIX ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
print("Assembling coupling combinations …")

if AXISYMMETRIC:
    # ── 1-D: m = 0 only ─────────────────────────────────────────────────────
    # For m=0 get_real_gaunt reduces to sqrt(4π)·G_code (Bug 1 fix only).
    for i, (l_val, _) in enumerate(mode_map):
        for j, (l_prime, _) in enumerate(mode_map):
            cell_sum_A = 0.0
            cell_sum_B = 0.0
            min_L = abs(l_val - l_prime)
            max_L = min(l_val + l_prime, lmax)
            for L in range(min_L, max_L + 1):
                if (l_val + l_prime + L) % 2 != 0:
                    continue
                w_A = W_numeric_A(l_val, l_prime, L, nu)
                w_B = W_numeric_B(l_val, l_prime, L, nu)
                if w_A == 0.0:
                    continue
                # BUG 1+3 FIX: use get_real_gaunt (= sqrt(4π)·G for m=0)
                q_val = get_real_gaunt(l_val, 0, L, 0, l_prime, 0)
                if q_val == 0.0:
                    continue
                D_val = float(find_custom_element(L, 0, Dlm_unstr))
                a_val = float(find_custom_element(L, 0, alm_unstr))
                cell_sum_A += w_A * D_val * q_val
                cell_sum_B += w_B * a_val * q_val

            val_A = cell_sum_A * scaler_A
            val_B = cell_sum_B * scaler_B
            if l_val == l_prime:
                val_A += buoy
            if val_A != 0.0:
                matrix_A_sparse[i, j] = val_A
            if val_B != 0.0:
                matrix_B_sparse[i, j] = val_B

else:
    # ── 2-D: all (l, m) ─────────────────────────────────────────────────────
    # BUG 3 FIX: use get_real_gaunt which correctly handles m ≠ 0 via the
    # complex decomposition sum.  The loop over M no longer uses the selection
    # rule M = m_out − m_prime; instead get_real_gaunt handles all sign
    # combinations internally via the _decompose_real_sh() expansion.
    for i, (l_val, m_val) in enumerate(mode_map):
        for j, (l_prime, m_prime) in enumerate(mode_map):
            cell_sum_A = 0.0
            cell_sum_B = 0.0

            min_L = abs(l_val - l_prime)
            max_L = min(l_val + l_prime, lmax)

            for L in range(min_L, max_L + 1):
                if (l_val + l_prime + L) % 2 != 0:
                    continue
                w_A = W_numeric_A(l_val, l_prime, L, nu)
                w_B = W_numeric_B(l_val, l_prime, L, nu)
                if w_A == 0.0:
                    continue

                # Sum over all M values at degree L
                for M in range(-L, L + 1):
                    # BUG 1+3 FIX: use get_real_gaunt (real SH, all m)
                    q_val = get_real_gaunt(l_val, m_val, L, M, l_prime, m_prime)
                    if q_val == 0.0:
                        continue
                    D_val = float(find_custom_element(L, M, Dlm_unstr))
                    a_val = float(find_custom_element(L, M, alm_unstr))
                    cell_sum_A += w_A * D_val * q_val
                    cell_sum_B += w_B * a_val * q_val

            val_A = cell_sum_A * scaler_A
            val_B = cell_sum_B * scaler_B
            if l_val == l_prime and m_val == m_prime:
                val_A += buoy
            if val_A != 0.0:
                matrix_A_sparse[i, j] = val_A
            if val_B != 0.0:
                matrix_B_sparse[i, j] = val_B

# ─────────────────────────────────────────────────────────────────────────────
#  FULL 2N × 2N SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
print("Combining sub-matrices into 2N × 2N architecture …")
M_system_sparse = sparse.bmat([
    [matrix_A_sparse,   matrix_a_l_sparse],
    [matrix_b_l_sparse, matrix_B_sparse]
], format="lil")

print("Zeroing degrees 0 and 1 …")
for idx, (l_val, _) in enumerate(mode_map):
    if l_val in (0, 1):
        M_system_sparse[idx, :]                       = 0.0
        M_system_sparse[idx, idx]                     = 1.0
        M_system_sparse[idx + N_modes, :]             = 0.0
        M_system_sparse[idx + N_modes, idx + N_modes] = 1.0

M_system_csr = M_system_sparse.tocsr()

# ─────────────────────────────────────────────────────────────────────────────
#  RHS LOADING VECTOR  (BUG 2 FIX: negative sign)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Solving for displacement (lmax={lmax}) …")

factors_y_lm = (Re/T_e_0)**3 * (rho_c * g0 * Re) / E

# BUG 2 FIX: negate the load so that positive topography → negative (downward) w.
y_lm_topo    = -factors_y_lm * (topo_clm.coeffs - geoid_clm.coeffs)
y_lm_unstr   = pysh.shio.SHCilmToVector(y_lm_topo)
y_lm_str     = np.array([find_custom_element(l, m, y_lm_unstr)
                         for l, m in mode_map])

rhs_dense = np.concatenate([y_lm_str, np.zeros(N_modes)])
for idx, (l_val, _) in enumerate(mode_map):
    if l_val in (0, 1):
        rhs_dense[idx]           = 0.0
        rhs_dense[idx + N_modes] = 0.0

sol_vector = spla.spsolve(M_system_csr, rhs_dense)
w_sol = sol_vector[:N_modes]

# ─────────────────────────────────────────────────────────────────────────────
#  MAP SOLUTION BACK TO SHCoeffs  (real SH, straightforward)
# ─────────────────────────────────────────────────────────────────────────────
# The matrix is entirely real and so is w_sol.  The mode_map uses signed-m
# real SH indices: m ≥ 0 → cosine (coeffs[0]), m < 0 → sine (coeffs[1]).
w_coeffs_np = np.zeros((2, lmax+1, lmax+1))
for idx, (l_val, m_val) in enumerate(mode_map):
    if m_val >= 0:
        w_coeffs_np[0, l_val,  m_val]  = float(w_sol[idx])
    else:
        w_coeffs_np[1, l_val, abs(m_val)] = float(w_sol[idx])

w_sol_clm_beuthe  = pysh.SHCoeffs.from_array(w_coeffs_np, normalization='4pi')
w_sol_grid_beuthe = w_sol_clm_beuthe.expand()

# ─────────────────────────────────────────────────────────────────────────────
#  TURCOTTE (TSA-T) SOLUTION
# ─────────────────────────────────────────────────────────────────────────────
print("Computing Turcotte reference …")
w_coeffs_turcotte = np.zeros((2, lmax+1, lmax+1))
T_e = np.mean(T_e_array)
for l_val in range(2, lmax+1):
    C_l = C_l_functional(l_val, nu, E, T_e, Re, rho_m, rho_c, g0)
    w_coeffs_turcotte[:, l_val, :l_val+1] = (
        rho_term * C_l
        * (topo_clm.coeffs[:, l_val, :l_val+1]
           - geoid_clm.coeffs[:, l_val, :l_val+1])
    )

w_sol_clm_turcotte  = pysh.SHCoeffs.from_array(w_coeffs_turcotte, normalization='4pi')
w_sol_grid_turcotte = w_sol_clm_turcotte.expand()


### Varying Te Turcotte solution (spatial)
topo_grid_norm = (topo_clm.coeffs - geoid_clm.coeffs).expand(normalization='4pi')

w_grid_spat = np.zeros(2*(lmax+1)+1, 4*(lmax+1)+1)
spatial_colatitudes = 90 - w_sol_grid_beuthe.lats()
spatial_longitudes = w_sol_grid_beuthe.lons()

for lat_idx in range(len(spatial_colatitudes)):
    for lon_idx in range(len(spatial_longitudes)):
        T_e_local1 = T_e_array[lat_idx, lon_idx]
        C_l_local1 = C_l_functional(l_val, nu, E, T_e_local1, Re, rho_m, rho_c, g0)

        w_grid_spat[lat_idx, lon_idx] = (
            rho_term * C_l_local1 * topo_grid_norm[lat_idx, lon_idx]
            )

w_sol_grid_turcotte_spat = pysh.SHGrid.from_array(w_grid_spat)


### Varying Te Turcotte solution (spectral)
w_coeffs_turcotte2 = np.zeros((2, lmax+1, lmax+1))
for l_val in range(2, lmax+1):
    C_l_local2 = 0
    for m_val in range(-l_val, l_val+1):
        T_e_local2 = T_e_clm[l_val, m_val]
        C_l_local2 += C_l_functional(l_val, nu, E, T_e_local2, Re, rho_m, rho_c, g0)
    
    w_coeffs_turcotte2[:, l_val, :l_val+1] = (
        rho_term * C_l_local2
        * (topo_clm.coeffs[:, l_val, :l_val+1]
           - geoid_clm.coeffs[:, l_val, :l_val+1])
    )
w_sol_clm_turcotte2  = pysh.SHCoeffs.from_array(w_coeffs_turcotte2, normalization='4pi')
w_sol_grid_turcotte2 = w_sol_clm_turcotte2.expand()

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

w_sol_grid_beuthe.plot(ax=ax1, cmap=mycmap, colorbar='right', cb_label='w [m]')
ax1.set_title('Beuthe matrix solution (TSA-B)')

w_sol_grid_turcotte.plot(ax=ax2, cmap=mycmap, colorbar='right', cb_label='w [m]')
ax2.set_title('Turcotte summation (TSA-T)')

diff_grid = w_sol_grid_beuthe - w_sol_grid_turcotte
diff_grid.plot(ax=ax3, cmap=mycmap, colorbar='right', cb_label='Misfit [m]')
ax3.set_title('Residual TSA-B − TSA-T')

plt.tight_layout()
# plt.savefig('TestDispMars_corrected.png', dpi=150)
plt.show()


# Comparisons of Turcotte outcomes using single thickness or spatially varying thickness
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

w_sol_grid_turcotte.plot(ax=ax1, cmap=mycmap, colorbar='right', cb_label='w [m]')
ax2.set_title('Turcotte single thickness')

w_sol_grid_turcotte_spat.plot(ax=ax2, cmap=mycmap, colorbar='right', cb_label='w [m]')
ax2.set_title('Turcotte varying thickness (spatial sol)')

w_sol_grid_turcotte2.plot(ax=ax3, cmap=mycmap, colorbar='right', cb_label='w [m]')
ax2.set_title('Turcotte varying thickness (spectral sol)')

plt.tight_layout()
# plt.savefig('TestDispMarsTurcotte_ConstantVsVarying.png', dpi=150)
plt.show()


end = time.time()
print(f"\n--- Done. Runtime: {round(end-start, 1)} s ---")

print("\nSanity check – displacement range:")
print(f"  Beuthe   min/max: {w_sol_grid_beuthe.data.min()/1e3:.2f} / "
      f"{w_sol_grid_beuthe.data.max()/1e3:.2f} km")
print(f"  Turcotte min/max: {w_sol_grid_turcotte.data.min()/1e3:.2f} / "
      f"{w_sol_grid_turcotte.data.max()/1e3:.2f} km")
print("  (Expected roughly −12.5 to +5 km for Mars at Te = 150 km)")