# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:22:58 2026

@author: Timov
"""

# -*- coding: utf-8 -*-
"""
M3 <-> DSP per-degree transfer diagnostic.

At CONSTANT Te the M3 system is block-diagonal per degree, so the deflection is
exactly a two-field per-degree response

        w_lm = P(l) * H_lm + Q(l) * G_lm           (H = topo-R,  G = geoid-R)

This module fits P(l), Q(l) per degree by least squares over m (no cross-power
ambiguity), reports the regression residual (a geoid-convention detector), and
- if a DSP w field is supplied - regresses DSP's w on the SAME observed (H,G)
and compares. It also returns the classical topography admittance
Z(l)=S_wH/S_HH for reference.

How to read the output
----------------------
  * M3 residual ~1e-12     -> M3 has cleanly reduced to the per-degree form
                              (validates the constant-Te decoupling). If it is
                              NOT tiny, the constant-Te run still has Gaunt
                              coupling / product aliasing -> investigate first.
  * DSP residual ~ M3's     -> DSP used (effectively) the observed geoid; the
                              gap is NOT the geoid -> look at P,Q ratios.
  * DSP residual >> M3's    -> DSP's effective load (its geoid) is not spanned
                              by the observed (H,G) -> geoid-convention is the gap.
  * P_M3/P_DSP flat != 1    -> a global scalar: reference-radius / scaler
                              convention (Re vs R, Kalousova scaler, ...).
  * P_M3/P_DSP ramps with l  -> a continuation-factor convention
                              (phi^(l+2), (2l+1), ...).
  * Q differs but P matches  -> geoid-coupling convention specifically.

Usage (paste this file next to your M3 script, then in __main__ after solving):

    from m3_admittance_diagnostic import admittance_diagnostic

    res = admittance_diagnostic(
        w_M3   = solutions_w[LMAX_REF, 0],   # SHCoeffs, metres
        topo   = topo_clm,                   # SHCoeffs (your solver input)
        geoid  = geoid_clm,                  # SHCoeffs (your solver input)
        R      = R,
        lmax   = LMAX_REF,
        w_DSP  = w_DSP,                      # SHGrid or SHCoeffs or None
        dsp_units_to_m = 1.0e3,              # DSP files look like km -> set 1e3
        lmin   = 2,
        make_plots = True,
    )

IMPORTANT: set dsp_units_to_m so that DSP w ends up in METRES (your M3 w is in
metres). Your DSP files are plotted with cb_label 'w [km]', so that is 1e3.
Run it once on the CONSTANT-Te M3 (strain=0) against the DSP file at the same
Te and Tc.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# ----------------------------------------------------------------------
#  adapters: accept SHCoeffs, SHGrid, or a raw (2,L+1,L+1) ndarray
# ----------------------------------------------------------------------
def _to_coeffs(obj, lmax):
    """Return a (2, lmax+1, lmax+1) coeffs array from an SHCoeffs, an SHGrid
    (it is expanded), or a raw ndarray. Truncates/zero-pads to lmax."""
    if obj is None:
        return None
    if hasattr(obj, "coeffs"):                 # SHCoeffs
        c = np.asarray(obj.coeffs)
    elif hasattr(obj, "expand") and hasattr(obj, "data"):   # SHGrid
        c = np.asarray(obj.expand().coeffs)
    else:                                      # already a coeffs ndarray
        c = np.asarray(obj)
    out = np.zeros((2, lmax + 1, lmax + 1))
    Lc = min(c.shape[1] - 1, lmax)
    out[:, :Lc + 1, :Lc + 1] = c[:, :Lc + 1, :Lc + 1]
    return out


def _degree_modes(clm, l):
    """Stack the 2l+1 real modes at degree l: cos m=0..l, then sin m=1..l."""
    return np.concatenate([clm[0, l, 0:l + 1], clm[1, l, 1:l + 1]])


# ----------------------------------------------------------------------
#  core fits
# ----------------------------------------------------------------------
def _fit_PQ(w_c, H_c, G_c, lmax, lmin=2):
    """Per-degree LSQ fit of  w_lm = P(l) H_lm + Q(l) G_lm.
    Returns P, Q and the per-degree RELATIVE residual norm."""
    P = np.full(lmax + 1, np.nan)
    Q = np.full(lmax + 1, np.nan)
    res = np.full(lmax + 1, np.nan)
    for l in range(lmin, lmax + 1):
        H = _degree_modes(H_c, l)
        G = _degree_modes(G_c, l)
        W = _degree_modes(w_c, l)
        nrm = np.linalg.norm(W)
        if nrm == 0.0:
            continue
        A = np.column_stack([H, G])
        # guard against an all-zero design row set (e.g. fully zeroed degree)
        if not np.any(A):
            continue
        sol, *_ = np.linalg.lstsq(A, W, rcond=None)
        P[l], Q[l] = sol
        res[l] = np.linalg.norm(A @ sol - W) / nrm
    return P, Q, res


def _classical_Z(w_c, H_c, lmax, lmin=2):
    """Classical topography admittance Z(l) = S_wH(l) / S_HH(l)."""
    Z = np.full(lmax + 1, np.nan)
    for l in range(lmin, lmax + 1):
        W = _degree_modes(w_c, l)
        H = _degree_modes(H_c, l)
        S_HH = float(np.sum(H * H))
        if S_HH != 0.0:
            Z[l] = float(np.sum(W * H)) / S_HH
    return Z


# ----------------------------------------------------------------------
#  main entry point
# ----------------------------------------------------------------------
def admittance_diagnostic(w_M3, topo, geoid, R, lmax,
                          w_DSP=None, dsp_units_to_m=1.0,
                          lmin=2, make_plots=True, title_tag=""):
    """See module docstring. Returns a dict of per-degree arrays."""
    w_c = _to_coeffs(w_M3, lmax)
    H_c = _to_coeffs(topo, lmax)
    G_c = _to_coeffs(geoid, lmax)

    # H = topo - R, G = geoid - R : only the l=0 monopole changes, and we fit
    # from lmin>=2, so the subtraction is cosmetic here. Do it anyway for clarity.
    H_c = H_c.copy(); H_c[0, 0, 0] -= R
    G_c = G_c.copy(); G_c[0, 0, 0] -= R

    ll = np.arange(lmax + 1)
    P_M3, Q_M3, r_M3 = _fit_PQ(w_c, H_c, G_c, lmax, lmin)
    Z_M3 = _classical_Z(w_c, H_c, lmax, lmin)

    out = dict(l=ll, P_M3=P_M3, Q_M3=Q_M3, resid_M3=r_M3, Z_M3=Z_M3)

    have_dsp = w_DSP is not None
    if have_dsp:
        wD_c = _to_coeffs(w_DSP, lmax) * dsp_units_to_m
        P_D, Q_D, r_D = _fit_PQ(wD_c, H_c, G_c, lmax, lmin)
        Z_D = _classical_Z(wD_c, H_c, lmax, lmin)
        out.update(P_DSP=P_D, Q_DSP=Q_D, resid_DSP=r_D, Z_DSP=Z_D)

    # ---- printed table ----
    print("\n" + "=" * 78)
    print(f"  M3 <-> DSP PER-DEGREE TRANSFER DIAGNOSTIC {title_tag}")
    print("=" * 78)
    if dsp_units_to_m == 1.0 and have_dsp:
        print("  NOTE: dsp_units_to_m=1.0 -- verify DSP w is in METRES (km files -> 1e3)")
    if have_dsp:
        print(f"{'l':>3} | {'P_M3':>9} {'P_DSP':>9} {'P_M3/P_DSP':>10} | "
              f"{'Q_M3':>9} {'Q_DSP':>9} | {'res_M3':>8} {'res_DSP':>8}")
        print("-" * 78)
        for l in range(lmin, lmax + 1):
            rat = P_M3[l] / P_D[l] if (P_D[l] not in (0.0,) and np.isfinite(P_D[l])) else np.nan
            print(f"{l:>3} | {P_M3[l]:9.3e} {P_D[l]:9.3e} {rat:10.4f} | "
                  f"{Q_M3[l]:9.3e} {Q_D[l]:9.3e} | {r_M3[l]:8.1e} {r_D[l]:8.1e}")
    else:
        print(f"{'l':>3} | {'P_M3':>10} {'Q_M3':>10} {'Z_M3':>10} {'resid_M3':>10}")
        print("-" * 78)
        for l in range(lmin, lmax + 1):
            print(f"{l:>3} | {P_M3[l]:10.3e} {Q_M3[l]:10.3e} {Z_M3[l]:10.3e} {r_M3[l]:10.1e}")
    print("-" * 78)
    print(f"  max M3 residual (constant-Te self-check, want ~1e-10): {np.nanmax(r_M3):.2e}")
    if have_dsp:
        print(f"  median DSP residual (>> M3 => geoid convention differs): {np.nanmedian(r_D):.2e}")
    print("=" * 78 + "\n")

    # ---- plots ----
    if make_plots and _HAVE_MPL:
        x = np.arange(lmin, lmax + 1)
        if have_dsp:
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            ax[0, 0].plot(x, P_M3[lmin:], 'o-', label='M3', ms=3)
            ax[0, 0].plot(x, P_D[lmin:], 's--', label='DSP', ms=3)
            ax[0, 0].set_title('Topography transfer P(l)'); ax[0, 0].legend()
            ax[0, 0].set_xlabel('degree l'); ax[0, 0].grid(True, alpha=.3)

            ax[0, 1].plot(x, Q_M3[lmin:], 'o-', label='M3', ms=3)
            ax[0, 1].plot(x, Q_D[lmin:], 's--', label='DSP', ms=3)
            ax[0, 1].set_title('Geoid transfer Q(l)'); ax[0, 1].legend()
            ax[0, 1].set_xlabel('degree l'); ax[0, 1].grid(True, alpha=.3)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratP = P_M3[lmin:] / P_D[lmin:]
                ratQ = Q_M3[lmin:] / Q_D[lmin:]
            ax[1, 0].axhline(1.0, color='k', lw=.8)
            ax[1, 0].plot(x, ratP, 'o-', label='P_M3/P_DSP', ms=3)
            ax[1, 0].plot(x, ratQ, '^-', label='Q_M3/Q_DSP', ms=3)
            ax[1, 0].set_title('Transfer ratios (flat=scaler, ramp=continuation)')
            ax[1, 0].legend(); ax[1, 0].set_xlabel('degree l'); ax[1, 0].grid(True, alpha=.3)

            ax[1, 1].semilogy(x, r_M3[lmin:], 'o-', label='M3 residual', ms=3)
            ax[1, 1].semilogy(x, r_D[lmin:], 's--', label='DSP residual', ms=3)
            ax[1, 1].set_title('Regression residual (DSP high => geoid differs)')
            ax[1, 1].legend(); ax[1, 1].set_xlabel('degree l'); ax[1, 1].grid(True, alpha=.3)
        else:
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
            ax[0].plot(x, P_M3[lmin:], 'o-', ms=3); ax[0].set_title('P(l) topo transfer')
            ax[1].plot(x, Q_M3[lmin:], 'o-', ms=3); ax[1].set_title('Q(l) geoid transfer')
            ax[2].semilogy(x, r_M3[lmin:], 'o-', ms=3); ax[2].set_title('residual (want ~0)')
            for a in ax:
                a.set_xlabel('degree l'); a.grid(True, alpha=.3)
        plt.suptitle(f'M3 per-degree transfer diagnostic {title_tag}')
        plt.tight_layout()
        plt.show()

    return out


# ----------------------------------------------------------------------
#  stand-alone self-test (runs with plain numpy, no pyshtools / no data)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L = 14

    def rand_clm(L):
        c = np.zeros((2, L + 1, L + 1))
        for l in range(L + 1):
            c[0, l, 0:l + 1] = rng.standard_normal(l + 1)
            c[1, l, 1:l + 1] = rng.standard_normal(l)
        return c

    class _FakeCoeffs:           # minimal stand-in for an SHCoeffs object
        def __init__(self, c): self.coeffs = c

    H = rand_clm(L); G = rand_clm(L); Gdsp = rand_clm(L)
    Pt = np.array([0, 0] + [-0.30 - 0.020 * l for l in range(2, L + 1)])
    Qt = np.array([0, 0] + [ 0.50 + 0.010 * l for l in range(2, L + 1)])

    def synth(Hf, Gf):
        w = np.zeros((2, L + 1, L + 1))
        for l in range(2, L + 1):
            w[0, l, 0:l + 1] = Pt[l] * Hf[0, l, 0:l + 1] + Qt[l] * Gf[0, l, 0:l + 1]
            w[1, l, 1:l + 1] = Pt[l] * Hf[1, l, 1:l + 1] + Qt[l] * Gf[1, l, 1:l + 1]
        return w

    w_M3 = synth(H, G)            # M3 with the observed geoid
    w_DSP = synth(H, Gdsp)        # DSP secretly using a different geoid

    print(">>> self-test: DSP residual should light up because its geoid differs")
    admittance_diagnostic(
        w_M3=_FakeCoeffs(w_M3), topo=_FakeCoeffs(H), geoid=_FakeCoeffs(G),
        R=0.0, lmax=L, w_DSP=_FakeCoeffs(w_DSP), dsp_units_to_m=1.0,
        make_plots=False, title_tag="(SELF-TEST)")