# %% BUILD / SAVE A TWO-VALUE Te MAP FROM A w<0 CONTOUR
"""
Builds a two-value Te field:  Te = Te_big everywhere, Te = Te_sml inside the
w < 0 region of a reference solution, restricted to a lat/lon box (Tharsis).

What is SAVED is the dimensionless indicator/taper field f in [0,1], NOT Te
itself, so that
        Te(theta,phi) = Te_big + (Te_sml - Te_big) * f(theta,phi)
can be regenerated for any (Te_big, Te_sml) pair without redoing the masking.

Grid convention: DH2, extend=False, lmax = grid_expansion_res  (same grid the
solver expands T_e_parent onto in derive_D_a / build_Omega_*).
"""

import os
import numpy as np
import pyshtools as pysh

TE_MAP_DIR = "Elastic_Thickness_Input_Maps"
os.makedirs(TE_MAP_DIR, exist_ok=True)


# ----------------------------------------------------------------- helpers --
def _w_grid_data(src, lmax_grid, lmax_calc):
    """Return w on a DH2, extend=False grid from SHCoeffs / SHGrid / ndarray."""
    if isinstance(src, pysh.SHCoeffs):
        return src.expand(lmax=lmax_grid, lmax_calc=lmax_calc,
                          grid='DH2', extend=False).data
    if isinstance(src, pysh.SHGrid):
        d = src.data
        if d.shape[0] % 2:                      # extended grid -> drop dup row/col
            d = d[:-1, :-1]
        clm = pysh.SHGrid.from_array(d).expand()
        clm = pysh.SHCoeffs.from_array(clm.coeffs[:, :lmax_calc + 1, :lmax_calc + 1])
        return clm.expand(lmax=lmax_grid, grid='DH2', extend=False).data
    d = np.asarray(src)
    if d.shape[0] % 2:
        d = d[:-1, :-1]
    return d


def _dh2_latlon(nlat):
    """Latitudes/longitudes of a DH2 extend=False grid with nlat rows."""
    lat = 90.0 - np.arange(nlat) * (180.0 / nlat)
    lon = np.arange(2 * nlat) * (360.0 / (2 * nlat))
    return lat, lon


def _sh_gaussian_smooth(field, sigma_deg, lmax_work):
    """
    Isotropic Gaussian smoothing on the sphere: B_l = exp(-l(l+1)*sigma^2/2).
    Positive kernel => output stays inside [min, max] of the input, and the
    degree-0 (global mean) is preserved exactly.
    """
    clm = pysh.SHGrid.from_array(field).expand()
    s = np.deg2rad(sigma_deg)
    l = np.arange(clm.lmax + 1)
    clm.coeffs *= np.exp(-l * (l + 1) * s ** 2 / 2.0)[None, :, None]
    return clm.expand(lmax=lmax_work, grid='DH2', extend=False).data


# ------------------------------------------------------------------ builder --
def build_Te_indicator(w_src, lmax_grid, lmax_calc,
                       lat_range=(-50.0, 70.0), lon_range=(225.0, 290.0),
                       sigma_deg=3.5, largest_blob=True, fill_holes=True,
                       renormalise=False):
    """
    Returns f (ndarray on the DH2 extend=False grid, values in [0,1]) and a
    metadata dict. f = 1 inside the target region, 0 outside, with a smooth
    transition of width ~2*sigma_deg across the w = 0 contour.

    sigma_deg : Gaussian smoothing width in degrees. This is what keeps the
                field band-limited; a hard 0/1 step truncated at lmax=45 rings
                by ~10% of (Te_big - Te_sml). ~3-4 deg is about one resolution
                element at lmax=45 (half-wavelength ~237 km ~ 4 deg on Mars).
    """
    w = _w_grid_data(w_src, lmax_grid, lmax_calc)
    nlat = w.shape[0]
    lat, lon = _dh2_latlon(nlat)
    LON, LAT = np.meshgrid(lon, lat)

    box = ((LAT >= lat_range[0]) & (LAT <= lat_range[1]) &
           (LON >= lon_range[0]) & (LON <= lon_range[1]))
    mask = (w < 0.0) & box

    if fill_holes or largest_blob:
        from scipy import ndimage
        if fill_holes:
            mask = ndimage.binary_fill_holes(mask)
        if largest_blob:
            lab, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                mask = (lab == (np.argmax(sizes) + 1))
            elif n == 0:
                raise RuntimeError("No w<0 pixels inside the requested box.")

    f = _sh_gaussian_smooth(mask.astype(float), sigma_deg, nlat // 2 - 1)
    if renormalise:                       # force the plateaus to hit 0 and 1
        f = (f - f.min()) / (f.max() - f.min())
    f = np.clip(f, 0.0, 1.0)

    cw = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(f)
    meta = dict(lmax_grid=lmax_grid, lmax_calc=lmax_calc, nlat=nlat,
                grid='DH2', extend=False, sigma_deg=sigma_deg,
                lat_range=lat_range, lon_range=lon_range,
                largest_blob=largest_blob, fill_holes=fill_holes,
                area_frac_sharp=float((cw * mask).sum() / cw.sum()),
                area_frac_smooth=float((cw * f).sum() / cw.sum()))
    return f, meta


def save_Te_indicator(f, meta, name="Te_twoval_tharsis"):
    path = os.path.join(TE_MAP_DIR, name + ".npz")
    np.savez(path, f=f, **{k: np.array(v) for k, v in meta.items()})
    print(f"Saved indicator field -> {path}")
    return path


def load_Te_indicator(name="Te_twoval_tharsis"):
    d = np.load(os.path.join(TE_MAP_DIR, name + ".npz"), allow_pickle=True)
    f = d['f']
    meta = {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k != 'f'}
    return f, meta


def Te_clm_from_indicator(f, Te_big, Te_sml, lmax_fit, verbose=True):
    """Te = Te_big + (Te_sml - Te_big)*f  ->  SHCoeffs truncated to lmax_fit."""
    Te_grid = Te_big + (Te_sml - Te_big) * f
    clm = pysh.SHGrid.from_array(Te_grid).expand()
    clm = pysh.SHCoeffs.from_array(clm.coeffs[:, :lmax_fit + 1, :lmax_fit + 1])
    if verbose:
        back = clm.expand(lmax=f.shape[0] // 2 - 1, grid='DH2', extend=False).data
        print(f"Te target [{Te_sml/1e3:.1f}, {Te_big/1e3:.1f}] km  ->  after "
              f"lmax={lmax_fit} truncation: min={back.min()/1e3:.2f}, "
              f"max={back.max()/1e3:.2f}, mean={clm.coeffs[0,0,0]/1e3:.2f} km")
    return clm, Te_grid


# ------------------------------------------------------------- sanity plot --
def plot_Te_map(f, Te_big, Te_sml, w_src=None, lmax_grid=None, lmax_calc=None,
                lmax_fit=45, grid=True):
    """Quick 2-panel check: the Te field, and the same field after the
    lmax_fit truncation the solver actually sees, with the w=0 contour."""
    import matplotlib.pyplot as plt
    clm, Te_grid = Te_clm_from_indicator(f, Te_big, Te_sml, lmax_fit, verbose=False)
    Te_trunc = clm.expand(lmax=f.shape[0] // 2 - 1, grid='DH2', extend=False).data

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ext = (0, 360, -90, 90)
    for a, d, t in zip(ax, [Te_grid, Te_trunc],
                       ['Te as designed (grid)',
                        f'Te after lmax={lmax_fit} truncation  '
                        f'[{Te_trunc.min()/1e3:.1f}, {Te_trunc.max()/1e3:.1f}] km']):
        im = a.imshow(d / 1e3, extent=ext, origin='upper', cmap='viridis',
                      vmin=min(Te_sml, Te_trunc.min()) / 1e3,
                      vmax=max(Te_big, Te_trunc.max()) / 1e3)
        if w_src is not None:
            w = _w_grid_data(w_src, lmax_grid, lmax_calc)
            a.contour(Te_grid > 100e3, levels=[0.99], extent=ext, colors='k', origin='upper')
        a.set_title(t)
        a.grid(grid)
        a.set_xticks(np.arange(0, 361, 30))
        a.set_yticks(np.arange(-90, 91, 30))
        fig.colorbar(im, ax=a, label='$T_e$ [km]')
    plt.tight_layout()
    plt.show()
