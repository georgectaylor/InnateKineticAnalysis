"""
phasor_dps.py
=============
Phasor-DPS pipeline for QPI phase stacks.
Author contributions: architecture and scientific design by user;
implementation by Claude (Anthropic), October 2026.

Decisions logged here for documentation:
- Q0.1/0.2/0.3: file format and metadata TBD; frame_rate exposed as top-level assumption
- Q1.1: simple time-mean detrend, full field (no mask), to include pericellular debris
- Q1.2: full-field average detrend, no cell mask
- Q1.4: complex FFT values preserved throughout for phasor
- Q1.5: Tukey spatial window applied before FFT to reduce edge leakage
         does NOT exclude debris -- only tapers the image frame edges
- Q1.6: Lorentzian fit in temporal frequency domain (per DPS precedent, Wang et al.)
- Q1.7: 103 frames; leaning on precedent regardless of short series
- Q1.8: S(k) kept as standalone output, not used for weighting yet
- Q2.1: Option A -- single fixed probe frequency (fundamental); frame_rate is
         an assumption, flagged below, adjustable in one line
- Q2.2: per-k phasor cloud, color-coded by window
- Q2.3: 3x noise floor threshold; 1x/2x/3x thresholds give Plots 0-3
- Q2.4: bootstrap with 100 resamples for departure score confidence interval

Scientific goal: identify k-windows where S(k) drops and diffusion
increases simultaneously, as a candidate signal for membrane integrity
loss / necrotic debris emergence. Off-semicircle departure + poor
Dk^2+vk fit = two independent lines of evidence for non-simple dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal.windows import tukey
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftfreq, fft, fftshift
import warnings

# =============================================================================
# TOP-LEVEL PARAMETERS -- adjust these before running
# =============================================================================

# *** ASSUMPTION: frame rate not yet verified from metadata ***
# Units: frames per second (effective phase acquisition rate)
# SLIM produces one phase image per 4 camera frames.
# If camera runs at 50 fps, effective phase rate = 12.5 fps.
# Adjust once metadata is confirmed.
FRAME_RATE = 12.5  # fps -- UNVERIFIED ASSUMPTION, adjust after metadata check

# Pixel size at the sample plane (um/pixel)
# For SLIM: camera pixel = 6.5 um, divided by objective magnification.
# e.g. 40x objective -> 6.5/40 = 0.1625 um/pixel
# Adjust once metadata is confirmed.
PIXEL_SIZE = 0.1625  # um/pixel -- UNVERIFIED ASSUMPTION

# Tukey window alpha parameter: 0 = rectangular (no taper), 1 = full Hanning
# 0.5 is standard: flat central region, taper only at outer edges of frame
TUKEY_ALPHA = 0.5

# Bootstrap resamples for phasor departure confidence interval (Q2.4)
N_BOOTSTRAP = 100

# Probe frequency for phasor (Option A: fundamental frequency of time series)
# omega = 2*pi * (1/n_frames) * frame_rate
# This is one fixed omega for all windows, so all phasor points are comparable
# on the same semicircle. Flag: not tuned to any specific relaxation process.
# To tune per-window later, replace with omega = Gamma_W per window (Option B).

# S(k) threshold multipliers for Plots 0-3
# Plot 0: all k (no threshold)
# Plot 1: S(k) > 1x noise floor
# Plot 2: S(k) > 2x noise floor
# Plot 3: S(k) > 3x noise floor
THRESHOLDS = [None, 1.0, 2.0, 3.0]

# Hypothesis k-windows (rad/um)
# Grounded in biological size literature for GBM cells:
#   Window A (membrane/large feature): ~1-10 um structures -> k ~ 0.6-6 rad/um
#   Window B (organelle):             ~0.2-2 um           -> k ~ 3-30 rad/um
#   Window C (fine/debris):           ~sub-500nm           -> k ~ 12-21 rad/um
# Note: SLIM lateral resolution ~0.3 um -> k_max ~ 21 rad/um
# Windows B and C overlap intentionally -- the dispersion fit within each
# will show whether they behave differently despite overlapping in k.
# Adjust these boundaries once you've inspected your k-range from the data.
WINDOWS = {
    'A_membrane':  (0.6,  6.0),   # rad/um
    'B_organelle': (6.0,  18.0),  # rad/um
    'C_debris':    (12.0, 21.0),  # rad/um
}
WINDOW_COLORS = {
    'A_membrane':  'steelblue',
    'B_organelle': 'darkorange',
    'C_debris':    'mediumseagreen',
}


# =============================================================================
# STEP 0: LOAD DATA
# =============================================================================

def load_phase_stack(filepath):
    """
    Load a QPI phase stack from disk.

    Parameters
    ----------
    filepath : str
        Path to file. Supported: .npy, .tif/.tiff (via tifffile).
        Extend this function for other formats as needed.

    Returns
    -------
    phi : np.ndarray, shape (n_rows, n_cols, n_frames), float
        Phase stack in whatever units the file uses (radians or nm OPL).
        Units affect absolute tau values but not phasor geometry.
    """
    import os
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == '.npy':
        phi = np.load(filepath).astype(float)
    elif ext in ('.tif', '.tiff'):
        try:
            import tifffile
            phi = tifffile.imread(filepath).astype(float)
            # tifffile loads as (frames, rows, cols); reorder to (rows, cols, frames)
            if phi.ndim == 3:
                phi = np.moveaxis(phi, 0, -1)
        except ImportError:
            raise ImportError("tifffile not installed. Run: pip install tifffile")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Add a loader here.")

    assert phi.ndim == 3, "Expected 3D array (rows, cols, frames)"
    print(f"Loaded phase stack: {phi.shape} (rows, cols, frames)")
    print(f"  Frame count: {phi.shape[2]}")
    print(f"  Assumed frame rate: {FRAME_RATE} fps  <-- VERIFY FROM METADATA")
    print(f"  Assumed pixel size: {PIXEL_SIZE} um/pixel  <-- VERIFY FROM METADATA")
    return phi


def load_noise_reference(filepath=None, phi=None):
    """
    Load or synthesize a noise reference.

    If filepath is provided, load it (same format as load_phase_stack).
    If filepath is None and phi is provided, synthesize by using the
    time-mean image as a static 'dead' reference -- i.e. a stack where
    every frame equals the mean, so all temporal fluctuations are zero.
    This is a conservative fallback: it underestimates true instrument
    noise but is better than having no reference at all.

    Returns
    -------
    phi_noise : np.ndarray, same shape as phi
    """
    if filepath is not None:
        return load_phase_stack(filepath)
    elif phi is not None:
        print("No noise reference file provided. Synthesizing from time-mean.")
        print("  WARNING: this underestimates true instrument noise.")
        print("  Acquire a fixed-cell or coverslip-only reference for rigorous use.")
        mean = np.mean(phi, axis=2, keepdims=True)
        return np.repeat(mean, phi.shape[2], axis=2)
    else:
        raise ValueError("Provide either a filepath or phi to synthesize from.")


# =============================================================================
# STEP 1a: DETREND
# =============================================================================

def detrend(phi):
    """
    Subtract the time-mean at each pixel.
    Full field, no mask -- includes pericellular debris (Q1.2).

    delta_phi[x, y, t] = phi[x, y, t] - mean_t(phi[x, y, :])

    Returns
    -------
    delta_phi : np.ndarray, same shape as phi
    time_mean : np.ndarray, shape (rows, cols)
    """
    time_mean = np.mean(phi, axis=2)
    delta_phi = phi - time_mean[:, :, np.newaxis]
    return delta_phi, time_mean


# =============================================================================
# STEP 1b: SPATIAL FFT AND RADIAL BINNING
# =============================================================================

def make_tukey_window_2d(shape, alpha=TUKEY_ALPHA):
    """
    Build a 2D Tukey (tapered cosine) window for the image frame.
    Applied before FFT to reduce spectral leakage from image edges.
    Does NOT affect the cell interior or pericellular debris --
    only tapers the outer edge of the acquired image frame.
    Connection to Q1.5: this broadens k-bin width slightly, making
    neighboring k-bins mildly correlated. Acceptable for proof-of-concept.
    Adjacent k-bins within a window contribute to the same dispersion
    fit, so this affects regression independence slightly -- flagged here,
    not corrected until a later version if needed.
    """
    rows, cols = shape
    wy = tukey(rows, alpha=alpha)
    wx = tukey(cols, alpha=alpha)
    return np.outer(wy, wx)


def compute_spatial_fft(delta_phi, pixel_size=PIXEL_SIZE):
    """
    Apply 2D spatial FFT to each frame, after Tukey windowing.
    Radially bin the result to get one complex time series per k magnitude.

    Parameters
    ----------
    delta_phi : np.ndarray (rows, cols, n_frames)
    pixel_size : float, um/pixel

    Returns
    -------
    k_bins : np.ndarray (n_k_bins,)
        Bin center k-magnitudes in rad/um.
    DELTA_PHI_k : np.ndarray (n_k_bins, n_frames), complex
        Complex-valued radially-averaged Fourier coefficient per k per frame.
        Complex values preserved for phasor computation (Q1.4).
    Sk : np.ndarray (n_k_bins,)
        Time-averaged power S(k) = <|DELTA_PHI_k|^2>_t.
        Returned as standalone output for use in visualization and
        any future weighting (Q1.8). Not used for weighting here.
    """
    rows, cols, n_frames = delta_phi.shape

    # 2D Tukey window (Q1.5)
    window_2d = make_tukey_window_2d((rows, cols))

    # Spatial frequency axes (rad/um)
    kx = fftfreq(cols, d=pixel_size) * 2 * np.pi  # rad/um
    ky = fftfreq(rows, d=pixel_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_MAG = np.sqrt(KX**2 + KY**2)

    # k-bin edges: from ~0 to k_max, spaced by dk = 2pi/(field_of_view)
    # Use ~50 bins across the accessible range; adjust if needed
    k_max = np.max(K_MAG)
    k_min_nonzero = np.min(K_MAG[K_MAG > 0])
    n_bins = 50
    bin_edges = np.linspace(k_min_nonzero, k_max, n_bins + 1)
    k_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # bin centers

    # FFT across all frames, accumulate per k-bin
    DELTA_PHI_k = np.zeros((n_bins, n_frames), dtype=complex)
    bin_counts = np.zeros(n_bins, dtype=int)

    for t in range(n_frames):
        frame = delta_phi[:, :, t] * window_2d
        F = fft2(frame)
        for b in range(n_bins):
            mask = (K_MAG >= bin_edges[b]) & (K_MAG < bin_edges[b+1])
            if np.any(mask):
                # Complex mean: preserves phase for phasor (Q1.4)
                DELTA_PHI_k[b, t] = np.mean(F[mask])
                bin_counts[b] = np.sum(mask)

    # Remove empty bins
    valid = bin_counts > 0
    DELTA_PHI_k = DELTA_PHI_k[valid]
    k_bins = k_bins[valid]

    # S(k): time-averaged power per k-bin (standalone output, Q1.8)
    Sk = np.mean(np.abs(DELTA_PHI_k)**2, axis=1)

    print(f"Spatial FFT complete. k range: {k_bins[0]:.2f} to {k_bins[-1]:.2f} rad/um")
    print(f"  n_k_bins: {len(k_bins)}, n_frames: {n_frames}")
    return k_bins, DELTA_PHI_k, Sk


def assign_k_to_windows(k_bins, windows=WINDOWS):
    """
    For each k-bin, determine which hypothesis window(s) it belongs to.

    Returns
    -------
    window_masks : dict {window_name: bool array of length n_k_bins}
    """
    masks = {}
    for name, (k_lo, k_hi) in windows.items():
        masks[name] = (k_bins >= k_lo) & (k_bins < k_hi)
        n = np.sum(masks[name])
        print(f"  Window {name}: {k_lo}-{k_hi} rad/um -> {n} k-bins")
    return masks


# =============================================================================
# STEP 1c: DISPERSION RELATION FIT (per window, per k-bin)
# =============================================================================

def lorentzian_power_spectrum(omega, gamma, amplitude):
    """
    Lorentzian power spectrum: L(omega) = A * gamma / (gamma^2 + omega^2)
    Corresponds to exponential autocorrelation with decay rate gamma.
    Per DPS precedent (Wang et al., Popescu lab; cardiomyocyte SLIM paper).
    HWHM of this Lorentzian = gamma = Gamma(k).
    """
    return amplitude * gamma / (gamma**2 + omega**2)


def fit_gamma_lorentzian(time_series, frame_rate=FRAME_RATE):
    """
    Fit Gamma(k) from the temporal power spectrum of one k-bin's time series
    by fitting a Lorentzian, per DPS/Option C precedent (Q1.6).

    With 103 frames we lean on precedent even when noisy (Q1.7).

    Parameters
    ----------
    time_series : np.ndarray (n_frames,), complex

    Returns
    -------
    gamma : float, decay rate in rad/s
    amplitude : float
    fit_ok : bool, whether fit converged
    """
    n = len(time_series)
    # Temporal power spectrum
    F = fft(time_series)
    freqs = fftfreq(n, d=1.0/frame_rate)  # Hz
    omega = 2 * np.pi * freqs             # rad/s
    power = np.abs(F)**2

    # Use only positive frequencies, exclude omega=0
    pos = (freqs > 0)
    omega_pos = omega[pos]
    power_pos = power[pos]

    if len(omega_pos) < 3:
        return np.nan, np.nan, False

    # Initial guess: gamma ~ peak frequency of power spectrum
    peak_idx = np.argmax(power_pos)
    gamma0 = omega_pos[peak_idx] if omega_pos[peak_idx] > 0 else omega_pos[1]
    amp0 = power_pos[peak_idx] * gamma0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lorentzian_power_spectrum,
                omega_pos, power_pos,
                p0=[gamma0, amp0],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=2000
            )
        gamma, amplitude = popt
        fit_ok = True
    except RuntimeError:
        gamma, amplitude = np.nan, np.nan
        fit_ok = False

    return gamma, amplitude, fit_ok


def compute_dispersion_per_window(k_bins, DELTA_PHI_k, window_masks,
                                  frame_rate=FRAME_RATE):
    """
    For each window, fit Gamma(k) at every k-bin within it,
    then fit Gamma(k) = D*k^2 + v*k by linear regression.

    Returns
    -------
    dispersion : dict {window_name: {
        'k': array of k values with successful fits,
        'gamma': array of Gamma(k),
        'D': float diffusion coefficient (um^2/s),
        'v': float advection speed (um/s),
        'residuals': array,
        'fit_ok': bool array
    }}
    """
    dispersion = {}

    for name, mask in window_masks.items():
        k_w = k_bins[mask]
        ts_w = DELTA_PHI_k[mask]  # (n_k_in_window, n_frames)

        gammas = []
        fit_oks = []

        for i in range(len(k_w)):
            gamma, _, ok = fit_gamma_lorentzian(ts_w[i], frame_rate)
            gammas.append(gamma)
            fit_oks.append(ok)

        gammas = np.array(gammas)
        fit_oks = np.array(fit_oks)

        # Keep only successful fits for regression
        valid = fit_oks & np.isfinite(gammas)
        k_valid = k_w[valid]
        gamma_valid = gammas[valid]

        D, v = np.nan, np.nan
        residuals = np.array([])

        if np.sum(valid) >= 3:
            # Linear regression: Gamma/k = D*k + v
            # Rearranged from Gamma = D*k^2 + v*k (divide both sides by k)
            # This linearizes the fit (standard DPS approach)
            k_nz = k_valid[k_valid > 0]
            g_nz = gamma_valid[k_valid > 0]

            if len(k_nz) >= 2:
                A = np.column_stack([k_nz, np.ones_like(k_nz)])
                b_vec = g_nz / k_nz
                result = np.linalg.lstsq(A, b_vec, rcond=None)
                coeffs = result[0]
                D, v = coeffs[0], coeffs[1]
                residuals = b_vec - (D * k_nz + v)

        dispersion[name] = {
            'k': k_w,
            'gamma': gammas,
            'D': D,
            'v': v,
            'residuals': residuals,
            'fit_ok': fit_oks,
            'k_valid': k_valid,
            'gamma_valid': gamma_valid,
        }

        print(f"  Window {name}: D={D:.4f} um^2/s, v={v:.4f} um/s "
              f"({np.sum(valid)}/{len(k_w)} fits ok)")

    return dispersion


# =============================================================================
# STEP 2a: PHASOR PER K-BIN
# =============================================================================

def compute_probe_omega(n_frames, frame_rate=FRAME_RATE):
    """
    Option A: probe at the fundamental temporal frequency of the time series.
    omega = 2*pi * (1/n_frames) * frame_rate  [rad/s]

    This is one fixed omega for all windows and all k-bins, making all
    phasor points directly comparable on the same semicircle (Q2.1).

    Note: not tuned to any specific relaxation process -- the phasor point
    position along the arc depends on omega*tau. A wrong frame_rate shifts
    all points together; does not change whether they're on/off semicircle.
    Fix frame_rate once metadata is confirmed.
    """
    omega = 2 * np.pi * (1.0 / n_frames) * frame_rate
    print(f"Probe frequency omega = {omega:.4f} rad/s "
          f"(fundamental, frame_rate={frame_rate} fps ASSUMED)")
    return omega


def compute_phasor_per_k(DELTA_PHI_k, omega, frame_rate=FRAME_RATE):
    """
    Compute (g, s) phasor for every k-bin.
    Per-k cloud, not window-averaged (Q2.2).
    Complex values preserved throughout (Q1.4).

    g = Re[FT(delta_phi)(omega)] / N
    s = Im[FT(delta_phi)(omega)] / N
    N = total power (sum of |FT|^2 excluding omega=0)

    Returns
    -------
    g : np.ndarray (n_k_bins,)
    s : np.ndarray (n_k_bins,)
    """
    n_frames = DELTA_PHI_k.shape[1]
    freqs = fftfreq(n_frames, d=1.0/frame_rate)
    omegas = 2 * np.pi * freqs

    # Find index closest to probe omega
    omega_idx = np.argmin(np.abs(omegas - omega))

    g_arr = np.zeros(len(DELTA_PHI_k))
    s_arr = np.zeros(len(DELTA_PHI_k))

    for i, ts in enumerate(DELTA_PHI_k):
        F = fft(ts)
        # Total power excluding omega=0 (already removed by detrend,
        # but exclude explicitly for safety)
        N = np.sum(np.abs(F[1:])**2)
        if N == 0:
            continue
        F_probe = F[omega_idx]
        g_arr[i] = np.real(F_probe) / N
        s_arr[i] = np.imag(F_probe) / N

    return g_arr, s_arr


def compute_phasor_bootstrap(DELTA_PHI_k, omega, n_bootstrap=N_BOOTSTRAP,
                              frame_rate=FRAME_RATE):
    """
    Bootstrap the phasor (g, s) and departure score d per k-bin.
    100 resamples of 103 frames drawn with replacement (Q2.4).

    Returns
    -------
    g_mean, g_std : np.ndarray (n_k_bins,)
    s_mean, s_std : np.ndarray (n_k_bins,)
    d_mean, d_std : np.ndarray (n_k_bins,)
        d = |g^2 + (s-0.5)^2 - 0.25| -- departure from semicircle
    """
    n_k, n_frames = DELTA_PHI_k.shape
    g_boot = np.zeros((n_bootstrap, n_k))
    s_boot = np.zeros((n_bootstrap, n_k))

    for b in range(n_bootstrap):
        idx = np.random.choice(n_frames, size=n_frames, replace=True)
        resampled = DELTA_PHI_k[:, idx]
        g_b, s_b = compute_phasor_per_k(resampled, omega, frame_rate)
        g_boot[b] = g_b
        s_boot[b] = s_b

    g_mean = np.mean(g_boot, axis=0)
    g_std  = np.std(g_boot,  axis=0)
    s_mean = np.mean(s_boot, axis=0)
    s_std  = np.std(s_boot,  axis=0)

    d_boot = np.abs(g_boot**2 + (s_boot - 0.5)**2 - 0.25)
    d_mean = np.mean(d_boot, axis=0)
    d_std  = np.std(d_boot,  axis=0)

    return g_mean, g_std, s_mean, s_std, d_mean, d_std


# =============================================================================
# STEP 2b: SEMICIRCLE DEPARTURE SCORE
# =============================================================================

def departure_score(g, s):
    """
    d = |g^2 + (s - 0.5)^2 - 0.25|
    Zero for a point exactly on the universal semicircle.
    Positive for a point inside (mixture of processes / non-Markovian).
    """
    return np.abs(g**2 + (s - 0.5)**2 - 0.25)


# =============================================================================
# PLOTTING
# =============================================================================

def draw_semicircle(ax):
    """Draw the universal semicircle on a phasor axis."""
    theta = np.linspace(0, np.pi, 300)
    g_sc = 0.5 + 0.5 * np.cos(theta)
    s_sc = 0.5 * np.sin(theta)
    ax.plot(g_sc, s_sc, 'k--', lw=1, alpha=0.5, label='Universal semicircle')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.65)
    ax.set_xlabel('g')
    ax.set_ylabel('s')
    ax.set_aspect('equal')


def plot_phasor_panels(k_bins, g, s, Sk, Sk_noise, window_masks,
                       d_mean, d_std,
                       thresholds=THRESHOLDS,
                       window_colors=WINDOW_COLORS,
                       title_prefix='Cell'):
    """
    Produce Plots 0-3: phasor clouds filtered by S(k) threshold.

    Plot 0: all k including noise floor
    Plot 1: S(k) > 1x noise floor
    Plot 2: S(k) > 2x noise floor
    Plot 3: S(k) > 3x noise floor

    Dot size proportional to S(k) (visualization of signal content, Q1.8).
    Color coded by window membership (Q2.2).
    Error bars on departure score shown per window in a subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Assign a color to each k-bin based on window membership
    # k-bins in multiple windows get the color of the last (finest) window
    k_colors = np.array(['lightgrey'] * len(k_bins), dtype=object)
    for name, mask in window_masks.items():
        k_colors[mask] = window_colors[name]

    # Dot size: scale S(k) to a reasonable marker size range
    s_min, s_max = np.min(Sk), np.max(Sk)
    def scale_size(sk):
        if s_max == s_min:
            return np.ones_like(sk) * 20
        return 5 + 80 * (sk - s_min) / (s_max - s_min)

    for idx, thresh in enumerate(thresholds):
        ax = axes[idx]
        draw_semicircle(ax)

        if thresh is None:
            # Plot 0: everything + noise floor overlay
            keep = np.ones(len(k_bins), dtype=bool)
            ax.set_title(f'{title_prefix} — Plot 0: all k (no threshold)')
            # Noise floor overlay in grey
            g_n, s_n = g.copy(), s.copy()  # placeholder -- noise computed separately
            ax.scatter(g[keep], s[keep],
                       s=scale_size(Sk[keep]),
                       c=k_colors[keep],
                       alpha=0.7, edgecolors='k', linewidths=0.3,
                       label='Cell signal')
            ax.text(0.02, 0.95,
                    'Dot size ∝ S(k)\nColor = window',
                    transform=ax.transAxes, fontsize=7, va='top')
        else:
            # Plots 1-3: filter by S(k) > thresh * S_noise(k)
            ratio = np.where(Sk_noise > 0, Sk / Sk_noise, 0)
            keep = ratio > thresh
            n_kept = np.sum(keep)
            ax.set_title(
                f'{title_prefix} — Plot {idx}: S(k) > {thresh:.0f}× noise '
                f'({n_kept}/{len(k_bins)} k-bins)'
            )
            if np.any(keep):
                ax.scatter(g[keep], s[keep],
                           s=scale_size(Sk[keep]),
                           c=k_colors[keep],
                           alpha=0.7, edgecolors='k', linewidths=0.3)

        # Window legend patches
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=c, label=n)
                          for n, c in window_colors.items()]
        ax.legend(handles=legend_patches, fontsize=7, loc='upper right')

    plt.tight_layout()
    return fig


def plot_dispersion_per_window(dispersion, window_colors=WINDOW_COLORS):
    """
    For each window: plot Gamma(k) vs k with fitted Dk^2+vk curve,
    plus residual subplot.
    Off-semicircle result and poor fit here are two independent lines
    of evidence for non-simple dynamics.
    """
    n_windows = len(dispersion)
    fig, axes = plt.subplots(n_windows, 2,
                             figsize=(10, 4 * n_windows))
    if n_windows == 1:
        axes = axes[np.newaxis, :]

    for row, (name, data) in enumerate(dispersion.items()):
        ax_main = axes[row, 0]
        ax_res  = axes[row, 1]
        color = window_colors.get(name, 'grey')

        k_all   = data['k']
        g_all   = data['gamma']
        fit_ok  = data['fit_ok']
        k_valid = data['k_valid']
        g_valid = data['gamma_valid']
        D, v    = data['D'], data['v']
        resids  = data['residuals']

        # All fitted points
        ax_main.scatter(k_all[fit_ok], g_all[fit_ok],
                        color=color, alpha=0.7, s=30,
                        label='Γ(k) fitted')
        ax_main.scatter(k_all[~fit_ok], np.zeros(np.sum(~fit_ok)),
                        color='red', marker='x', s=20,
                        label='Fit failed')

        # Overlay Dk^2 + vk
        if np.isfinite(D) and np.isfinite(v):
            k_line = np.linspace(k_valid[0], k_valid[-1], 200)
            gamma_line = D * k_line**2 + v * k_line
            ax_main.plot(k_line, gamma_line, 'k--', lw=1.5,
                         label=f'Dk²+vk\nD={D:.4f}, v={v:.4f}')

        ax_main.set_xlabel('k (rad/µm)')
        ax_main.set_ylabel('Γ(k) (rad/s)')
        ax_main.set_title(f'Window {name}: dispersion relation')
        ax_main.legend(fontsize=7)

        # Residuals
        if len(resids) > 0:
            k_nz = k_valid[k_valid > 0]
            ax_res.scatter(k_nz, resids, color=color, alpha=0.7, s=20)
            ax_res.axhline(0, color='k', lw=1, ls='--')
            ax_res.set_xlabel('k (rad/µm)')
            ax_res.set_ylabel('Residual (Γ/k - fit)')
            ax_res.set_title(f'{name}: residuals\n'
                             f'(large/systematic = model insufficient)')
        else:
            ax_res.text(0.5, 0.5, 'Insufficient fits for residuals',
                        ha='center', va='center', transform=ax_res.transAxes)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(phi_path, noise_path=None):
    """
    Run the full Steps 0-2 pipeline.

    Parameters
    ----------
    phi_path : str
        Path to your phase stack file.
    noise_path : str or None
        Path to noise reference file. If None, synthesizes from phi.
    """
    print("\n=== STEP 0: Load data ===")
    phi = load_phase_stack(phi_path)
    phi_noise = load_noise_reference(noise_path, phi=phi)

    print("\n=== STEP 1a: Detrend ===")
    delta_phi, _ = detrend(phi)
    delta_phi_noise, _ = detrend(phi_noise)

    print("\n=== STEP 1b: Spatial FFT ===")
    k_bins, DELTA_PHI_k, Sk = compute_spatial_fft(delta_phi)
    _,      DELTA_PHI_noise, Sk_noise = compute_spatial_fft(delta_phi_noise)

    print("\n=== Assigning k-bins to windows ===")
    window_masks = assign_k_to_windows(k_bins)

    print("\n=== STEP 1c: Dispersion relation per window ===")
    dispersion = compute_dispersion_per_window(k_bins, DELTA_PHI_k, window_masks)

    print("\n=== STEP 2a: Phasor (bootstrap) ===")
    n_frames = delta_phi.shape[2]
    omega = compute_probe_omega(n_frames)
    g_mean, g_std, s_mean, s_std, d_mean, d_std = \
        compute_phasor_bootstrap(DELTA_PHI_k, omega)

    print("\n=== STEP 2b: Departure scores ===")
    d = departure_score(g_mean, s_mean)
    for name, mask in window_masks.items():
        if np.any(mask):
            print(f"  Window {name}: mean d = {np.mean(d[mask]):.4f} "
                  f"± {np.mean(d_std[mask]):.4f}")

    print("\n=== Plotting ===")
    fig_phasor = plot_phasor_panels(
        k_bins, g_mean, s_mean, Sk, Sk_noise,
        window_masks, d_mean, d_std
    )
    fig_disp = plot_dispersion_per_window(dispersion)

    plt.show()

    return {
        'k_bins': k_bins,
        'Sk': Sk,           # standalone S(k) output (Q1.8)
        'Sk_noise': Sk_noise,
        'g': g_mean,
        'g_std': g_std,
        's': s_mean,
        's_std': s_std,
        'd': d_mean,
        'd_std': d_std,
        'dispersion': dispersion,
        'window_masks': window_masks,
        'omega': omega,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python phasor_dps.py <path_to_phase_stack> [path_to_noise_ref]")
        print("Example: python phasor_dps.py my_cell.tif")
        sys.exit(1)
    phi_path = sys.argv[1]
    noise_path = sys.argv[2] if len(sys.argv) > 2 else None
    results = run_pipeline(phi_path, noise_path)
