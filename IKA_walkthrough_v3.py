# -------------------------------------------------------------------
# IKA — Innate Kinetic Analysis
# Phasor-DPS pipeline for quantitative phase imaging data
# George Taylor, 2026
#
# To convert to Jupyter notebook:
#   pip install jupytext
#   jupytext --to notebook IKA_walkthrough_v3.py
#
# Or paste each cell block (between # %% markers) into a new
# Jupyter notebook manually. Blocks marked # %% [markdown] become
# markdown cells (Escape → M in Jupyter).
# -------------------------------------------------------------------

# %% [markdown]
# # IKA — Innate Kinetic Analysis
# ### Phasor-DPS pipeline for quantitative phase imaging data
# George Taylor · October 2026
#
# This notebook demonstrates the IKA pipeline on a single
# glioblastoma (GBM) cell imaged by SLIM (Spatial Light Interference
# Microscopy) at 20-minute intervals over approximately 34 hours
# (103 frames).
#
# **Central question:** At each spatial scale, does intracellular
# mass transport follow a simple two-component model (diffusive +
# advective)? The phasor semicircle is the geometric locus of dynamics
# adequately described by a single exponential relaxation rate. Departure
# from the semicircle quantifies model inadequacy per scale — a
# physics-grounded fingerprint that requires no segmentation, no
# labeling, and no prior assumption about which cellular structures
# are present.
#
# **Output:** One phasor plot. Each dot is one spatial scale (k-bin),
# colored by structural size window, sized by signal power S(k).
#
# ---
# **Goals for future development**
# - Independent Python implementation of core functions
# - Unit tests against synthetic ground truth with known D and v
# - Packaging for pip-installable distribution
# - Temporal window analysis: track how phasor position evolves
#   within a session
# - Multi-channel extension: fluorescence phasors alongside QPI
#   dry-mass phasors for direct comparison of labeled and bulk dynamics
# - Trajectory embedding (PHATE/UMAP) once temporal windows are
#   validated — deferred pending interpretability benchmarking

# %% [markdown]
# ## Cell 1 — Imports and parameters
#
# All adjustable parameters are defined here. To adapt this pipeline
# to a new dataset, only this cell should need to change.

# %%
import os
import re
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import fft2, fft, fftfreq
from scipy.signal.windows import tukey
import tifffile

# --- Data location ---
GBM_FOLDER = 'onesingleGBMcell'   # folder containing per-frame TIFFs

# --- Acquisition parameters ---
PIXEL_SIZE     = 0.1625   # um/pixel  (SLIM, 40x objective)
FRAME_INTERVAL = 1200     # seconds between frames (1 frame per 20 min)

# --- Spatial frequency bin count ---
N_BINS = 50

# --- Tukey window parameter ---
# Controls edge tapering before spatial FFT (0 = rectangular, 1 = Hann)
TUKEY_ALPHA = 0.5

# --- Spatial scale windows (rad/um) ---
# Each window corresponds to a structural size range in the cell.
# Adjust boundaries here; all downstream code uses these variables.
WINDOW_A_MIN, WINDOW_A_MAX = 0.6,  6.0    # ~1–10 um: membrane, large features
WINDOW_B_MIN, WINDOW_B_MAX = 6.0,  18.0   # ~0.2–2 um: organelles
WINDOW_C_MIN, WINDOW_C_MAX = 12.0, 21.0   # <500 nm: fine structure

WINDOWS = {
    'A_membrane':  (WINDOW_A_MIN, WINDOW_A_MAX),
    'B_organelle': (WINDOW_B_MIN, WINDOW_B_MAX),
    'C_fine':      (WINDOW_C_MIN, WINDOW_C_MAX),
}
WINDOW_COLORS = {
    'A_membrane':  'steelblue',
    'B_organelle': 'darkorange',
    'C_fine':      'mediumseagreen',
}

# %% [markdown]
# ## Cell 2 — Load data

# %%
def load_slim_folder(folder_path):
    """
    Load a SLIM time-lapse from a folder of per-frame TIFF files.

    Files are sorted by the integer index after '_i' in the filename
    rather than alphabetically (alphabetical order mis-sorts i10
    before i2, scrambling the time series).

    Returns phi: ndarray (rows, cols, n_frames), float64.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    def get_i_index(filename):
        match = re.search(r'_i(\d+)', filename)
        return int(match.group(1)) if match else 0

    files_sorted = sorted(files, key=get_i_index)
    frames = [tifffile.imread(os.path.join(folder_path, f))
              for f in files_sorted]
    phi = np.stack(frames, axis=-1).astype(float)
    print(f"Loaded: {phi.shape}  (rows, cols, frames)")
    print(f"Total session duration: "
          f"{phi.shape[2] * FRAME_INTERVAL / 3600:.1f} hours")
    return phi


phi = load_slim_folder(GBM_FOLDER)

# %% [markdown]
# ## Cell 3 — Pipeline functions

# %%
def detrend(phi):
    """
    Remove per-pixel time mean from phase stack.

    Eliminates static phase structure so that only temporal
    fluctuations remain, preventing a DC spike in the temporal FFT.
    """
    return phi - np.mean(phi, axis=2, keepdims=True)


def compute_relaxation_modes(delta_phi,
                             pixel_size=PIXEL_SIZE,
                             n_bins=N_BINS,
                             tukey_alpha=TUKEY_ALPHA):
    """
    Radially averaged spatial power per k-bin per frame.

    Two implementation choices worth noting:
    1. Tukey window applied before spatial FFT to reduce edge leakage.
    2. |FFT| magnitude taken before radial averaging — critical for
       large fields (2048×2048). Complex averaging causes near-complete
       cancellation of positive and negative imaginary parts, collapsing
       values to ~1e-17. Magnitude averaging preserves real temporal
       variation (verified std/mean ≈ 0.125 on this dataset).

    Returns
    -------
    k_bins      : 1D array, bin-center spatial frequencies (rad/um)
    DELTA_PHI_k : 2D array (n_valid_bins, n_frames), real-valued
    Sk          : 1D array, time-mean power per k-bin (signal detectability)
    """
    rows, cols, n_frames = delta_phi.shape

    wy = tukey(rows, alpha=tukey_alpha)
    wx = tukey(cols, alpha=tukey_alpha)
    window_2d = np.outer(wy, wx)

    kx = fftfreq(cols, d=pixel_size) * 2 * np.pi
    ky = fftfreq(rows, d=pixel_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_MAG = np.sqrt(KX**2 + KY**2)

    k_max_val = np.max(K_MAG)
    k_min_val = np.min(K_MAG[K_MAG > 0])
    bin_edges = np.linspace(k_min_val, k_max_val, n_bins + 1)
    k_bins_all = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    DELTA_PHI_k_all = np.zeros((n_bins, n_frames))
    bin_counts = np.zeros(n_bins, dtype=int)

    t0 = time.time()
    for t in range(n_frames):
        if t % 10 == 0:
            print(f"  frame {t}/{n_frames}")
        frame = delta_phi[:, :, t] * window_2d
        F_mag = np.abs(fft2(frame))
        for b in range(n_bins):
            mask = (K_MAG >= bin_edges[b]) & (K_MAG < bin_edges[b + 1])
            if np.any(mask):
                DELTA_PHI_k_all[b, t] = np.mean(F_mag[mask])
                bin_counts[b] = np.sum(mask)

    valid = bin_counts > 0
    DELTA_PHI_k = DELTA_PHI_k_all[valid]
    k_bins = k_bins_all[valid]
    Sk = np.mean(DELTA_PHI_k**2, axis=1)

    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  k range: {k_bins[0]:.2f}–{k_bins[-1]:.2f} rad/um, "
          f"{len(k_bins)} bins")
    return k_bins, DELTA_PHI_k, Sk


def assign_windows(k_bins, windows=WINDOWS):
    """
    Return a boolean mask per window for each k-bin.
    Windows are defined in Cell 1 and passed through here,
    so changing WINDOW_*_MIN/MAX propagates automatically.
    """
    return {name: (k_bins >= lo) & (k_bins < hi)
            for name, (lo, hi) in windows.items()}


def compute_phasors(DELTA_PHI_k, frame_interval=FRAME_INTERVAL):
    """
    Compute (g, s) phasor coordinates per k-bin via temporal FFT.

    Probe frequency: the fundamental omega = 2*pi / T_total.
    Normalization: sum of positive-frequency amplitudes, making
    (g, s) dimensionless with the semicircle criterion fixed at
    g^2 + (s - 0.5)^2 = 0.25 regardless of signal amplitude.

    Returns
    -------
    g_arr, s_arr : 1D arrays, phasor coordinates per k-bin
    """
    n_k, n_frames = DELTA_PHI_k.shape
    freqs = fftfreq(n_frames, d=frame_interval)
    omegas = 2 * np.pi * freqs
    omega = 2 * np.pi * (1.0 / n_frames) / frame_interval
    omega_idx = np.argmin(np.abs(omegas - omega))

    g_arr = np.zeros(n_k)
    s_arr = np.zeros(n_k)
    for i in range(n_k):
        ts = DELTA_PHI_k[i]
        F = fft(ts)
        N = np.sum(np.abs(F[1:n_frames // 2]))
        if N == 0:
            continue
        g_arr[i] = np.real(F[omega_idx]) / N
        s_arr[i] = np.imag(F[omega_idx]) / N
    return g_arr, s_arr


def departure_score(g, s):
    """
    Per-k-bin distance from the phasor semicircle.

    d(k) = |g^2 + (s - 0.5)^2 - 0.25|

    d = 0 : dynamics exactly described by single-rate exponential
    d > 0 : mixture of processes or non-Markovian dynamics
    """
    return np.abs(g**2 + (s - 0.5)**2 - 0.25)

# %% [markdown]
# ## Cell 4 — Run pipeline

# %%
print("Detrending...")
delta_phi = detrend(phi)

print("\nComputing relaxation modes...")
k_bins, DELTA_PHI_k, Sk = compute_relaxation_modes(delta_phi)

print("\nAssigning windows...")
window_masks = assign_windows(k_bins)

print("\nComputing phasors...")
g, s = compute_phasors(DELTA_PHI_k)

print("\nComputing departure scores...")
d = departure_score(g, s)

print("\nDone.")

# %% [markdown]
# ## Cell 5 — Phasor plot
#
# Each dot is one spatial scale (k-bin).
# **Color** = structural size window (see legend).
# **Size** ∝ S(k), the time-mean signal power at that scale —
# larger dots are scales where the phase signal is stronger.
# The dashed semicircle is the locus of dynamics adequately described
# by a single exponential relaxation rate. Points inside or outside
# the semicircle indicate that scale's dynamics require a more complex
# model than simple two-component transport.

# %%
theta = np.linspace(0, np.pi, 300)
sc_g = 0.5 + 0.5 * np.cos(theta)
sc_s = 0.5 * np.sin(theta)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(sc_g, sc_s, 'k--', lw=1.2, alpha=0.5, label='Semicircle')
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 0.7)
ax.set_xlabel('g', fontsize=12)
ax.set_ylabel('s', fontsize=12)
ax.set_title('IKA phasor — GBM cell (n=1, preliminary)\n'
             'dot size ∝ S(k) signal power', fontsize=11)
ax.axhline(0, color='grey', lw=0.5, alpha=0.4)
ax.axvline(0, color='grey', lw=0.5, alpha=0.4)

s_min, s_max = Sk.min(), Sk.max()
dot_sizes = 8 + 80 * (Sk - s_min) / (s_max - s_min + 1e-10)

k_colors = np.array(['lightgrey'] * len(k_bins), dtype=object)
for wname, wmask in window_masks.items():
    k_colors[wmask] = WINDOW_COLORS[wname]

ax.scatter(g, s, s=dot_sizes, c=k_colors,
           alpha=0.75, edgecolors='k', linewidths=0.3, zorder=3)

patches = [mpatches.Patch(color=WINDOW_COLORS[w],
                           label=w.replace('_', ' '))
           for w in WINDOWS]
ax.legend(handles=patches, fontsize=9, loc='upper right', framealpha=0.8)

plt.tight_layout()
plt.savefig('IKA_phasor_GBM.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: IKA_phasor_GBM.png")
