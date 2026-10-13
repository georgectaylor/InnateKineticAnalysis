# -------------------------------------------------------------------
# IKA — Innate Kinetic Analysis
# Walkthrough notebook: phasor analysis of QPI data
# George Taylor, 2026
#
# To convert to Jupyter notebook:
#   pip install jupytext
#   jupytext --to notebook IKA_walkthrough.py
#
# Or paste each cell block into a new Jupyter notebook manually.
# Cells are delimited by # %% markers.
#
# NOTE: Plots in this notebook were generated on GBM data not
# included in this repository. Cells marked [SYNTHETIC] reproduce
# independently without any external data files.
# -------------------------------------------------------------------

# %% [markdown]
# # IKA — Innate Kinetic Analysis
# ### Phasor-DPS pipeline for quantitative phase imaging data
#
# This notebook demonstrates the core IKA pipeline on two datasets:
# 1. **Synthetic reconstituted cytoskeletal composites** [SYNTHETIC] —
#    generated with known transport parameters to validate that the
#    pipeline recovers expected phasor behavior at each spatial scale.
# 2. **Real GBM cell (QPI/SLIM)** — preliminary n=1 proof-of-concept
#    on a single glioblastoma cell imaged by SLIM.
#
# The pipeline asks: *at each spatial scale, does mass transport in
# this sample follow a simple two-component (diffusive + advective)
# model?* Departure from the phasor semicircle quantifies model
# inadequacy per scale, providing a physics-grounded fingerprint of
# cytoskeletal dynamics.
#
# ---
# **Goals for future development**
# - Independent Python implementation of core functions (currently
#   AI-assisted from pseudocode)
# - Unit tests against synthetic ground truth with known D and v
# - Packaging for pip-installable distribution
# - Temporal window analysis: track how phasor position evolves over
#   time within a session
# - Multi-channel extension: overlay fluorescence-channel phasors
#   alongside QPI dry-mass phasors
# - Trajectory embedding via PHATE/UMAP once temporal windows are
#   implemented (t-SNE here is illustrative only; n=4 is too small
#   for statistical interpretation)

# %% [markdown]
# ## Cell 1 — Imports and constants

# %%
import os
import re
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import fft2, fft, fftfreq
from scipy.signal.windows import tukey
from sklearn.manifold import TSNE
import tifffile

# --- Acquisition parameters ---
# Verify both values against your microscope metadata before running.
PIXEL_SIZE     = 0.1625   # um/pixel  (SLIM, 40x objective)
FRAME_INTERVAL = 1200     # seconds between frames (1 frame per 20 min)

# --- Spatial frequency windows (rad/um) ---
# Grounded in known size ranges of GBM cell structures:
#   Window A  ~1–10 um  : membrane features, large organelles
#   Window B  ~0.2–2 um : mitochondria, lysosomes
#   Window C  <500 nm   : fine structure, debris
WINDOWS = {
    'A_membrane':  (0.6,  6.0),
    'B_organelle': (6.0,  18.0),
    'C_fine':      (12.0, 21.0),
}
WINDOW_COLORS = {
    'A_membrane':  'steelblue',
    'B_organelle': 'darkorange',
    'C_fine':      'mediumseagreen',
}

# --- Synthetic data dimensions ---
# Kept smaller than real data (256 vs 2048) for fast generation.
# Frame count matches the real GBM session (103 frames).
SYN_ROWS, SYN_COLS, SYN_FRAMES = 256, 256, 103

# %% [markdown]
# ## Cell 2 — Data loading and synthetic generation

# %%
# --- Load real GBM data ---
def load_slim_folder(folder_path):
    """
    Load a SLIM time-lapse from a folder of per-frame TIFF files.

    Files are sorted by the integer index after '_i' in the filename
    (e.g. '…_i007_…tif') rather than alphabetically, which would
    mis-order frame 10 before frame 2.

    Returns phi: ndarray of shape (rows, cols, n_frames), float64.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    def get_i_index(filename):
        match = re.search(r'_i(\d+)', filename)
        return int(match.group(1)) if match else 0

    files_sorted = sorted(files, key=get_i_index)
    frames = [tifffile.imread(os.path.join(folder_path, f))
              for f in files_sorted]
    phi = np.stack(frames, axis=-1).astype(float)
    print(f"Loaded GBM stack: {phi.shape}  (rows, cols, frames)")
    return phi


# --- Generate synthetic stack with scale-specific transport ---
def generate_synthetic_stack(n_rows, n_cols, n_frames,
                              scale_params, pixel_size,
                              frame_interval, noise_level=0.01,
                              seed=42):
    """
    Generate a synthetic QPI phase stack with known transport properties
    per spatial scale window.

    Each spatial Fourier mode k is assigned a decay rate
        Gamma(k) = D*k^2 + v*k
    drawn from scale_params, so the pipeline's recovered phasor
    positions are ground-truth verifiable.

    Parameters
    ----------
    scale_params : list of dict, each with keys
        'k_min', 'k_max' (rad/um), 'D' (um^2/s), 'v' (um/s)
    pixel_size   : float, um/pixel
    frame_interval : float, seconds between frames
    noise_level  : float, amplitude of white noise added each frame
    seed         : int, random seed for reproducibility

    Returns
    -------
    phi_stack : ndarray (n_rows, n_cols, n_frames)
    """
    rng = np.random.default_rng(seed)

    kx = fftfreq(n_cols, d=pixel_size) * 2 * np.pi
    ky = fftfreq(n_rows, d=pixel_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_MAG = np.sqrt(KX**2 + KY**2)
    K_MAG[0, 0] = 1e-10   # avoid division by zero at DC

    # Build per-pixel decay-rate map from scale_params
    GAMMA = np.zeros_like(K_MAG)
    for p in scale_params:
        mask = (K_MAG >= p['k_min']) & (K_MAG < p['k_max'])
        GAMMA[mask] = p['D'] * K_MAG[mask]**2 + p['v'] * K_MAG[mask]
    # Default for k outside all specified ranges
    GAMMA[GAMMA == 0] = 0.1 * K_MAG[GAMMA == 0]**2

    # Random initial complex Fourier field
    F0 = (rng.standard_normal((n_rows, n_cols)) +
          1j * rng.standard_normal((n_rows, n_cols)))

    phi_stack = np.zeros((n_rows, n_cols, n_frames))
    for t in range(n_frames):
        dt = t * frame_interval
        decay = np.exp(-GAMMA * dt)
        noise = noise_level * (rng.standard_normal((n_rows, n_cols)) +
                               1j * rng.standard_normal((n_rows, n_cols)))
        F_t = F0 * decay + noise
        phi_stack[:, :, t] = np.real(np.fft.ifft2(F_t))

    return phi_stack


# Load real GBM data
GBM_FOLDER = 'onesingleGBMcell'   # adjust path if needed
phi_gbm = load_slim_folder(GBM_FOLDER)

# --- Synthetic condition 1: purely diffusive at all scales ---
# Represents unpolymerized monomer solution; no directed transport.
# All windows should land near the diffusive end of the semicircle.
phi_diffusive = generate_synthetic_stack(
    SYN_ROWS, SYN_COLS, SYN_FRAMES,
    scale_params=[
        {'k_min': 0.6,  'k_max': 6.0,  'D': 0.1, 'v': 0.0},
        {'k_min': 6.0,  'k_max': 18.0, 'D': 0.5, 'v': 0.0},
        {'k_min': 12.0, 'k_max': 21.0, 'D': 1.0, 'v': 0.0},
    ],
    pixel_size=PIXEL_SIZE, frame_interval=FRAME_INTERVAL, seed=1)
print(f"Generated diffusive stack:  {phi_diffusive.shape}")

# --- Synthetic condition 2: scale-selective mixed dynamics ---
# Represents actin/microtubule composite with motor activity.
# Window A (large scale) advective; Window B mixed; Window C diffusive.
# This is the most biologically relevant synthetic condition.
phi_composite = generate_synthetic_stack(
    SYN_ROWS, SYN_COLS, SYN_FRAMES,
    scale_params=[
        {'k_min': 0.6,  'k_max': 6.0,  'D': 0.1, 'v': 0.8},
        {'k_min': 6.0,  'k_max': 18.0, 'D': 0.3, 'v': 0.3},
        {'k_min': 12.0, 'k_max': 21.0, 'D': 1.0, 'v': 0.0},
    ],
    pixel_size=PIXEL_SIZE, frame_interval=FRAME_INTERVAL, seed=2)
print(f"Generated composite stack:  {phi_composite.shape}")

# --- Synthetic condition 3: Window A arrested ---
# Large-scale dynamics suppressed (very low D, v=0);
# other windows unchanged from composite condition.
# Demonstrates scale-selective loss of dynamics without assuming
# a specific drug mechanism.
phi_arrested = generate_synthetic_stack(
    SYN_ROWS, SYN_COLS, SYN_FRAMES,
    scale_params=[
        {'k_min': 0.6,  'k_max': 6.0,  'D': 0.001, 'v': 0.0},
        {'k_min': 6.0,  'k_max': 18.0, 'D': 0.3,   'v': 0.3},
        {'k_min': 12.0, 'k_max': 21.0, 'D': 1.0,   'v': 0.0},
    ],
    pixel_size=PIXEL_SIZE, frame_interval=FRAME_INTERVAL, seed=3)
print(f"Generated arrested stack:   {phi_arrested.shape}")

# %% [markdown]
# ## Cell 3 — Pipeline function definitions
#
# All functions are defined silently here and called in Cell 4.
#
# **Pipeline steps:**
# 1. `detrend` — remove per-pixel time mean (eliminates static structure;
#    prevents a DC spike in the temporal FFT)
# 2. `compute_relaxation_modes` — spatial FFT per frame, magnitude-averaged
#    per k-bin → DELTA_PHI_k(k, t)
# 3. `assign_windows` — label each k-bin with its biological size window
# 4. `compute_phasors` — temporal FFT of each k-bin's time series →
#    g and s coordinates on the phasor semicircle
# 5. `departure_score` — Euclidean distance from the semicircle per k-bin
#
# **Implementation note — magnitude before radial averaging:**
# A naive implementation takes the complex mean of all FFT coefficients
# within a k-bin. In large fields (2048×2048 pixels), the positive and
# negative imaginary parts cancel nearly completely, producing DELTA_PHI_k
# values ~1e-17 and a normalization collapse. Taking |FFT| before averaging
# preserves the real temporal variation (verified: std/mean ≈ 0.125 on GBM
# data vs. ~1e-14 with complex averaging). This is a non-obvious but
# critical implementation detail for large-format QPI data.

# %%
def detrend(phi):
    """Remove per-pixel time mean from phase stack."""
    return phi - np.mean(phi, axis=2, keepdims=True)


def compute_relaxation_modes(delta_phi, pixel_size=PIXEL_SIZE,
                             n_bins=50, tukey_alpha=0.5):
    """
    Compute radially averaged spatial power per k-bin per frame.

    Uses a 2D Tukey window before spatial FFT to reduce edge leakage,
    and takes |FFT| magnitude before radial averaging to prevent
    complex cancellation in large fields.

    Returns
    -------
    k_bins       : 1D array, bin-center spatial frequencies (rad/um)
    DELTA_PHI_k  : 2D array (n_valid_bins, n_frames), real-valued
    Sk           : 1D array, time-mean power per k-bin (detectability)
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
            print(f"  FFT frame {t}/{n_frames}")
        frame = delta_phi[:, :, t] * window_2d
        F = fft2(frame)
        F_mag = np.abs(F)   # ← magnitude before averaging
        for b in range(n_bins):
            mask = (K_MAG >= bin_edges[b]) & (K_MAG < bin_edges[b + 1])
            if np.any(mask):
                DELTA_PHI_k_all[b, t] = np.mean(F_mag[mask])
                bin_counts[b] = np.sum(mask)

    valid = bin_counts > 0
    DELTA_PHI_k = DELTA_PHI_k_all[valid]
    k_bins = k_bins_all[valid]
    Sk = np.mean(DELTA_PHI_k**2, axis=1)
    print(f"  Done in {time.time()-t0:.1f}s — "
          f"k range: {k_bins[0]:.2f}–{k_bins[-1]:.2f} rad/um, "
          f"{len(k_bins)} bins")
    return k_bins, DELTA_PHI_k, Sk


def assign_windows(k_bins, windows=WINDOWS):
    """Return a dict of boolean masks mapping each k-bin to a window."""
    return {name: (k_bins >= lo) & (k_bins < hi)
            for name, (lo, hi) in windows.items()}


def compute_phasors(DELTA_PHI_k, frame_interval=FRAME_INTERVAL):
    """
    Compute (g, s) phasor coordinates per k-bin.

    The probe frequency is the fundamental: omega = 2*pi / T_total.
    Normalization uses the sum of positive-frequency amplitudes so that
    (g, s) are dimensionless and the semicircle criterion g^2+(s-0.5)^2=0.25
    has a fixed reference independent of signal amplitude.

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
        pos_amps = np.abs(F[1:n_frames // 2])
        N = np.sum(pos_amps)
        if N == 0:
            continue
        g_arr[i] = np.real(F[omega_idx]) / N
        s_arr[i] = np.imag(F[omega_idx]) / N

    return g_arr, s_arr


def departure_score(g, s):
    """
    Per-k-bin distance from the phasor semicircle.

    d(k) = |g^2 + (s - 0.5)^2 - 0.25|

    d = 0  : dynamics exactly described by single-rate exponential
    d > 0  : mixture of processes, or non-Markovian dynamics
    """
    return np.abs(g**2 + (s - 0.5)**2 - 0.25)

# %% [markdown]
# ## Cell 4 — Run pipeline on all datasets

# %%
datasets = {
    'Diffusive (synthetic)':  phi_diffusive,
    'Composite (synthetic)':  phi_composite,
    'Arrested (synthetic)':   phi_arrested,
    'GBM cell (real, n=1)':   phi_gbm,
}

results = {}
for name, phi in datasets.items():
    print(f"\n=== {name} ===")
    delta_phi = detrend(phi)
    k_bins, DELTA_PHI_k, Sk = compute_relaxation_modes(delta_phi)
    window_masks = assign_windows(k_bins)
    g, s = compute_phasors(DELTA_PHI_k)
    d = departure_score(g, s)
    results[name] = dict(k_bins=k_bins, Sk=Sk,
                         g=g, s=s, d=d,
                         window_masks=window_masks)

# Stack departure score vectors for t-SNE
# Trim to shortest k_bins length (real and synthetic may differ slightly)
min_bins = min(results[n]['d'].shape[0] for n in datasets)
d_matrix = np.stack([results[n]['d'][:min_bins] for n in datasets], axis=0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tsne = TSNE(n_components=2, perplexity=2, random_state=42, n_iter=2000)
    embedding = tsne.fit_transform(d_matrix)

# %% [markdown]
# ## Cell 5 — Plots
#
# **Figure 1:** Phasor plots for all four conditions.
# Dot color = spatial scale window; dot size ∝ S(k) (signal power).
# Dashed semicircle = locus of single-rate exponential relaxation.
# Points inside/outside the semicircle indicate model inadequacy.
#
# **Figure 2:** t-SNE of departure score vectors d(k).
# *Note: with n=4 conditions this is illustrative only — t-SNE requires
# substantially more points for statistical interpretation. Included
# here to demonstrate the embedding concept; meaningful results require
# multiple replicates and/or drug conditions.*

# %%
condition_names = list(datasets.keys())
condition_colors = ['#4477AA', '#66CCEE', '#CCBB44', '#EE6677']

# Semicircle reference curve
theta = np.linspace(0, np.pi, 300)
sc_g = 0.5 + 0.5 * np.cos(theta)
sc_s = 0.5 * np.sin(theta)

# --- Figure 1: Phasor plots ---
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(sc_g, sc_s, 'k--', lw=1.2, alpha=0.5, label='Semicircle')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 0.7)
    ax.set_xlabel('g', fontsize=11)
    ax.set_ylabel('s', fontsize=11)
    ax.set_title(name, fontsize=11)
    ax.axhline(0, color='grey', lw=0.5, alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, alpha=0.4)

    g, s, Sk = res['g'], res['s'], res['Sk']
    s_min, s_max = Sk.min(), Sk.max()
    dot_sizes = 8 + 80 * (Sk - s_min) / (s_max - s_min + 1e-10)

    k_colors = np.array(['lightgrey'] * len(res['k_bins']), dtype=object)
    for wname, wmask in res['window_masks'].items():
        k_colors[wmask] = WINDOW_COLORS[wname]

    ax.scatter(g, s, s=dot_sizes, c=k_colors,
               alpha=0.75, edgecolors='k', linewidths=0.2, zorder=3)

    patches = [mpatches.Patch(color=WINDOW_COLORS[w],
                               label=w.replace('_', ' '))
               for w in WINDOWS]
    ax.legend(handles=patches, fontsize=7, loc='upper right',
              framealpha=0.7)

fig1.suptitle('IKA phasor plots — dot size ∝ S(k) signal power',
              fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('IKA_phasors.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: IKA_phasors.png")

# --- Figure 2: t-SNE embedding ---
fig2, ax2 = plt.subplots(figsize=(6, 5))

for i, (name, color) in enumerate(zip(condition_names, condition_colors)):
    ax2.scatter(embedding[i, 0], embedding[i, 1],
                color=color, s=140, zorder=3, edgecolors='k',
                linewidths=0.5)
    ax2.annotate(name,
                 (embedding[i, 0], embedding[i, 1]),
                 textcoords='offset points',
                 xytext=(8, 4), fontsize=8)

ax2.set_title('t-SNE of departure score vectors d(k)\n'
              '(n=4 — illustrative only; '
              'not statistically interpretable at this n)',
              fontsize=9)
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('IKA_tsne.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: IKA_tsne.png")
