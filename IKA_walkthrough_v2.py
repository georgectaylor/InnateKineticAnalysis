# -------------------------------------------------------------------
# IKA — Innate Kinetic Analysis
# Walkthrough notebook v2: phasor analysis + temporal t-SNE
# George Taylor, 2026
#
# To convert to Jupyter notebook:
#   pip install jupytext
#   jupytext --to notebook IKA_walkthrough_v2.py
#
# Or paste each cell block (between # %% markers) into a new
# Jupyter notebook manually. Blocks marked # %% [markdown] become
# markdown cells (Escape → M in Jupyter).
#
# NOTE: Plots were generated on GBM data not included in this
# repository. Cells marked [SYNTHETIC] reproduce independently.
# -------------------------------------------------------------------

# %% [markdown]
# # IKA — Innate Kinetic Analysis
# ### Phasor-DPS pipeline for quantitative phase imaging data
# George Taylor · October 2026
#
# This notebook demonstrates the core IKA pipeline on two datasets
# treated in parallel:
#
# 1. **Synthetic reconstituted cytoskeletal composite** [SYNTHETIC] —
#    a single simulated stack where each spatial scale window was
#    assigned known transport parameters (Window A advective, Window B
#    mixed, Window C diffusive). This is the direct cell-free analogue
#    of the planned reconstituted actin/microtubule experiment, generated
#    here as ground-truth validation.
#
# 2. **Real GBM cell (QPI/SLIM)** — preliminary n=1 proof-of-concept
#    on a single glioblastoma cell imaged by SLIM at 20-minute intervals.
#
# **Central question:** At each spatial scale, does mass transport follow
# a simple two-component (diffusive + advective) model? And do the scales
# move through dynamical state space together (coupled) or independently
# (decoupled/chaotic)?
#
# **Four output plots:**
# - Phasor plot, synthetic composite — ground truth scale separation
# - Phasor plot, GBM cell — real scale structure
# - t-SNE of windowed departure scores, synthetic — temporal validation
# - t-SNE of windowed departure scores, GBM — trajectory of real dynamics
#
# ---
# **Goals for future development**
# - Independent Python implementation of core functions
# - Unit tests against synthetic ground truth with known D and v
# - Packaging for pip-installable distribution
# - Replace t-SNE with PHATE to recover trajectory geometry
#   (loops = limit cycle, drift = attractor approach, scatter = chaos)
# - Multi-channel extension: fluorescence phasors alongside QPI dry mass
# - Temporal window analysis extended to multi-channel comparison

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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.fft import fft2, fft, fftfreq
from scipy.signal.windows import tukey
from sklearn.manifold import TSNE
import tifffile

# --- Acquisition parameters ---
# Verify both against microscope metadata before running.
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

# --- Temporal sliding window parameters ---
# Each window covers WINDOW_WIDTH frames; windows step by WINDOW_STEP.
# At 20-minute intervals, 10 frames = ~3.3 hours of real time.
WINDOW_WIDTH = 10   # frames per window
WINDOW_STEP  = 5    # frames between window starts

# --- Synthetic data dimensions ---
SYN_ROWS, SYN_COLS, SYN_FRAMES = 256, 256, 103

# %% [markdown]
# ## Cell 2 — Data loading and synthetic generation

# %%
def load_slim_folder(folder_path):
    """
    Load a SLIM time-lapse from a folder of per-frame TIFF files.

    Files are sorted by the integer index after '_i' in the filename
    rather than alphabetically (alphabetical order mis-sorts i10 before i2).

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
    print(f"Loaded GBM stack: {phi.shape}  (rows, cols, frames)")
    return phi


def generate_synthetic_stack(n_rows, n_cols, n_frames,
                              scale_params, pixel_size,
                              frame_interval, noise_level=0.01,
                              seed=42):
    """
    Generate a single synthetic QPI phase stack with scale-specific
    transport parameters assigned per spatial frequency window.

    Each Fourier mode k is assigned decay rate Gamma(k) = D*k^2 + v*k
    from whichever entry in scale_params contains that k value.
    This produces one composite dataset where different spatial scales
    have different known dynamics simultaneously — the direct synthetic
    analogue of a reconstituted cytoskeletal composite.

    Parameters
    ----------
    scale_params : list of dict, each with
        'k_min', 'k_max' (rad/um), 'D' (um^2/s), 'v' (um/s)
    """
    rng = np.random.default_rng(seed)

    kx = fftfreq(n_cols, d=pixel_size) * 2 * np.pi
    ky = fftfreq(n_rows, d=pixel_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_MAG = np.sqrt(KX**2 + KY**2)
    K_MAG[0, 0] = 1e-10

    GAMMA = np.zeros_like(K_MAG)
    for p in scale_params:
        mask = (K_MAG >= p['k_min']) & (K_MAG < p['k_max'])
        GAMMA[mask] = p['D'] * K_MAG[mask]**2 + p['v'] * K_MAG[mask]
    GAMMA[GAMMA == 0] = 0.1 * K_MAG[GAMMA == 0]**2

    F0 = (rng.standard_normal((n_rows, n_cols)) +
          1j * rng.standard_normal((n_rows, n_cols)))

    phi_stack = np.zeros((n_rows, n_cols, n_frames))
    for t in range(n_frames):
        dt = t * frame_interval
        decay = np.exp(-GAMMA * dt)
        noise = noise_level * (rng.standard_normal((n_rows, n_cols)) +
                               1j * rng.standard_normal((n_rows, n_cols)))
        phi_stack[:, :, t] = np.real(np.fft.ifft2(F0 * decay + noise))

    return phi_stack


# Load real GBM data
GBM_FOLDER = 'onesingleGBMcell'   # adjust path if needed
phi_gbm = load_slim_folder(GBM_FOLDER)

# Generate single synthetic composite stack:
# Window A (large scale): advective  — D low, v high
# Window B (organelle):   mixed      — D and v both moderate
# Window C (fine scale):  diffusive  — D high, v=0
# This is one dataset, not three — scales coexist within the same stack.
phi_synthetic = generate_synthetic_stack(
    SYN_ROWS, SYN_COLS, SYN_FRAMES,
    scale_params=[
        {'k_min': 0.6,  'k_max': 6.0,  'D': 0.1, 'v': 0.8},  # A: advective
        {'k_min': 6.0,  'k_max': 18.0, 'D': 0.3, 'v': 0.3},  # B: mixed
        {'k_min': 12.0, 'k_max': 21.0, 'D': 1.0, 'v': 0.0},  # C: diffusive
    ],
    pixel_size=PIXEL_SIZE, frame_interval=FRAME_INTERVAL, seed=42)
print(f"Generated synthetic composite: {phi_synthetic.shape}")

# %% [markdown]
# ## Cell 3 — Pipeline function definitions
#
# **Implementation note — magnitude before radial averaging:**
# Taking |FFT| before averaging across a k-bin is critical for large
# fields (2048×2048 pixels). Complex averaging causes near-complete
# cancellation of positive and negative imaginary parts, collapsing
# DELTA_PHI_k to ~1e-17 and causing normalization failure. Magnitude
# averaging preserves real temporal variation (verified std/mean ≈ 0.125
# on GBM data). This is a non-obvious but essential step for large-format
# QPI data.

# %%
def detrend(phi):
    """Remove per-pixel time mean. Eliminates static structure and
    prevents a DC spike dominating the temporal FFT."""
    return phi - np.mean(phi, axis=2, keepdims=True)


def compute_relaxation_modes(delta_phi, pixel_size=PIXEL_SIZE,
                             n_bins=50, tukey_alpha=0.5):
    """
    Radially averaged spatial power per k-bin per frame.

    Returns
    -------
    k_bins      : 1D array, bin-center spatial frequencies (rad/um)
    DELTA_PHI_k : 2D array (n_bins, n_frames), real-valued
    Sk          : 1D array, time-mean power per k-bin
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
        F_mag = np.abs(fft2(frame))   # magnitude before averaging
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
          f"k: {k_bins[0]:.2f}–{k_bins[-1]:.2f} rad/um, "
          f"{len(k_bins)} bins")
    return k_bins, DELTA_PHI_k, Sk


def assign_windows(k_bins, windows=WINDOWS):
    """Boolean mask per window for each k-bin."""
    return {name: (k_bins >= lo) & (k_bins < hi)
            for name, (lo, hi) in windows.items()}


def compute_phasors(DELTA_PHI_k, frame_interval=FRAME_INTERVAL):
    """
    (g, s) phasor coordinates per k-bin via temporal FFT.

    Probe frequency = fundamental (omega = 2*pi / T_total).
    Normalization by sum of positive-frequency amplitudes makes
    (g, s) dimensionless with fixed semicircle reference.
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
    Distance from the phasor semicircle per k-bin.
    d = 0: single-rate exponential relaxation (model adequate).
    d > 0: mixture of processes or non-Markovian dynamics.
    """
    return np.abs(g**2 + (s - 0.5)**2 - 0.25)


def windowed_departure_scores(DELTA_PHI_k, frame_interval=FRAME_INTERVAL,
                               width=WINDOW_WIDTH, step=WINDOW_STEP):
    """
    Compute departure score vectors over sliding temporal windows.

    For each window, compute phasors from that window's frames only,
    then compute departure scores. Returns one d(k) vector per window.

    Returns
    -------
    d_windows : 2D array (n_windows, n_k_bins)
    window_centers : 1D array, center frame index of each window
    """
    n_k, n_frames = DELTA_PHI_k.shape
    starts = range(0, n_frames - width + 1, step)
    d_windows = []
    window_centers = []

    for start in starts:
        segment = DELTA_PHI_k[:, start:start + width]
        g, s = compute_phasors(segment,
                               frame_interval=frame_interval)
        d_windows.append(departure_score(g, s))
        window_centers.append(start + width // 2)

    return np.array(d_windows), np.array(window_centers)

# %% [markdown]
# ## Cell 4 — Run pipeline on both datasets

# %%
datasets = {
    'Synthetic composite': phi_synthetic,
    'GBM cell':            phi_gbm,
}

results = {}
for name, phi in datasets.items():
    print(f"\n=== {name} ===")
    delta_phi = detrend(phi)
    k_bins, DELTA_PHI_k, Sk = compute_relaxation_modes(delta_phi)
    window_masks = assign_windows(k_bins)
    g, s = compute_phasors(DELTA_PHI_k)
    d = departure_score(g, s)
    d_windows, window_centers = windowed_departure_scores(DELTA_PHI_k)
    results[name] = dict(k_bins=k_bins, Sk=Sk,
                         g=g, s=s, d=d,
                         d_windows=d_windows,
                         window_centers=window_centers,
                         window_masks=window_masks)

# t-SNE per dataset on windowed departure score vectors
# Each point = one time window; trajectory shows how dynamics evolve
for name in datasets:
    res = results[name]
    n_windows = res['d_windows'].shape[0]
    print(f"\nt-SNE: {name} — {n_windows} windows")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perp = min(5, n_windows - 1)
        tsne = TSNE(n_components=2, perplexity=perp,
                    random_state=42, n_iter=2000)
        res['embedding'] = tsne.fit_transform(res['d_windows'])

# %% [markdown]
# ## Cell 5 — Plots
#
# **Phasor plots:** each dot is one k-bin, color = spatial scale window,
# size ∝ S(k) signal power. Dashed semicircle = locus of single-rate
# exponential relaxation. Points off the semicircle indicate model
# inadequacy at that scale.
#
# **t-SNE plots:** each dot is one sliding time window (~3.3 hours of
# real time at 20-minute frame intervals). Color progresses from light
# (early) to dark (late), tracing the trajectory through dynamical
# state space.
# - Tight cluster: stationary dynamics, same behavior throughout
# - Directional drift: cell evolving, approaching or leaving an attractor
# - Loop: periodic return to same dynamical state (limit cycle)
# - Scatter: history-dependent, potentially chaotic
#
# *Note: t-SNE preserves local clustering but not global distances or
# trajectory geometry. Future work will replace with PHATE, which
# explicitly recovers tree and loop structure in trajectories.*

# %%
# Semicircle reference
theta = np.linspace(0, np.pi, 300)
sc_g = 0.5 + 0.5 * np.cos(theta)
sc_s = 0.5 * np.sin(theta)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

dataset_names = list(datasets.keys())

for col, name in enumerate(dataset_names):
    res = results[name]

    # --- Phasor plot (top row) ---
    ax_p = axes[0, col]
    ax_p.plot(sc_g, sc_s, 'k--', lw=1.2, alpha=0.5)
    ax_p.set_xlim(-0.2, 1.2)
    ax_p.set_ylim(-0.2, 0.7)
    ax_p.set_xlabel('g', fontsize=11)
    ax_p.set_ylabel('s', fontsize=11)
    ax_p.set_title(f'Phasor — {name}', fontsize=11)
    ax_p.axhline(0, color='grey', lw=0.5, alpha=0.4)
    ax_p.axvline(0, color='grey', lw=0.5, alpha=0.4)

    g, s, Sk = res['g'], res['s'], res['Sk']
    s_min, s_max = Sk.min(), Sk.max()
    dot_sizes = 8 + 80 * (Sk - s_min) / (s_max - s_min + 1e-10)

    k_colors = np.array(['lightgrey'] * len(res['k_bins']), dtype=object)
    for wname, wmask in res['window_masks'].items():
        k_colors[wmask] = WINDOW_COLORS[wname]

    ax_p.scatter(g, s, s=dot_sizes, c=k_colors,
                 alpha=0.75, edgecolors='k', linewidths=0.2, zorder=3)

    patches = [mpatches.Patch(color=WINDOW_COLORS[w],
                               label=w.replace('_', ' '))
               for w in WINDOWS]
    ax_p.legend(handles=patches, fontsize=7, loc='upper right',
                framealpha=0.7)

    # --- t-SNE plot (bottom row) ---
    ax_t = axes[1, col]
    emb = res['embedding']
    wc  = res['window_centers']

    norm = Normalize(vmin=wc.min(), vmax=wc.max())
    cmap = plt.cm.viridis
    colors_tsne = cmap(norm(wc))

    # Draw trajectory lines between consecutive windows
    for i in range(len(wc) - 1):
        ax_t.plot(emb[i:i+2, 0], emb[i:i+2, 1],
                  color='lightgrey', lw=0.8, zorder=1)

    sc = ax_t.scatter(emb[:, 0], emb[:, 1],
                      c=wc, cmap='viridis', s=60,
                      edgecolors='k', linewidths=0.4, zorder=3)

    # Label first and last window
    ax_t.annotate('start', emb[0],  xytext=(5, 5),
                  textcoords='offset points', fontsize=7, color='navy')
    ax_t.annotate('end',   emb[-1], xytext=(5, 5),
                  textcoords='offset points', fontsize=7, color='darkred')

    plt.colorbar(sc, ax=ax_t, label='Window center (frame)')
    ax_t.set_xlabel('t-SNE 1', fontsize=10)
    ax_t.set_ylabel('t-SNE 2', fontsize=10)
    ax_t.set_title(f't-SNE trajectory — {name}\n'
                   f'({len(wc)} windows, {WINDOW_WIDTH} frames each '
                   f'≈ {WINDOW_WIDTH * FRAME_INTERVAL / 3600:.1f} hr)',
                   fontsize=10)

fig.suptitle('IKA — phasor structure and temporal dynamics\n'
             'dot size ∝ S(k) signal power  |  '
             't-SNE color: early (light) → late (dark)',
             fontsize=12, y=1.01)

plt.tight_layout()
plt.savefig('IKA_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: IKA_results.png")
