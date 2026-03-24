# Personalized HRF × The Virtual Brain

*Pre-GSoC prototype · INCF GSoC 2026 Project #27*

The hemodynamic response function (HRF) is the brain's vascular "blur" — the lag between a neuron firing and the BOLD signal peaking in fMRI. Every computational model of brain activity, including The Virtual Brain, assumes this blur is identical across regions and subjects. It isn't. This repository builds the pipeline to estimate a subject-specific, region-specific HRF from resting/naturalistic fMRI using `rsHRF`, inject it into TVB's BOLD monitor in place of the standard Gamma Kernel, and measure whether the personalized simulation better reproduces empirical functional connectivity.

---

## The Core Idea

```
Standard TVB pipeline:
  Neural activity (simulated) → Gamma Kernel HRF (fixed, canonical) → Predicted BOLD

This project:
  Neural activity (simulated) → Empirical HRF per region (from rsHRF) → Predicted BOLD
```

The delta is one targeted swap in TVB's BOLD monitor. The question is whether that swap produces a measurable difference in simulated FC — and whether the personalized version is closer to empirical FC.

---

## Dataset

**OpenNeuro ds000228** — 155 subjects watched Pixar short films in an fMRI scanner (`task-pixar`). fMRIPrep-style preprocessing applied (SPM pipeline: `swrf` prefix = smoothed, warped to MNI, realigned). TR = 2.0s.

Per subject, two files are used:
- `sub-pixarXXX_task-pixar_run-001_swrf_bold.nii.gz` — preprocessed 4D BOLD (~204k voxels × ~168 timepoints)
- `sub-pixarXXX_analysis_mask.nii.gz` — binary brain mask

---

## Phase 1 — HRF Estimation with rsHRF

rsHRF runs on the preprocessed BOLD using a **point process model**: it detects spontaneous BOLD peaks, treats them as pseudo-events with no assumed shape, and fits an HRF to each voxel's timeseries independently. No task, no known onsets required.

**Docker command used:**
```bash
docker run -ti --rm \
  -v /path/to/sub-pixar001:/data:ro \
  -v /path/to/output:/output \
  bids/rshrf \
  --input_file /data/sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz \
  --atlas /data/sub-pixar001_analysis_mask.nii.gz \
  --estimation canon2dd \
  --output_dir /output \
  -TR 2.0
```

**Estimation method:** `canon2dd` — fits the HRF as a weighted combination of the canonical SPM HRF + its time derivative + dispersion derivative, giving each voxel flexibility to capture its own timing and width.

**Output files:**

| File | Contents |
|------|----------|
| `_hrf.mat` | Core output — HRF timeseries per voxel, shape `(37, 204275)` |
| `_height.nii.gz` | Peak amplitude map (voxelwise) |
| `_T2P.nii.gz` | Time-to-peak map (voxelwise) |
| `_FWHM.nii.gz` | Full width at half maximum map (voxelwise) |
| `_eventnumber.nii.gz` | Number of detected spontaneous events per voxel |
| `_deconv.nii.gz` | Neural activity timeseries after HRF deconvolution |
| `_hrf_plot.png` | Mean HRF shape across brain |
| `_deconvolution_plot.png` | BOLD vs deconvolved signal with detected events |

**Key finding for sub-pixar001:**
- Estimated HRF peaks at **~8–9 seconds** (canonical SPM HRF peaks at ~5–6s — this subject is hemodynamically slower than average)
- Clean post-undershoot at ~18–20s
- Physiologically realistic shape — rsHRF is working correctly

> 📊 **Figure placeholder: `figures/fig1_hrf_shape.png`**
> *Mean estimated HRF for sub-pixar001 (the `_hrf_plot.png` output from rsHRF). Shows time-to-peak, undershoot, and return to baseline. Compare against the canonical SPM HRF overlaid.*

> 📊 **Figure placeholder: `figures/fig2_deconvolution.png`**
> *Single-voxel deconvolution plot (`_deconvolution_plot.png`). Blue = raw BOLD, red = deconvolved neural signal, black arrows = detected spontaneous events.*

> 📊 **Figure placeholder: `figures/fig3_T2P_map.png`**
> *Whole-brain time-to-peak map (`_T2P.nii.gz` rendered as a brain slice). Shows spatial heterogeneity of HRF timing across cortex — the variation this project is trying to exploit.*

---

## Phase 2 — Baseline TVB Simulation

The Virtual Brain (EBRAINS cloud instance) was configured with:

| Parameter | Value |
|-----------|-------|
| Connectivity | Hagmann 66-region human connectome |
| Neural mass model | Generic 2D Oscillator |
| Integrator | Heun Deterministic, dt=0.5ms |
| BOLD monitor | Gamma Kernel HRF (TVB default) |
| Simulation length | 20,000 ms |

This is the **legacy approach** — one canonical HRF kernel applied uniformly to all 66 regions for all subjects. Output: `bold_*.h5` and `raw_*.h5` (HDF5 format).

> 📊 **Figure placeholder: `figures/fig4_tvb_baseline_fc.png`**
> *Functional connectivity matrix (66×66) computed from the baseline TVB simulation BOLD output. This is the standard model's prediction of FC.*

---

## Phase 3 — Regional HRF Extraction *(in progress)*

The `hrfa` matrix from rsHRF (`shape: 37 × 204,275`) needs to be collapsed from voxel space to region space for use in TVB. Pipeline:

1. Load `hrfa` from `.mat` file with `scipy.io`
2. Load AAL atlas (MNI space, fetched via `nilearn.datasets.fetch_atlas_aal()`)
3. Resample atlas to match BOLD voxel grid with `nilearn.image.resample_to_img`
4. For each atlas region: extract all voxels belonging to that region, average their HRF timeseries
5. Output: `regional_hrfs.npy` — shape `(n_regions × 37)`

> 📊 **Figure placeholder: `figures/fig5_regional_hrfs.png`**
> *Grid of HRF curves — one per brain region. Shows the spread of HRF shapes across regions: some faster, some slower, some with larger undershoot. This is the variation the standard model ignores.*

---

## Phase 4 — Custom TVB BOLD Monitor *(in progress)*

TVB's BOLD monitor (`tvb.simulator.monitors.Bold`) uses an `equation` object as its HRF kernel — the same kernel for every region. The modification:

```python
class EmpiricalHRFMonitor(Bold):
    """
    Drop-in replacement for TVB's Bold monitor.
    Uses subject-specific, region-specific HRFs estimated by rsHRF
    instead of a fixed canonical kernel.
    """
    def __init__(self, regional_hrfs, **kwargs):
        super().__init__(**kwargs)
        self.regional_hrfs = regional_hrfs  # shape: (n_regions, hrf_length)

    def sample(self, step, state):
        result = np.zeros((self.n_regions,))
        for r in range(self.n_regions):
            neural = state[:, r]
            hrf = self.regional_hrfs[r]
            result[r] = np.convolve(neural, hrf, mode='full')[step]
        return result
```

---

## Phase 5 — Comparison *(in progress)*

Two simulations, identical neural model, identical connectivity, one difference: the HRF.

```
Simulation A: TVB default Gamma Kernel → FC_standard (66×66)
Simulation B: rsHRF empirical HRFs     → FC_empirical (66×66)
```

Comparison metrics:
- Visual difference of FC matrices
- `r = np.corrcoef(FC_A.ravel(), FC_B.ravel())` — how much does HRF choice change FC?
- Correlation of each simulated FC against empirical FC from ds000228 BOLD data — which is closer to ground truth?

> 📊 **Figure placeholder: `figures/fig6_fc_comparison.png`**
> *Side-by-side: FC_standard vs FC_empirical as 66×66 correlation matrices. The visual difference — if present — is the PoC result.*

> 📊 **Figure placeholder: `figures/fig7_model_fit.png`**
> *Scatter plot: simulated FC (standard) vs empirical FC on x-axis, simulated FC (personalized HRF) vs empirical FC on y-axis. Higher r = better model. This is the headline number.*

---

## Figures to Generate After PoC Runs

| Placeholder | What to do | Source file |
|-------------|------------|-------------|
| `fig1_hrf_shape.png` | Screenshot / save the rsHRF output plot, overlay canonical HRF | `_hrf_plot.png` + matplotlib |
| `fig2_deconvolution.png` | Screenshot / save the deconvolution plot | `_deconvolution_plot.png` |
| `fig3_T2P_map.png` | Load `_T2P.nii.gz`, plot axial slice with nilearn `plot_stat_map` | `_T2P.nii.gz` |
| `fig4_tvb_baseline_fc.png` | Load `bold_*.h5`, compute FC, plot with `plt.imshow` | TVB H5 output |
| `fig5_regional_hrfs.png` | Plot `regional_hrfs.npy` as grid of curves | Phase 3 output |
| `fig6_fc_comparison.png` | Side-by-side imshow of FC_standard vs FC_empirical | Phase 5 output |
| `fig7_model_fit.png` | Scatter of simulated vs empirical FC for both conditions | Phase 5 output |

---

## Running It

```bash
# 1. Pull rsHRF Docker image
docker pull bids/rshrf

# 2. Run HRF estimation on one subject
docker run -ti --rm \
  -v /path/to/subject:/data:ro \
  -v /path/to/output:/output \
  bids/rshrf \
  --input_file /data/sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz \
  --atlas /data/sub-pixar001_analysis_mask.nii.gz \
  --estimation canon2dd \
  --output_dir /output \
  -TR 2.0

# 3. Install Python dependencies
pip install nibabel nilearn scipy numpy matplotlib tvb-library tvb-data

# 4. Extract regional HRFs (Phase 3 - in progress)
python extract_regional_hrfs.py

# 5. Run TVB comparison (Phase 4-5 - in progress)
python run_tvb_comparison.py
```

---

## Repository Structure *(target)*

```
GSOC_27/
  data/
    sub-pixar001_hrf.mat           ← rsHRF voxelwise HRF output
    regional_hrfs.npy              ← (n_regions × 37) averaged HRFs
    fc_standard.npy                ← FC from TVB default Gamma Kernel
    fc_empirical.npy               ← FC from TVB + rsHRF HRFs
    fc_bold_empirical.npy          ← FC from actual ds000228 BOLD
  figures/
  extract_regional_hrfs.py        ← mat → regional average HRFs
  run_tvb_comparison.py           ← run both simulations, compare FC
  requirements.txt
```

---

## What a Full GSoC Project Would Add

- Per-region HRF extraction across all 155 subjects (not just sub-pixar001)
- Modified `tvb.simulator.monitors.Bold` with `regional_hrfs` as a proper TVB datatype
- Balloon-Windkessel comparison alongside the HRF kernel approach
- Multi-subject FC comparison with statistics
- EBRAINS-ready notebook and Docker container
- Classical resting-state dataset validation alongside naturalistic viewing

---

## Current Status

| Phase | Status |
|-------|--------|
| rsHRF Docker setup | ✅ Done |
| Data acquisition (ds000228, 155 subjects) | ✅ Done |
| HRF estimation — sub-pixar001 | ✅ Done |
| Baseline TVB simulation | ✅ Done |
| Regional HRF extraction |  ✅ Done |
| Custom TVB BOLD monitor | 🔄 In progress |
| FC comparison | ⏳ Pending |

---

*OpenNeuro ds000228 · AAL atlas · rsHRF (Wu et al., Neuroimage 2021) · The Virtual Brain (EBRAINS) · Hagmann 66-region connectome*