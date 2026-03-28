from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import os

out_data = r"E:\outputTVB"
out_figs = r"E:\GSOC\figures"
os.makedirs(out_data, exist_ok=True)
os.makedirs(out_figs, exist_ok=True)

# ── STEP 1: Run TVB simulation (Raw + BOLD) ──────────────────────────────────
print("Step 1: Running TVB simulation...")

model = models.Generic2dOscillator()
model.a = np.array([0.5])
model.b = np.array([-10.0])
model.c = np.array([0.0])
model.d = np.array([0.02])
model.I = np.array([0.0])
model.tau = np.array([1.0])

conn = connectivity.Connectivity.from_file()
conn.speed = np.array([4.0])

coupl = coupling.Linear(a=np.array([0.0042]))

integrator = integrators.HeunStochastic(dt=0.1)
integrator.noise.nsig = np.array([0.01])

mon_raw = monitors.Raw()
mon_raw.period = 2000.0

mon_bold = monitors.Bold()
mon_bold.period = 2000.0

sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupl,
    integrator=integrator,
    monitors=[mon_raw, mon_bold],
    simulation_length=100000.0   # 100s — enough timepoints, small footprint
)
sim.configure()

print("  Simulating...")
results = sim.run()
(time_raw, data_raw) = results[0]
(time_bold, data_bold) = results[1]

neural = data_raw[:, 0, :, 0].T    # (n_regions, T)
bold_canonical = data_bold[:, 0, :, 0].T  # (n_regions, T)
n_regions = neural.shape[0]

print(f"  Neural shape: {neural.shape}, std: {neural.std():.4f}")
print(f"  BOLD shape: {bold_canonical.shape}, std: {bold_canonical.std():.4f}")

np.save(os.path.join(out_data, "neural_activity.npy"), neural)
np.save(os.path.join(out_data, "bold_canonical.npy"), bold_canonical)
print("  Saved neural and canonical BOLD.")

# ── STEP 2: Load and prepare HRFs ────────────────────────────────────────────
print("\nStep 2: Preparing HRFs...")

hrfs = np.load(os.path.join(out_data, "hrfs_76.npy"))  # (76, 15)
hrfs = hrfs[:n_regions, :]
print(f"  HRF shape: {hrfs.shape}, peak at index {np.argmax(hrfs[0])} = {np.argmax(hrfs[0])*2}s")

# ── STEP 3: Empirical HRF convolution ────────────────────────────────────────
print("\nStep 3: Convolving neural activity with empirical HRFs...")

T = neural.shape[1]
bold_empirical = np.zeros((n_regions, T))

for i in range(n_regions):
    conv = fftconvolve(neural[i], hrfs[i], mode='full')
    bold_empirical[i] = conv[:T]

# Normalize
def normalize(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)

bold_empirical = normalize(bold_empirical)
bold_canonical_n = normalize(bold_canonical)

np.save(os.path.join(out_data, "bold_empirical.npy"), bold_empirical)
print(f"  Empirical BOLD shape: {bold_empirical.shape}")

# ── STEP 4: FC matrices ───────────────────────────────────────────────────────
print("\nStep 4: Computing FC matrices...")

FC_canonical = np.corrcoef(bold_canonical_n)
FC_empirical = np.corrcoef(bold_empirical)
FC_diff = FC_empirical - FC_canonical

np.save(os.path.join(out_data, "FC_canonical.npy"), FC_canonical)
np.save(os.path.join(out_data, "FC_empirical.npy"), FC_empirical)

rmse = np.sqrt(np.mean(FC_diff**2))
corr = np.corrcoef(FC_canonical.flatten(), FC_empirical.flatten())[0, 1]
print(f"  RMSE: {rmse:.4f}")
print(f"  FC correlation: {corr:.4f}")

# ── STEP 5: Generate all figures ──────────────────────────────────────────────
print("\nStep 5: Generating figures...")

# fig7 — BOLD timeseries
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(5):
    ax.plot(bold_canonical[i], label=f'Region {i+1}', alpha=0.8)
ax.set_title('TVB Canonical BOLD Timeseries (First 5 Regions)')
ax.set_xlabel('Timepoint')
ax.set_ylabel('BOLD Signal')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_figs, "fig7_tvb_bold_timeseries.png"), dpi=150)
plt.close()
print("  fig7 saved")

# fig8 — FC Canonical
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(FC_canonical, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title('FC Matrix — Canonical HRF (TVB)')
ax.set_xlabel('Region')
ax.set_ylabel('Region')
plt.tight_layout()
plt.savefig(os.path.join(out_figs, "fig8_fc_canonical.png"), dpi=150)
plt.close()
print("  fig8 saved")

# fig9 — FC Empirical
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(FC_empirical, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title('FC Matrix — Empirical rsHRF (TVB)')
ax.set_xlabel('Region')
ax.set_ylabel('Region')
plt.tight_layout()
plt.savefig(os.path.join(out_figs, "fig9_fc_empirical.png"), dpi=150)
plt.close()
print("  fig9 saved")

# fig10 — FC comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, mat, title in zip(axes,
    [FC_canonical, FC_empirical, FC_diff],
    ['Canonical HRF', 'Empirical rsHRF', 'Difference (Empirical - Canonical)']):
    im = ax.imshow(mat, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title(title)
    ax.set_xlabel('Region')
    ax.set_ylabel('Region')
plt.tight_layout()
plt.savefig(os.path.join(out_figs, "fig10_fc_comparison.png"), dpi=150)
plt.close()
print("  fig10 saved")

print(f"\n All done!")
print(f"   RMSE: {rmse:.4f}")
print(f"   FC correlation: {corr:.4f}")
print(f"   Figures saved to: {out_figs}")
print(f"   Data saved to: {out_data}")