import numpy as np
from scipy.interpolate import interp1d

hrfs_full = np.load(r"E:\New folder\GSOC\output\regional_hrfs.npy")  # (155, 37)

# Current time axis — wrong, treated as 2000ms steps giving 20s peak
# Actual TR of ds000228 = 2000ms, but peak should be ~8s = index 4 at TR=2s
# So actual HRF TR is 2s, 37 points = 74s total, peak at 10*2s = 20s
# This is too long — canon2dd likely used TR=0.8s internally

# Define correct time axes
TR_actual = 0.8          # seconds, implied by peak calculation
TR_target = 2.0          # seconds, BOLD TR we want to match
n_points = 37

t_original = np.arange(n_points) * TR_actual          # 0 to 29.6s
t_target = np.arange(0, t_original[-1], TR_target)    # 0 to 29.6s in 2s steps

print(f"Original time axis: 0 to {t_original[-1]:.1f}s in {TR_actual}s steps")
print(f"Target time axis: 0 to {t_target[-1]:.1f}s in {TR_target}s steps")
print(f"New HRF length: {len(t_target)} timepoints")

hrfs_resampled = np.zeros((155, len(t_target)))

for i in range(155):
    f = interp1d(t_original, hrfs_full[i], kind='linear', fill_value=0, bounds_error=False)
    hrfs_resampled[i] = f(t_target)

# Verify peak
peak_idx = np.argmax(hrfs_resampled[0])
print(f"New peak at index {peak_idx} = {peak_idx * TR_target:.1f}s")

# Take first 76
hrfs_76 = hrfs_resampled[:76, :]

# Normalize
for i in range(76):
    peak = np.max(np.abs(hrfs_76[i]))
    if peak > 0:
        hrfs_76[i] = hrfs_76[i] / peak

np.save(r"E:\outputTVB\hrfs_76.npy", hrfs_76)
print("HRFs resampled and saved. Shape:", hrfs_76.shape)