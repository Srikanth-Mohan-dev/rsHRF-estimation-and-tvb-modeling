import numpy as np
import matplotlib.pyplot as plt

regional_hrfs = np.load(r"E:\GSOC\output\regional_hrfs.npy")
region_labels = np.load(r"E:\GSOC\output\region_labels.npy")

tr = 2.0
time = np.arange(regional_hrfs.shape[1]) * tr

# Only search for peak in first 20s (first 10 timepoints)
peak_window = int(20 / tr)  # = 10

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: all regional HRFs overlaid
ax = axes[0]
for i in range(len(regional_hrfs)):
    ax.plot(time, regional_hrfs[i], alpha=0.3, linewidth=0.8, color='steelblue')
ax.plot(time, np.mean(regional_hrfs, axis=0), color='red',
        linewidth=2.5, label='Mean across regions')
ax.set_xlabel('Time (s)')
ax.set_ylabel('HRF amplitude')
ax.set_title('Regional HRFs — sub-pixar001\n(155 AAL regions, all overlaid)')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.legend()

# Right: time-to-peak within first 20s only
ax = axes[1]
t2p = [time[np.argmax(regional_hrfs[i, :peak_window])] 
       for i in range(len(regional_hrfs))]
ax.hist(t2p, bins=20, color='steelblue', edgecolor='white')
ax.set_xlabel('Time to peak (s)')
ax.set_ylabel('Number of regions')
ax.set_title('HRF time-to-peak distribution\nacross 155 regions (0–20s window)')
ax.axvline(np.mean(t2p), color='red', linewidth=2, 
           label=f'Mean: {np.mean(t2p):.1f}s')
ax.legend()

plt.tight_layout()
plt.savefig(r"E:\GSOC\output\figures\fig5_regional_hrfs.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Time-to-peak range: {min(t2p):.1f}s – {max(t2p):.1f}s")
print(f"Mean time-to-peak: {np.mean(t2p):.1f}s")