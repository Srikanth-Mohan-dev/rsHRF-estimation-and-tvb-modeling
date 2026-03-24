import numpy as np
import nibabel as nib
import scipy.io
from nilearn.image import resample_to_img

HRF_MAT    = r"E:\GSOC\output\sub-pixar001_task-pixar_run-001_swrf_hrf.mat"
BOLD_FILE  = r"E:\ds000228\derivatives\fmriprep\sub-pixar001\sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz"
ATLAS_FILE = r"C:\Users\mohan\nilearn_data\aal_3v2\AAL3\AAL3v1.nii"
MASK_FILE  = r"E:\ds000228\derivatives\fmriprep\sub-pixar001\sub-pixar001_analysis_mask.nii.gz"
OUT_DIR    = r"E:\GSOC\output"

# Step 1: Load HRF mat
print("Loading HRF mat...")
mat = scipy.io.loadmat(HRF_MAT)
hrfa = mat['hrfa']  # (37, 204275)
print(f"hrfa shape: {hrfa.shape}")

# Step 2: Derive mask directly from hrfa — non-zero columns only
# hrfa columns with any non-zero value = voxels rsHRF actually processed
print("Deriving exact rsHRF mask from hrfa non-zero columns...")
active_cols = np.any(hrfa != 0, axis=0)  # (204275,) bool
print(f"Non-zero hrfa columns: {np.sum(active_cols)}")

# Step 3: Load analysis mask and get its voxel indices
mask_img = nib.load(MASK_FILE)
mask_data = mask_img.get_fdata().astype(bool)
mask_indices = np.where(mask_data.ravel())[0]  # flat indices of mask voxels
print(f"Analysis mask voxels: {len(mask_indices)}")

# Step 4: Build 3D HRF volume — place hrfa columns into mask voxel positions
bold_img = nib.load(BOLD_FILE)
bold_shape = bold_img.shape[:3]
hrf_len = hrfa.shape[0]
hrf_volume = np.zeros((np.prod(bold_shape), hrf_len))

# Map each hrfa column to its corresponding flat voxel index
# rsHRF processes voxels in mask order — direct assignment
hrf_volume[mask_indices[:hrfa.shape[1]]] = hrfa.T
hrf_volume = hrf_volume.reshape((*bold_shape, hrf_len))
print(f"HRF volume shape: {hrf_volume.shape}")

# Step 5: Resample atlas to BOLD space
print("Resampling atlas to BOLD space...")
atlas_img = nib.load(ATLAS_FILE)
atlas_resampled = resample_to_img(atlas_img, bold_img, interpolation='nearest')
atlas_data = atlas_resampled.get_fdata()

# Step 6: Extract per-region HRFs
print("Extracting regional HRFs...")
region_ids = np.unique(atlas_data)
region_ids = region_ids[region_ids > 0].astype(int)

regional_hrfs = []
region_labels = []

for rid in region_ids:
    region_mask = (atlas_data == rid) & mask_data
    n_vox = np.sum(region_mask)
    if n_vox == 0:
        continue
    mean_hrf = np.mean(hrf_volume[region_mask], axis=0)
    regional_hrfs.append(mean_hrf)
    region_labels.append(rid)

regional_hrfs = np.array(regional_hrfs)
print(f"Final regional_hrfs shape: {regional_hrfs.shape}")

# Step 7: Save
np.save(rf"{OUT_DIR}\regional_hrfs.npy", regional_hrfs)
np.save(rf"{OUT_DIR}\region_labels.npy", np.array(region_labels))
print(f"Saved! Shape: {regional_hrfs.shape}")
print("Done!")