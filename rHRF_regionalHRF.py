import scipy.io
import nibabel as nib
import numpy as np

# 1. Load the HRF mat file
hrf_data = scipy.io.loadmat('output/sub-pixar001_task-pixar_run-001_swrf_hrf.mat')
# explore keys first: print(hrf_data.keys())

# 2. Load an MNI atlas (e.g. AAL90 or Hagmann parcellation)
atlas = nib.load('atlas.nii.gz')
atlas_data = atlas.get_fdata()  # integer labels per voxel

# 3. For each region, average the HRFs of all voxels in that region
n_regions = 66
regional_hrfs = []
for region_id in range(1, n_regions+1):
    voxel_indices = np.where(atlas_data == region_id)
    region_hrfs = hrf_data[voxel_indices]  # shape: [n_voxels_in_region × hrf_length]
    mean_hrf = np.mean(region_hrfs, axis=0)
    regional_hrfs.append(mean_hrf)

regional_hrfs = np.array(regional_hrfs)  # shape: [66 × hrf_length]