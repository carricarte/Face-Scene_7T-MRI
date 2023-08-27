from os.path import join
import numpy as np
import sys
from skimage.morphology import binary_erosion, binary_dilation
from nilearn.image import load_img, new_img_like

wm_file = sys.argv[1]
mask_file = sys.argv[2]
output_dir = sys.argv[3]

wm_niimg = load_img(wm_file).get_fdata()
mask_niimg = load_img(mask_file).get_fdata()

# mask_niimg = np.logical_or(mask_niimg1 != 0, mask_niimg2 != 0)
wm_niimg[wm_niimg != 0] = 1

edges = wm_niimg

for d in np.arange(0, wm_niimg.shape[2]):
    # edges[:, :, d] = binary_dilation(wm_niimg[:, :, d]) - wm_niimg[:, :, d]
    edges[:, :, d] = wm_niimg[:, :, d] - binary_erosion(wm_niimg[:, :, d])

edges[mask_niimg == 0] = 0
edges[edges != 0] = 2
edges = new_img_like(mask_file, edges)
edges.to_filename(join(output_dir, "edge_mask_FFA.nii"))
