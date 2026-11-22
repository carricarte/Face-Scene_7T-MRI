import os
import nibabel as nb
import numpy as np
from nilearn.image import index_img


def load_nifti(nifti):
    """Return the data array for a file path, a nibabel image, or an array."""
    if isinstance(nifti, (str, os.PathLike)):
        return nb.load(nifti).get_fdata()
    if isinstance(nifti, nb.spatialimages.SpatialImage):
        return nifti.get_fdata()
    return np.asarray(nifti)


def _all_nonzero(*masks):
    """Voxel-wise AND of `mask != 0` over every mask given."""
    return np.logical_and.reduce([load_nifti(m) != 0 for m in masks])


def threshold_mask(map_file, threshold, *masks):
    """`map >= threshold`, restricted to the nonzero voxels of the given masks.

    masks[0] is the main mask; if masks[1] is given it is a 4D image whose first
    volume sets the non-NaN voxels; any further masks are ANDed with masks[0].
    """
    above = load_nifti(map_file) >= threshold
    if not masks:
        return above
    if len(masks) > 1:
        above = np.logical_and(nan_mask(masks[1]), above)
    return np.logical_and(above, _all_nonzero(masks[0], *masks[2:]))


def img_mask(img, mask, *more_masks):
    """Values of `img` inside the nonzero voxels of every mask given."""
    return load_nifti(img)[_all_nonzero(mask, *more_masks)]


def mean_roi(map_file, mask):
    t_map = load_nifti(map_file)
    return np.mean(t_map[np.logical_and(load_nifti(mask) != 0, np.logical_not(np.isnan(t_map)))])


def time_series_to_matrix(nifti_file, mask_file):
    epi_4d = load_nifti(nifti_file)
    mask = load_nifti(mask_file)

    features = epi_4d.reshape(-1, epi_4d.shape[-1])
    return features[mask.reshape(-1) != 0]


def time_series_to_mat(nifti_file):
    epi_4d = load_nifti(nifti_file)
    features = epi_4d.reshape(-1, epi_4d.shape[-1])
    return features


def volume_to_vector(nifti_file, mask_file):
    epi = load_nifti(nifti_file)
    mask = load_nifti(mask_file)
    epi_vector = epi.reshape(-1)
    mask_vector = mask.reshape(-1)
    return np.array(epi_vector[mask_vector != 0]).astype(int)


def vol_to_vector(nifti_file):
    epi = load_nifti(nifti_file)
    epi_vector = epi.reshape(-1)
    return np.array(epi_vector).astype(int)


def nan_mask(nifti_file):
    return np.logical_not(np.isnan(index_img(nifti_file, 0).get_fdata()))


def get_layer(ribbon_file):
    ribbon_mask = load_nifti(ribbon_file)
    layers = np.unique(ribbon_mask)

    mask_list = []
    for l in layers[layers != 0]:
        mask_list.append(ribbon_mask == l)
    return mask_list
