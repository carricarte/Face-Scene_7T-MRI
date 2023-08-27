import nibabel as nb
import numpy as np
from nilearn.image import index_img


def load_nifti(nifti_file):
    if str(type(nifti_file)) == "<class 'nibabel.nifti1.Nifti1Image'>":
        return nifti_file.get_fdata()
    else:
        return nb.load(nifti_file).get_fdata()


def threshold_mask(*args):
    map_file = args[0]
    threshold = args[1]

    if isinstance(map_file, str):

        map_file = load_nifti(map_file)

    elif str(type(map_file)) == "<class 'nibabel.nifti1.Nifti1Image'>":

        map_file = map_file.get_fdata()

    if len(args) > 2:

        new_mask = args[2]
        if isinstance(new_mask, str):

            new_mask = load_nifti(new_mask)

        elif str(type(new_mask)) == "<class 'nibabel.nifti1.Nifti1Image'>":

            new_mask = new_mask.get_fdata()

        if len(args) > 3:
            mask_nan = nan_mask(args[3])

            if len(args) > 4:

                for arg in args[4:]:

                    if isinstance(arg, str):

                        arg = load_nifti(arg)

                    elif str(type(arg)) == "<class 'nibabel.nifti1.Nifti1Image'>":

                        arg = arg.get_fdata()

                    new_mask = np.logical_and(arg != 0, new_mask != 0)

            return np.logical_and(np.logical_and(mask_nan, map_file >= threshold), new_mask != 0)

        return np.logical_and(map_file >= threshold, new_mask != 0)
    else:
        return np.array(map_file >= threshold).astype(bool)



def img_mask(*args):

    img = args[0]
    mask = args[1]

    if isinstance(img, str):

        img = load_nifti(img)

    elif str(type(img)) == "<class 'nibabel.nifti1.Nifti1Image'>":

        img = img.get_fdata()

    if isinstance(mask, str):

        mask = load_nifti(mask)

    elif str(type(mask)) == "<class 'nibabel.nifti1.Nifti1Image'>":

        mask = mask.get_fdata()

    if len(args) > 2:

        for arg in args[2:]:

            if isinstance(arg, str):

                arg = load_nifti(arg)

            elif str(type(arg)) == "<class 'nibabel.nifti1.Nifti1Image'>":

                arg = arg.get_fdata()


            mask = np.logical_and(arg != 0, mask != 0)

    return img[mask != 0]


def mean_roi(map_file, mask):

    t_map = load_nifti(map_file)

    if isinstance(mask, str):

        mask = load_nifti(mask)

    elif str(type(mask)) == "<class 'nibabel.nifti1.Nifti1Image'>":

        mask = mask.get_fdata()


    return np.mean(t_map[np.logical_and(mask != 0, np.logical_not(np.isnan(t_map)))])


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
