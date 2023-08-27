def interpolator(values, points, tr, new_tr):
    import numpy as np
    from scipy.interpolate import interp1d

    time_points = np.arange(points) * tr
    new_points = np.arange(0, (points - 1) * tr + new_tr, new_tr)
    interpol = interp1d(time_points, values, kind='linear')
    return interpol(new_points)


def nifti2dataframe(*args):
    nifti_list = args[0]
    mask_f = args[1]
    t_map_file = args[2]
    layer_f = args[3]
    n_vox = args[4]

    t_map = join(t_map_file, "spmT_0006.nii")
    # t_map_file = join(t_map_file, "spmT_0010.nii")

    mask_dir, filename = split(mask_file)

    layer = image_stats.load_nifti(layer_f)
    n_lay = max(np.unique(layer)).astype(int)
    t = 4.76
    mask_layer_list = image_stats.get_layer(layer_f)
    mask = np.zeros(shape=layer.shape)
    for i, layer_mask in enumerate(mask_layer_list):
        l_mask = new_img_like(layer_f, layer_mask)
        new_layer_file = join(mask_dir, 'layer_{}_mask.nii'.format(i + 1))
        l_mask.to_filename(new_layer_file)
        tval = image_stats.img_mask(t_map, mask_f, new_layer_file)
        tval.sort()
        tval = tval[::-1]
        t = tval[n_voxel]
        mask_ffa = np.logical_or(mask, image_stats.threshold_mask(t_map, t, nifti_list[0], mask_f, new_layer_file))
        # mask_ppa = np.logical_or(mask, image_stats.threshold_mask(t_map_ppa_file, t, nifti_list[0], mask_f, new_layer_file))
        mask = np.logical_or(mask_ffa, mask_ffa)

    mask_img = new_img_like(mask_f, mask)
    new_mask_file = join(mask_dir, 'layer_mask_{}.nii'.format(n_voxel + 1))
    mask_img.to_filename(new_mask_file)

    # concatenate all functional volumes across runs in a 4D matrix
    features = image_stats.time_series_to_matrix(nifti_list[0], new_mask_file)
    # time interpolation from 3s to 0.5s
    interpolated_features = interpolator(features, features.shape[1], TR, newTR)

    feature_matrix = np.zeros(
        shape=(len(interpolated_features), interpolated_features.shape[1] * (len(nifti_list) + 1)))  # plus one
    feature_matrix[:, 0:interpolated_features.shape[1]] = interpolated_features
    last_index = np.array([0])
    last_index = np.concatenate((last_index, np.array([last_index[i] + interpolated_features.shape[1]])),
                                axis=0)

    # small bug in last index CHECK BUG
    for i, nifti_file in enumerate(nifti_list[1::]):
        print(nifti_file)
        features = image_stats.time_series_to_matrix(nifti_file, new_mask_file)

        interpolated_features = interpolator(features, features.shape[1], TR, newTR)

        if i < len(nifti_list[1::]) - 1:
            last_index = np.concatenate((last_index, np.array([last_index[i + 1] + interpolated_features.shape[1]])),
                                        axis=0)
        feature_matrix[:,
        last_index[i + 1]: last_index[i + 1] + interpolated_features.shape[1]] = interpolated_features

    feature_matrix = feature_matrix.T

    vector_layer_mask = image_stats.volume_to_vector(layer_f, new_mask_file)

    df_featuers = pd.DataFrame(data=feature_matrix)
    df_featuers.to_csv(join(output_dir, 'features_{0}_layer-{1}_vox-{2}.tsv'.format("FFA_wb", n_lay, n_vox + 1))
                       , sep='\t', index=True)
    df_volumes = pd.DataFrame(data=last_index)
    df_volumes.to_csv(join(output_dir, 'vol_x_run.tsv'), sep='\t', index=True)
    df_layer = pd.DataFrame(data=vector_layer_mask)
    df_layer.to_csv(join(output_dir, 'layer_mask_{0}_layer-{1}_vox-{2}.tsv'.format("FFA_wb", n_lay, n_vox + 1))
                    , sep='\t', index=True)

    print(feature_matrix.shape)


import sys
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, split
import image_stats
from nilearn.image import new_img_like

# directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
tmap_file = sys.argv[4]
layer_file = sys.argv[5]

# get epi and data
TR = 3
newTR = 0.5
# voxels = [0]
voxels = np.arange(20, 3000, 150) - 1


fnifti = []
[fnifti.append(join(input_dir, f)) for f in listdir(input_dir) if "cr_rsub" in f and "devein" not in f and
 f.endswith(".nii") and "._" not in f and "00" not in f and "wcr" not in f and "sm_" not in f and "ALF" not in f]
fnifti.sort()

for n_voxel in voxels:
    nifti2dataframe(fnifti, mask_file, tmap_file, layer_file, n_voxel)
