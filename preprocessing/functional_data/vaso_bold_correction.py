from nipype.interfaces.spm import Realign
from os.path import join
from nilearn.image import index_img, new_img_like
import nibabel as nb
from os.path import splitext, split
from shutil import copyfile
from os import rename, listdir
import numpy as np
import sys
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
import image_stats as st

dph = 2.75
TR = 4.826


def interpolator(values, points, tr, phase):
    import numpy as np
    from scipy.interpolate import interp1d

    time_points = np.arange(points) * tr
    new_points = time_points - phase
    # new_points = np.arange(0, (points - 1) * tr + new_tr, new_tr)
    interpol = interp1d(time_points, values, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interpol(new_points)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

vaso_list = []
[vaso_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "rsub" in _ and "loc" not in _ and "wcr_rsub" not in _ and "intr" not in _
 and "cr_rsub" not in _ and "mean" not in _ and "gz" not in _ and "vaso" in _ and "corr" not in _
 and "cr_2" not in _ and "bc" not in _ and "tr" not in _]
vaso_list.sort()

bold_list = []
[bold_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "rsub" in _ and "loc" not in _ and "wcr_rsub" not in _ and "intr" not in _
 and "cr_rsub" not in _ and "mean" not in _ and "gz" not in _ and "bold" in _ and "corr" not in _
 and "cr_2" not in _ and "bc" not in _ and "tr" not in _ and "intemp" not in _]
bold_list.sort()

prefix = "corr_"

for i in np.arange(0, len(vaso_list)):

    vaso_nii = vaso_list[i]
    bold_nii = bold_list[i]

    ref_vol = nb.load(vaso_nii)
    ref_bold = nb.load(bold_nii)
    ori_shape = np.array(np.array(ref_vol.get_fdata()).shape)
    # new_shape = ori_shape
    # new_shape[3] = ori_shape[3] * 2 - 1

    filepath, vaso_filename = split(vaso_nii)
    filepath, bold_filename = split(bold_nii)

    print(vaso_nii)
    print(bold_nii)
    print("___________________")

    vaso_features = st.time_series_to_mat(vaso_nii)
    bold_features = st.time_series_to_mat(bold_nii)

    print(vaso_features.shape)
    # print(bold_features_1.shape)
    print("___________________")

    # bold_features_2 = bold_features_1[:, 1:]
    # bold_features_2 = np.concatenate((bold_features_2, np.array(bold_features_1[:, bold_features_1.shape[1] - 1]).reshape(len(bold_features_1), 1)), axis=1)

    # corr_vaso = np.array(vaso_features/(0.5*bold_features_1 + 0.5*bold_features_2)).astype(np.float32)

    # vaso_interpolated_features = interpolator(vaso_features, vaso_features.shape[1], TR, new_TR)
    # corr_vaso = interpolator(vaso_features, vaso_features.shape[1], TR, new_TR)

    bold_interpolated_features = interpolator(bold_features, bold_features.shape[1], TR, dph)
    corr_vaso = np.array(vaso_features/bold_interpolated_features).astype(np.float32)

    # itr_vaso = vaso_interpolated_features.reshape(tuple(new_shape))

    #
    # itr_vaso = new_img_like(ref_vol, itr_vaso)
    #
    # pixdim = np.array(ref_vol.header.get_zooms())
    # new_pixdim = pixdim
    # new_pixdim[3] = pixdim[3]/2
    # ref_vol.header.set_zooms(tuple(new_pixdim))
    itr_bold = bold_interpolated_features.reshape(tuple(ori_shape))
    itr_bold = new_img_like(ref_bold, itr_bold)
    corr_vaso = corr_vaso.reshape(tuple(ori_shape))
    corr_vaso = new_img_like(ref_vol, corr_vaso)
    corr_vaso.to_filename(join(output_dir, prefix + vaso_filename))
    itr_bold.to_filename(join(output_dir, "intemp_" + bold_filename))

    # itr_vaso.to_filename(join(output_dir, "intr_" + vaso_filename))
    # print(epi_file)
