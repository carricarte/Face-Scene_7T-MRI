from nilearn.image import new_img_like
from os.path import split, join
from os import listdir
import sys
import numpy as np

sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
import image_stats as st

nifti_dir = sys.argv[1]
layer_file = sys.argv[2]
column_file = sys.argv[3]
ALF_file = sys.argv[4]

mode = "deconv"

run_list = []
[run_list.append(join(nifti_dir, f)) for f in listdir(nifti_dir) if "run-" in f and '._' not in f]
run_list.sort()

layer_data = st.vol_to_vector(layer_file)
column_data = st.vol_to_vector(column_file)
ALF_data = st.vol_to_vector(ALF_file)
max_layer = np.max(np.unique(layer_data))
# [fnifti.append(join(nifti_dir, f)) for f in listdir(nifti_dir) if "cr_rsub" in f and f.endswith(".nii")
#  and "._" not in f and "devein" not in f and "wcr" not in f and "sm_" not in f and "loc" not in f]
# fnifti.sort()

if mode == "linear":

    linear_factor = np.zeros(len(layer_data))

    for l in np.arange(1, max_layer + 1):
        linear_factor[layer_data == l] = 1. - (layer_data[layer_data == l] - 0.5) / max_layer

else:

    max_column = np.max(np.unique(column_data))
    lambda_ = 0.25

# for r in run_list:
fnifti = []
[fnifti.append(join(nifti_dir, f)) for f in listdir(nifti_dir) if "devein" not in f and f.endswith(".nii")
 and "._" not in f]
fnifti.sort()

for epi in fnifti:

    epi_data = st.load_nifti(epi)
    original_shape = epi_data.shape
    print(original_shape)
    path_file, name = split(epi)

    if len(original_shape) > 3:

        data = epi_data.reshape(-1, epi_data.shape[-1]).T
        t = original_shape[3]

    else:

        data = epi_data.reshape(1, -1)
        t = 1

    if mode == "linear":

        new_file = join(path_file, name[:-4]) + "_devein_linear.nii"
        deveined_data = np.float32(data * linear_factor)
        data[:, layer_data == 0] = 0

    else:

        deveined_data = np.zeros(shape=data.shape).astype(np.float32)

        for c in np.arange(1, max_column + 1):

            vec1 = np.zeros(shape=[t, max_layer]).astype(np.float32)
            vec2 = np.zeros(shape=[t, max_layer]).astype(np.float32)
            vecALF = np.zeros(shape=[max_layer]).astype(np.float32)
            vec_nr_voxels = np.zeros(shape=[max_layer]).astype(np.float32)

            for l in np.arange(1, max_layer + 1):
                log_vec = np.logical_and(layer_data == l, column_data == c)
                vec_nr_voxels[l - 1] = np.sum(log_vec)
                vecALF[l - 1] = np.sum(ALF_data[log_vec])
                vec1[:, l - 1] = np.sum(data[:, log_vec], 1)

            vec1[:, vec_nr_voxels > 0] = vec1[:, vec_nr_voxels > 0] / vec_nr_voxels[vec_nr_voxels > 0]
            vecALF[vec_nr_voxels > 0] = vecALF[vec_nr_voxels > 0] / vec_nr_voxels[vec_nr_voxels > 0]

            # Normalize amplitude of low frequencies (ALF)
            divisor = np.sum(vecALF[vec_nr_voxels > 0])
            if divisor != float(0):
                vecALF = vecALF / np.sum(vecALF[vec_nr_voxels > 0])
            sum_vec = np.zeros(max_layer).astype(np.float32)

            if mode == "CBV":
                for l in range(0, max_layer - 1):
                    new_file = join(path_file, name[:-4]) + "_devein_cbv.nii"
                    vec2[:, vec_nr_voxels > 0] = vec1[:, vec_nr_voxels > 0] / vecALF[vec_nr_voxels > 0] * float(
                        max_layer)

            elif mode == "deconv":

                for l in range(0, max_layer - 1):
                    new_file = join(path_file, name[:-4]) + "_devein_deconv.nii"
                    false_arr = np.zeros(len(vec_nr_voxels)).astype(bool)
                    false_arr[:l + 1] = True
                    log_vec = np.logical_and(vec_nr_voxels > 0, false_arr)
                    vec_temp = vec1[:, log_vec] / max_layer / vecALF[log_vec] * lambda_
                    sum_vec[l + 1] = np.sum(vec_temp[vec_temp == vec_temp])

                vec2[:, vec_nr_voxels > 0] = vec1[:, vec_nr_voxels > 0] - sum_vec[vec_nr_voxels > 0]

            for l in np.arange(1, max_layer + 1):

                log_vec = np.logical_and(layer_data == l, column_data == c)
                indexes = np.array(np.where(log_vec == 1))[0]
                # if deveined_data.ndim == 1:
                #
                #     deveined_data[log_vec] = vec2[:, l - 1]
                #
                # elif deveined_data.dmin == 2:
                for ind in indexes:
                    deveined_data[:, ind] = vec2[:, l - 1]

    new_img = new_img_like(epi, np.array(np.array(deveined_data.T).reshape(original_shape)).astype(np.float32))
    new_img.to_filename(new_file)
    print(new_file)
