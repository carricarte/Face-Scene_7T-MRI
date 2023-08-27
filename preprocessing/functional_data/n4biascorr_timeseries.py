import sys
from os import listdir
from os.path import split, join
from nilearn.image import math_img, load_img, new_img_like, index_img

epi_dir = sys.argv[1]
bias_file = sys.argv[2]

# epi_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/func"
# bias_file = "/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-04/func/BiasField.nii"

output_dir, filename = split(epi_dir)

epi_files = []
[epi_files.append(join(epi_dir, _)) for _ in listdir(epi_dir) if _.endswith(".nii") and "._" not in _ and "rsub" in _
 and "corr_" not in _ and "wcr_rsub" not in _ and "cr_" not in _ and "mean" not in _ and "vaso" in _ and "tr" not in _
 and "run-01" in _]
epi_files.sort()

bias_img = load_img(bias_file).get_fdata()
for epi in epi_files:

    nifti_img = load_img(epi).get_fdata()
    result_img = load_img(epi).get_fdata()

    for v in range(0, result_img.shape[3]):
        output_dir, filename = split(epi)
        result_img[:, :, :, v] = nifti_img[:, :, :, v]/bias_img
        # result_img[:, :, :, v] = math_img("img1 / img2", img1=index_img(epi, v), img2=bias_file).get_fdata()
    result_img = new_img_like(epi, result_img)
    result_img.to_filename(join(output_dir, "bc_" + filename))
