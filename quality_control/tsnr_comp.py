from os.path import join, split
from nilearn import image
import sys

input_dir = sys.argv[1]

path_file, name_file = split(input_dir)
tsnr_func = image.math_img('img.mean(axis=3) / img.std(axis=3)', img=input_dir)
tsnr_func.to_filename(join(path_file, "vaso_tsnr.nii"))
