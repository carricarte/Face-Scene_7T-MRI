import sys
import numpy as np
from os import listdir
import nibabel as nb
from os.path import split, join
from os import remove


def to_float16(path_to_file):
    path, file = split(path_to_file)
    output_img = join(path, "float_" + file)

    volume_obj = nb.load(path_to_file)
    volume_data = np.array(volume_obj.get_fdata()).astype(np.single)
    new_volume = nb.Nifti1Image(volume_data, volume_obj.affine, volume_obj.header)
    new_volume.header.set_data_dtype(np.single)
    nb.save(new_volume, output_img)


input_dir = sys.argv[1]

run_list = []
[run_list.append(join(input_dir, f)) for f in listdir(input_dir) if "cr_" in f and '._' not in f
 and "00" not in f and "sm_" not in f]
run_list.sort()

for run in run_list:
    to_float16(run)
    # [remove(join(run, x)) for x in listdir(run) if "r_" in x]
    # [to_float16(join(run, x)) for x in listdir(run) if "._" not in x and ".mat" not in x and "r_" not in x]
