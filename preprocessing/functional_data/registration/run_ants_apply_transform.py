from os.path import join
from os import listdir
import subprocess
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]
fixed_img = sys.argv[3]
script_dir = sys.argv[4]
generic_affine = sys.argv[5]
deformation_field = sys.argv[6]

epi_list = []
[epi_list.append(_) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _ and "corrected" in _]
epi_list.sort()

for r in epi_list:
    
    moving_img = join(input_dir, r)
    output_img = join(output_dir, 'cr_' + r)
    subprocess.call(
        ['sbatch', join(script_dir, "ants_apply_transform"), fixed_img, moving_img, generic_affine,
        deformation_field, output_img])