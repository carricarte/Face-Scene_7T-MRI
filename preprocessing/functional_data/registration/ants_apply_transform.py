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
#collapsed_field_img = join(output_dir, 'registered_collapsedWarp.nii.gz')

#subprocess.call(
#    ['sbatch', join(script_dir, "combine_deformation"), fixed_img, generic_affine,
#     deformation_field, collapsed_field_img])
seq = "bold"
epi_list = []
[epi_list.append(_) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _ and "rsub" in _
 and "cr_" not in _ and "bc" not in _ and "corr" not in _ and "mean" not in _ and "tr" not in _
 and seq in _ and "AFL" not in _ and "loc" not in _]
epi_list.sort()

# [epi_list.append(_) for _ in listdir(input_dir) if _.endswith(".nii") and "._"
# not in _ and "rsub" in _ and "cr_" not in _ and "mean" not in _ and "loc" in _
#  and "bold" in _ and "img" not in _]
epi_list.sort()

for r in epi_list:
    print(r)
    moving_img = join(input_dir, r)
    output_img = join(output_dir, 'cr_' + r)
    subprocess.call(
        ['sbatch', join(script_dir, "ants_apply_transform"), fixed_img, moving_img, generic_affine, deformation_field,
         output_img])