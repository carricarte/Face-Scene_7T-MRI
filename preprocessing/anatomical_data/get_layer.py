import nibabel as nb
from nighres.surface import probability_to_levelset
from nighres.laminar import volumetric_layering
import sys

# parameters
input_dir = sys.argv[1]
output_dir = sys.argv[2]

gm_val = [3, 42]
wm_val = [2, 41]
n_layers = 3

""" do not edit below """

# wm mask
wm_img = nb.load(input_dir)
wm_array = wm_img.get_fdata()
wm_array[wm_array == gm_val[0]] = 0
wm_array[wm_array == gm_val[1]] = 0
wm_array[wm_array != 0] = 1
wm_mask = nb.Nifti1Image(wm_array, wm_img.affine, wm_img.header)

# csf mask
csf_img = nb.load(input_dir)
csf_array = csf_img.get_fdata()
csf_array[csf_array != 0] = 1
csf_mask = nb.Nifti1Image(csf_array, csf_img.affine, csf_img.header)

# probability to levelset
wm_level = probability_to_levelset(wm_mask)
csf_level = probability_to_levelset(csf_mask)

# layering
volumetric_layering(wm_level["result"],
                    csf_level["result"],
                    n_layers=n_layers,
                    topology_lut_dir=None,
                    save_data=True,
                    overwrite=True,
                    output_dir=output_dir,
                    file_name="corr")
