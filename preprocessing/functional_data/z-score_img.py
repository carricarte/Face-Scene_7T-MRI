import sys
from os.path import join, split
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis')
import image_stats
import numpy as np
from nilearn.image import load_img, new_img_like

input_dir = sys.argv[1]
layer_file = sys.argv[2]

# input_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-01/func/sub-01_task-img_run-01_vaso.nii"
# layer_file = "/Users/carricarte/PhD/Debugging/vaso/main/sub-01/mask/cr_2vaso_layer_layering-layers.nii"

layer = ["deep", "middle", "superficial"]
hemi = ["left", "right"]

output_dir, basename = split(input_dir)
mask_dir, basename2 = split(layer_file)

left_hemisphere_file = join(mask_dir, "left_mask.nii")
right_hemisphere_file = join(mask_dir, "right_mask.nii")

hemi_left_mask = load_img(left_hemisphere_file).get_fdata()
hemi_right_mask = load_img(right_hemisphere_file).get_fdata()

def get_layer(ribbon_file):
    ribbon_mask = load_img(ribbon_file).get_fdata()
    layers = np.unique(ribbon_mask)

    mask_list = []
    for l in layers[layers != 0]:
        mask_list.append(ribbon_mask == l)
    return mask_list

mask_layer_list = get_layer(layer_file)
ori_shape = mask_layer_list[0].shape
niimg = image_stats.time_series_to_mat(input_dir)
z_niimg = np.zeros(shape=niimg.shape)

for k, layer_mask in enumerate(mask_layer_list):
    for h in hemi:
        if h == "left":
            hemi_layer_mask = np.logical_and(layer_mask != 0, hemi_left_mask != 0)
        elif h == "right":
            hemi_layer_mask = np.logical_and(layer_mask != 0, hemi_right_mask != 0)

        hemi_layer_mask = image_stats.vol_to_vector(hemi_layer_mask)
        masked_data = niimg[hemi_layer_mask != 0]
        # m = np.mean(masked_data, 1)
        # s = np.std(masked_data,1)
        z_niimg[hemi_layer_mask != 0] = (masked_data - np.mean(np.mean(masked_data, 0))) / np.std(np.std(masked_data, 0))
        # z_data = (masked_data - m.reshape(len(m), 1))/s.reshape(len(m), 1)
        # print("hemisphere: {} layer: {} mean {}".format(h, layer[k], m))

z_niimg = z_niimg.reshape(ori_shape + z_niimg.shape[-1:])
z_niimg = new_img_like(input_dir, np.array(z_niimg).astype(np.float32))
z_niimg.to_filename(join(output_dir, "z_" + basename))
