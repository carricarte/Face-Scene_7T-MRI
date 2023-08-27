import numpy as np
from os.path import join, split
import sys
import image_stats
from nilearn.image import new_img_like
from nilearn.masking import apply_mask

# loc_tmap_file = sys.argv[1]
# manual_mask_file = sys.argv[2]

# sub = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-07"]
sub = ["sub-08", "sub-09"]
hemisphere = ["left", "right"]
for s in sub:
    loc_tmap_file = "/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/" + s + "/analysis/localizer/spmT_0010.nii"
    manual_mask_file = "/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/" + s + "/mask/mask_PPA_vaso.nii"

    v = 400

    path_file, name_file = split(manual_mask_file)

    manual_mask = image_stats.load_nifti(manual_mask_file)
    loc_tmap = image_stats.load_nifti(loc_tmap_file)

    for h in hemisphere:

        if h == "left":
            file_name = "left_mask.nii"

        elif h == "right":
            file_name = "right_mask.nii"

        hemisphere_mask = join(path_file, file_name)
        mask = np.logical_and(manual_mask != 0,
                                     image_stats.load_nifti(hemisphere_mask) != 0)

        tval = list(loc_tmap[mask != 0])
        tval.sort()
        tval = tval[::-1]
        t = tval[v]
        loc_mask = image_stats.threshold_mask(loc_tmap_file, t, mask)

        mask_nifti = new_img_like(manual_mask_file, loc_mask)
        ribbon_epi_mask = join(path_file, "loc_" + file_name)
        mask_nifti.to_filename(ribbon_epi_mask)

    whole_mask = np.logical_or(image_stats.load_nifti(join(path_file, "loc_right_mask.nii")) != 0,
                                     image_stats.load_nifti(join(path_file, "loc_left_mask.nii")) != 0)

    mask_nifti = new_img_like(manual_mask_file, whole_mask)
    ribbon_epi_mask = join(path_file, "loc_wh_mask.nii")
    mask_nifti.to_filename(ribbon_epi_mask)
