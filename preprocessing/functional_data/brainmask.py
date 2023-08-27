from os.path import join, split
from os import listdir
from nilearn.masking import compute_epi_mask, unmask, apply_mask
from nilearn.maskers import NiftiMasker
from nilearn.image import smooth_img, binarize_img, load_img, new_img_like
import sys

# ma = ImageMath(op1="", operation="FillHoles")
# ma.
niimg = sys.argv[1]
mask = sys.argv[2]

pathname, filename = split(niimg)
# nifti_list = []
# [nifti_list.append(join(input_dir, nifti)) for nifti in listdir(input_dir) if "bc_mean" in nifti
#  and nifti.endswith("nii") and "str" not in nifti]
# nifti_list.sort()
#
# for niimg in nifti_list:
#     pathname, filename = split(niimg)
#
#     # brainmask = compute_epi_mask(niimg, opening=3)
#     # masked_brain = apply_mask(niimg, brainmask)
#     # masked_brain = unmask(masked_brain, brainmask)
#
#     if "loc" in niimg:
#         # sub-01 use these params and not the bias field corrected mean image
#         # masker = NiftiMasker(mask_strategy='epi', mask_args=dict(upper_cutoff=.8, lower_cutoff=.7,
#         #                                                          opening=False))
#         # mask = masker.fit(niimg).mask_img_
#         # mask = smooth_img(mask, fwhm=3)
#         # mask = binarize_img(mask, threshold=0.001)
#         # mask.set_data_dtype(float)
#
#         masker = NiftiMasker(mask_strategy='epi', mask_args=dict(upper_cutoff=.61, lower_cutoff=.59,
#                                                                  opening=False))
#         mask = masker.fit(niimg).mask_img_
#         mask = smooth_img(mask, fwhm=3)
#         mask = binarize_img(mask, threshold=0.00125)
#         mask.set_data_dtype(float)
#
#     else:
#
#         masker = NiftiMasker(mask_strategy='epi', mask_args=dict(opening=6))
#         mask = masker.fit(niimg).mask_img_
#         mask = smooth_img(mask, fwhm=6)
#         mask = binarize_img(mask, threshold=0.2)
#         mask.set_data_dtype(float)

# masked_brain = apply_mask(niimg, mask)
# masked_brain = unmask(masked_brain, mask)
# masked_brain.to_filename(join(pathname, "str_" + filename))

mask_img = load_img(mask).get_fdata()
epi = load_img(niimg).get_fdata()

epi[mask_img == 0] = 0
masked_epi = new_img_like(niimg, epi)
masked_epi.to_filename(join(pathname, "str_" + filename))
