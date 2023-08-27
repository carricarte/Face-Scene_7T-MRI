from os.path import join
from nilearn import image
import nibabel as nb
import sys
from os.path import split

input_dir = sys.argv[1]

pathname, filename = split(input_dir)
fwhm = 6.  # value of 6 for the bold pilot study

nii_obj = nb.load(input_dir)
smoothed_img = image.smooth_img(nii_obj, fwhm)
smoothed_img.to_filename(join(pathname, 'sm_' + filename))
