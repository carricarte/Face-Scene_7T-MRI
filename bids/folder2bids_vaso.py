import sys
from os import listdir, makedirs
from os.path import join, splitext, exists
import shutil
import re


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


input_dir = sys.argv[1]
output_dir = sys.argv[2]
s = sys.argv[3]

func_dir = join(output_dir, "rawdata", "sub-" + s, "func")
anat_dir = join(output_dir, "rawdata", "sub-" + s, "anat")
fmap_dir = join(output_dir, "rawdata", "sub-" + s, "fmap")
deriv_func_dir = join(output_dir, "derivatives", "sub-" + s, "func")
deriv_anat_dir = join(output_dir, "derivatives", "sub-" + s, "anat")
roi_dir = join(output_dir, "derivatives", "sub-" + s, "mask")
deriv_fmap_dir = join(output_dir, "derivatives", "sub-" + s, "fmap")
mriqc_dir = join(output_dir, "derivatives", "mriqc")

if not exists(func_dir):
    makedirs(func_dir)

if not exists(anat_dir):
    makedirs(anat_dir)

if not exists(deriv_func_dir):
    makedirs(deriv_func_dir)

if not exists(deriv_anat_dir):
    makedirs(deriv_anat_dir)

if not exists(deriv_fmap_dir):
    makedirs(deriv_fmap_dir)

if not exists(roi_dir):
    makedirs(roi_dir)

if not exists(fmap_dir):
    makedirs(fmap_dir)

if not exists(mriqc_dir):
    makedirs(mriqc_dir)

subject_dir = join(input_dir, f"sub-{s}")
niftifiles = []
jsonfiles = []
img_r = 0
prf_r = 0
topup_r = 0
nifti_files = [f for f in listdir(subject_dir) if "._" not in f and "json" not in f]
sort_nicely(nifti_files)

for f in nifti_files:
    filename, extension = splitext(f)
    template = []

    if "bold" in filename or "cmrr" in filename:

        template = "sub-" + s + "_task-loc_run-01_bold"

    elif "_vaso" in f and "top-up" not in f:

        # template = filename[0:-4]
        img_r += 1
        template = "sub-" + s + "_task-img_run-%02d_vaso" % img_r

    elif "_vaso" in f and "top-up" in f:

        template = "sub-" + s + "_task-img_dir-down_run-%02d_vaso" % img_r

    elif "PRF" in f and "topup" in f:

        topup_r = topup_r + 1
        template = "sub-" + s + "_task-prf_dir-down_run-%02d" % topup_r

    elif "PRF" in f and "topup" not in f:

        prf_r = prf_r + 1
        template = "sub-" + s + "_task-prf_run-%02d" % prf_r

    elif "mp2rage" in filename:

        if "_0p7" in filename:

            a = 1
            if "_INV1" in filename:

                template = "sub-" + s + "_acq-INV1_run-%02d_MP2RAGE" % a

            elif "_INV2" in filename:

                template = "sub-" + s + "_acq-INV2_run-%02d_MP2RAGE" % a

            elif "_UNI_" in filename:

                template = "sub-" + s + "_acq-UNI_run-%02d_MP2RAGE" % a

    if template:
        if extension == ".gz":
            template = template + ".nii"
        if "mp2rage" in filename:
            shutil.copyfile(join(subject_dir, f), join(anat_dir, template + extension))
            shutil.copyfile(join(subject_dir, filename[0:-4] + ".json"), join(anat_dir, template + ".json"))
        elif "top-up" in filename or "topup" in filename:
            shutil.copyfile(join(subject_dir, f), join(fmap_dir, template + extension))
            if "vaso" not in filename:
                shutil.copyfile(join(subject_dir, filename[0:-4] + ".json"), join(fmap_dir, template + ".json"))
        else:
            shutil.copyfile(join(subject_dir, f), join(func_dir, template + extension))
            # shutil.copyfile(join(subject_dir, filename[0:-4] + ".json"), join(func_dir, template + ".json"))
        print("copying file: " + f + " to " + template)
