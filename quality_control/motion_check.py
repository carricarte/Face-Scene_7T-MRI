
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
import pandas as pd

input_dir = '/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-07/func'
# output_dir = '/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-04/analysis/vaso'
# tsv_dir = '/Users/carricarte/scratch/projects/imagery/main/vaso/behdata/sub-04/'

bold_motion_files = []
[bold_motion_files.append(join(input_dir, f)) for f in listdir(input_dir) if "rp_sub" in f and
 "bold" in f and f.endswith(".txt") and "._" not in f and "mean" not in f and "loc" not in f]
bold_motion_files.sort()

vaso_motion_files = []
[vaso_motion_files.append(join(input_dir, f)) for f in listdir(input_dir) if "rp_sub" in f and
 "vaso" in f and f.endswith(".txt") and "._" not in f and "mean" not in f and "loc" not in f]
vaso_motion_files.sort()

for i, e in enumerate(vaso_motion_files):
    df_vaso = pd.read_table(vaso_motion_files[i], sep='  ', header=None)
    df_bold = pd.read_table(bold_motion_files[i], sep='  ', header=None)
    plt.plot(df_vaso.iloc[:, :3], color="blue", linewidth=0.2)
    plt.plot(df_bold.iloc[:, :3], color="red", linewidth=0.2)
    plt.show()
print("hello world")
