import subprocess as sp

program = 'matlab'
script = '/Users/carricarte/home/layer_mri/pipeline/preprocessing/anatomical_data/triangle_area_fun;exit'
b = 5
h = 3
args = '({},{})'.format(b, h)
fun = [program, '-nodesktop', '-nodisplay', 'r', script]

print('command:', fun)

print('running MATLAB script...')
sp.Popen(fun)

print('done')
