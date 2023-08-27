import sys
import os
import json

main_dir = sys.argv[1]
key = sys.argv[2]
subject = sys.argv[3]

paths_dict = os.path.join(main_dir, "paths.json")

if os.path.exists(paths_dict) is None:
    sys.exit(1)

file = open(paths_dict, "r")
directories = json.load(file)

status = 0
all_dir = eval(directories[key][0])

if not "EXCLUDE" in all_dir:
    if os.path.exists(all_dir) == False:
        status = 1

for dir_exp in directories[key][1::]:
    
    dir = eval(dir_exp)

    if not "EXCLUDE" in dir:
        if os.path.exists(dir) == False:
            # print(dir)
            status = 1
    all_dir = all_dir + ';' + dir

sys.stdout.write(all_dir)
sys.exit(status)


