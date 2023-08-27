import sys
import os
import json

print("setting json file protocol")
subdir = sys.argv[1]

def protocol(subdir):
    protocol_dir = os.path.join(subdir, "derivatives", "conversion", "Protocol_Translator.json")
    
    if not os.path.exists(protocol_dir):
        print("file doesn't exist", protocol_dir)
        return

    #Debugging on local computer
    #subdir = "/Users/carricarte/Desktop"
    #protocol_dir = os.path.join(subdir, "Protocol_Translator.json")

    file = open(protocol_dir, "r")
    protocol = json.load(file)

    for k in protocol:

        value = ["EXCLUDE_BIDS_Directory", "EXCLUDE_BIDS_Nmae", "UNASSIGNED"]

        if "cmrr" in k or "mp2rage" in k:

            if "_FA5" in k and "INV1" in k:
                value[0] = "anat"
                value[1] = "acq-INV1_MP2RAGE"
                value[2] = "UNASSIGNED"

            elif "_FA5" in k and "INV2" in k:
                value[0] = "anat"
                value[1] = "acq-INV2_MP2RAGE"
                value[2] = "UNASSIGNED"

            elif "_FA5" in k and "UNI" in k:
                value[0] = "anat"
                value[1] = "acq-UNI_MP2RAGE"
                value[2] = "UNASSIGNED"

            elif "TR3000" in k and "_loc" in k:
                value[0] = "func"
                value[1] = "task-loc_bold"
                value[2] = "UNASSIGNED"

            elif "TR3000" in k and "_loc" not in k:
                value[0] = "func"
                value[1] = "task-imagery_bold"
                value[2] = "UNASSIGNED"

            print(k)
            protocol[k] = value

    #protocol["dUNI"] = ["anat", "acq-MP2RAGE_T1w", "UNASSIGNED"]

    file = open(protocol_dir, "w")
    json.dump(protocol, file)
    file.close()

if __name__ == "__main__":
    protocol(subdir)

