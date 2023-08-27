#!/bin/sh

#SBATCH --job-name=bids_conversion
#SBATCH --nodes=1
#SBATCH --time=60:00
#SBATCH --mem=4096
#SBATCH --qos=standard
#SBATCH --mail-type=END
#SBATCH --mail-user=carricarte@zedat.fu-berlin.de

INPUTDIR="$1"
PIPELINEDIR="$2"

python3 "$PIPELINEDIR/set_protocol.py" "$INPUTDIR"
singularity run --bind "$INPUTDIR":/mnt bidskit.sif  --indir=/mnt/sourcedata --outdir=/mnt/rawdata --no-sessions