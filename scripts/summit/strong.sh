#!/bin/bash
#BSUB -P [REDACTED]
#BSUB -W 0:30
#BSUB -nnodes 1024
#BSUB -J scuda
#BSUB -N

PROG=./cuda

# https://jsrunvisualizer.olcf.ornl.gov/index.html?s4f1o01n1c42g1r11d1b1l0=
runone() {
  NNODE=$[$1 * 6]
  jsrun --nrs $NNODE --tasks_per_rs 1 --cpu_per_rs 7 --gpu_per_rs 1 --rs_per_host 6 \
    --smpiargs="-gpu" \
    --bind rs $PROG -d $DSIZE,$DSIZE,$DSIZE -I $2
}

date

echo "Experiment strong scaling from 8 to 1024 nodes with $DSIZE ^3 for $PROG"
runone 8 400
runone 16 600
runone 32 900
runone 64 1500
runone 128 2200
runone 256 3200
runone 512 5500
runone 1024 7000

