#!/bin/bash
#BSUB -P [REDACTED]
#BSUB -W 2:00
#BSUB -nnodes 8
#BSUB -J scuda
#BSUB -N

PROG=./cuda

# https://jsrunvisualizer.olcf.ornl.gov/index.html?s4f1o01n1c42g1r11d1b1l0=
runone() {
  DSIZE=$[$1 * 2]
  jsrun --nrs $NNODE --tasks_per_rs 1 --cpu_per_rs 7 --gpu_per_rs 1 --rs_per_host 1 \
    --smpiargs="-gpu" \
    --bind rs $PROG -s $1,$1,$1 -I $2
}

date

echo "Experiment $PROG using $NNODE nodes with $NPROC per node"
echo "  from a minimum of $MINSIZE^3 to a total of $MAXSIZE^3 domain"

runone 16 16000
runone 32 8000
runone 64 4000
runone 128 2000
runone 256 1000
runone 512 500
