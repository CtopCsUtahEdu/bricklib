#!/bin/bash
#COBALT -t 60
#COBALT -n 1024
#COBALT -q default
#COBALT --attrs mcdram=flat:numa=quad
#COBALT -A [REDACTED]

export OMP_PLACES=cores
export KMP_HOT_TEAMS_MODE=1
export KMP_HOT_TEAMS_MAX_LEVEL=2
export OMP_NUM_THREADS=256
SIZE=2048
PROG=./cpu

runone() {
	aprun -n $1 -N 1 \
	--env OMP_NUM_THREADS=256 --env OMP_PLACES=cores --env KMP_HOT_TEAMS_MODE=1 --env KMP_HOT_TEAMS_MAX_LEVEL=2 \
	--cc depth -d 256 -j 4 numactl -m 1 $PROG -d $SIZE,$SIZE,$SIZE -I 500
}

runone 32
runone 64
runone 128
runone 256
runone 512
runone 1024

