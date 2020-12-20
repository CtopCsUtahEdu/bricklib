#!/bin/bash
#COBALT -t 60
#COBALT -n 1
#COBALT -q debug-flat-quad
#COBALT --attrs mcdram=flat:numa=quad
#COBALT -A [REDACTED]

export OMP_PLACES=cores
export KMP_HOT_TEAMS_MODE=1
export KMP_HOT_TEAMS_MAX_LEVEL=2
export OMP_NUM_THREADS=256

runone() {
	aprun -n 1 -N 1 \
	--env OMP_NUM_THREADS=256 --env OMP_PLACES=cores --env KMP_HOT_TEAMS_MODE=1 --env KMP_HOT_TEAMS_MAX_LEVEL=2 \
	--cc depth -d 256 -j 4 numactl -m 1 ./cpu -s $1,$1,$1 -I $2
}

runone 512 100
runone 256 200
runone 128 300
runone 64  500
runone 32  700
runone 16 1000
