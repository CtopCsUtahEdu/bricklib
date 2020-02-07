# This will compile for OpenSHMEM in weak/shmem.cpp

CC=oshc++

all:shmem

shmem:weak/shmem-out.cpp src/brick-mpi.cpp src/memfd.cpp src/zmort.cpp stencils/brickcompare.cpp stencils/multiarray.cpp
	$(CC) $^ -o $@ -std=c++11 -Iinclude -I. -lrt -lmpi_cxx -lmpi -fopenmp -O3

clean:
	-rm shmem

.PHONY:all clean
