all: highlife_mpi.c highlife.cu
	mpixlc -O3 highlife_mpi.c -c -o highlife-mpi.o
	nvcc -O3 -arch=sm_70 highlife.cu -c -o highlife-cuda.o
	mpicc -O3 highlife-mpi.o highlife-cuda.o -o highlife-exe \
	-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++