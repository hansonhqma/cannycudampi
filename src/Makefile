all: main.c canny.cu
	mpixlc -O3 main.c -c -o main.o
	nvcc -O3 -arch=sm_70 canny.cu -c -o canny.o --expt-relaxed-constexpr
	mpicc -O3 main.o canny.o -o canny \
	-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++