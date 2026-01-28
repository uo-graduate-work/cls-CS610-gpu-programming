#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 2048
#define M 1024
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__global__ void matrixAdd(int *a, int *b, int *c, int n, int m) {
	//* 2D indexing
	int col = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                                
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//* Bounds check
	if (col >= m || row >= n) return;

	int idx = row * m + col;
	c[idx] = a[idx] + b[idx];
}

__host__ void matrixAddCPU(int *a, int *b, int *c, int max) {
	for (int i=0; i < N * M; i++){
		c[i] = a[i] + b[i];
	};
}

__host__ void validate(int *c_ref, int *c) {
	int errors = 0;
	for (int i=0; i < N * M; i++){
		if (c_ref[i] != c[i])
		{
			printf("Validate error at idx: %d\n", i);
			errors++;
		};
	};
	printf("\nFound %d total errors.\n", errors);
}

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * M * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	dim3 threadsPerBlock(16, 16);                                                                                                          
	dim3 blocksPerGrid((M+15)/16, (N+15)/16);                                                                                                                         
	matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, M); 

	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	matrixAddCPU(a, b, c_ref, N);
	validate(c_ref, c);

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

//* Generate NxM matrix
void random_ints(int *a)
{
	for (unsigned int i = 0; i < N * M; i++){
		a[i] = rand();
	}
}
