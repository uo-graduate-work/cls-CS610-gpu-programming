#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//* Bounds check
	if (i >= max) return;
	//* Fixed, was minus.
	c[i] = a[i] + b[i];
}

__host__ void vectorAddCPU(int *a, int *b, int *c, int max) {
	for (int i=0; i < N; i++){
		c[i] = a[i] + b[i];
	};
}

__host__ void validate(int *c_ref, int *c) {
	int errors = 0;
	for (int i=0; i < N; i++){
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
	unsigned int size = N * sizeof(int);

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
	vectorAdd <<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(d_a, d_b, d_c, N);

	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	vectorAddCPU(a, b, c_ref, N);
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

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}
