#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <cstdlib>

#define M 10000
#define K 10000
#define N 10000

// Default block dimensions; can be overridden via -DBLOCK_SIZE=... at compile time
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

void checkCUDAError(const char*);
void random_ints(int *a, int size);

static inline void checkCuda(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// Tiled matrix multiplication with compile-time BLOCK_SIZE
__global__ void matrixMult(int *a, int *b, int *c, int m, int k, int n) {
	//* 2D indexing

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * BLOCK_SIZE + ty;
	int col = blockIdx.x * BLOCK_SIZE + tx;

	__shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

	int acc = 0;

	int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

	for (int t = 0; t < numTiles; t++) {

		int aCol = t * BLOCK_SIZE + tx;
		int bRow = t * BLOCK_SIZE + ty;

		if (aCol < k && row < m) {
			As[ty][tx] = a[row * k + aCol];
		}
		else{
			As[ty][tx] = 0;
		}

		if (bRow < k && col < n) {
			Bs[ty][tx] = b[bRow * n + col];
		}
		else{
			Bs[ty][tx] = 0;
		}

		__syncthreads();

		for (int i = 0; i < BLOCK_SIZE; i++) {
			acc += As[ty][i] * Bs[i][tx];
		}

		// Ensure all threads are done using the current tile before loading the next one
		__syncthreads();
	}

	if (row < m && col < n) {
		c[row * n + col] = acc;
	}
}


__host__ void validate(int *c_ref, int *c) {
	int errors = 0;
	bool foundError = false;
	for (int i=0; i < N * M; i++){
		if (c_ref[i] != c[i])
		{
			if (foundError == false) {
				foundError = true;
				printf("Validate error at idx: %d\n", i);
				errors++;
			}
			else {
				errors++;
			}
		};
		
	};
	printf("\nFound %d total errors.\n", errors);
}

/* A = (m, k), B = (k, n), C = (m, n) */
__host__ void matrixMultCPU(int *a, int *b, int *c, int m, int k, int n) {
    //* Parallelizing with OpenMP so I don't have to wait as long. Results will be the EXACT same as there is no thread contention.
    int num_threads = omp_get_max_threads();  // or a constant like 8, 16
	omp_set_num_threads(num_threads);
	printf("Parallelizing with %d threads\n", num_threads);
	#pragma omp parallel for
	for (int i=0; i < m; i++){
		for (int j=0; j < n; j++){
			c[i*n+j] = 0;
			for (int t = 0; t < k; t++){
				c[i*n+j] += a[i*k+t] * b[t*n+j];
			}
		}
	}
}

int main(int argc, char** argv) {
	int *a, *b, *c, *c_ref;			    // host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	unsigned int a_size = M * K * sizeof(int);
	unsigned int b_size = K * N * sizeof(int);
	unsigned int c_size = M * N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, a_size);
	cudaMalloc((void **)&d_b, b_size);
	cudaMalloc((void **)&d_c, c_size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(a_size); random_ints(a, a_size / sizeof(int));
	b = (int *)malloc(b_size); random_ints(b, b_size / sizeof(int));
	c = (int *)malloc(c_size);
	c_ref = (int *)malloc(c_size);

    cudaEvent_t full_start, full_stop;
    checkCuda(cudaEventCreate(&full_start), "cudaEventCreate(full_start)");
    checkCuda(cudaEventCreate(&full_stop), "cudaEventCreate(full_stop)");
    checkCuda(cudaEventRecord(full_start), "cudaEventRecord(full_start)");

	// Copy inputs to device
	cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch matrixMult() kernel on GPU
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
	                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	printf("Launch config: grid=(%u,%u) block=(%u,%u)\n",
	       blocksPerGrid.x, blocksPerGrid.y,
	       threadsPerBlock.x, threadsPerBlock.y);

    //* Time kernel execution
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

	matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N); 
	checkCUDAError("CUDA kernel launch");

    checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    printf("Kernel execution time: %f milliseconds\n", milliseconds);
    printf("GFLOPS: %f\n", (2.0 * M * K * N) / 1e9 / (milliseconds / 1e3));
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, c_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

    checkCuda(cudaEventRecord(full_stop), "cudaEventRecord(full_stop)");
    checkCuda(cudaEventSynchronize(full_stop), "cudaEventSynchronize(full_stop)");
    float full_milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&full_milliseconds, full_start, full_stop), "cudaEventElapsedTime(full_milliseconds)");
    printf("Full execution time (including memcpy): %f milliseconds\n", full_milliseconds);
    checkCuda(cudaEventDestroy(full_start), "cudaEventDestroy(full_start)");
    checkCuda(cudaEventDestroy(full_stop), "cudaEventDestroy(full_stop)");

    //* Time CPU execution
    auto t0 = std::chrono::high_resolution_clock::now();
	matrixMultCPU(a, b, c_ref, M, K, N);
    auto t1= std::chrono::high_resolution_clock::now();
    float cpu_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("CPU execution time: %f milliseconds\n", cpu_milliseconds);
    printf("GFLOPS: %f\n", (2.0 * M * K * N) / 1e9 / (cpu_milliseconds / 1e3));
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
void random_ints(int *a, int size)
{
	for (unsigned int i = 0; i < size; i++){
		a[i] = rand();
	}
}
