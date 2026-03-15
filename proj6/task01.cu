#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#define HISTO_SIZE 256
#define BLOCK_SIZE 256

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void generate_data(char * data, unsigned int length) {
	for (int i = 0; i < length; ++i) {
		data[i] = (char)rand() % HISTO_SIZE;
	}
}

__global__ void GPU_histogram(char * data, unsigned int length, unsigned int* histo) {
	__shared__ unsigned int temp[HISTO_SIZE];
	
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < HISTO_SIZE) {
		temp[tid] = 0;
	}
	__syncthreads();
	
	if (gid < length) {
		atomicAdd(&temp[(unsigned char)data[gid]], 1);
	}
	__syncthreads();
	
	if (tid < HISTO_SIZE) {
		atomicAdd(&histo[tid], temp[tid]);
	}
}

void CPU_histogram(char * data, unsigned int length, unsigned int* histo) {
	memset(histo, 0, HISTO_SIZE * sizeof(unsigned int));
	for (unsigned int i = 0; i < length; i++) {
		histo[(unsigned char)data[i]]++;
	}
}

/* Host code */
int main(void) {
	unsigned int input_length = 2048;
	char * h_data, * d_data;
	unsigned int * h_histo, * d_histo;
	cudaEvent_t start, stop;
	float ms;

	unsigned int data_size = input_length * sizeof(char);
	unsigned int histo_size = HISTO_SIZE * sizeof(unsigned int);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&d_data, data_size);
	cudaMalloc((void**)&d_histo, histo_size);
	cudaMemset(d_histo, 0, histo_size);
	checkCUDAError("CUDA malloc");

	h_data = (char*)malloc(data_size);
	h_histo = (unsigned int*)malloc(histo_size);
	generate_data(h_data, input_length);

	cudaMemcpy(d_data, h_data, input_length, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");
	
	cudaEventRecord(start, 0);
	
	unsigned int numBlocks = (input_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
	GPU_histogram<<<numBlocks, BLOCK_SIZE>>>(d_data, input_length, d_histo);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel normal");


	cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	printf("Execution time:\t%f\n", ms);

	unsigned int * cpu_histo = (unsigned int*)malloc(histo_size);
	CPU_histogram(h_data, input_length, cpu_histo);
	
	int verified = 1;
	for (int i = 0; i < HISTO_SIZE; i++) {
		if (h_histo[i] != cpu_histo[i]) {
			verified = 0;
			break;
		}
	}
	printf("Verification: %s\n", verified ? "PASSED" : "FAILED");
	
	for (int i = 0; i < 10; i++) {
		printf("histo[%d] = %u (CPU: %u)\n", i, h_histo[i], cpu_histo[i]);
	}
	
	free(cpu_histo);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_data);
	cudaFree(d_histo);
	free(h_data);
	free(h_histo);

	return 0;
}