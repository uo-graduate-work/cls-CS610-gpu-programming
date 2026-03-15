#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

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

void generate_data(float * data, unsigned int length) {
	for (int i = 0; i < length; ++i) {
		data[i] = (float)rand() / RAND_MAX;
	}
}

__global__ void GPU_scan(float * X, float * Y, unsigned int length) {
	__shared__ float temp[BLOCK_SIZE];
	
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (gid < length) temp[tid] = X[gid];
	else temp[tid] = 0;
	
	__syncthreads();
	
	for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
		if (tid >= stride) {
			temp[tid] += temp[tid - stride];
		}
		__syncthreads();
	}
	
	if (gid < length) Y[gid] = temp[tid];
}

__global__ void GPU_add_block_sums(float * Y, float * block_sums, unsigned int length) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (gid < length) {
		float old_val = Y[gid];
		Y[gid] += block_sums[blockIdx.x];
		printf("Block %d: Y[%d] = %f + %f = %f\n", 
		       blockIdx.x, gid, old_val, block_sums[blockIdx.x], Y[gid]);
	}
}

void CPU_scan(float * X, float * Y, unsigned int length) {
	Y[0] = X[0];
	for (unsigned int i = 1; i < length; i++) {
		Y[i] = Y[i-1] + X[i];
	}
}

/* Host code */
int main(void) {
	unsigned int input_length = 2048;
	float * h_input, * d_input, * h_output, * d_output;
	float * h_block_sums, * d_block_sums;
	cudaEvent_t start, stop;
	float ms;

	unsigned int data_size = input_length * sizeof(float);
	unsigned int numBlocks = (input_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int block_sums_size = numBlocks * sizeof(float);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&d_input, data_size);
	cudaMalloc((void**)&d_output, data_size);
	cudaMalloc((void**)&d_block_sums, block_sums_size);
	checkCUDAError("CUDA malloc");

	h_input = (float*)malloc(data_size);
	h_output = (float*)malloc(data_size);
	h_block_sums = (float*)malloc(block_sums_size);
	generate_data(h_input, input_length);

	cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");
	
	cudaEventRecord(start, 0);
	
	GPU_scan<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, input_length);
	cudaDeviceSynchronize();
	checkCUDAError("GPU_scan");
	
	cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
	
	float * temp_buffer = (float*)malloc(data_size);
	cudaMemcpy(temp_buffer, d_output, data_size, cudaMemcpyDeviceToHost);
	
	printf("After scan (sample indices):\n");
	printf("  Y[255] = %f (block 0 last)\n", temp_buffer[255]);
	printf("  Y[511] = %f (block 1 last)\n", temp_buffer[511]);
	printf("  Y[512] = %f (block 2 first)\n", temp_buffer[512]);
	printf("  Y[767] = %f (block 2 last)\n", temp_buffer[767]);
	
	float total = 0;
	for (unsigned int i = 0; i < numBlocks; i++) {
		unsigned int block_end = (i + 1) * BLOCK_SIZE;
		if (block_end > input_length) block_end = input_length;
		unsigned int last_idx = block_end - 1;
		
		h_block_sums[i] = total;
		printf("  Block %u: last_idx=%u, Y[last]=%f, block_sum[%u]=%f\n", 
		       i, last_idx, temp_buffer[last_idx], i, total);
		total += temp_buffer[last_idx];
	}
	free(temp_buffer);
	
	cudaMemcpy(d_block_sums, h_block_sums, block_sums_size, cudaMemcpyHostToDevice);
	
	GPU_add_block_sums<<<numBlocks, BLOCK_SIZE>>>(d_output, d_block_sums, input_length);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel normal");


	cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	printf("Execution time:\t%f\n", ms);

	float * cpu_output = (float*)malloc(data_size);
	CPU_scan(h_input, cpu_output, input_length);
	
	int verified = 1;
	unsigned int fail_idx = 0;
	for (unsigned int i = 0; i < input_length; i++) {
		if (fabs(h_output[i] - cpu_output[i]) > 0.001) {
			verified = 0;
			fail_idx = i;
			break;
		}
	}
	printf("Verification: %s\n", verified ? "PASSED" : "FAILED");
	
	if (!verified) {
		printf("First failure at index %u: GPU=%f, CPU=%f\n", fail_idx, h_output[fail_idx], cpu_output[fail_idx]);
	}
	
	for (unsigned int i = 0; i < 10; i++) {
		printf("output[%u] = %f (CPU: %f)\n", i, h_output[i], cpu_output[i]);
	}
	
	free(cpu_output);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_block_sums);
	free(h_input);
	free(h_output);
	free(h_block_sums);

	return 0;
}