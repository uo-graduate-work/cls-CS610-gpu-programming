#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>
#include <iostream>

#define TILE_SIZE 256

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
	if (length == 0) return;
	for (int i = 0; i < length; ++i) {
		data[i] = (float)rand() / RAND_MAX;
	}
	std::sort(data, data + length);
}

__device__ unsigned int findCoRank(unsigned int k, float * A, unsigned int m, float * B, unsigned int n) {
	unsigned int i = (k > m) ? m : k;
	unsigned int j = k - i;
	
	while (i > 0 && j < n && A[i-1] > B[j]) {
		unsigned int diff = (unsigned int)((A[i-1] - B[j]) * 1000);
		unsigned int step = (diff > i * 1000) ? i / 2 : 1;
		j += step;
		i -= step;
	}
	
	if (i > 0 && j < n && A[i-1] > B[j]) {
		i = 0;
	} else {
		while (i < m && j > 0 && B[j-1] >= A[i]) {
			i++;
			j--;
		}
	}
	
	return i;
}

__global__ void GPU_merge(float * A, unsigned int m, float * B, unsigned int n, float * C, unsigned int tile_size) {
	__shared__ float s_A[256];
	__shared__ float s_B[256];
	
	unsigned int tid = threadIdx.x;
	unsigned int total = m + n;
	unsigned int c_idx = blockIdx.x * tile_size;
	
	unsigned int a_pos = 0;
	unsigned int b_pos = 0;
	
	while (c_idx < total) {
		unsigned int k = tile_size;
		if (c_idx + k > total) k = total - c_idx;
		
		unsigned int a_end = a_pos + k;
		if (a_end > m) a_end = m;
		unsigned int b_end = b_pos + k;
		if (b_end > n) b_end = n;
		
		unsigned int a_count = a_end - a_pos;
		unsigned int b_count = b_end - b_pos;
		
		if (tid < a_count) {
			s_A[tid] = A[a_pos + tid];
		}
		if (tid < b_count) {
			s_B[tid] = B[b_pos + tid];
		}
		__syncthreads();
		
		if (tid < k) {
			unsigned int a_local = findCoRank(tid, s_A, a_count, s_B, b_count);
			unsigned int b_local = tid - a_local;
			
			float a_val = (a_local < a_count) ? s_A[a_local] : 1e10f;
			float b_val = (b_local < b_count) ? s_B[b_local] : 1e10f;
			
			C[c_idx + tid] = (a_val <= b_val) ? a_val : b_val;
		}
		__syncthreads();
		
		unsigned int consumed_a = 0, consumed_b = 0;
		if (tid == 0) {
			unsigned int last_a_local = findCoRank(k-1, s_A, a_count, s_B, b_count);
			unsigned int last_b_local = (k-1) - last_a_local;
			consumed_a = last_a_local + 1;
			consumed_b = last_b_local + 1;
			if (last_a_local == a_count) consumed_a = a_count;
			if (last_b_local == b_count) consumed_b = b_count;
			a_pos += consumed_a;
			b_pos += consumed_b;
		}
		__syncthreads();
		
		c_idx += tile_size;
	}
}

void CPU_merge(float * A, unsigned int m, float * B, unsigned int n, float * C) {
	unsigned int i = 0, j = 0, k = 0;
	while (i < m && j < n) {
		if (A[i] <= B[j]) {
			C[k++] = A[i++];
		} else {
			C[k++] = B[j++];
		}
	}
	while (i < m) C[k++] = A[i++];
	while (j < n) C[k++] = B[j++];
}

/* Host code */
int main(void) {
	unsigned int test_sizes[][2] = {
		{20, 20},     // Original - evenly divisible
		{1, 50},      // Edge case: very small A
		{50, 1},      // Edge case: very small B
		{0, 50},      // Edge case: empty A
		{50, 0},      // Edge case: empty B
	};
	
	int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
	
	for (int t = 0; t < num_tests; t++) {
		unsigned int input_length_A = test_sizes[t][0];
		unsigned int input_length_B = test_sizes[t][1];
		unsigned int output_length_C = input_length_A + input_length_B;
	float * h_A, * d_A, * h_B, * d_B, * h_C, * d_C;
	cudaEvent_t start, stop;
	float ms;

	unsigned int data_size_A = input_length_A * sizeof(float);
	unsigned int data_size_B = input_length_B * sizeof(float);
	unsigned int data_size_C = output_length_C * sizeof(float);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&d_A, data_size_A);
	cudaMalloc((void**)&d_B, data_size_B);
	cudaMalloc((void**)&d_C, data_size_C);
	checkCUDAError("CUDA malloc");

	h_A = (float*)malloc(data_size_A);
	h_B = (float*)malloc(data_size_B);
	h_C = (float*)malloc(data_size_C);
	generate_data(h_A, input_length_A);
	generate_data(h_B, input_length_B);

	cudaMemcpy(d_A, h_A, data_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, data_size_B, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");
	
	cudaEventRecord(start, 0);
	
	GPU_merge<<<1, TILE_SIZE>>>(d_A, input_length_A, d_B, input_length_B, d_C, TILE_SIZE);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel normal");


	cudaMemcpy(h_C, d_C, data_size_C, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	printf("Execution time:\t%f\n", ms);

	float * cpu_C = (float*)malloc(data_size_C);
	CPU_merge(h_A, input_length_A, h_B, input_length_B, cpu_C);
	
	int verified = 1;
	for (unsigned int i = 0; i < output_length_C; i++) {
		if (h_C[i] != cpu_C[i]) {
			verified = 0;
			break;
		}
	}
	printf("Verification: %s\n", verified ? "PASSED" : "FAILED");
	
	printf("A: ");
	for (unsigned int i = 0; i < input_length_A; i++) printf("%.2f ", h_A[i]);
	printf("\n");
	printf("B: ");
	for (unsigned int i = 0; i < input_length_B; i++) printf("%.2f ", h_B[i]);
	printf("\n");
	printf("C: ");
	for (unsigned int i = 0; i < output_length_C; i++) printf("%.2f ", h_C[i]);
	printf("\n");
	
	free(cpu_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	} // end test loop

	return 0;
}