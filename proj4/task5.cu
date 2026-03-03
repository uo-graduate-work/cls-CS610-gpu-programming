#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// include kernels and cuda headers after definitions of structures
#include "task5_kernels.cuh"


void checkCUDAError(const char*);
void readRecords(student_record *records);
void studentRecordAOS2SOA(student_record *aos, student_records *soa);
void cpuValidation(student_records *h_records, float gpu_max_mark, int gpu_max_student_id, const char *method_name);
void maximumMark_atomic(student_records*, student_records*, student_records*, student_records*);
void maximumMark_recursive(student_records*, student_records*, student_records*, student_records*);
void maximumMark_SM(student_records*, student_records*, student_records*, student_records*);
void maximumMark_shuffle(student_records*, student_records*, student_records*, student_records*);


int main(void) {
	student_record *recordsAOS;
	student_records *h_records;
	student_records *h_records_result;
	student_records *d_records;
	student_records *d_records_result;

	//host allocation
	recordsAOS = (student_record*)malloc(sizeof(student_record)*NUM_RECORDS);
	h_records = (student_records*)malloc(sizeof(student_records));
	h_records_result = (student_records*)malloc(sizeof(student_records));

	//device allocation
	cudaMalloc((void**)&d_records, sizeof(student_records));
	cudaMalloc((void**)&d_records_result, sizeof(student_records));
	checkCUDAError("CUDA malloc");

	//read file
	readRecords(recordsAOS);
	studentRecordAOS2SOA(recordsAOS, h_records);

	//free AOS as it is no longer needed
	free(recordsAOS);

	//apply each approach in turn
	maximumMark_atomic(h_records, h_records_result, d_records, d_records_result);
	maximumMark_recursive(h_records, h_records_result, d_records, d_records_result);
	maximumMark_SM(h_records, h_records_result, d_records, d_records_result);
	maximumMark_shuffle(h_records, h_records_result, d_records, d_records_result);


	// Cleanup
	free(h_records);
	free(h_records_result);
	cudaFree(d_records);
	cudaFree(d_records_result);
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

void readRecords(student_record *records){
	FILE *f = NULL;
	f = fopen("Student_large.dat", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find file \n");
		exit(1);
	}

	//read student data
	if (fread(records, sizeof(student_record), NUM_RECORDS, f) != NUM_RECORDS){
		fprintf(stderr, "Error: Unexpected end of file!\n");
		exit(1);
	}
	fclose(f);
}

// Task 0.1) Convert the array of structures (AOS) into structure of arrays (SOA) format.
//
// AOS layout: each element is a student_record { student_id, assignment_mark }
//   Memory: [id0, mark0, id1, mark1, id2, mark2, ...]
//
// SOA layout: student_records has separate arrays for IDs and marks
//   Memory: [id0, id1, id2, ...] [mark0, mark1, mark2, ...]
//
// SOA is better for GPU coalesced memory access because threads in a warp
// access consecutive elements of the SAME field, which maps to consecutive
// memory addresses. With AOS, threads would access strided memory locations.
void studentRecordAOS2SOA(student_record *aos, student_records *soa){
	for (int i = 0; i < NUM_RECORDS; i++) {
		soa->student_ids[i] = aos[i].student_id;
		soa->assignment_marks[i] = aos[i].assignment_mark;
	}
}

// Task 1.5) CPU version validation function
// Computes the maximum mark sequentially on the CPU, then compares
// the GPU result against the CPU result and prints PASS or FAIL.
void cpuValidation(student_records *h_records, float gpu_max_mark, int gpu_max_student_id, const char *method_name){
	float cpu_max = 0;
	int cpu_max_id = 0;
	for (int i = 0; i < NUM_RECORDS; i++) {
		if (h_records->assignment_marks[i] > cpu_max) {
			cpu_max = h_records->assignment_marks[i];
			cpu_max_id = h_records->student_ids[i];
		}
	}
	// Compare GPU result against CPU ground truth
	if (gpu_max_mark == cpu_max && gpu_max_student_id == cpu_max_id) {
		printf("\t%s PASSED validation (CPU: mark=%f, student=%d)\n", method_name, cpu_max, cpu_max_id);
	} else {
		// Mark still matches but student ID may differ (e.g. atomics non-deterministic tie-breaking)
		if (gpu_max_mark == cpu_max) {
			printf("\t%s PASSED (mark matches CPU=%f, student IDs differ: GPU=%d CPU=%d -- tie-breaking)\n",
				method_name, cpu_max, gpu_max_student_id, cpu_max_id);
		} else {
			printf("\t%s FAILED validation! GPU: mark=%f student=%d, CPU: mark=%f student=%d\n",
				method_name, gpu_max_mark, gpu_max_student_id, cpu_max, cpu_max_id);
		}
	}
}


void maximumMark_atomic(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0;
	max_mark_student_id = 0;

	// Reset device variables to ensure clean state
	float zero_f = 0.0f;
	int zero_i = 0;
	cudaMemcpyToSymbol(d_max_mark, &zero_f, sizeof(float));
	cudaMemcpyToSymbol(d_max_mark_student_id, &zero_i, sizeof(int));
	cudaMemcpyToSymbol(d_lock, &zero_i, sizeof(int));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Atomics: CUDA memcpy");

	cudaEventRecord(start, 0);

	// Task 1.2) Configure the kernel
	// NUM_RECORDS threads total, THREADS_PER_BLOCK per block
	int num_blocks = NUM_RECORDS / THREADS_PER_BLOCK;

	// Task 1.3) Launch and synchronize the kernel
	maximumMark_atomic_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_records);
	cudaDeviceSynchronize();
	checkCUDAError("Atomics: kernel launch");

	// Task 1.4) Copy result back to host using cudaMemcpyFromSymbol
	// These are __device__ global variables, not regular device memory,
	// so we must use cudaMemcpyFromSymbol instead of cudaMemcpy.
	cudaMemcpyFromSymbol(&max_mark, d_max_mark, sizeof(float));
	cudaMemcpyFromSymbol(&max_mark_student_id, d_max_mark_student_id, sizeof(int));
	checkCUDAError("Atomics: cudaMemcpyFromSymbol");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// Task 1.5) Validate GPU result against CPU validation function
	//output result
	printf("Atomics: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);
	cpuValidation(h_records, max_mark, max_mark_student_id, "Atomics");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Task 2)
void maximumMark_recursive(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	int i;
	float max_mark;
	int max_mark_student_id;
	student_records *d_records_temp;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0;
	max_mark_student_id = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Recursive: CUDA memcpy");

	cudaEventRecord(start, 0);

	//Task 2.3) Recursively call GPU steps until there are THREADS_PER_BLOCK values left
	//
	// Each kernel call halves the data: N threads produce N/2 outputs (even threads
	// compare with their neighbor and write the max compactly).
	//
	// For NUM_RECORDS=2048, THREADS_PER_BLOCK=256:
	//   Call 1: 2048 input -> 8 blocks -> 1024 output
	//   Call 2: 1024 input -> 4 blocks ->  512 output
	//   Call 3:  512 input -> 2 blocks ->  256 output  (= THREADS_PER_BLOCK, stop)
	//
	// After each call, swap the input/output pointers so the output
	// of the current call becomes the input of the next call.
	int num_records_current = NUM_RECORDS;
	while (num_records_current > THREADS_PER_BLOCK) {
		int num_blocks = num_records_current / THREADS_PER_BLOCK;
		maximumMark_recursive_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_records, d_records_result);
		cudaDeviceSynchronize();
		checkCUDAError("Recursive: kernel launch");

		num_records_current /= 2;

		// Swap pointers: output becomes the new input
		d_records_temp = d_records;
		d_records = d_records_result;
		d_records_result = d_records_temp;
	}

	//Task 2.4) copy back the final THREADS_PER_BLOCK values
	// After the loop, d_records points to the buffer with the final 256 values
	cudaMemcpy(h_records_result, d_records, sizeof(student_records), cudaMemcpyDeviceToHost);
	checkCUDAError("Recursive: CUDA memcpy back");

	//Task 2.5) reduce the final THREADS_PER_BLOCK values on CPU
	for (i = 0; i < THREADS_PER_BLOCK; i++) {
		if (h_records_result->assignment_marks[i] > max_mark) {
			max_mark = h_records_result->assignment_marks[i];
			max_mark_student_id = h_records_result->student_ids[i];
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output the result
	printf("Recursive: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);
	cpuValidation(h_records, max_mark, max_mark_student_id, "Recursive");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Task 3)
void maximumMark_SM(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0;
	max_mark_student_id = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("SM: CUDA memcpy");

	cudaEventRecord(start, 0);

	//Task 3.4) Call the shared memory reduction kernel
	// Single kernel launch: each block fully reduces its THREADS_PER_BLOCK
	// values down to 1 result using tree-based parallel reduction in shared memory.
	// Output has num_blocks values (one per block).
	int num_blocks = NUM_RECORDS / THREADS_PER_BLOCK;
	maximumMark_SM_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_records, d_records_result);
	cudaDeviceSynchronize();
	checkCUDAError("SM: kernel launch");

	//Task 3.5) Copy the final block values back to CPU
	cudaMemcpy(h_records_result, d_records_result, sizeof(student_records), cudaMemcpyDeviceToHost);
	checkCUDAError("SM: CUDA memcpy back");

	//Task 3.6) Reduce the block level results on CPU
	// We have num_blocks values (one maximum per block). Find the overall maximum.
	for (i = 0; i < (unsigned int)num_blocks; i++) {
		if (h_records_result->assignment_marks[i] > max_mark) {
			max_mark = h_records_result->assignment_marks[i];
			max_mark_student_id = h_records_result->student_ids[i];
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("SM: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);
	cpuValidation(h_records, max_mark, max_mark_student_id, "SM");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Task 4)
void maximumMark_shuffle(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	unsigned int warps_per_grid;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0;
	max_mark_student_id = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Shuffle: CUDA memcpy");

	cudaEventRecord(start, 0);

	//Task 4.2) Execute the kernel, copy back result, reduce final values on CPU
	//
	// The shuffle kernel reduces within each warp (32 threads) using
	// __shfl_down_sync -- no shared memory needed.
	// Each warp produces 1 result, so output has warps_per_grid values.
	int num_blocks = NUM_RECORDS / THREADS_PER_BLOCK;
	warps_per_grid = NUM_RECORDS / 32;

	maximumMark_shuffle_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_records, d_records_result);
	cudaDeviceSynchronize();
	checkCUDAError("Shuffle: kernel launch");

	// Copy warp-level results back to host
	cudaMemcpy(h_records_result, d_records_result, sizeof(student_records), cudaMemcpyDeviceToHost);
	checkCUDAError("Shuffle: CUDA memcpy back");

	// Reduce the per-warp results on CPU
	for (i = 0; i < warps_per_grid; i++) {
		if (h_records_result->assignment_marks[i] > max_mark) {
			max_mark = h_records_result->assignment_marks[i];
			max_mark_student_id = h_records_result->student_ids[i];
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("Shuffle: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);
	cpuValidation(h_records, max_mark, max_mark_student_id, "Shuffle");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
