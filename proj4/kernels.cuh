#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

#define NUM_RECORDS 2048
#define THREADS_PER_BLOCK 256


struct student_record{
	int student_id;
	float assignment_mark;
};

struct student_records{
	int student_ids[NUM_RECORDS];
	float assignment_marks[NUM_RECORDS];
};

typedef struct student_record student_record;
typedef struct student_records student_records;

__device__ float d_max_mark = 0;
__device__ int d_max_mark_student_id = 0;
__device__ int d_lock = 0;


// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float mark = d_records->assignment_marks[idx];
	int id = d_records->student_ids[idx];

	// Task 1.1) Use atomicCAS function to create a critical section
	//           that updates d_max_mark and d_max_mark_student_id
	//
	// Strategy: Use atomicCAS on a lock variable to implement a spin-lock.
	// Only one thread at a time can enter the critical section.
	// Inside, we compare the thread's mark with the global max and update if larger.
	bool done = false;
	while (!done) {
		// Try to acquire the lock: atomicCAS returns the OLD value at &d_lock.
		// If old value was 0 (unlocked), it sets d_lock to 1 (locked) and returns 0.
		if (atomicCAS(&d_lock, 0, 1) == 0) {
			// --- Critical section: we hold the lock ---
			if (mark > d_max_mark) {
				d_max_mark = mark;
				d_max_mark_student_id = id;
			}
			// Ensure writes to d_max_mark and d_max_mark_student_id are visible
			// to all threads before we release the lock
			__threadfence();
			// Release the lock atomically
			atomicExch(&d_lock, 0);
			done = true;
		}
	}
}

//Task 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Task 2.1) Load a single student record into shared memory
	// Using student_record (AOS) in shared memory -- no penalty for AOS in shared mem
	__shared__ student_record shared_records[THREADS_PER_BLOCK];
	shared_records[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	shared_records[threadIdx.x].student_id = d_records->student_ids[idx];

	__syncthreads();

	//Task 2.2) Compare two values and write the result to d_reduced_records
	// Even threads compare their value with the proceeding (next) thread's value.
	// Write the maximum to d_reduced_records compactly at position idx/2,
	// resulting in an output array with half the number of values.
	if (threadIdx.x % 2 == 0) {
		int out_idx = idx / 2;
		if (shared_records[threadIdx.x].assignment_mark >= shared_records[threadIdx.x + 1].assignment_mark) {
			d_reduced_records->assignment_marks[out_idx] = shared_records[threadIdx.x].assignment_mark;
			d_reduced_records->student_ids[out_idx] = shared_records[threadIdx.x].student_id;
		} else {
			d_reduced_records->assignment_marks[out_idx] = shared_records[threadIdx.x + 1].assignment_mark;
			d_reduced_records->student_ids[out_idx] = shared_records[threadIdx.x + 1].student_id;
		}
	}
}


//Task 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Task 3.1) Load a single student record into shared memory
	__shared__ student_record shared_records[THREADS_PER_BLOCK];
	shared_records[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	shared_records[threadIdx.x].student_id = d_records->student_ids[idx];

	__syncthreads();

	//Task 3.2) Reduce in shared memory in parallel
	// Classic tree-based parallel reduction:
	// At each step, threads in the first half compare with corresponding
	// threads in the second half, halving the active set each iteration.
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			if (shared_records[threadIdx.x + stride].assignment_mark > shared_records[threadIdx.x].assignment_mark) {
				shared_records[threadIdx.x] = shared_records[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}

	//Task 3.3) Write the result
	// Thread 0 in each block writes the block's maximum to d_reduced_records.
	// Result array contains one record per thread block.
	if (threadIdx.x == 0) {
		d_reduced_records->assignment_marks[blockIdx.x] = shared_records[0].assignment_mark;
		d_reduced_records->student_ids[blockIdx.x] = shared_records[0].student_id;
	}
}

//Task 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Task 4.1) Complete the kernel
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = idx / 32;       // Global warp index
	int lane_id = threadIdx.x % 32; // Lane within the warp (0-31)

	// Each thread loads its student record from global memory
	float mark = d_records->assignment_marks[idx];
	int id = d_records->student_ids[idx];

	// Warp shuffle reduction: use __shfl_down_sync to exchange values
	// between threads within the same warp WITHOUT shared memory.
	// At each step, each thread gets the value from the thread 'offset'
	// lanes below it. We compare and keep the maximum.
	// offset: 16 -> 8 -> 4 -> 2 -> 1  (5 steps to reduce 32 lanes to 1)
	for (int offset = 16; offset > 0; offset >>= 1) {
		float other_mark = __shfl_down_sync(0xFFFFFFFF, mark, offset);
		int other_id = __shfl_down_sync(0xFFFFFFFF, id, offset);
		if (other_mark > mark) {
			mark = other_mark;
			id = other_id;
		}
	}

	// Lane 0 of each warp holds the warp's maximum -- write it to output
	if (lane_id == 0) {
		d_reduced_records->assignment_marks[warp_id] = mark;
		d_reduced_records->student_ids[warp_id] = id;
	}
}

#endif //KERNEL_H
