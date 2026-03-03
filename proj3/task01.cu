#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define K 10
#define M 10000
#define N 10000
#define P 10000

// Small dimensions for CPU verification (10K x 10K CPU matmul is infeasible)
#define VERIFY_M 256
#define VERIFY_N 256
#define VERIFY_P 256

#define BLOCK_SIZE 16

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- CPU baseline for verification ----
void cpu_matrix_multiply(float* A, float* B, float* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// ---- GPU matmul kernel ----
__global__ void matrix_multiply_kernel(float* A, float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// ---- Verify GPU kernel correctness using small matrices ----
void verify_correctness() {
    printf("=== Correctness Verification (small matrices %dx%d) ===\n", VERIFY_M, VERIFY_N);

    size_t sizeA = VERIFY_M * VERIFY_N * sizeof(float);
    size_t sizeB = VERIFY_N * VERIFY_P * sizeof(float);
    size_t sizeC = VERIFY_M * VERIFY_P * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_cpu = (float*)malloc(sizeC);
    float* h_C_gpu = (float*)malloc(sizeC);

    srand(42);
    for (int j = 0; j < VERIFY_M * VERIFY_N; j++)
        h_A[j] = (float)(rand() % 10) / 10.0f;
    for (int j = 0; j < VERIFY_N * VERIFY_P; j++)
        h_B[j] = (float)(rand() % 10) / 10.0f;

    // CPU reference
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, VERIFY_M, VERIFY_N, VERIFY_P);

    // GPU
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((VERIFY_P + BLOCK_SIZE - 1) / BLOCK_SIZE, (VERIFY_M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrix_multiply_kernel<<<grid, block>>>(d_A, d_B, d_C, VERIFY_M, VERIFY_N, VERIFY_P);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compare
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < VERIFY_M * VERIFY_P; i++) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (fabsf(h_C_cpu[i]) > 1e-5f) {
            float rel = diff / fabsf(h_C_cpu[i]);
            if (rel > max_rel) max_rel = rel;
        }
    }
    printf("Max absolute diff: %e, max relative error: %e\n", max_diff, max_rel);
    if (max_rel < 1e-3f)
        printf("PASSED: GPU kernel matches CPU reference.\n\n");
    else
        printf("FAILED: GPU kernel does NOT match CPU reference!\n\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

// ---- Without pre-fetch: serial H2D -> compute -> D2H per batch, single stream ----
void run_without_prefetch(float** h_A, float** h_B, float** h_C,
                          int m, int n, int p,
                          cudaEvent_t start, cudaEvent_t stop) {
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    // Allocate 1 set of device buffers (reuse per batch)
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int i = 0; i < K; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_A, h_A[i], sizeA, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_B, h_B[i], sizeB, cudaMemcpyHostToDevice, stream));
        matrix_multiply_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, m, n, p);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C, sizeC, cudaMemcpyDeviceToHost, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaStreamDestroy(stream);
}

// ---- With pre-fetch pipeline: overlap H2D of next batch with compute of current ----
// Uses 2 streams: one for transfers, one for compute, with events for sync.
// Ping-pong device buffers (2 sets) so next H2D doesn't clobber current compute data.
void run_with_prefetch(float** h_A, float** h_B, float** h_C,
                       int m, int n, int p,
                       cudaEvent_t start, cudaEvent_t stop) {
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    // Ping-pong: 2 sets of device buffers
    float *d_A[2], *d_B[2], *d_C[2];
    for (int s = 0; s < 2; s++) {
        CHECK_CUDA(cudaMalloc(&d_A[s], sizeA));
        CHECK_CUDA(cudaMalloc(&d_B[s], sizeB));
        CHECK_CUDA(cudaMalloc(&d_C[s], sizeC));
    }

    cudaStream_t stream_transfer, stream_compute;
    CHECK_CUDA(cudaStreamCreate(&stream_transfer));
    CHECK_CUDA(cudaStreamCreate(&stream_compute));

    // Events for synchronization between streams
    cudaEvent_t transfer_done[K], compute_done[K];
    for (int i = 0; i < K; i++) {
        CHECK_CUDA(cudaEventCreate(&transfer_done[i]));
        CHECK_CUDA(cudaEventCreate(&compute_done[i]));
    }

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CHECK_CUDA(cudaEventRecord(start, stream_transfer));

    // Pre-fetch batch 0
    int buf = 0;
    CHECK_CUDA(cudaMemcpyAsync(d_A[buf], h_A[0], sizeA, cudaMemcpyHostToDevice, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(d_B[buf], h_B[0], sizeB, cudaMemcpyHostToDevice, stream_transfer));
    CHECK_CUDA(cudaEventRecord(transfer_done[0], stream_transfer));

    for (int i = 0; i < K; i++) {
        int cur_buf = i % 2;
        int nxt_buf = (i + 1) % 2;

        // Pre-fetch next batch (overlaps with current compute)
        if (i + 1 < K) {
            // Wait for previous compute on nxt_buf to finish before overwriting
            if (i >= 1) {
                CHECK_CUDA(cudaStreamWaitEvent(stream_transfer, compute_done[i - 1], 0));
            }
            CHECK_CUDA(cudaMemcpyAsync(d_A[nxt_buf], h_A[i + 1], sizeA, cudaMemcpyHostToDevice, stream_transfer));
            CHECK_CUDA(cudaMemcpyAsync(d_B[nxt_buf], h_B[i + 1], sizeB, cudaMemcpyHostToDevice, stream_transfer));
            CHECK_CUDA(cudaEventRecord(transfer_done[i + 1], stream_transfer));
        }

        // Wait for current batch's H2D to complete before computing
        CHECK_CUDA(cudaStreamWaitEvent(stream_compute, transfer_done[i], 0));

        // Compute current batch
        matrix_multiply_kernel<<<grid, block, 0, stream_compute>>>(d_A[cur_buf], d_B[cur_buf], d_C[cur_buf], m, n, p);
        CHECK_CUDA(cudaGetLastError());

        // Copy result back (on compute stream, so it waits for kernel)
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C[cur_buf], sizeC, cudaMemcpyDeviceToHost, stream_compute));

        CHECK_CUDA(cudaEventRecord(compute_done[i], stream_compute));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream_compute));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Cleanup
    for (int i = 0; i < K; i++) {
        cudaEventDestroy(transfer_done[i]);
        cudaEventDestroy(compute_done[i]);
    }
    for (int s = 0; s < 2; s++) {
        cudaFree(d_A[s]); cudaFree(d_B[s]); cudaFree(d_C[s]);
    }
    cudaStreamDestroy(stream_transfer);
    cudaStreamDestroy(stream_compute);
}

// ---- Verify random subset of large matrices ----
void verify_large_subset(float** h_A, float** h_B, float** h_C, int m, int n, int p, int k_batches) {
    printf("=== Large Matrix Random Subset Verification ===\n");
    int num_checks = 100;
    float max_diff = 0.0f;
    float max_rel = 0.0f;

    for (int check = 0; check < num_checks; check++) {
        int b = rand() % k_batches;
        int row = rand() % m;
        int col = rand() % p;

        float cpu_val = 0.0f;
        for (int k = 0; k < n; k++) {
            cpu_val += h_A[b][row * n + k] * h_B[b][k * p + col];
        }

        float gpu_val = h_C[b][row * p + col];
        float diff = fabsf(cpu_val - gpu_val);
        if (diff > max_diff) max_diff = diff;
        
        if (fabsf(cpu_val) > 1e-5f) {
            float rel = diff / fabsf(cpu_val);
            if (rel > max_rel) max_rel = rel;
        }
    }
    printf("Checked %d random elements across all batches.\n", num_checks);
    printf("Max absolute diff: %e, max relative error: %e\n", max_diff, max_rel);
    if (max_rel < 1e-3f)
        printf("PASSED: Large matrix GPU results match CPU reference (sampled).\n\n");
    else
        printf("FAILED: Large matrix GPU results do NOT match CPU reference!\n\n");
}

int main() {
    // ---- Step 1: Verify kernel correctness with small matrices ----
    verify_correctness();

    // ---- Step 2: Benchmark with full-size matrices ----
    printf("=== Batched Matrix Multiplication Benchmark ===\n");
    printf("K=%d batches, m=%d, k=%d, n=%d\n\n", K, M, N, P);

    size_t sizeA = (size_t)M * N * sizeof(float);
    size_t sizeB = (size_t)N * P * sizeof(float);
    size_t sizeC = (size_t)M * P * sizeof(float);

    // Allocate PINNED host memory (required for true async transfers)
    float** h_A = (float**)malloc(K * sizeof(float*));
    float** h_B = (float**)malloc(K * sizeof(float*));
    float** h_C_no_prefetch = (float**)malloc(K * sizeof(float*));
    float** h_C_prefetch = (float**)malloc(K * sizeof(float*));

    printf("Allocating pinned host memory...\n");
    srand(42);
    for (int i = 0; i < K; i++) {
        CHECK_CUDA(cudaMallocHost(&h_A[i], sizeA));
        CHECK_CUDA(cudaMallocHost(&h_B[i], sizeB));
        CHECK_CUDA(cudaMallocHost(&h_C_no_prefetch[i], sizeC));
        CHECK_CUDA(cudaMallocHost(&h_C_prefetch[i], sizeC));

        for (int j = 0; j < M * N; j++)
            h_A[i][j] = (float)(rand() % 10) / 10.0f;
        for (int j = 0; j < N * P; j++)
            h_B[i][j] = (float)(rand() % 10) / 10.0f;
    }
    printf("Done.\n\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ---- Run without pre-fetch ----
    printf("Running WITHOUT pre-fetch (serial H2D -> compute -> D2H per batch)...\n");
    run_without_prefetch(h_A, h_B, h_C_no_prefetch, M, N, P, start, stop);
    float time_no_prefetch;
    CHECK_CUDA(cudaEventElapsedTime(&time_no_prefetch, start, stop));
    printf("Time without pre-fetch: %.2f ms\n\n", time_no_prefetch);

    // ---- Run with pre-fetch pipeline ----
    printf("Running WITH pre-fetch pipeline (overlap H2D of next with compute of current)...\n");
    run_with_prefetch(h_A, h_B, h_C_prefetch, M, N, P, start, stop);
    float time_prefetch;
    CHECK_CUDA(cudaEventElapsedTime(&time_prefetch, start, stop));
    printf("Time with pre-fetch: %.2f ms\n\n", time_prefetch);

    // ---- Cross-check: both GPU runs should produce same results ----
    printf("Cross-checking GPU results (prefetch vs no-prefetch)...\n");
    float max_diff = 0.0f;
    for (int b = 0; b < K; b++) {
        for (int j = 0; j < M * P; j++) {
            float diff = fabsf(h_C_no_prefetch[b][j] - h_C_prefetch[b][j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    printf("Max difference between prefetch/no-prefetch results: %e\n", max_diff);
    if (max_diff < 1e-5f)
        printf("PASSED: Both GPU methods produce identical results.\n\n");
    else
        printf("WARNING: Results differ (floating point order differences expected).\n\n");

    // ---- Step 3: Verify subset of large results against CPU ----
    verify_large_subset(h_A, h_B, h_C_prefetch, M, N, P, K);

    // ---- Summary ----
    printf("=== Performance Summary ===\n");
    printf("Without pre-fetch: %.2f ms\n", time_no_prefetch);
    printf("With pre-fetch:    %.2f ms\n", time_prefetch);
    printf("Speedup:           %.2fx\n", time_no_prefetch / time_prefetch);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    for (int i = 0; i < K; i++) {
        cudaFreeHost(h_A[i]);
        cudaFreeHost(h_B[i]);
        cudaFreeHost(h_C_no_prefetch[i]);
        cudaFreeHost(h_C_prefetch[i]);
    }
    free(h_A); free(h_B);
    free(h_C_no_prefetch); free(h_C_prefetch);

    return 0;
}
