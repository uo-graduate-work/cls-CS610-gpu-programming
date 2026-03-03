#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 4194304
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void vectorAdd(int max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < max) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void) {
    int *a, *b, *c;
    int errors;
    unsigned int size = N * sizeof(int);
    cudaEvent_t start, stop;
    float kernelTime;
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Memory Clock Rate (kHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);

    float memoryClockGHz = prop.memoryClockRate / 1000000.0f;
    float memoryBusWidthGB = prop.memoryBusWidth / 8.0f;
    float theoreticalBW = memoryClockGHz * memoryBusWidthGB * 2;
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoreticalBW);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    random_ints(a);
    random_ints(b);

    cudaMemcpyToSymbol(d_a, a, size);
    cudaMemcpyToSymbol(d_b, b, size);
    checkCUDAError("CUDA memcpy to device");

    vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(N);
    checkCUDAError("CUDA kernel");

    cudaEventRecord(start, 0);
    vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);
    checkCUDAError("CUDA kernel timing");

    printf("Kernel Execution Time: %.3f ms\n", kernelTime);

    cudaMemcpyFromSymbol(c, d_c, size);
    checkCUDAError("CUDA memcpy from device");

    long long bytesRead = (long long)N * sizeof(int) * 2;
    long long bytesWritten = (long long)N * sizeof(int);
    long long totalBytes = bytesRead + bytesWritten;
    double timeSeconds = kernelTime / 1000.0;
    double measuredBW = (totalBytes / timeSeconds) / (1024.0 * 1024.0 * 1024.0);
    printf("Measured Bandwidth: %.2f GB/s\n", measuredBW);
    printf("Ratio (Measured/Theoretical): %.2f%%\n", (measuredBW / theoreticalBW) * 100.0);

    errors = 0;
    for (unsigned int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            errors++;
        }
    }
    printf("Verification: %d errors found\n", errors);

    free(a);
    free(b);
    free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
