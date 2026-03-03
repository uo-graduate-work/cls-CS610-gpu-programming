#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#define IMAGE_DIM 2048

using uchar = unsigned char;

void output_image_file(uchar* image);
void output_image_file_named(const char* filename, uchar* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);

__global__ void image_to_grayscale(uchar3 *image, uchar *image_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = image[idx];
        double gray = pixel.x * 0.21 + pixel.y * 0.72 + pixel.z * 0.07;
        image_output[idx] = (uchar)gray;
    }
}

__global__ void image_to_grayscale_soa(uchar *r_channel, uchar *g_channel, uchar *b_channel, 
                                        uchar *image_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar r = r_channel[idx];
        uchar g = g_channel[idx];
        uchar b = b_channel[idx];
        double gray = r * 0.21 + g * 0.72 + b * 0.07;
        image_output[idx] = (uchar)gray;
    }
}

int main(void) {
    unsigned int image_size, image_output_size;
    uchar3 *d_image, *h_image;
    uchar  *d_image_output, *h_image_output;
    cudaEvent_t start, stop;
    float ms_aos = 0.0f, ms_soa = 0.0f;

    image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar3);
    image_output_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_image = (uchar3*)malloc(image_size);
    h_image_output = (uchar*)malloc(image_output_size);
    input_image_file("input.ppm", h_image);

    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_image_output, image_output_size);
    checkCUDAError("CUDA malloc");

    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy to device");

    dim3 blockAos(16, 16);
    dim3 gridAos((IMAGE_DIM + blockAos.x - 1) / blockAos.x, (IMAGE_DIM + blockAos.y - 1) / blockAos.y);

    // Warm-up launch to eliminate GPU context initialization from timing
    image_to_grayscale<<<gridAos, blockAos>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    image_to_grayscale<<<gridAos, blockAos>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_aos, start, stop);
    checkCUDAError("Kernel launch (AoS)");

    cudaMemcpy(h_image_output, d_image_output, image_output_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device (AoS)");

    output_image_file_named("output_aos.ppm", h_image_output);

    printf("Execution time (AoS - Array of Structures): %f ms\n", ms_aos);

    uchar *d_r, *d_g, *d_b;
    uchar *h_r = (uchar*)malloc(image_output_size);
    uchar *h_g = (uchar*)malloc(image_output_size);
    uchar *h_b = (uchar*)malloc(image_output_size);

    cudaMalloc(&d_r, image_output_size);
    cudaMalloc(&d_g, image_output_size);
    cudaMalloc(&d_b, image_output_size);
    checkCUDAError("CUDA malloc (SoA)");

    for (int y = 0; y < IMAGE_DIM; y++) {
        for (int x = 0; x < IMAGE_DIM; x++) {
            int idx = y * IMAGE_DIM + x;
            h_r[idx] = h_image[idx].x;
            h_g[idx] = h_image[idx].y;
            h_b[idx] = h_image[idx].z;
        }
    }

    cudaMemcpy(d_r, h_r, image_output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, image_output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, image_output_size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy to device (SoA)");

    dim3 blockSoa(16, 16);
    dim3 gridSoa((IMAGE_DIM + blockSoa.x - 1) / blockSoa.x, (IMAGE_DIM + blockSoa.y - 1) / blockSoa.y);

    cudaEventRecord(start);
    image_to_grayscale_soa<<<gridSoa, blockSoa>>>(d_r, d_g, d_b, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_soa, start, stop);
    checkCUDAError("Kernel launch (SoA)");

    cudaMemcpy(h_image_output, d_image_output, image_output_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device (SoA)");

    output_image_file_named("output_soa.ppm", h_image_output);
    output_image_file(h_image_output);

    printf("Execution time (SoA - Structure of Arrays): %f ms\n", ms_soa);
    printf("Speedup (AoS -> SoA): %.2fx\n", ms_aos / ms_soa);

    cudaFree(d_image);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_image_output);
    checkCUDAError("CUDA free");

    free(h_image);
    free(h_image_output);
    free(h_r);
    free(h_g);
    free(h_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

void output_image_file(uchar* image)
{
    output_image_file_named("output.ppm", image);
}

void output_image_file_named(const char* filename, uchar* image)
{
    FILE *f;
    f = fopen(filename, "wb");
    if (f == NULL){
        fprintf(stderr, "Error opening '%s' output file\n", filename);
        exit(1);
    }
    fprintf(f, "P5\n");
    fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
    for (int x = 0; x < IMAGE_DIM; x++){
        for (int y = 0; y < IMAGE_DIM; y++){
            int i = x + y*IMAGE_DIM;
            fwrite(&image[i], sizeof(unsigned char), 1, f);
        }
    }
    fclose(f);
}

void input_image_file(char* filename, uchar3* image)
{
    FILE *f;
    char temp[256];
    unsigned int x, y, s;

    f = fopen("input.ppm", "rb");
    if (f == NULL){
        fprintf(stderr, "Error opening 'input.ppm' input file\n");
        exit(1);
    }
    fscanf(f, "%s\n", temp);
    fscanf(f, "%d %d\n", &x, &y);
    fscanf(f, "%d\n",&s);
    if ((x != y) && (x != IMAGE_DIM)){
        fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
        exit(1);
    }

    for (int x = 0; x < IMAGE_DIM; x++){
        for (int y = 0; y < IMAGE_DIM; y++){
            int i = x + y*IMAGE_DIM;
            fread(&image[i], sizeof(unsigned char), 3, f);
        }
    }

    fclose(f);
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
