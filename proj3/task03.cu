#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#define IMAGE_DIM 2048
#define BLOCK_SIZE 16

using uchar = unsigned char;

void output_image_file(uchar3* image);
void output_image_file_named(const char* filename, uchar3* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);

inline __device__ int wrap_index(int idx, int dim) {
    if (idx < 0) return idx + dim;
    if (idx >= dim) return idx - dim;
    return idx;
}

__global__ void image_blur_A_basic(uchar3 *image, uchar3 *image_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 1;
    float weight = 1.0f / 9.0f;

    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int wx = wrap_index(x + dx, width);
                int wy = wrap_index(y + dy, height);
                uchar3 pixel = image[wy * width + wx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

__global__ void image_blur_B_basic(uchar3 *image, uchar3 *image_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 2;
    float weight = 1.0f / 25.0f;

    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int wx = wrap_index(x + dx, width);
                int wy = wrap_index(y + dy, height);
                uchar3 pixel = image[wy * width + wx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

__global__ void image_blur_C_basic(uchar3 *image, uchar3 *image_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 4;
    float weight = 1.0f / 81.0f;

    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int wx = wrap_index(x + dx, width);
                int wy = wrap_index(y + dy, height);
                uchar3 pixel = image[wy * width + wx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

__global__ void image_blur_D_basic(uchar3 *image, uchar3 *image_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 8;
    float weight = 1.0f / 289.0f;

    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int wx = wrap_index(x + dx, width);
                int wy = wrap_index(y + dy, height);
                uchar3 pixel = image[wy * width + wx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

template <int TILE_SIZE, int RADIUS>
__global__ void image_blur_A_tiled(uchar3 *image, uchar3 *image_output, int width, int height) {
    __shared__ uchar3 tile[(TILE_SIZE + 2 * RADIUS)][(TILE_SIZE + 2 * RADIUS)];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    int tile_dim = TILE_SIZE + 2 * RADIUS;
    
    for (int j = 0; j < tile_dim; j += TILE_SIZE) {
        for (int i = 0; i < tile_dim; i += TILE_SIZE) {
            int shared_y = ty + j;
            int shared_x = tx + i;
            int img_y = wrap_index(blockIdx.y * TILE_SIZE + shared_y - RADIUS, height);
            int img_x = wrap_index(blockIdx.x * TILE_SIZE + shared_x - RADIUS, width);
            tile[shared_y][shared_x] = image[img_y * width + img_x];
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        float weight = 1.0f / 9.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                uchar3 pixel = tile[ty + RADIUS + dy][tx + RADIUS + dx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

template <int TILE_SIZE, int RADIUS>
__global__ void image_blur_B_tiled(uchar3 *image, uchar3 *image_output, int width, int height) {
    __shared__ uchar3 tile[(TILE_SIZE + 2 * RADIUS)][(TILE_SIZE + 2 * RADIUS)];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    int tile_dim = TILE_SIZE + 2 * RADIUS;
    
    for (int j = 0; j < tile_dim; j += TILE_SIZE) {
        for (int i = 0; i < tile_dim; i += TILE_SIZE) {
            int shared_y = ty + j;
            int shared_x = tx + i;
            int img_y = wrap_index(blockIdx.y * TILE_SIZE + shared_y - RADIUS, height);
            int img_x = wrap_index(blockIdx.x * TILE_SIZE + shared_x - RADIUS, width);
            tile[shared_y][shared_x] = image[img_y * width + img_x];
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        float weight = 1.0f / 25.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                uchar3 pixel = tile[ty + RADIUS + dy][tx + RADIUS + dx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

template <int TILE_SIZE, int RADIUS>
__global__ void image_blur_C_tiled(uchar3 *image, uchar3 *image_output, int width, int height) {
    __shared__ uchar3 tile[(TILE_SIZE + 2 * RADIUS)][(TILE_SIZE + 2 * RADIUS)];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    int tile_dim = TILE_SIZE + 2 * RADIUS;
    
    for (int j = 0; j < tile_dim; j += TILE_SIZE) {
        for (int i = 0; i < tile_dim; i += TILE_SIZE) {
            int shared_y = ty + j;
            int shared_x = tx + i;
            int img_y = wrap_index(blockIdx.y * TILE_SIZE + shared_y - RADIUS, height);
            int img_x = wrap_index(blockIdx.x * TILE_SIZE + shared_x - RADIUS, width);
            tile[shared_y][shared_x] = image[img_y * width + img_x];
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        float weight = 1.0f / 81.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                uchar3 pixel = tile[ty + RADIUS + dy][tx + RADIUS + dx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

template <int TILE_SIZE, int RADIUS>
__global__ void image_blur_D_tiled(uchar3 *image, uchar3 *image_output, int width, int height) {
    __shared__ uchar3 tile[(TILE_SIZE + 2 * RADIUS)][(TILE_SIZE + 2 * RADIUS)];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    int tile_dim = TILE_SIZE + 2 * RADIUS;
    
    for (int j = 0; j < tile_dim; j += TILE_SIZE) {
        for (int i = 0; i < tile_dim; i += TILE_SIZE) {
            int shared_y = ty + j;
            int shared_x = tx + i;
            int img_y = wrap_index(blockIdx.y * TILE_SIZE + shared_y - RADIUS, height);
            int img_x = wrap_index(blockIdx.x * TILE_SIZE + shared_x - RADIUS, width);
            tile[shared_y][shared_x] = image[img_y * width + img_x];
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        float weight = 1.0f / 289.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                uchar3 pixel = tile[ty + RADIUS + dy][tx + RADIUS + dx];
                r_sum += pixel.x;
                g_sum += pixel.y;
                b_sum += pixel.z;
            }
        }
        
        image_output[y * width + x] = make_uchar3(
            (uchar)(r_sum * weight),
            (uchar)(g_sum * weight),
            (uchar)(b_sum * weight)
        );
    }
}

int main(int argc, char **argv) {
    unsigned int image_size;
    uchar3 *d_image, *d_image_output;
    uchar3 *h_image;
    cudaEvent_t start, stop;
    float ms;

    const char *mode = NULL;
    const char *output_file = "output.ppm";
    int radius = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && (i + 1) < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--radius") == 0 && (i + 1) < argc) {
            radius = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && (i + 1) < argc) {
            output_file = argv[++i];
        }
    }

    image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar3);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_image = (uchar3*)malloc(image_size);
    input_image_file((char*)"input.ppm", h_image);

    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_image_output, image_size);
    checkCUDAError("CUDA malloc");

    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy to device");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((IMAGE_DIM + block.x - 1) / block.x, (IMAGE_DIM + block.y - 1) / block.y);

    if (mode != NULL && radius != 0) {
        if (strcmp(mode, "basic") != 0 && strcmp(mode, "tiled") != 0) {
            fprintf(stderr, "Usage: %s [--mode basic|tiled] [--radius 1|2|4|8] [--output filename]\n", argv[0]);
            return 1;
        }

        if (strcmp(mode, "basic") == 0) {
            if (radius == 1) {
                image_blur_A_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else if (radius == 2) {
                image_blur_B_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else if (radius == 4) {
                image_blur_C_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else if (radius == 8) {
                image_blur_D_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else {
                fprintf(stderr, "Invalid radius %d. Use 1, 2, 4, or 8.\n", radius);
                return 1;
            }
        } else {
            if (radius == 1) {
                image_blur_A_tiled<BLOCK_SIZE, 1><<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else if (radius == 2) {
                image_blur_B_tiled<BLOCK_SIZE, 2><<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else if (radius == 4) {
                image_blur_C_tiled<BLOCK_SIZE, 4><<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else if (radius == 8) {
                image_blur_D_tiled<BLOCK_SIZE, 8><<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
            } else {
                fprintf(stderr, "Invalid radius %d. Use 1, 2, 4, or 8.\n", radius);
                return 1;
            }
        }

        cudaDeviceSynchronize();
        checkCUDAError("Kernel launch (single mode)");

        cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
        checkCUDAError("CUDA memcpy from device");

        output_image_file_named(output_file, h_image);

        cudaFree(d_image);
        cudaFree(d_image_output);
        checkCUDAError("CUDA free");

        free(h_image);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return 0;
    }

    printf("=== Basic Kernels (Without Tiling) ===\n");

    // Warm-up launch
    image_blur_A_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    image_blur_A_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (A basic)");
    printf("Filter A (radius=1) basic: %f ms\n", ms);

    cudaEventRecord(start);
    image_blur_B_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (B basic)");
    printf("Filter B (radius=2) basic: %f ms\n", ms);

    cudaEventRecord(start);
    image_blur_C_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (C basic)");
    printf("Filter C (radius=4) basic: %f ms\n", ms);

    cudaEventRecord(start);
    image_blur_D_basic<<<grid, block>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (D basic)");
    printf("Filter D (radius=8) basic: %f ms\n", ms);

    cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device");
    output_image_file(h_image);

    printf("\n=== Tiled Kernels (With Shared Memory) ===\n");

    dim3 blockTiled(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start);
    image_blur_A_tiled<BLOCK_SIZE, 1><<<grid, blockTiled>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (A tiled)");
    printf("Filter A (radius=1) tiled: %f ms\n", ms);

    cudaEventRecord(start);
    image_blur_B_tiled<BLOCK_SIZE, 2><<<grid, blockTiled>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (B tiled)");
    printf("Filter B (radius=2) tiled: %f ms\n", ms);

    cudaEventRecord(start);
    image_blur_C_tiled<BLOCK_SIZE, 4><<<grid, blockTiled>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (C tiled)");
    printf("Filter C (radius=4) tiled: %f ms\n", ms);

    cudaEventRecord(start);
    image_blur_D_tiled<BLOCK_SIZE, 8><<<grid, blockTiled>>>(d_image, d_image_output, IMAGE_DIM, IMAGE_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("Kernel launch (D tiled)");
    printf("Filter D (radius=8) tiled: %f ms\n", ms);

    cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device");
    output_image_file(h_image);

    printf("\n=== Summary ===\n");
    printf("All optimizations applied. See timing results above.\n");

    cudaFree(d_image);
    cudaFree(d_image_output);
    checkCUDAError("CUDA free");

    free(h_image);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

void output_image_file(uchar3* image)
{
    output_image_file_named("output.ppm", image);
}

void output_image_file_named(const char* filename, uchar3* image)
{
    FILE *f;
    f = fopen(filename, "wb");
    if (f == NULL){
        fprintf(stderr, "Error opening '%s' output file\n", filename);
        exit(1);
    }
    fprintf(f, "P6\n");
    fprintf(f, "# CS 629/729 Lab 05 Task02\n");
    fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
    for (int x = 0; x < IMAGE_DIM; x++){
        for (int y = 0; y < IMAGE_DIM; y++){
            int i = x + y*IMAGE_DIM;
            fwrite(&image[i], sizeof(unsigned char), 3, f);
        }
    }
    fclose(f);
}

void input_image_file(char* filename, uchar3* image)
{
    FILE *f;
    char temp[256];
    unsigned int x, y, s;

    f = fopen((char*)"input.ppm", "rb");
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
