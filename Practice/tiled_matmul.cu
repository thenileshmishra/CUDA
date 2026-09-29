#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void tiledMatMul(float *C, const float *A, const float *B,
                            int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for tiles
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    float sum = 0.0f;

    // Process one tile at a time
    for (int t = 0; t < K / TILE; t++) {

        // Load A tile into shared memory
        tileA[threadIdx.y][threadIdx.x] =
            A[row * K + t * TILE + threadIdx.x];

        // Load B tile into shared memory
        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        // Wait for all threads to finish loading
        __syncthreads();

        // Multiply the tiles
        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        // Make sure all threads finish before loading next tile
        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


int main() {

    const int M = 1024, N = 1024, K = 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    // Initialize matrices
    for (int i = 0; i < M * K; i++)
        h_A[i] = 1.0f;

    for (int i = 0; i < K * N; i++)
        h_B[i] = 1.0f;


    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);


    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);


    // 16 × 16 threads per block
    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks(
        (N + 15) / 16,
        (M + 15) / 16
    );


    // Launch tiled matrix multiplication
    tiledMatMul<<<numBlocks, threadsPerBlock>>>(
        d_C, d_A, d_B, M, N, K
    );


    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);


    printf("Tiled Matrix Mul: %dx%dx%d\n", M, K, N);
    printf("Verification: C[0] = %f\n", h_C[0]);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
