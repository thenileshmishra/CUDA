#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        return 1; \
    } \
} while (0)


__global__ void subKernel(int *c, int *a, int *b, int size){
    int i = blockIdx.x* blockDim.x + threadIdx.x;
    if( i < size) c[i] = a[i] - b[i];
}


int main(){

    int size = 1000000, bytes = size * sizeof(int);

    int *h_a = (int*)malloc(bytes), *h_b = (int*)malloc(bytes), *h_c = (int*)malloc(bytes);

    for(int i = 0; i < size ; i++){
        h_a[i] = i*3;
        h_b[i] = i;
    }

    int *dev_a, *dev_b, *dev_c;
    CHECK_CUDA(cudaMalloc(&dev_a, bytes));
    CHECK_CUDA(cudaMalloc(&dev_b, bytes));
    CHECK_CUDA(cudaMalloc(&dev_c, bytes));

    CHECK_CUDA(cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256, blockPerGrid = (size+255)/256;

    subKernel <<<blockPerGrid, threadsPerBlock>>> (dev_c, dev_a, dev_b, size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_c, dev_c, bytes, cudaMemcpyDeviceToHost));

    printf("Vector Subtraction of %d elements \n", size);
    printf("Verification: c[0] = %d, c[999999]=%d\n", h_c[0], h_c[999999]);

    CHECK_CUDA(cudaFree(dev_a));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}