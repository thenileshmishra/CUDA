#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) c[i] = a[i] + b[i];
}


int main(){
    const int size = 1000000, bytes = size * sizeof(int);

    int *h_a = (int*)malloc(bytes), *h_b = (int*)malloc(bytes), *h_c = (int*)malloc(bytes);

    for(int i = 0 ; i < size ; i++) {
        h_a[i] = i; 
        h_b[i] = i*2;
    }

    int *dev_a, *dev_b, *dev_c;

    // GPU par bytes size ki memory reserve karo aur uska address dev_a mein daal do.
    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);
    

    // cudaMemcpy(destination, source, size, direction);
        // h_a: CPU/RAM mein input array
        // dev_a: GPU memory mein allocated array
        // bytes: kitna data copy karna hai
        // cudaMemcpyHostToDevice: CPU se GPU direction
        
    cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256, blockPerGrid = (size + 255) / 256;
    addKernel<<<blockPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
    cudaMemcpy(h_c, dev_c, bytes, cudaMemcpyDeviceToHost);
     
    printf("Vector adddition of %d elements\n", size);
    printf("Verification: c[0]= %d, c[999999]=%d\n", h_c[0], h_c[999999]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;

}