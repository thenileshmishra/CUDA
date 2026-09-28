#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main(){
    int device = 0;

    // cudaDeviceProp is a container that holds information about your GPU
    cudaDeviceProp prop;

    // cudaGetDeviceProperties() as the function that fills that container
    cudaGetDeviceProperties(&prop, device);

    cout << "GPU Name           :" << prop.name << endl;
    cout << "Number of SMs      :" << prop.multiProcessorCount << endl;
    cout << "Max Threads/Block  :" << prop.maxThreadsPerBlock << endl;
    cout << "Warp Size          :" << prop.warpSize << endl;
    cout << "Shared Mem/Block   :" << prop.sharedMemPerBlock << "bytes" << endl;

    return 0;
}


// SM — Streaming Multiprocessor
// SM = Streaming Multiprocessor.
// An SM is a major execution unit inside an NVIDIA GPU.
// A GPU contains multiple SMs. Your RTX 6000 Ada has 142 SMs.
// Each SM contains resources such as:
// CUDA cores
// Warp schedulers
// Registers
// Shared memory
// Load/store units
// Special-function units
// CUDA blocks are assigned to SMs for execution.
// A block stays on one SM while it executes; it is not split across multiple SMs.
// Each block is divided into warps of 32 threads.
// An SM can execute/manage multiple warps and blocks concurrently, depending on available resources.
// More SMs generally provide more parallel execution capacity, but actual performance also depends on memory, occupancy, instruction mix, etc.
