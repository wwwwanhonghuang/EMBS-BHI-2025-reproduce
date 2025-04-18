#ifndef DEVICE_MANAGEMENT_CUH
#define DEVICE_MANAGEMENT_CUH
#include <cuda_runtime.h>
#include "macros.def"
#include <iostream>

int select_cuda_device(int device_id);



class CudaGC {
public:
    // Template function to allocate memory on the device
    template <typename T>
    __device_pt__ T* allocate(int element_count) {
        // Allocate device memory using cudaMalloc
        __device_pt__ T* d_ptr = nullptr;
        cudaError_t err = cudaMalloc((void**)&d_ptr, element_count * sizeof(T));
        if (err != cudaSuccess) {
            printf("CUDA memory allocation failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        return d_ptr;
    }

    // Function to deallocate memory on the device
    template <typename T>
    void deallocate(__device_pt__ T* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};
#endif