#ifndef DEVICE_MANAGEMENT_CUH
#define DEVICE_MANAGEMENT_CUH
#include <cuda_runtime.h>
#include "macros.def"
#include <iostream>

int select_cuda_device(int device_id);

template<typename T>
struct cuda_gc_managed_pt{
    __device_pt__ T* ptr;
    int element_count;
    cuda_gc_managed_pt() : ptr(nullptr), element_count(0) {}

    cuda_gc_managed_pt(__device_pt__ T* ptr, int element_count) {
        this->ptr = ptr;
        this->element_count = element_count;
    }
};

class CudaGC {
public:
    // Template function to allocate memory on the device
    template <typename T>
    cuda_gc_managed_pt<T> allocate(int element_count) {
        // Allocate device memory using cudaMalloc
        __device_pt__ T* d_ptr = nullptr;
        cudaError_t err = cudaMalloc((void**)&d_ptr, element_count * sizeof(T));
        if (err != cudaSuccess) {
            printf("CUDA memory allocation failed: %s\n", cudaGetErrorString(err));
            return cuda_gc_managed_pt<T>();
        }
        return cuda_gc_managed_pt<T>(d_ptr, element_count);
    }

    // Function to deallocate memory on the device
    template <typename T>
    void deallocate(cuda_gc_managed_pt<T> ptr) {
        if (ptr.ptr) {
            cudaFree(ptr.ptr);
        }
    }

    template <typename T>
    void zerolize(cuda_gc_managed_pt<T>& ptr) {
        size_t buffer_size = ptr.element_count * sizeof(T);
        cudaError_t err = cudaMemset(ptr.ptr, 0, buffer_size);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory zeroing failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
};
#endif