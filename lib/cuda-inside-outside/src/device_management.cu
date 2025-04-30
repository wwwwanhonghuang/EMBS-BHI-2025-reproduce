
#include "device_management.cuh"
#include <iostream>

int select_cuda_device(int device_id){
    // Query the number of available CUDA devices
    int device_count;
    cudaGetDeviceCount(&device_count);

    // Check if the specified device_id is valid
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Invalid device ID. Available devices: 0 to " << device_count - 1 << std::endl;
        return -1;
    }

    // Set the device using the provided device ID
    cudaSetDevice(device_id);
    return 0;
}

template <>
void CudaGC::fill(cuda_gc_managed_pt<float>& ptr, float value) {
    float* data = ptr.ptr;
    size_t count = ptr.element_count;
    std::fill(data, data + count, value);

    // Optional: prefetch to GPU if you plan to use it on the device
    cudaMemPrefetchAsync(data, count * sizeof(float), 0); // 0 = default device
    cudaDeviceSynchronize(); // Wait for prefetch if needed
}