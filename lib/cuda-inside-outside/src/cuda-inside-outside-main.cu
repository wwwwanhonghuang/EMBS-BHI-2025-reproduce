#include <cuda_runtime.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>
#include "device_management.cuh"
#include "macros.def"




// __global__ void initialize_buffers(){
    
// }
// Kernel function to initialize the CKY buffer
struct AlgorithmContext{
    int S = 0;
    int MAX_SEQ_LEN = 0;
    __device_pt__ float* CKY = nullptr; // Device pointer

};


__global__ void initialize_CKY(__device_pt__ float* CKY, AlgorithmContext context) {
    int x = threadIdx.x;
    int y = threadIdx.y;

    if (x < context.S && y < context.MAX_SEQ_LEN) {
        int index = x * context.MAX_SEQ_LEN * context.MAX_SEQ_LEN + y * context.MAX_SEQ_LEN + y;
        CKY[index] = 0.0f;
    }
}


int main(int argc, char* argv[]) {
    std::string configuration_file_path = "./configurations/config.yaml";
    CudaGC cuda_gc;
    AlgorithmContext context;
    
    if (argc >= 2) {
        configuration_file_path = std::string(argv[1]);
    }

    YAML::Node config;
    try {
        config = YAML::LoadFile(configuration_file_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return -1;
    }

    int use_device_id = config["cuda_device"]["use_device_id"].as<int>();
    if(select_cuda_device(use_device_id) == 0){
        std::cout << "Use CUDA Device ID: " << use_device_id << std::endl;
    }else{
        return -1;
    }


    int S = config["cky_buffer"]["size"]["S"].as<int>();
    int MAX_SEQ_LEN = config["cky_buffer"]["size"]["max_seq_len"].as<int>();


    context.S = S;
    context.MAX_SEQ_LEN = MAX_SEQ_LEN;



    size_t n_cky_buffer_elements = S * MAX_SEQ_LEN * MAX_SEQ_LEN;

    __device_pt__ float* d_CKY = cuda_gc.allocate<float>(n_cky_buffer_elements);
    

    // Define the size of the grid for initialization
    dim3 threadsPerBlock(16, 16);  // Define block size
    dim3 numBlocks((S + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (MAX_SEQ_LEN + threadsPerBlock.y - 1) / threadsPerBlock.y); // Define grid size

    // Launch the kernel to initialize the CKY buffer
    initialize_CKY<<<numBlocks, threadsPerBlock>>>(d_CKY, context);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // For demonstration: copy a small part of CKY to the host and print a value
    __host_pt__ float* h_CKY = new float[n_cky_buffer_elements];  // Allocate host memory
    cudaMemcpy(h_CKY, d_CKY, n_cky_buffer_elements * sizeof(float), cudaMemcpyDeviceToHost);  // Copy data from device to host

    // Print a value for demonstration (example: CKY[0][0][0])
    std::cout << "CKY[0][0][0]: " << h_CKY[0 * MAX_SEQ_LEN * MAX_SEQ_LEN + 0 * MAX_SEQ_LEN + 0] << std::endl;

    // Clean up
    delete[] h_CKY;  // Free host memory
    cuda_gc.deallocate<float>(d_CKY);
    cudaDeviceReset();

    return 0;
}
