#include <cuda_runtime.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>
#include <memory>
#include "device_management.cuh"
#include "macros.def"

// __global__ void initialize_buffers(){
    
// }
// Kernel function to initialize the CKY buffer


struct AlgorithmContext{
    int S = 0;
    int MAX_SEQ_LEN = 0;
    cuda_gc_managed_pt<float> CKY;
    cuda_gc_managed_pt<float> grammar;
    cuda_gc_managed_pt<int> sequence;
    cuda_gc_managed_pt<float> intermediate_results_buffer;
    std::shared_ptr<CudaGC> cuda_gc;

};


void initialize_buffers(AlgorithmContext context){
    context.cuda_gc->zerolize(context.CKY);
}


YAML::Node read_yaml_configuration(const std::string& configuration_file_path){
    try {
        YAML::Node config = YAML::LoadFile(configuration_file_path);
        return config;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return YAML::Node();
    }
}

__global__ void cky_initialization_kernel(int S, int MAX_SEQ_LEN, 
    __device_pt__ float* cky_ptr, __device_pt__ float* grammar_ptr, __device_pt__ int* sequence){
    
        // Grid-striding for BOTH sequence position (i) and symbol (s_A)
    for (int s_A = blockIdx.y * blockDim.y + threadIdx.y; 
        s_A < S; 
        s_A += blockDim.y * gridDim.y) {
       
       for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < MAX_SEQ_LEN; 
            i += blockDim.x * gridDim.x) {
           
        
           // Grammar access: shape [S, S, S]
           int grammar_idx = s_A * (MAX_SEQ_LEN * MAX_SEQ_LEN) + sequence[i] * MAX_SEQ_LEN;
           float rule_val = grammar_ptr[grammar_idx];
           
           // CKY table access: shape [S, MAX_SEQ_LEN, MAX_SEQ_LEN]
           int cky_idx = s_A * (MAX_SEQ_LEN * MAX_SEQ_LEN) + i * MAX_SEQ_LEN + i;
           cky_ptr[cky_idx] = rule_val;
       }
   }
}

__global__ void cky_span_processing_kernel_calculate_intermediate_results(
    int span_length, int S, int MAX_SEQ_LEN, 
    float* cky, float* grammar, int* sequence, float* intermediate_results_buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int s_A = threadIdx.y + blockIdx.y * blockDim.y;
    int s_B = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= MAX_SEQ_LEN || s_A >= S || s_B >= S)
        return;

    // This loop could also be made parallel if desired
    for (int i = 0; i < MAX_SEQ_LEN - span_length; i += blockIdx.x * blockDim.x) { // parallel i
        int right_boundary = i + span_length;

        for (int k = i; k < right_boundary; k++) { // k-loop cannot be parallel. 
            for(s_B = s_A,)
            int id_s_B_i_k = s_B * MAX_SEQ_LEN * MAX_SEQ_LEN + i * MAX_SEQ_LEN + k;
            int id_s_C_k_p1_j = s_C * MAX_SEQ_LEN * MAX_SEQ_LEN + (k + 1) * MAX_SEQ_LEN + (i + span_length);
            int id_grammar_A_B_C = s_A * S * S + s_B * S + s_C;

            float cky_val = cky[id_s_B_i_k] * cky[id_s_C_k_p1_j] * grammar[id_grammar_A_B_C];

            int index = s_A * S * S * MAX_SEQ_LEN * MAX_SEQ_LEN + 
                        s_B * S * MAX_SEQ_LEN * MAX_SEQ_LEN +
                        s_C * MAX_SEQ_LEN * MAX_SEQ_LEN +
                        i * MAX_SEQ_LEN + k;

            intermediate_results_buffer[index] = cky_val;
        }
    }
}


void cuda_cky_algorithm(AlgorithmContext context) {
    std::cout << "Begin CKY algorithm..." << std::endl;
    std::cout << "Zero out CKY Buffer..." << std::endl;

    initialize_buffers(context);
    std::cout << "[Completed] Zero out CKY Buffer." << std::endl;

    // Launch the kernel to initialize the CKY table
    const int BLOCK_X = 1024;  
    const int BLOCK_Y = 8; 

    // Compute grid size (adjust based on your GPU limits)
    int grid_x = min((context.MAX_SEQ_LEN + BLOCK_X - 1) / BLOCK_X, 65535);
    int grid_y = min((context.S + BLOCK_Y - 1) / BLOCK_Y, 65535);

    dim3 blocks(grid_x, grid_y);
    dim3 threads(BLOCK_X, BLOCK_Y);


    std::cout << "Launch CKY span 1 calcualtion kernel..." << std::endl;

    cky_initialization_kernel<<<blocks, threads>>>(context.S, context.MAX_SEQ_LEN, 
        context.CKY.ptr, context.grammar.ptr, context.sequence.ptr);

    std::cout << "[Completed] CKY span 1 calcualtion." << std::endl;

    /* In the CKY algorithm, tasks with a particular span length represent the largest
       parallelizable units of computation. Therefore, we set the largest grain of
       parallelism to the computation over a specific span length. */
    for(int span_length = 2; span_length < context.MAX_SEQ_LEN; span_length++) {
        cky_span_processing_kernel<<<16, 16>>>(span_length, context.S, context.MAX_SEQ_LEN, context.CKY.ptr, contex.sequence.ptr);
        cudaDeviceSynchronize();
    }
    std::cout << "[Completed] CKY Algorithm." << std::endl;

}


int main(int argc, char* argv[]) {
    std::string configuration_file_path = "./configurations/config.yaml";
    AlgorithmContext context;

    std::shared_ptr<CudaGC> cuda_gc = std::shared_ptr<CudaGC>();
    context.cuda_gc = cuda_gc;

    if (argc >= 2) {
        configuration_file_path = std::string(argv[1]);
    }

    YAML::Node config = read_yaml_configuration(configuration_file_path);
    if (config.IsNull()) {
        std::cerr << "Failed to load configuration file!" << std::endl;
        return -1;  // Handle the error
    }

    
    int use_device_id = config["cuda_device"]["use_device_id"].as<int>();
    if(select_cuda_device(use_device_id) == 0){
        std::cout << "Use CUDA Device ID: " << use_device_id << std::endl;
    }else{
        return -1;
    }

    int S = config["cky_buffer"]["size"]["S"].as<int>();
    int MAX_SEQ_LEN = config["cky_buffer"]["size"]["max_seq_len"].as<int>();
    
    size_t n_cky_buffer_elements = S * MAX_SEQ_LEN * MAX_SEQ_LEN;
    size_t n_grammar_buffer_elements = S * S * S; // A -> B C
    size_t n_sequence_buffer_elements = MAX_SEQ_LEN; // A -> B C
    size_t n_intermediate_results_buffer_elements = S * S * S * MAX_SEQ_LEN * MAX_SEQ_LEN; // A -> B C

    cuda_gc_managed_pt<float> d_CKY = cuda_gc->allocate<float>(n_cky_buffer_elements);
    cuda_gc_managed_pt<float> grammar = cuda_gc->allocate<float>(n_grammar_buffer_elements);
    cuda_gc_managed_pt<float> sequence = cuda_gc->allocate<int>(n_sequence_buffer_elements);
    cuda_gc_managed_pt<float> intermediate_results_buffer = cuda_gc->allocate<float>(n_intermediate_results_buffer_elements);


    context.S = S;
    context.MAX_SEQ_LEN = MAX_SEQ_LEN;
    context.CKY = d_CKY;
    context.grammar = grammar;
    contex.intermediate_results_buffer = intermediate_results_buffer;

    initialize_buffers(context);

    cudaDeviceSynchronize();

    cuda_cky_algorithm(context);


    /* Process data in host. */
    // For demonstration: copy a small part of CKY to the host and print a value
    __host_pt__ float* h_CKY = new float[n_cky_buffer_elements];  // Allocate host memory
    cudaMemcpy(h_CKY, d_CKY.ptr, n_cky_buffer_elements * sizeof(float), cudaMemcpyDeviceToHost);  // Copy data from device to host

    // Print a value for demonstration (example: CKY[0][0][0])
    std::cout << "CKY[0][0][0]: " << h_CKY[0 * MAX_SEQ_LEN * MAX_SEQ_LEN + 0 * MAX_SEQ_LEN + 0] << std::endl;

    // Clean up
    delete[] h_CKY; 
    cuda_gc->deallocate<float>(d_CKY);
    cudaDeviceReset();

    return 0;
}
