#include <cuda_runtime.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>
#include <memory>
#include <grammar/grammar_parser.hpp>
#include "utils/grammar_loader.cuh"
#include "device_management.cuh"
#include "macros.def"
#include <stdio.h>

// __global__ void initialize_buffers(){
    
// }
// Kernel function to initialize the CKY buffer

__device__ float logsumexpf(float a, float b);

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
    context.cuda_gc->fill(context.CKY, -INFINITY);

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
           int grammar_idx = s_A * (S + 1) * (S + 1) + sequence[i] * (S + 1);
           float rule_val = grammar_ptr[grammar_idx];
           int cky_idx = s_A * (MAX_SEQ_LEN * MAX_SEQ_LEN) + i * MAX_SEQ_LEN + i;

           printf("Grammar[%d, %d, 0] == %lf, %lf\n",  s_A, sequence[i], rule_val, grammar_ptr[0]);


           // CKY table access: shape [S, MAX_SEQ_LEN, MAX_SEQ_LEN]
           cky_ptr[cky_idx] = rule_val;

        }
   }
}

// Helper function for atomic float max
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}


__global__ void cky_reduce_kernel(
    int S, 
    int MAX_SEQ_LEN,
    float* __restrict__ cky_table,
    float* __restrict__ intermediate_buffer
){
    // Parallelize across 3D grid: s_A, i, j
    int s_A = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Boundary checks
    if (s_A >= S || i >= MAX_SEQ_LEN || j >= MAX_SEQ_LEN) 
        return;

    float reduced_val = -INFINITY;
    int base_idx = s_A * S * MAX_SEQ_LEN * MAX_SEQ_LEN + i * MAX_SEQ_LEN + j;

    // Each thread reduces across s_B dimension
    for (int s_B = 0; s_B < S; s_B++) {
        int buffer_idx = base_idx + s_B * MAX_SEQ_LEN * MAX_SEQ_LEN;
        reduced_val = logsumexpf(reduced_val, intermediate_buffer[buffer_idx]);
    }

    // Write reduced result to CKY table
    cky_table[s_A * MAX_SEQ_LEN * MAX_SEQ_LEN + i * MAX_SEQ_LEN + j] = reduced_val;
}


__device__ float logsumexpf(float a, float b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    float max_ab = fmaxf(a, b);
    return max_ab + logf(expf(a - max_ab) + expf(b - max_ab));
}
__global__ void cky_span_processing_kernel(
    int span_length, int S, int MAX_SEQ_LEN,
    float* __restrict__ cky,
    float* __restrict__ grammar,
    float* __restrict__ results)
{
    // Parallelize over 3 axes: i, s_A, s_B
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s_A = blockIdx.y * blockDim.y + threadIdx.y;
    int s_B = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary checks
    if (i >= MAX_SEQ_LEN - span_length || s_A >= S || s_B >= S) return;

    int j = i + span_length;
    if (j >= MAX_SEQ_LEN) return;

    float total_score = -INFINITY;
    const int grammar_stride = (S + 1) * (S + 1);

    // Prefetch the base grammar rule for epsilon productions
    float epsilon_rule = grammar[s_A * grammar_stride + s_B * (S + 1) + 0];

    // Process all possible splits
    for (int k_offset = 0; k_offset < span_length; k_offset++) {
        int k = i + k_offset;
        if (k > j) continue;
        total_score = -INFINITY;
        float left_score = cky[s_B * MAX_SEQ_LEN * MAX_SEQ_LEN + i * MAX_SEQ_LEN + k];
        
        // Handle epsilon production case
        if (k == j) {
            total_score = logsumexpf(total_score, left_score + epsilon_rule);
        }else{
            // Process all right children
            for (int s_C = 1; s_C < S + 1; s_C++) { // s_C cannot equal to 0 (epsilon). 
                float right_score = cky[s_C * MAX_SEQ_LEN * MAX_SEQ_LEN + (k + 1) * MAX_SEQ_LEN + j];
                float rule = grammar[s_A * grammar_stride + s_B * (S + 1) + s_C];
                total_score = logsumexpf(total_score, left_score + right_score + rule);
            }
        }       

    }
    int index_sA_sB_i_j = s_A * (S + 1) * MAX_SEQ_LEN * MAX_SEQ_LEN + s_B * (MAX_SEQ_LEN * MAX_SEQ_LEN) + i * MAX_SEQ_LEN + j;
    results[index_sA_sB_i_j] = logsumexpf(results[index_sA_sB_i_j], total_score); // [s_A, s_B, i, k]

    // // Write result only if valid
    // if (j < MAX_SEQ_LEN) {
    //     if (i == j)
    //         printf("set CKY[%d, %d, %d] = %lf\n", s_A, i, j, total_score);
    //     results[s_A * MAX_SEQ_LEN * MAX_SEQ_LEN + i * MAX_SEQ_LEN + j] = total_score;
    // }
}

void cuda_cky_algorithm(AlgorithmContext context) {
    std::cout << "Begin CKY algorithm..." << std::endl;
    std::cout << "Zero out CKY Buffer..." << std::endl;

    initialize_buffers(context);
    std::cout << "[Completed] Zero out CKY Buffer." << std::endl;

    // Launch the kernel to initialize the CKY table
    const int BLOCK_X = 128;  
    const int BLOCK_Y = 8; 

    // Compute grid size (adjust based on your GPU limits)
    int grid_x = min((context.MAX_SEQ_LEN + BLOCK_X - 1) / BLOCK_X, 65535);
    int grid_y = min((context.S + BLOCK_Y - 1) / BLOCK_Y, 65535);

    dim3 blocks(grid_x, grid_y);
    dim3 threads(BLOCK_X, BLOCK_Y);


    std::cout << "Launch CKY span 1 calcualtion kernel..." <<  std::endl;

    cky_initialization_kernel<<<blocks, threads>>>(context.S, context.MAX_SEQ_LEN, 
        context.CKY.ptr, context.grammar.ptr, context.sequence.ptr);
    

    cudaError_t cudaerr = cudaPeekAtLastError();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    std::cout << "[Completed] CKY span 1 calcualtion." << std::endl;
    
    /* In the CKY algorithm, tasks with a particular span length represent the largest
       parallelizable units of computation. Therefore, we set the largest grain of
       parallelism to the computation over a specific span length. */
    // dim3 cky_blockDim(64, 4, 4);  // Each block has N x S x S threads
    // dim3 cky_gridDim((context.MAX_SEQ_LEN + 64 - 1) / 64, (context.S + 4 - 1) / 4, (context.S + 4 - 1) / 4); 
    // for(int span_length = 2; span_length < context.MAX_SEQ_LEN; span_length++) {
       
    //     cky_span_processing_kernel<<<cky_gridDim, cky_blockDim>>>(
    //         span_length, context.S, context.MAX_SEQ_LEN, 
    //         context.CKY.ptr, context.grammar.ptr, context.intermediate_results_buffer.ptr);
    //     cudaDeviceSynchronize();
    // }
    // cky_reduce_kernel<<<cky_gridDim, cky_blockDim>>>(context.S, context.MAX_SEQ_LEN, context.CKY.ptr, context.intermediate_results_buffer.ptr);
    
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

    int MAX_SEQ_LEN = config["cky_buffer"]["size"]["max_seq_len"].as<int>();
    const std::string& grammar_file_path =  config["grammar"]["file_path"].as<std::string>();
    std::cout << "grammar file path = " << grammar_file_path << std::endl;
    pcfg* parsed_pcfg = prepare_grammar(grammar_file_path);
    __host_pt__ float* host_grammar_buffer = initialize_grammar_buffer_from_pcfg(parsed_pcfg);
    int S = parsed_pcfg->nonterminate_map.size() + parsed_pcfg->terminate_map.size();


    size_t n_cky_buffer_elements = (S + 1) * MAX_SEQ_LEN * MAX_SEQ_LEN;
    size_t n_grammar_buffer_elements = (S + 1) * (S + 1) * (S + 1);
    size_t n_sequence_buffer_elements = MAX_SEQ_LEN; // A -> B C
    long n_intermediate_results_buffer_elements = (S + 1) * (S + 1) * MAX_SEQ_LEN * MAX_SEQ_LEN; // [A, B, i, j]
    std::cout << MAX_SEQ_LEN << "," << S << ", " << S * S * MAX_SEQ_LEN * MAX_SEQ_LEN  << std::endl;
    cuda_gc_managed_pt<float> d_CKY = cuda_gc->allocate<float>(n_cky_buffer_elements);
    cuda_gc_managed_pt<float> grammar = cuda_gc->allocate<float>(n_grammar_buffer_elements);
    cuda_gc_managed_pt<int> sequence = cuda_gc->allocate<int>(n_sequence_buffer_elements);
    cuda_gc->zerolize(grammar);
    cuda_gc->zerolize(sequence);

    cuda_gc_managed_pt<float> intermediate_results_buffer = cuda_gc->allocate<float>(n_intermediate_results_buffer_elements);
    context.S = S;
    context.MAX_SEQ_LEN = MAX_SEQ_LEN;
    context.CKY = d_CKY;
    context.intermediate_results_buffer = intermediate_results_buffer;
    
    initialize_buffers(context);
    cuda_gc->fill(intermediate_results_buffer, -INFINITY);
    cudaDeviceSynchronize();
    __host_pt__ int* host_sequence = new int[MAX_SEQ_LEN];
    
    /* [fish people fish tanks]'s ID sequence == [10 9 10 11] + 1*/
    host_sequence[0] = 11;
    host_sequence[1] = 10;
    host_sequence[2] = 11;
    host_sequence[3] = 12;
    host_sequence[4] = 0;
    host_sequence[5] = 0;
    host_sequence[6] = 0;
    host_sequence[7] = 0;
    host_sequence[8] = 0;
    host_sequence[9] = 0;

    cudaMemcpy(grammar.ptr, host_grammar_buffer, n_grammar_buffer_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sequence.ptr, host_sequence, n_sequence_buffer_elements * sizeof(int), cudaMemcpyHostToDevice);
    context.sequence = sequence;
    context.grammar = grammar;
    cuda_cky_algorithm(context);

    /* Process data in host. */
    // For demonstration: copy a small part of CKY to the host and print a value
    __host_pt__ float* h_CKY = new float[n_cky_buffer_elements];  // Allocate host memory
    cudaMemcpy(h_CKY, d_CKY.ptr, n_cky_buffer_elements * sizeof(float), cudaMemcpyDeviceToHost);  // Copy data from device to host

    // Print a value for demonstration (example: CKY[0][0][0])
    for(int s = 0; s < S; s++){
        for(int i = 0; i < 4; i++){
            for(int j = i; j < 4; j++){
                std::cout << "CKY[" << s << "][" << i << "][" << j << "]: " 
                << h_CKY[s * MAX_SEQ_LEN * MAX_SEQ_LEN + i * MAX_SEQ_LEN + j] << std::endl;
            }
        }
    }

    // Clean up
    delete[] h_CKY; 
    cuda_gc->deallocate<float>(d_CKY);
    cudaDeviceReset();

    return 0;
}
