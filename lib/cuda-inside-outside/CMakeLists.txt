cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CudaInsideOutside)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

# Define the CUDA source file
set(SOURCE_FILES src/cuda-inside-outside-main.cu src/device_management.cu)
include_directories(include)

# Add executable target and specify the source files
cuda_add_executable(cuda_inside_outside ${SOURCE_FILES})


# Link necessary libraries
target_link_libraries(cuda_inside_outside yaml-cpp cuda)

# Set any necessary compile options (optional)
set_target_properties(cuda_inside_outside PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(cuda_inside_outside PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Optional: Specify the include directories if you have any custom headers
# target_include_directories(cuda_inside_outside PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Set the required CUDA architecture
set(CUDA_ARCH "sm_86") # You can adjust the compute capability as needed

# Ensure proper CUDA flags for the target
set_target_properties(cuda_inside_outside PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH}"
)

# Add any additional flags or compile options
target_compile_options(cuda_inside_outside PRIVATE -Werr)
