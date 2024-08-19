
# **CUDA-Based Convolution Operation**

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Compilation and Execution](#compilation-and-execution)
4. [Input and Output](#input-and-output)
5. [CUDA Kernels and Memory Management](#cuda-kernels-and-memory-management)
6. [Performance Measurement](#performance-measurement)


## **Project Overview**

This project implements a 2D convolution operation using CUDA for parallel processing. The convolution is applied to a matrix with a specified filter. The program efficiently manages GPU memory to handle large matrices and filters, ensuring minimal memory usage and avoiding running out of memory during execution.

## **Prerequisites**

- **CUDA Toolkit**: Ensure that CUDA is installed on your system.
- **NVIDIA GPU**: A compatible GPU is required to run the CUDA code.
- **C++ Compiler**: A compiler like `g++` that supports C++11 or later.

## **Compilation and Execution**

### **Compilation**

To compile the code, use the following command:

```bash
nvcc cuda_convolution.cu -o cuda_convolution
```

### **Execution**

To run the compiled program:

```bash
./cuda_convolution < input_file
```

- `<input_file>`: Path to the input file containing the matrix dimensions, filter size, matrix elements, and filter elements.

The output will be written to `cuda.out`, and the execution time will be recorded in `cuda_timing.out`.

## **Input and Output**

### **Input Format**

The input should be provided via a file or standard input with the following format:

1. **Matrix Dimensions (m, n)**: Two integers representing the number of rows `m` and columns `n` of the matrix.
2. **Filter Size (k)**: One integer `k` representing the dimensions of the filter, which is `k x k`.
3. **Matrix Elements**: `m * n` integers representing the elements of the matrix, provided row-wise.
4. **Filter Elements**: `k * k` integers representing the elements of the filter, provided row-wise.

### **Output Files**

- **`cuda.out`**: This file contains the result of the convolution operation, formatted as a matrix with `m` rows and `n` columns.
- **`cuda_timing.out`**: This file contains the time taken to execute the CUDA kernel in seconds.

## **CUDA Kernels and Memory Management**

### **CUDA Kernel: `PerformCudaConvolutionOperation`**

This kernel performs the convolution operation on the input matrix using a filter. The filter is stored in constant memory for efficient access during the computation. Shared memory is used to hold portions of the matrix for fast access within each block.

### **Memory Management**

The program carefully allocates and frees memory on the GPU:

- **Memory Allocation**: Memory for the matrix and output is allocated on the GPU using `cudaMalloc`.
- **Constant Memory**: The filter is stored in constant memory using `cudaMemcpyToSymbol` for fast access.
- **Memory Transfer**: Data is transferred between the host (CPU) and device (GPU) using `cudaMemcpy`.
- **Memory Deallocation**: GPU memory is freed using `cudaFree` after the computation to prevent memory leaks.

## **Performance Measurement**

The execution time of the CUDA kernel is measured using the `chrono` library in C++. The time taken is recorded in the `cuda_timing.out` file, providing insight into the performance of the convolution operation on the GPU.

