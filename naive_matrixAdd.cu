﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>



cudaError_t addVecWithCuda(int *c,  int *a,  int *b, const unsigned int size);

cudaError_t addMatrixWithCuda(int** c, int** a, int** b, const unsigned int rows, const unsigned int cols);

__global__ void addVecKernel(int* c, int* a, int* b, const unsigned int size)
{
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void addMatrixKernel(int** c, int** a, int** b, const unsigned int rows, const unsigned int cols)
{
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < cols * rows) {
        c[i / rows][i % cols] = a[i / rows][i % cols] + b[i / rows][i % cols];
    }
}

void genRandVec(int* v, const unsigned int arraySize) {
    for (unsigned int i = 0; i < arraySize; i++) {
        v[i] = rand() % 100;
    }
}

void genRandMatrix(int** M, const unsigned int n, const unsigned int m) {
    for (unsigned int i = 0; i < m; i++) {
        genRandVec(M[i], n);
    }
}

void printVec(int* v, const unsigned int arraySize) {
    for (unsigned int i = 0; i < arraySize; i++) {
        std::cout << v[i] << " ";
    }
    //std::cout << std::endl;
}

void errorCheckVecAdd(int* a, int* b, int* c, const unsigned int arraySize) {
    for (unsigned int i = 0; i < arraySize; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

void errorCheckMatrixAdd(int** a, int** b, int** c, const unsigned int rows, const unsigned int cols) {
    for (unsigned int i = 0; i < cols; i++) {
        for (unsigned int j = 0; j < rows; j++) {
            c[j][i] = a[j][i] + b[j][i];
        }
    }
}

int main() 
{
    const unsigned int arraySize = 500;
    int a[arraySize] = {};
    int b[arraySize];
    genRandVec(a, arraySize);
    genRandVec(b, arraySize);
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addVecWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    //printVec(a, arraySize); std::cout << "+ "; printVec(b, arraySize); std::cout << "= "; printVec(c, arraySize);
    
    //copy(&b[0], &b[arraySize], std::ostream_iterator<std::string>(std::cout, " "));
    //copy(&c[0], &c[arraySize], std::ostream_iterator<std::string>(std::cout, " "));
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    errorCheckVecAdd(a, b, c, arraySize);
    printf("Successful! with vectors of length %d", arraySize);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, int *b, const unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addVecKernel<<<1, size>>>(dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
