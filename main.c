#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define GIG 1073741824

int main() {
    size_t N = GIG; // 2 GB for float (4 bytes each)
    float *host_array = (float*)malloc(N * sizeof(float));
    if (!host_array) {
        printf("Host malloc failed\n");
        return 1;
    }

    for (size_t i = 0; i < N; ++i) host_array[i] = (float)i;

    float *device_array;
    cudaError_t err = cudaMalloc((void**)&device_array, N * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(host_array);
        return 1;
    }

    err = cudaMemcpy(device_array, host_array, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_array);
        free(host_array);
        return 1;
    }

  free(host_array); 
  printf("Allocated and copied 2 GB to GPU. Sleeping for 10 seconds...\n");
  sleep(100);

    cudaFree(device_array);
    
    return 0;
}
