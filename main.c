#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "/opt/cuda/include/cuda_runtime_api.h"


#define GIG (1024 * 1024 * 1024) / 4

int main() {
    size_t gigs = 4;
    size_t N = GIG*gigs; 
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
  u_int sleep_time = 100;
  printf("Allocated and copied %zu GB to GPU. Sleeping for %u seconds...\n", gigs, sleep_time);
  sleep(sleep_time);

    cudaFree(device_array);
    
    return 0;
}
