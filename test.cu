#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define TGIG 536870912
#define SLEEP_TIME 100

int main() {
    float N = TGIG*3.2; 
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
    printf("Allocated and copied %f GB to GPU. Sleeping for %d seconds...\nWill Now start deallocating from RAM\n", (N*4)/1000000000, SLEEP_TIME);
    sleep(SLEEP_TIME);
    cudaFree(device_array);
    return 0;
}
