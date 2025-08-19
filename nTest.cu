#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <random>
#include <thread>


#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
#define CUERR {                                                            \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                                           \
    }                                                                      \
}  
#define JOIN {                          \
    for (int k = 0; k < nThread; k++)   \
    {                                   \
        threads[k].join();              \
    }                                   \
}
#define deviceId 0
#define nKernel 64
#define transient 3
#define nIteration 500000
#define nThread 8
#define nKernelPerThread 8

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;

cudaError_t dynamicTransformation(float* a, unsigned int sized);
cudaError_t dynamicTransformation(float* a, unsigned int size, const int n);
cudaError_t dynamicTransformation(float* a, unsigned int size, const int n, const int t);
cudaError_t dynamicTransformation(float* a, unsigned int size, const int n, const int t, void* dummy);
void test(float* c, float* check, int arraySize);
void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void initLaunch(float** dev_a, float* a, void* kernelArgs[nKernel][1], cudaStream_t stream[nKernel], int size, int k);
void parallelStreamSync(cudaStream_t stream[nKernel], int k);
void launchKernel(void* kernelArgs[nKernel][1], cudaStream_t stream[nKernel], int size, int k);
void syncLaunchSync(void* kernelArgs[nKernel][1], cudaStream_t stream[nKernel], int size, int k);



__global__ void dynamicTransformationKernel(float *a)
{
    int i = threadIdx.x;
    unsigned long long int N = nIteration;
    float ra = a[i];
    // this_grid().sync();
#pragma unroll
    for (int k = 0; k < N; k++)
    {
        if (0 <= ra < 0.5)
        {
            ra = 2 * ra;
        }
        else if (0.5 <= ra < 1)
        {
            ra = 2 * ra - 1;
        }
    }
    a[i] = ra;
    // this_grid().sync();
}

int main()
{
    const int arraySize = 1024;
    float a[arraySize];
    float check[arraySize];
    // Random number generation setup
    std::random_device rd;                          // Seed generator (non-deterministic if available)
    std::mt19937 gen(rd());                         // Mersenne Twister PRNG
    std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution in range [0, 1)

    // Fill the array with random values
    for (int i = 0; i < arraySize; ++i) {
        a[i] = dis(gen);
    }

    float c[arraySize] = { 0 };

    // init, and time kernel runs
    // run serial kernels
    dynamicTransformation(a, arraySize); CUERR;
    // run parallel kernels
    dynamicTransformation(a, arraySize, nKernel); CUERR;
    // run parallel kernels, with parallel cpu thread launch
    dynamicTransformation(a, arraySize, nKernel, nThread); CUERR;
    // run parallel kernels, with parallel cpu thread launch, full parallel launch
    dynamicTransformation(a, arraySize, nKernel, nThread, (void*)&a); CUERR;
    



    test(c, check, arraySize);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset(); CUERR;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t dynamicTransformation(float *a, unsigned int size)
{
    float* dev_a = 0;

    cudaError_t cudaStatus;
    cudaStream_t stream;
    cudaStream_t stream2;
    dim3 dimBlock(size, 1, 1);
    dim3 dimGrid(1, 1, 1);
    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0); CUERR;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_a, size * sizeof(int)); CUERR;

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); CUERR;
    
    // Setting kernel arguments
    void* kernelArgs[] = {(void*)&dev_a};
    
    // Creating the steam 
    cudaStreamCreate(&stream); CUERR;

    // Setting stream attributes
    // cudaStreamSetAttribute(stream, )

    // Setting device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);  CUERR;

    // deviceProp.concurrentKernels = 1;
    deviceProp.cooperativeLaunch = 1;

    // Benchmark single kernel launch
    printf("Benchmark single serial kernel launch, more launch to get through GPU clock ramp-up\n");
    for (int i = 0; i < transient; i++)
    {
        begin = std::chrono::steady_clock::now();
        for (int k = 0; k < nKernel; k++)
        {
            cudaStreamSynchronize(stream); CUERR;
            // cudaDeviceSynchronize(); CUERR;
            cudaLaunchCooperativeKernel((void*)dynamicTransformationKernel, dimGrid, dimBlock, kernelArgs, 0, stream);
            // cudaDeviceSynchronize(); CUERR;
        }
        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end);
    }
    // Check for any errors launching the kernel
    cudaGetLastError(); CUERR;
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStreamSynchronize(stream); CUERR;
    cudaStreamDestroy(stream); CUERR;


    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost); CUERR;

    cudaFree(dev_a);
    cudaStatus = cudaGetLastError();
    return cudaStatus;
}

// Launch n number of cooperative parallel kernels
cudaError_t dynamicTransformation(float* a, unsigned int size, const int n)
{
    float* dev_a[nKernel];
    void* kernelArgs[nKernel][1];
    cudaError_t cudaStatus;
    cudaStream_t stream[nKernel];
    dim3 dimBlock(size, 1, 1);
    dim3 dimGrid(1, 1, 1);
    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0); CUERR;

    for (int k = 0; k < nKernel; k++)
    {
        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaMalloc((void**)&dev_a[k], size * sizeof(float)); CUERR;

        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(dev_a[k], a, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

        // Setting kernel arguments
        kernelArgs[k][0] = (void*)&dev_a;

        // Creating the steam 
        cudaStreamCreate(&(stream[k])); CUERR;
    }

    // Setting stream attributes
    // cudaStreamSetAttribute(stream, )

    // Setting device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);  CUERR;

    deviceProp.concurrentKernels = 1;
    deviceProp.cooperativeLaunch = 1;

    // Benchmark nKernel number of kernel launch
    printf("Benchmark parallel, nKernel number of kernel launch, more runs to get through GPU clock ramp-up\n");
    for (int i = 0; i < transient; i++)
    {
        begin = std::chrono::steady_clock::now();
#pragma unroll
        for (int k = 0; k < nKernel; k++) { cudaStreamSynchronize(stream[k]); CUERR; }
        // cudaDeviceSynchronize(); CUERR;
#pragma unroll
        for (int k = 0; k < nKernel; k++)
        {
            // Launch a kernel on the GPU
            cudaLaunchCooperativeKernel((void*)dynamicTransformationKernel, dimGrid, dimBlock, kernelArgs[k], 0, stream[k]);
        }
        // cudaDeviceSynchronize(); CUERR;
#pragma unroll
        for (int k = 0; k < nKernel; k++) { cudaStreamSynchronize(stream[k]); CUERR; }
        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end);
    }
    // Check for any errors launching the kernel
    cudaGetLastError(); CUERR;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    for (int k = 0; k < nKernel; k++) { cudaStreamDestroy(stream[k]); CUERR; }

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(a, dev_a[0], size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

    cudaFree(dev_a);
    cudaStatus = cudaGetLastError();
    return cudaStatus;
}

// Launch n number of cooperative parallel kernels, with parallel cpu thread kernel launch
cudaError_t dynamicTransformation(float* a, unsigned int size, const int n, const int t)
{
    float* dev_a[nKernel];
    void* kernelArgs[nKernel][1];
    cudaError_t cudaStatus;
    cudaStream_t stream[nKernel];
    dim3 dimBlock(size, 1, 1);
    dim3 dimGrid(1, 1, 1);
    thread threads[nThread];

    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0); CUERR;
    // memory allocation and initialization
    for (int k = 0; k < nThread; k++)
    {
        threads[k] = thread(initLaunch, dev_a, a, kernelArgs, stream, size, k);
    } JOIN;

    // Setting stream attributes
    // cudaStreamSetAttribute(stream, )

    // Setting device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);  CUERR;

    deviceProp.concurrentKernels = 1;
    deviceProp.cooperativeLaunch = 1;

    // Benchmark nKernel number of kernel launch
    printf("Benchmark parallel, nKernel number of kernel launch, with parallel cpu thread launches, more runs to get through GPU clock ramp-up\n");
    for (int i = 0; i < transient; i++)
    {
        begin = std::chrono::steady_clock::now();
        // parralel stream sync
        for (int k = 0; k < nThread; k++)
        {
            threads[k] = thread(parallelStreamSync, stream, k);
            
        } JOIN;
        // cudaDeviceSynchronize(); CUERR;
        // for (int k = 0; k < nThread; k++)
        for (int k = 0; k < nThread; k++)
        {
            // Launch a kernel on the GPU
            threads[k] = thread(launchKernel, kernelArgs, stream, size, k);

            // cudaLaunchCooperativeKernel((void*)dynamicTransformationKernel, dimGrid, dimBlock, kernelArgs[k], 0, stream[k]);

        } JOIN;
        // cudaDeviceSynchronize(); CUERR;
        // parralel stream sync
        for (int k = 0; k < nThread; k++)
        {
            threads[k] = thread(parallelStreamSync, stream, k);

        } JOIN;
        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end);
    }
    // Check for any errors launching the kernel
    cudaGetLastError(); CUERR;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    for (int k = 0; k < nKernel; k++) { cudaStreamDestroy(stream[k]); CUERR; }

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(a, dev_a[0], size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

    cudaFree(dev_a);
    cudaStatus = cudaGetLastError();
    return cudaStatus;
}

// Launch n number of cooperative parallel kernels, with parallel cpu thread kernel launch, full parallel launch
cudaError_t dynamicTransformation(float* a, unsigned int size, const int n, const int t, void* dummy)
{
    float* dev_a[nKernel];
    void* kernelArgs[nKernel][1];
    cudaError_t cudaStatus;
    cudaStream_t stream[nKernel];
    dim3 dimBlock(size, 1, 1);
    dim3 dimGrid(1, 1, 1);
    thread threads[nThread];

    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0); CUERR;
    // memory allocation and initialization
    for (int k = 0; k < nThread; k++)
    {
        threads[k] = thread(initLaunch, dev_a, a, kernelArgs, stream, size, k);
    } JOIN;

    // Setting stream attributes
    // cudaStreamSetAttribute(stream, )

    // Setting device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);  CUERR;

    deviceProp.concurrentKernels = 1;
    deviceProp.cooperativeLaunch = 1;

    // Benchmark nKernel number of kernel launch
    printf("Benchmark parallel, nKernel number of kernel launch, with parallel cpu thread launches, more runs to get through GPU clock ramp-up\n");
    for (int i = 0; i < transient; i++)
    {
        begin = std::chrono::steady_clock::now();
        // parralel stream sync
        for (int k = 0; k < nThread; k++)
        {
            threads[k] = thread( syncLaunchSync, kernelArgs, stream, size, k );

        } JOIN;
        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end);
    }
    // Check for any errors launching the kernel
    cudaGetLastError(); CUERR;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    for (int k = 0; k < nKernel; k++) { cudaStreamDestroy(stream[k]); CUERR; }

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(a, dev_a[0], size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

    cudaFree(dev_a);
    cudaStatus = cudaGetLastError();
    return cudaStatus;
}

void test(float* c, float* check, int arraySize) {
    for (int i = 0; i < arraySize; i++)
    {
        if (c[i] != check[i])
        {
            // printf("calculation failed: c[%d] = %d, check[%d] = %d\n", i, c[i], i, check[i]);
        }
    }
    printf("calculation completed successfuly ???\n");
}

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
}

void initLaunch(float** dev_a, float* a, void* kernelArgs[nKernel][1], cudaStream_t stream[nKernel], int size, int k)
{
    // std::cout << "Thread " << k << " started\n";
    for ( int i = k * nKernelPerThread; ( i - k * nKernelPerThread ) < nKernelPerThread; i++ )
    {
        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaMalloc((void**)&dev_a[i], size * sizeof(float)); CUERR;
        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(dev_a[i], a, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;
        // Setting kernel argument
        kernelArgs[i][0] = (void*)&dev_a[i];
        // Creating the steam 
        cudaStreamCreate(&(stream[i])); CUERR;
        // cout << i << endl;
    }
}

void parallelStreamSync(cudaStream_t stream[nKernel], int k)
{
    for ( int i = k * nKernelPerThread; ( i - k * nKernelPerThread ) < nKernelPerThread; i++ )
    {
        cudaStreamSynchronize(stream[i]); CUERR;
    }
}

void launchKernel( void* kernelArgs[nKernel][1], cudaStream_t stream[nKernel], int size, int k )
{
    dim3 dimBlock(size, 1, 1);
    dim3 dimGrid(1, 1, 1);
    for (int i = k * nKernelPerThread; (i - k * nKernelPerThread) < nKernelPerThread; i++)
    {
        cudaLaunchCooperativeKernel((void*)dynamicTransformationKernel, dimGrid, dimBlock, kernelArgs[k], 0, stream[k]);
    }
}

void syncLaunchSync(void* kernelArgs[nKernel][1], cudaStream_t stream[nKernel], int size, int k)
{
    dim3 dimBlock(size, 1, 1);
    dim3 dimGrid(1, 1, 1);

    for (int i = k * nKernelPerThread; (i - k * nKernelPerThread) < nKernelPerThread; i++)
    {
        cudaStreamSynchronize(stream[i]); CUERR;
    }
    // cudaDeviceSynchronize(); CUERR;
    for (int i = k * nKernelPerThread; (i - k * nKernelPerThread) < nKernelPerThread; i++)
    {
        cudaLaunchCooperativeKernel((void*)dynamicTransformationKernel, dimGrid, dimBlock, kernelArgs[k], 0, stream[k]);
    }
    // cudaDeviceSynchronize(); CUERR;
    for (int i = k * nKernelPerThread; (i - k * nKernelPerThread) < nKernelPerThread; i++)
    {
        cudaStreamSynchronize(stream[i]); CUERR;
    }
}