// unified
//-------------------------------------------------
//---------- DEFINE WHICH KERNELS TO RUN ----------
//-------------------------------------------------
#define run_naive 0
#define run_global_copy 0
#define run_gsync 0
#define run_gsync_int_friend 0
#define run_gsync_int_friend_persistent_L2 0
#define run_gsync_restruct 0
#define run_gsync_restruct_int_friend 0
#define run_gsync_restruct_int_friend_v2 0
#define run_gsync_restruct_int_friend_v2_threadfence 0
#define run_gsync_restruct_int_friend_v2_threadfence_for 0
#define run_n_kernel 1
#define nKernel 8  // can be in the set of { ? }
                    // in case n = 32: each block has 1 warp
                    // n = 8 => 256 / 8 = 32 => if n increases more then a warp wont be filled with threads (case A)
#define jmax 32 // = res / nKernel :   256 / 2 = 128
//                                      256 / 4 = 64
//                                      256 / 8 = 32

#define CUERR {                                                            \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                                           \
    }                                                                      \
}

#define PASS printf(" PASSED FILE: %s, LINE: %d\n", __FILE__, __LINE__);

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "copy_gridsync.h"
#include "copy_gridsync_n.cuh"
#include "iterate_gridsync.h"
#include "iterate_gridsync_restructure.h"
#include "iterate_gridsync_restructure_n.cuh"
#include "iterate_gridsync_restructure_int_friend.h"
#include "iterate_gridsync_restructure_int_friend_v2.h"
#include "iterate_gridsync_restructure_int_friend_v2_threadfence.h"
#include "iterate_gridsync_restructure_int_friend_v2_threadfence_for.h"
#include "iterate_gridsync_intfriend.h"
#include "iterate_naive.h"
#include "g_to_g_test.h"
#include "unifi.h"
#include "unifi_n.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

#define DEVICE_ID 1
#define ITERATIONS 5000

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;


cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res);
void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);

int main()
{
    const unsigned int res = 256;
    const unsigned int size = res * res;
    float* u = new float[size];
    float* v = new float[size];
    float* u_sol = new float[size];

    text_to_array(u, v);

    // Add vectors in parallel.
    cudaError_t cudaStatus = iterate_prep(u, v, u_sol, size, res); CUERR;
    PASS;
    // array_to_text(u, res);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset(); CUERR;
    PASS;
    delete[] u;
    delete[] u_sol;
    delete[] v;
    PASS;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res)
{
    //const unsigned int res = 128;
    float* dev_ua = 0;
    float* dev_ub = 0;
    float* dev_v = 0;
    float* dev_2_u = 0;
    // float* u_2 = 0;

    //measure at this iteration
    int iteration_limit_per_2 = ITERATIONS / 2;
    // int measure_at = iteration_limit_per_2 - 1;

    // cooperative kernel launch parameters
    void* kernelArgs[] = { (void*)&dev_ua, (void*)&dev_ub, (void*)&dev_v, (void*)&res, (void*)&iteration_limit_per_2 };
    int* ivar;
    void* kernel_args_for[] = { (void*)&dev_ua, (void*)&dev_ub, (void*)&dev_v, (void*)&res, (void*)&iteration_limit_per_2, (void*)&ivar };
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(8, 8, 1);

    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    // int average_iteration_runtime;

    // declare persisting L2 cache variables
    cudaDeviceProp prop;
    cudaStream_t stream_1;
    size_t L2_size;

    cudaError_t cudaStatus;

    // for testing return
    // bool test_bool;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(DEVICE_ID); CUERR;
    
    //-------------------------------------------------------------------------------------------------
    //---------------------------------------- GLOBAL COPY KERNEL -------------------------------------
    //-------------------------------------------------------------------------------------------------
    if (run_global_copy)
    {
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID); CUERR;

        cudaDeviceSynchronize();
        g_to_g_test << <dimGrid, dimBlock >> > (dev_ua, dev_ub);
        cudaDeviceSynchronize();
    }

    //-------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC KERNEL ----------------------------------------
    //-------------------------------------------------------------------------------------------------
    if (run_gsync)
    {
        cout << "gridsync \n";
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterateKernel, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }
    //-------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT KERNEL -------------------------------
    //-------------------------------------------------------------------------------------------------
    if (run_gsync_restruct)
    {
        cout << "gridsync restruct \n";
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterate_kernel_gsync_restruct, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }

    //-----------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL -------------------------------
    //-----------------------------------------------------------------------------------------------------------
    if (run_gsync_restruct_int_friend)
    {
        cout << "gridsync restruct \n";
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterate_kernel_gsync_restruct_int_friend, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }

    //---------------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL V2 -------------------------------
    //---------------------------------------------------------------------------------------------------------------
    if (run_gsync_restruct_int_friend_v2)
    {
        cout << "gridsync restruct \n";
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterate_kernel_gsync_restruct_int_friend_v2, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }

    //--------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL V2 THREADFENCE-------------------------------
    //--------------------------------------------------------------------------------------------------------------------------
    if (run_gsync_restruct_int_friend_v2_threadfence)
    {
        cout << "gridsync restruct \n";
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterate_kernel_gsync_restruct_int_friend_v2_threadfence, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }

    //------------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL V2 THREADFENCE FOR-------------------------------
    //------------------------------------------------------------------------------------------------------------------------------
    if (run_gsync_restruct_int_friend_v2_threadfence_for)
    {
        int count = 0;
        for (size_t i = 0; i < 10000; i++)
        {
            // cout << "gridsync restruct \n";
            // Allocate GPU buffers
            cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

            cudaDeviceSynchronize();
            // ---------- Launch a kernel on the GPU with one thread for each element. ----------
            begin = std::chrono::steady_clock::now();

            cudaLaunchCooperativeKernel((void*)iterate_kernel_gsync_restruct_int_friend_v2_threadfence, dimGrid, dimBlock, kernelArgs);
            cudaDeviceSynchronize();

            end = std::chrono::steady_clock::now();

            print_runtimes(begin, end);

            // destroy stream
            //cudaStreamDestroy(stream_1);

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

            // Copy output vector from GPU buffer to host memory.
            cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

            cudaStatus = cudaDeviceSynchronize();

            array_to_text(u, res);

            bool tmp = test(u, size);
            if (!tmp)
            {
                printf("test failed at %d", i);
                count++;
            }
            
        }
        printf("failed %d times", count);
    }

    //----------------------------------------------------------------------------------------------
    //---------------------------------------- NAIVE KERNEL ----------------------------------------
    //----------------------------------------------------------------------------------------------
    if (run_naive)
    {

        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        // Launch a kernel on the GPU with one thread for each element.
        /*
        for (size_t i = 0; i < iteration_limit_per_2; i++)
        {
            iterateKernelExpNaive << <dimGrid, dimBlock >> > (dev_ua, dev_ub, dev_v, res);
            cudaDeviceSynchronize();

            no_flux_kernel << < dimGrid, dimBlock >> > (dev_ub, res);
            cudaDeviceSynchronize();

            iterateKernelExpNaive << < dimGrid, dimBlock >> > (dev_ub, dev_ua, dev_v, res);
            cudaDeviceSynchronize();

            no_flux_kernel << < dimGrid, dimBlock >> > (dev_ua, res);
            cudaDeviceSynchronize();
        }
        */

        iterateKernelExpNaive << <dimGrid, dimBlock >> > (dev_ua, dev_ub, dev_v, res);
        cudaDeviceSynchronize();

        no_flux_kernel << < dimGrid, dimBlock >> > (dev_ub, res);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }

    //------------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC INTFIENDLY KERNEL ----------------------------------------
    //------------------------------------------------------------------------------------------------------------
    if (run_gsync_int_friend)
    {
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterate_kernel_gi, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        //cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }
    //------------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC INTFIENDLY PERSISTENT L2 KERNEL --------------------------
    //------------------------------------------------------------------------------------------------------------
    if (run_gsync_int_friend_persistent_L2)
    {
        // Allocate GPU buffers
        cudaStatus = init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed! -----init-----"); goto Error; }

        // ----- Set up persisting cache -----
        // set persisting cache maximum limit
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        L2_size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, L2_size);
        // setting stream parameters
        cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
        cudaStreamCreate(&stream_1);
        stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(dev_2_u); // Global Memory data pointer
        stream_attribute.accessPolicyWindow.num_bytes = 2 * res * res * sizeof(float);                    // Number of bytes for persistence access.
        // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
        stream_attribute.accessPolicyWindow.hitRatio = 1;                          // Hint for cache hit ratio
        stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
        stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

        //Set the attributes to a CUDA stream of type cudaStream_t
        cudaStreamSetAttribute(stream_1, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        cudaLaunchCooperativeKernel((void*)iterate_kernel_gi, dimGrid, dimBlock, kernelArgs);
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        print_runtimes(begin, end);

        // destroy stream
        cudaStreamDestroy(stream_1);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Gsync Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

        cudaStatus = cudaDeviceSynchronize();

        array_to_text(u, res);

        test(u, size);
    }

    //-------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT KERNEL RUN N KERNELS -----------------
    //-------------------------------------------------------------------------------------------------
    if (run_n_kernel)
    {
        // case B
        dim3 dimBlock_n(32, 32 / nKernel, 1); // case B
        dim3 dimGrid(8, 8, 1);
        float** dev_ua_n = (float**)malloc( nKernel * sizeof(float*) );
        float** dev_ub_n = (float**)malloc( nKernel * sizeof(float*) );
        float** dev_v_n = (float**)malloc( nKernel * sizeof(float*) );
        cout << "n kernel \n";
        // Allocate GPU buffers
        init_n(u, v, &dev_ua_n, &dev_ub_n, &dev_v_n, size, nKernel, DEVICE_ID); CUERR;
        // organising kernel inputs into an array
        const int argCount = 5;
        PASS;
        void*** kernelArgsN  = (void***)malloc( nKernel * sizeof(void**) );
        // allocate memory
        for(int i = 0; i < nKernel; i++){ kernelArgsN[i] = (void**)malloc( argCount * sizeof(void*) ); }
        // allocate memory is missing
        PASS;
        for(int k = 0; k < nKernel; k++)
        {
            PASS;
            kernelArgsN[k][0] = (void*)&(dev_ua_n[k]);
            kernelArgsN[k][1] = (void*)&(dev_ub_n[k]);
            kernelArgsN[k][2] = (void*)&(dev_v_n[k]);
            kernelArgsN[k][3] = (void*)&res;
            kernelArgsN[k][4] = (void*)&iteration_limit_per_2;
        }
        PASS;
        // Setting up n number of streams
        cudaStream_t stream_n[nKernel];
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, DEVICE_ID);  CUERR;
        deviceProp.concurrentKernels = 1;
        deviceProp.cooperativeLaunch = 1;

        // create streams
        for(int k = 0; k < nKernel; k++) { cudaStreamCreate(&stream_n[k]); CUERR; }
        begin = std::chrono::steady_clock::now();
        // sync streams
        for(int k = 0; k < nKernel; k++){ cudaStreamSynchronize(stream_n[k]); CUERR; }
        // launch kernels
        for(int k = 0; k < nKernel; k++)
        {
            cout << "n =" << k << endl;
            // void* kernelArgs_n[] = { (void*)&(dev_ua_n[k]), (void*)&(dev_ub_n[k]), (void*)&(dev_v_n[k]), (void*)&res, (void*)&iteration_limit_per_2 };
            cudaLaunchCooperativeKernel((void*)iterate_kernel_gsync_restruct_n_B, dimGrid, dimBlock_n, kernelArgsN[k], 0, stream_n[k]);    
        } // end of launching n kernels
        // sync streams
        for(int k = 0; k < nKernel; k++){ cudaStreamSynchronize(stream_n[k]); CUERR;}
        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end);
        // Check for any errors launching the kernel
        cudaGetLastError(); CUERR;
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaDeviceSynchronize(); CUERR;
        // Copy output vector from GPU buffer to host memory.
        // cudaMemcpy(u, dev_ua, size * sizeof(int), cudaMemcpyDeviceToHost); CUERR;
        cudaDeviceSynchronize();
        array_to_text(u, res);
        // test_n(u, size);
        PASS;
    }
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------


    cudaFree(dev_ua);
    cudaFree(dev_ub);
    cudaFree(dev_v);
    cudaFree(dev_2_u);

Error:
    cudaFree(dev_ua);
    cudaFree(dev_ub);
    cudaFree(dev_v);
    cudaFree(dev_2_u);
    //cudaFree(dev_u_sol);

    return cudaStatus;
}

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    //measure at this iteration
    int iteration_limit_per_2 = ITERATIONS / 2;
    int average_iteration_runtime;
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    // std::cout << "Time difference (whole simulation) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[us]" << endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[ns]" << endl;
}