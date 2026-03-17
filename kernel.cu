//-------------------------------------------------------
//----------DEFINE RUN PARAMETERS------------------------
//-------------------------------------------------------

#define run_gsync_restruct_int_friend 1
#define DEVICE_ID 0
#define ITERATIONS 500000
#define SNAPSHOT 1			// 1 => save snapshots
#define SNAP_MODULUS 5000		// take snapshots if k % SNAP_MODULUS == 0

#define CUERR {																					\
cudaError_t err;																				\
    if ((err = cudaGetLastError()) != cudaSuccess) {											\
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
        exit(1);																				\
    }																							\
}

#define PASS printf(" PASSED FILE: %s, LINE: %d\n", __FILE__, __LINE__);

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "copy_gridsync.cpp"
#include "iterate_gridsync_restructure_int_friend.cu"
#include "unifi.cu"
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
    // array_to_text(u, res);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset(); CUERR;
    delete[] u;
    delete[] u_sol;
    delete[] v;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res)
{
    float* dev_ua = 0;
    float* dev_ub = 0;
    float* dev_v = 0;
    float* dev_2_u = 0;
	float* dev_snap = 0;
	void* kernelArgs[6];
	float* uSnap = new float[ size * int( ITERATIONS / SNAP_MODULUS / 2 ) ];
    //measure at this iteration
    int iteration_limit_per_2 = ITERATIONS / 2;

    // cooperative kernel launch parameters
	if ( SNAPSHOT ){
		kernelArgs[0] = (void*)&(dev_ua);
		kernelArgs[1] = (void*)&(dev_ub);
		kernelArgs[2] = (void*)&(dev_v);
		kernelArgs[3] = (void*)&res;
		kernelArgs[4] = (void*)&iteration_limit_per_2;
		kernelArgs[5] = (void*)&(dev_snap);
	}
	else{
		kernelArgs[0] = (void*)&(dev_ua);
		kernelArgs[1] = (void*)&(dev_ub);
		kernelArgs[2] = (void*)&(dev_v);
		kernelArgs[3] = (void*)&res;
		kernelArgs[4] = (void*)&iteration_limit_per_2;
	}

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(8, 8, 1);

    //measure elapsed time
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(DEVICE_ID); CUERR;

    //-----------------------------------------------------------------------------------------------------------
    //---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL -------------------------------
    //-----------------------------------------------------------------------------------------------------------
    if (run_gsync_restruct_int_friend)
    {
        cout << "gridsync restruct \n";
        // Allocate GPU buffers
		if ( SNAPSHOT ){
			init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, &dev_snap); CUERR;
		}
		else{
			init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID); CUERR;
		}
        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();

        if ( SNAPSHOT ){ cudaLaunchCooperativeKernel((void*)iterate_snap, dimGrid, dimBlock, kernelArgs); }
		else{ cudaLaunchCooperativeKernel((void*)iterate, dimGrid, dimBlock, kernelArgs); }
        cudaDeviceSynchronize(); CUERR;

        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end);

        // Check for any errors launching the kernel
        cudaGetLastError(); CUERR;

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaDeviceSynchronize(); CUERR;

        // Copy output vector from GPU buffer to host memory.
        cudaMemcpy(u, dev_ua, size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

		cudaMemcpy(uSnap, dev_snap, size * sizeof(float) * int( ITERATIONS / SNAP_MODULUS / 2 ), cudaMemcpyDeviceToHost); CUERR;

        cudaDeviceSynchronize(); CUERR;

        array_to_text(u, res);

		snap_to_text(uSnap, res);

        test(u, size);
    }

// Error:
    cudaFree(dev_v); CUERR;
    cudaFree(dev_2_u); CUERR;
	cudaFree(dev_snap); CUERR;
	delete[] uSnap;

    return cudaStatus;
}

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    //measure at this iteration
    int iteration_limit_per_2 = ITERATIONS / 2;
    int average_iteration_runtime;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[us]" << endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel) : " << average_iteration_runtime << "[ns]" << endl;
}
