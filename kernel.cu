//-------------------------------------------------------
//----------DEFINE RUN PARAMETERS------------------------
//-------------------------------------------------------


#define run_gsync_restruct_int_friend 1
#define run_bubble_solver 1
#define DEVICE_ID 0
// #define ITERATIONS 1000000

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
#include "param.h"
#include "paramSetup.cpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdio>
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
#include "solve2.cuh"

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;


cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res, SimParamStruct params);
cudaError_t iterate_prep_segment(float* u, float* v, float* u_sol, unsigned int size, int res, SimParamStruct params);
void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params);

int main()
{
    const unsigned int res = 256; // swich
    const unsigned int size = res * res;
    float* u = new float[size];
    float* v = new float[size];
    float* u_sol = new float[size];
	// read ic
    // text_to_array(u, v); 

    // Add vectors in parallel.
	cudaError_t cudaStatus;
	SimParamStruct params = paramSetup();
	if(params.segment){ cudaStatus = solve2(u, v, u_sol, size, res, params); CUERR; }
    // if(params.segment){ cudaStatus = iterate_prep_segment(u, v, u_sol, size, res, params); CUERR; }
	else{ cudaStatus = iterate_prep(u, v, u_sol, size, res, params); CUERR; }
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
cudaError_t iterate_prep_segment(float* u, float* v, float* u_sol, unsigned int size, int res, SimParamStruct params)
{
	float* u2 = new float[ size ];
	float* u1 = new float[ size ];
	for(int i = 0; i<size; i++){ u1[i]=u[i]; u2[i]=u[i]; }
	cudaError_t cudaStatus;
	int iteration_limit_per_2 = 0;

	int activeSteps;
	float* dev_ua = 0;
	float* dev_ub = 0;
	float* dev_2_u = 0;
	float* dev_snap = 0;
	void* kernelArgs[6];
	int kStepStart = 0;
	kernelArgs[0] = (void*)&(dev_ua);
	kernelArgs[1] = (void*)&(dev_ub);
	kernelArgs[2] = (void*)&iteration_limit_per_2;
	kernelArgs[3] = (void*)&(dev_snap);
	kernelArgs[4] = (void*)&params;
	kernelArgs[5] = (void*)&kStepStart;

	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(4, 4, 1); // swich

	//measure elapsed time
	std::chrono::steady_clock::time_point beginTot;
	std::chrono::steady_clock::time_point endTot;

	beginTot = std::chrono::steady_clock::now();


	for(int segment = 0; segment < params.segments; segment++ )
	{
		activeSteps = ( segment < params.segments-1 ) ? params.maxStepPerSegment : params.remainder;
		float* uSnap = new float[ size * int( activeSteps ) ];

		//measure at this iteration
		iteration_limit_per_2 += activeSteps / 2;
		// printf("active steps: %d\n", activeSteps);
		// printf("iter lim / 2: %d\n", iteration_limit_per_2);
		printf("segment/segments: %d / %d \n", segment, params.segments);

		kStepStart = segment * params.maxStepPerSegment / 2;

		// cooperative kernel launch parameters
		
		//measure elapsed time
		// std::chrono::steady_clock::time_point begin;
		// std::chrono::steady_clock::time_point end;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(DEVICE_ID); CUERR;

		//-----------------------------------------------------------------------------------------------------------
		//---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL -------------------------------
		//-----------------------------------------------------------------------------------------------------------
		if (run_gsync_restruct_int_friend)
		{
			// cout << "gridsync restruct \n";
			// Allocate GPU buffers


			// init(u1, u2, dev_2_u, dev_ua, dev_ub, dev_snap, params, activeSteps)
			init(u1, u2, &dev_2_u, &dev_ua, &dev_ub, &dev_snap, params, activeSteps); CUERR;
			// init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, &dev_snap, params, &dev_snap_gradx, &dev_snap_grady); CUERR;
			cudaDeviceSynchronize();
			// ---------- Launch a kernel on the GPU with one thread for each element. ----------
			// begin = std::chrono::steady_clock::now();
			cudaLaunchCooperativeKernel((void*)iterate_snap_segment, dimGrid, dimBlock, kernelArgs);
			cudaDeviceSynchronize(); CUERR;

			// end = std::chrono::steady_clock::now();
			// print_runtimes(begin, end, params);

			// Check for any errors launching the kernel
			cudaGetLastError(); CUERR;

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaDeviceSynchronize(); CUERR;

			
			// Copy output vector from GPU buffer to host memory.
			cudaMemcpy(u2, dev_ua, size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;
			cudaMemcpy(u1, dev_ub, size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

			if ( params.SNAPSHOT ) { cudaMemcpy(uSnap, dev_snap, size * sizeof(float) * int( activeSteps ), cudaMemcpyDeviceToHost); CUERR; }

			cudaDeviceSynchronize(); CUERR;

			char fname[100];
			snprintf(fname, sizeof(fname), "u_snap_f_%.2f_df_%.2f_s_%d.bin",params.f, params.df,segment);
			snap_to_text(uSnap, params, fname, activeSteps);
		}

	// Error:
		cudaFree(dev_2_u); CUERR;
		
		cudaFree(dev_snap); CUERR;
		
		delete[] uSnap;
	}

	endTot = std::chrono::steady_clock::now();
	print_runtimes(beginTot, endTot, params);
	
    return cudaStatus;
}


cudaError_t iterate_prep(float* u, float* v, float* u_sol, unsigned int size, int res, SimParamStruct params)
{
    float* dev_ua = 0;
    float* dev_ub = 0;
    float* dev_v = 0;
    float* dev_2_u = 0;
	float* dev_snap = 0;
	float* dev_snap_gradx = 0;
	float* dev_snap_grady = 0;
	void* kernelArgs[9];
	float* uSnap = new float[ size * int( params.tsc ) ];
	float* uSnapGradx = new float[ size * int( params.tsc ) ];
	float* uSnapGrady = new float[ size * int( params.tsc ) ];
    //measure at this iteration
    int iteration_limit_per_2 = params.ITERATIONS / 2;

    // cooperative kernel launch parameters
	if ( params.segment ){
		kernelArgs[0] = (void*)&(dev_ua);
		kernelArgs[1] = (void*)&(dev_ub);
		kernelArgs[2] = (void*)&(dev_v);
		kernelArgs[3] = (void*)&res;
		kernelArgs[4] = (void*)&iteration_limit_per_2;
		kernelArgs[5] = (void*)&(dev_snap);
		kernelArgs[6] = (void*)&params;
	}
	else if ( params.all ){
		kernelArgs[0] = (void*)&(dev_ua);
		kernelArgs[1] = (void*)&(dev_ub);
		kernelArgs[2] = (void*)&(dev_v);
		kernelArgs[3] = (void*)&res;
		kernelArgs[4] = (void*)&iteration_limit_per_2;
		kernelArgs[5] = (void*)&(dev_snap);
		kernelArgs[6] = (void*)&params;
		kernelArgs[7] = (void*)&(dev_snap_gradx);
		kernelArgs[8] = (void*)&(dev_snap_grady);
	}
	else if ( params.SNAPSHOT ){
		kernelArgs[0] = (void*)&(dev_ua);
		kernelArgs[1] = (void*)&(dev_ub);
		kernelArgs[2] = (void*)&(dev_v);
		kernelArgs[3] = (void*)&res;
		kernelArgs[4] = (void*)&iteration_limit_per_2;
		kernelArgs[5] = (void*)&(dev_snap);
		kernelArgs[6] = (void*)&params;
	}
	else{
		kernelArgs[0] = (void*)&(dev_ua);
		kernelArgs[1] = (void*)&(dev_ub);
		kernelArgs[2] = (void*)&(dev_v);
		kernelArgs[3] = (void*)&res;
		kernelArgs[4] = (void*)&iteration_limit_per_2;
		kernelArgs[5] = (void*)&params;
	}

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(4, 4, 1);

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
		if (params.segment){
			init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, &dev_snap, params, &dev_snap_gradx, &dev_snap_grady); CUERR;
		}
		else if ( params.all){
			init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, &dev_snap, params, &dev_snap_gradx, &dev_snap_grady); CUERR;
		}
		else if ( params.SNAPSHOT ){
			init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, &dev_snap, params); CUERR;
		}
		else{
			init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, params); CUERR;
		}
        cudaDeviceSynchronize();
        // ---------- Launch a kernel on the GPU with one thread for each element. ----------
        begin = std::chrono::steady_clock::now();
		// if ( params.segment ){ cudaLaunchCooperativeKernel((void*)iterate_snap_segment, dimGrid, dimBlock, kernelArgs); }
		if ( params.all ){ cudaLaunchCooperativeKernel((void*)iterate_snap_grad, dimGrid, dimBlock, kernelArgs); }
		else if ( params.SNAPSHOT ){ cudaLaunchCooperativeKernel((void*)iterate_snap, dimGrid, dimBlock, kernelArgs); }
		else{ cudaLaunchCooperativeKernel((void*)iterate, dimGrid, dimBlock, kernelArgs); }
        cudaDeviceSynchronize(); CUERR;
		

        end = std::chrono::steady_clock::now();
        print_runtimes(begin, end, params);

        // Check for any errors launching the kernel
        cudaGetLastError(); CUERR;

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaDeviceSynchronize(); CUERR;

		
        // Copy output vector from GPU buffer to host memory.
        cudaMemcpy(u, dev_ua, size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

		
		if ( params.all ) {
			cudaMemcpy(uSnapGradx, dev_snap_gradx, size * sizeof(float) * int( params.tsc ), cudaMemcpyDeviceToHost); CUERR;
			
			cudaMemcpy(uSnapGrady, dev_snap_grady, size * sizeof(float) * int( params.tsc ), cudaMemcpyDeviceToHost); CUERR;
		}
		if ( params.SNAPSHOT ) { cudaMemcpy(uSnap, dev_snap, size * sizeof(float) * int( params.tsc ), cudaMemcpyDeviceToHost); CUERR; }

        cudaDeviceSynchronize(); CUERR;

        // array_to_text(u, res);
		
		snap_to_text(uSnap, res, params, "u_snap.bin");
		
		if ( params.all ) {
			snap_to_text(uSnapGradx, res, params, "u_snap_gradx.bin");
		
			snap_to_text(uSnapGrady, res, params, "u_snap_grady.bin");
		}
		
        // test(u, size);
    }

// Error:
    cudaFree(dev_v); CUERR;
	
    cudaFree(dev_2_u); CUERR;
	
	cudaFree(dev_snap); CUERR;
	
	cudaFree(dev_snap_gradx); CUERR;
	
	cudaFree(dev_snap_grady); CUERR;
	
	delete[] uSnap;
	
	delete[] uSnapGradx;
	
	delete[] uSnapGrady;
	

    return cudaStatus;
}

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params)
{
    //measure at this iteration
    int iteration_limit_per_2 = params.ITERATIONS / 2;
    int average_iteration_runtime;
    int total_runtime;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel loop) : " << average_iteration_runtime << "[us]" << endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel loop) : " << average_iteration_runtime << "[ns]" << endl;

    total_runtime = std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();
	cout << "total runtime : " << total_runtime << "[s]" << endl; 
	total_runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
	cout << "total runtime : " << total_runtime << "[ms]" << endl; 
	total_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count();
	cout << "total runtime : " << total_runtime << "[us]" << endl; 
	
	cout << "number of timesteps : " << params.ITERATIONS << endl;

}
