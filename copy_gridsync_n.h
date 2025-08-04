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

#include <iomanip>
#include <limits> 

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;

// copy from global memory to the registers
static __device__ __forceinline__ void copy_g_to_r_n_B(float* g_ua, float* g_v, float r_u[n_kernel], float r_v[n_kernel], int* n_offset)
{
    for (int k = 0; k < n_kernel; k++)
    {
        r_u[k] = g_ua[n_offset[k]];
        r_v[k] = g_v[n_offset[k]];
    }
}

// copy from registers to global memory
static __device__ __forceinline__ void copy_r_to_g_n_B(float* g_ub, float* g_v, float* r_u, float* r_v, int* n_offset)
{
    for (int k = 0; k < n_kernel; k++)
    {
        g_ub[n_offset[k]] = r_u[k];
        // g_v[n_offset[k]] = r_v[k];
    }
}