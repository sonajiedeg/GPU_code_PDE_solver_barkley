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
static __device__ __forceinline__ void copy_g_to_r_n_B(float* g_ua, float* g_v, float r_u[nKernel], float r_v[nKernel], int* n_offset)
{
    for (int k = 0; k < nKernel; k++)
    {
        r_u[k] = g_ua[n_offset[k]];
        r_v[k] = g_v[n_offset[k]];
    }
}
/*
// copy from registers to global memory
static __device__ __forceinline__ void copy_r_to_g_n_B(float* g_ub, float* g_v, float* r_u, float* r_v, int* n_offset)
{
    for (int k = 0; k < nKernel; k++)
    {
        g_ub[n_offset[k]] = r_u[k];
        // g_v[n_offset[k]] = r_v[k];
    }
}
*/

// write n sim result to a txt file
void array_to_text_n(float* u_sol, int res)
{
    const unsigned int s = res * res;
    fstream newfile;
    newfile.open("u_sol_n.txt", ios::out);
    if (newfile.is_open())
    {
        newfile << scientific << setprecision(17);
        // max_digits10 ensures no loss of precision

        for (int i = 0; i < s * nKernel; i++)
        {
            newfile << u_sol[i] << "\n";
        }

        newfile.close();
    }
};
