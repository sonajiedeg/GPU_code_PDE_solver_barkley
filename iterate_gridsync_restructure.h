    #include "cuda_runtime.h"



static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset);

__global__ void iterate_kernel_gsync_restruct(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();

    if (i < res && j < res)            // INTO REG
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)       // INTO REG
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_v, ThreadOffset);
        //whole_grid.sync();

//#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            iterate_without_laplace(r_ua, r_v, &r_un, &r_vn);
            
            // ----- Laplace -----
            laplace(g_ua, &r_un, res, i, j, ThreadOffset);

            // ----- "Unload" Data And No Flux Bc -----
            copy_r_to_g_and_no_flux(g_ub, r_un, res, i, j, ThreadOffset);
            whole_grid.sync();

            // ----- Iteration step 2 
            iterate_without_laplace(r_un, r_vn, &r_ua, &r_v);
            
            // ----- Laplace -----
            laplace(g_ub, &r_ua, res, i, j, ThreadOffset);

            // ----- "Unload" Data And No Flux Bc -----
            copy_r_to_g_and_no_flux(g_ua, r_ua, res, i, j, ThreadOffset);
            whole_grid.sync();
        }
    }
}

// no flux boundary condition
static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset)
{
    // ----- Copy To GlobalMem -----
    if ( i >= 1 && j >= 1 && i <= res - 2 && j <= res - 2)
    {
        g_ub[ThreadOffset] = r_un;
    }

    // ----- No Flux BC -----
    //s
    if (j == 1) { g_ub[ThreadOffset - res] = r_un; }
    //w
    if (i == 1) { g_ub[ThreadOffset - 1] = r_un; }
    //e
    if (i == res - 2) { g_ub[ThreadOffset + 1] = r_un; }
    //n
    if (j == res - 2) { g_ub[ThreadOffset + res] = r_un; }
    //sw
    if (i == 1 && j == 1) { g_ub[ThreadOffset - res - 1] = r_un; }
    //se
    if (i == res - 2 && j == 1) { g_ub[ThreadOffset - res + 1] = r_un; }
    //nw
    if (i == 1 && j == res - 2) { g_ub[ThreadOffset + res - 1] = r_un; }
    //ne
    if (i == res - 2 && j == res - 2) { g_ub[ThreadOffset + res + 1] = r_un; }
}

