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
        float r_va;
        float r_ub;
        float r_vb;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
        //whole_grid.sync();

        // ----- Iteration step 1 -----
        iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
            // ----- Laplace 1 -----
            laplace(g_ua, &r_ub, res, i, j, ThreadOffset);

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, r_vb, &r_ua, &r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux(g_ub, r_ub, res, i, j, ThreadOffset);
            whole_grid.sync();
            
            // ----- Laplace 2 -----
            laplace(g_ub, &r_ua, res, i, j, ThreadOffset);

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux(g_ua, r_ua, res, i, j, ThreadOffset);
            //whole_grid.sync();
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

