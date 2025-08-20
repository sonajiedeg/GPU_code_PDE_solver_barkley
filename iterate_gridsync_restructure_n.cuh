#include "cuda_runtime.h"

static __device__ __forceinline__ void no_flux_n_B(float* g_ub, int res, int i, int j, int ThreadOffset);
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_n_B(float* g_ub, float* r_un, int res, int i, int j, int* nOffset);

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace_n_B(float* r_u, float* r_v, float* r_un, float* r_vn)
{
    float u_th;
    for (int k = 0; k < nKernel; k++)
    {
        u_th = ( r_v[k] + b ) * ra;
        r_vn[k] = r_v[k] + dt * ( r_u[k] - r_v[k] );
        r_un[k] = r_u[k] + dt * repsilon * r_u[k] * ( 1 - r_u[k] ) * (r_u[k] - u_th);
    }
}

// deal with laplace
static __device__ __forceinline__ void laplace_n_B(float* g_ua, float* r_un, int res, int i, int j, int* n_offset)
{   
    for(int k = 0; k < nKernel; k++)
    {
        if (i < res - 1 && j * nKernel + k < res - 1 && i > 0 && j * nKernel + k  > 0)
        {
            float r_L = 0;

            float* g_ua_act = &g_ua[n_offset[k]]; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD     // INTO REG

            r_L = r_L-20 * g_ua_act[0];
            r_L = 4 * g_ua_act[+1] + r_L;
            r_L = 4 * g_ua_act[-1] + r_L;

            g_ua_act = g_ua_act - res; // Eltoljuk a pointert 1 sorral feljebb, 1 INT ADD         // INTO REG
            r_L = 4 * g_ua_act[0] + r_L;
            r_L = 1 * g_ua_act[+1] + r_L;
            r_L = 1 * g_ua_act[-1] + r_L;

            g_ua_act = g_ua_act + 2 * res; // Eltoljuk a pointert 2 sorral lejjebb, 1 INT FMA       // INTO REG
            r_L = 4 * g_ua_act[0] + r_L;
            r_L = 1 * g_ua_act[+1] + r_L;
            r_L = 1 * g_ua_act[-1] + r_L;

            r_un[k] = ccc * r_L + r_un[k];
        }
    }
}

__global__ void iterate_kernel_gsync_restruct_n_B(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    // global unique thread identifiers in 2D
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();

    if (i < res && j < jmax)            // INTO REG
    {
        // int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)       // INTO REG
        int nOffset[nKernel];
        for(int k = 0; k < nKernel; k++)
        {
            nOffset[k] = i + nKernel * j * res + k * res;
        }
        // ----- Variables -----
        float r_ua[nKernel];
        float r_va[nKernel];
        float r_ub[nKernel];
        float r_vb[nKernel];

        // ----- Load Data -----
        copy_g_to_r_n_B(g_ua, g_v, r_ua, r_va, nOffset);
        //whole_grid.sync();

        // ----- Iteration step 1 -----
        iterate_without_laplace_n_B(r_ua, r_va, r_ub, r_vb);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
            // ----- Laplace 1 -----
            laplace_n_B(g_ua, r_ub, res, i, j, nOffset);

            // ----- Iteration step 2 
            iterate_without_laplace_n_B(r_ub, r_vb, r_ua, r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_n_B(g_ub, r_ub, res, i, j, nOffset);
            whole_grid.sync();
            
            // ----- Laplace 2 -----
            laplace_n_B(g_ub, r_ua, res, i, j, nOffset);

            // ----- Iteration step 3 -----
            iterate_without_laplace_n_B(r_ua, r_va, r_ub, r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_n_B(g_ua, r_ua, res, i, j, nOffset);
            //whole_grid.sync();
        }
    }
}

// no flux boundary condition
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_n_B(float* g_ub, float* r_un, int res, int i, int j, int* nOffset)
{
    for(int k = 0; k < nKernel; k++)
    {
        // point id from j and k
        int jp = j * nKernel + k;
        // index of the point in global memory
        int nOffsetTmp = nOffset[k];
        // ----- Copy To GlobalMem -----
        if( i > 1 && jp > 1 && i < res - 1 && jp < res - 1 ) { g_ub[nOffsetTmp] = r_un[k]; }

        // ----- No Flux BC -----
        //s
        if( jp == 1 ) { g_ub[nOffsetTmp - res] = r_un[k]; }
        //w
        if( i == 1 ) { g_ub[nOffsetTmp - 1] = r_un[k]; }
        //e
        if( i == res - 2 ) { g_ub[nOffsetTmp + 1] = r_un[k]; }
        //n
        if( jp == res - 2 ) { g_ub[nOffsetTmp + res] = r_un[k]; }
        //sw
        if( jp == 1 && i == 1 ) { g_ub[nOffsetTmp - res - 1] = r_un[k]; }
        //se 
        if( jp == 1 && i == res - 2 ) { g_ub[nOffsetTmp - res + 1] = r_un[k]; }
        //nw
        if( jp == res - 2 && i == 1 ) { g_ub[nOffsetTmp + res - 1] = r_un[k]; }
        //ne
        if( jp == res - 2 && i == res - 2 ) { g_ub[nOffsetTmp + res + 1] = r_un[k]; }
    }
}

/*
// copy from registers to global memory
static __device__ __forceinline__ void copy_r_to_g_n_B(float* g_ub, float* g_v, float* r_u, float* r_v, int* n_offset, int n_kernel)
{
    for (int k = 0; k < n_kernel; i++)
    {
        g_ub[n_offset[k]] = r_u[k];
        g_v[n_offset[k]] = r_v[k];
    }
}
*/