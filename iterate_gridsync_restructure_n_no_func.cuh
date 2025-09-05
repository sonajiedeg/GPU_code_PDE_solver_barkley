#include "cuda_runtime.h"

__global__ void iterate_kernel_gsync_restruct_n_B_no_func(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    // global unique thread identifiers in 2D
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float u_th;
    float r_L = 0;

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
        // copy from global memory to the registers
        for (int k = 0; k < nKernel; k++)
        {
            r_ua[k] = g_ua[nOffset[k]];
            r_va[k] = g_v[nOffset[k]];
        }

        // ----- Iteration step 1 -----
        // iterate one timestep without laplace
        for (int k = 0; k < nKernel; k++)
        {
            u_th = ( r_va[k] + b ) * ra;
            r_vb[k] = r_va[k] + dt * ( r_va[k] - r_va[k] );
            r_ub[k] = r_ua[k] + dt * repsilon * r_ua[k] * ( 1 - r_ua[k] ) * (r_ua[k] - u_th);
        }

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            this_grid().sync();
            // ----- Laplace 1 -----
            // deal with laplace   
            for(int k = 0; k < nKernel; k++)
            {
                if (i < res - 1 && j * nKernel + k < res - 1 && i > 0 && j * nKernel + k  > 0)
                {
                    float r_L = 0;

                    float* g_ua_act = &g_ua[nOffset[k]]; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD     // INTO REG

                    r_L = r_L - 20 * g_ua_act[0];
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

                    r_ub[k] = ccc * r_L + r_ub[k];
                }
            }

            // ----- Iteration step 2 
            // iterate one timestep without laplace
            for (int k = 0; k < nKernel; k++)
            {
                u_th = ( r_vb[k] + b ) * ra;
                r_va[k] = r_vb[k] + dt * ( r_ub[k] - r_vb[k] );
                r_ua[k] = r_ub[k] + dt * repsilon * r_ub[k] * ( 1 - r_ub[k] ) * (r_ub[k] - u_th);
            }

            // ----- "Unload" Data And No Flux Bc 1 -----
            // no flux boundary condition
            for(int k = 0; k < nKernel; k++)
            {
                // point id from j and k
                int jp = j * nKernel + k;
                // index of the point in global memory
                int nOffsetTmp = nOffset[k];
                // ----- Copy To GlobalMem -----
                if( i > 1 && jp > 1 && i < res - 1 && jp < res - 1 ) { g_ub[nOffsetTmp] = r_ub[k]; }

                // ----- No Flux BC -----
                //s
                if( jp == 1 ) { g_ub[nOffsetTmp - res] = r_ub[k]; }
                //w
                if( i == 1 ) { g_ub[nOffsetTmp - 1] = r_ub[k]; }
                //e
                if( i == res - 2 ) { g_ub[nOffsetTmp + 1] = r_ub[k]; }
                //n
                if( jp == res - 2 ) { g_ub[nOffsetTmp + res] = r_ub[k]; }
                //sw
                if( jp == 1 && i == 1 ) { g_ub[nOffsetTmp - res - 1] = r_ub[k]; }
                //se 
                if( jp == 1 && i == res - 2 ) { g_ub[nOffsetTmp - res + 1] = r_ub[k]; }
                //nw
                if( jp == res - 2 && i == 1 ) { g_ub[nOffsetTmp + res - 1] = r_ub[k]; }
                //ne
                if( jp == res - 2 && i == res - 2 ) { g_ub[nOffsetTmp + res + 1] = r_ub[k]; }
            }

            this_grid().sync();
            
            // ----- Laplace 2 -----
            // deal with laplace
            for(int k = 0; k < nKernel; k++)
            {
                if (i < res - 1 && j * nKernel + k < res - 1 && i > 0 && j * nKernel + k  > 0)
                {
                    float r_L = 0;

                    float* g_ua_act = &g_ub[nOffset[k]]; // Létrehozunk egy új pointert ami az aktuális thread megfelelő elemére mutat, 1 INT ADD     // INTO REG

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

                    r_ua[k] = ccc * r_L + r_ua[k];
                }
            }

            // ----- Iteration step 3 -----
            // iterate one timestep without laplace
            for (int k = 0; k < nKernel; k++)
            {
                u_th = ( r_va[k] + b ) * ra;
                r_vb[k] = r_va[k] + dt * ( r_ua[k] - r_va[k] );
                r_ub[k] = r_ua[k] + dt * repsilon * r_ua[k] * ( 1 - r_ua[k] ) * (r_ua[k] - u_th);
            }

            // ----- "Unload" Data And No Flux Bc 2 -----
            // no flux boundary condition
            for(int k = 0; k < nKernel; k++)
            {
                // point id from j and k
                int jp = j * nKernel + k;
                // index of the point in global memory
                int nOffsetTmp = nOffset[k];
                // ----- Copy To GlobalMem -----
                if( i > 1 && jp > 1 && i < res - 1 && jp < res - 1 ) { g_ua[nOffsetTmp] = r_ua[k]; }

                // ----- No Flux BC -----
                //s
                if( jp == 1 ) { g_ua[nOffsetTmp - res] = r_ua[k]; }
                //w
                if( i == 1 ) { g_ua[nOffsetTmp - 1] = r_ua[k]; }
                //e
                if( i == res - 2 ) { g_ua[nOffsetTmp + 1] = r_ua[k]; }
                //n
                if( jp == res - 2 ) { g_ua[nOffsetTmp + res] = r_ua[k]; }
                //sw
                if( jp == 1 && i == 1 ) { g_ua[nOffsetTmp - res - 1] = r_ua[k]; }
                //se 
                if( jp == 1 && i == res - 2 ) { g_ua[nOffsetTmp - res + 1] = r_ua[k]; }
                //nw
                if( jp == res - 2 && i == 1 ) { g_ua[nOffsetTmp + res - 1] = r_ua[k]; }
                //ne
                if( jp == res - 2 && i == res - 2 ) { g_ua[nOffsetTmp + res + 1] = r_ua[k]; }
            }
        }
    }
}