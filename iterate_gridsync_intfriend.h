#include "cuda_runtime.h"

/*
__const__ float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
__const__ float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
__const__ float b = 0.05;
__const__ float dt = 0.0076893502846360206604003906250000000000;
__const__ float ccc = 0.0335952863097190856933593750000000000000;
*/

static __device__ __forceinline__ void no_flux_gi(float* g_ub, int res, int i, int j, int ThreadOffset, int threadoffset_p_one, int threadoffset_m_one, int threadoffset_p_res, int threadoffset_m_res, int res_m_one);
// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace_gi(const float r_ua, const float r_v, float* r_un, float* r_vn)
{
    float u_th;

    // ----- constants -----
    // Ezeket vagy a constant vagy a shared memóriába raknám.
    // Constant memóriába úgy is be tudod "rakni" ha beégeted a számot ahol használod.
    // Ha változtatni akarod, akkor lehet #define -t is használni (a global függvényen kívül természetesen).
    // DE NEM KRITIKUS
    /*
    const float epsilon = 0.02;	// Inkább: repsilon = 1.0/0.02
    const float a = 0.64;		// Inkább: ra = 1.0/0.64
    */


    // Az osztásokat kicserélnémreciprokkal való szorzásra! Lásd a kommenteket fent.
    // Az osztások nagyon lassúak, kb 50-70 szer lassabb mint egy szorzás vagy összeadás/kivonás
    u_th = (r_v + b) * ra; // ( r_v + b ) * ra // ez a sor összevonható lenne a +2. sorral a +2. sorban
    *r_vn = r_v + dt * (r_ua - r_v);
    *r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th); // ... dt * repsilon * r_ua ...

}

// deal with laplace
static __device__ __forceinline__ void laplace_gi(float* g_ua, float* r_un, int res, int i, int j, int ThreadOffset, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s)
{
    if (within_simspace_inner)       // INTO REG: within_simspace_inner
    {
        float r_L = 0;
        // Nem nagy a stencil, kilehet bontani teljesen:
        // float* g_ua_act = g_ua + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelõ elemére mutat, 1 INT ADD     // INTO REG: lap_p
        r_L += -20 * lap_p[0];
        r_L += 4 * lap_p[+1];
        r_L += 4 * lap_p[-1];

        // g_ua_act = g_ua_act - res; // Eltoljuk a pointert 1 sorral feljebb, 1 INT ADD         // INTO REG: lap_s
        r_L += 4 * lap_s[0];
        r_L += 1 * lap_s[+1];
        r_L += 1 * lap_s[-1];

        // g_ua_act = g_ua_act + 2 * res; // Eltoljuk a pointert 2 sorral lejjebb, 1 INT FMA       // INTO REG: lap_n
        r_L += 4 * lap_n[0];
        r_L += 1 * lap_n[+1];
        r_L += 1 * lap_n[-1];

        *r_un += ccc * r_L;
    }
}

__global__ void iterate_kernel_gi(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int within_simspace_inner = i < res - 1 && j < res - 1 && i > 0 && j > 0;

    // Nem nagy a stencil, kilehet bontani teljesen:
    

    //threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, res_m_one

    grid_group whole_grid = this_grid();

    if (i < res && j < res)            // INTO REG
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c mûveletet 1 órajel alatt megcsinálja)       // INTO REG, allready done

        float* lap_pa = g_ua + ThreadOffset; // INTO REG: lap_p
        float* lap_sa = lap_pa - res; // INTO REG: lap_s
        float* lap_na = lap_pa + res; // INTO REG: lap_n

        float* lap_pb = g_ub + ThreadOffset; // INTO REG: lap_p
        float* lap_sb = lap_pb - res; // INTO REG: lap_s
        float* lap_nb = lap_pb + res; // INTO REG: lap_n

        int threadoffset_p_one = ThreadOffset + 1;
        int threadoffset_m_one = ThreadOffset - 1;
        int threadoffset_p_res = ThreadOffset + res;
        int threadoffset_m_res = ThreadOffset - res;
        int res_m_one = res - 1;

        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_v, ThreadOffset);
        //whole_grid.sync();

#pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            // ----- Iteration step 1 -----
            //r_un = 0;
            iterate_without_laplace_gi(r_ua, r_v, &r_un, &r_vn);
            //whole_grid.sync();

            // ----- Laplace -----
            laplace_gi(g_ua, &r_un, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);
            //whole_grid.sync();

            // ----- "Unload" Data -----
            copy_r_to_g(g_ub, g_v, r_un, r_vn, ThreadOffset);
            __syncthreads();

            // ----- No Flux Bc -----
            no_flux_gi(g_ub, res, i, j, ThreadOffset, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, res_m_one);
            whole_grid.sync();

            // ----- Iteration step 2 -----
            //r_ua = 0;
            iterate_without_laplace_gi(r_un, r_vn, &r_ua, &r_v);
            //whole_grid.sync();

            // ----- Laplace -----
            laplace_gi(g_ub, &r_ua, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);

            // ----- "Unload" Data -----
            copy_r_to_g(g_ua, g_v, r_ua, r_v, ThreadOffset);
            __syncthreads();

            // ----- No Flux Bc -----
            no_flux_gi(g_ua, res, i, j, ThreadOffset, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, res_m_one);
            whole_grid.sync();
        }
    }
}

// no flux boundary condition
static __device__ __forceinline__ void no_flux_gi(float* g_ub, int res, int i, int j, int ThreadOffset, int threadoffset_p_one, int threadoffset_m_one, int threadoffset_p_res, int threadoffset_m_res, int res_m_one)
{
    // ----- No Flux BC -----
    if (i == 0) { g_ub[ThreadOffset] = g_ub[threadoffset_p_one]; }                // INTO REG: threadoffset_p_one

    if (i == res_m_one) { g_ub[ThreadOffset] = g_ub[threadoffset_m_one]; }       // INTO REG: res_m_one, threadoffset_m_one
    __syncthreads();
    if (j == 0) { g_ub[ThreadOffset] = g_ub[threadoffset_p_res]; }              // INTO REG: threadoffset_p_res

    if (j == res_m_one) { g_ub[ThreadOffset] = g_ub[threadoffset_m_res]; }       // INTO REG: res_m_one, threadoffset_m_res
}

