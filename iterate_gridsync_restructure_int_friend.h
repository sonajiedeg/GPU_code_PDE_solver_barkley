#include "cuda_runtime.h"

__const__ float repsilon = 1 / 0.02;	// Inkább: repsilon = 1.0/0.02
__const__ float ra = 1 / 0.64;		// Inkább: ra = 1.0/0.64
__const__ float b = 0.05;
__const__ float dt = 0.0076893502846360206604003906250000000000;
__const__ float ccc = 0.0335952863097190856933593750000000000000;

static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset);
static __device__ __forceinline__ void laplace_gi(float* g_ua, float* r_un, int res, int i, int j, int ThreadOffset, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s);
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one, int threadoffset_p_res, int threadoffset_m_res, int within_simspace_inner);

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

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(const float r_ua, const float r_v, float* r_un, float* r_vn)
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


__global__ void iterate_kernel_gsync_restruct_int_friend(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int within_simspace_inner = i < res - 1 && j < res - 1 && i > 0 && j > 0;


    if (i < res && j < res)            // INTO REG
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)       // INTO REG, allready done

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

        bool j_eq_1 = j == 1;
        bool j_eq_res_m_2 = j == res - 2;
        bool i_eq_1 = i == 1;
        bool i_eq_res_m_2 = i == res - 2;

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
            laplace_gi(g_ua, &r_ub, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, r_vb, &r_ua, &r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner);
            // copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, res, i, j, ThreadOffset);
            whole_grid.sync();
            
            // ----- Laplace 2 -----
            laplace_gi(g_ub, &r_ua, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner);
            // copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, res, i, j, ThreadOffset);
            //whole_grid.sync();
        }
    }
}

// no flux boundary condition INT FRIEND V2, testing register usage
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner)
{
    // ----- Copy To GlobalMem -----
    if (within_simspace_inner)
    {
        g_ub[ThreadOffset] = r_un;
    }

    // ----- No Flux BC -----
    //s
    if (j_eq_1) { g_ub[threadoffset_m_res] = r_un; }
    //w
    if (i_eq_1) { g_ub[threadoffset_m_one] = r_un; }
    //e
    if (i_eq_res_m_2) { g_ub[threadoffset_p_one] = r_un; }
    //n
    if (j_eq_res_m_2) { g_ub[threadoffset_p_res] = r_un; }
    //sw
    if (i_eq_1 && j_eq_1) { g_ub[threadoffset_m_res - 1] = r_un; }
    //se
    if (i_eq_res_m_2 && j_eq_1) { g_ub[threadoffset_m_res + 1] = r_un; }
    //nw
    if (i_eq_1 && j_eq_res_m_2) { g_ub[threadoffset_p_res - 1] = r_un; }
    //ne
    if (i_eq_res_m_2 && j_eq_res_m_2) { g_ub[threadoffset_p_res + 1] = r_un; }
}
