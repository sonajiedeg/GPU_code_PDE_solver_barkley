#include "cuda_runtime.h"



// static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset);
static __device__ __forceinline__ void laplace_gi(float* g_ua, float* r_un, int res, int i, int j, int ThreadOffset, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s);
// static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend_v2(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one, int threadoffset_p_res, int threadoffset_m_res, int within_simspace_inner, bool i_eq_1_and_j_eq_1, bool i_eq_res_m_2_and_j_eq_1, bool i_eq_1_and_j_eq_res_m_2, bool i_eq_res_m_2_and_j_eq_res_m_2, int threadoffset_m_res_m_1, int threadoffset_m_res_p_1, int threadoffset_p_res_m_1, int threadoffset_p_resp_1);


__global__ void iterate_kernel_gsync_restruct_int_friend_v2_threadfence_for(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, int* ivar)
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

        int threadoffset_m_res_m_1 = threadoffset_m_res - 1;
        int threadoffset_m_res_p_1 = threadoffset_m_res + 1;
        int threadoffset_p_res_m_1 = threadoffset_p_res - 1;
        int threadoffset_p_resp_1 = threadoffset_p_res + 1;

        bool j_eq_1 = j == 1;
        bool j_eq_res_m_2 = j == res - 2;
        bool i_eq_1 = i == 1;
        bool i_eq_res_m_2 = i == res - 2;

        bool i_eq_1_and_j_eq_1 = i_eq_1 && j_eq_1;
        bool i_eq_res_m_2_and_j_eq_1 = i_eq_res_m_2 && j_eq_1;
        bool i_eq_1_and_j_eq_res_m_2 = i_eq_1 && j_eq_res_m_2;
        bool i_eq_res_m_2_and_j_eq_res_m_2 = i_eq_res_m_2 && j_eq_res_m_2;

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
            switch ( (*ivar) % 3 )
            {
            case 0:
                whole_grid.sync();
            case 1:
                __threadfence();
            case 2:
                __syncthreads();
            default:
                break;
            }
            // whole_grid.sync();
            // __threadfence();
            // __syncthreads();
            // 
            // ----- Laplace 1 -----
            laplace_gi(g_ua, &r_ub, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, r_vb, &r_ua, &r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend_v2(g_ub, r_ub, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, i_eq_1_and_j_eq_1, i_eq_res_m_2_and_j_eq_1, i_eq_1_and_j_eq_res_m_2, i_eq_res_m_2_and_j_eq_res_m_2, threadoffset_m_res_m_1, threadoffset_m_res_p_1, threadoffset_p_res_m_1, threadoffset_p_resp_1);
            // copy_r_to_g_and_no_flux_int_friend_v2(g_ub, r_ub, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner);
            // copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, res, i, j, ThreadOffset);
            // whole_grid.sync();
            // __threadfence();
            // __syncthreads();
            switch ((*ivar) / 3)
            {
            case 0:
                whole_grid.sync();
            case 1:
                __threadfence();
            case 2:
                __syncthreads();
            default:
                break;
            }

            // ----- Laplace 2 -----
            laplace_gi(g_ub, &r_ua, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend_v2(g_ua, r_ua, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, i_eq_1_and_j_eq_1, i_eq_res_m_2_and_j_eq_1, i_eq_1_and_j_eq_res_m_2, i_eq_res_m_2_and_j_eq_res_m_2, threadoffset_m_res_m_1, threadoffset_m_res_p_1, threadoffset_p_res_m_1, threadoffset_p_resp_1);
            // copy_r_to_g_and_no_flux_int_friend_v2(g_ua, r_ua, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner);
            // copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, res, i, j, ThreadOffset);
            //whole_grid.sync();
        }
    }
}