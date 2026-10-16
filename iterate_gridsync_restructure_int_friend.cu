#include "cuda_runtime.h"
#include "param.h"
#include "bc.h"


static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_ua, float r_ub, float r_uc, int res, int i, int j, int ThreadOffset);

static __device__ __forceinline__ void laplace_gi(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params);

static __device__ __forceinline__ void laplace_grad(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady, int snapCount, int ThreadOffset);

static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_u2, float r_u2, float r_u1, float r_u0, float* r_u1_bc, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k, SimParamStruct params);


__global__ void iterate_snap_segment(float* g_ua, float* g_ub, int iteration_limit_per_2, float* g_snap, SimParamStruct params, int k0);

__global__ void iterate_snap_grad(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady);

__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params);

__global__ void iterate(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, SimParamStruct params);


static __device__ __forceinline__ void laplace_grad(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady, int snapCount, int ThreadOffset)
{
    if (within_simspace_inner)       // INTO REG: within_simspace_inner
    {
        float r_L = 0;
		// Patra-Karttunen (mabey try Oono-Puri)
        r_L += -20 * lap_p[0];

		float n = lap_n[0];
		float s = lap_s[0];
		float e = lap_p[+1];
		float w = lap_p[-1];

		r_L += 4 * e;
        r_L += 4 * w;
        r_L += 4 * s;
        r_L += 4 * n;

		/*
        r_L += 1 * lap_p[+1];
        r_L += 1 * lap_p[-1];
        r_L += 1 * lap_s[0];
        r_L += 1 * lap_n[0];
		*/
		
        r_L += 1 * lap_s[+1];
        r_L += 1 * lap_s[-1];
        r_L += 1 * lap_n[+1];
        r_L += 1 * lap_n[-1];
		
		// save gradients
 		// int size = 256 * 256; // for REMOTE
 		int size = 256 * 256; // for LOCAL
		g_snap_gradx[ ThreadOffset + snapCount * size ]	= ( e - w ) * 0.5;
		g_snap_grady[ ThreadOffset + snapCount * size ]	= ( n - s ) * 0.5;
		// g_snap_gradx[ ThreadOffset + snapCount * size ]	= ( lap_p[+1] - lap_p[-1] ) / 2;
	    // g_snap_grady[ ThreadOffset + snapCount * size ]	= ( lap_n[0] - lap_s[0] ) / 2;

        *r_un += params.K * r_L;
    }
}

// deal with laplace
static __device__ __forceinline__ void laplace_gi(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params)
{
    if (within_simspace_inner)       // INTO REG: within_simspace_inner
    {
        float r_L = 0;
		// Patra-Karttunen (mabey try Oono-Puri)
        r_L += -20 * lap_p[0];

        r_L += 4 * lap_p[+1];
        r_L += 4 * lap_p[-1];
        r_L += 4 * lap_s[0];
        r_L += 4 * lap_n[0];

        r_L += 1 * lap_s[+1];
        r_L += 1 * lap_s[-1];
        r_L += 1 * lap_n[+1];
        r_L += 1 * lap_n[-1];

        *r_un += params.K * r_L;
    }
}

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(float r_ua, float* r_ub, float r_uc)
{
	*r_ub = 2 * r_ua - r_uc; // TODO: Check if it is correct, but try first...
}

__global__ void iterate_snap_segment(float* g_ua, float* g_ub, int iteration_limit_per_2, float* g_snap, SimParamStruct params, int k0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int within_simspace_inner = i < params.res - 1 && j < params.res - 1 && i > 0 && j > 0;

    if (i < params.res && j < params.res)            // INTO REG
    {
        int ThreadOffset = i + j * params.res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)       // INTO REG, allready done

        float* lap_pa = g_ua + ThreadOffset; // INTO REG: lap_p
        float* lap_sa = lap_pa - params.res; // INTO REG: lap_s
        float* lap_na = lap_pa + params.res; // INTO REG: lap_n

        float* lap_pb = g_ub + ThreadOffset; // INTO REG: lap_p
        float* lap_sb = lap_pb - params.res; // INTO REG: lap_s
        float* lap_nb = lap_pb + params.res; // INTO REG: lap_n

        int threadoffset_p_one = ThreadOffset + 1;
        int threadoffset_m_one = ThreadOffset - 1;
        int threadoffset_p_res = ThreadOffset + params.res;
        int threadoffset_m_res = ThreadOffset - params.res;

        bool j_eq_1 = j == 1;
        bool j_eq_res_m_2 = j == params.res - 2;
        bool i_eq_1 = i == 1;
        bool i_eq_res_m_2 = i == params.res - 2;

        // ----- Variables -----
        float r_ua;
        float r_ub = 0;
		float r_u_bc = 0;
		if(i==1){r_u_bc = g_ua[ThreadOffset-1];}
		if(j==1){r_u_bc = g_ua[ThreadOffset-params.res];}
		if(i==params.res-2){r_u_bc = g_ua[ThreadOffset+1];}
		if(j==params.res-2){r_u_bc = g_ua[ThreadOffset+params.res];}
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = params.res * params.res;
		// if(i==0&&j==0){ PASS }
        // ----- Load Data -----
        copy_g_to_r(g_ua, g_ub, &r_ua, &r_ub, ThreadOffset);
        //whole_grid.sync();

		// ----- save u_n{n-1} for BC at u_{n+1} 1-----
		float r_uc1 = 0;
		float r_uc2 = 0;
		r_uc1 = r_ub;
		r_uc2 = r_ub;

        // ----- Iteration step 1 -----
        iterate_without_laplace(r_ua, &r_ub, r_uc2);

// #pragma unroll

		// if(i==1&&j==1){printf("debug gpu: iteration_limit_per_2: %d",iteration_limit_per_2);}
        for (int k = k0; k < iteration_limit_per_2; k++)
        {
			// if(i==1&&j==1&&k%10==0){printf("debug gpu: r_ub: %f",r_ub);}
            whole_grid.sync();
			// ----- Laplace 1 -----

            laplace_gi(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params);
            // laplace_grad(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params, g_snap_gradx, g_snap_grady, countSnap, ThreadOffset);

			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc2 = r_ua;

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, &r_ua, r_uc2);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, r_ua, r_uc1, &r_u_bc, params.res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k*2, params);
			// ----- save snapshot to global memory -----
            whole_grid.sync();
			if ( (k*2) % params.SNAP_MODULUS == 0 ){
				if(within_simspace_inner){
					g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ub;
				}
				else{ g_snap[ThreadOffset + ( countSnap++ ) * size] = g_ua[ThreadOffset]; }
			}

            whole_grid.sync();
            
            // ----- Laplace 2 -----

            laplace_gi(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);
            // laplace_grad(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params, g_snap_gradx, g_snap_grady, countSnap, ThreadOffset);
            // laplace_gi(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);


			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc1 = r_ub;

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, &r_ub, r_uc1);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, r_ub, r_uc2, &r_u_bc, params.res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k*2+1, params);
			// ----- save snapshot to global memory -----
            whole_grid.sync();
			if ( (k*2+1) % params.SNAP_MODULUS == 0 ){
				if(within_simspace_inner){
					g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ua;
				}
				else{ g_snap[ThreadOffset + ( countSnap++ ) * size] = g_ua[ThreadOffset]; }
			}
        }
    }
}

__global__ void iterate_snap_grad(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady)
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
        float r_ub = 0;
		float r_u_bc = 0;
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;
		// if(i==0&&j==0){ PASS }
        // ----- Load Data -----
        copy_g_to_r(g_ua, g_ub, &r_ua, &r_ub, ThreadOffset);
        //whole_grid.sync();

		// ----- save u_n{n-1} for BC at u_{n+1} 1-----
		float r_uc1 = 0;
		float r_uc2 = 0;
		r_uc1 = r_ub;
		r_uc2 = r_ub;

        // ----- Iteration step 1 -----
        iterate_without_laplace(r_ua, &r_ub, r_uc2);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
			// ----- Laplace 1 -----
            // laplace_gi(&r_ub, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);
            laplace_grad(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params, g_snap_gradx, g_snap_grady, countSnap, ThreadOffset);

			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc2 = r_ua;

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, &r_ua, r_uc2);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, r_ua, r_uc1, &r_u_bc, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k*2, params);
			// ----- save snapshot to global memory -----
            whole_grid.sync();
			if ( (k*2) % params.SNAP_MODULUS == 0 ){
				if(within_simspace_inner){
					g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ub;
				}
				else{ g_snap[ThreadOffset + ( countSnap++ ) * size] = g_ua[ThreadOffset]; }
			}

            whole_grid.sync();
            
            // ----- Laplace 2 -----
            //laplace_gi(&r_ua, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);
            laplace_grad(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params, g_snap_gradx, g_snap_grady, countSnap, ThreadOffset);
            // laplace_gi(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);


			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc1 = r_ub;

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, &r_ub, r_uc1);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, r_ub, r_uc2, &r_u_bc, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k*2+1, params);
			// ----- save snapshot to global memory -----
            whole_grid.sync();
			if ( (k*2+1) % params.SNAP_MODULUS == 0 ){
				if(within_simspace_inner){
					g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ua;
				}
				else{ g_snap[ThreadOffset + ( countSnap++ ) * size] = g_ua[ThreadOffset]; }
			}
        }
    }
}

__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params)
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
        float r_ub = 0;
		float r_u_bc = 0;
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;

        // ----- Load Data -----
		// copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
        copy_g_to_r(g_ua, g_ub, &r_ua, &r_ub, ThreadOffset);
        //whole_grid.sync();

		// ----- save u_n{n-1} for BC at u_{n+1} 1-----
		float r_uc1 = 0;
		float r_uc2 = 0;
		r_uc1 = r_ub;

        // ----- Iteration step 1 -----
        iterate_without_laplace(r_ua, &r_ub, r_uc2);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
			// ----- Laplace 1 -----
            // laplace_gi(&r_ub, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);
            laplace_gi(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params);

			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc2 = r_ua;

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, &r_ua, r_uc2);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, r_ua, r_uc1, &r_u_bc, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k, params);
			// ----- save snapshot to global memory -----
			if ( (k*2) % params.SNAP_MODULUS == 0 ){ g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ua; }

            whole_grid.sync();
            
            // ----- Laplace 2 -----
            //laplace_gi(&r_ua, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);
            laplace_gi(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);


			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc1 = r_ub;

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, &r_ub, r_uc1);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, r_ub, r_uc2, &r_u_bc, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k+1, params);
			// ----- save snapshot to global memory -----
			if ( (k*2+1) % params.SNAP_MODULUS == 0 ){ g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ua; }
        }
    }
}

__global__ void iterate(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, SimParamStruct params)
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
        float r_ub = 0;

		// ----- init previous boundary data:
		float r_u_bc = 0;
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

        // ----- Load Data -----
        // copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
        copy_g_to_r(g_ua, g_ub, &r_ua, &r_ub, ThreadOffset);
        //whole_grid.sync();

		// ----- save u_n{n-1} for BC at u_{n+1} 1-----
		float r_uc1 = 0;
		float r_uc2 = 0;
		r_uc1 = r_ub;


        // ----- Iteration step 1 -----
        iterate_without_laplace(r_ua, &r_ub, r_uc1);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
            // ----- Laplace 1 -----
            laplace_gi(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params);

			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc2 = r_ua;

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, &r_ua, r_uc2);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, r_uc2, r_uc1, &r_u_bc, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k, params);
            whole_grid.sync();
            
            // ----- Laplace 2 -----

            laplace_gi(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);

			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc1 = r_ub;

            // ----- Iteration step 3 -----
			iterate_without_laplace(r_ua, &r_ub, r_uc1);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, r_uc1, r_uc2, &r_u_bc, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k, params);
        }
    }
}

// no flux boundary condition INT FRIEND V2, testing register usage
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_u2, float r_u2, float r_u1, float r_u0, float* r_u1_bc, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k, SimParamStruct params)
{
    // ----- Copy To GlobalMem -----
    if (within_simspace_inner)
    {
        g_u2[ThreadOffset] = r_u2;
    }

    // ----- No Flux BC -----
    // ============================== S
	// neumann
	// if (j_eq_1) { g_u2[threadoffset_m_res] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// if (j_eq_1) { g_u2[threadoffset_m_res] = r_u2; }
	// sommerfeld/impedance
	// if (j_eq_1) { float r_u_tmp = *r_u1_bc + dt*Z*c/dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_m_res] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order
	if (j_eq_1) {
		// float r_u_tmp = *r_u1_bc + params.dt * params.c / params.dx * ( r_u1 - *r_u1_bc ); // somewhat worked
		float r_u_tmp = *r_u1_bc * params.som1 + params.som2 * ( r_u2 );
		g_u2[threadoffset_m_res] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
		//sw
		if (i_eq_1 && j_eq_1) { g_u2[threadoffset_m_res - 1] = r_u_tmp; }
		//se
		if (i_eq_res_m_2 && j_eq_1) { g_u2[threadoffset_m_res + 1] = r_u_tmp; }
	}
    // ============================== W
    // impedance/sommerfeld
	// if (i_eq_1) { float r_u_tmp = *r_u1_bc + params.dt*params.Z*params.c/params.dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_m_one] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order
	// u[0, 1:ny-1, 0] = (u[1, 1:ny-1, 1] - kappa * ( u[0, 1:ny-1, 1] - u[1, 1:ny-1, 0]) )
		
    // u2[0] = u1[0] + dt * Z * c / dx * ( u1[1] - u1[0] )
	/*
	if (i_eq_1) {
		// float r_u_tmp = r_u1 - params.kappa * ( r_u2 - *r_u1_bc );
		// float r_u_tmp = *r_u1_bc + params.dt * params.c / params.dx * ( r_u1 - *r_u1_bc ); // somewhat worked
		float r_u_tmp = *r_u1_bc / ( 1 + params.dt * params.c / params.dx ) + params.dt * params.c / params.dx * ( r_u2 ) / ( 1 + params.dt * params.c / params.dx );
		g_u2[threadoffset_m_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	} // temp 1st order impl. TODO: 2nd order
	*/
	/*
	if (i_eq_1) {
		// if ( 64 <= j && j <= 192 ) { g_u2[threadoffset_p_one] = r_u2 + params.sin_coeff * __sinf( params.omega_dt * k ) - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
		if ( 32 <= j && j <= 96 ) { g_u2[threadoffset_m_one] = r_u2 + (1 - 1/32*abs(j-64)) * params.dx_rho_d_omega2 * __sinf( params.omega_dt * k + 1.5707963267948966192313216916398); }
		// else { g_u2[threadoffset_p_one] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); } // idk, mabey 2nd order neumann
		else {
			// float r_u_tmp = *r_u1_bc + params.dt * params.c / params.dx * ( r_u1 - *r_u1_bc ); // somewhat worked
			float r_u_tmp = *r_u1_bc * params.som1 + params.som2 * ( r_u2 );
			g_u2[threadoffset_m_one] = r_u_tmp;
			*r_u1_bc = r_u_tmp;
		} // sommerfeld
	}
	*/
	// mur's first order 

	/*
    if (i_eq_1) {
		float r_u_tmp = r_u1 + ( params.c * params.dt - params.dx ) / ( params.c * params.dt + params.dx ) * ( r_u2 - *r_u1_bc );
		g_u2[threadoffset_m_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	} // temp 1st order impl. TODO: 2nd order
	*/
	// Kalmi's transducer
    if (i_eq_1) {
		if ( 64 <= j && j <= 191 ) {
			// float phase_offset = 3.14159265359 * 1 * (k / 10236.0 );
			float phase_offset = 0.0;
			float r_u_tmp = params.bcC1 * (*r_u1_bc) + params.bcC3 * r_u2 - params.bcC2 * __sinf( params.omega_dt * k - phase_offset );
			g_u2[threadoffset_m_one] = r_u_tmp;
			(*r_u1_bc) = r_u_tmp;
		}
		// else { g_u2[threadoffset_m_one] = r_u2; }
		else{
			float r_u_tmp = *r_u1_bc * params.som1 + params.som2 * ( r_u2 );
			g_u2[threadoffset_m_one] = r_u_tmp;
			//sw
			if (i_eq_1 && j_eq_1) { g_u2[threadoffset_m_res - 1] = r_u_tmp; }
			//nw
			if (i_eq_1 && j_eq_res_m_2) { g_u2[threadoffset_p_res - 1] = r_u_tmp; }

			*r_u1_bc = r_u_tmp;

		}
	}
	// dirichlet
	// if (i_eq_1) { g_u2[threadoffset_m_one] = 0; }
	// neumann
    // if (i_eq_1) { g_u2[threadoffset_m_one] = r_u2 + 1/2/K * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// if (i_eq_1) { g_u2[threadoffset_m_one] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// if (i_eq_1) { g_u2[threadoffset_m_one] = r_u2; }
	
    // ============================== E
    //e transducer
#if eBC == 0
	// dirichlet
	if (i_eq_res_m_2) { g_u2[threadoffset_p_one] = 0; }
#elif eBC == 1
	// neumann
	if (i_eq_res_m_2) { g_u2[threadoffset_p_one] = r_u2; }
#elif eBC == 2
	// sommerfeld
	if (i_eq_res_m_2)
	{
		float r_u_tmp = *r_u1_bc * params.som1 + params.som2 * ( r_u2 );
		g_u2[threadoffset_p_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	}
#elif eBC == 3
	// transducer and sommerfeld
    if (i_eq_res_m_2) {
		// if ( 64 <= j && j <= 192 ) { g_u2[threadoffset_p_one] = r_u2 + params.sin_coeff * __sinf( params.omega_dt * k ) - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
		if ( 60 <= j && j <= 68 ) { g_u2[threadoffset_p_one] = r_u2 + (1 - 1/32*abs(j-64)) * params.dx_rho_d_omega2 * __sinf( params.omega_dt * k ); }
		// else { g_u2[threadoffset_p_one] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); } // idk, mabey 2nd order neumann
		else {
			// float r_u_tmp = *r_u1_bc + params.dt * params.c / params.dx * ( r_u1 - *r_u1_bc ); // somewhat worked
			//float r_u_tmp = *r_u1_bc / ( 1 + params.dt * params.c / params.dx ) + params.dt * params.c / params.dx * ( r_u2 ) / ( 1 + params.dt * params.c / params.dx );
			float r_u_tmp = *r_u1_bc * params.som1 + params.som2 * ( r_u2 );
			g_u2[threadoffset_p_one] = r_u_tmp;
			*r_u1_bc = r_u_tmp;
		} // sommerfeld
	}
#elif eBC == 4
// Kalmi's transducer
    if (i_eq_res_m_2) {
		if ( 64 <= j && j <= 191 ) {
			// float phase_offset = 3.14159265359 * 1 * (k / 10236.0 );
			float phase_offset = params.PhaseOffsetAmplitude*__sinf(params.PhaseOffsetAngularVelocity*params.dt*k);
			// if(j==32&&k%200==0){printf("phase offset amp: %f\n",params.PhaseOffsetAmplitude);};
			// if(j==32&&k%200==0){printf("phase offset in sin: %f\n",params.PhaseOffsetAngularVelocity*params.dt*k);};
			// if(j==32&&k%200==0){printf("phase offset sin: %f\n",__sinf(params.PhaseOffsetAngularVelocity*params.dt*k));};
			// if(j==32&&k%200==0){printf("phase offset: %f\n",phase_offset);};
			// if (k<=5120) { phase_offset = 3.14159265359 * params.C_amp * ( 3.14159265359*0.5 + atanf( params.s * (k - params.C_off ) ) ); }
			// if (k<=10240) { phase_offset = 4*3.14159265359*sinf( k / 3.14159265359 / 2560 ); }
			// if (k<=10240) { phase_offset = 4*3.14159265359*k/10240; }
			// else { phase_offset = params.C_amp*( 3.14159265359*0.5 - atanf( params.s * (k - 3.2 * params.C_off ) ) ); }
			// if(j == 64 && k%10==0){printf("offset = %f\n",phase_offset);}
			// float r_u_tmp = params.bcC1 * (*r_u1_bc) + params.bcC3 * r_u2 + params.bcC2 * __cosf( params.omega_dt * k + phase_offset + 3.14159265359*0.5);
			// phase_offset = 0.0;
			float r_u_tmp = params.bcC1 * (*r_u1_bc) + params.bcC3 * r_u2 - params.bcC2 * __sinf( params.omega_dt * k + phase_offset );
			g_u2[threadoffset_p_one] = r_u_tmp;
			(*r_u1_bc) = r_u_tmp;
		}
		// else { g_u2[threadoffset_p_one] = r_u2; }
		else{
			//float r_u_tmp = *r_u1_bc / ( 1 + params.dt * params.c / params.dx ) + params.dt * params.c / params.dx * ( r_u2 ) / ( 1 + params.dt * params.c / params.dx );
			float r_u_tmp = params.som1 * (*r_u1_bc) + params.som2 * ( r_u2 );
			g_u2[threadoffset_p_one] = r_u_tmp;
			//se
			if (i_eq_res_m_2 && j_eq_1) { g_u2[threadoffset_m_res + 1] = r_u_tmp; }
			//ne
			if (i_eq_res_m_2 && j_eq_res_m_2) { g_u2[threadoffset_p_res + 1] = r_u_tmp; }

			*r_u1_bc = r_u_tmp;

		}
	}
#endif
	// Kalmi's transducer

	/*	
    if (i_eq_res_m_2) {
		if ( 32 <= j && j <= 96 ) {
			float r_u_tmp = params.bcC1 * (*r_u1_bc) + params.bcC3 * r_u2 + params.bcC2 * __cosf( params.omega_dt * k );
			g_u2[threadoffset_p_one] = r_u_tmp;
			*r_u1_bc = r_u_tmp;
		}
		else { g_u2[threadoffset_p_one] = r_u2; }
	}
	*/
	
	
	    // ============================== N
	// neumann
    // if (j_eq_res_m_2) { g_u2[threadoffset_p_res] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    // if (j_eq_res_m_2) { g_u2[threadoffset_p_res] = r_u2; }
	// dirichlet
    // if (j_eq_res_m_2) { g_u2[threadoffset_p_res] = 0; }
	// if (j_eq_res_m_2) { float r_u_tmp = *r_u1_bc - dt*Z*c/dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_p_res] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order
	// sommerfeld
	if (j_eq_res_m_2) {
		// float r_u_tmp = *r_u1_bc + params.dt * params.c / params.dx * ( r_u1 - *r_u1_bc ); // somewhat worked
		float r_u_tmp = *r_u1_bc * params.som1 + params.som2 * ( r_u2 );
		g_u2[threadoffset_p_res] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
		//nw
		if (i_eq_1 && j_eq_res_m_2) { g_u2[threadoffset_p_res - 1] = r_u_tmp; }
		//ne
		if (i_eq_res_m_2 && j_eq_res_m_2) { g_u2[threadoffset_p_res + 1] = r_u_tmp; }

	}

	/*
    //sw
	// if (i_eq_1 && j_eq_1) { g_u2[threadoffset_m_res - 1] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    if (i_eq_1 && j_eq_1) { g_u2[threadoffset_m_res - 1] = r_u2; }
    //se
	// if (i_eq_res_m_2 && j_eq_1) { g_u2[threadoffset_m_res + 1] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    if (i_eq_res_m_2 && j_eq_1) { g_u2[threadoffset_m_res + 1] = r_u2; }
    //nw
    // if (i_eq_1 && j_eq_res_m_2) { g_u2[threadoffset_p_res - 1] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    if (i_eq_1 && j_eq_res_m_2) { g_u2[threadoffset_p_res - 1] = r_u2; }
    //ne
    // if (i_eq_res_m_2 && j_eq_res_m_2) { g_u2[threadoffset_p_res + 1] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    if (i_eq_res_m_2 && j_eq_res_m_2) { g_u2[threadoffset_p_res + 1] = r_u2; }
	*/
}
