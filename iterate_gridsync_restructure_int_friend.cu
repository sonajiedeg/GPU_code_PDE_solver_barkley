#include "cuda_runtime.h"
#include "param.h"


static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_ua, float r_ub, float r_uc, int res, int i, int j, int ThreadOffset);

static __device__ __forceinline__ void laplace_gi(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params);

static __device__ __forceinline__ void laplace_grad(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady, int snapCount, int ThreadOffset, int k);

static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_u2, float r_u2, float r_u1, float r_u0, float* r_u1_bc, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k, SimParamStruct params);

static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_u2, float* r_u2, float* r_u1, float* r_u0, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k, SimParamStruct params);


__global__ void iterate_snap_grad(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady);

__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params);

__global__ void iterate_snapR(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params, float* g_phi1a, float* g_phi1b, float* g_phi2a, float* g_phi2b);

__global__ void iterate(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, SimParamStruct params);


static __device__ __forceinline__ void laplace_grad(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params, float* g_snap_gradx, float* g_snap_grady, int snapCount, int ThreadOffset, int k)
{
    if (within_simspace_inner)       // INTO REG: within_simspace_inner
    {
        float r_L = 0;
		// Patra-Karttunen (mabey try Oono-Puri)
        r_L += -4 * lap_p[0];

		float n = lap_n[0];
		float s = lap_s[0];
		float e = lap_p[+1];
		float w = lap_p[-1];

		r_L += 1 * e;
        r_L += 1 * w;
        r_L += 1 * s;
        r_L += 1 * n;

		/*
        r_L += 1 * lap_p[+1];
        r_L += 1 * lap_p[-1];
        r_L += 1 * lap_s[0];
        r_L += 1 * lap_n[0];
		*/
		/*
        r_L += 1 * lap_s[+1];
        r_L += 1 * lap_s[-1];
        r_L += 1 * lap_n[+1];
        r_L += 1 * lap_n[-1];
		*/
		// save gradients
 		int size = 256 * 256;

		if ( (k*2) % params.SNAP_MODULUS == 0 ){
			g_snap_gradx[ ThreadOffset + snapCount * size ]	= ( e - w ) / 2;
			g_snap_grady[ ThreadOffset + snapCount * size ]	= ( n - s ) / 2;
		}
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
        r_L += -4 * lap_p[0];

        r_L += 1 * lap_p[+1];
        r_L += 1 * lap_p[-1];
        r_L += 1 * lap_s[0];
        r_L += 1 * lap_n[0];

        // r_L += 1 * lap_s[+1];
        // r_L += 1 * lap_s[-1];
        // r_L += 1 * lap_n[+1];
        // r_L += 1 * lap_n[-1];

        *r_un += params.K * r_L;
    }
}

static __device__ __forceinline__ void laplace_gi(float* r_un, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s, SimParamStruct params, )
{
    if (within_simspace_inner)       // INTO REG: within_simspace_inner
    {
        float r_L = 0;
		// Patra-Karttunen (mabey try Oono-Puri)
        r_L += -4 * lap_p[0];

        r_L += 1 * lap_p[+1];
        r_L += 1 * lap_p[-1];
        r_L += 1 * lap_s[0];
        r_L += 1 * lap_n[0];

        // r_L += 1 * lap_s[+1];
        // r_L += 1 * lap_s[-1];
        // r_L += 1 * lap_n[+1];
        // r_L += 1 * lap_n[-1];

        *r_un += params.K * r_L;
    }
}

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(float r_ua, float* r_ub, float r_uc)
{
	*r_ub = 2 * r_ua - r_uc; // TODO: Check if it is correct, but try first...
}

static __device__ __forceinline__ void iterate_without_laplaceR(float* r_ua, float* r_ub, float* r_uc, float* eta, float** cx, SimParamStruct params, float** r_phia, r_phib)
{
	r_ub[0] = cx[0][0] * r_ua[0] + cx[1][0] * r_ub[0];
	r_ub[1] = cx[0][1] * r_ua[1] + cx[1][1] * r_ub[1];
	r_phib[0][0] = r_phib[0][0] - 2 * params.dt * eta[0] * r_rphia[0][0];	
	r_phib[0][1] = r_phib[0][1] - 2 * params.dt * eta[1] * r_rphia[0][1];
	r_phib[1][0] = r_phib[1][0];
	r_phib[1][1] = r_phib[1][1];
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
        float r_va;
        float r_ub = 0;
        float r_vb;
		float r_u_bc = 0;
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
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
            laplace_grad(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params, g_snap_gradx, g_snap_grady, countSnap, ThreadOffset, k);

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
            laplace_grad(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params, g_snap_gradx, g_snap_grady, countSnap, ThreadOffset, k);
            // laplace_gi(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);


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
        float r_va;
        float r_ub = 0;
        float r_vb;
		float r_u_bc = 0;
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
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


__global__ void iterate_snapR(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap, SimParamStruct params, float* g_phi1a, float* g_phi1b, float* g_phi2a, float* g_phi2b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    grid_group whole_grid = this_grid();
    int within_simspace_inner = i < res - 1 && j < res - 1 && i > 0 && j > 0;


    if (i < res && j < res)            // INTO REG
    {
        int ThreadOffset = 2 * i + 2 * j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c műveletet 1 órajel alatt megcsinálja)       // INTO REG, allready done

        float* lap_pa = g_ua + ThreadOffset; // INTO REG: lap_p
        float* lap_sa = lap_pa - 2 * res; // INTO REG: lap_s
        float* lap_na = lap_pa + 2 * res; // INTO REG: lap_n

        float* lap_pb = g_ub + ThreadOffset; // INTO REG: lap_p
        float* lap_sb = lap_pb - 2 * res; // INTO REG: lap_s
        float* lap_nb = lap_pb + 2 * res; // INTO REG: lap_n

        int threadoffset_p_one = ThreadOffset + 1;
        int threadoffset_m_one = ThreadOffset - 1;
        int threadoffset_p_res = ThreadOffset + 2 * res;
        int threadoffset_m_res = ThreadOffset - 2 * res;

        bool j_eq_1 = j == 1;
        bool j_eq_res_m_2 = j == res - 2;
        bool i_eq_1 = i == 1;
        bool i_eq_res_m_2 = i == res - 2;

        // ----- Variables -----
        float r_ua[2] = {0.0};
        float r_ub[2] = {0.0};
		float r_phi1a[2] = {0.0};
		float r_phi1b[2] = {0.0};
		float r_phi2a[2] = {0.0};
		float r_phi2b[2] = {0.0};
		float* r_phia = { r_phi1a, r_phi2a };
		float* r_phib = { r_phi1b, r_phi2b };

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = 2 * res * res;

		// ----- Damping coeff -----
		float eta[2];
		if j>=128{
			eta[0] = 0.0;
			eta[1] = 0.0;
		}
		else{
			eta[0] = ( 2 * j - 255) * ( 2 * j - 255) / 255 / 255;
			eta[1] = ( 2 * j + 1 - 255 ) * ( 2 * j + 1 - 255 ) / 255 / 255;
		}
		
		// coefficients
		float cx0[2] = { (2) / ( 1 + eta[0] * params.dt / 2 ), (2) / ( 1 + eta[1] * params.dt / 2 ) };
		float cx1[2] = { ( params.dt / 2 * eta[0] - 1 ) / ( 1 + eta[0] * params.dt / 2 ), ( params.dt / 2 * eta[1] - 1 ) / ( 1 + eta[1] * params.dt / 2 ) };
		float cx2[2] = { -1 * params.dt * params.c2 / params.dx * eta[0], -1 * params.dt * params.c2 / params.dx * eta[1] };
		float cx3[2] = { params.K / ( 1 + eta[0] * params.dt / 2 ), params.K / ( 1 + eta[1] * params.dt / 2 ) };
		float cx4[2] = { params.K / params.dx / ( 1 + eta[0] * params.dt / 2 ), params.K / params.dx / ( 1 + eta[1] * params.dt / 2 ) };
		float* cx[5] = { cx0, cx1, cx2, cx3, cx4 };

        // ----- Load Data -----
        // copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
		r_ua[0] = g_ua[ThreadOffset];
		r_ua[1] = g_ua[ThreadOffset+1];
		r_phi1a[0] = g_phi1a[ThreadOffset];
		r_phi1a[1] = g_phi1a[ThreadOffset+1];
		r_phi1b[0] = g_phi1b[ThreadOffset];
		r_phi1b[1] = g_phi1b[ThreadOffset+1];
		r_phi2a[0] = g_phi2a[ThreadOffset];
		r_phi2a[1] = g_phi2a[ThreadOffset+1];
		r_phi2b[0] = g_phi2b[ThreadOffset];
		r_phi2b[1] = g_phi2b[ThreadOffset+1];
        //whole_grid.sync();

		// ----- save u_n{n-1} for BC at u_{n+1} 1-----
		/*
		float r_uc1[2] = {0.0};
		float r_uc2[2] = {0.0};
		float r_phi1c1[2] = {0.0};
		float r_phi1c2[2] = {0.0};
		float r_phi2c1[2] = {0.0};
		float r_phi2c2[2] = {0.0};
		r_uc1[0] = r_ub[0];
		r_uc1[1] = r_ub[1];
		r_phi1c1[0] = r_phi1b[0];
		r_phi1c1[1] = r_phi1b[1];
		r_phi2c1[0] = r_phi2b[0];
		r_phi2c1[1] = r_phi2b[1];
		*/
		
        // ----- Iteration step 1 -----
        iterate_without_laplaceR(r_ua, r_ub, eta, cx, params, r_phia, r_phib);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
			// ----- Laplace 1 -----
            // laplace_gi(&r_ub, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);
            laplace_giR(&r_ub, within_simspace_inner, lap_pa, lap_na, lap_sa, params);

			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc2 = r_ua;

            // ----- Iteration step 2 
            iterate_without_laplaceR(r_ub, &r_ua, r_uc2);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friendR(g_ub, r_ub, r_ua, r_uc1, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k, params);
			// ----- save snapshot to global memory -----
			if ( (k*2) % params.SNAP_MODULUS == 0 ){ g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ua; }

            whole_grid.sync();
            
            // ----- Laplace 2 -----
            //laplace_gi(&r_ua, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);
            laplace_giR(&r_ua, within_simspace_inner, lap_pb, lap_nb, lap_sb, params);


			// ----- save u_n{n-1} for BC at u_{n+1} 2-----
			r_uc1 = r_ub;

            // ----- Iteration step 3 -----
            iterate_without_laplaceR(r_ua, &r_ub, r_uc1);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friendR(g_ua, r_ua, r_ub, r_uc2, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k+1, params);
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
        float r_va;
        float r_ub = 0;
        float r_vb;

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;

		// ----- init previous boundary data:
		float r_u_bc = 0;
		// float r_ua_bc = 0;
		// float r_ub_bc = 0;

        // ----- Load Data -----
        copy_g_to_r(g_ua, g_v, &r_ua, &r_va, ThreadOffset);
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
	if (j_eq_1) { g_u2[threadoffset_m_res] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// sommerfeld/impedance
	// if (j_eq_1) { float r_u_tmp = *r_u1_bc + dt*Z*c/dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_m_res] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order

    // ============================== W
    // impedance/sommerfeld
	// if (i_eq_1) { float r_u_tmp = *r_u1_bc + params.dt*params.Z*params.c/params.dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_m_one] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order
	// u[0, 1:ny-1, 0] = (u[1, 1:ny-1, 1] - kappa * ( u[0, 1:ny-1, 1] - u[1, 1:ny-1, 0]) )
	/*
	if (i_eq_1) {
		float r_u_tmp = r_u1 - params.kappa * ( r_u2 - *r_u1_bc );
		g_u2[threadoffset_m_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	} // temp 1st order impl. TODO: 2nd order
	*/
	// mur's first order 

	if (i_eq_1) {
		float r_u_tmp = r_u1 + ( params.c * params.dt - params.dx ) / ( params.c * params.dt + params.dx ) * ( r_u2 - *r_u1_bc );
		g_u2[threadoffset_m_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	} // temp 1st order impl. TODO: 2nd order
	
	// dirichlet
	// if (i_eq_1) { g_u2[threadoffset_m_one] = 0; }
	// neumann
	// if (i_eq_1) { g_u2[threadoffset_m_one] = r_u2 + 1/2/K * ( r_u2 - 2 * r_u1 + r_u0 ); }
    // ============================== E
    //e transducer
    if (i_eq_res_m_2) {
		// if ( 64 <= j && j <= 192 ) { g_u2[threadoffset_p_one] = r_u2 + params.sin_coeff * sinf( params.omega_dt * k ) - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
		if ( 64 <= j && j <= 192 ) { g_u2[threadoffset_p_one] = r_u2 + params.dt_rho_d_omega2 * sinf( params.omega_dt * k ); }
		else { g_u2[threadoffset_p_one] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	}
    // ============================== N
    if (j_eq_res_m_2) { g_u2[threadoffset_p_res] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// if (j_eq_res_m_2) { float r_u_tmp = *r_u1_bc - dt*Z*c/dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_p_res] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order

    //sw
    if (i_eq_1 && j_eq_1) { g_u2[threadoffset_m_res - 1] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    //se
    if (i_eq_res_m_2 && j_eq_1) { g_u2[threadoffset_m_res + 1] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    //nw
    if (i_eq_1 && j_eq_res_m_2) { g_u2[threadoffset_p_res - 1] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    //ne
    if (i_eq_res_m_2 && j_eq_res_m_2) { g_u2[threadoffset_p_res + 1] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
}

static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_u2, float r_u2, float r_u1, float r_u0, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k, SimParamStruct params)
{
    // ----- Copy To GlobalMem -----
    if (within_simspace_inner)
    {
        g_u2[ThreadOffset] = r_u2;
    }

    // ----- No Flux BC -----
    // ============================== S
	// neumann
	if (j_eq_1) { g_u2[threadoffset_m_res] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// sommerfeld/impedance
	// if (j_eq_1) { float r_u_tmp = *r_u1_bc + dt*Z*c/dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_m_res] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order

    // ============================== W
    // impedance/sommerfeld
	// if (i_eq_1) { float r_u_tmp = *r_u1_bc + params.dt*params.Z*params.c/params.dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_m_one] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order
	// u[0, 1:ny-1, 0] = (u[1, 1:ny-1, 1] - kappa * ( u[0, 1:ny-1, 1] - u[1, 1:ny-1, 0]) )
	/*
	if (i_eq_1) {
		float r_u_tmp = r_u1 - params.kappa * ( r_u2 - *r_u1_bc );
		g_u2[threadoffset_m_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	} // temp 1st order impl. TODO: 2nd order
	*/
	// mur's first order 

	if (i_eq_1) {
		float r_u_tmp = r_u1 + ( params.c * params.dt - params.dx ) / ( params.c * params.dt + params.dx ) * ( r_u2 - *r_u1_bc );
		g_u2[threadoffset_m_one] = r_u_tmp;
		*r_u1_bc = r_u_tmp;
	} // temp 1st order impl. TODO: 2nd order
	
	// dirichlet
	// if (i_eq_1) { g_u2[threadoffset_m_one] = 0; }
	// neumann
	// if (i_eq_1) { g_u2[threadoffset_m_one] = r_u2 + 1/2/K * ( r_u2 - 2 * r_u1 + r_u0 ); }
    // ============================== E
    //e transducer
    if (i_eq_res_m_2) {
		// if ( 64 <= j && j <= 192 ) { g_u2[threadoffset_p_one] = r_u2 + params.sin_coeff * sinf( params.omega_dt * k ) - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
		if ( 64 <= j && j <= 192 ) { g_u2[threadoffset_p_one] = r_u2 + params.dt_rho_d_omega2 * sinf( params.omega_dt * k ); }
		else { g_u2[threadoffset_p_one] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	}
    // ============================== N
    if (j_eq_res_m_2) { g_u2[threadoffset_p_res] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
	// if (j_eq_res_m_2) { float r_u_tmp = *r_u1_bc - dt*Z*c/dx * ( r_u1 - *r_u1_bc ); g_u2[threadoffset_p_res] = r_u_tmp; *r_u1_bc = r_u_tmp; } // temp 1st order impl. TODO: 2nd order

    //sw
    if (i_eq_1 && j_eq_1) { g_u2[threadoffset_m_res - 1] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    //se
    if (i_eq_res_m_2 && j_eq_1) { g_u2[threadoffset_m_res + 1] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    //nw
    if (i_eq_1 && j_eq_res_m_2) { g_u2[threadoffset_p_res - 1] = r_u2 + params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
    //ne
    if (i_eq_res_m_2 && j_eq_res_m_2) { g_u2[threadoffset_p_res + 1] = r_u2 - params.bck * ( r_u2 - 2 * r_u1 + r_u0 ); }
}
