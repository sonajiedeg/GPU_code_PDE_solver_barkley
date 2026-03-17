#include "cuda_runtime.h"

__const__ float dx = 0.075 * 3 / 256; //  m divided by 256
__const__ float c = 1500;   // c ~= 1500 m/s, then take its square
__const__ float c2 = c * c; 
__const__ float dt = 5.859374999999999e-9 / 4; // CFL criterion with Courant number C_CFL = small
__const__ float K = ( dt * c2 ) / ( 6 * dx * dx ); // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
					  // the E.E. time discretization
					  // and from the 9 point stencil of the Laplace operator
__const__ float rho = 1e3;
__const__ float d = 10e-6;
__const__ float omega = 2 * 3.14159265359 * 20e3;
__const__ float dt_rho_d_omega2 = dt * rho * d * omega * omega;
__const__ float omega_dt = omega * dt;


static __device__ __forceinline__ void copy_r_to_g_and_no_flux(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset);
static __device__ __forceinline__ void laplace_gi(float* g_ua, float* r_un, int res, int i, int j, int ThreadOffset, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s);
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one, int threadoffset_p_res, int threadoffset_m_res, int within_simspace_inner, int k);


__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap);

__global__ void iterate(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2);


// deal with laplace
static __device__ __forceinline__ void laplace_gi(float* g_ua, float* r_vn, int res, int i, int j, int ThreadOffset, int within_simspace_inner, float* lap_p, float* lap_n, float* lap_s)
{
    if (within_simspace_inner)       // INTO REG: within_simspace_inner
    {
        float r_L = 0;
		// Patra-Karttunen (mabey try Oono-Puri)
        r_L += -20 * lap_p[0];
        r_L += 4 * lap_p[+1];
        r_L += 4 * lap_p[-1];

        r_L += 4 * lap_s[0];
        r_L += 1 * lap_s[+1];
        r_L += 1 * lap_s[-1];

        r_L += 4 * lap_n[0];
        r_L += 1 * lap_n[+1];
        r_L += 1 * lap_n[-1];

        *r_vn += K * r_L;
    }
}

// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(const float r_ua, const float r_va, float* r_ub, float* r_vb)
{
    *r_vb = r_va; // laplace part added in an other function
    *r_ub = r_ua + dt * r_va;
    
}


__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap)
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

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;

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
            laplace_gi(g_ua, &r_vb, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, r_vb, &r_ua, &r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k);
            whole_grid.sync();
            
            // ----- Laplace 2 -----
            laplace_gi(g_ub, &r_va, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k);

			// ----- save snapshot to global memory -----
			if ( k % SNAP_MODULUS == 0 ){ g_snap[ThreadOffset + ( countSnap++ ) * size] = r_ua; }
        }
    }
}

__global__ void iterate(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2)
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

		// ----- Counter for Snapshots -----
		int countSnap = 0;
		int size = res * res;

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
            laplace_gi(g_ua, &r_vb, res, i, j, ThreadOffset, within_simspace_inner, lap_pa, lap_na, lap_sa);

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, r_vb, &r_ua, &r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k);
            whole_grid.sync();
            
            // ----- Laplace 2 -----
            laplace_gi(g_ub, &r_va, res, i, j, ThreadOffset, within_simspace_inner, lap_pb, lap_nb, lap_sb);

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, res, i, j, ThreadOffset, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k);
        }
    }
}

// no flux boundary condition INT FRIEND V2, testing register usage
static __device__ __forceinline__ void copy_r_to_g_and_no_flux_int_friend(float* g_ub, float r_un, int res, int i, int j, int ThreadOffset, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k)
{
    // ----- Copy To GlobalMem -----
    if (within_simspace_inner)
    {
        g_ub[ThreadOffset] = r_un;
    }

    // ----- No Flux BC -----
    //s wall
    if (j_eq_1) { g_ub[threadoffset_m_res] = r_un; }
    //w outlet
    if (i_eq_1) { g_ub[threadoffset_m_one] = 0; }
    //e transducer
    if (i_eq_res_m_2) {
		if ( j > 64 && j < 192 ) { g_ub[threadoffset_p_one] = r_un + dt_rho_d_omega2 * sinf( omega_dt * k ); }
		else { g_ub[threadoffset_p_one] = r_un; }
	}
    //n wall
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
