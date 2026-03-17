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




__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap);





// iterate one timestep without laplace
static __device__ __forceinline__ void iterate_without_laplace(const float r_ua, const float r_va, float* r_ub, float* r_vb)
{
    *r_vb = r_va; // laplace part added in an other function
    *r_ub = r_ua + dt * r_va;
    
}


__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, int iteration_limit_per_2, float* g_snap)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    grid_group whole_grid = this_grid();
    int within_simspace_inner = i < res - 1 && i > 0;


    if (i < res)            // INTO REG
    {
        // ----- Variables -----
        float r_ua;
        float r_va;
        float r_ub;
        float r_vb;
		// ----- Counter for Snapshots -----
		int countSnap = 0;
        // ----- Load Data -----
		r_ua = g_ua[i];
		r_va = g_v[i];
    
        // ----- Iteration step 1 -----
        iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

// #pragma unroll
        for (int k = 0; k < iteration_limit_per_2; k++)
        {
            whole_grid.sync();
            // ----- Laplace 1 -----
            laplace_gi(g_ua, &r_vb, res, i, j, i, within_simspace_inner, lap_pa, lap_na, lap_sa);

            // ----- Iteration step 2 
            iterate_without_laplace(r_ub, r_vb, &r_ua, &r_va);

            // ----- "Unload" Data And No Flux Bc 1 -----
            copy_r_to_g_and_no_flux_int_friend(g_ub, r_ub, res, i, j, i, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k);
            whole_grid.sync();
            
            // ----- Laplace 2 -----
            laplace_gi(g_ub, &r_va, res, i, j, i, within_simspace_inner, lap_pb, lap_nb, lap_sb);

            // ----- Iteration step 3 -----
            iterate_without_laplace(r_ua, r_va, &r_ub, &r_vb);

            // ----- "Unload" Data And No Flux Bc 2 -----
            copy_r_to_g_and_no_flux_int_friend(g_ua, r_ua, res, i, j, i, j_eq_1, j_eq_res_m_2, i_eq_1, i_eq_res_m_2, threadoffset_p_one, threadoffset_m_one, threadoffset_p_res, threadoffset_m_res, within_simspace_inner, k);

			// ----- save snapshot to global memory -----
			if ( k % SNAP_MODULUS == 0 ){ g_snap[i + ( countSnap++ ) * size] = r_ua; }
        }
    }
}



// no flux boundary condition INT FRIEND V2, testing register usage
static __device__ __forceinline__ void copy_r_to_g_and_bc(float* g_ub, float r_un, int res, int i, int j, int i, bool j_eq_1, bool j_eq_res_m_2, bool i_eq_1, bool i_eq_res_m_2, int threadoffset_p_one, int threadoffset_m_one , int threadoffset_p_res , int threadoffset_m_res, int within_simspace_inner, int k)
{
    // ----- Copy To GlobalMem -----
    if (within_simspace_inner)
    {
        g_ub[i] = r_un;
    }

    // ----- No Flux BC -----
    //w outlet
    if (i_eq_1) { g_ub[threadoffset_m_one] = 0; }
    //e transducer
    if (i_eq_res_m_2) {
		if ( j > 64 && j < 192 ) { g_ub[threadoffset_p_one] = r_un + dt_rho_d_omega2 * sinf( omega_dt * k ); }
		else { g_ub[threadoffset_p_one] = r_un; }
	}
}
