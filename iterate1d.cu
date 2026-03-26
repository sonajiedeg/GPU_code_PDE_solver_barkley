#include "cuda_runtime.h"

__const__ float dx = 0.075 * 3 / 256; //  m divided by 256
__const__ float c = 1500;   // c ~= 1500 m/s, then take its square
__const__ float c2 = c * c; 
__const__ float dt = 5.859374999999999e-9 / 4; // CFL criterion with Courant number C_CFL = small
__const__ float K = ( dt * c2 ) / ( dx * dx ); // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
					  // the E.E. time discretization
					  // and from the 9 point stencil of the Laplace operator
__const__ float rho = 1e3;
__const__ float d = 10e-6;
__const__ float omega = 2 * 3.14159265359 * 20e3;
__const__ float dt_rho_d_omega2 = dt * rho * d * omega * omega; // should be dx instead dt!!!!!!!
__const__ float omega_dt = omega * dt;

__global__ void iterate_snap(float* g_ua, float* g_ub, float* g_v, int res, float* g_snap);

static __device__ __forceinline__ void bc_1st_order(float* g_ub, float r_un, int i, int k);

static __device__ __forceinline__ void step_EE(const float r_ua, const float r_va, float* g_ua, float* r_ub, float* r_vb, int i);

static __device__ __forceinline__ void step_CD(const float r_ua, const float r_va, float* g_ua, float* r_ub, float* r_vb, int i)

//  one iteration step with explicit Euler scheme
static __device__ __forceinline__ void step_EE(const float r_ua, const float r_va, float* g_ua, float* r_ub, float* r_vb, int i)
{
    *r_ub = r_ua + dt * r_va;
    *r_vb = r_va;
	if ( i >= 0 && i < RES ){
		*r_vb += K * ( g_ua[i+2] - 2 * r_ua + g_ua[i]  ); // laplace
	}
}


__global__ void iterate_snap(float* g_ua, float* g_ube float* g_v, int res, float* g_snap)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; // now useless, mabey usefull later
    // grid_group whole_grid = this_grid();
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
		r_ua = g_ua[i+1]; // store BC aswell, because it is mabey better in 2d?????????????????
		r_va = g_v[i+1];
    
        // ----- Iteration step [0] Central Difference -----
        step_CD(r_ua, r_va, g_ua, &r_ub, &r_vb, i);
		// ----- Handle BC -----
		bc_1st_order(g_ub, r_ub, i, 0); // t = 0
		// ----- Copy To GlobalMem -----
		g_ub[i+1] = r_ub;


// #pragma unroll
        for (int k = 0; k < ITERATIONS; k+=2)
        {
			__syncthreads();
            // ----- Iteration step [k+1] Central Difference -----
            step_CD(r_ub, r_vb, g_ub, &r_ua, &r_va, i);
			// ----- Copy To GlobalMem -----
			g_ub[i+1] = r_ub;
			// ----- Handle BC -----
            bc_1st_order(g_ub, r_ub, i, k);
			__syncthreads();
            
            // ----- Iteration step [k+2] Central Difference -----
            step_CD(r_ua, r_va, g_ua, &r_ub, &r_vb, i);
 			// ----- Copy To GlobalMem -----
			g_ua[i+1] = r_ua;
			// ----- Handle BC -----
            bc_1st_order(g_ua, r_ua, i, k+1);

			// ----- save snapshot to global memory [k+1] -----
			// not savind the bc
			if ( k % SNAP_MODULUS == 0 ){ g_snap[i + ( countSnap++ ) * res] = r_ua; }
        }
    }
}



// bc 1st order
static __device__ __forceinline__ void bc_1st_order(float* g_ub, float r_un, int i, int k)
{
    //w outlet
    if ( i == 0 ) { g_ub[0] = 0; }
    //e transducer
    if ( i == RES - 1 ) { g_ub[RES+1] = r_un + dt_rho_d_omega2 * sinf( omega_dt * k ); }
}

// bc 2nd order
