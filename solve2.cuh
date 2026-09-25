#include "loadp.cpp"
#include "interp.cpp"
#include "solSave.cpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>

// Define a type for our state vector (equivalent to y(:, i))
using State = std::vector<float>;
// Define a type for our ODE function: dydt = f(y)
// using OdeFunc = std::function<State(const State&, const float, const float, const float)>;
using OdeFunc = std::function<State(const State&, const float, const float, const float, const float, const float, const float [3], const float [12], const float [3], const SimParamStruct)>;

struct Solution {
    std::vector<float> t;
    std::vector<State> y;
    std::vector<float> H;
    std::vector<float> error;
	// Returns the time series for state component 'state_idx'
    std::vector<float> get_state_series(std::size_t state_idx) const {
        std::vector<float> series;
        series.reserve(y.size()); // Pre-allocate memory for performance

        for (const auto& state : y) {
            series.push_back(state[state_idx]);
        }

        return series;
    }
};

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params);
void loadp(float*pArr, float* pSnap, const State& st, float t0, SimParamStruct params, int segment);
void interp(float* pArr, const State& st, float t,float* p,float* dpdx,float* dpdy,float* dpdt,float* dpdtdx,SimParamStruct params);
void solSave(char* fname, std::vector<float> sol, SimParamStruct params);

// Helper function for using CUDA to add vectors in parallel.
cudaError_t solve2(float* u, float* v, float* u_sol, unsigned int size, int res, SimParamStruct params)
{
	float* u2 = new float[ size ];
	float* u1 = new float[ size ];
	for(int i = 0; i<size; i++){ u1[i]=u[i]; u2[i]=u[i]; }
	cudaError_t cudaStatus;
	int iteration_limit_per_2 = 0;

	int activeSteps;
	float* dev_ua = 0;
	float* dev_ub = 0;
	float* dev_2_u = 0;
	float* dev_snap = 0;
	void* kernelArgs[6];
	int kStepStart = 0;
	kernelArgs[0] = (void*)&(dev_ua);
	kernelArgs[1] = (void*)&(dev_ub);
	kernelArgs[2] = (void*)&iteration_limit_per_2;
	kernelArgs[3] = (void*)&(dev_snap);
	kernelArgs[4] = (void*)&params;
	kernelArgs[5] = (void*)&kStepStart;

	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(4, 4, 1); // swich

	// setup bubble solver params
	int itLim = 1000000;
	float abstol = 1e-5;
	float t0,t1,tSegmentEnd,x0,x1,v0,v1,y0,y1,w0,w1,p0,pb,dpdx,dpdy,dpdt,dpdtdx,tmp;
	float pArr[8];
	int stateVarCount = 6; // R V x v y u
	float* S0 = new float[stateVarCount]; 
	float* S1 = new float[stateVarCount]; 
	for(int i=0;i<stateVarCount;i++){ S0[i]=0; S1[i]=0; }
	p0 = 101325.0f;
	pb = p0;
	t0 = 0.0f;
	x0 = params.lInWavelenght * params.c / params.f / 2.0f;	
	y0 = params.lInWavelenght * params.c / params.f / 2.0f;	
	v0 = 0.0f;
	w0 = 0.0f;
	float up[3] = {0.0f};
	float pp[12] = {0.0f};
	float sp[3] = {0.0f};
	paramSetupBubbleSolver(up, pp, sp, params);

	// 1. Define ODE function (e.g., a simple harmonic oscillator or 2D system)
	/*
    OdeFunc dSdtOFF = [](const State& Y, const float pb, const float dpx, const float dpy) -> State {
        // Example: Y[0]' = Y[1], Y[1]' = -Y[0]
        return { Y[1], -40000000*Y[0], Y[3], -Y[2], Y[5], -Y[4] };
    };
	*/

	// -------------------------------------------------
	OdeFunc dSdt = [](const State& Y, const float pb, const float dpdx, const float dpdy, const float dpdt, const float dpdtdx, const float up[3], const float pp[12], const float sp[3], const SimParamStruct params) -> State {
		// Radial Oscillation
		float R = Y[0];        // Dimensionless Radius
		float U = Y[3];        // Dimensionless Wall velocity
		float px =Y[1];        // Dimensionless position
		float vx =Y[4];        // Dimensionless translation velocity

		State F(Y.size(), 0.0);

		float inv_R = 1.0 / R;
		float p = (float)std::pow(up[2] * inv_R,sp[0]);

		// -- Radial Oscillation --
		float N = (up[0] + up[1]*U) * p; 
		N-= pp[0] * (1 + pp[5]*U) - pp[1] * inv_R - pp[2]*U*inv_R;
		N-= 1.5 * (1.0 - pp[5]*U * (1.0/3.0))*U*U;
		N-= (1 + pp[5]*U) * pp[3] * pb - pp[4] * dpdt * params.f * R;
		N+= 0.25 * vx * vx;
		

		float D = R - pp[5]*R*U + pp[2]*pp[5];
		float rD = 1.0 / D;
		// -- Translational motion --
		float Fb = -pp[7]*R*R*R * dpdx / (params.c/params.f); // Primary Bjerknes Force
		float u = - dpdtdx / params.rho / params.f / (params.c/params.f);
		float Fd = -pp[8]*R * (vx * pp[9] - u);     // Drag Force
		// hereby, u(x,t) is the liquid velocity induced by the acousitc irradiaiton calculated at the center of the bubble [ref...]
		float dv = 3*(Fb + Fd)*pp[6]*inv_R*inv_R*inv_R - 3.0*U*inv_R*vx;

		F[0] = U;
		F[1] = vx;
		F[3] = rD * N;
		F[4] = dv;

        return { F[0], F[1], F[2], F[3], F[4], F[5] };
    };

	// -------------------------------------------------

	float t_start = 0.0f;

	// Data from the Butcher tableau for Fehlberg4(5) pair
    const float b[5] = { 25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0 }; // 4th order
    const float bh[6] = { 16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0 }; // 5th order

    std::vector<std::vector<float>> a(6, std::vector<float>(6, 0.0));
    a[1][0] = 1.0 / 4.0;
    a[2][0] = 3.0 / 32.0;       a[2][1] = 9.0 / 32.0;
    a[3][0] = 1932.0 / 2197.0;  a[3][1] = -7200.0 / 2197.0;  a[3][2] = 7296.0 / 2197.0;
    a[4][0] = 439.0 / 216.0;    a[4][1] = -8.0;              a[4][2] = 3680.0 / 513.0;    a[4][3] = -845.0 / 4104.0;
    a[5][0] = -8.0 / 27.0;      a[5][1] = 2.0;               a[5][2] = -3544.0 / 2565.0;  a[5][3] = 1859.0 / 4104.0;  a[5][4] = -11.0 / 40.0;

	// const State ic = { 10.f, 0.0f, 0, 0, 0, 0 };
	const State ic = { 1.0f, x0/(params.c/params.f), y0/(params.c/params.f), 0, 0, 0 }; // R x y U vx vy
    int num_eq = ic.size();
    
    // Initialize results structure
    Solution sol;
    sol.t.push_back(t_start);
    sol.y.push_back(ic);

    // Set parameters for error control
	// float h_init = params.dt / 4.0f;
	float h_init = 1e-6;
	// float h_max = params.dt / 2.0f;
	float h_max = params.dt * params.f / 2;
    float h = h_init;
    float abstolmin = abstol * 0.1;
    float abstolmax = abstol * 10.0;
    float frac = 0.9;
    float p = 4.0;

	//measure elapsed time
	std::chrono::steady_clock::time_point beginTot;
	std::chrono::steady_clock::time_point endTot;

	beginTot = std::chrono::steady_clock::now();


	for(int segment = 0; segment < params.segments; segment++ )
	{
		activeSteps = ( segment < params.segments-1 ) ? params.maxStepPerSegment : params.remainder;
		float* uSnap = new float[ size * int( activeSteps ) ];


		//measure at this iteration
		iteration_limit_per_2 += activeSteps / 2;
		// printf("active steps: %d\n", activeSteps);
		// printf("iter lim / 2: %d\n", iteration_limit_per_2);
		printf("segment/segments: %d / %d \n", segment, params.segments);

		kStepStart = segment * params.maxStepPerSegment / 2;

		// cooperative kernel launch parameters
		
		//measure elapsed time
		// std::chrono::steady_clock::time_point begin;
		// std::chrono::steady_clock::time_point end;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(DEVICE_ID); CUERR;

		//-----------------------------------------------------------------------------------------------------------
		//---------------------------------------- GRIDSYNC RESTRUCT INT FRIEND KERNEL -------------------------------
		//-----------------------------------------------------------------------------------------------------------
		if (run_gsync_restruct_int_friend)
		{
			// cout << "gridsync restruct \n";
			// Allocate GPU buffers


			// init(u1, u2, dev_2_u, dev_ua, dev_ub, dev_snap, params, activeSteps)
			init(u1, u2, &dev_2_u, &dev_ua, &dev_ub, &dev_snap, params, activeSteps); CUERR;
			// init(u, v, &dev_2_u, &dev_ua, &dev_ub, &dev_v, size, DEVICE_ID, &dev_snap, params, &dev_snap_gradx, &dev_snap_grady); CUERR;
			cudaDeviceSynchronize();
			// ---------- Launch a kernel on the GPU with one thread for each element. ----------
			// begin = std::chrono::steady_clock::now();
			cudaLaunchCooperativeKernel((void*)iterate_snap_segment, dimGrid, dimBlock, kernelArgs);
			cudaDeviceSynchronize(); CUERR;

			// end = std::chrono::steady_clock::now();
			// print_runtimes(begin, end, params);

			// Check for any errors launching the kernel
			cudaGetLastError(); CUERR;

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaDeviceSynchronize(); CUERR;

			
			// Copy output vector from GPU buffer to host memory.
			cudaMemcpy(u2, dev_ua, size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;
			cudaMemcpy(u1, dev_ub, size * sizeof(float), cudaMemcpyDeviceToHost); CUERR;

			if ( params.SNAPSHOT ) { cudaMemcpy(uSnap, dev_snap, size * sizeof(float) * int( activeSteps ), cudaMemcpyDeviceToHost); CUERR; }

			cudaDeviceSynchronize(); CUERR;

			char fname[100];
			snprintf(fname, sizeof(fname), "u_snap_f_%.2f_df_%.2f_s_%d.bin",params.f, params.df,segment);
			snap_to_text(uSnap, params, fname, activeSteps);

			
		}


		if(run_bubble_solver)
		{
			tSegmentEnd = ( segment * params.maxStepPerSegment + activeSteps ) * params.dt;
			int it = 0;
			while(sol.t.back() < tSegmentEnd*params.f && it < itLim)
			{
					
				bool error_not_in_range = true;
				State y_next(num_eq, 0.0);
				State yh(num_eq, 0.0);
				float max_err = 0.0;
				float current_accepted_h = h;

				while (error_not_in_range) {
					loadp(pArr, uSnap, sol.y.back(), sol.t.back(), params, segment);
					interp(pArr, sol.y.back(), sol.t.back(), &pb, &dpdx, &dpdy, &dpdy, &dpdtdx, params);
					pb+=p0;

					std::vector<State> k(6, State(num_eq, 0.0));

					// Walk through the 6 stages
					for (int j = 0; j < 6; j++) {
						State Y = sol.y.back(); // calculating the y value where f will be evaluated
						for (int n = 0; n < j; n++) {
							if (a[j][n] != 0.0) { // optimization to skip 0 multiplications
								for (int dim = 0; dim < num_eq; dim++) {
									Y[dim] += h * a[j][n] * k[n][dim];
								}
							}
						}
						k[j] = dSdt(Y,pb,dpdx,dpdy,dpdt,dpdtdx,up,pp,sp,params); // Evaluate function
					}
					// Calculate 4th order estimate (y_next)
					y_next = sol.y.back();
					for (int j = 0; j < 5; j++) {
						if (b[j] != 0.0) {
							for (int dim = 0; dim < num_eq; dim++) {
								y_next[dim] += h * b[j] * k[j][dim];
							}
						}
					}

					// Calculate 5th order estimate (yh)
					yh = sol.y.back();
					for (int j = 0; j < 6; j++) {
						if (bh[j] != 0.0) {
							for (int dim = 0; dim < num_eq; dim++) {
								yh[dim] += h * bh[j] * k[j][dim];
							}
						}
					}

					// Examine error
					max_err = 0.0;
					float sum_err = 0.0;
					for (int dim = 0; dim < num_eq; dim++) {
						float e = std::abs(yh[dim] - y_next[dim]);
						if (e > max_err) max_err = e;
						sum_err += e;
					}

					if (sum_err == 0.0f) sum_err = 1e-15f;

					// Manipulate step size accordingly
					if (max_err > abstolmax) {
						// Step solution rejected
						error_not_in_range = true;
						h = std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h;

						// h = std::max(1e-8f, (float)std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h);
					}
					else if (max_err < abstolmin && sum_err != 0.0) {
						// Step solution accepted, but computation is slow (error too small)
						error_not_in_range = false;
						current_accepted_h = h;
						float h_tmp = std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h;

						// h = std::min(h_tmp, h_max);
						if (h_tmp <= h_max) {
							h = h_tmp; // Prepare step size for NEXT iteration
						}
					}
					else {
						// Step solution accepted, optimal range
						error_not_in_range = false;
						current_accepted_h = h;
					}
				} // End of one step attempt (rejected => repeated)
				// Store accepted values
				sol.y.push_back(y_next);
				sol.error.push_back(max_err);
				sol.H.push_back(current_accepted_h);
				
				// Advance time using the 'h' that was accepted for THIS step
				sol.t.push_back(sol.t.back() + current_accepted_h); 


				// ode45();
				it+=1;
				if(it == itLim){printf("iteration limit hit\n");}
			}
		}

	// Error:
		cudaFree(dev_2_u); CUERR;
		
		cudaFree(dev_snap); CUERR;
		
		delete[] uSnap;
	}

	endTot = std::chrono::steady_clock::now();
	print_runtimes(beginTot, endTot, params);
	
	char fname[100];
	snprintf(fname, sizeof(fname), "R_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	printf("size: %lu\n",sol.t.size());
	solSave(fname, sol.get_state_series(1), params);
	snprintf(fname, sizeof(fname), "x_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	solSave(fname, sol.get_state_series(3), params);
	snprintf(fname, sizeof(fname), "y_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	solSave(fname, sol.get_state_series(5), params);
	snprintf(fname, sizeof(fname), "t_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	solSave(fname, sol.t, params);

   
    return cudaStatus;
}


