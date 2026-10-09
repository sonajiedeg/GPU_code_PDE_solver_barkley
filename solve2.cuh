#include "loadp.cpp"
#include "interp.cpp"
#include "solSave.cpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>

// Define a type for our state vector (equivalent to y(:, i))
using State = std::vector<double>;
// Define a type for our ODE function: dydt = f(y)
// using OdeFunc = std::function<State(const State&, const float, const float, const float)>;

using OdeFunc = std::function<State(const double, const State&, const double [3], const double [12], const double [4], const SimParamStruct, float*, int)>;

struct Solution {
    std::vector<double> t;
    std::vector<State> y;
    std::vector<double> H;
    std::vector<double> Fb;
    std::vector<double> error;
	// Returns the time series for state component 'state_idx'
    std::vector<double> get_state_series(std::size_t state_idx) const {
        std::vector<double> series;
        series.reserve(y.size()); // Pre-allocate memory for performance

        for (const auto& state : y) {
            series.push_back(state[state_idx]);
        }

        return series;
    }
};

/*
struct PositionReset {
	bool xLow, yLow, xHigh, yHigh;
};
*/

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params);


void loadp(double* pArr, float* pSnap, const State& st, double tau0, SimParamStruct params, int segment);
// void loadp(double* pArr, float* pSnap, const State& st, double t0, SimParamStruct params, int segment);

// void interp(double* pArr, const State& st, double tau,double* p,double* dpdx,double* dpdy,double* dpdt,double* dpdtdx, SimParamStruct params);
void interp(double* pArr, const State& st, double t, double* p, double* dpdx, double* dpdy, double* dpdt, double* dpdtdx,SimParamStruct params);

void solSave(char* fname, std::vector<double> sol, SimParamStruct params);

// Helper function for using CUDA to add vectors in parallel.
cudaError_t solve2(float* u, float* v, float* u_sol, unsigned int size, int res, SimParamStruct params)
{
	PositionReset pr;
	pr.xLow = 0;
	pr.yLow = 0;
	pr.xHigh = 0;
	pr.yHigh = 0;
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
	dim3 dimGrid(8, 8, 1); // swich

	// setup bubble solver params
	unsigned long long int itLim = 10000000;
	double abstol = 1e-9;
	double t0,t1,tSegmentEnd,x0,x1,v0,v1,y0,y1,w0,w1,p0,pb,dpdx,dpdy,dpdt,dpdtdx,tmp;
	double pArr[8];
	int stateVarCount = 6; // R V x v y u
	p0 = 101325.0;
	pb = p0;
	t0 = 0.0f;
	x0 = params.lInWavelenght * params.c / params.f / 2.0f;	
	y0 = params.lInWavelenght * params.c / params.f / 2.0f;	
	v0 = 0.0f;
	w0 = 0.0f;
	double up[3] = {0.0};
	double pp[12] = {0.0};
	double sp[4] = {0.0};
	paramSetupBubbleSolver(up, pp, sp, params);

	// 1. Define ODE function (e.g., a simple harmonic oscillator or 2D system)
	/*
    OdeFunc dSdtOFF = [](const State& Y, const float pb, const float dpx, const float dpy) -> State {
        // Example: Y[0]' = Y[1], Y[1]' = -Y[0]
        return { Y[1], -40000000*Y[0], Y[3], -Y[2], Y[5], -Y[4] };
    };
	*/

	// -------------------------------------------------
	OdeFunc dSdt = [](const double tau, const State& Y, const double up[3], const double pp[12], const double sp[4], const SimParamStruct params, float* pSnap, int segment) -> State {
		// Radial Oscillation
		double R = Y[0];        // Dimensionless Radius
		double U = Y[3];        // Dimensionless Wall velocity
		double px =Y[1];        // Dimensionless position
		double vx =Y[4];        // Dimensionless translation velocity

		State F(Y.size(), 0.0);

		if (R <= 1e-9) {
			return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; 
		}

		double pb,dpdx,dpdy,dpdt,dpdtdx;


		double pArr[8];

		// void loadp(double* pArr, float* pSnap, State& st, double tau0, SimParamStruct params, int segment){
		loadp(pArr, pSnap, Y, tau, params, segment);

		// interp(float* pArr, const State& st, float tau,float* p,float* dpdx,float* dpdy,float* dpdt,float* dpdtdx, SimParamStruct params){
		interp(pArr, Y, tau, &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params);
		// interpSep(Y, tau, &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params);

		double inv_R = 1.0 / R;
    
		double base = up[2] * inv_R;
		if (base <= 0.0) base = 1e-9; 
		double p = (double)std::pow(base, sp[0]);

		// -- Radial Oscillation --

		double N = (up[0] + up[1]*U) * p; 
		N-= pp[0] * (1 + pp[5]*U) + pp[1] * inv_R + pp[2]*U*inv_R;
		N-= 1.5 * (1.0 - pp[5]*U * (1.0/3.0))*U*U;
		N-= (1 + pp[5]*U) * pp[3] * pb + pp[4] * dpdt * R;
		N+= 0.25 * vx * vx * params.c * params.c / params.f / params.f  / pp[11] / pp[11];
		
		double D = R - pp[5]*R*U + pp[2]*pp[5];
		double rD = 1.0 / D;
		// -- Translational motion --
		// double Fb = -pp[7]*R*R*R * dpdx / (params.c/params.f);
		// double u = - dpdtdx / params.rho / params.f / (params.c/params.f);
		double Fb = -pp[7]*R*R*R * dpdx;
		// double Fb = -pp[7]*R*R*R * dpdx;
		// double Fb = -pp[7]*R*R*R * dpdx;
		const double pi = 3.14159265359;
		double u = -params.Pa / (params.rho * params.c) * std::sin(2.0 * pi * px) * std::cos(2.0 * pi * tau);
		// double Fd = -pp[8] * R * (vx / sp[1]/sp[2] - u);
		double Fd = -pp[8] * R * (vx * sp[3] - u);
		// mod speed of sound
		// double Fd = -pp[8] * R * (vx * pp[9] - u);
		// double u = - dpdtdx / params.rho / params.omega / params.omega;
		// double Fd = -pp[8]*R * (vx * pp[9] - u);
		// hereby, u(x,t) is the liquid velocity induced by the acousitc irradiaiton calculated at the center of the bubble [ref...]
		double dv = 3*(Fb + Fd)*pp[6]*inv_R*inv_R*inv_R - 3.0*U*inv_R*vx;

		F[0] = U;
		F[1] = vx;
		F[2] = 0.0;
		F[3] = rD * N;
		F[4] = dv;
		F[5] = 0.0;
		F[6] = 0.0;
		// mod
		/*
		F[1] = vx;
		double zeta = 2.05;
		double omega_n = 0.0308;
		dv = -2 * zeta * omega_n* vx - omega_n * omega_n * px;
		F[4] = dv;
		*/

        return { F[0], F[1], F[2], F[3], F[4], F[5], F[6] };
    };

	// -------------------------------------------------

	double t_start = 0.0;

	// RKCK45
	const int o4stages = 6;
	const int o5stages = 6;
	const double b[6]  = { 2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0 }; // 4th order
    const double bh[6] = { 37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0 }; // 5th order

    const double c[6] = { 0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0 }; 
    std::vector<std::vector<double>> a(6, std::vector<double>(6, 0.0));
    a[1][0] = 1.0 / 5.0;
    a[2][0] = 3.0 / 40.0;       a[2][1] = 9.0 / 40.0;
    a[3][0] = 3.0 / 10.0;       a[3][1] = -9.0 / 10.0;      a[3][2] = 6.0 / 5.0;
    a[4][0] = -11.0 / 54.0;     a[4][1] = 5.0 / 2.0;        a[4][2] = -70.0 / 27.0;     a[4][3] = 35.0 / 27.0;
    a[5][0] = 1631.0 / 55296.0; a[5][1] = 175.0 / 512.0;    a[5][2] = 575.0 / 13824.0;  a[5][3] = 44275.0 / 110592.0; a[5][4] = 253.0 / 4096.0;

	// RKF45 
	/*
	const int o4stages = 5;
	const int o5stages = 6;

    const double b[5] = { 25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0 }; // 4th order
    const double bh[6] = { 16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0 }; // 5th order

    const double c[6] = { 0, 1/4, 3/8, 12/13, 1, 1/2 }; 
    std::vector<std::vector<double>> a(6, std::vector<double>(6, 0.0));
    a[1][0] = 1.0 / 4.0;
    a[2][0] = 3.0 / 32.0;       a[2][1] = 9.0 / 32.0;
    a[3][0] = 1932.0 / 2197.0;  a[3][1] = -7200.0 / 2197.0;  a[3][2] = 7296.0 / 2197.0;
    a[4][0] = 439.0 / 216.0;    a[4][1] = -8.0;              a[4][2] = 3680.0 / 513.0;    a[4][3] = -845.0 / 4104.0;
    a[5][0] = -8.0 / 27.0;      a[5][1] = 2.0;               a[5][2] = -3544.0 / 2565.0;  a[5][3] = 1859.0 / 4104.0;  a[5][4] = -11.0 / 40.0;
	*/

	double R0 = 80e-6;
	// const State ic = { 10.f, 0.0f, 0, 0, 0, 0 };
	const State ic = { 1.0, x0/(params.c/params.f), y0/(params.c/params.f), 0, 0, 0 }; // R x y U vx vy
	// const State ic = { 1.0, 0.009, y0/(params.c/params.f), 0, 0, 0 }; // R x y U vx vy
    int num_eq = ic.size();
    
    // Initialize results structure
    Solution sol;
    sol.t.push_back(t_start);
    sol.y.push_back(ic);
	sol.Fb.push_back(0.0);

    // Set parameters for error control
	// float h_init = params.dt / 4.0f;
	double h_init = 1e-6;
	double h_max = params.dt * params.f / 2.0;
	double h_min = 1e-12;
	// float h_max = params.dt * params.f / 2.0f;
	// float h_init = h_max;
	// float h_max = 1e-9;
    double h = h_init;
    double abstolmin = abstol * 0.1;
    double abstolmax = abstol * 10.0;
    double frac = 0.9;
    double p = 4.0;

	//measure elapsed time
	std::chrono::steady_clock::time_point beginTot;
	std::chrono::steady_clock::time_point endTot;
	beginTot = std::chrono::steady_clock::now();

	float* uSegmentEnd = new float[size*2]{};

	for(int segment = 0; segment < params.segments; segment++ )
	{
		printf("segment / segments: %d / %d\n",segment,params.segments);
		activeSteps = ( segment < params.segments-1 ) ? params.maxStepPerSegment : params.remainder;
		size_t snapSize = (size_t)size * (size_t)(activeSteps+2) ;
		float* uSnap = new float[snapSize]{};


		//measure at this iteration
		iteration_limit_per_2 += activeSteps / 2;
		// printf("active steps: %d\n", activeSteps);
		// printf("iter lim / 2: %d\n", iteration_limit_per_2);
		// printf("segment/segments: %d / %d \n", segment, params.segments);

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
			cudaMemcpy(u2, dev_ua, (size_t)size * (size_t)sizeof(float), cudaMemcpyDeviceToHost); CUERR;
			cudaMemcpy(u1, dev_ub, (size_t)size * (size_t)sizeof(float), cudaMemcpyDeviceToHost); CUERR;

			if ( params.SNAPSHOT ) { cudaMemcpy(&(uSnap[2*size]), dev_snap, (size_t)size * (size_t)sizeof(float) * (size_t)activeSteps, cudaMemcpyDeviceToHost); CUERR; }
			// append previous segment end 2 timesteps
			for(int i = 0; i<2*size; i++){ uSnap[i] = uSegmentEnd[i]; }

			cudaDeviceSynchronize(); CUERR;
	
			/*
			char fname[100];
			snprintf(fname, sizeof(fname), "uf%.0fdf%.0fs%d.bin",params.f, params.df,segment);
			snap_to_text(uSnap, params, fname, activeSteps);
			*/
		}


		if(run_bubble_solver)
		{
			// where to start the time????
			tSegmentEnd = ( segment * params.maxStepPerSegment + activeSteps - 1 ) * params.dt;
			int it = 0; // step counter
			int eit = 0; // error iteration counter
			State y_next(num_eq, 0.0);
			State yh(num_eq, 0.0);
			std::vector<State> k(6, State(num_eq, 0.0));

			while(sol.t.back() + 1*h < tSegmentEnd*params.f && it < itLim)
			{
					
				bool error_not_in_range = true;
				double max_err = 0.0;
				double current_accepted_h = h;
				eit = 0;
				while (error_not_in_range && eit < itLim) {
					// printf("segment: %d\t it: %d\t eit: %d\n",segment,it,eit);
					// loadp(pArr, uSnap, sol.y.back(), sol.t.back(), params, segment);
					// interp(pArr, sol.y.back(), sol.t.back(), &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params);
					interpSep(sol.y.back(), sol.t.back(), &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params);

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
						// k[j] = dSdt(Y,pb,dpdx,dpdy,dpdt,dpdtdx,up,pp,sp,params); // Evaluate function
						double tj = sol.t.back() + c[j] * h;
						k[j] = dSdt(tj,Y,up,pp,sp,params,uSnap,segment);
					}
					// Calculate 4th order estimate (y_next)
					y_next = sol.y.back();
					for (int j = 0; j < o4stages; j++) {
						if (b[j] != 0.0) {
							for (int dim = 0; dim < num_eq; dim++) {
								y_next[dim] += h * b[j] * k[j][dim];
							}
						}
					}

					// Calculate 5th order estimate (yh)
					yh = sol.y.back();
					for (int j = 0; j < o5stages; j++) {
						if (bh[j] != 0.0) {
							for (int dim = 0; dim < num_eq; dim++) {
								yh[dim] += h * bh[j] * k[j][dim];
							}
						}
					}

					// Examine error
					max_err = 0.0;
					double sum_err = 0.0;
					for (int dim = 0; dim < num_eq; dim++) {
						double e = std::abs(yh[dim] - y_next[dim]);
						if (e > max_err) max_err = e;
						sum_err += e;
					}

					// if (sum_err == 0.0f) sum_err = 1e-15f;

					// Manipulate step size accordingly
					if (max_err > abstolmax) {
						if(h < h_min){
							printf("WARNING: absolute error tolerance of %f was not reached\n",abstol);
							current_accepted_h = h;
							error_not_in_range = false;
						}
						else{
							// Step solution rejected
							error_not_in_range = true;
							float h_tmp = std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h;
							h = (h_tmp < h_min) ? h_min : h_tmp;
						}

						// h = std::max(1e-8f, (float)std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h);
					}
					else if (max_err < abstolmin) {
						// Step solution accepted, but computation is slow (error too small)
						if(sum_err == 0.0){ sum_err = 1e-15; }
						error_not_in_range = false;
						current_accepted_h = h;
						double h_tmp = std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h;

						// h = std::min(h_tmp, h_max);
						if (h_tmp <= h_max) {
							h = h_tmp; // Prepare step size for NEXT iteration
						}
						else { h = h_max; }
					}
					else {
						// Step solution accepted, optimal range
						error_not_in_range = false;
						current_accepted_h = h;
					}
					eit += 1;
					// if(eit%100 == 0 ){ printf("error iteration step: %d\n",eit); }
					if(eit == itLim){ printf("error iteration limit hit\n"); }
				} // End of one step attempt (rejected => repeated)
				// Store accepted values
				// if(y_next[1] != y_next[1]){ printf("yayay\n"); }

				// limit the position, but the dynamics are not realistic if this happens
				
				if(y_next[1]<0.0){y_next[1]=0.0;printf("Warning: Bubble left the simulation domain!\n");}
				else if(y_next[1]>params.lInWavelenght){y_next[1]=params.lInWavelenght*0.99999999999;printf("Warning: Bubble left the simulation domain!\n");}
				if(y_next[2]<0.0){y_next[2]=0.0;printf("Warning: Bubble left the simulation domain!\n");}
				else if(y_next[2]>params.lInWavelenght){y_next[2]=params.lInWavelenght*0.99999999999;printf("Warning: Bubble left the simulation domain!\n");}

				sol.y.push_back(y_next);
				sol.Fb.push_back( -pp[7] * std::pow(sol.y.back()[0],3) * dpdx );
				sol.error.push_back(max_err);
				sol.H.push_back(current_accepted_h);
				
				// Advance time using the 'h' that was accepted for THIS step
				sol.t.push_back(sol.t.back() + current_accepted_h); 


				// ode45();
				it+=1;
				// if(it%100 == 0 ){ printf("timestep: %d\n",it); }
				if(it == itLim){printf("iteration limit hit\n");}
			} // end of RK45 timesteps

			size_t offset = (activeSteps - 2)*size; 	
			for(int i=0; i<2*size; i++){uSegmentEnd[i] = uSnap[offset + i];}

			// printf("size: %lu\n",sol.t.size());
			/*
			char fname[100];
			snprintf(fname, sizeof(fname), "R_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
			solSave(fname, sol.get_state_series(0), params);

			snprintf(fname, sizeof(fname), "x_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
			solSave(fname, sol.get_state_series(1), params);

			snprintf(fname, sizeof(fname), "y_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
			solSave(fname, sol.get_state_series(2), params);

			snprintf(fname, sizeof(fname), "t_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
			solSave(fname, sol.t, params);

			snprintf(fname, sizeof(fname), "H_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
			solSave(fname, sol.H, params);

			snprintf(fname, sizeof(fname), "e_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
			solSave(fname, sol.error, params);
			*/


		}

	// Error:
		cudaFree(dev_2_u); CUERR;
		
		cudaFree(dev_snap); CUERR;
		
		delete[] uSnap;
	} // end of segment iterations

	endTot = std::chrono::steady_clock::now();
	print_runtimes(beginTot, endTot, params);
	
	char fname[100];
	snprintf(fname, sizeof(fname), "R_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(0), params);

	snprintf(fname, sizeof(fname), "x_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(1), params);

	snprintf(fname, sizeof(fname), "y_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(2), params);

	snprintf(fname, sizeof(fname), "t_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.t, params);

	snprintf(fname, sizeof(fname), "H_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.H, params);

	snprintf(fname, sizeof(fname), "e_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.error, params);



	/*
	char fname[100];
	snprintf(fname, sizeof(fname), "R_%.0f_df_%.0f.txt",params.f/1000.0, params.df,0);
	// snprintf(fname, sizeof(fname), "R_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	printf("size: %lu\n",sol.t.size());
	solSave(fname, sol.get_state_series(1), params);
	snprintf(fname, sizeof(fname), "x_%.0f_df_%.0f.txt",params.f/1000.0, params.df,0);
	// snprintf(fname, sizeof(fname), "x_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	solSave(fname, sol.get_state_series(3), params);
	snprintf(fname, sizeof(fname), "y_%.0f_df_%.0f.txt",params.f/1000.0, params.df,0);
	// snprintf(fname, sizeof(fname), "y_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	solSave(fname, sol.get_state_series(5), params);
	snprintf(fname, sizeof(fname), "t_%.0f_df_%.0f.txt",params.f/1000.0, params.df,0);
	// snprintf(fname, sizeof(fname), "t_%.2f_df_%.2f_s_%d.txt",params.f, params.df,0);
	solSave(fname, sol.t, params);
	*/

   
	delete[] u1;
	delete[] u2;

    return cudaStatus;
}
