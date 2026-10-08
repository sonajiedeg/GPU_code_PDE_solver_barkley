#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>

//-------------------------------------------------------
//----------DEFINE RUN PARAMETERS------------------------
//-------------------------------------------------------


#define run_gsync_restruct_int_friend 1
#define run_bubble_solver 1
#define DEVICE_ID 0
// #define ITERATIONS 1000000

#define CUERR {																					\
cudaError_t err;																				\
    if ((err = cudaGetLastError()) != cudaSuccess) {											\
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
        exit(1);																				\
    }																							\
}

#define PASS printf(" PASSED FILE: %s, LINE: %d\n", __FILE__, __LINE__);

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdio>
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
#include "copy_gridsync.cpp"
#include "iterate_gridsync_restructure_int_friend.cu"
#include "unifi.cu"
#include "param.h"
#include "paramSetup.cpp"
#include "loadp.cpp"
#include "interp.cpp"
#include "solSave.cpp"

using namespace std;
using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;




using State = std::vector<double>;
using OdeFunc = std::function<State(const double, const State&, const double [3], const double [12], const double [4], const SimParamStruct)>;

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

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params);

void loadp(double*pArr, double* pSnap, const State& st, double t0, SimParamStruct params, int segment);

void interp(double* pArr, const State& st, double t,double* p,double* dpdx,double* dpdy,double* dpdt,double* dpdtdx,SimParamStruct params);

void solSave(char* fname, std::vector<double> sol, SimParamStruct params);

void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params);

int main()
{
	unsigned int size = 65536;
	int res = 256;
	double* u = new double[ size ];
	double* u_sol = new double[ size ];
	SimParamStruct params = paramSetup(200.0);
	
	double* u2 = new double[ size ];
	double* u1 = new double[ size ];
	for(int i = 0; i<size; i++){ u1[i]=u[i]; u2[i]=u[i]; }
	int iteration_limit_per_2 = 0;
	int activeSteps;
	unsigned long long int itLim = 100000;
	double abstol = 1e-9;
	double t0,t1,tSegmentEnd,x0,x1,v0,v1,y0,y1,w0,w1,p0,pb,dpdx,dpdy,dpdt,dpdtdx,tmp;
	double pArr[8];
	int stateVarCount = 6; // R V x v y u
	double* S0 = new double[stateVarCount]; 
	double* S1 = new double[stateVarCount]; 
	for(int i=0;i<stateVarCount;i++){ S0[i]=0; S1[i]=0; }
	p0 = 101325.0;
	pb = p0;
	t0 = 0.0;
	x0 = params.lInWavelenght * params.c / params.f / 2.0;	
	y0 = params.lInWavelenght * params.c / params.f / 2.0;	
	v0 = 0.0;
	w0 = 0.0;
	double up[3] = {0.0};
	double pp[12] = {0.0};
	double sp[4] = {0.0};
	paramSetupBubbleSolver(up, pp, sp, params);

	// -------------------------------------------------
	OdeFunc dSdt = [](const double tau, const State& Y, const double up[3], const double pp[12], const double sp[4], const SimParamStruct params) -> State {
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


		interpSep(Y, tau, &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params);

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
	// const State ic = { 1.0, x0/R0*1.25, y0/R0, 0, 0, 0 }; // R x y U vx vy
	const State ic = { 1.0, 0.009, y0/R0, 0.0, 0.0, 0.0 }; // R x y U vx vy
    int num_eq = ic.size();
    
    Solution sol;
    sol.t.push_back(t_start);
    sol.y.push_back(ic);
	sol.Fb.push_back(0.0);

    // Set parameters for error control
	double h_init = 1e-6;
	double h_max = params.dt * params.f / 2.0;
	double h_min = 1e-12;
    double h = h_init;
    double abstolmin = abstol * 0.1;
    double abstolmax = abstol * 10.0;
    double frac = 0.9;
    double p = 4.0;
	int kStepStart = 0;

	//measure elapsed time
	std::chrono::steady_clock::time_point beginTot;
	std::chrono::steady_clock::time_point endTot;
	beginTot = std::chrono::steady_clock::now();

	for(int segment = 0; segment < params.segments; segment++ )
	{
		printf("segment / segments: %d / %d\n",segment,params.segments);
		activeSteps = ( segment < params.segments-1 ) ? params.maxStepPerSegment : params.remainder;
		size_t snapSize = (size_t)size * (size_t)activeSteps ;
		// double* uSnap = new double[ snapSize];


		//measure at this iteration
		iteration_limit_per_2 += activeSteps / 2;
		kStepStart = segment * params.maxStepPerSegment / 2;

		if(run_bubble_solver)
		{
			// where to start the time????
			tSegmentEnd = ( segment * params.maxStepPerSegment + activeSteps ) * params.dt;
			int it = 0; // step counter
			int eit = 0; // error iteration counter
			State y_next(num_eq, 0.0);
			State yh(num_eq, 0.0);
			std::vector<State> k(6, State(num_eq, 0.0));

			while(sol.t.back() < tSegmentEnd*params.f && it < itLim)
			{
				bool error_not_in_range = true;
				double max_err = 0.0;
				double current_accepted_h = h;
				eit = 0;
				while (error_not_in_range && eit < itLim) {
					// loadpSep(pArr, uSnap, sol.y.back(), sol.t.back(), params, segment);
					interpSep(sol.y.back(), sol.t.back(), &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params);
					// pb+=0.0;
					// pb+=p0;


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
						double tj = sol.t.back() + c[j] * h;
						k[j] = dSdt(tj,Y,up,pp,sp,params); // Evaluate function
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
						if(h <= h_min){
							printf("WARNING: absolute error tolerance of %f was not reached\n",abstol);
							current_accepted_h = h;
							error_not_in_range = false;
						}
						else{
							// Step solution rejected
							error_not_in_range = true;
							double h_tmp = std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h;
							h = (h_tmp < h_min) ? h_min : h_tmp;
						}

						// h = std::max(1e-8f, (double)std::pow((frac * abstol) / sum_err, 1.0 / (p + 1.0)) * h);
					}
					// else if (max_err < abstolmin && sum_err != 0.0) {
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
					// if(eit == itLim){ printf("error iteration limit hit\n"); }
				} // End of one step attempt (rejected => repeated)
				// Store accepted values
				// if(y_next[1] != y_next[1]){ printf("yayay\n"); }
				sol.y.push_back(y_next);
				sol.Fb.push_back( -pp[7] * std::pow(sol.y.back()[0],3) * dpdx );
				sol.error.push_back(max_err);
				sol.H.push_back(current_accepted_h);
				
				// Advance time using the 'h' that was accepted for THIS step
				sol.t.push_back(sol.t.back() + current_accepted_h); 


				// ode45();
				it+=1;
				// if(it%100 == 0 ){ printf("timestep: %d\n",it); }
				// if(it == itLim){printf("iteration limit hit\n");}
			} // end of RK45 timesteps

			// printf("size: %lu\n",sol.t.size());
			
		}
		
		// delete[] uSnap;
	} // end of segment iterations

	endTot = std::chrono::steady_clock::now();
	print_runtimes(beginTot, endTot, params);


	char fname[100];
	snprintf(fname, sizeof(fname), "R_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(0), params);

	snprintf(fname, sizeof(fname), "x_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(1), params);

	snprintf(fname, sizeof(fname), "vx_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(4), params);

	snprintf(fname, sizeof(fname), "y_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.get_state_series(2), params);

	snprintf(fname, sizeof(fname), "t_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.t, params);

	snprintf(fname, sizeof(fname), "H_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.H, params);

	snprintf(fname, sizeof(fname), "e_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.error, params);

	snprintf(fname, sizeof(fname), "Fb_%.0f_df_%.0f.txt",params.f/1000.0, params.df);
	solSave(fname, sol.Fb, params);


	PASS

	
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
	delete[] S0;
	delete[] S1;

    return 0;
}


void print_runtimes(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, SimParamStruct params)
{
    //measure at this iteration
    int iteration_limit_per_2 = params.ITERATIONS / 2.0;
	if(iteration_limit_per_2 == 0){iteration_limit_per_2 = 1;}

    int average_iteration_runtime;
    int total_runtime;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel loop) : " << average_iteration_runtime << "[us]" << endl;
    average_iteration_runtime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 2 / iteration_limit_per_2;
    cout << "average iteration runtime (count one of 2 iteration per kernel loop) : " << average_iteration_runtime << "[ns]" << endl;

    total_runtime = std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();
	cout << "total runtime : " << total_runtime << "[s]" << endl; 
	total_runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
	cout << "total runtime : " << total_runtime << "[ms]" << endl; 
	total_runtime = std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count();
	cout << "total runtime : " << total_runtime << "[us]" << endl; 
	
	cout << "number of timesteps : " << params.ITERATIONS << endl;

}
