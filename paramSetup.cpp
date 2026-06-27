#include "param.h"

SimParamStruct paramSetup(){
	float lInWavelenght = 30;
	int res = 256;
	SimParamStruct params;
	params.f = 20e3;
	params.c = 1500;   // c ~= 1500 m/s, then take its square
	params.c2 = params.c * params.c; 
	params.dx = params.c / params.f * lInWavelenght / res; //  m divide
	// CFL criterion with Courant number C_CFL = smalld by 256
	// params.dt = 5.859374999999999e-9 / 4;
	params.sigma = 0.4; // CFL
	params.dt = params.sigma * params.dx / params.c;

	// for 9 point stencil
	// params.K = ( params.dt * params.dt * params.c2 ) / ( 6 * params.dx * params.dx ); // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
	// for 5 point stencil
	 params.K = params.dt*params.dt * params.c*params.c / params.dx / params.dx; // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
						  // the E.E. time discretization
						  // and from the 9 point stencil of the Laplace operator
	// params.bck = 1/2/params.K;					  //
	params.bck = 1/2/(params.dt*params.dt * params.c*params.c / params.dx / params.dx);
	params.rho = 1e3;
	params.d = 10e-6;
	params.omega = 2 * 3.14159265359 * params.f;
	params.dt_rho_d_omega2 = params.dt * params.rho * params.d * params.omega * params.omega;
	params.omega_dt = params.omega * params.dt;
	params.sin_coeff = params.c / params.f * lInWavelenght / res * params.rho * params.d * params.omega * params.omega;
	params.Z = 1;
	/*
	 * c
	 * L = 3 lambda = 3 c / f
	 * c = L / t
	 * t = L / c = 3 * c / f / c = 3 / f
	 *
	 * t / dt = 3 / f / ( sig * dx / c )
	 *		  = 3 / f / ( sig * c / f * 3 / res / c)
	 *		  = 3 / f / ( sig / f * 3 / res )
	 *		  = 3 / f / ( (3 sig) / (f * res) )
	 *		  = ( 3 f res) / ( f 3 sig )
	 *		  = res / sig
	*/
	float t = lInWavelenght / params.f; // i want to set it so it just reaches the other wall
	params.ITERATIONS = round( t / params.dt / 4 ) * 2;
	// params.ITERATIONS = round( t / params.dt / 2 ) * 2 * 3;
	params.tsc = 100;
	params.SNAPSHOT = 1;
	params.SNAP_MODULUS = floor( params.ITERATIONS / params.tsc );
	// printf("%d",params.ITERATIONS);
	// save all timesteps, save all p and dp_dx and dp_dy
	params.all = 0;
	if(params.all){
		params.snap_grad = 1;
		params.SNAPSHOT = 1;
		params.SNAP_MODULUS = 1;
		params.tsc = params.ITERATIONS;
	}
	return params;
}
