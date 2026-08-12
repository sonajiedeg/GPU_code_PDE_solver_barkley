#include "param.h"



SimParamStruct paramSetup(){
	float lInWavelenght = 2;
	int res = 128;
	SimParamStruct params;
	params.f = 20e3;
	params.c = 1482;   // c ~= 1500 m/s, then take its square
	params.c2 = params.c * params.c; 
	params.dx = params.c / params.f * lInWavelenght / res; //  m divide
	// params.dx = 1e-3 / 256; // for the validation case
	// CFL criterion with Courant number C_CFL = smalld by 256
	// params.dt = 5.859374999999999e-9 / 4;
	params.sigma = 0.6; // CFL
	params.dt = params.sigma * params.dx / params.c;

	// for 9 point stencil
	params.K = ( params.dt * params.dt * params.c2 ) / ( 6 * params.dx * params.dx ); // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
	// for 5 point stencil
	// params.K = params.dt*params.dt * params.c*params.c / params.dx / params.dx; // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
						  // the E.E. time discretization
						  // and from the 9 point stencil of the Laplace operator
	// params.bck = 1/2/params.K;					  //
	params.bck = 1/2/(params.dt*params.dt * params.c*params.c / params.dx / params.dx);
	params.rho = 998.2;
	params.d = 1e-6;
	params.omega = 2 * 3.14159265359 * params.f;
	params.dx_rho_d_omega2 = params.dx * params.rho * params.d * params.omega * params.omega;
	params.omega_dt = params.omega * params.dt;
	params.sin_coeff = params.c / params.f * lInWavelenght / res * params.rho * params.d * params.omega * params.omega;
	params.Z = 1;
	params.kappa = ( 1 - params.sigma ) / ( 1 + params.sigma );

    // ua[0] = ub[0] + dt * Z * c / dx * ( ub[1] - ub[0] )
	// sommerfeld params
	params.som1 = 1 / ( 1 + params.dt * params.c / params.dx );
	params.som2 = params.dt * params.c / params.dx / ( 1 + params.dt * params.c / params.dx );
	
	params.Pa = 2 * 50000; // assigned the forced oscillation amplitide at the transducers
	params.bcC1 = 1 / ( 1 + params.dt * params.c / params.dx );
	params.bcC2 = params.dt * params.bcC1 * params.omega;
	params.bcC3 = params.dt * params.c / params.dx * params.bcC1;
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
	float t = lInWavelenght / params.f * 8; // i want to set it so it just reaches the other wall
	// t = 2.5e-6; // for the validation case
	// params.ITERATIONS = round( t / params.dt );
    params.ITERATIONS = round( t / params.dt / 2 ) * 2;
    // params.ITERATIONS = 4;
	// params.ITERATIONS = round( t / params.dt / 2 ) * 2 * 3;
	params.tsc = 100;
	params.SNAPSHOT = 1;
	params.SNAP_MODULUS = floor( params.ITERATIONS / params.tsc );
	// printf("%d",params.ITERATIONS);
	// save all timesteps, save all p and dp_dx and dp_dy
	params.all = 1;
	if(params.all){
		params.snap_grad = 1;
		params.SNAPSHOT = 1;
		params.SNAP_MODULUS = 1;
		params.tsc = params.ITERATIONS;
	}

	return params;
}
