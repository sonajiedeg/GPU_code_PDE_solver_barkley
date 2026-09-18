#include "param.h"
#include <cstdio>

SimParamStruct paramSetup(float df = 0.0f){
	
	float lInWavelenght = 1.0/2.0;
	// float res = 256;
	// int res = 256;
	SimParamStruct params;
	params.res = 256;
	params.f = 44e3;
	params.df = df;
	params.c = 1482;   // c ~= 1500 m/s, then take its square
	params.c2 = params.c * params.c; 
	params.dx = params.c / params.f * lInWavelenght / (1.0f * params.res); //  m divide
	// params.dx = 1e-3 / 256; // for the validation case
	// CFL criterion with Courant number C_CFL = smalld by 256
	// params.dt = 5.859374999999999e-9 / 4;
	params.sigma = 0.8; // CFL
						// rounding to, to, to, .... do something... idk, i want to sleep
	float dt_tmp = params.sigma * params.dx / params.c;
	float k_tmp = 1 / params.f / dt_tmp;
	int k = int(k_tmp); // i think this truncates, rounds to floor, good, cfl wont be violated
	params.dt = 1 / params.f / k;

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
	params.sin_coeff = params.c / params.f * lInWavelenght / (1.0f * params.res) * params.rho * params.d * params.omega * params.omega;
	params.Z = 1;
	params.kappa = ( 1 - params.sigma ) / ( 1 + params.sigma );

    // ua[0] = ub[0] + dt * Z * c / dx * ( ub[1] - ub[0] )
	// sommerfeld params
	params.som1 = 1 / ( 1 + params.dt * params.c * params.Z / params.dx );
	params.som2 = params.dt * params.c * params.Z / params.dx / ( 1 + params.dt * params.c * params.Z / params.dx );
	
	params.Pa = 2 * 50000; // assigned the forced oscillation amplitide at the transducers
	params.bcC1 = 1 / ( 1 + params.dt * params.c / params.dx );
	params.bcC2 = params.dt * params.bcC1 * params.omega;
	params.bcC3 = params.dt * params.c / params.dx * params.bcC1;

	params.C_amp = 4*4; // it is a unit multiplied by pi: if ==1 => will result in pi
	float K = 1 / params.f / params.dt; // T / dt, time period in unit of timesteps
	params.C_off = 70 * K; // time offset, will turn on the phase offet after this, has unit of [number of oscillations * number of timesteps]
	params.s = 0.001;
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
	float t = lInWavelenght / params.f * 12; // i want to set it so it just reaches the other wall
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
	params.all = 0;
	params.segment = 1;
	if(params.segment){
		params.snap_grad = 1;
		params.SNAPSHOT = 1;
		params.SNAP_MODULUS = 1;
		params.tsc = params.ITERATIONS;
		params.maxStepPerSegment = 2;
		params.remainder = params.ITERATIONS % params.maxStepPerSegment;
		params.segments = ( params.ITERATIONS - params.remainder ) / params.maxStepPerSegment + 1;
	}
	else if(params.all){
		params.snap_grad = 1;
		params.SNAPSHOT = 1;
		params.SNAP_MODULUS = 1;
		params.tsc = params.ITERATIONS;
	}
	printf("f: %f (Hz)\n",params.f);
	printf("c: %f (m/s)\n",params.c);
	printf("rho: %f (kg/m^3)\n",params.rho);
	printf("CFL target: %f (-)\n",params.sigma);
	printf("CFL actual: %f (-)\n",params.dt * params.c / params.dx);
	printf("dx: %f (m)\n",params.dx);
	printf("dx: %.9g (m)\n",params.dx);
	printf("dt: %.9g (s)\n",params.dt);
	printf("t_end: %f (s)\n",t);
	// float tmp = 1.40625014e-06;
	// printf("1.40625014e-06 == dt: %d (-)\n", tmp == params.dt);
	printf("t_end / (L / lam) * f: %f (-)\n",t * params.f / lInWavelenght);
	printf("L / lam: %f (-)\n",lInWavelenght);
	printf("L: %f (m)\n",params.c / params.f * lInWavelenght);
	printf("max step per segment (-): %d\n", params.maxStepPerSegment);
	printf("remainder (-): %d\n", params.remainder);
	printf("segments (-): %d\n", params.segments);

	// write params to the file
	char fname[100];
	snprintf(fname, sizeof(fname), "params_f_%.2f_df_%.2f.txt",params.f, df);
	FILE* file = fopen(fname, "w");
	// fwrite(uSnap, sizeof(float), res * res * int( params.tsc ) , file);
	if(file){
		fprintf(file, "f (Hz)\t\t %.9g\n", params.f);
		fprintf(file, "df (Hz)\t\t %.9g\n", params.df);
		fprintf(file, "c (m/s)\t\t %.9g\n", params.c);
		fprintf(file, "rho (kg/m^3)\t\t %.9g\n", params.rho);
		fprintf(file, "CFL target (-)\t\t %.9g\n", params.sigma);
		fprintf(file, "CFL actual (-)\t\t %.9g\n", params.dt * params.c / params.dx);
		fprintf(file, "dx (m)\t\t %.9g\n", params.dx);
		fprintf(file, "dx (m)\t\t %.9g\n", params.dx);
		fprintf(file, "dt (s)\t\t %.9g\n", params.dt);
		fprintf(file, "t_end (s)\t\t %.9g\n", t);
		fprintf(file, "t_end / (L / lam) * f (-)\t\t %.9g\n", t * params.f / lInWavelenght);
		fprintf(file, "L / lam (-)\t\t %.9g\n", lInWavelenght);
		fprintf(file, "L (m)\t\t %.9g\n", params.c / params.f * lInWavelenght);
		fprintf(file, "max step per segment ()\t\t%d \n", params.maxStepPerSegment);
		fprintf(file, "remainder ()\t\t%d \n", params.remainder);
		fprintf(file, "segments ()\t\t%d \n", params.segments);
		fclose(file);
	}


	return params;
}
