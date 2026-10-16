#pragma once
#include "param.h"
#include <cstdio>

SimParamStruct paramSetup(float df = 50.0f){
	
	// float res = 256;
	// int res = 256;
	SimParamStruct params;
	params.lInWavelenght = 1.0f/1.0f;
	params.res = 256;
	params.f = 25e3;
	// params.c = 1482.0f;   // c ~= 1500 m/s, then take its square
	// pmod
	params.c = 1500.0f;
	float pi = 3.14159265359;

	float lambda = params.c / params.f;
	float dAx = 1623.7864077669904e-6; // maximum measured oscillation amplitude of antinode
									   //
	printf("%f\t%f\t%f\n",pi,dAx,1.0f/lambda);
	params.PhaseOffsetAmplitude = 2.0f * pi * dAx / lambda;
	params.PhaseOffsetAngularVelocity = 2.0f * pi * df;
	params.df = df;
	params.c2 = params.c * params.c; 
	params.dx = params.c / params.f * params.lInWavelenght / (1.0f * params.res); //  m divide
	// params.dx = 1e-3 / 256; // for the validation case
	// CFL criterion with Courant number C_CFL = smalld by 256
	// params.dt = 5.859374999999999e-9 / 4;
	params.sigma = 0.8; // CFL
						// rounding to, to, to, .... do something... idk, i want to sleep
	float dt_tmp = params.sigma * params.dx / params.c;
	float k_tmp = 1 / params.f / dt_tmp;
	int k = int(k_tmp); // i think this truncates, rounds to floor, good, cfl wont be violated
	params.dt = 1 / params.f / (1.0f * k);

	// for 9 point stencil
    params.K = ( params.dt * params.dt * params.c2 ) / ( 6 * params.dx * params.dx ); // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
	// for 5 point stencil
	// params.K = params.dt*params.dt * params.c*params.c / params.dx / params.dx; // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
	// params.K = (params.dt * params.dt * params.c2) / (params.dx * params.dx);
						  // the E.E. time discretization
						  // and from the 9 point stencil of the Laplace operator
	// params.bck = 1/2/params.K;					  //
	params.bck = 1/2.0f/(params.dt*params.dt * params.c*params.c / params.dx / params.dx);
	// params.rho = 998.2;
	// pmod
	params.rho = 998.0f;
	params.d = 1e-6;
	params.omega = 2 * 3.14159265359 * params.f;
	params.dx_rho_d_omega2 = params.dx * params.rho * params.d * params.omega * params.omega;
	params.omega_dt = params.omega * params.dt;
	params.sin_coeff = params.c / params.f * params.lInWavelenght / (1.0f * params.res) * params.rho * params.d * params.omega * params.omega;
	params.Z = 1.0f;
	params.kappa = ( 1.0f - params.sigma ) / ( 1.0f + params.sigma );

    // ua[0] = ub[0] + dt * Z * c / dx * ( ub[1] - ub[0] )
	// sommerfeld params
	params.som1 = 1.0f / ( 1.0f + params.dt * params.c * params.Z / params.dx );
	params.som2 = params.dt * params.c * params.Z / params.dx / ( 1.0f + params.dt * params.c * params.Z / params.dx );
	
	params.Pa = 75000.0f; // assigned the forced oscillation amplitide at the transducers
	// params.Pa = 285250.0f; 
	params.bcC1 = 1 / ( 1.0f + params.dt * params.c / params.dx );
	params.bcC2 = params.Pa * params.dt * params.bcC1 * params.omega;
	params.bcC3 = params.dt * params.c / params.dx * params.bcC1;

	params.C_amp = 4.0f*4.0f; // it is a unit multiplied by pi: if ==1 => will result in pi
	float K = 1 / params.f / params.dt; // T / dt, time period in unit of timesteps
	params.C_off = 70.0f * K; // time offset, will turn on the phase offet after this, has unit of [number of oscillations * number of timesteps]
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
	// float t = params.lInWavelenght / params.f * 12.0f*80.0f; // i want to set it so it just reaches the other wall
	// float t = 0.004;
	float t = 0.04;
	// t = 2.5e-6; // for the validation case
	// params.ITERATIONS = round( t / params.dt );
    params.ITERATIONS = round( t / params.dt / 2.0f ) * 2.0f;
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
		params.maxStepPerSegment = 4*1024*2;
		params.remainder = params.ITERATIONS % params.maxStepPerSegment;
		params.segments = ( params.ITERATIONS - params.remainder ) / params.maxStepPerSegment + 1;
		if (params.remainder == 0){ params.segments -= 1; params.remainder = params.maxStepPerSegment; }
	}
	else if(params.all){
		params.snap_grad = 1;
		params.SNAPSHOT = 1;
		params.SNAP_MODULUS = 1;
		params.tsc = params.ITERATIONS;
	}
	/*
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
	printf("t_end / (L / lam) * f: %f (-)\n",t * params.f / params.lInWavelenght);
	printf("L / lam: %f (-)\n",params.lInWavelenght);
	printf("L: %f (m)\n",params.c / params.f * params.lInWavelenght);
	printf("max step per segment (-): %d\n", params.maxStepPerSegment);
	printf("remainder (-): %d\n", params.remainder);
	printf("segments (-): %d\n", params.segments);
	*/

	printf("CFL actual: %f (-)\n",params.dt * params.c / params.dx);
	printf("t_end: %f (s)\n",t);
	printf("t_end / (L / lam) * f: %f (-)\n",t * params.f / params.lInWavelenght);
	printf("L / lam: %f (-)\n",params.lInWavelenght);
	printf("L: %f (m)\n",params.c / params.f * params.lInWavelenght);
	printf("wavelenght: %f (m)\n",params.c / params.f);


	printf("res: %d\n",params.res);
	printf("ITERATIONS: %d\n",params.ITERATIONS);
	printf("SNAPSHOT: %d\n",params.SNAPSHOT);	
	printf("SNAP_MODULUS: %d\n",params.SNAP_MODULUS);	
	printf("tsc: %d\n",params.tsc); 
	printf("all: %d\n",params.all);
	printf("segment: %d\n",params.segment);
	printf("snap_grad: %d\n",params.snap_grad);
	printf("sigma: %.9g\n",params.sigma);
	printf("f: %.9g\n",params.f);
	printf("c: %.9g\n",params.c);   
	printf("c2: %.9g\n",params.c2); 
	printf("dx: %.9g\n",params.dx); 
	printf("sigma: %.9g\n",params.sigma);
	printf("dt: %.9g\n",params.dt);
	printf("K: %.9g\n",params.K); 
	printf("bck: %.9g\n",params.bck);
	printf("rho: %.9g\n",params.rho);
	printf("d: %.9g\n",params.d);
	printf("omega: %.9g\n",params.omega);
	printf("dx_rho_d_omega2: %.9g\n",params.dx_rho_d_omega2);
	printf("omega_dt: %.9g\n",params.omega_dt);
	printf("sin_coeff: %.9g\n",params.sin_coeff);
	printf("Z: %.9g\n",params.Z);
	printf("kappa: %.9g\n",params.kappa);
	printf("Pa: %.9g\n",params.Pa);
	printf("bcC1: %.9g\n",params.bcC1);
	printf("bcC2: %.9g\n",params.bcC2);
	printf("bcC3: %.9g\n",params.bcC3);
	printf("som1: %.9g\n",params.som1);
	printf("som2: %.9g\n",params.som2);
	printf("C_amp: %.9g\n",params.C_amp);
	printf("C_off: %.9g\n",params.C_off);
	printf("s: %.9g\n",params.s);
	printf("df: %.9g\n",params.df);
	printf("maxStepPerSegment: %d\n",params.maxStepPerSegment);
	printf("segments: %d\n",params.segments);
	printf("remainder: %d\n",params.remainder);
	printf("PhaseOffsetAmplitude: %f\n",params.PhaseOffsetAmplitude);
	printf("PhaseOffsetAngularVelocity: %f\n",params.PhaseOffsetAngularVelocity);




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
		fprintf(file, "t_end / (L / lam) * f (-)\t\t %.9g\n", t * params.f / params.lInWavelenght);
		fprintf(file, "L / lam (-)\t\t %.9g\n", params.lInWavelenght);
		fprintf(file, "L (m)\t\t %.9g\n", params.c / params.f * params.lInWavelenght);
		fprintf(file, "max step per segment ()\t\t%d \n", params.maxStepPerSegment);
		fprintf(file, "remainder ()\t\t%d \n", params.remainder);
		fprintf(file, "segments ()\t\t%d \n", params.segments);
		fclose(file);
	}


	return params;
}

void paramSetupBubbleSolver(double* up, double* pp, double* sp, SimParamStruct params){

	double ST = 0.0725; // N/m
	double R0 = 80e-6; // m
	// double P0 = 101325; // Pa
	// pmod
	double P0 = 1.0e5; // Pa
	double PV = 0.0;
	double pi = 3.14159265359;
	// double Rref = params.c / params.f;
	double Rref = R0;
	double omega = 2*pi*params.f; // Hz
	// double RHO = 998.2; // kg/m^3
	// pmod
	double RHO = 998.0; // kg/m^3
	double PE = 1.4; // <<<----------------------?????????????????????????
	double CL = params.c; // m/s
	double VIS = 0.001; // Pa s
	// double  = Z;


	double PG0 = (2.0 * ST / R0 + P0 - PV);

	up[0] = PG0 * std::pow(2.0 * pi / Rref / omega,2.0) / RHO;
	up[1] = (1.0 - 3.0 * PE) * PG0 * (2.0 * pi / Rref / omega) / CL / RHO;
	up[2] = R0 / Rref;

	//  Population-specific constants
	// Keller--Miksis Equation
	pp[0] = (P0 - PV) * std::pow(2.0 *pi / Rref / omega,2.0) / RHO;                 // C0
	pp[1] = (2.0 * ST / Rref / RHO) * std::pow(2.0 * pi / Rref / omega,2.0);        // C1
	pp[2] = 4.0 * VIS / RHO / (Rref*Rref) * (2.0* pi / omega);               // C2
	pp[3] = (std::pow(2.0 * pi / Rref / omega,2.0)) / RHO;                          // C3
	pp[4] = (std::pow(2.0 * pi / omega,2.0)) / CL / RHO / Rref;                    // C4
	pp[5] = Rref * omega / (2 * pi) / CL;                                    // C5
	// Translational Motion
	// pp[6] = 2.0 * pi / std::pow(Rref*Rref * omega,2.0) / RHO;                           // C6
	// mod 
	pp[6] = 2.0 * pi / std::pow(Rref * omega,2.0) / Rref / (params.c / params.f) / RHO;                           // C6
	pp[7] = 4.0 * pi / 3 * std::pow(Rref,3.0);                                          // C7
	pp[8] = 12.0 * pi * VIS * Rref;                                          // C8
	pp[9] = Rref * omega / (2.0 * pi);                                       // C9
	pp[10] = 36.0 * pi / omega / (Rref*Rref) * VIS / RHO;                         // C10 (csak ha van coupling)
	pp[11] = Rref;


	// System-specific constants
	sp[0] = 3.0 * PE;
	sp[1] = 1.0 / omega;
	sp[2] = omega / CL;
	sp[3] = CL;
}
