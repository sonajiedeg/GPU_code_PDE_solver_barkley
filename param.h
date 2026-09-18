#pragma once

struct SimParamStruct{
	int res;
	int ITERATIONS;
	int SNAPSHOT;			// 1 => save snapshots
	int SNAP_MODULUS;		// take snapshots if k % SNAP_MODULUS == 0
	int tsc; // target shot count
	int all;
	int segment;
	int snap_grad;
	float f;
	float c;   // c ~= 1500 m/s, then take its square
	float c2; 
	float dx; //  m divide
// CFL criterion with Courant number C_CFL = smalld by 256
// 	float dt;
	float sigma;
	float dt;

// 	float K; // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
	float K; // ( dt * c^2 ) / ( 6 * dx^2 ) this comes from
					  // the E.E. time discretization
					  // and from the 9 point stencil of the Laplace operator
	float bck;
	float rho;
	float d;
	float omega;
	float dx_rho_d_omega2;
	float omega_dt;
	float sin_coeff;
	float Z;
	float kappa;

	float Pa;
	float bcC1;
	float bcC2;
	float bcC3;
	float som1;
	float som2;
	float C_amp;
	float C_off;
	float s;
	float df;
	int maxStepPerSegment;
	int segments;
	int remainder;
};
