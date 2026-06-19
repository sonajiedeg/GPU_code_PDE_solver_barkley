#pragma once

struct SimParamStruct{
	int ITERATIONS;
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
	float rho;
	float d;
	float omega;
	float dt_rho_d_omega2;
	float omega_dt;
	float sin_coeff;
	float Z;
};
