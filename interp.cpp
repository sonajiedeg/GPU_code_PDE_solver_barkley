#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#define _USE_MATH_DEFINES

using State = std::vector<double>;

// void interp(float* pArr, const State& st, float tau,float* p,float* dpdx,float* dpdy,float* dpdt,float* dpdtdx, SimParamStruct params){
void interp(double* pArr, const State& st, double tau, double* p, double* dpdx, double* dpdy, double* dpdt, double* dpdtdx,SimParamStruct params, int segment){
	float x = st[1] * (params.c/params.f);
	float y = st[2] * (params.c/params.f);
	float t = tau / params.f;
	int i0, j0, k0;
	i0 = floor( x / (params.lInWavelenght * params.c / params.f) * params.res );
	j0 = floor( y / (params.lInWavelenght * params.c / params.f) * params.res );
	k0 = floor( t / params.dt - segment * params.maxStepPerSegment + 2);
	float x0,x1,y0,y1,t0,t1;
	x0 = i0 * params.dx; x1 = (i0+1) * params.dx;
	y0 = j0 * params.dx; y1 = (j0+1) * params.dx; t0 = (segment * params.maxStepPerSegment + k0 - 2) * params.dt;
	t1 = (k0+1) * params.dt;
	float xd,yd,td;
	xd = ( x - x0 ) / ( x1 - x0 );
	yd = ( y - y0 ) / ( y1 - y0 );
	td = ( t - t0 ) / ( t1 - t0 );
	float p00,p01,p10,p11,p0,p1;
	p00 = pArr[0+2*0+4*0] * (1 - xd) + pArr[1+2*0+4*0] * xd;
	p01 = pArr[0+2*0+4*1] * (1 - xd) + pArr[1+2*0+4*1] * xd;
	p10 = pArr[0+2*1+4*0] * (1 - xd) + pArr[1+2*1+4*0] * xd;
	p11 = pArr[0+2*1+4*1] * (1 - xd) + pArr[1+2*1+4*1] * xd;
	p0 = p00 * (1 - yd) + p10 * yd;
	p1 = p01 * (1 - yd) + p11 * yd;
	*p = p0 * (1 - td) + p1 * td;
	float dpdx00,dpdx01,dpdx10,dpdx11,dpdx0,dpdx1;
	dpdx00 = ( pArr[1+2*0+4*0] - pArr[0+2*0+4*0] ) / params.dx;
	dpdx01 = ( pArr[1+2*0+4*1] - pArr[0+2*0+4*1] ) / params.dx;
	dpdx10 = ( pArr[1+2*1+4*0] - pArr[0+2*1+4*0] ) / params.dx;
	dpdx11 = ( pArr[1+2*1+4*1] - pArr[0+2*1+4*1] ) / params.dx;
	dpdx0 = dpdx00 * (1 - yd) + dpdx10 * yd;
	dpdx1 = dpdx01 * (1 - yd) + dpdx11 * yd;
	*dpdx = dpdx0 * (1 - td) + dpdx1 * td;
	float dpdy00,dpdy01,dpdy10,dpdy11,dpdy0,dpdy1;
	dpdy00 = ( pArr[0+2*1+4*0] - pArr[0+2*0+4*0] ) / params.dx;
	dpdy01 = ( pArr[0+2*1+4*1] - pArr[0+2*0+4*1] ) / params.dx;
	dpdy10 = ( pArr[1+2*1+4*0] - pArr[1+2*0+4*0] ) / params.dx;
	dpdy11 = ( pArr[1+2*1+4*1] - pArr[1+2*0+4*1] ) / params.dx;
	dpdy0 = dpdy00 * (1 - xd) + dpdy10 * xd;
	dpdy1 = dpdy01 * (1 - xd) + dpdy11 * xd;
	*dpdy = dpdy0 * (1 - td) + dpdy1 * td;
	float dpdt00,dpdt01,dpdt10,dpdt11,dpdt0,dpdt1;
	dpdt00 = ( pArr[0+2*0+4*1] - pArr[0+2*0+4*0] ) / params.dt;
	dpdt01 = ( pArr[0+2*1+4*1] - pArr[0+2*1+4*0] ) / params.dt;
	dpdt10 = ( pArr[1+2*0+4*0] - pArr[1+2*0+4*0] ) / params.dt;
	dpdt11 = ( pArr[1+2*1+4*1] - pArr[1+2*1+4*0] ) / params.dt;
	dpdt0 = dpdt00 * (1 - xd) + dpdt10 * xd;
	dpdt1 = dpdt01 * (1 - xd) + dpdt11 * xd;
	*dpdt = dpdt0 * (1 - yd) + dpdt1 * yd;
	float dpdtdx00,dpdtdx01,dpdtdx10,dpdtdx11,dpdtdx0,dpdtdx1;
	dpdtdx00 = ( pArr[0+2*0+4*1] - pArr[0+2*0+4*0] ) / params.dt;
	dpdtdx01 = ( pArr[0+2*1+4*1] - pArr[0+2*1+4*0] ) / params.dt;
	dpdtdx10 = ( pArr[1+2*0+4*0] - pArr[1+2*0+4*0] ) / params.dt;
	dpdtdx11 = ( pArr[1+2*1+4*1] - pArr[1+2*1+4*0] ) / params.dt;
	dpdtdx0 =  ( dpdtdx10 - dpdtdx00 ) / params.dx;
	dpdtdx1 =  ( dpdtdx11 - dpdtdx01 ) / params.dx;
	*dpdtdx = dpdtdx0 * (1 - yd) + dpdtdx1 * yd;

}


void interpSep(const State& st, double tau,double* p,double* dpdx,double* dpdy,double* dpdt,double* dpdtdx, SimParamStruct params){
	double xi = st[1];
	*p = params.Pa * std::cos(2*M_PI*xi) * std::sin(2*M_PI*tau);
	*dpdx = - params.Pa * 2 * M_PI / (params.c / params.f)* std::sin(2*M_PI*xi) * std::sin(2*M_PI*tau);
	*dpdt = params.Pa * params.omega * std::cos(2*M_PI*xi) * std::cos(2*M_PI*tau);
	*dpdtdx = - params.Pa * params.omega * params.omega / params.c * std::sin(2*M_PI*xi) * std::cos(2*M_PI*tau);
	*dpdy = 0.0;
}
