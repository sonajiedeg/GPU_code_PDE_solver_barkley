#pragma once
#include <iostream>
#include <vector>
#include "paramSetup.cpp"

using State = std::vector<double>;

struct PositionReset {
	bool xLow, yLow, xHigh, yHigh;
};


void loadp(double* pArr, float* pSnap, const State& stIn, double tau0, SimParamStruct params, int segment, int activeSteps){
	// limiting the position within the simulaion domain
	State st = stIn;
	if(st[1]<0.0){st[1]=0.0;}
	else if(st[1]>params.lInWavelenght){st[1]=params.lInWavelenght;}
	if(st[2]<0.0){st[2]=0.0;}
	else if(st[2]>params.lInWavelenght){st[2]=params.lInWavelenght;}

	/*
	if(st[1]<0.0){st[1]=0.0;pr.xLow=1;};
	else if(st[1]>params.lInWavelenght){st[1]=params.lInWavelenght;pr.xHigh=1;};
	if(st[2]<0.0){st[2]=0.0;pr.yLow=1;};
	else if(st[2]>params.lInWavelenght){st[2]=params.lInWavelenght;pr.yHigh=1;};
	*/

	// float x0 = st[1] * (params.c/params.f);
	// float y0 = st[2] * (params.c/params.f);
	double t0 = tau0 / (double)params.f;
	int i0, j0, k0;
	i0 = floor( st[1] / (double)params.lInWavelenght * (double)params.res );
	j0 = floor( st[2] / (double)params.lInWavelenght * (double)params.res );
	k0 = floor( t0 / (double)params.dt - (double)segment * (double)params.maxStepPerSegment + 3.0 );

    if (k0 < 0){k0 = 0;}
    // Assuming maxStepPerSegment defines the maximum valid steps in pSnap buffer
    if (k0 > activeSteps + 1){k0 = activeSteps + 1;}
	
	// printf("t0: %f\n",t0);
	// printf("t0 / dt: %f\n",t0 / params.dt);
	// printf("i0: %d\n",i0);
	// printf("j0: %d\n",j0);
	// printf("k0: %d\n",k0);
	// std::cout<< "pSnap idx0 " << (i0+0)+(j0+0)*params.res + (k0+0)*params.res*params.res << " i0 " << i0 << " k0 " << k0 << "\n";

	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				// printf("%d\t%d\t%d\t\n",i,j,k);
				// if(x0 != x0){ printf("yay\n"); }
				// printf("trace i0:\n%f\n%f\n%f\n%f\n",x0,(params.lInWavelenght * params.c / params.f),( x0 / (params.lInWavelenght * params.c / params.f) * params.res ),st[1] * (params.c/params.f));
				// printf("%d\t%d\t%d\t\n",i0,j0,k0);
				// printf("pArr %d\n",i+2*j+4*k);
				// unsigned long long int idx = (i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res;
				std::size_t idx = (std::size_t)(i0+i)+(std::size_t)(j0+j)*(std::size_t)params.res + (std::size_t)(k0+k)*(std::size_t)params.res*(std::size_t)params.res;
				// printf("pSnap %d\n",(i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res);
				// printf("pSnap %llu\n",(i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res);
				// std::cout<< "pSnap idx" << idx << "\n";
				// std::cout<< "pArr idx" << i+2*j+4*k<< "\n";
				// pArr[i+2*j+4*k] = pSnap[(i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res];
				// while(1){ printf("%d\t%f\n",idx,pSnap[idx]); idx++; }
				// PASS
				pArr[i+2*j+4*k] = (double)pSnap[idx];
			}
		}
	}
}

void loadpSep(float*pArr, float* pSnap, const State& st, float tau0, SimParamStruct params, int segment){
	// PASS
	float x0 = st[1] * (params.c/params.f);
	float y0 = st[2] * (params.c/params.f);
	float t0 = tau0 / params.f;
	int i0, j0, k0;
	i0 = floor( x0 / (params.lInWavelenght * params.c / params.f) * params.res );
	j0 = floor( y0 / (params.lInWavelenght * params.c / params.f) * params.res );
	k0 = floor( t0 / params.dt - segment * params.maxStepPerSegment  );
	
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				if(x0 != x0){ printf("yay\n"); }
				std::size_t idx = (i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res;

				pArr[i+2*j+4*k] = pSnap[idx];
			}
		}
	}
}
