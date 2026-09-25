#include <iostream>
#include <vector>

using State = std::vector<float>;

void loadp(float*pArr, float* pSnap, const State& st, float tau0, SimParamStruct params, int segment){
	PASS
	float x0 = st[1] * (params.c/params.f);
	float y0 = st[2] * (params.c/params.f);
	float t0 = tau0 / params.f;
	int i0, j0, k0;
	i0 = floor( x0 / (params.lInWavelenght * params.c / params.f) * params.res );
	j0 = floor( y0 / (params.lInWavelenght * params.c / params.f) * params.res );
	k0 = floor( t0 / params.dt - segment * params.maxStepPerSegment  );
	/*
	printf("t0: %f\n",t0);
	printf("t0 / dt: %f\n",t0 / params.dt);
	printf("i0: %d\n",i0);
	printf("j0: %d\n",j0);
	printf("k0: %d\n",k0);
	*/
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				printf("%d\t%d\t%d\t\n",i,j,k);
				printf("%d\t%d\t%d\t\n",i0,j0,k0);
				// printf("pArr %d\n",i+2*j+4*k);
				// unsigned long long int idx = (i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res;
				std::size_t idx = (i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res;
				// printf("pSnap %d\n",(i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res);
				// printf("pSnap %llu\n",(i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res);
				std::cout<< "pSnap" << idx << "\n";
				// pArr[i+2*j+4*k] = pSnap[(i0+i)+(j0+j)*params.res + (k0+k)*params.res*params.res];
				// while(1){ printf("%d\t%f\n",idx,pSnap[idx]); idx++; }
				PASS
				pArr[i+2*j+4*k] = pSnap[idx];
			}
		}
	}
}
