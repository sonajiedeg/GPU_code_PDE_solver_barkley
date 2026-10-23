#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdio>

using namespace std;

template <typename T>
void DiffNormComp(T* pd, T* pdn){
	T p,ph,pl,phn,pln;
	// T pd,pdn;
	T div = 1.01f;
	p = 1.0e5f;
	ph = p;
	pl = p/div;
	phn = 1.0f;
	pln = 1.0f/div;
	*pd = ph - pl;
	*pdn = p*(phn - pln);
	printf("not norm.: %8.8f\n",*pd);
	printf("    norm.: %8.8f\n",*pdn);
	printf("diff:      %8.8f\n", (T)(*pd - *pdn));
	printf("rel diff:  %8.8f\n", (T)((*pd - *pdn)/(p)));
}


int main()
{
	printf("compare normalized and not n. differences!\n");
	float fpd, fpdn;
	printf("float\n");
	DiffNormComp<float>(&fpd, &fpdn);	
	double dpd, dpdn;
	printf("double\n");
	DiffNormComp<double>(&dpd, &dpdn);	


	return 0;
}


