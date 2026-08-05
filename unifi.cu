#include <string>
#include "param.h"


cudaError_t init(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, float** dev_snap, SimParamStruct params, float** dev_snap_gradx, float** dev_snap_grady);

cudaError_t init(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, float** dev_snap, SimParamStruct params);
	
cudaError_t initR(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, float** dev_snap, SimParamStruct params, float* dev_phi1a, float* dev_phi1b, float* dev_phi2a, float* dev_phi2b);
// init 
cudaError_t init(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, SimParamStruct params);

cudaError_t init(float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int res, float** dev_snap, SimParamStruct params);

cudaError_t init(float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int res, SimParamStruct params);

cudaError_t init(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, float** dev_snap, SimParamStruct params, float** dev_snap_gradx, float** dev_snap_grady)
{
    /*
    cudaFree(*dev_2_u);
    cudaFree(*dev_ua);
    cudaFree(*dev_ub);
    cudaFree(*dev_v);
    */

    // READ INPUTS
    text_to_array(u, v);

    cudaError_t cudaStatus;

    // Allocate GPU buffers (two input, one output).
    cudaStatus = cudaMalloc((void**)dev_2_u, 2 * size * sizeof(float)); CUERR;

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + size;

    cudaStatus = cudaMalloc((void**)dev_v, size * sizeof(float)); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ua, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ub, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;
	// allocate mem for snapshots
    cudaStatus = cudaMalloc((void**)dev_snap, size * sizeof(float) * ( int( params.tsc ) ) ); CUERR;
    cudaStatus = cudaMalloc((void**)dev_snap_gradx, size * sizeof(float) * ( int( params.tsc ) ) ); CUERR;
    cudaStatus = cudaMalloc((void**)dev_snap_grady, size * sizeof(float) * ( int( params.tsc ) ) ); CUERR;

	cudaStatus = cudaGetLastError();
    return cudaStatus;  // Successful allocation
}

cudaError_t init(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, float** dev_snap, SimParamStruct params)
{
    /*
    cudaFree(*dev_2_u);
    cudaFree(*dev_ua);
    cudaFree(*dev_ub);
    cudaFree(*dev_v);
    */

    // READ INPUTS
    text_to_array(u, v);

    cudaError_t cudaStatus;

    // Allocate GPU buffers (two input, one output).
    cudaStatus = cudaMalloc((void**)dev_2_u, 2 * size * sizeof(float)); CUERR;

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + size;

    cudaStatus = cudaMalloc((void**)dev_v, size * sizeof(float)); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ua, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ub, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;
	// allocate mem for snapshots
    cudaStatus = cudaMalloc((void**)dev_snap, size * sizeof(float) * ( int( params.tsc ) ) ); CUERR;

	cudaStatus = cudaGetLastError();
    return cudaStatus;  // Successful allocation
}

cudaError_t initR(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, float** dev_snap, SimParamStruct params, float* dev_phi1a, float* dev_phi1b, float* dev_phi2a, float* dev_phi2b)
{
    /*
    cudaFree(*dev_2_u);
    cudaFree(*dev_ua);
    cudaFree(*dev_ub);
    cudaFree(*dev_v);
    */

    // READ INPUTS
    text_to_array(u, v);

    cudaError_t cudaStatus;

    // Allocate GPU buffers (two input, one output).
    cudaStatus = cudaMalloc((void**)dev_2_u, 2 * size * sizeof(float)); CUERR;

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + size;

    cudaStatus = cudaMalloc((void**)dev_v, size * sizeof(float)); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ua, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ub, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;
	// allocate mem for snapshots
    cudaStatus = cudaMalloc((void**)dev_snap, size * sizeof(float) * ( int( params.tsc ) ) ); CUERR;
	
	float* tmp = 0;
	
	cudaStatus = cudaMalloc((void**)tmp, size * sizeof(float)); CUERR;
	*dev_phi1a = *tmp;
	*dev_phi1b = *tmp + size;
	*dev_phi2a = *tmp + 2 * size;
	*dev_phi2b = *tmp + 3 * size;

	cudaStatus = cudaGetLastError();
    return cudaStatus;  // Successful allocation
}

cudaError_t init(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id, SimParamStruct params)
{
    /*
    cudaFree(*dev_2_u);
    cudaFree(*dev_ua);
    cudaFree(*dev_ub);
    cudaFree(*dev_v);
    */

    // READ INPUTS
    text_to_array(u, v);

    cudaError_t cudaStatus;

    // Allocate GPU buffers (two input, one output).
    cudaStatus = cudaMalloc((void**)dev_2_u, 2 * size * sizeof(float)); CUERR;

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + size;

    cudaStatus = cudaMalloc((void**)dev_v, size * sizeof(float)); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ua, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_ub, u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

    // cudaStatus = cudaMemcpy(*dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

	cudaStatus = cudaGetLastError();
    return cudaStatus;  // Successful allocation
}

bool test(float* u, int size)
{
    float* u_comp_test = new float[size];

    fstream newfile;

    newfile.open("u_comp_test_5000.txt", ios::in); //open a file to perform read operation using file object
    if (newfile.is_open())  //checking whether the file is open
    {
        string tp;
        int i = 0;
        while (getline(newfile, tp))  //read data from file object and put it into string.
        {
            //cout << tp << "\n"; //print the data of the string
            u_comp_test[i++] = stof(tp);
        }
        newfile.close(); //close the file object.
    }
    

    float avg_abs_diff = 0;
    for (int i = 0; i < size; i++)
    {
        if ( u[i] != u_comp_test[i] )
        {
            cout << "i:" << i << "\n"; //difference
            cout << "difference:" << u[i] - u_comp_test[i] << "\n"; //difference
            cout << "u[i]          " << u[i] << "\n";
            cout << "u_comp_test[i]" << u_comp_test[i] << "\n";
            cout << "test failed!" << "\n";

            
            for (int j = 0; j < size; j++)
            {
                if (u[j] != u_comp_test[j])
                {
                    avg_abs_diff += abs(u[j] - u_comp_test[j]) / 65536;
                }
            }

            cout << "average absolut difference: " << avg_abs_diff << "\n";

            return false;
        }
    }


    cout << "test passed?" << "\n";
    delete[] u_comp_test;
    return true;
}

cudaError_t init(float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int res, float** dev_snap, SimParamStruct params)
{
    cudaError_t cudaStatus;

    // Allocate GPU buffers
    cudaMalloc((void**)dev_2_u, 3 * ( res+2 ) * sizeof(float)); CUERR;

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + res;
	*dev_v = *dev_2_u + 2 * res;

	// allocate mem for snapshots
	cudaMalloc((void**)dev_snap, res * sizeof(float) * ( int( params.tsc ) ) ); CUERR;

	// cudaMemcpyToSymbol(d_params, &h_params, sizeof(SimParams));

	cudaStatus = cudaGetLastError();
    return cudaStatus;  // Successful allocation
}

cudaError_t init(float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int res, SimParamStruct params)
{
    cudaError_t cudaStatus;

    // Allocate GPU buffers
    cudaMalloc((void**)dev_2_u, 3 * ( res+2 )  * sizeof(float)); CUERR;

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + res;
	*dev_v = *dev_2_u + 2 * res;

	cudaStatus = cudaGetLastError();
    return cudaStatus;  // Successful allocation
}
