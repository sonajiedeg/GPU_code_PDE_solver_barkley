#include <string>

// allocate for n kernels
// cudaError_t init_n(float* u, float* v, float*** dev_ua, float*** dev_ub, float*** dev_v, unsigned int size, unsigned int n, int device_id) 
cudaError_t init_n(float* u, float* v, float** dev_ua[nKernel], float** dev_ub[nKernel], float** dev_v[nKernel], unsigned int size, unsigned int n, int device_id) 
{
    float* dev_alloc_tmp;
    // READ INPUTS
    text_to_array(u, v);

    cudaError_t cudaStatus;
    // Allocate GPU memory
    cudaStatus = cudaMalloc((void**)(&dev_alloc_tmp), 3 * n * size * sizeof(float)); CUERR;
    // splitting the allcated memory for ua, ub and v for each n
    for (int i = 0; i < n; i++)
    {
        PASS;
        (*dev_ua)[i] = &( dev_alloc_tmp[ i * 3 * size ] );
        (*dev_ub)[i] = &( dev_alloc_tmp[ i * 3 * size + size ] );
        (*dev_v)[i] = &( dev_alloc_tmp[ i * 3 * size + 2 * size ] );
        // cudaMalloc((void**)(dev_ua[i]), size * sizeof(float)); CUERR;
        // cudaMalloc((void**)(dev_ub[i]), size * sizeof(float)); CUERR;
        // cudaMalloc((void**)(dev_v[i]), size * sizeof(float)); CUERR;
    }
    PASS;
    for(int i = 0; i < n; i++)
    {
        cudaStatus = cudaMemcpy((*dev_ua)[i], u, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;

        cudaStatus = cudaMemcpy((*dev_v)[i], v, size * sizeof(float), cudaMemcpyHostToDevice); CUERR;
    }
    return cudaSuccess;  // Successful allocation
}

bool test_n(float* u, int size)
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
	for (int i = 0; i < nKernel; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if( u[ i*size + j ] != u_comp_test[j] )
			{
				cout << "kernel number: " << i << ", difference at: " << j << "\n";
				cout << "difference: " << u[ i*size + j ] - u_comp_test[j] << "\n";
				cotu << "u[j]: " << u[ i*size + j ] << ", u_comp[j]: " << u_comp_test[j] << "\n";
				cout << "test failed!" << "\n";

	            for (int k = 0; k < size; k++)
				{
					if (u[k] != u_comp_test[k])
					{
						avg_abs_diff += abs(u[ i*size + k ] - u_comp_test[k]) / 65536;
					}
				}

				cout << "average absolut difference: " << avg_abs_diff << "\n";

				return false;
			}
		}
	}


    cout << "test passed?" << "\n";
    delete[] u_comp_test;
    return true;
}
