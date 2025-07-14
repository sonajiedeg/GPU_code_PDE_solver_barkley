#include <string>


cudaError_t init_n(float* u, float* v, float** dev_2_u, float** dev_ua, float** dev_ub, float** dev_v, unsigned int size, int device_id)
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
    cudaStatus = cudaMalloc((void**)dev_2_u, 2 * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed ---init function---!\n");
        return cudaStatus;  // Return error without using goto
    }

    // splitting the allcated memory for u
    *dev_ua = *dev_2_u;
    *dev_ub = *dev_2_u + size;

    cudaStatus = cudaMalloc((void**)dev_v, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed ---init function---!\n");
        return cudaStatus;  // Return error without using goto
    }

    cudaStatus = cudaMemcpy(*dev_ua, u, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(*dev_ub, u, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(*dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
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


    cout << "test passed?" << "\n";
    delete[] u_comp_test;
    return true;
}