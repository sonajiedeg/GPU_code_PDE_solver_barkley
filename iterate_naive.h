__global__ void iterateKernelExpNaive(float* g_ua, float* g_ub, float* g_v, int res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float u_th;

    // ----- constants -----
    const float repsilon = 1 / 0.02;
    const float ra = 1 / 0.64;
    const float b = 0.05;
    const float dt = 0.0076893502846360206604003906250000000000;

    if (i < res && j < res)
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c mûveletet 1 órajel alatt megcsinálja)
        // ----- Variables -----
        float r_ua;
        float r_v;
        float r_un;
        float r_vn;
        float r_L;
        // const float dx = 0.195312500000000;
        // const float dt = 0.003051800000000;
        // float ccc = dt / 6 / dx / dx;
        //const float ccc = 0.0133335171;
        // const float ccc = 0.013333517126739025115966796875000000000000000000000000000000;

        // ----- Load Data -----
        //copy_g_to_r(g_ua, g_v, &r_ua, &r_v, ThreadOffset);
        r_ua = g_ua[ThreadOffset];
        r_v = g_v[ThreadOffset];
        __syncthreads();
        // ----- Iteration step -----
        //iterate_without_laplace(&r_ua, &r_v, &r_un, &r_vn);
        u_th = (r_v + b) * ra;
        r_vn = r_v + dt * (r_ua - r_v);
        r_un = r_ua + dt * repsilon * r_ua * (1 - r_ua) * (r_ua - u_th);
        __syncthreads();
        // ----- Laplace -----
        //laplace(g_ua, &r_un, res, i, j, ThreadOffset);
        if (i < res - 1 && j < res - 1 && i > 0 && j > 0)
        {
            r_L = 0;
            // Nem nagy a stencil, kilehet bontani teljesen:
            float* g_ua_act = g_ua + ThreadOffset; // Létrehozunk egy új pointert ami az aktuális thread megfelelõ elemére mutat, 1 INT ADD
            r_L += -20 * g_ua_act[0];
            r_L += 4 * g_ua_act[+1];
            r_L += 4 * g_ua_act[-1];

            g_ua_act = g_ua_act - res; // Eltoljuk a pointert 1 sorral feljebb, 1 INT ADD
            r_L += 4 * g_ua_act[0];
            r_L += 1 * g_ua_act[+1];
            r_L += 1 * g_ua_act[-1];

            g_ua_act = g_ua_act + 2 * res; // Eltoljuk a pointert 2 sorral lejjebb, 1 INT FMA
            r_L += 4 * g_ua_act[0];
            r_L += 1 * g_ua_act[+1];
            r_L += 1 * g_ua_act[-1];

            r_L = ccc * r_L;

            r_un += r_L;
        }
        __syncthreads();
        // ----- "Unload" Data -----
        //copy_r_to_g(g_ub, g_v, r_un, r_vn, ThreadOffset);
        g_ub[ThreadOffset] = r_un;
        g_v[ThreadOffset] = r_vn;
        __syncthreads();
    }
}

__global__ void no_flux_kernel(float* g_ub, int res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < res && j < res)
    {
        int ThreadOffset = i + j * res; // 1 INT FMA (FMA = Fused Multiple Add, az a+b*c mûveletet 1 órajel alatt megcsinálja)

        // ----- No Flux Boundary Condition -----
        no_flux(g_ub, res, i, j, ThreadOffset);
    }
}