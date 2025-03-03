__global__ void g_to_g_test(float* g_ua, float* g_ub)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int ThreadOffset = i + 32 * 32 * j;
#pragma unroll
	for (int i = 0; i < 50000; i++)
	{
		g_ua[ThreadOffset] = g_ub[ThreadOffset];
		g_ub[ThreadOffset] = g_ua[ThreadOffset];
	}
}
