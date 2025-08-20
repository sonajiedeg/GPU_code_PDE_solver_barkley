CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_86 -lineinfo --fmad=true --prec-div=true --prec-sqrt=true 
SOURCE   = kernel.cu
all: kernel.exe
kernel.exe: $(SOURCE)
	nvcc -o	kernel.exe $(SOURCE) $(CMPL_OPT)

clean:
	rm -f kernel.exe
	rm -f u_sol.txt
	
re:
	rm -f kernel.exe
	rm -f u_sol.txt
	make
	./kernel.exe

nTest.exe:
	# nvcc -o	nTest.exe nTest.cu $(CMPL_OPT)
	nvcc --default-stream per-thread ./nTest.cu -o nTest.exe
	# nvcc --default-stream per-thread ./pthread_test.cu -o pthreads_per_thread
