ARCH=sm_75
NVCC=nvcc
NVFLAGS=-Xptxas -v

compile:
	$(NVCC) $(NVFLAGS) bandwidth.cu -o bandwidth.x
clean:
	rm -r *.x
