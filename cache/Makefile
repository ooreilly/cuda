arch=sm_75
flags=-lnvidia-ml

compile:
	nvcc -arch=$(arch) $(flags) cache.cu -o cache
run:
	./cache > cpi.txt

clean:
	rm -f cache
