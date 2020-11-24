arch=sm_75
sm=68
metrics=gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_active.avg
filename=prof.txt
n=1e8
cmd=./readonly $(n)
flags=-Xptxas -v 

all: init compile profile

init:
	mkdir -p logs

compile:
	nvcc -arch=$(arch) $(flags) readonly.cu -o readonly

profile:
	ncu --log-file logs/$(filename) --metrics $(metrics) $(cmd)
	@grep peak logs/$(filename)

clean:
	rm ./readonly
