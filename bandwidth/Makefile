arch=sm_75
sm=68
metrics=gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
filename=prof.txt
n=1e6
cmd=./bandwidth $(n)
flags=-Xptxas -v

all: init compile profile

init:
	mkdir -p logs

compile:
	nvcc -arch=$(arch) $(flags) bandwidth.cu -o bandwidth

profile:
	ncu --log-file logs/$(filename) --metrics $(metrics) $(cmd)
	@grep $(metrics) logs/$(filename)

clean:
	rm ./bandwidth
