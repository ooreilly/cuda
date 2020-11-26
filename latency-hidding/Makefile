arch=sm_75
sm=68
metrics=gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_active.avg,sm__warps_active.avg.per_cycle_active,dram__bytes.sum.per_second
kernel=readonly_float4
filename=prof.txt
smem=1e2
n=1e8
cmd=./readonly $(n) $(smem)
flags=-Xptxas -v 

all: init compile profile

init:
	mkdir -p logs

compile:
	nvcc -arch=$(arch) $(flags) readonly.cu -o readonly

profile:
	ncu --log-file logs/$(filename) -k $(kernel) --metrics $(metrics) $(cmd)
	@grep warps logs/$(filename)
	@grep peak logs/$(filename)
	@grep dram logs/$(filename)

clean:
	rm ./readonly
