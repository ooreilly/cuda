arch=sm_75
sm=68
metrics=gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_active.avg,sm__warps_active.avg.per_cycle_active,dram__bytes.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
kernel=readonly_float4
filename=prof.txt
smem=0
n=1e7
warps_per_block=2
cmd=./readonly_profile $(n) $(smem) $(warps_per_block)
flags=-Xptxas -v -g

all: init compile

init:
	mkdir -p logs data figures

compile:
	nvcc -arch=$(arch) $(flags) -DPROFILE=0 -DWRITE_DATA=0 -DSTRIDED_ACCESS=0 readonly.cu -o readonly

compile-profile:
	nvcc -arch=$(arch) $(flags) -DPROFILE=0 -DWRITE_DATA=0 -DSTRIDED_ACCESS=0 readonly.cu -o readonly_profile

profile:
	ncu --log-file logs/$(filename) -k $(kernel) --metrics $(metrics) $(cmd)
	@grep warps logs/$(filename)
	@grep peak logs/$(filename)
	@grep dram logs/$(filename)
	@grep sectors logs/$(filename)

clean:
	rm ./readonly
