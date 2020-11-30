n=139264
let n4=n*4
for warps_per_block in 2 4 8 16;
do
        ./readonly ${n} 0 ${warps_per_block}
        mv data/readonly_baseline.bin data/readonly_baseline_${warps_per_block}.bin
        ./readonly ${n4} 0 ${warps_per_block}
        mv data/readonly_float4.bin data/readonly_float4_${warps_per_block}.bin
        python3 plot_warps.py readonly_baseline_${warps_per_block} figures/warps_readonly_baseline_${warps_per_block}.svg
        python3 plot_warps.py readonly_float4_${warps_per_block} figures/warps_readonly_float4_${warps_per_block}.svg
done
