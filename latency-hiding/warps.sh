kernel=$1
n=139264
let n4=n*4
for warps_per_block in 2 4 8 16;
do
        ./readonly ${n} 0 ${warps_per_block}
        mv data/${kernel}.bin data/${kernel}_${warps_per_block}.bin
        python3 plot_warps.py data/${kernel}_${warps_per_block}.bin figures/warps_${kernel}_${warps_per_block}.svg
done
