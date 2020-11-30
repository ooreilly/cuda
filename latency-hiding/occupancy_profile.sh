kernel=$1
warps_per_block=$2
rm -f data/${kernel}_${warps_per_block}.txt
touch data/${kernel}_${warps_per_block}.txt
echo "Profiling: ${kernel}"
for mem in `seq 65536 -2048 0`;
do
        echo ${mem}
        make profile smem=${mem} kernel=${kernel} filename=prof_${warps_per_block}.txt warps_per_block=${warps_per_block}
        grep warp logs/prof_${warps_per_block}.txt | awk '{printf $3 ;}' >> data/${kernel}_${warps_per_block}.txt  
        printf " " >> data/${kernel}_${warps_per_block}.txt
        grep dram logs/prof_${warps_per_block}.txt | awk '{print $3;}' >> data/${kernel}_${warps_per_block}.txt  
done
