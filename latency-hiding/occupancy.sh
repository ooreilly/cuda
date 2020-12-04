kernel=$1
rm -f data/${kernel}.txt

echo "Block Size | Shared Memory (B) | Theoretical Occupancy (warps / SM) | Cache lines / SM | Bandwidth (GB/s) " >  data/${kernel}.txt
echo "Profiling: ${kernel}"
block_size=1
for block_size in 1 2;
do
occ=0
for mem in $(cat data/shared_memory_size.txt)
do
        echo "Shared memory: ${mem} B"
        ./readonly 1e9 ${mem} ${block_size} > tmp.txt
        var=`grep occupancy tmp.txt | awk '{print $6;}'`
        cache_lines=`grep occupancy tmp.txt | awk '{print $10;}'`
        bandwidth=`grep occupancy tmp.txt | awk '{print $12;}'`

        let threads=${block_size}*32
        echo "${threads} ${mem} ${occ} ${cache_lines} ${bandwidth} ">> data/${kernel}.txt
done
done
