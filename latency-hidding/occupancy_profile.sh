kernel=$1
rm -f data/${kernel}.txt
touch data/${kernel}.txt
echo "Profiling: ${kernel}"
for mem in `seq 65536 -2048 0`;
do
        echo ${mem}
        make profile smem=${mem} kernel=${kernel} filename=prof.txt
        grep warps logs/prof.txt | awk '{printf $3 ;}' >> data/${kernel}.txt  
        printf " " >> data/${kernel}.txt
        grep dram logs/prof.txt | awk '{print $3;}' >> data/${kernel}.txt  
done
