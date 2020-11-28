kernel=$1
sm=68 # Number of SMs
filename=${kernel}_bytes_in_flight.txt
rm -f data/${filename}
touch data/${filename}
echo "Profiling: ${kernel}"
for n in `seq 1024 4096 2097152`;
do
        let nsm=$n*${sm}
        make profile n=${nsm} filename=prof.txt
        printf "${n} " >> data/${filename}
        grep dram logs/prof.txt | awk '{print $3;}' >> data/${filename}
done
