#!/usr/bin/evn bash

{
    hadoop dfs -rmr yjahn/KDtree/output
} || {
    printf "nothing to remove\n"
}

start1=$SECONDS


python partition_s.py

end1=$SECONDS
{
    hadoop dfs -rm yjahn/KDtree/input.txt
} || {
    printf "nothing to remove\n"
}

hadoop dfs -copyFromLocal input.txt yjahn/KDtree

start2=$SECONDS
hadoop jar /opt/cloudera/parcels/CDH-5.11.2-1.cdh5.11.2.p0.4/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-mr1.jar -file mapper.py -mapper mapper.py -file reducer_s.py -reducer reducer_s.py -input yjahn/KDtree/input.txt -output yjahn/KDtree/output

end2=$SECONDS

{
    rm output/*
} || {
    printf "nothing to remonve\n"
}

hadoop dfs -copyToLocal yjahn/KDtree/output

time1=$(($end1 - $start1))
time2=$(($end2 - $start2))
printf "partition time : $time1 \n"
printf "reduce time : $time2 \n"
