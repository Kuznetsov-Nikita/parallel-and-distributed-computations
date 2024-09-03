#!/usr/bin/env bash

OUTPUT_DIRECTORY="streaming_wc_result"
REDUCERS_COUNT=8

hdfs dfs -rm -r -skipTrash $OUTPUT_DIRECTORY* > /dev/null

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.reduces=${REDUCERS_COUNT} \
    -files mapper.py,reducer.py \
    -mapper mapper.py \
    -reducer reducer.py \
    -input /data/ids \
    -output $OUTPUT_DIRECTORY > /dev/null
    
hdfs dfs -cat ${OUTPUT_DIRECTORY}/part-00000 2> /dev/null | head -50

