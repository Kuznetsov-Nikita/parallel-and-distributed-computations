#!/usr/bin/env bash

COUNT_OUTPUT_DIRECTORY="count_result"
REDUCERS_COUNT=8

hdfs dfs -rm -r -skipTrash $COUNT_OUTPUT_DIRECTORY* > /dev/null

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.reduces=${REDUCERS_COUNT} \
    -files count_mapper.py,count_reducer.py \
    -mapper count_mapper.py \
    -reducer count_reducer.py \
    -input /data/wiki/en_articles \
    -output $COUNT_OUTPUT_DIRECTORY > /dev/null
    
SORT_OUTPUT_DIRECTORY="sort_result"
REDUCERS_COUNT=1

hdfs dfs -rm -r -skipTrash $SORT_OUTPUT_DIRECTORY* > /dev/null

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.reduces=${REDUCERS_COUNT} \
    -D map.output.key.field.separator=\t \
    -D mapreduce.partition.keycomparator.options=-k1,1nr \
    -D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
    -files sort_mapper.py,sort_reducer.py \
    -mapper sort_mapper.py \
    -reducer sort_reducer.py \
    -input $COUNT_OUTPUT_DIRECTORY \
    -output $SORT_OUTPUT_DIRECTORY > /dev/null


hdfs dfs -cat ${SORT_OUTPUT_DIRECTORY}/part-00000 2> /dev/null | head -10

