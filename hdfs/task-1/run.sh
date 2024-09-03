#!/usr/bin/env bash

FILE_NAME=$1

BLOCK_ID=`hdfs fsck "$1" -files -blocks -locations 2>/dev/null | grep -o 'blk_[0-9]*'`
NODE_NAME=`hdfs fsck -blockId $BLOCK_ID 2> /dev/null | grep -o 'Block replica on datanode/rack: .*/default' | head -1 | awk '{ print $5 }' | sed 's/\/default//'`

echo $NODE_NAME

