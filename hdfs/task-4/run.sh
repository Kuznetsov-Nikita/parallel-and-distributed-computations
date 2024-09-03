#!/usr/bin/env bash

BLOCK_ID=$1

NODE_NAME=`hdfs fsck -blockId $BLOCK_ID 2> /dev/null | grep -o 'Block replica on datanode/rack: .*/default' | head -1 | awk '{ print $5 }' | sed 's/\/default//'`
PATH=`sudo -u hdfsuser ssh hdfsuser@$NODE_NAME "locate $BLOCK_ID && exit" 2> /dev/null | head -1`

echo $NODE_NAME:$PATH

