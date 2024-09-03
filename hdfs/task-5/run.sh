#!/usr/bin/env bash

FILE_SIZE=$1

dd if=/dev/urandom of=file.txt bs=$FILE_SIZE count=1

hdfs dfs -Ddfs.replication=1 -put file.txt
BLOCK_IDS=`hdfs fsck /user/hjudge/file.txt -files -blocks -locations 2> /dev/null | grep -oP 'blk_[0-9]*'`

for BLOCK_ID in $BLOCK_IDS
do
	NODE_NAME=`hdfs fsck -blockId $BLOCK_ID 2> /dev/null | grep -o 'Block replica on datanode/rack: .*/default' | head -1 | awk '{ print $5 }' | sed 's/\/default//'`
	echo $NODE_NAME
	sudo -u hdfsuser ssh hdfsuser@$NODE_NAME "cd /dfs/; find -name $BLOCK_ID" 2> /dev/null
done > tmp.txt

BLOCKS_SIZE=0

while read NODE_NAME
do
	read BLOCK_PATH
	BLOCK_LOCATION=`echo $BLOCK_PATH | awk '{ print "/dfs"substr($0, 2) }'`
	BLOCK_SIZE=`sudo -u hdfsuser ssh hdfsuser@$NODE_NAME "ls -l $BLOCK_LOCATION" 2> /dev/null | awk '{ print $5 }'`
	BLOCKS_SIZE=$(( BLOCKS_SIZE + BLOCK_SIZE ))
done < tmp.txt

RESULT=$(( FILE_SIZE - BLOCKS_SIZE ))
echo $RESULT

rm tmp.txt
hdfs dfs -rm file.txt 2> /dev/null
rm file.txt

