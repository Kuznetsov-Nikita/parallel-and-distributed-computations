#!/usr/bin/env bash

FILE_NAME=$1

hdfs fsck -blocks $FILE_NAME 2> /dev/null | grep 'Total blocks' | awk '{ print $4 }'

