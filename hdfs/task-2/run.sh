#!/usr/bin/env bash

FILE_NAME=$1

HOST=`curl -i "http://mipt-master.atp-fivt.org:50070/webhdfs/v1$FILE_NAME?op=OPEN" 2> /dev/null | grep -oP '(?<=Location: ).*(?=.)'`
curl -i "$HOST&length=10" 2> /dev/null | tail -n +8

