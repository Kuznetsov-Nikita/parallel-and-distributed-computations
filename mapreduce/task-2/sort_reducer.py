#!/usr/bin/env python

import sys

for line in sys.stdin:
    try:
        count, word = line.strip().split('\t')
    except ValueError as e:
        continue
    
    print("{}\t{}".format(word, count))

