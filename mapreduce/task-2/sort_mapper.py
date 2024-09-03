#!/usr/bin/env python

import sys

for line in sys.stdin:
    try:
        word, count = line.strip().split('\t')
    except ValueError as e:
        continue
    
    print("{}\t{}".format(count, word))

