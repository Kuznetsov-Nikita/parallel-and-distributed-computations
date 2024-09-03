#!/usr/bin/env python

import sys

current_word = None
current_count_no_proper = 0
current_count_proper = 0

for line in sys.stdin:
    try:
        word, count = line.strip().split('\t')
    except ValueError as e:
        continue
    
    if current_word != word:
        if current_word and current_count_no_proper == 0:
            print("{}\t{}".format(current_word, current_count_proper))
            
        current_word = word
        if int(count) == 0:
            current_count_proper = 0
            current_count_no_proper = 1
        else:
            current_count_proper = 1
            current_count_no_proper = 0
    else:
        if int(count) == 0:
            current_count_no_proper += 1
        else:
            current_count_proper += 1

if current_word and current_count_no_proper == 0:
    print("{}\t{}".format(current_word, current_count_proper))

