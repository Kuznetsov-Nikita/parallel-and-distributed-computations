#!/usr/bin/env python

import re
import sys

def is_proper_name(word):
    if len(word) > 9 or len(word) < 6:
        return False
    return word.isalpha() and word[0].isupper() and word[1:].islower()    

for line in sys.stdin:
    try:
        id, text = line.strip().split('\t', 1)
    except ValueError as e:
        continue
        
    words = re.sub('\W', ' ', text).split()
    
    for word in words:
        if is_proper_name(word):
            print("{}\t{}".format(word.lower(), 1))
        else:
            print("{}\t{}".format(word.lower(), 0))

