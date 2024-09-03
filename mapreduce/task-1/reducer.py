#!/usr/bin/env python

import random
import sys

line_ids_count = random.randint(1, 5)
line_ids = list()

for line in sys.stdin:
    try:
        num, id = line.strip().split(" ")
    except ValueError as e:
        continue
        
    line_ids.append(id)
    line_ids_count -= 1
        
    if line_ids_count == 0:
        print(",".join(line_ids))
            
        line_ids_count = random.randint(1, 5)
        line_ids = list()

if len(line_ids) > 0:
    print(",".join(line_ids))

