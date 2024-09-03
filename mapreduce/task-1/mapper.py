#!/usr/bin/env python

import random
import sys

max_num = 2 ** 64

for line in sys.stdin:
    try:
        id = line.strip()
        num = random.randint(0, max_num)
        print("{} {}".format(num, id))
    except ValueError as e:
        continue

