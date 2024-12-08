#!/usr/bin/env python3
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("uno", type=int)
parser.add_argument("dos", type=int)

args = parser.parse_args()

# Add parsed values
result = args.uno + args.dos

with open('outputtest.txt', 'w') as f:
    f.write(str(result))
    