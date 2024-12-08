#!/usr/bin/env python3
uno = 1
dos = 2

result = uno + dos

with open('outputtest.txt', 'w') as f:
    f.write(result)
    