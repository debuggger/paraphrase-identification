import sys
import numpy as np

def process_glove_file(filename):
    wv = {}
    lines = open(filename).readlines()
    for line in lines:
        tok = line.split()
        wv[tok[0]] = np.array(map(float, tok[1:]))

    return wv
