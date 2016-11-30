import numpy as np
import pickle
from scipy.spatial import distance
import heapq

def parse(filename):
    vocab = {}
    with open(filename) as f:
        line = f.readline()
        while len(line) > 0:
            tokens = line.split()
            vocab[tokens[0]] = np.array([float(num) for num in tokens[1:]], dtype=np.float32)
            line = f.readline()

    return vocab

def getNearWords(srcWord, vocab, numClosest=10, vicinity=0, threshold=1.0):
    srcWordVec = vocab[srcWord]
    heap = []
    for destWord, destWordVec in vocab.iteritems():
        dist = distance.euclidean(srcWordVec, destWordVec)
        if abs(dist-vicinity) < threshold:
            heapq.heappush(heap, (dist, destWord))

    res = []
    for i in range(min(numClosest, len(heap))):
        res.append(heapq.heappop(heap)[1])

    return res
