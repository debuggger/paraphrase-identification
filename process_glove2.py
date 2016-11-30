import numpy as np
import pickle
from scipy.spatial import distance
import heapq

def parse(filename):
    return pickle.load(open(filename))

def getNearWords(srcWord, word2VecFile, numClosest=10, vicinity=0, threshold=1.0):
    vocab = parse(word2VecFile)
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
