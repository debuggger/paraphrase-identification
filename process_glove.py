import numpy as np
import pickle
from scipy.spatial import distance
import heapq
import msgpack

def parse(vocabFile, vecFile):
    word2Id = msgpack.unpack(open(vocabFile))
    id2Word = dict([(id, w) for w,(id, _) in word2Id.items()])
    vectors = pickle.load(open(vecFile))

    return word2Id, id2Word, vectors

def getNearWords(srcWord, vocab, numClosest=10, vicinity=0, threshold=1.0):
    srcWordVec = vector[vocab[srcWord][0]]
    heap = []
    for destWord, destWordVec in vocab.iteritems():
        dist = distance.euclidean(srcWordVec, destWordVec)
        if abs(dist-vicinity) < threshold:
            heapq.heappush(heap, (dist, destWord))

    res = []
    for i in range(min(numClosest, len(heap))):
        res.append(heapq.heappop(heap)[1])

    return res
