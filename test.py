import heapq
import numpy as np
from scipy.spatial import distance

def getNear(word, W_prime, vocab, id2word):
	heap = []
	res = []
	for i in range(len(vocab)):
		dist = distance.euclidean(W_prime[vocab[word][0]], W_prime[i])
        	heapq.heappush(heap, (dist, i))
	for i in range(20):
	        res.append(id2word[heapq.heappop(heap)[1]])	
	return res

