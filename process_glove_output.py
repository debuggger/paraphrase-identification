import pickle
import msgpack
import sys


vocabFile = sys.argv[1]
vectorFile = sys.argv[2]
outputFile = sys.argv[3]

vocab2Id = msgpack.unpack(open(vocabFile))
vectors = pickle.load(open(vectorFile))

vocab = {}

id2Word = dict([(id, word) for word, (id, _) in vocab2Id.items()])

for i, vec in enumerate(vectors):
    vocab[id2Word[i]] = vec

pickle.dump(vocab, open(outputFile, 'w'))
