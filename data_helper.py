import pickle
import numpy as np

def getNextBatch(textFile, vocab, batchsize = 500):
    with open(textFile) as f:
        lines = f.readlines()
        maxlen = max([len(sent.split()) for line in lines for sent in line.split('\t')[-2:]])
        for i in range(0, len(lines), batchsize):
            x1 = []
            x2 = []
            y = []
            lines[i: i+1]
            for line in lines[i: min(i+batchsize, len(lines))]:
                try:
                    label, _, _, sent1, sent2 = line.strip().split('\t')
                    sent1 += ' <PAD>'*(maxlen-len(sent1.split()))
                    sent2 += ' <PAD>'*(maxlen-len(sent2.split()))
                    x1.append(map(lambda word: vocab[word], sent1.split()))
                    x2.append(map(lambda word: vocab[word], sent2.split()))
                    cl = np.array([0, 0])
                    cl[int(label)] = 1
                    y.append(cl)
                    #print sent1, sent2, y[-1]
                except:
                    continue
            yield np.array(x1), np.array(x2), np.array(y)
            
