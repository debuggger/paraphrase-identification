import pickle
import numpy as np

def getNextBatch(textFile, vocab, start=0, batchsize = 500, maxlen=-1):
    f = open(textFile)
    lines = f.readlines()
    f.close()
    if batchsize == -1: batchsize = len(lines)

    if maxlen == -1:
        maxlen = getMaxSentLen(textFile)
    
    x1 = []
    x2 = []
    y = []
    for line in lines[start: min(start+batchsize, len(lines))]:
        #print line
        label, _, _, sent1, sent2 = line.strip().split('\t')
        sent1 += ' <PAD>'*(maxlen-len(sent1.split()))
        sent2 += ' <PAD>'*(maxlen-len(sent2.split()))
        x1.append(map(lambda word: vocab[word] if word in vocab else vocab['<PAD>'], sent1.split()[:maxlen]))
        x2.append(map(lambda word: vocab[word] if word in vocab else vocab['<PAD>'], sent1.split()[:maxlen]))
        x1.append(map(lambda word: vocab[word] if word in vocab else vocab['<PAD>'], sent2.split()[:maxlen]))
        x2.append(map(lambda word: vocab[word] if word in vocab else vocab['<PAD>'], sent2.split()[:maxlen]))
        cl = np.array([0, 0])
        cl[int(label)] = 1
        y.append(cl)
        y.append(cl)

    return np.array(x1), np.array(x2), np.array(y)

def getNextBatchLM(textFile, vocab, start=0, batchsize = 500, maxlen=-1):
    f = open(textFile)
    lines = f.readlines()
    f.close()
    if batchsize == -1: batchsize = len(lines)

    if maxlen == -1:
        maxlen = getMaxSentLen(textFile)
    
    x = []
    x1 = []
    x2 = []
    x3 = []
    y = []
    for line in lines[start: min(start+batchsize, len(lines))]:
        #print line
        sent = line.strip().split()
        x.append(map(lambda word: vocab[word] if word in vocab else vocab['<PAD>'] , sent))
        x1.append(vocab[sent[0]] if sent[0] in vocab else vocab['<PAD>'])
        x2.append(vocab[sent[1]] if sent[1] in vocab else vocab['<PAD>'])
        x3.append(vocab[sent[2]] if sent[2] in vocab else vocab['<PAD>'])
        y.append(vocab[sent[3]] if sent[3] in vocab else vocab['<PAD>'])

    return np.array(x), np.array(x1), np.array(x2), np.array(x3), np.array(y)

def getMaxSentLenLM(textFile):
    with open(textFile) as f:
        lines = f.readlines()
        return max([len(line.split()) for line in lines])
        #return max([len(sent.split()) for line in lines for sent in line.split('\t')[-2:]])

def getMaxSentLen(textFile):
    with open(textFile) as f:
        lines = f.readlines()
        return max([len(sent.split()) for line in lines for sent in line.split('\t')[-2:]])
