import json
from nltk.corpus import stopwords
import pickle, gensim, random
import numpy as np

f               = open("data/dwords.p",'rb')
dtr             = pickle.load(f, encoding='latin1')

print ("Loading Word2Vec...")
model = gensim.models.KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz",binary=True)
print ("Word2Vec loaded!")

def load_data_from_json():
    with open('data/data.txt', 'r') as f:
        result = [json.loads(x) for x in f.readlines()]
    data = []
    for i in result:
        item = []
        item.append(i['sentence1'])
        item.append(i['sentence2'])
        item.append(i['score'])
        data.append(item)
    return data

def embed(stmx):
    dmtr=np.zeros((stmx.shape[0],300), dtype=np.float32)
    count=0
    while(count<len(stmx)):
        if stmx[count]=='<end>':
            count+=1
            continue
        if stmx[count] in dtr:
            dmtr[count]=model[dtr[stmx[count]]]
            count+=1
        elif stmx[count] in model:
            dmtr[count]=model[stmx[count]]
            count+=1
        else:
            dmtr[count]=dmtr[count]
            count+=1
    return dmtr

def getmtr(xa, maxlen):
    ls=[]
    for i in range(0,len(xa)):
        q=xa[i].split()
        while(len(q)<maxlen):
            q.append('<end>')
        else:
            q = q[:maxlen]
        ls.append(q)      
    xa=np.array(ls)
    return xa

def prepare_data(data, maxlen):
    xa1 = [ data[i][0] for i in range(0,len(data)) ]
    xb1 = [ data[i][1] for i in range(0,len(data)) ]
    y   = [ data[i][2] for i in range(0,len(data)) ]

    lengths1, lengths2 =[],[]
    for i in xa1:
        if len(i.split()) > maxlen:
            lengths1.append(maxlen)
        else:
            lengths1.append(len(i.split()))
    for i in xb1:
        if len(i.split()) > maxlen:
            lengths2.append(maxlen)
        else:
            lengths2.append(len(i.split()))


    words1 = getmtr(xa1, maxlen)
    emb1   = [ embed(words) for words in words1]
    words2 = getmtr(xb1, maxlen)
    emb2   = [ embed(words) for words in words2]

    y = np.array(y, dtype=np.float32)

    return [ emb1, lengths1, emb2, lengths2, y ]