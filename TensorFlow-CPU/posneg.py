import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random, pickle
from collections import Counter
import tensorflow as tf
import os

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexitcon(pos,neg):
    lexicon = []
    for file in [pos,neg]:
        with open (file,'r') as f:
            contents = f.readlines()
            for lne in contents[:hm_lines]:
                allwords = word_tokenize(lne.lower())
                lexicon += list(allwords)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    l2 = []
    for w in w_counts:
        #we dont care about common words, or rare words
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for lne in contents[:hm_lines]:
            currentWords = word_tokenize(lne.lower())
            currentWords = [lemmatizer.lemmatize(i for i in currentWords)]
            features = np.zeros(len(lexicon))
            for word in currentWords:
                if word.lower() in lexicon:
                    indexvalue = lexicon.index(word.lower())
                    features[indexvalue] += 1
            features = list(features)
            featureset.append([features,classification])
    return featureset

def create_featureset_and_lables(pos,neg,testsize=0.1):
    lexicon = create_lexitcon(pos,neg)
    features = []
    features += sample_handling(pos,lexicon,[1,0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testingSize = int(testsize*len(features))
    train_x = list(features[:,0][:-testingSize]) #get all 0 index elements which are the features
    train_y = list(features[:, 1][:-testingSize])

    test_x = list(features[:, 0][-testingSize:])
    test_y = list(features[:, 1][-testingSize:])

    return  train_x,train_y,test_x,test_y

if __name__ == '__main__':
    print(os.getcwd())

    train_x, train_y, test_x, test_y = create_featureset_and_lables('MachineLearning/Datasets/PosNeg_pt5/pos.txt','MachineLearning/Datasets/PosNeg_pt5/neg.txt')
    with open('posneg.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y],f)







