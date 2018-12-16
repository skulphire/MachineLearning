import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def init_process(fin,fout):
    outfile = open(fout,'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                elif initial_polarity == '4':
                    initial_polarity = [0,1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()

#files are on server
init_process('training.csv','train_set.csv')
init_process('testing.csv','test_set.csv')

def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open('lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f)

create_lexicon('train_set.csv')

def create_test_data_pickle(fin):

    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)

create_test_data_pickle('processed-test-set.csv')

