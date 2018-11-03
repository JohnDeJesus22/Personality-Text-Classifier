# Personality Type CNN with Tensorflow and Keras

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import tensorflow as tf
from nltk import TweetTokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, Model

# predefining parameters
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10


# change directory
os.chdir('D:\\glovePretrainedFlies\\glove.twitter.27B')
glove_file = 'glove.twitter.27B.50d.txt'

# load pretrained twitter glove vectors into dictionary
print('Loading pretrained glove file')
word2vec={}
with open(os.path.join(glove_file),encoding = 'utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype = 'float32')
        word2vec[word]= vec
print(f'Found {len(word2vec)} word vectors')
        
        
# load dataset
print('Loading tweet data')
os.chdir('D:\\')
data = pd.read_csv('mbt1_modified.csv',encoding = 'latin1')
tweets = data['individualPosts'].values