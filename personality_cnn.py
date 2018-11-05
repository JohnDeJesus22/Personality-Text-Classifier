# Personality Type CNN with Tensorflow and Keras

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model

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
        
        
# load dataset and split into tweets and target types
print('Loading tweet data')
os.chdir('D:\\')
data = pd.read_csv('mbt1_modified.csv',encoding = 'latin1')
tweets = data['individualPosts'].values
personality_type = data['IntroExtro'].values
labelencoder = LabelEncoder()
personality_type = labelencoder.fit_transform(personality_type)
#onehotencoder=OneHotEncoder(categorical_features=[1])
#personality_type=onehotencoder.fit_transform([personality_type]).toarray()
#personality_type = personality_type.reshape(395472,-1)

# tokenize and convert tweets to integers
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# get word to integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

# get descriptive statistics on the sequences
print('max sequence len:', max(len(s) for s in sequences))
print('min sequence len:', min(len(s) for s in sequences))
seq_sorted = sorted(len(s) for s in sequences)
print('median sequence length:', seq_sorted[len(seq_sorted)//2])

# pad sequences
padded_seq = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', padded_seq.shape)

# prep embedded matrix
print('prepping embedding matrix')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedded_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word,index in word2idx.items():
    if index < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedded_matrix[index]=embedding_vector
            
# place embeddings into an embedding layer
embedding_layer = layers.Embedding(num_words, EMBEDDING_DIM, weights=[embedded_matrix],
                                   input_length = MAX_SEQUENCE_LENGTH,
                                   trainable = False)

# construct model
print('Initiating model construction')

inputs = layers.Input(shape = (MAX_SEQUENCE_LENGTH,))
x = embedding_layer(inputs)
x = layers.Conv1D(128,3,activation = 'relu')(x)
x = layers.MaxPooling1D(3)(x)
x = layers.Conv1D(128,3,activation = 'relu')(x)
x = layers.MaxPooling1D(3)(x)
x = layers.Conv1D(128,3, activation = 'relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128,activation = 'relu')(x)
output = layers.Dense(1,activation = 'sigmoid')(x)

model = Model(inputs,output)
model.compile(loss = 'binary_crossentropy',optimizer = 'rmsprop',metrics=['accuracy'])

print('Training Model')
classifier = model.fit(padded_seq, personality_type,
                       batch_size = BATCH_SIZE,epochs = EPOCHS,
                       validation_split = VALIDATION_SPLIT)
print('Training Completed')

# save the model
os.chdir('D:\\PacktTensorflowCourseMaterials\\PersonalityCNN')

save_model(model=model,filepath = './personality-model-1',overwrite = True,
                 include_optimizer = True)

# plot loss and validation loss
plt.plot(classifier.history['loss'],label='training loss')
plt.plot(classifier.history['val_loss'],label = 'validation_loss')
plt.legend()
plt.show()
# validation loss increased. Probably due to the imbalance in the classes

# plot accuracy and validation accuracy
plt.plot(classifier.history['acc'], label = 'training accuracy')
plt.plot(classifier.history['val_acc'], label = 'validation accuracy')
plt.legend()
plt.show()
# val acc dipped at 8th epoch but then stabilized afterward

# evaluate with roc_auc_score
p = model.predict(padded_seq)
aucs = []
for j in range(6):
    auc = roc_auc_score(personality_type[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))

# test tweet
new_tweet = 'hi there everyone! can not wait to hangout with you all'
new_tweet = tokenizer.texts_to_sequences(new_tweet)
new_tweet = pad_sequences(new_tweet, maxlen= MAX_SEQUENCE_LENGTH)
labelencoder.inverse_transform(model.predict(new_tweet))