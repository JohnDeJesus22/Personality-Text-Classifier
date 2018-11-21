# Personality Type CNN with Tensorflow and Keras

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

# predefining parameters
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10


# change directory
os.chdir('D:\\glovePretrainedFlies\\glove.6B')
glove_file = 'glove.6B.50d.txt'

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
data = pd.read_csv('mbt1_modified.csv',encoding = 'latin1',usecols=['individualPosts',
                                                                    'text_length',
                                                                     'IntroExtro'])
tweets = data['individualPosts'][data.text_length >= 3].values
personality_type = data['IntroExtro'][data.text_length >= 3].values
labelencoder = LabelEncoder()
personality_type = labelencoder.fit_transform(personality_type)


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
model.compile(loss = 'binary_crossentropy',optimizer = 'rmsprop',
              metrics=['accuracy'])

# oversampling with smote(testing after bbg and fit_generator)
print('Beginning SMOTE sampling')
class_totals = Counter(personality_type)
sm = SMOTE(random_state = 1,sampling_strategy = 'minority')                                                 
padded_seq_res, personality_type_res = sm.fit_resample(padded_seq, personality_type)
res_class_totals = Counter(personality_type_res)
print('Sampling complete:', res_class_totals)


# training the model with keras fit
print('Training Model')
classifier = model.fit(padded_seq_res, personality_type_res,
                       batch_size = BATCH_SIZE,epochs = EPOCHS,
                       validation_split = VALIDATION_SPLIT)
                       
print('Training Completed')

# save the model
os.chdir('D:\\PacktTensorflowCourseMaterials\\PersonalityCNN')

save_model(model=model,filepath = './cnn-model-2',overwrite = True,
                 include_optimizer = True)

# plot loss and validation loss
plt.plot(classifier.history['loss'],label='training loss')
plt.plot(classifier.history['val_loss'],label = 'validation_loss')
plt.legend()
plt.show()


# plot accuracy and validation accuracy
plt.plot(classifier.history['acc'], label = 'training accuracy')
plt.plot(classifier.history['val_acc'], label = 'validation accuracy')
plt.legend()
plt.show()


# evaluate with confusion matrix
p = model.predict_classes(padded_seq)
real_p = (p>.5)

cm = confusion_matrix(personality_type,real_p)


# test tweet
new_tweet = np.array(["I prefer to be alone."])
new_tweet = tokenizer.texts_to_sequences(new_tweet)
new_tweet = pad_sequences(new_tweet, maxlen = MAX_SEQUENCE_LENGTH)
result = labelencoder.inverse_transform(model.predict(new_tweet))
print(result[0])

# Hyper-Parameter tuning with Bayesian Optimization

# setup parameters to tune

dim_num_epochs = Integer(low = 10, high = 30, name = 'num_epochs')
dim_batch_size = Integer(low = 100, high = 140, name = 'batch_size')
dim_optimizer = Categorical(categories = ['rmsprop','adam'], name = 'optimizer')
dim_activation = Categorical(categories = ['relu', 'sigmoid'], 
                              name = 'activation')
dim_max_sequence_length = Integer(low = 100, high = 240, name ='max_sequence_length' )

dimensions = [dim_num_epochs, dim_batch_size, 
              dim_optimizer, dim_activation, dim_max_sequence_length]

# set default parameters
default_parameters = [10,128, 'rmsprop','relu',100]


# helper function to engage with tensorboard
def log_dir_name(num_epochs, batch_size, optimizer, 
                 activation, max_sequence_length):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(num_epochs,
                       batch_size,
                       optimizer,
                       activation,
                       max_sequence_length)

    return log_dir

# function to create model
def create_model(num_epochs, batch_size, optimizer,
                 activation, max_sequence_length):
    
    # optimize max sequence length
    padded_seq = pad_sequences(sequences, maxlen = max_sequence_length)
    print('Shape of data tensor:', padded_seq.shape)
    
    num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
    embedded_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word,index in word2idx.items():
        if index < MAX_VOCAB_SIZE:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedded_matrix[index]=embedding_vector
            
    # place embeddings into an embedding layer
    embedding_layer = layers.Embedding(num_words, EMBEDDING_DIM, weights=[embedded_matrix],
                                       input_length = max_sequence_length,
                                       trainable = False)
    
    print('Initiating model construction')
    
    # optimize activation and optimizer
    inputs = layers.Input(shape = (max_sequence_length,))
    x = embedding_layer(inputs)
    x = layers.Conv1D(128,3,activation = activation)(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(128,3,activation = activation)(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(128,3, activation = activation)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128,activation = activation)(x)
    output = layers.Dense(1,activation = 'sigmoid')(x)
    
    model = Model(inputs,output)
    model.compile(loss = 'binary_crossentropy',optimizer = optimizer,
                  metrics=['accuracy'])
    
    print('Model construnction complete.')
    
    return model
    
# initialize best model name and best accuracy
name_best_model = 'best_model_text_cnn'
best_accuracy = 0.0

@use_named_args(dimensions = dimensions)
def fitness(num_epochs, batch_size, optimizer,
                 activation, max_sequence_length):
    
    print('Number of Epochs:', num_epochs)
    print('Batch Size:', batch_size)
    print('Optimizer:', optimizer)
    print('Activation:', activation)
    print('Max Sequence Length:', max_sequence_length)
    print('')
    
    # create model
    model = create_model(num_epochs = num_epochs,
                         batch_size = batch_size,
                         optimizer = optimizer,
                         activation = activation,
                         max_sequence_length= max_sequence_length)

    
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(num_epochs, batch_size,optimizer,
                           activation,max_sequence_length)
    
    # callback-log
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
    
    # train the model
    history = model.fit(padded_seq,
                        personality_type,
                        epochs = num_epochs,
                        batch_size = batch_size,
                        validation_split = VALIDATION_SPLIT,
                        callbacks = [callback_log])
    
    # get accuracy of validation set after the last training
    accuracy = history.history['val_acc'][-1]
    
    print()
    print('Accuracy:', accuracy)
    print()
    
    # get best_accuracy
    global best_accuracy
    
    if accuracy > best_accuracy:
        
        model.save(name_best_model)
        
        best_accuracy = accuracy
        
    # delete the model to prepare for next one
    del model
    
    # clear keras session
    K.clear_session()
    
    # Return -accuracy since this value needs to be minizied to get the lowest fitness 
    # score and the highest classification accuracy
    return -accuracy

# test with defaults (successful 11/20/18)
fitness(x = default_parameters)

# run optimization
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=11,
                            x0=default_parameters)
