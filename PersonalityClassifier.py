#Personality Classifier Script

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

#change directory and load csv
os.chdir('D:\\')
data=pd.read_csv('mbt1_modified.csv', encoding='latin1')

#import libraries to clean texts
import re 
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

start=time.time()
corpus=[]
for i in range(data.shape[0]):
    posts=re.sub('[^a-zA-z]',' ',data['individualPosts'][i])
    posts=posts.lower()
    posts=posts.split() 
    ps=PorterStemmer()
    posts=[ps.stem(word) for word in posts if not word in set(stopwords.words('english'))]
    posts=' '.join(posts)
    corpus.append(posts)
elapsed=time.time()-start
print(elapsed.seconds)