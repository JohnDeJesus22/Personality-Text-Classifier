# Data Prep for Personality Text Classifier

#import libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import re

#import data set
df=pd.read_csv('mbti_1.csv')
df.info()

#number of columns to house each entry
columnlist=[str(i) for i in range(50)]

#split comments in posts column into idividual columns to melt
df[columnlist] = df['posts'].str.split(r'\|\|\|',n=49,expand=True)


#drop original group posts column
df.drop(['posts'],inplace=True, axis=1)

#put all posts into a single column called individualPosts
#then drop variable column
df_melt=pd.melt(df,id_vars=['type'],value_vars=columnlist, value_name='individualPosts')
df_melt.drop(['variable'],inplace=True, axis=1)

#filter out comments that contain links to sites and/or images
df_melt=df_melt[df_melt['individualPosts'].str.contains('http')==False]
#total remaining is 396313 rows

#check for number of nulls, result=0
nulls=sum(df_melt['individualPosts'].isnull()==True)#

#strip each " ' " at the beginning of each comment
df_melt['individualPosts']=df_melt['individualPosts'].str.lstrip("'")

#create column for post length
df_melt['text_length'] = df_melt['individualPosts'].apply(lambda x: len(x))

#Send melted df to csv
df_melt.to_csv('mbt_1_Melted.csv',index=False)