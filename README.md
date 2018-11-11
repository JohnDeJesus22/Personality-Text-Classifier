# Personality-Text-Classifier

Using data from [Myers Briggs Personality Type test data found on kaggle](https://www.kaggle.com/mahekhooda/introvert-extroverts/data),
I will make a classifier to determine the type of personality based on the text input. 

This repo will house the code for the classifier along with scripts of data wrangling/prep, EDA, and assessments of the classifier.

## Classifications
The data contains classes for 16 different personality types. From a binary perspective, the personality is either introvert or extrovert. With some feature engineering the binary column with those previous mentioned personality types was created. 

## Model
The model used will be a CNN using an embedded layer with pretrained 50 dimensional glove vectors for tweets.

### Attempt One:
Data was sigificantly imbalanced with the introverts. I was aware of this intially but wanted to try it since this was my first time with such a model. Accuracy averaged around 76% on training and validation but contained many false negatives due to the imbalance

### Attempt Two:
Applied SMOTE oversampling from imblearn api before training the model. Improved accuracy on both the the training (about 79%) and the validation sets (about 91%).

11/10/18: will attempt SMOTE with ENN and then will apply bayesian optimization to improve the overall model.

