# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer

# dataset import using pandas
dataset = pd.read_csv('./reviews.tsv', delimiter='\t', quoting = 3)

# for stemming the word - prefix/sufix
ps = PorterStemmer()

# uncomment this line if 'stopwords are not downloaded'
# nltk.download('stopwords')

# imporing stopwords
from nltk.corpus import stopwords

corpus = []
# filtering the text
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =  ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 18)

# Fitting Naive Bayes to the Training set
# you can use MultinomialNB, BernoulliNB or GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix to check accurate answers
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# function that reply whether the review is good or bad 
def ask():
    arr = list()
    input_string = str(input('submit comment = '))
    review = re.sub('[^a-zA-Z]', ' ', input_string).lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =  ' '.join(review)
    arr.append(review)
    cou = cv.transform(arr).toarray()
    pred_orig = classifier.predict(cou)
    if(pred_orig[0] == 1):
        print('good review')
    else:
        print('bad review')

ask()
