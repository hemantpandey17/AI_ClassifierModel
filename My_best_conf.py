import sys
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import os
import nltk
from sklearn.feature_selection import SelectPercentile, chi2
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import _pickle as cPickle

#Defining Categories which are included in the dataset from 20newsgroups dataset
categories = ['rec.sport.hockey',  'sci.med', 'soc.religion.christian', 'talk.religion.misc']
#Importing dataset using sklearn
twenty_train = sklearn.datasets.load_files(container_path= r'Training', encoding='latin1')

print(len(twenty_train.data))


from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
twenty_test = sklearn.datasets.load_files(container_path= r'Judge', encoding= 'latin1' )
docs_test = twenty_test.data

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

import string
length = len(twenty_train.data)
value = []
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
print("Using Linear SVC Classifier")
for n in range(100, length, 200):
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),lowercase=True, stop_words='english')),
                        ('tfidf', TfidfTransformer(norm='l2')),
                        ('clf', LinearSVC(C=1 ,loss='squared_hinge', dual = False, penalty='l2', tol=1e-6, max_iter=1000, random_state=None,
                                        fit_intercept = True))])
    text_clf = text_clf.fit(twenty_train.data[1:n], twenty_train.target[1:n])
    predicted = text_clf.predict(docs_test)
    print(np.mean(predicted == twenty_test.target))
    from sklearn import metrics
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    value.append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))

print("Precision Recall Graph of whole dataset \n")
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

with open('my_own_best_classifier.pkl', 'wb') as fid:
    cPickle.dump(text_clf, fid)

print(value)
import matplotlib.pyplot as plt
xaxis = [n for n in range(100, length, 200)]
plt.plot(xaxis, value, 'r-', label='LinearSVC')
plt.xlabel('Training samples')
plt.ylabel('F1 scores')
plt.show()

