import sys
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#Defining Categories which are included in the dataset from 20newsgroups dataset
categories = ['rec.sport.hockey',  'sci.med', 'soc.religion.christian', 'talk.religion.misc']
twenty_train = sklearn.datasets.load_files(container_path= r'Training', encoding='latin1')
# Length of the data
print(len(twenty_train.data))

#Displaying first four lines of first file of dataset
print("\n".join(twenty_train.data[0].split("\n")[:3]))

print(twenty_train.target_names[twenty_train.target[0]])

no_of_grams = int(input('Enter 1 if you want to use Unigrams, 2 for Bigrams \n'))
print(no_of_grams)
from sklearn.feature_extraction.text import CountVectorizer
if no_of_grams == 1:
    print("Using Unigrams")
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(ngram_range=(1,1))
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print(count_vect)

elif no_of_grams == 2:
    print("Using Bigrams")
    from sklearn.feature_extraction.text import CountVectorizer
    print("Here")
    count_vect = CountVectorizer(ngram_range=(2, 2))
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print("there")
    print(count_vect)


#print(X_train_counts)
print(X_train_counts.shape)

print(count_vect.vocabulary_.get(u'algorithm'))

# Computing tf-idf scores
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
print(X_train_tfidf.shape)

'''classifier_type = int(input("Enter the type of classifier as follows: \n"
                        "1. Naive Baye's \n"
                        "2. Logistic Regression \n"
                        "3. Support Vctor Machines \n"
                        "4. Random Forest \n" ))'''


#Working on sample data
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

length = len(twenty_train.data)
#predicted = clf.predict(X_new_tfidf)

import numpy as np
twenty_test = sklearn.datasets.load_files(container_path= r'Test', encoding= 'latin1' )
docs_test = twenty_test.data

value = []
i = -1
for classifier_type in range(1,5):
    i = i+1
    value.append([])
    for n in range(100, length, 200):
        if(classifier_type == 1):
            from sklearn.pipeline import Pipeline
            from sklearn.naive_bayes import MultinomialNB
            print("Using Naive Baye's")
            from sklearn.naive_bayes import MultinomialNB

            text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultinomialNB()),
                                ])
            text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
            predicted = text_clf.predict(docs_test)
            print(np.mean(predicted == twenty_test.target))
            from sklearn import metrics
            print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
            value[i].append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))

        elif(classifier_type == 3):
            from sklearn.pipeline import Pipeline
            from sklearn.svm import LinearSVC
            print("Using Linear SVC Classifier")
            clf = LinearSVC().fit(X_train_tfidf, twenty_train.target)
            text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', LinearSVC())])
            text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
            predicted = text_clf.predict(docs_test)
            print(np.mean(predicted == twenty_test.target))
            from sklearn import metrics
            print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
            value[i].append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))

        elif(classifier_type == 4):
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestClassifier
            print("Using Random Forest Classifier")
            clf = RandomForestClassifier().fit(X_train_tfidf, twenty_train.target)
            text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', RandomForestClassifier())])
            text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
            predicted = text_clf.predict(docs_test)
            print(np.mean(predicted == twenty_test.target))
            from sklearn import metrics
            print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
            value[i].append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))

        elif(classifier_type == 2):
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            print("Using Logistic Regression")
            clf = LogisticRegression().fit(X_train_tfidf, twenty_train.target)
            text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', LogisticRegression())])
            text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
            predicted = text_clf.predict(docs_test)
            print(np.mean(predicted == twenty_test.target))
            from sklearn import metrics
            print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
            value[i].append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))

print(value)
import matplotlib.pyplot as plt
xaxis = [n for n in range(100, length, 200)]
plt.plot(xaxis, value[0], 'r-', label='NB')
plt.plot(xaxis, value[1], 'b-', label='LR')
plt.plot(xaxis, value[2], 'g-', label='SV')
plt.plot(xaxis, value[3], 'y-', label='RF')
plt.xlabel('Training samples')
plt.ylabel('F1 scores')
plt.show()
