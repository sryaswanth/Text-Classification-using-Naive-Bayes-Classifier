# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:49:04 2019

@author: sryas
"""

import datetime
starttime = datetime.datetime.now()
import spacy

nlp = spacy.load("en_core_web_md")

import os
import gc

path1 = r"H:/train/rec.autos"
path2 = r"H:/train/rec.sport.hockey"
testpath1 = r"G:/test/rec.autos"
testpath2 = r"G:/test/rec.sport.hockey"

testlists = []
lists = []
unique_words = []
unique_word_types = []
count = []


def preprocess(myfilep):
    string = ''
    myfile = nlp(myfilep)
    for token in (myfile):
        
            #token = token.text
            if not token.pos == "SYM":
                if not token.is_stop:
                    
                    string =string+" "+(token.lemma_).lower()
                    if (token.lemma_).lower() not in unique_words:
                        unique_words.append((token.lemma_).lower())
                    a= token.pos_
                    if a not in unique_word_types:
                        unique_word_types.append(a)
    lists.append(string)
    
                        
                        
def preprocesstest(myfilep):
    stringtest = ''
    myfile = nlp(myfilep)
    for token in (myfile):
        
            #token = token.text
            if not token.pos == "SYM":
                if not token.is_stop:
                    
                    stringtest = stringtest+" "+(token.lemma_).lower()
                    if (token.lemma_).lower() not in unique_words:
                        unique_words.append((token.lemma_).lower())
                    a= token.pos_
                    if a not in unique_word_types:
                        unique_word_types.append(a)
                        
    testlists.append(stringtest)

    

def fileload(path):
    count=0
    dir = os.listdir(path)
    for files in dir:
    
        f=path+'/'+files
        with open(f, encoding = "utf-8", errors = 'ignore') as file:
                preprocess(file.read())
                file.close()
        gc.collect()
        count+=1
    return count
        
def testfileload(path):
    count =0 
    dir = os.listdir(path)
   
    for files in dir:
        f=path+'/'+files
        with open(f, encoding = "utf-8", errors = 'ignore') as file:
                preprocesstest(file.read())
                file.close()
        gc.collect()
        count+=1
    return count
                
count1 = fileload(path1)
print("Total documents in Training set of rec.autos:",count1)
gc.collect()
count2 = fileload(path2)
print("Total documents in Training set of rec.sport.hockey:",count2)

count3 = testfileload(testpath1)
print("Total documents in testing set of rec.autos:",count3)
count4 = testfileload(testpath2)
print("Total documents in testing set of rec.sport.hockey:",count4)


print("Total Number of Documents in Training Set:", len(lists))
print("Total Number of Documents in Test set:", len(testlists))
print("Total Documents:", len(lists)+len(testlists))
print("Total Unique Words:",len(unique_words))
print("Total Unique Word types:",len(unique_word_types))


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_X = vectorizer.fit_transform(lists)
#print(vectorizer.get_feature_names())
#print(train_X.toarray())
print("train_X.shape is:",train_X.shape)
nr_count = len(train_X.toarray())
test_X = vectorizer.transform(testlists)
#print(vectorizer.get_feature_names())
#print(test_X.toarray())
print("test_X.shape is:",test_X.shape)
train_y = np.vstack((np.zeros([count1,1], dtype=int), np.ones([count2,1], dtype=int)))
print("train_y.shape is:",train_y.shape)
test_y = np.vstack((np.zeros([count3,1], dtype=int), np.ones([count4,1], dtype=int)))
print("test_y.shape is:",test_y.shape)




from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
naivebayes = MultinomialNB()
naivebayes.fit(train_X,train_y.ravel())
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#print(naivebayes.predict(test_X))
a = naivebayes.predict(test_X)
#print(a)
list1=[]
for i in test_y:
        list1.append(i[0])
print("Accuracy Score:",accuracy_score(naivebayes.predict(test_X),list1))
print("F1 Score:",f1_score(test_y,a, average='binary'))

#print(test_y)

endtime = datetime.datetime.now()
print("Total Execution Time in HH:MM:SS Format:", endtime-starttime)