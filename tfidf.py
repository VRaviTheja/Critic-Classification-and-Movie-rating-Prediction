# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:42:45 2018

@author: Ravi Theja
"""
  
import numpy as np
import math
import sword
import re
import mlr as mlr


def tokzr_WORD(txt): return (re.findall(r'(?ms)\W*(\w+)', txt))  # split words

stop_words = sword.stop_words

def tokenize(text):
    words = tokzr_WORD(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]


def freq(term, document):
  return document.count(term)


def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount 

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)


def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

'''          ADDED         #keyword   '''
def fit_trans(vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index,words1):
    tfidf_documents = []
    jo = 0
    for document in words1:
        print(jo)
        sam = np.zeros(VOCABULARY_SIZE)       #keyword[jo]
        for term in document:
            if term in vocabulary:
                tf = sublinear_term_frequency(term, document)
                sam[word_index[term]] = (tf * my_idf_vector[word_index[term]])
        tfidf_documents.append(sam)
        jo = jo + 1
    tfidf_documents = np.asarray(tfidf_documents)
    return tfidf_documents


def one_hotencoding(y_train):
    onehot_encoded = []
    for temp in y_train:
        if temp == 1:
            onehot_encoded.append([1,0,0,0])
        if temp == 2:
            onehot_encoded.append([0,1,0,0])
        if temp == 3:
            onehot_encoded.append([0,0,1,0])
        if temp == 4:
            onehot_encoded.append([0,0,0,1])
    
    print(len(onehot_encoded))
    
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    
    for temp in onehot_encoded:
        y1.append([temp[0]])
        y2.append([temp[1]])
        y3.append([temp[2]])
        y4.append([temp[3]])
    return y1,y2,y3,y4
    

''' TRAINING FUNCTION '''
def tfidf_train(X_train, y_train, X_train_o, X_train_k):
    
    global vocabulary
    vocabulary = set()
    global words
    words = []
    global words1
    words1 = []
    
    for temp in X_train:
        words1.append(tokenize(temp))
        words = words + tokenize(temp)
    
    vocabulary.update(words)
    
    new_vocab = set()
    vocab = list(vocabulary)
    bank = []
    ho = 0
    for p in vocab:
        ho = ho + 1
        count = 0
        for c in words1:
            if p in c:
                count = count + 1
        if (count>=5) and (count<(0.8*len(X_train))):
            bank.append(p)
        if(ho%500 == 0):
            print("adding... ", ho)
    new_vocab.update(bank)
    
    vocabulary = new_vocab
    
    vocabulary = list(vocabulary)
    
    global word_index
    word_index = {w: idx for idx, w in enumerate(vocabulary)}
    global VOCABULARY_SIZE, DOCUMENTS_COUNT
    VOCABULARY_SIZE = len(vocabulary)
    DOCUMENTS_COUNT = len(X_train)
     
    print(VOCABULARY_SIZE, DOCUMENTS_COUNT)
    print("Calucalating idf........")
    global my_idf_vector
    my_idf_vector = [idf(word, words1) for word in vocabulary]
    '''     ADDING   '''
    keyword_data = []
    for temp in X_train_k:
        keyword_data.append(tokenize(temp))
    ''' ADDING       '''
    global tfidf_documents
    tfidf_documents = fit_trans(vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, words1)
    
    y1,y2,y3,y4 = one_hotencoding(y_train)
    
    
    weights, mins, maxs, rng = mlr.multi_svm(tfidf_documents, y1,y2,y3,y4, X_train_o)
    return weights, vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng

'''   TESTING FUNCTION  '''
def tfidf_test(X_test, X_test_o, X_test_k, weights, vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng, mins1, maxs1, rng1):
    global words_t
    words_t = []
    global words1_t
    words1_t = []
    
    for temp in X_test:
        words1_t.append(tokenize(temp))
        words_t = words_t + tokenize(temp)

    '''ADDING'''
    keyword_data = []
    for temp in X_test_k:
        keyword_data.append(tokenize(temp))
    '''ADDING'''
    
    
    global tfidf_documents_test
    tfidf_documents_test = fit_trans(vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, words1_t)
    print(len(tfidf_documents_test), len(tfidf_documents_test[0]), len(X_test_o), len(X_test_o[0]), type(tfidf_documents_test), type(X_test_o), type(tfidf_documents_test[0]), type(X_test_o[0]), type(tfidf_documents_test[0][0]), type(X_test_o[0][0]), tfidf_documents_test[0][0], X_test_o[0][0])
        
    tfidf_documents_test = np.concatenate((X_test_o, tfidf_documents_test),axis=1)

    maxs2 = np.concatenate((maxs1,maxs), axis=0)
    rng2 = np.concatenate((rng1,rng), axis=0)
    high = 1.0
    low = 0.0
    tfidf_documents_test11 = high - (((high - low) * (maxs2 - tfidf_documents_test)) / rng2)

    result = mlr.predict(tfidf_documents_test11, weights)
    return result





