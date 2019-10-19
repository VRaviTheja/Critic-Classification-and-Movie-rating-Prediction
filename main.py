# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:30:42 2018

@author: Ravi Theja
"""
import numpy as np
import config as cfg
import read as rd
import tfidf as ti
import mlr as mlr
import discovery as ds
import pickle
from operator import itemgetter

        
def read_upload(collection_id, environment_id, filename, size, dire, istrain):
    path = rd.text(filename, istrain)      # Splitting train.json to multiple files
    print(path)
    
    count = ds.add_documents(collection_id, environment_id, size, dire)    # Adding Training documents
    return count


''' Main Training Function   '''
def training():
    
    collection_id = cfg.collection_id_train    # Training IDS
    environment_id = cfg.environment_id_train

    if cfg.uploaded_t == False:
        samp = read_upload(collection_id, environment_id, cfg.trainig_filename, cfg.train_size, cfg.train_dir, True)
        cfg.uploaded_t = True
        
    ''' Querying from discovery '''
    response = ds.query_docs(collection_id, environment_id, cfg.count_t)
    
    '''  Taking required data for model '''
    data = [{'Review': review['Review'], 'Critic': review['Critic'], 'Stars': review['Stars'], 'Sentiment':review['enriched_Review']['sentiment']['document']['score'], 'keywords': " ".join([mose['text'] for mose in review['enriched_Review']['keywords']]), 'disgust': review['enriched_Review']['emotion']['document']['emotion']['disgust'], 'joy': review['enriched_Review']['emotion']['document']['emotion']['joy'], 'anger': review['enriched_Review']['emotion']['document']['emotion']['anger'], 'fear': review['enriched_Review']['emotion']['document']['emotion']['fear'], 'sadness': review['enriched_Review']['emotion']['document']['emotion']['sadness'], 'rev_len': len(review['Review']), 'rev': review['Review'], 'que_mark': review['Review'].count('?')} for review in response['results']]
    XX, YY, XX_other_features, XX_keywords = ds.filter_data(data)
    
    ''' Scaling numerical features  '''
    XX_other_features, mins1, maxs1, rng1 = ds.scaling_x(XX_other_features, high=0.5)
    
    weights, vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng = ti.tfidf_train(XX, YY, XX_other_features, XX_keywords) 

    with open('objects.pkl', 'wb') as f: 
        pickle.dump([vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng, mins1, maxs1, rng1, weights], f)
        
    
''' Main Testing function  '''
def predicting(weights, vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng, mins1, maxs1, rng1):
    
    collection_id = cfg.collection_id_test    # Testing IDS
    environment_id = cfg.environment_id_test
    
    if cfg.uploaded_test == False:
        samp = read_upload(collection_id, environment_id, cfg.testing_filename, cfg.test_size, cfg.test_dir, False)
        cfg.uploaded_test = True
    
    ''' Querying from discovery '''
    response = ds.query_docs(collection_id, environment_id, cfg.count_test)
    
    '''  Taking required data for model '''
    data = [{'name': review['extracted_metadata']['filename'], 'Review': review['Review'], 'Stars': review['Stars'], 'Sentiment':review['enriched_Review']['sentiment']['document']['score'], 'keywords': " ".join([mose['text'] for mose in review['enriched_Review']['keywords']]), 'disgust': review['enriched_Review']['emotion']['document']['emotion']['disgust'], 'joy': review['enriched_Review']['emotion']['document']['emotion']['joy'], 'anger': review['enriched_Review']['emotion']['document']['emotion']['anger'], 'fear': review['enriched_Review']['emotion']['document']['emotion']['fear'], 'sadness': review['enriched_Review']['emotion']['document']['emotion']['sadness'], 'rev_len': len(review['Review']), 'rev': review['Review'], 'que_mark': review['Review'].count('?')} for review in response['results']]
    newlist = sorted(data, key=itemgetter('name'))
    
    XX, XX_other_features, XX_keywords = ds.filter_data_test(newlist)
    
    ''' Scaling Numerical features '''
    high = 0.5
    low = 0.0
    XX_other_features = high - (((high - low) * (maxs1 - XX_other_features)) / rng1)
    
    result = ti.tfidf_test(XX, XX_other_features, XX_keywords, weights, vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng, mins1, maxs1, rng1)
    
    val = {'1' : 'Alpha', '2': 'Beta', '3': 'Gamma', '4': 'Delta'}
    with open("output.txt", "w") as f:
        for s in result:
            f.write(val[str(s)] +"\n")
    
    return result



def predict_calling():
    with open('objects.pkl', 'rb') as f:  
        vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng, mins1, maxs1, rng1, weights = pickle.load(f)
        result = predicting(weights, vocabulary, VOCABULARY_SIZE, my_idf_vector, word_index, mins, maxs, rng, mins1, maxs1, rng1)
        return result


if __name__ == "__main__":
    if cfg.training_needed:     # Enable accordingly in config file
        print("Training.....")
        training()
        print("Predicting.....")
        result = predict_calling()
        print(result)
    else:
        print("Prediction: ")
        result = predict_calling()
        print(result)

    
    
    