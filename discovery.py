from watson_developer_cloud import DiscoveryV1
import numpy as np
import os
import mlr as mlr
import config as cfg
import json

discovery = DiscoveryV1(
      username=cfg.discovery_username,
      password=cfg.discovery_password,
      version="2018-04-28"
    )                                          #Discovery instance

def add_documents(collection_id, environment_id, size, dire):
    count = 0
    for i in range(size):
        try:
            with open(os.path.join(os.getcwd(), dire, 'rev_{0}.json'.format(i+1))) as fileinfo:
                add_doc = discovery.add_document(environment_id, collection_id, file=fileinfo)
            print(f'Added document {i}')
            count = count + 1
        except:
            print(f'Error uploading document {i}')
    return count


''' Searching discovery to get target passage for the question asked   que==question,    ints == intents'''
def query_docs(collection_id, environment_id, size):
    my_query = discovery.query(environment_id, collection_id, natural_language_query='', count=size, deduplicate=False)
    return my_query

def deleting_collection(collection_id = cfg.collection_id_test, environment_id = cfg.environment_id_test):
    delete_collection = discovery.delete_collection(environment_id, collection_id)
    print(json.dumps(delete_collection, indent=2))

def creating_collection(environment_id = cfg.environment_id_test, configuration_id = cfg.configuration_id, name = cfg.name_of_collection):
    new_collection = discovery.create_collection(environment_id=environment_id, configuration_id=configuration_id, name=name, description='Enrichment for prediction')
    #cfg.collection_id_test = new_collection["collection_id"]
    print(json.dumps(new_collection, indent=2))


def filter_data(data):
    val = {'Alpha' : 1, 'Beta': 2, 'Gamma': 3, 'Delta': 4}
    XX = []
    YY = []
    XX_o = []
    XX_k = []
    for temp in data:
        XX.append(temp['Review'])
        XX_k.append(temp['keywords'])
        XX_o.append([temp['Stars'], temp['Sentiment'], temp['disgust'], temp['joy'], temp['anger'], temp['fear'], temp['sadness'], len(temp['Review']), temp['Review'].count('?')])
        YY.append(val[temp['Critic']])
    return XX, YY, XX_o, XX_k
  

def filter_data_test(data):
    XX = []
    XX_o = []
    XX_k = []
    for temp in data:
        XX.append(temp['Review'])
        XX_k.append(temp['keywords'])
        XX_o.append([temp['Stars'], temp['Sentiment'], temp['disgust'], temp['joy'], temp['anger'], temp['fear'], temp['sadness'], len(temp['Review']), temp['Review'].count('?')])
    return XX, XX_o, XX_k
          

def scaling_x(X, high=1.0, low=0.0, enable=False):
    if enable == True:
        mins = mlr.mins
        maxs = mlr.maxs
        rng = maxs - mins
    else:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        rng = maxs - mins
    scaled_points = high - (((high - low) * (maxs - X)) / rng)
    X = scaled_points
    return X, mins, maxs, rng
         
        
def accuracy_cal(res, y):
    j = 0
    correct = 0
    for com in res:
        if com == y[j]:
            correct = correct+1
        j = j+1
    return correct


