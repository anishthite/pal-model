# import random
# class Retriever():
#     def __init__(self, dataset):
#         with open(dataset,'r') as datasetfile:
#             self.dataset = [line.rstrip('\n') for line in datasetfile]

#     def predict(self, query):
#         answers = [line for line in self.dataset if query in line]
#         if answers is not []:
#             return random.choice(answers)
#         return "Sorry I don't have a joke about that right now"

import gensim
import gensim.downloader as api
import scipy
import numpy as np
# DATASET = 'humor_challenge_data/bot_data/qa_total_word2vec.csv'
# dataset = pd.read_csv(DATASET)
# dataset['word2vec'],

# dataset['dataset'][np.argmin(dataset[word2vec].apply(lambda x: scipy.spatial.distance.cosine(x, model[query])).values)]



import random
import pandas as pd
class Retriever():
    def __init__(self, dataset, tokenized_dataset, word2vec_dataset):
        self.dataset = pd.read_csv(dataset)
        self.tokenized_dataset =  pd.read_csv(tokenized_dataset)
        # self.word2vec_dataset = pd.read_csv(word2vec_dataset)
        self.word2vec_dataset=np.load(word2vec_dataset)['arr_0']
        self.model = api.load("word2vec-google-news-300")


    def predict(self, query):
        print(type(query))
        # query = query['history']
        a = sum([self.model[word] for word in query.split(' ') if word in self.model.vocab])
        # mn = np.argmin(self.word2vec_dataset['word2vec'].apply(lambda x: scipy.spatial.distance.cosine(x,self.model[query])).values)
        mn = np.argpartition(np.array([scipy.spatial.distance.cosine(row,a) for row in self.word2vec_dataset]),10)[10]
        mn_data = self.dataset.values[mn]
        if not mn_data.empty:
            sample = dataset_with_query.sample(1)
            return str(sample['title'].values[0]).strip('\n') +' '+str(sample['selftext'].values[0]).strip('\n') 
        return "Sorry I don't have a joke about that right now"
        # return str(mn_data[0].strip('\n') + ' ' + str(mn_data[1].strip('\n')))
        # if len(query.split(' ')) > 1:
        #     dataset_with_query = self.dataset[self.dataset['title'].str.contains(query) | self.dataset['selftext'].str.contains(query)]
        # else:
        #     dataset_with_query = self.dataset[self.tokenized_dataset['0'].apply(lambda x : query.lower() in x)]
        # if not dataset_with_query.empty:
        #     sample = dataset_with_query.sample(1)
        #     return str(sample['title'].values[0]).strip('\n') +' '+str(sample['selftext'].values[0]).strip('\n') 
        # return "Sorry I don't have a joke about that right now"
