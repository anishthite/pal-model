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
import profanity_check as pc




import random
import pandas as pd
from bert.bertrun import *
import pickle


class Retriever():
    
    def __init__(self, dataset, tokenized_dataset):
        self.dataset = pd.read_csv(dataset)
        self.dataset = self.dataset.fillna('')
        # self.tokenized_dataset =  pd.read_csv(tokenized_dataset)
        with open(tokenized_dataset, "rb") as fp:   # Unpickling
            self.tokenized_dataset = pickle.load(fp)
        # self.word2vec_dataset = pd.read_csv(word2vec_dataset)
        # self.word2vec_dataset=np.load(word2vec_dataset)['arr_0']
        self.word2vec_model = api.load("word2vec-google-news-300")
        model_path='/nethome/ilee300/Workspace/pal-model/trained_models/bettertrainbert_medium_joker_50066.pt'
        self.bert_model = HumorDetector(model_path)


    def predict(self, query):
        #encode
        if pc.predict([query])[0] ==1
            return "Joke is not appropriate"
        n = 10
        l = self.word2vec_model.most_similar([query], topn = n)
        words = [i[0] for i in l]
        # return str(mn_data[0].strip('\n') + ' ' + str(mn_data[1].strip('\n')))
        if len(query.split(' ')) > 1:
            # self.dataset.apply(lambda x: any(j in x for j in words))
            dataset_with_query = self.dataset[self.dataset['title'].apply(lambda x: any(j in str(x) for j in words)) | self.dataset['selftext'].apply(lambda x: any(j in str(x) for j in words))]
            # dataset_with_query = self.dataset[  self.dataset['title'].str.contains(query) | self.dataset['selftext'].str.contains(query)].values
        else:

            dataset_with_query = self.dataset[pd.Series([any(j in x for j in words) for x in self.tokenized_dataset])]
            # dataset_with_query = self.dataset[self.tokenized_dataset['0'].apply(lambda x : query.lower() in x)]
        if not len(dataset_with_query)>0:
            # dataset_with_query = dataset_with_query.fillna('')
            dataset_with_query = (dataset['title'].str.strip('\n')+' '+dataset['selftext'].str.strip('\n')).values
            jokeprobs = jokeProbs(dataset_with_query)
            m = 10
            k = min(m,len(jokeprobs)-1)
            return np.random.choice(dataset_with_query[np.argpartition(jokeprobs,k)[:k]],1)

            # sample = dataset_with_query.sample(1)
            # return str(sample['title'].values[0]).strip('\n') +' '+str(sample['selftext'].values[0]).strip('\n') 
        return "Sorry I don't have a joke about that right now"



    def jokeCheck(thresh, jokeList):
        '''
            Randomly selects jokes among those that have probability above the threshold
            Parameter:
                threshold : the threshold that is used to evaluate whether the joke is "good"
                jokeList: list of jokes retrieved from the dataset

            Return:
                None: if the jokeList is empty
                str: 1 joke with a probability higher than the threshold is randomly chosen. If nth is above the threshold, the max is chosen.

        '''
        if not jokeList:
            return None
        
        self.bert_model.model.eval()
        above_thresh = []
        max_joke = (0,'')
        for jokes in jokeList:
            answer, probs = self.bert_model(jokes)
            if answer == 1:
                if( probs > thresh):
                    above_thresh.append(jokes)
                    if probs > max_joke[0]:
                        max_joke = (probs, jokes)
        if above_thresh:
            print("above thresh : ", above_thresh)
            return np.random.choice(above_thresh,1)
        elif max_joke[0] != 0:
            print("nth above thresh")
            return max_joke[1]
        else:
            print("no joke found")
            return None 

    def jokeProbs(jokeList)
        if not jokeList:
                return None
        self.bert_model.model.eval()
        return np.array([self.bert_model(joke)[1] for joke in jokelist])
   
# if __name__ == "__main__":
#     jokeList = ["What do you call a dinosaur that is sleeping? A dino-snore!", "cow on the moon goes moo", "What did the left eye say to the right eye? Between us, something smells"]
#     ans = Retriever.jokeCheck(0.3, jokeList)
#     print("\n\n\n", ans, "\n\n")

