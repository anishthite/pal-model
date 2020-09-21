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



import random
import pandas as pd
from bert.bertrun import *
# DATASET = 'humor_challenge_data/bot_data/qa_total_word2vec.csv'
# dataset = pd.read_csv(DATASET)
# dataset['word2vec'],''

# dataset['dataset'][np.argmin(dataset[word2vec].apply(lambda x: scipy.spatial.distance.cosine(x, model[query])).values)]


model_path='/nethome/ilee300/Workspace/pal-model/trained_models/bettertrainbert_medium_joker_50066.pt'
mymodel = HumorDetector(model_path)
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
        # a = sum([self.model[word] for word in query.split(' ') if word in self.model.vocab])
        # mn = np.argpartition(np.array([scipy.spatial.distance.cosine(row,a) for row in self.word2vec_dataset]),10)[10]
        # mn_data = self.dataset.values[mn]
        # if not mn_data.empty:
        #     sample = dataset_with_query.sample(1)
        #     return str(sample['title'].values[0]).strip('\n') +' '+str(sample['selftext'].values[0]).strip('\n') 
        # return "Sorry I don't have a joke about that right now"

        # return str(mn_data[0].strip('\n') + ' ' + str(mn_data[1].strip('\n')))
        if len(query.split(' ')) > 1:
            dataset_with_query = self.dataset[self.dataset['title'].str.contains(query) | self.dataset['selftext'].str.contains(query)]
        else:
            dataset_with_query = self.dataset[self.tokenized_dataset['0'].apply(lambda x : query.lower() in x)]
        if not dataset_with_query.empty:
            dataset_with_query = dataset_with_query.fillna('')
            dataset_with_query = (dataset['title'].str.strip('\n')+' '+dataset['selftext'].str.strip('\n')).values
            jokeprobs = jokeProbs(dataset_with_query)
            k = min(10,len(jokeprobs)-1)
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
        
        mymodel.model.eval()
        above_thresh = []
        max_joke = (0,'')
        for jokes in jokeList:
            answer, probs = mymodel(jokes)
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
        mymodel.model.eval()
        return np.array([mymodel(joke)[1] for joke in jokelist])
   
if __name__ == "__main__":
    jokeList = ["What do you call a dinosaur that is sleeping? A dino-snore!", "cow on the moon goes moo", "What did the left eye say to the right eye? Between us, something smells"]
    ans = Retriever.jokeCheck(0.3, jokeList)
    print("\n\n\n", ans, "\n\n")

