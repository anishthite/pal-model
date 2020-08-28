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

import random
import pandas as pd
class Retriever():
    def __init__(self, dataset):
        self.dataset = pd.read_csv(dataset)

    def predict(self, query):
        answers = [line for line in self.dataset if query in line]
        dataset_with_query = self.dataset[self.dataset['title'].str.contains(query) | self.dataset['selftext'].str.contains(query)]
        if not dataset_with_query.empty:
            sample = dataset_with_query.sample(1)
            return sample['title'].values[0].strip('\n') +' ' +sample['selftext'].values[0].strip('\n') 
        return "Sorry I don't have a joke about that right now"

