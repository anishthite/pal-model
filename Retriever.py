import random
class Retriever():
    def __init__(self, dataset):
        with open(dataset,'r') as datasetfile:
            self.dataset = [line.rstrip('\n') for line in datasetfile]

    def predict(self, query):
        answers = [line for line in self.dataset if query in line]
        if answers is not []:
            return random.choice(answers)
        return "Sorry I don't have a joke about that right now"



