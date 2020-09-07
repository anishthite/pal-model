import torch
import argparse
import pickle
import numpy as np
from bertrun import *
from nltk.tokenize import word_tokenize


class AnchorFinder:
    def __init__(self, model_path):
        self.model = HumorDetector(model_path)
    
    def getScore(input):
        answer, probs = mymodel(input)
        if answer==0:
            return 0
        else:
            return probs
    
    def findAnchors(sentence, threshold):
        tokens = word_tokenize(sentence)
        orgScore = getScore(sentence)
        anchorList = []
        for token in tokens:
            candSentence = sentence.lstrip(tokens).trim()
            probs = getScore(candSentence)
            if orgScore - probs >= threshold:
                anchorList.append(token)

class AnchorDictionary:
    def __init__(self, saved_file= None):
        self.dictionary = dict()
        if saved_file is not None:
            dict_file = open(saved_file,"rb")
            self.dictionary = pickle.load(dict_file)
            dict_file.close()
    
    def save_file(self,file_name):
        dict_file = open(file_name,"wb")
        pickle.dump(self.dictionary, dict_file)
        dict_file.close()

    def add_item(anchor_list, joke):
        #TODO: finish this method.



