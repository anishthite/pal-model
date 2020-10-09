import torch
import argparse
import pickle
import pandas as pd
import numpy as np
from bertrun import *
from nltk.tokenize import word_tokenize


class AnchorFinder:
    def __init__(self, model_path):
        self.model = HumorDetector(model_path)
    
    def getScore(self,input):
        answer, probs = self.model(input)
        # if answer == 0:
        #     print("not joke :( probs " + str(probs))
        # else:    
        #     print("Joke ;) probs " + str(probs))
        if answer == 0:
            return -1
        else:
            return probs
    
    def findAnchors(self,sentence, threshold):
        tokens = word_tokenize(sentence)
        orgScore = self.getScore(sentence)
        #print("original score:", orgScore)
        anchorList = []
        basic_words = ["it", "a","The", "A", "."]

        for token in tokens:
            if token in basic_words:
                continue
            candSentence = sentence.replace(token,"")
            #print(candSentence)
            probs = self.getScore(candSentence)
            #print("token: ", token, "\n probability:", probs,"diff: ",(orgScore - probs),"\n\n")
            if orgScore - probs >= threshold:
                if token not in anchorList:
                    anchorList.append(token)

            # for token2 in tokens:
            #     candSentence = sentence.replace(token,"").replace(token2,"")
            #     print(candSentence)
            #     probs = self.getScore(candSentence)
            #     print("token: ", token, token2, "\n probability:", probs,"diff: ",(orgScore - probs),"\n\n")
            #     if orgScore - probs >= threshold:
            #         if token not in anchorList:
            #             anchorList.append(token)
            #         if token2 not in anchorList:
            #             anchorList.append(token2)
        return anchorList

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

    def add_item(self,anchor_list, joke):
        for anchor in anchor_list:
            if anchor in self.dictionary:
                self.dictionary[anchor] += [joke]
            else:
                self.dictionary[anchor] = [joke]
        #TODO: finish this method.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/nethome/ilee300/Workspace/pal-model/trained_models/bettertrainbert_medium_joker_50066.pt', type=str, required=False)
    parser.add_argument('--test', default=False, action='store_true', help='Bool type')
    parser.add_argument("--saveFile", default="anchorDictionary2.pkl",type=str,required=False)
    parser.add_argument("--dataFile", default="/nethome/ilee300/Workspace/pal-model/bert_train_data/all_jokes.csv",type=str,required=False)
    parser.add_argument("--threshold", default=0.1,type=float,required=False )

    args = parser.parse_args()
    finder = AnchorFinder(args.modelpath)
    anchor_dictionary = AnchorDictionary()
    print(args.test)
    if args.test:
        while True:
            query = input("Enter joke: ")
            if query == 'q':
                break
            anchorList = finder.findAnchors(query, args.threshold)
            print(anchorList)
            anchor_dictionary.add_item(anchorList,query)
            anchor_dictionary.save_file(args.saveFile)
        print(anchor_dictionary.dictionary)
    else:
        #dataset = open("../humor_challenge_data/bot_data/pun_list.txt", 'r')
        # for joke in dataset:
        #     if joke != "\n":
        #         print(joke)
        #     anchorList = finder.findAnchors(joke, args.threshold)
        #     print(anchorList)
        #     anchor_dictionary.add_item(anchorList,joke)
        # print(anchor_dictionary.dictionary)
        dataset = pd.read_csv(args.dataFile)
        jokeList = dataset['text']
        for joke in jokeList:
            if type(joke) !=str:
                continue
            anchorList = finder.findAnchors(joke, args.threshold)
            anchor_dictionary.add_item(anchorList,joke)
        anchor_dictionary.save_file(args.saveFile)
        print("Done")



