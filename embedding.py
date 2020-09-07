import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.downloader as api

class WordEmbedder:
    def __init__(self):
        #glove_file = datapath('/Users/irene/desktop/humor/pal-model/glove.6B/glove.6B.100d.txt')
        word2vec_glove_file = get_tmpfile("/Users/irene/desktop/humor/pal-model/glove.6B/glove.6B.100d.word2vec.txt")
        #glove2word2vec(glove_file, word2vec_glove_file)
        self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file)


    def findKeyword(self,strList):
        return self.model.most_similar(positive=strList)


if __name__ == "__main__":
    embedder = WordEmbedder()
    while True:
        query = input("Enter comma separated words (type q to exit):")
        if query == 'q':
            break
        strList = [ s.strip() for s in query.split(',')]
        print(strList)
        print(embedder.findKeyword(strList))