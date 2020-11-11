import torch
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from profanity_filter import ProfanityFilter
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Model
import tensorflow
import pandas as pd

#device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

import toxicity

class HumorGenGPT:
    def __init__(self, modelpath):
        
        model_state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
        self.model = GPT2LMHeadModel.from_pretrained(None, config= '/home/tobias/humor/pal-model/gpt2/trained_models/gpt2config.json', state_dict=model_state_dict)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        special_tokens_dict = {'sep_token': '<SEP>','bos_token': '<BOS>','pad_token': '<PAD>', 'eos_token': '<|endoftext|>'}

        #special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<|endoftext|>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        assert self.tokenizer.sep_token == '<SEP>'
        assert self.tokenizer.eos_token == '<|endoftext|>'

        #self.model = self.model.to(device)
        self.model.eval()
        self.pf = ProfanityFilter()
        self.tokenizer = toxicity.tknzr()
        matrix = toxicity.embeddingmatrix(self.tokenizer)
        self.toxicity_model = toxicity.NeuralNet(matrix, 7)
        self.toxicity_model.load_state_dict(torch.load('model epoch:3.pt'), map_location=device)
        self.toxicity_model.eval()

            
    def __call__(self, query, **kwargs):
        return self.predict(query, **kwargs)

    def predict(self, query, **kwargs):
        
        #encode
        query = query + ' <BOS> '
        #print(query)
        tokens = self.tokenizer.encode(query)
        #print(tokens)
        inputs = torch.tensor([tokens], dtype=torch.long)
        #print(inputs)
        #inputs = inputs.to(device)
        #predict
        for i in range(2):
            with torch.no_grad():
                output = self.model.generate(inputs, **kwargs) #temp 0.8
            output = self.tokenizer.decode(output.tolist()[0]) 
            eos_index = output.find('<|endoftext|>')
            if eos_index != -1:
                output = output[:eos_index]
            bos_index = output.find('<BOS>')
            if bos_index != -1:
                output = output[bos_index+5:]
            # if self.pf.is_clean(output) and output.find('<BOS>') == -1 and output.find('<SEP>') == -1:
            #     return output
            # model = tensorflow.keras.models.load_model('models/model2')
            # if np.round(model.predict(sequence.pad_sequences(tokenizer.texts_to_sequences([['bitch']]), maxlen=maxlen))).flatten()[0]<.5:
                # return output
            MAX_LEN = 220
            if toxicity.sigmoid(self.toxicity_model(torch.tensor(sequence.pad_sequences(self.tokenizer.texts_to_sequences(np.array([toxicity.clean_special_chars(output)])), MAX_LEN), dtype=torch.long).numpy()))[0][0]>0.5:
                return output
        return "None Generated!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/home/tobias/humor/pal-model/gpt2/trained_models/gpt2_tokens_tag_10056.pt', type=str, required=False)
    args = parser.parse_args()
    print(args.modelpath)
    mymodel = HumorGenGPT(args.modelpath)
    while True:
        query = input("Enter Question: ")
        answer = mymodel(query, do_sample=True)
        print(answer)
        answer = mymodel(query)
        print(answer)
