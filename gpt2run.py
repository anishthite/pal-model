import torch
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel


#device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class HumorGenGPT:
    def __init__(self, modelpath):
        
        model_state_dict = torch.load(modelpath)
        self.model = GPT2LMHeadModel.from_pretrained(None, config= 'trained_models/gpt2config.json', state_dict=model_state_dict)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

        special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<|endoftext|>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        assert self.tokenizer.sep_token == '<SEP>'
        assert self.tokenizer.eos_token == '<|endoftext|>'

        self.model = self.model.to(device)
        self.model.eval()
            
    def __call__(self, query):
        return self.predict(query)

    def predict(self, query):
        #encode
        query = query + ' <SEP> '
        print(query)
        tokens = self.tokenizer.encode(query)
        print(tokens)
        inputs = torch.tensor([tokens], dtype=torch.long)
        print(inputs)
        inputs = inputs.to(device)
        #predict
        with torch.no_grad():
            output = self.model.generate(inputs, top_k=50)
        return self.tokenizer.decode(output.tolist()[0]) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/home/anishthite/Workspace/pal-model/models/bettertraingpt2_medium_joker_10066.pt', type=str, required=False)
    args = parser.parse_args()
    mymodel = HumorGenGPT(args.modelpath)
    while True:
        query = input("Enter Question: ")
        answer = mymodel(query)
        print(answer)
