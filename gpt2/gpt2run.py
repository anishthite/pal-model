import torch
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel


#device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

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
            
    def __call__(self, query):
        return self.predict(query)

    def predict(self, query):
        #encode
        query = query + ' <BOS> '
        print(query)
        tokens = self.tokenizer.encode(query)
        print(tokens)
        inputs = torch.tensor([tokens], dtype=torch.long)
        print(inputs)
        #inputs = inputs.to(device)
        #predict
        with torch.no_grad():
            output = self.model.generate(inputs, top_k=50)
        output = self.tokenizer.decode(output.tolist()[0]) 
        eos_index = output.find('<|endoftext|>')
        if eos_index != -1:
            output = output[:eos_index]
        bos_index = output.find('<BOS>')
        if bos_index != -1:
            output = output[bos_index+5:]
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/home/tobias/humor/pal-model/gpt2/trained_models/gpt2_tokens_tag_10056.pt', type=str, required=False)
    args = parser.parse_args()
    mymodel = HumorGenGPT(args.modelpath)
    while True:
        query = input("Enter Question: ")
        answer = mymodel(query)
        print(answer)
