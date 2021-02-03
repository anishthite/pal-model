import torch
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from profanity_filter import ProfanityFilter
#from profanity_check import predict, predict_prob
import profanity_check as pc

import pickle
from pytorch_pretrained_bert import BertConfig, BertForSequenceClassification

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class HumorGenGPT:
    def __init__(self, modelpath):
        
        model_state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
        self.model = GPT2LMHeadModel.from_pretrained(None, config= 'trained_models/gpt2config.json', state_dict=model_state_dict)
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
        with open('models/bert-toxicity/bert_tokenizer.pickle', 'rb') as handle:
            self.toxicity_tokenizer = pickle.load(handle)
        # device2 = torch.device(device)
        bert_config = BertConfig('models/bert-toxicity/bert_config.json')
        self.toxicity_model = BertForSequenceClassification(bert_config, num_labels=1)
        self.toxicity_model.load_state_dict(torch.load("models/bert-toxicity/bert_pytorch.bin", map_location=torch.device('cpu')))
        self.toxicity_model.to(torch.device(device))
        for param in self.toxicity_model.parameters():
            param.requires_grad = False
        self.toxicity_model.eval()

        
            
    def __call__(self, query, **kwargs):
        return self.predict(query, **kwargs)

    def predict(self, query, **kwargs):
        
        #encode
        if pc.predict([query])[0] ==1
            return "Joke is not appropriate"
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
            all_tokens = []
            longer = 0
            max_seq_length =220-2
            tokens_a = self.toxicity_tokenizer.tokenize(output)
            if len(tokens_a)>max_seq_length:
                    tokens_a = tokens_a[:max_seq_length]
                    longer += 1
            one_token = self.toxicity_tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
            all_tokens.append(one_token)

            if torch.sigmoid(self.toxicity_model(torch.tensor(np.array(all_tokens)).to(device), attention_mask=(torch.tensor(np.array(all_tokens)).to(device) > 0), labels=None))[0][0].item()<=.5:
                return output


        return "Sorry I don't have a joke about that right now"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='trained_models/gpt2_tokens_tag_10056.pt', type=str, required=False)
    args = parser.parse_args()
    print(args.modelpath)
    mymodel = HumorGenGPT(args.modelpath)
    while True:
        query = input("Enter Question: ")
        answer = mymodel(query, do_sample=True)
        print(answer)
        answer = mymodel(query)
        print(answer)
