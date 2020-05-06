import torch
import argparse
import numpy as np
from transformers import BertTokenizer
from transformers import EncoderDecoderModel
import torch
from torch.nn import functional as F

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class HumorGenBart:
    def __init__(self, modelpath):
        
        
        model_state_dict = torch.load(modelpath)
        
        #self.model = EncoderDecoderModel.from_pretrained(None, config= 'trained_models/combined.json', state_dict=model_state_dict)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('combinersencoder','combinersdecoder')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<|endoftext|>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))
        assert self.tokenizer.sep_token == '<SEP>'
        assert self.tokenizer.eos_token == '<|endoftext|>'

        self.model = self.model.to(device)
        self.model.eval()
            
    def __call__(self, query):
        return self.predict(query)

    def predict(self, query):
        #encode
        query = query + ' <SEP> '
        tokens = self.tokenizer.encode(query, max_length=100)
        inputs = torch.tensor([tokens], dtype=torch.long)
        inputs = inputs.to(device)
        #predict
        print(query)
        print(tokens)
        with torch.no_grad():


            outputs = self.model(input_ids=inputs, decoder_input_ids=inputs)
            print(outputs)
            print(self.tokenizer.decode(torch.argmax(F.softmax(outputs[0], dim=-1), dim=-1)[0]))
            output = self.model.generate(inputs, decoder_start_token_id=self.model.config.decoder.pad_token_id, top_p=0.9)
        return self.tokenizer.decode(output.tolist()[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/home/anish/projects/humor/trained_models/bettertraingpt2_medium_joker_10066.pt', type=str, required=False)
    args = parser.parse_args()
    mymodel = HumorGenBart(args.modelpath)
    while True:
        query = input("Enter Question: ")
        answer = mymodel(query)
        print(answer)
