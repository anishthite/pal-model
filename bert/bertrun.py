import torch
import argparse
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class HumorDetector:
    def __init__(self, modelpath):
        
        
        model_state_dict = torch.load(modelpath, map_location=torch.device('cpu'))

        self.model = BertForSequenceClassification.from_pretrained(None, config= 'bert.json', state_dict=model_state_dict)
        
        #self.model.load_state_dict(torch.load(modelpath))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<|endoftext|>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        assert self.tokenizer.sep_token == '<SEP>'
        assert self.tokenizer.eos_token == '<|endoftext|>'

        self.model = self.model.to(device)
        self.model.train()
            
    def __call__(self, query):
        return self.predict(query)

    def predict(self, query):
        #encode
        tokens = self.tokenizer.encode(query, max_length=100, pad_to_max_length = True)
        inputs = torch.tensor([tokens], dtype=torch.long)
        inputs = inputs.to(device)
        #predict
        with torch.no_grad():
            output = self.model(inputs)
            #print(torch.sigmoid(output[0]))
            prediction = np.argmax(np.round(torch.sigmoid(output[0]).cpu()), axis=1)
        return prediction, torch.max(torch.sigmoid(output[0])) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/nethome/ilee300/Workspace/bettertrainbert_medium_joker_10066.pt', type=str, required=False)
    args = parser.parse_args()
    mymodel = HumorDetector(args.modelpath)
    mymodel.model.eval()
    while True:
        query = input("Enter joke or not joke: ")
        answer, probs = mymodel(query)
        if answer == 0:
            print("not joke :( probs " + str(probs))
        else:    
            print("Joke ;) probs " + str(probs))
