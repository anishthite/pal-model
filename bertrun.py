import torch
import argparse
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification



device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class HumorDetector:
    def __init__(self, modelpath):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-cased')
        self.model.load_state_dict(torch.load(modelpath))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = model.to(device)
        model.eval()
            
    def __call__(self, query):
        return self.predict(query)

    def predict(self, query):
        #encode
        inputs = torch.tensor(self.tokenizer.encode(query, pad_to_max_length = True))
        inputs.to(device)
        #predict
        with torch.no_grad():
            output = self.model(inputs)
            prediction = np.argmax(np.round(torch.sigmoid(output).cpu()), axis=1)
        return prediction



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default='/home/anish/projects/humor/trained_models/bettertrainbert_medium_joker_10066.pt', type=str, required=True)
    args = parser.parse_args()
    mymodel = HumorDetector(args.modelpath)
    while True:
        query = input("Enter joke or not joke: ")
        answer = mymodel(query)
        if answer == 0:
            print("not joke :(")
        print("Joke ;)")
