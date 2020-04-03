import argparse
import torch
from  betterberthumorclassifier import evaluate
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BertTokenizer, BertForSequenceClassification
MODELPATH = '/home/anish/projects/humor/trained_models/gpt2_medium_joker_5004.pt'
MODELPATH = '/home/anish/projects/humor/trained_models/bettertrainbert_medium_joker_10066.pt'
parser = argparse.ArgumentParser()
parser.add_argument("--traindataset", default=None, type=str, required=True) 
parser.add_argument("--evaldataset", default=None, type=str, required=True)
parser.add_argument("--outputfile", default=None, type=str, required=True)
parser.add_argument("--epochs", default=5, type=int, required=True)
parser.add_argument("--gradient_acums", default=6, type=int, required=True)
parser.add_argument("--maxseqlen", default=500, type=int, required=True)
parser.add_argument("--batch", default=6, type=int, required=True)
args = parser.parse_args()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
#model = GPT2LMHeadModel.from_pretrained('bert-based-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased')

model.load_state_dict(torch.load(MODELPATH))




model.eval()
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

evaluate(args, model, tokenizer)



