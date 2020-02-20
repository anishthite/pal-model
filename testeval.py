import torch
from simpletrain import evaluate
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODELPATH = '/home/anish/projects/humor/trained_models/gpt2_medium_joker_5004.pt'


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.load_state_dict(torch.load(MODELPATH))
model.eval()
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

evaluate(model, tokenizer)
