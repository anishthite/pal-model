import argparse
import torch
from  bettertrain import bleu_evaluate, evaluate
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BertTokenizer, BertForSequenceClassification
MODELPATH = '/home/anish/projects/humor/trained_models/gpt2_tokens_tag_10056.pt'
parser = argparse.ArgumentParser()
parser.add_argument("--traindataset", default=None, type=str, required=True) 
parser.add_argument("--evaldataset", default=None, type=str, required=True)
parser.add_argument("--outputfile", default=None, type=str, required=True)
parser.add_argument("--epochs", default=5, type=int, required=True)
parser.add_argument("--gradient_acums", default=6, type=int, required=True)
parser.add_argument("--maxseqlen", default=500, type=int, required=True)
parser.add_argument("--batch", default=6, type=int, required=True)
parser.add_argument("--model", default=None, type=str, required=True)
parser.add_argument("--config", default=None,  type=str, required=True)
args = parser.parse_args()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
#model = GPT2LMHeadModel.from_pretrained('bert-based-cased')

model_state_dict = torch.load(args.model)

model = GPT2LMHeadModel.from_pretrained(None, config= args.config, state_dict=model_state_dict)
#model.load_state_dict(torch.load(MODELPATH))



model.eval()
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
special_tokens_dict = {'sep_token': '<SEP>','bos_token': '<BOS>','pad_token': '<PAD>', 'eos_token': '<|endoftext|>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
assert tokenizer.sep_token == '<SEP>'
assert tokenizer.eos_token == '<|endoftext|>'


evaluate(args, model, tokenizer)


