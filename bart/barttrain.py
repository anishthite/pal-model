import argparse
import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bart.modeling_bart import shift_tokens_right
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import os
import json
import csv
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')
from nltk.translate.bleu_score import sentence_bleu
import sys


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


TRAINPATH = '/home/anish/projects/humor/'

LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000


#tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
#model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

#tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
#model = GPT2LMHeadModel.from_pretrained('gpt2-large')

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

#tokenizer = GPT2TokenizerFast.from_pretrained('microsoft/DialoGPT-medium')
#model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')
#special_tokens_dict = {'sep_token': '<SEP>','bos_token': '<BOS>','pad_token': '<PAD>', 'eos_token': '<|endoftext|>'}
#tokenizer.add_special_tokens(special_tokens_dict)
#model.resize_token_embeddings(len(tokenizer))
#assert tokenizer.sep_token == '<SEP>'
#assert tokenizer.eos_token == '<|endoftext|>'

#model = torch.nn.DataParallel(model)

model = model.to(device)
class JokesDataset(Dataset):
    def __init__(self, dataset = 'humor_challenge_jokes_gpt2_better_qa_train.txt', block_size=512):
        super().__init__()

        self.joke_list = []
        self.examples = []
        self.labels = []
        with open(dataset) as csv_file:
            #lines = csv_file.readlines()
                         
            pad = tokenizer.pad_token_id
            #self.examples = tokenizer.encode_batch([line.rstrip() for line in lines])
            for line in tqdm(csv_file):
            #    self.joke_list.append(line)
                #tok_line = tokenizer.encode(line.rstrip().replace(' <|endoftext|>',''), padding="max_length",  max_length=block_size ) + [pad]
                begin = line.split(" <BOS> ")[0]
                label = line.split(" <BOS> ")[1]
                
                tok_begin = tokenizer(' ' + begin.rstrip().replace(' <|endoftext|>',''), 
                                      max_length=block_size,
                                      padding="max_length",
                                      return_tensors='pt')
                #if len(tok_begin) < block_size:
                #    tok_begin = tok_begin + [pad for _ in range(block_size-len(tok_begin))]
                #if len(tok_begin) > block_size:
                #    print("skipping")
                #    continue
                
                tok_label = tokenizer(' ' + label.rstrip().replace(' <|endoftext|>',''), 
                                      max_length=block_size,
                                      padding="max_length",
                                      return_tensors='pt')
                #if len(tok_label) < block_size:
                #    tok_label = tok_label + [pad for _ in range(block_size-len(tok_label))]
                #if len(tok_label) > block_size:
                #    print("skipping")
                #    continue
                if list(tok_begin['input_ids'].size()) != [1,200]:
                    continue
                if list(tok_label['input_ids'].size()) != [1,200]:
                    continue
                self.examples.append(tok_begin)
                self.labels.append(tok_label)
#        text = ''.join(self.joke_list)
        print(self.examples[:2])
        sys.stdout.flush()

#        tokenized_text = tokenizer.encode(text)
        

#        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
#            self.examples.append(tokenized_text[i : i + block_size])
        
        self.joke_list = []
        tokenized_text = ''
        text = ''

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (self.examples[item]['input_ids'], self.examples[item]['attention_mask'], self.labels[item]['input_ids'])

model.train()


models_folder = "models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

def bleu_evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = JokesDataset(dataset = args.evaldataset, block_size=args.maxseqlen)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch)

    # Eval!
    print("***** Running bleu evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.gradient_acums)
    bleu = 0.0
    nb_eval_steps = 0
    model.eval()

    bad_jokes = 0
    for idx,joke in enumerate(eval_dataloader):
        print(str(idx) + ' ' + str(len(eval_dataloader)))
        inputs, labels = (joke[0], joke[2])
        #inputs = tokenizer.decode(inputs.tolist()[0])
        #inputs_list  = inputs.split(' <BOS>')
        #if len(inputs_list) < 2:
        #    bad_jokes +=1
        #    continue
        #inputs = inputs_list[0] + ' <BOS>'
        #labels = inputs_list[1]
        #inputs = torch.tensor([tokenizer.encode(inputs)])
        with torch.no_grad():
            #output = model.generate(inputs, top_p=0.9, max_length=200, do_sample=True)
            
            inputs = torch.reshape(inputs, (1,200)).to(device)
            labels= torch.reshape(labels, (1,200))
            
            output = model.generate(inputs, max_length=200)
            output = tokenizer.decode(output.tolist()[0])
            #eosindex = labels.find('<|endoftext|>')
            #if eosindex != -1:
            #    labels = labels[:eosindex]

            labels = tokenizer.decode(labels.tolist()[0])
            
            bosindex = output.rfind('<s>')
            if bosindex != -1:
                output = output[bosindex+4:].strip()
            
            eosindex = output.find('</s>')
            if eosindex != -1:
                output = output[:eosindex].strip()
            
            bosindex = labels.rfind('<s>')
            if bosindex != -1:
                labels = labels[bosindex+4:].strip()
            
            eosindex = labels.find('</s>')
            if eosindex != -1:
                labels = labels[:eosindex].strip()
            
            print(labels)
            print(output)
            bleu += sentence_bleu([labels], output)
        sys.stdout.flush()
        
    print(bad_jokes)
    bleu = bleu / (len(eval_dataset)-bad_jokes)
    print(bleu)
    result = {"bleu": bleu}
    
    with open(args.outputfile, "a") as writer:
        writer.write(str(args.maxseqlen) + str(args.gradient_acums) + str(result))


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = JokesDataset(dataset = args.evaldataset, block_size=args.maxseqlen)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.gradient_acums)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for idx,joke in list(enumerate(eval_dataloader))[:-1]:
        print(str(idx) + ' ' + str(len(eval_dataloader)))
        inputs, mask, labels = (joke[0], joke[1], joke[2])
        #print(inputs)
        #print(labels)
        #inputs = inputs.to(device)
        #mask = mask.to(device)
        #labels = labels.to(device)
        inputs = torch.reshape(inputs, (1,200)).to(device)
        mask = torch.reshape(mask, (1,200)).to(device)
        labels = torch.reshape(labels, (1,200)).to(device)
        print(labels.size())
        decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id, tokenizer.bos_token_id).to(device)
        with torch.no_grad():
            outputs = model.forward(inputs, attention_mask=mask, decoder_input_ids=decoder_input_ids, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
            #eval_loss += lm_loss
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    with open(args.outputfile, "a") as writer:
        writer.write(str(args.maxseqlen) + str(args.gradient_acums) + str(result))
    return result

def train(args, model, tokenizer):

    dataset = JokesDataset(dataset=args.traindataset, block_size=args.maxseqlen)
    joke_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    t_total = len(joke_loader) // args.gradient_acums * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = t_total)
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    models_folder = args.outputdir
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    for epoch in range(args.epochs):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx,joke in tqdm(list(enumerate(joke_loader))[:-1], total=len(joke_loader)):
            #print(str(idx) + ' ' + str(len(joke_loader)))
            sys.stdout.flush()
            inputs, mask, labels = (joke[0], joke[1], joke[2])
            inputs = torch.squeeze(inputs).to(device)
            mask = torch.squeeze(mask).to(device)
            labels = torch.squeeze(labels).to(device)
            #torch.set_printoptions(threshold=50000)
            #print(joke)
            #print(joke.shape)
            decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id, tokenizer.bos_token_id).to(device)
            outputs = model.forward(inputs, attention_mask=mask, decoder_input_ids=decoder_input_ids, labels=labels)
            loss, logits = outputs[:2]
            loss = loss / args.gradient_acums
            loss.backward()
            sum_loss = sum_loss + loss.detach().data
                        
            #proc_seq_count = proc_seq_count + 1
            #if proc_seq_count == args.gradient_acums:
            #    proc_seq_count = 0    
            batch_count += 1
            if (idx + 1) % args.gradient_acums == 0:
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0
        
        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(models_folder, f"{models_folder}_{args.maxseqlen}{args.gradient_acums}{args.batch}{epoch}.pt"))
        model.save_pretrained(models_folder)
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindataset", default=None, type=str, required=True) 
    parser.add_argument("--evaldataset", default=None, type=str, required=True)
    parser.add_argument("--outputfile", default=None, type=str, required=True)
    parser.add_argument("--outputdir", default=None, type=str, required=True)
    parser.add_argument("--epochs", default=5, type=int, required=True)
    parser.add_argument("--gradient_acums", default=6, type=int, required=True)
    parser.add_argument("--maxseqlen", default=500, type=int, required=True)
    parser.add_argument("--batch", default=6, type=int, required=False)
    args = parser.parse_args()
    train(args, model, tokenizer)
