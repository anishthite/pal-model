from transformers import BertTokenizer, BertForSequenceClassification
import torch

import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


TRAINPATH = '/home/anish/projects/humor/'


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
model = model.to(device)

class JokesDataset(Dataset):
    def __init__(self, dataset = 'humor_challenge_jokes_gpt2_better_qa_train.txt'):
        super().__init__()

        # short_jokes_path = os.path.join(jokes_dataset_path, dataset)

        self.joke_list = []
        self.labellist = []
        self.end_of_text_token = "<|endoftext|>"
        
        with open(dataset) as csv_file:
            for line in csv_file:
                myline, label = line.split('\t')
                self.joke_list.append(myline)
                self.labellist.append(label)
        
    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item], self.labellist[item]


#BATCH_SIZE = 6
#EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
#MAX_SEQ_LEN = 500

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


model = model.to(device)
model.train()

tmp_jokes_tens = None
models_folder = "models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = JokesDataset(dataset = args.evaldataset)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.gradient_acums)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.gradient_acums)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for idx,joke in enumerate(eval_dataloader):
        print(str(idx) + ' ' + str(len(eval_dataloader)))
        joke_tens = torch.tensor(tokenizer.encode(joke[0], add_special_tokens=True)).unsqueeze(0).to(device)
        label_tens = torch.tensor(float(joke[1][0][:-2])).to(device)
        inputs, labels = (joke_tens, label_tens)
        #print(inputs)
        #print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"loss": torch.tensor(eval_loss)}

    with open(args.outputfile, "a") as writer:
        writer.write(str(args.maxseqlen) + str(args.gradient_acums) + str(result))
    return result

def train(args, model, tokenizer):

    dataset = JokesDataset(dataset=args.traindataset)
    joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    tmp_jokes_tens = None
    models_folder = "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    for epoch in range(args.epochs):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx,joke in enumerate(joke_loader):
            print(str(idx) + ' ' + str(len(joke_loader)))
            joke_tens = torch.tensor(tokenizer.encode(joke[0][0], add_special_tokens=True)).unsqueeze(0).to(device)
            label_tens = torch.tensor(float(joke[1][0][:-2])).to(device)
            if joke_tens.size()[1] > args.maxseqlen:
                continue
            if not torch.is_tensor(tmp_jokes_tens):
                tmp_jokes_tens = joke_tens
                tmp_label_tens = label_tens.reshape(1)
                continue
            else:
                #The next joke does not fit in so we process the sequence and leave the last joke 
                #as the start for next sequence 
                if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > args.maxseqlen:
                    work_jokes_tens = tmp_jokes_tens
                    tmp_jokes_tens = joke_tens
                    
                    work_label_tens = tmp_label_tens
                    tmp_label_tens = label_tens 
                else:
                    # Add the joke to sequence, continue and try to add more
                    tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:,1:]], dim=1)
                    tmp_label_tens = torch.cat([tmp_label_tens, label_tens.reshape(1)])
                    #print(joke[0][0])
                    #print(joke_tens)
                    continue
            ################## Sequence ready, process it trough the model ##################
            print(work_jokes_tens.shape)
            print(work_label_tens.shape)
            outputs = model(work_jokes_tens, labels=work_label_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data
                        
            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == args.gradient_acums:
                proc_seq_count = 0    
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0
        
        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_joker_{args.maxseqlen}{epoch}{args.gradient_acums}.pt"))
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindataset", default=None, type=str, required=True) 
    parser.add_argument("--evaldataset", default=None, type=str, required=True)
    parser.add_argument("--outputfile", default=None, type=str, required=True)
    parser.add_argument("--epochs", default=5, type=int, required=True)
    parser.add_argument("--gradient_acums", default=6, type=int, required=True)
    parser.add_argument("--maxseqlen", default=500, type=int, required=True)
    args = parser.parse_args()
    train(args, model, tokenizer)