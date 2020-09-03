import argparse
import torch
from transformers import BertTokenizer
from transformers import EncoderDecoderModel
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

LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000


tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-large-cased', 'bert-large-cased')
model = model.to(device)

special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<|endoftext|>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.encoder.resize_token_embeddings(len(tokenizer))
model.decoder.resize_token_embeddings(len(tokenizer))
assert tokenizer.sep_token == '<SEP>'
assert tokenizer.eos_token == '<|endoftext|>'


class JokesDataset(Dataset):
    def __init__(self, dataset = 'humor_challenge_jokes_gpt2_better_qa_train.txt', block_size=512):
        super().__init__()

        self.examples = []
        self.labels = []

        with open(dataset) as csv_file:
            for line in csv_file:
                splitindex = line.index('<SEP>')
                tokenized_input = tokenizer.encode(line[:splitindex], max_length=block_size, pad_to_max_length = True, add_special_tokens=False)
                tokenized_label = tokenizer.encode(line[splitindex:], max_length=block_size, pad_to_max_length = True, add_special_tokens=False)
                self.examples.append(tokenized_input)
                self.labels.append(tokenized_label)           
        tokenized_input = ''
        text = ''

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.labels[item])

#model = model.to(device)
model.train()


models_folder = "models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)



def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = JokesDataset(dataset = args.evaldataset, block_size=args.maxseqlen)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.gradient_acums)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for idx,joke in enumerate(eval_dataloader):
        print(str(idx) + ' ' + str(len(eval_dataloader)))
        inputs, labels = (joke[0], joke[1])
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs, decoder_input_ids=labels, lm_labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
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
    models_folder = "combinerslargeencoder"
    models_folder2 = "combinerslargedecoder"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    if not os.path.exists(models_folder2):
        os.mkdir(models_folder2)
    for epoch in range(args.epochs):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx,joke in enumerate(joke_loader):
            print(str(idx) + ' ' + str(len(joke_loader)))
            #joke = joke.to(device)
            #torch.set_printoptions(threshold=50000)
            #print(joke)
            #print(joke.shape)
            inputs, labels = (joke[0], joke[1])
            #print(inputs)
            #print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=inputs, decoder_input_ids=labels, lm_labels=labels)
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
        torch.save(model.state_dict(), os.path.join(models_folder, f"combined_joker_{args.maxseqlen}{epoch}{args.gradient_acums}.pt"))
        model.save_pretrained(models_folder)
        model.encoder.save_pretrained(models_folder)
        model.decoder.save_pretrained(models_folder2)
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindataset", default=None, type=str, required=True) 
    parser.add_argument("--evaldataset", default=None, type=str, required=True)
    parser.add_argument("--outputfile", default=None, type=str, required=True)
    parser.add_argument("--epochs", default=5, type=int, required=True)
    parser.add_argument("--gradient_acums", default=6, type=int, required=True)
    parser.add_argument("--maxseqlen", default=500, type=int, required=True)
    parser.add_argument("--batch", default=6, type=int, required=True)
    args = parser.parse_args()
    train(args, model, tokenizer)
