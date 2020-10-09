import argparse
from transformers import BertTokenizer, BertForSequenceClassification
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
import numpy as np
from bertrun import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


TRAINPATH = '/home/anish/projects/humor/'

LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
model = model.to(device)


special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<|endoftext|>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer)) 
assert tokenizer.sep_token == '<SEP>'
assert tokenizer.eos_token == '<|endoftext|>'

class JokesDataset(Dataset):
    def __init__(self, dataset = 'humor_challenge_jokes_gpt2_better_qa_train.txt', block_size=512):
        super().__init__()

        self.joke_list = []
        self.examples = []
        self.labels = []

        with open(dataset) as csv_file:
            for line in csv_file:
                try:
                    myline, label = line.split('\t')
                    label = float(label)
                    
                except:
                    continue
                self.examples.append(tokenizer.encode(myline, max_length=block_size, pad_to_max_length = True))
                self.labels.append(label)
        print("done")
        # text = ''.join(self.joke_list)

        # tokenized_text = tokenizer.encode(text)
        # for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
        #     self.examples.append(tokenized_text[i : i + block_size])
        
        # self.joke_list = []
        # tokenized_text = ''
        # text = ''

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.labels[item], dtype=torch.long)

model = model.to(device)
model.train()


models_folder = "models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)



def evaluate(args, model, tokenizer, epoch_num, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = None
    if epoch_num == args.epochs:
        eval_dataset = JokesDataset(dataset = args.testdataset, block_size=args.maxseqlen)
    else:
        eval_dataset = JokesDataset(dataset = args.devdataset, block_size=args.maxseqlen)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.gradient_acums)
    eval_loss = 0.0
    acc = 0.0
    nb_eval_steps = 0
    model.eval()
    true_pos =true_neg=false_pos=false_neg = 0

    for idx,joke in enumerate(eval_dataloader):
        print(str(idx) + ' ' + str(len(eval_dataloader)))
        inputs, labels = (joke[0], joke[1])
        #print(inputs)
        #print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            #outputs = model(inputs)
            prediction = np.argmax(np.round(torch.sigmoid(outputs[1]).cpu()), axis=1)
            acc += np.count_nonzero(prediction==labels.cpu())
            # print(prediction)
            # print(labels)
            # print(acc)
            for i in range(len(prediction)):
                if prediction[i] == 1:
                    if labels.cpu()[i] == 1:
                        true_pos +=1
                    else:
                        false_pos +=1
                else:
                    if labels.cpu()[i] == 1:
                        false_neg +=1
                    else:
                        true_neg+=1
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    acc = acc / len(eval_dataset)
    perplexity = torch.exp(torch.tensor(eval_loss))
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1 = 2*(precision * recall) / (precision + recall)
    result = {"eval_loss": eval_loss, "acc": acc}

    epoch_header = ""
    if epoch_num == args.epochs:
        epoch_header = "Final test  ||"
    else:
        epoch_header = "epoch[{}] ||".format(epoch_num)
    save_log = "{:<15}  maxSequence length : {} gradeint accumulation: {}, eval_loss: {}, accuracy: {}, precision : {}, recall : {}, f1 : {}\n\n".format(
        epoch_header, args.maxseqlen, args.gradient_acums, eval_loss,acc,precision,recall,f1
    )
    
    print(save_log)
    with open(args.outputfile, "a") as writer:
        # writer.write(str(args.maxseqlen) + str(args.gradient_acums) + str(result))
        writer.write(save_log)
        writer.close()
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
    models_folder = "../trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    for epoch in range(args.epochs):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        total_batch = len(joke_loader)
        for idx,joke in enumerate(joke_loader):
            #print(str(idx) + ' ' + str(len(joke_loader)))
            inputs, labels = (joke[0], joke[1])
            #print(inputs)
            #print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            #torch.set_printoptions(threshold=50000)
            #print(joke)be
            #print(joke.shape)
            outputs = model(inputs, labels=labels)
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
            if batch_count % 100 == 0:
                print(f"batch {batch_count :<4} / {total_batch :<4} :\t sum loss {sum_loss}")
                # batch_count = 0
                sum_loss = 0.0

        evaluate(args, model, tokenizer,epoch)

        # Store the model after each epoch to compare the performance of them
        if(epoch == args.epochs - 1):
            evaluate(args, model, tokenizer,epoch+1)

            model.config.save_pretrained(models_folder)
            torch.save(model.state_dict(), os.path.join(models_folder, f"newsave.pt"))
            model.save_pretrained(models_folder)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindataset", default=None, type=str, required=True) 
    parser.add_argument("--devdataset", default=None, type=str, required=True)
    parser.add_argument("--testdataset", default=None, type=str, required=True)

    parser.add_argument("--outputfile", default=None, type=str, required=True)
    parser.add_argument("--epochs", default=5, type=int, required=True)
    parser.add_argument("--gradient_acums", default=6, type=int, required=False)
    parser.add_argument("--maxseqlen", default=500, type=int, required=False)
    parser.add_argument("--batch", default=6, type=int, required=False)
    parser.add_argument('--pretrained', default=False, action='store_true', help='Bool type')
    args = parser.parse_args()
    if (args.pretrained) :
        model_path = "/nethome/ilee300/Workspace/pal-model/trained_models/bettertrainbert_medium_joker_50016.pt"
        model = HumorDetector(model_path).model
    train(args, model, tokenizer)