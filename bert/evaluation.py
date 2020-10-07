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

model_path = "/nethome/ilee300/Workspace/bettertrainbert_medium_joker_10066.pt"
model = HumorDetector(model_path).model

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.labels[item], dtype=torch.long)

model = model.to(device)
model.eval()





def evaluate(args, model, tokenizer, prefix=""):
    print("evaluation started")
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = JokesDataset(dataset = args.evaldataset, block_size=args.maxseqlen)
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
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0


    for idx,joke in enumerate(eval_dataloader):
        print(str(idx) + ' ' + str(len(eval_dataloader)))
        inputs, labels = (joke[0], joke[1])
        #print(inputs)
        #print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            prediction = np.argmax(np.round(torch.sigmoid(outputs[1]).cpu()), axis=1)
            acc += np.count_nonzero(prediction==labels.cpu())
            for i in range(len(prediction)):
                if prediction[i] == 1:
                    if labels.cpu()[i] == 1:
                        true_positive +=1
                    else:
                        false_positive +=1
                else:
                    if labels.cpu()[i] == 1:
                        false_positive +=1
                    else:
                        true_negative+=1

            # print(prediction)
            # print(labels)
            # print(true_positive, false_positive, true_negative, false_negative, sep="   ")
            # print(acc)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    acc = acc / len(eval_dataset)
    perplexity = torch.exp(torch.tensor(eval_loss))
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1_score =(2* precision * recall)/(precision+recall)
    result = {"eval_loss": eval_loss, "acc": acc, "f1 score" : f1_score}
    print("evaluation done. Saving the output")
    with open(args.outputfile, "a") as writer:
        writer.write(str(args.maxseqlen) + " " + str(args.gradient_acums) + str(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaldataset", default=None, type=str, required=True)
    parser.add_argument("--outputfile", default=None, type=str, required=True)
    parser.add_argument("--gradient_acums", default=6, type=int, required=False)
    parser.add_argument("--maxseqlen", default=500, type=int, required=False)
    parser.add_argument("--batch", default=6, type=int, required=False)
    parser.add_argument('--pretrained', default=False, action='store_true', help='Bool type')
    args = parser.parse_args()
    # if (args.pretrained) :
    #     model_path = "/nethome/ilee300/Workspace/bettertrainbert_medium_joker_10066.pt"
    #     model = HumorDetector(model_path).model
    # train(args, model, tokenizer)
    evaluate(args, model, tokenizer)
