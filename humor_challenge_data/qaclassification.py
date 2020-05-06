import csv
import random
from sklearn.model_selection import train_test_split
 #your data

jokelist = []
with open("./raw/humor_challenge_jokes.csv", 'r') as infile:
    for line in infile:
        line = line.replace('\n',' <|endoftext|>')
        if line.count('?') == 1 and len(line) < 200 and line not in jokelist:
            jokelist.append(''.join([line[:line.index('?') + 1], ' <SEP> ', line[line.index('?') + 1:], '\t', '1.0', '\n']))
with open("./raw/qajokes_cleaned.tsv") as cleanjokes:
    for line in cleanjokes:
        line = line.replace('\t', ' <SEP> ').replace('\n',' <|endoftext|>')
        if line not in jokelist:
       	    jokelist.append(''.join([line, '\t', '1.0', '\n']))

with open("./raw/negative_examples_adj.tsv", 'r') as negfile:
    for line in negfile:
        line = line.replace('\t', ' <SEP> ').replace('\n',' <|endoftext|>')
        if line not in jokelist:
            jokelist.append(''.join([line, '\t', '0.0', '\n']))
    random.shuffle(jokelist)
    train, test = train_test_split(jokelist, train_size=0.9, test_size=0.1)


    with open("qa_classification_train_tokens.txt", 'w') as trainfile:
        trainfile.write(''.join(train))
    with open("qa_classification_test_tokens.txt", 'w') as testfile:
        testfile.write(''.join(test))
    
    # import pandas as pd

    # traindf = pd.read_csv("humor_challenge_jokes_qa_classification_train.txt", sep='\t')
    # testdf = pd.read_csv("humor_challenge_jokes_qa_classification_test.txt", sep='\t')

    #i#reader = csv.reader(infile, dialect="excel")    
    #with open("qa_jokes_cleaned_gpt2.txt", mode="w") as outfile:
     #   writer = csv.writer(outfile, delimiter=str(' <|endoftext|> '))
      #  writer.writerows(reader)
