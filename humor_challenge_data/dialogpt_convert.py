import csv
from sklearn.model_selection import train_test_split
 #your data

jokelist = []
with open("./raw/humor_challenge_jokes.csv", 'r') as infile:
    for line in infile:
        line = line.replace('\n',' <|endoftext|> \n')
        if line.count('?') == 1 and len(line) < 200 and line not in jokelist:
            line = line.replace('?', ' <|endoftext|> \n')
            jokelist.append(line)
with open("./raw/qajokes_cleaned.tsv") as cleanjokes:
    for line in cleanjokes:
        line = line.replace('\t', ' ').replace('\n',' <|endoftext|> \n')
        if line not in jokelist:
            line = line.replace('?', ' <|endoftext|> \n')
       	    jokelist.append(line)
    train, test = train_test_split(jokelist, train_size=0.99, test_size=0.01)
    with open("humor_dialogpt.txt", 'w') as trainfile:
        trainfile.write(''.join(train))