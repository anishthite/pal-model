import csv
from sklearn.model_selection import train_test_split
 #your data

jokelist = []
with open("humor_challenge_jokes.csv", 'r') as infile:
    for line in infile:
        jokelist.append(''.join([line.replace('\n',' <|endoftext|> ')]))
    train, test = train_test_split(jokelist, train_size=0.9, test_size=0.1)
    with open("humor_challenge_jokes_gpt2_train.txt", 'w') as trainfile:
        trainfile.write(''.join(train))
    with open("humor_challenge_jokes_gpt2_test.txt", 'w') as testfile:
        testfile.write(''.join(test))
    
    
    #i#reader = csv.reader(infile, dialect="excel")    
    #with open("qa_jokes_cleaned_gpt2.txt", mode="w") as outfile:
     #   writer = csv.writer(outfile, delimiter=str(' <|endoftext|> '))
      #  writer.writerows(reader)
