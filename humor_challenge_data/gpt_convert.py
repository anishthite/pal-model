import csv
from sklearn.model_selection import train_test_split
 #your data

badwords = ['nigger', 'fuck', 'sex', 'negro', 'cunt', 'bitch', 'cum', 'gay', 'dyke', 'fag', 'faggot']
redditwords = ['Edit:', 'edit:']


def check_bad_word(badwordlist, sentence):
    sentencecopy = sentence.split(' ')
    for word in badwords:
        if word in sentencecopy:
            return False
    return True


jokelist = []
with open("./raw/humor_challenge_jokes.csv", 'r') as infile:
    for line in infile:
        line = line.replace('\n',' <|endoftext|> \n')
        if line.count('?') == 1 and len(line) < 200 and check_bad_word(badwords, line):
            newline = ''.join([line[:line.index('?') + 1], ' <SEP> ', line[line.index('?') + 1:]])
            if newline not in jokelist:
                jokelist.append(newline)
with open("./raw/qajokes_cleaned.tsv") as cleanjokes:
    for line in cleanjokes:
        line = line.replace('\t', ' <SEP> ').replace('\n',' <|endoftext|> \n')
        if line not in jokelist and check_bad_word(badwords, line):
       	    jokelist.append(''.join([line]))
with open("./raw/rjokes.csv", 'r') as redditjokes:
    myreader = csv.reader(redditjokes)
    for row in myreader:
        row[1] = row[1].replace('\n','')
        row[2] = row[2].replace('\n','')
        for word in redditwords:
            if word in row[2]:
                row[2] = row[2][:row[2].index(word)]
        line = ''.join([row[1], ' <SEP> ', row[2], ' <|endoftext|> \n'])
        if line not in jokelist and check_bad_word(badwords, line) and line.count('?') == 1 and len(line) < 250:
            jokelist.append(line)


    train, test = train_test_split(jokelist, train_size=0.9, test_size=0.1)
    with open("jokes_cleaned_train.txt", 'w') as trainfile:
        trainfile.write(''.join(train))
    with open("jokes_cleaned_test.txt", 'w') as testfile:
        testfile.write(''.join(test))
    
    #i#reader = csv.reader(infile, dialect="excel")    
    #with open("qa_jokes_cleaned_gpt2.txt", mode="w") as outfile:
     #   writer = csv.writer(outfile, delimiter=str(' <|endoftext|> '))
      #  writer.writerows(reader)
