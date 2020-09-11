import pandas as pd
import nltk
DATASET = 'humor_challenge_data/bot_data/non_qa_total.csv'
dataset = pd.read_csv(DATASET)
def f(x):
    return nltk.word_tokenize(str(x[0]).lower()) + nltk.word_tokenize(str(x[1]).lower())
# tokenized_tile = dataset['title'].apply(lambda x: nltk.word_tokenize(str(x)))
# tokenized_selftext = dataset['selftext'].apply(lambda x: nltk.word_tokenize(str(x)))
tokenized_title = dataset[['title','selftext']].apply(f, axis =1)
print(len(dataset), len(tokenized_title))
# tokenized_title.to_csv('humor_challenge_data/bot_data/non_qa_total_tokenized.csv', index = False)
# # print(tokenized_title)

# a = pd.read_csv('humor_challenge_data/bot_data/qa_total_tokenized.csv')
# print(a['0'].apply(lambda x : 'hello' in x).sum())

# print(a[a['0'].apply(lambda x : 'hello' in x)])
# print(a['hello' in a[0] ])




# dataset['selftext'] = dataset['selftext'].apply(lambda x: x[:x.find('Edit:')] if 'Edit:' in str(x) and type(x)==str else x)

# dataset = dataset[dataset['selftext'] != '[deleted]']
# dataset = dataset[dataset['selftext'] != '[removed]']
# dataset = dataset[~dataset['selftext'].isnull()]
# dataset.to_csv('humor_challenge_data/bot_data/qa_total.csv', index = False)