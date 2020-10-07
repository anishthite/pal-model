from gpt2.gpt2run import HumorGenGPT
from bert.bertrun import HumorDetector 
from gpt2.bettertrain import JokesDataset
from torch.utils.data import Dataset, DataLoader, SequentialSampler


print('loading gen')
generator = HumorGenGPT('/home/tobias/humor/pal-model/gpt2/trained_models/gpt2_tokens_tag_10056.pt')
print('loading class')
classifier = HumorDetector('/home/tobias/humor/pal-model/bert/bettertrainbert_medium_joker_50066.pt') 

print('adding data')
eval_dataset = JokesDataset(dataset = '/home/tobias/humor/pal-model/total_jokes_tags_test.txt', block_size=1000)
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

# Eval!
print("***** Running humor evaluation *****")
print("  Num examples = %d", len(eval_dataset))
humorscore = 0.0
labelscore = 0.0
avg = 0
gold_ans = 0
nb_eval_steps = 0

bad_jokes = 0
for idx,joke in enumerate(eval_dataloader):
    print(str(idx) + ' ' + str(len(eval_dataloader)))
    inputs, labels = (joke, joke)
    inputs = generator.tokenizer.decode(inputs.tolist()[0])
    inputs_list  = inputs.split('<BOS>')
    if len(inputs_list) < 2:
        bad_jokes +=1
        continue
    inputs = inputs_list[0]
    #output = generator.generate(inputs, max_length=len(inputs[0]) + 10)
    output = generator(inputs)
    print(inputs_list[1].split("<|endoftext|>")[0])
    answer, probs = classifier(inputs_list[1].split("<|endoftext|>")[0])
    score = probs
    gold_ans += answer
    answer, probs = classifier(output)
    humorscore += probs
    avg += answer
    labelscore += probs/score
    nb_eval_steps +=1
print("humorscore: ")
print(str(humorscore/nb_eval_steps))
print("humorescore/labelscore")
print(str(labelscore/nb_eval_steps))
print("simple acc")
print(str(avg))
print(str(gold_ans))
print(str(avg//gold_ans))
