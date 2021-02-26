from gpt2.gpt2run import HumorGenGPT
from bert.bertrun import HumorDetector
import argparse
import numpy as np

class HumorPipeline():
    def __init__(self, genpath, classifierpath, num_gens=20,  eval_mode=False):
        self.generator = HumorGenGPT(genpath) 
        self.classifier = HumorDetector(classifierpath)
        self.num_gens = num_gens
        self.eval_mode = eval_mode
    
    def __call__(self, query, **kwargs):
        return self.predict(query, **kwargs)

    def predict(self, inputs, **kwargs):
        import time
        answerlist = []
        start = time.perf_counter()
        answerlist = [self.generator(inputs, **kwargs)[0] for _ in range(self.num_gens)]
        print(f"{time.perf_counter() - start}")
        start = time.perf_counter()
        answer = answerlist[np.argmax([self.classifier(answer)[1] for answer in answerlist])]
        print(f"{time.perf_counter() - start}")
        return answer




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpathbert", default='/home/tobias/humor/pal-model/bettertrainbert_medium_joker_10066.pt', type=str, required=False)
    parser.add_argument("--modelpathgpt2", default='/home/tobias/humor/pal-model/gpt2/trained_models/gpt2_tokens_tag_10056.pt', type=str, required=False)
    #parser.add_argument("--modelpathgpt2", default='/home/tobias/humor/pal-model/gpt2/trained_models/dialogpt2_tokens_tag.pt', type=str, required=False)
    args = parser.parse_args()
    pipeline = HumorPipeline(args.modelpathgpt2, args.modelpathbert)
    while True:
        query = input("Enter Question: ")
        answer = pipeline(query, do_sample=True)
        print(answer)
