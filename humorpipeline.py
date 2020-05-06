from gpt2run import HumorGenGPT
from bertrun import HumorDetector
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpathbert", default='/home/anishthite/Workspace/pal-model/models/bettertraingpt2_medium_joker_10066.pt', type=str, required=False)
    parser.add_argument("--modelpathgpt2", default='/home/anishthite/Workspace/pal-model/models/bettertraingpt2_medium_joker_10066.pt', type=str, required=False)
    args = parser.parse_args()
    generator = HumorGenGPT(args.modelpathgpt2)
    classifier = HumorDetector(args.modelpathbert)

    while True:
        query = input("Enter Question: ")
        answer = generator(query)
        print(answer)
        answer, probs = classifier(answer)
        if answer == 0:
            print("not joke :( probs " + str(probs))
        else:    
            print("Joke ;) probs " + str(probs))
