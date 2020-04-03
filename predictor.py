import wget
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import generator
import os
from argparse import ArgumentParser

DEFAULT_MODEL_PATH = "./models/medium"

##medium_config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)
#config = GPT2Config.from_json_file(os.path.join(DEFAULT_MODEL_PATH, 'config.json'))
#model = GPT2LMHeadModel(config)
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

def init(model_path, metadata):

    #weights = torch.load(os.path.join(DEFAULT_MODEL_PATH, 'medium_ft.pkl'))
    #weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    #weights.pop("lm_head.decoder.weight", None)
    #model.load_state_dict(weights)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(metadata["device"])


def predict(metadata):
    conditioned_tokens = sum([tokenizer.encode(utterance) + [generator.END_OF_TEXT] for utterance in metadata["history"]],[])
    prediction = generator.generate(model, conditioned_tokens, metadata["device"], metadata["temperature"], metadata["top_k"], metadata["top_p"])
    prediction = prediction.tolist()
    return tokenizer.decode(generator.cut_seq_to_eos(prediction[0])).encode('ascii','ignore').decode('ascii')
    #return tokenizer.decode(prediction)

##Testing
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--max_history", type=int, default=20, help="Number of previous utterances to keep in history")
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    metadata = {}
    metadata["device"] = "cuda"
    metadata["top_p"] = args.top_p
    metadata["top_k"] = args.top_k
    metadata["temperature"] = args.temperature
    init(args.model_path, metadata)
    while True:
        raw_text = input("USER >> ")
        response = predict(metadata)
        print(response)
        # response = predictd({"history": raw_text}, metadata)
        metadata["history"].append(response)
        metadata["history"] = metadata["history"][-(2*args.max_history+1):]
        print(response)
