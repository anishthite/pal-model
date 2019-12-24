# This file includes code which was modified from https://colab.research.google.com/drive/1KTLqiAOdKM_3RnBWfqgrvOQLqumUyOdA

import torch
import torch.nn.functional as F


END_OF_TEXT = 50256


def generate(model, conditioned_tokens, device, temperature, top_k, top_p):
    generated_tokens = []
    while True:
        result = generate_next_token(model, conditioned_tokens, generated_tokens, device, temperature, top_k, top_p)
        if result == END_OF_TEXT:
            return generated_tokens[:-1]


def generate_next_token(model, conditioned_tokens, generated_tokens, device, temperature, top_k, top_p):
    indexed_tokens = conditioned_tokens + generated_tokens
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    logits = predictions[0, -1, :] / temperature
    filtered_logits = top_filtering(logits, top_k=top_k, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)
    
    #TESTING FOR PROBS
    #from predictor import tokenizer
    #all_tokens = torch.multinomial(probabilities, 10)
    #torch.set_printoptions(profile="full")
    # for x in range(len(probabilities)):
    #    if probabilities[x] != 0.0000000000000000000000000:
    #        print(tokenizer.decode(x))
    #        print(probabilities[x])
    #torch.set_printoptions(profile="default")
    next_token = torch.multinomial(probabilities, 1)
    generated_tokens.append(next_token.item())
    return next_token.item()

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits