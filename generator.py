# This file includes code which was modified from https://colab.research.google.com/drive/1KTLqiAOdKM_3RnBWfqgrvOQLqumUyOdA

import torch
import torch.nn.functional as F


END_OF_TEXT = 50256

def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != END_OF_TEXT:
            sent.append(s)
        else:
            break
    return sent


def generate(model, conditioned_tokens, device, temperature, top_k, top_p, max_len=20):
    past = None
    context_tokens = torch.tensor(conditioned_tokens, device=device, dtype=torch.long).unsqueeze(0)
    position_ids = torch.arange(0, context_tokens.size(-1), dtype=torch.long, device=context_tokens.device)
    
    output = context_tokens.new_zeros([context_tokens.size(0),0])
    prev = context_tokens
    for x in range(max_len):
        prev, probs, past = generate_next_token(model, context_tokens, position_ids, prev, past, device, temperature, top_k, top_p)
        output = torch.cat((output, prev), dim=1)
        # if END_OF_TEXT in prev:
        #     return output
    return output


def generate_next_token(model, context_tokens, position_ids, prev, past, device, temperature, top_k, top_p):
    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids=position_ids, past=past, token_type_ids=None )
        else:
            hidden_states, past = model.transformer(prev, past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits.unsqueeze(0), dim=-1)
        prev = torch.multinomial(probs, 1)
        return prev, probs[0][prev], past

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