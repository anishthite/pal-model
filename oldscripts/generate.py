MODEL_EPOCH = 4

models_folder = "trained_models"

model_path = os.path.join(models_folder, f"gpt2_medium_joker_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

jokes_output_file_path = f'generated_{MODEL_EPOCH}.jokes'

model.eval()
if os.path.exists(jokes_output_file_path):
    os.remove(jokes_output_file_path)
    
joke_num = 0
with torch.no_grad():
   
        for joke_idx in range(1000):
        
            joke_finished = False

            cur_ids = torch.tensor(tokenizer.encode("JOKE:")).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            
            if joke_finished:
                
                joke_num = joke_num + 1
                
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)

                with open(jokes_output_file_path, 'a') as f:
                    f.write(f"{output_text} \n\n")