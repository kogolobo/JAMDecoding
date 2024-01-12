"""This file support src/stylometric_method.py
"""
import torch
import numpy as np
import spacy


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def find_cosine_similarity(original_embedding, word_embeddings):
  original_embedding_norm = original_embedding  / original_embedding .norm(dim=1)[:, None]
  word_embeddings_norm = word_embeddings / word_embeddings.norm(dim=1)[:, None]
  cos_score = torch.mm(original_embedding_norm, word_embeddings_norm.transpose(0,1))
  return(cos_score)

def average_repeated_values(candidates, similarity_scores, top_k_threshold, tokenizer, original_raw, original_spacy, device):
  vals, idx = torch.sort(similarity_scores, descending = True)
  nlp = spacy.load("en_core_web_sm")
  candidate_ls = [original_raw]
  idx_norepeat = []
  vals_norepeat = []
  average_vals = {original_raw:[vals[0][0]]}
  topk = 1
  
  for i in range(1,len(idx[0])):
      candidate =  candidates[idx[0][i].item()].strip().lower()
      candidate_spacy = nlp(candidate)
      if len(candidate_spacy) > 0:
        # make sure that verbs of the same tense and nouns are same tag (plural or not)
        if original_spacy.pos_ in ["VERB", "NOUN"]:
          if candidate_spacy[0].tag_ != original_spacy.tag_:
            continue       
      if candidate in candidate_ls:
        average_vals[candidate].append(vals[0][i])
      else:
          topk+=1
          average_vals[candidate] = []
          average_vals[candidate].append(vals[0][i])
          candidate_ls.append(candidate)
      if topk == top_k_threshold:
          break
  for k in average_vals.keys():
    vals_norepeat.append(torch.mean(torch.stack(average_vals[k])))
  for c in candidate_ls:
    token_ls = tokenizer(c)['input_ids']
    # remove space token
    if 3 in token_ls:
      token_ls.remove(3)
    # remove end token
    if 1 in token_ls:
      token_ls.remove(1)
    idx_norepeat.append(token_ls)
  idx_norepeat = [torch.tensor(i).to(device) for i in idx_norepeat]
  return(torch.tensor([(vals_norepeat)]).to(device), idx_norepeat, candidate_ls)

def evaluation_quality(obfuscated_text, device, model, tokenizer):
    prob_good = []
    for obf in obfuscated_text:
        tokenize_input = tokenizer.tokenize(obf)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        output=model(tensor_input)
        prob_good.append(output.logits.softmax(-1)[0][1].item())
    return(prob_good, np.average(prob_good))

def find_sentence_attribute(spacy_original_prompt, step, sent_model, tokenizer, input_ids, candidate_ls, device, eval_model, eval_tokenizer, fill_in = True, metric = "cosine_similarity"):
  # Get original sentence
  original_prompt = str(spacy_original_prompt) 
  original_word = str(spacy_original_prompt[np.max([0, step-3]):step+3]) 
  # Place an indicator word and then split on sentences
  new_original_prompt = original_prompt.replace(original_word, "INDICATOR-WORD").replace('.', '****').replace('?', '****').replace('!', '****')
  original_prompt_split = new_original_prompt.split('****')
  # Find correct original sentence
  for i,p in enumerate(original_prompt_split):
    if "INDICATOR-WORD" in p:
        sent_num = i
  original_sent = original_prompt_split[sent_num].replace("INDICATOR-WORD", original_word)
  original_embedding = torch.tensor(sent_model.encode(original_sent)).unsqueeze(0)

  # Get generated part of sentence
  if input_ids != None:
    generated_text = tokenizer.decode(input_ids[0].int(), skip_special_tokens = True)
  else:
    generated_text = ""
  # split into sentences
  new_generated = generated_text.replace('.', '****').replace('?', '****').replace('!', '****')
  generated_prompt_split = new_generated.split('****')
  generated_sentence = generated_prompt_split[-1]

  # Extract rest of sentence after current word (original prompt)
  original_sent_end = str(spacy_original_prompt[step+1:]) 
  new_original_sent = original_sent_end.replace('.', '****').replace('?', '****').replace('!', '****')
  original_prompt_split = new_original_sent.split('****')
  original_prompt_sent = original_prompt_split[0]

  cosine_similarity_score = []
  cola_average = []
  for c in candidate_ls:
    candidate_sentence = generated_sentence + c + " "
    if fill_in: 
      if step < len(spacy_original_prompt)-1:
        candidate_sentence = candidate_sentence + original_prompt_sent
    candidate_embedding = torch.tensor(sent_model.encode(candidate_sentence)).unsqueeze(0)
    if metric == "cosine_similarity":
      cosine_similarity_score.append(find_cosine_similarity(original_embedding, candidate_embedding))
    elif metric == "cola":
      cola_average.append(evaluation_quality([candidate_sentence.replace("<pad>", " ")], device, eval_model, eval_tokenizer)[1])
    elif metric == "cola_difference":
      cola_new = evaluation_quality([candidate_sentence.replace("<pad>", " ")], device, eval_model, eval_tokenizer)[1]
      cola_old = evaluation_quality([original_sent.replace("<pad>", " ")], device, eval_model, eval_tokenizer)[1]
      cola_average.append(cola_old - cola_new)
    else:
      print("Metric not found")
      break  
  if metric == "cosine_similarity":
    output = torch.stack([v.squeeze(0) for v in cosine_similarity_score]).transpose(0,1)
  elif metric == "cola":
    output = cola_average
  elif metric == "cola_difference":
    output = cola_average[0]
  return(output)
