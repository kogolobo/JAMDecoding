"""This file support main_keywordextraction.py, "main_generation.py", "main_filtering.py" ,and "main_evaluation.py"
"""
import torch
import re
from datetime import date
import numpy as np
import logging
import random
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from keyword_extraction_methods.liklihood_extractors import get_likelihood_constraints_gpt2, get_likelihood_constraints_infill
from keyword_extraction_methods.content_words_extractors import get_contentwords_constraints
from keyword_extraction_methods.auto_extractors import get_keybert_constraints, get_rake_constraints
from neurologic_super_fast.constraints import PhrasalConstraint, DisjunctiveConstraint

# Helper Functions
def load_gpt2_models_tokenizer(args):
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2-xl", cache_dir = args.cache_dir)
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
    tokenizer_gpt2.padding_side = "left"

    model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2-xl", cache_dir = args.cache_dir).to(args.device)
    model_gpt2.config.pad_token_id = model_gpt2.config.eos_token_id
    model_gpt2.parallelize()
    
    bad_token = ['\n',':', "'", '-', '_', '@', 'Ċ', 'Ġ:']
    bad_words_ids = [tokenizer_gpt2.convert_tokens_to_ids([t]) for t in bad_token]
    return(tokenizer_gpt2, model_gpt2, bad_words_ids)

# Step 0: Preprocess the Raw Data
def uniform_paragraph_symbols(string):
    new_string = ""
    prev_char = ""
    for char in string:
        if char == "\t":
            continue
        if char == "\n":
            if prev_char != "\n":
                new_string += "\n\t"
        else:
            new_string += char
        prev_char = char
    return new_string

# Step 1: Constraint Preparation and Processing
def prepare_constraints(tokenizer, word_list):
    _constraints = []
    for constraint_idx, word in enumerate(word_list):
        if type(word) == str:
            _constraint = PhrasalConstraint(
                tokenizer(word, add_prefix_space=True, add_special_tokens=False).input_ids, constraint_idx
            )
        else:  # disjunctive constraints
            _constraint = DisjunctiveConstraint(
                tokenizer(word, add_prefix_space=True, add_special_tokens=False).input_ids, constraint_idx
            )
        _constraints.append(_constraint)
    return _constraints

def pre_process_constraints(constraint_list, y_orig, prefix):
    constraint_words = []
    y_orig_lower = y_orig.lower()
    y_orig_lower_words = y_orig.lower().split()
    for word in constraint_list:
        # If word is empty then ignore
        if word in ["", " "]:
            continue
        # remove spaces in front of word/phrases
        if word[0] == " ":
            word = word[1:]
        # If word is not in y_orig, then remove
        if word.lower() not in y_orig_lower_words:
            continue
        # remove repeats of words
        if word in constraint_words:
            continue
        # if word is already in constraint_words continue
        if len(constraint_words)>0 and (word in [w.split() for w in constraint_words][0]):
            continue
        constraint_words.append(word)
        
    return(constraint_words)

def combine_constraints(constraint_words, like_ls):
    final_ls = []
    for i in range(len(constraint_words)):
        if like_ls[i] == []:
            final_ls.append(constraint_words[i])
        else:
            if len(like_ls[i])==1 and like_ls[i][0]==constraint_words[i]:
                final_ls.append(like_ls[i][0])
            else:  
                final_ls.append([constraint_words[i]]+like_ls[i])
    return(final_ls)

def skip_generation(args, y_orig, prefix, save_idx, results):
    results[str(save_idx)]= { 'x_l':prefix, 
            'y_orig': y_orig,
            'keywords':{"skipped generation"},
            'keywords_raw':{"skipped generation"},
            'generations': y_orig}
    # re-save for each y_orig
    date_var = str(date.today().strftime("%b-%d-%Y"))
    torch.save(results, args.save_dir + date_var)
    return(results)

def get_constraints(keyword_extractor, y_orig, args, save_idx, likelihood_constraints_gpt2_ls = None, model_t5=None, tokenizer_t5=None):
    # Likelihood Based
    if keyword_extractor == "likelihood-infill":
        logging.info("Starting likelihood-infill")
        keyword_list = get_likelihood_constraints_infill(y_orig, model_t5, tokenizer_t5, args)
        logging.info("Finished likelihood-infill")
    elif keyword_extractor == "likelihood-gpt2":
        keyword_list = likelihood_constraints_gpt2_ls[save_idx]
        logging.info("Finished likelihood-gpt2")
    # Content Words
    elif keyword_extractor == "content_words":
        logging.info("Starting content_words")
        keyword_list = get_contentwords_constraints(y_orig)
        random.seed(1)
        keyword_list = random.sample(keyword_list, int(len(keyword_list) * args.p_content_words))
        logging.info("Finished content_words")
    # Auto Extractors
    elif keyword_extractor == "keybert":
        logging.info("Starting keybert")
        keyword_list = get_keybert_constraints(y_orig, args)
        logging.info("Finished keybert")
    elif keyword_extractor == "rake":
        logging.info("Starting rake")
        keyword_list = get_rake_constraints(y_orig)
        logging.info("Finished rake")
    else:
        print("Invalid type of keyword extractor choosen. Please choose from: 'likelihood-infill', 'likelihood-gpt2', 'content_words', 'keybert', and 'rake'.")
        exit()
    return(keyword_list)

# Step 2: Generations
def group_sentence_for_generation(data, args, continuing_text, num_text, dir_list):
    # Group sentences for batch generation based on length of y_orig
    sentence_length = [len(data[key]['y_orig'].split(" ")) for key in data.keys()]
    # Check length of y_orig and if too short then skip generations
    sentence_length = [0 if val <= args.min_length_to_generate else val for val in sentence_length]
    sorted_val = np.sort(sentence_length)
    all_sorted_indicies = np.argsort(sentence_length)
    sorted_indices = [str(all_sorted_indicies[i]) for i in range(len(all_sorted_indicies)) if sorted_val[i]!= 0]
    number_groups = int(np.ceil(len(sorted_indices)/args.generation_batch_size))
    start_group = 0 # change this if continuing text (below)
    # if we are continuing a text, then we need to find the place to start
    if continuing_text:
        unfinished_indices = [i for i in range(len(sorted_indices)) if "generations" not in list(data[sorted_indices[i]].keys())]
        start_group = (unfinished_indices[0])//args.generation_batch_size
        logging.info('Continuing generation for %s / %s on group %s', num_text, len(dir_list), start_group)
    return(start_group, number_groups, sorted_indices)

# 3. Filtering
def create_valid_filter_keys(sampled, ordered, diversity):
    valid_filter_keys = []
    if sampled == True:
        sampled_options = ["True", "False"]
    else:
        sampled_options = ["False"]
    if ordered == True:
        ordered_options = ["True", "False"]
    else:
        ordered_options = ["False"]
    if diversity == True:
        diversity_options = ["True", "False"]
    else:
        diversity_options = ["False"]
    for s in sampled_options:
        for o in ordered_options:
            for d in diversity_options:
                valid_filter_keys.append(s+"_"+o+"_"+d)
    return(valid_filter_keys)

def skip_filtering(key, generation, save_idx, args, data):
    data[key]['final_generations'] = generation
    data[key]['filtered_topk'] =  "skipped filtering"
    data[key]['cola_values_topk']= "skipped filtering"
    data[key]['nli_values_topk']= "skipped filtering"
    data[key]['filtered_greedy'] =  "skipped filtering"
    data[key]['cola_values_greedy']= "skipped filtering"
    data[key]['filtered_threshold'] = "skipped filtering"
    data[key]['cola_values_threshold'] = "skipped filtering"
    date_var = str(date.today().strftime("%b-%d-%Y"))
    torch.save(data, args.save_dir + date_var)
    return(data)



def calc_cola_score(text, model, tokenizer, device):
    prob_good = []
    tokenize_input = tokenizer.tokenize(text)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    output=model(tensor_input)
    # 0 = probability of bad grammar, 1 = probability of good grammar
    prob_good.append(output.logits.softmax(-1)[0][1].item())
    return(prob_good)

def clean_text(text):
    clean_text = []
    for t in text:
        t_clean = [
            line.replace("\xa0", " ")  # breaking spaces
            for line in t.text.split("\n") 
            if line != ''  # drop empty lines
        ]
        # rejoin into one formatted snippet
        clean_text.append("\n".join(t_clean)) 

def create_complete_sentence(generations):
    final_generation = []
    for g in generations:
        sentences = re.split(r'(?<=[.!?]) +', g)
        sentences = [s for s in sentences if s != ""]
        # If it is less than 1 sentence, then remove generation
        if (len(g.split(".")) <=1) and (len(g.split("!")) <=1) and (len(g.split("?")) <=1):
            # g = g + "."
            continue
        # If it ends in punctuation, than do not alter
        if sentences[-1].endswith(".") or sentences[-1].endswith("!") or sentences[-1].endswith("?"):
            final_generation.append(g)
        else:
            # If it is more than 1 sentence than drop last unfinished sentence
            g = remove_unfinished_sentences(g)
            final_generation.append(g)
    return(final_generation)

def remove_unfinished_sentences(text):
    # Split the text into sentences using a regular expression
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Iterate through the sentences to find the last complete sentence
    last_sentence = ""
    complete_sentences = []
    for sentence in sentences:
        if sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?"):
            last_sentence = sentence
            complete_sentences.append(sentence)
        else:
            break

    return " ".join(complete_sentences)

def skip_filtering(key, generation, save_idx, args, data):
    data[key]['final_generations'] = generation
    data[key]['filtered_topk'] =  "skipped filtering"
    data[key]['cola_values_topk']= "skipped filtering"
    data[key]['nli_values_topk']= "skipped filtering"
    data[key]['filtered_greedy'] =  "skipped filtering"
    data[key]['cola_values_greedy']= "skipped filtering"
    data[key]['filtered_threshold'] = "skipped filtering"
    data[key]['cola_values_threshold'] = "skipped filtering"
    return(data)

def cola_score(text, cola_tokenizer, cola_model, device):
    tokenize_input = cola_tokenizer.tokenize(text)
    tensor_input = torch.tensor([cola_tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    output=cola_model(tensor_input)
    return output.logits.softmax(-1)[0][1].item()

class Dataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    
    def __getitem__(self, i):
        return self.xs[i], self.ys[i]
    
    def __len__(self):
        return len(self.xs)