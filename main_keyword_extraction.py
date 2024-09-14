"""This file extracts the keywords for a given set of text (must be preprocessed using process_raw_data.py)
"""
import os
import sys
from argparse import ArgumentParser
import logging
import time
import random
import nltk
from datetime import date
from torch.utils.data import DataLoader

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from keyword_extraction_methods.likelihood_extractors import get_likelihood_constraints_gpt2
from src.utils import prepare_constraints, pre_process_constraints, skip_generation, combine_constraints, Dataset, load_gpt2_models_tokenizer, get_constraints
from src.medium_constraint import get_synon_words, get_lemmatized_words, remove_repeated_tokens

def parse_args():
    parser = ArgumentParser()
    # Directories and Experimental arguments 
    parser.add_argument("--device_id", default =0, type=int, help="Specify which GPU to use")
    parser.add_argument("--data_path", type=str, help="Location of raw data", required=True)
    parser.add_argument("--output_path", type=str, help="Location of results", required=True)
    
    parser.add_argument("--prefix_context_size", default=0, type=int, help="Optional way to create prefix_ls, using the previous x sentences (x = context_size), if set to 0, then must input custom prefix_ls") 
    parser.add_argument("--min_length_to_generate", default=3, type=int, help="Minimum value of original sentence to actually create generations") 

    # Constraint arguments
    parser.add_argument("--keyword_extractors", nargs='+', default = ["likelihood-gpt2" , "likelihood-infill",  "keybert"], help="Options: 'likelihood-infill', 'likelihood-gpt2', 'content_words', 'keybert', and 'rake'")
    parser.add_argument("--likelihood_p_threshold", default =0.5, type=float, help=" Bottom percentage p of likelihoods to use") 
    parser.add_argument("--p_content_words", default = 0.8, type=float, help="Percentage of randomly selected content words to include as constraints")  
    parser.add_argument("--keybert_length", default =2, type=int, help="Returns len(y_orig)/keybert_length number of constraint words") 
    parser.add_argument("--batch_size", default =4, type=int, help="Batch size to get constraints from likelihood_gpt2") 

    # Medium Constraint arguments
    parser.add_argument("--top_k_medium_constraints", default =4, type=int, help="Returns len(y_orig)/keybert_length number of constraint words")
    parser.add_argument("--like_words", action='store_false', help= " Add disjunctive to like words (same lemma) for adj, noun, verbs")
    parser.add_argument("--similar_words", action='store_false', help= " Add disjunctive to similar words (synonyms) for adj, noun, verbs")

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device_id}")
    return args


if __name__ == "__main__":
    args = parse_args()
    
#0. Set Parameters and load models/data
    exp_start_time = time.time()
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',  stream=sys.stdout)
    logging.info('Arguments: %s', args)
    logging.info("START KEYWORD EXTRACTION FOR %s", args.data_path)

    # Create a new directory if it does not exist
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    # Load Models
    logging.info("Loading models/data")
    # GPT2
    tokenizer_gpt2, model_gpt2, bad_words_ids = load_gpt2_models_tokenizer(args)
    # T5
    model_t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer_t5 = T5Tokenizer.from_pretrained('t5-base')

    # nltk.data.path.append(args.nltk_data_dir)
    # nltk.data.path = nltk.data.path[1:]

# 1. Prepare inputs
    logging.info('Prepare Data')
    paragraphs = torch.load(args.data_path)
    processed_paragraphs = []
    for paragraph_id, paragraph in enumerate(paragraphs):
        logging.info('Starting keyword extraction for %s / %s', paragraph_id+1, len(paragraphs))
        y_orig_ls = [sent.replace("\n","") for sent in paragraph['y_orig']]

        # Either use pre-made prefix's or use previous x sentences (x = args.prefix_context_size)
        prefix_ls = []
        if args.prefix_context_size > 0:
            for i in range(len(y_orig_ls)):
                if i in [0,1]:
                    prefix_ls.append(y_orig_ls[0])
                else:
                    prefix_ls.append(" ".join(y_orig_ls[i-args.prefix_context_size:i]))
        else:
            prefix_ls = paragraph['x_l']

        # remove paragraph symbol
        y_orig_ls = [o.replace("\n", "") for o in y_orig_ls]
        prefix_ls = [p.replace("\n", "") for p in prefix_ls]

        # create dataset/dataloader to run through gpt2 keyword extraction
        dataset = Dataset(y_orig_ls, prefix_ls)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Can generate constraints from gpt2 in batches (do this before "for" loop below)
        likelihood_constraints_gpt2_ls = []
        if "likelihood-gpt2" in args.keyword_extractors:
            logging.info("Starting likelihood-gpt2")
            for y_orig, prefix in dataloader:
                keyword_list = get_likelihood_constraints_gpt2(tokenizer_gpt2, model_gpt2, y_orig, prefix, args)
                likelihood_constraints_gpt2_ls.extend(keyword_list)
            logging.info("Finished likelihood-gpt2")

        # loop through each sentence to find rest of constraints
        paragraph_results = {}
        for save_idx, (prefix, y_orig) in enumerate(zip(prefix_ls, y_orig_ls)):
            # log info
            sent_start_time = time.time()
            logging.info('Starting Sample %s out of %s', save_idx+1, len(prefix_ls))
            logging.info('y_orig: %s', y_orig)
            logging.info('x_l: %s', prefix)
            prefix_list = [prefix]
            # input_encoding = tokenizer_gpt2(prefix_list, padding=True, return_tensors="pt").to(args.device)
            logging.info('Extract Keywords from y_orig as Constraints')
            output_sequences = []

            # Check length of y_orig and if too short then skip generations
            if len(y_orig.split(" ")) <= args.min_length_to_generate:
                paragraph_results[str(save_idx)] = skip_generation(args, y_orig, prefix, save_idx, paragraph_results)
                continue

# 2. Get constraint words using defined methods 
            constraint_ls = {}
            constraint_raw_ls = {}
            for keyword_extractor in args.keyword_extractors:
                logging.info("Keyword Extractor: %s", keyword_extractor)
                keyword_list = get_constraints(keyword_extractor, y_orig, args, save_idx, likelihood_constraints_gpt2_ls, model_t5, tokenizer_t5)    
                    
# 3. Process Constraints
                # put keywords in order
                y_orig_lower = y_orig.lower()
                ordered_keyword_list = []
                for keyword in keyword_list:
                    # put's words in order
                    clean_keyword = keyword.lower().strip()
                    keyword_start_idx = y_orig_lower.find(clean_keyword)
                    ordered_keyword_list.append((y_orig[keyword_start_idx:keyword_start_idx + len(clean_keyword)],
                                                keyword_start_idx))
                ordered_keyword_list = sorted(ordered_keyword_list, key=lambda x: x[-1])
                constraint_words = [o[0] for o in ordered_keyword_list]
                logging.info("Finished putting words in order")

                # processes constraint to include space
                constraint_words = pre_process_constraints(constraint_words, y_orig, prefix)
                # cannot have no keywords, so if none chosen by method then choose a random words
                search_try = 0
                while constraint_words == [] and search_try <= 100:
                    keyword_list = random.sample([y for y in y_orig.split(" ") if y != ""], 1)
                    constraint_words = pre_process_constraints(keyword_list, y_orig, prefix)
                    search_try += 1
                    if search_try > 100:
                        print("Not able to find at least 1 constraint word even though sentence is above min requirements")
                print("Original Constraints:", constraint_words)

#4. Create medium constraints (optional)
                # Use nltk - wordnet to find like words (same lemma) and similar words (synonyms)
                similar_ls = [[]]*len(constraint_words)
                like_ls = [[]]*len(constraint_words)
                if args.similar_words:
                    similar_ls = get_synon_words(constraint_words, y_orig)
                if args.like_words:
                    like_ls = get_lemmatized_words(constraint_words)
                
                # combine lists
                constraint_list_like = combine_constraints(constraint_words, like_ls)
                constraint_list_like_similar = combine_constraints(constraint_words, [s+l for s,l in zip(similar_ls, like_ls)])

                # remove repeated tokens (needed for constraint decoding)
                constraint_list_like = remove_repeated_tokens(constraint_list_like, tokenizer_gpt2)
                constraint_list_like_similar = remove_repeated_tokens(constraint_list_like_similar, tokenizer_gpt2)
                logging.info("Finished finding like and similar words")

                end = time.time()
                # Create constraints
                constraint_ls[keyword_extractor] = {"constraints_like_similar": [prepare_constraints(tokenizer_gpt2, constraint_list_like_similar)],
                "constraints_like": [prepare_constraints(tokenizer_gpt2, constraint_list_like)],
                "constraints_only": [prepare_constraints(tokenizer_gpt2, constraint_words)]}
                constraint_raw_ls[keyword_extractor] = {"constraints_like_similar_raw": constraint_list_like_similar,
                "constraints_like_raw": constraint_list_like,
                "constraints_only_raw":constraint_words}
                
                logging.info("Finished constraints for %s", keyword_extractor)
                
#5. Save constraints
            paragraph_results[str(save_idx)] = {
                'x_l':prefix, 
                'y_orig': y_orig,
                'keywords':constraint_ls,
                'keywords_raw':constraint_raw_ls
            }

        processed_paragraphs.append(paragraph_results)
    
    torch.save(processed_paragraphs, args.output_path)
    exp_total_time = time.time() - exp_start_time
    print(f"Total Keyword Extraction Time for {paragraph_id}: {exp_total_time/60} minutes")
    print(f"Results output to {args.output_path}")
