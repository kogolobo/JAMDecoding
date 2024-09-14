"""This file creates generations using keywords extracted from main_keyword_extraction.py
"""
import os

import sys
from argparse import ArgumentParser
import logging
import time
import numpy as np
from datetime import date

import torch
from transformers import GenerationConfig
from neurologic_super_fast.generate import GenerationWrapper
from neurologic_super_fast.util import NeuroLogicConfig
from src.utils import group_sentence_for_generation, load_gpt2_models_tokenizer

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def parse_args():
    parser = ArgumentParser()
    # Directories and Experimental arguments
    parser.add_argument("--device_id", default =0, type=int)
    parser.add_argument("--data_path", type=str, help="Location of raw data", required=True)
    parser.add_argument("--output_path", type=str, help="Location of results", required=True)
    
    
    # parser.add_argument("--data_dir", default =None, type=str, help="Specify specific directory of keyword extraction or uses 'None' to use the same save_directory from main_keyword_extraction")
    # parser.add_argument("--dataset", default ="amt", type=str)
    # parser.add_argument("--num_authors", default =3, type=int)
    # parser.add_argument("--save_dir", default ="/results/", type=str)
    parser.add_argument("--min_length_to_generate", default=3, type=int, help="Minimum value of original sentence to actually create generations") 

    # Generation Config
    parser.add_argument("--generation_batch_size", default =4, type=int, help="Number of sentences to generate at once") 
    parser.add_argument("--max_new_tokens", default =2, type=int, help = "Max number of tokens for generation (number of times the original length)") 
    parser.add_argument("--num_beams", default =10, type=int)
    parser.add_argument("--num_return_sequences", default =10, type=int, help="Number of beams to select, must be less than or equal to num_beams") 
    parser.add_argument("--no_repeat_ngram", default =3, type=int)
    parser.add_argument("--do_sample", default = [True, False], nargs='+', help = "Use sampling for next token in beams instead of greedy decoding")
    parser.add_argument("--min_length", default =2, type=int)
    parser.add_argument("--repetition_penalty", default =1.0, type=int) 

    # Neurologic Decoding Config
    parser.add_argument("--ordered", default = [True, False], nargs='+', help="Make constraints be satisfied in order") 
    parser.add_argument("--grouping_strategy", default ='type', type=str, help="Options for beam choice: 'number' or 'type'") 
    parser.add_argument("--likelihood_prune_factor", default =0.4, type=float, help="Creates cut-off for likelihood tolerance in pruning step (p percentage)") 
    parser.add_argument("--constraint_prune_factor", default =0.6, type=float, help = "Creates cut-off for constraint tolerance in pruning step ( at least p percentage of constraints), higher p =  harder to pass") 
    parser.add_argument("--diversity", default = [True, False], nargs='+', help="Add diversity to pre-processing of logits")   
    parser.add_argument("--do_early_stopping", default = True, type=bool, help="Stop beam search early if candidates are not better than last") 

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
    logging.info("START GENERATING GENERATIONS FOR %s", args.data_path)

    # GPT2 tokenizer and model
    tokenizer_gpt2, model_gpt2, bad_words_ids = load_gpt2_models_tokenizer(args)

# 1. Prepare inputs
    paragraphs = torch.load(args.data_path)
    
# 2. Start generation
    # Cycle through all data in each file in the data_dir
    start = time.time()
    processed_paragraphs = []
    for pagraph_id, paragraph in enumerate(paragraphs):
        logging.info('Starting generation for %s / %s', pagraph_id, len(paragraphs))

        continuing_text = False
        save_idx = 0
        
        # Group sentences for batch generation based on length of y_orig
        start_group, number_groups, sorted_indices = group_sentence_for_generation(paragraph, args, continuing_text, pagraph_id, dir_list=[])
        
        # run through each group
        for i in range(start_group, number_groups):
            start_idx = i*args.generation_batch_size
            end_idx = (i+1)*args.generation_batch_size
            current_indices = sorted_indices[start_idx:end_idx]
            y_orig = [paragraph[key]['y_orig'] for key in current_indices]
            prefix = [paragraph[key]['x_l'] for key in current_indices]
            input_encoding = tokenizer_gpt2(prefix, padding=True, return_tensors="pt").to(args.device)
            for key in current_indices:
                paragraph[key]['generations'] = {}
            output_sequences = {}
            sent_start_time = time.time()
            logging.info('Starting Group %s out of %s of size %s', i, number_groups, args.generation_batch_size)
            logging.info('y_orig: %s', y_orig)
            logging.info('x_l: %s', prefix)

# 2.1 Prepare generation config
            for keywords_name in list(paragraph[sorted_indices[start_idx]]['keywords'].keys()):
                # New generations can be up to max_new_tokens times the largest text in a batch 
                max_new_length = np.max([int(len(tokenizer_gpt2(y)['input_ids'])*args.max_new_tokens) for y in y_orig])
                keywords_ls = [paragraph[key]['keywords'][keywords_name] for key in current_indices]
                output_sequences[keywords_name] = {}
                logging.info('Start Generation for %s', keywords_name)
                for i, constraints_ls in enumerate(keywords_ls[0].keys()):
                    constraints = [keyword_ls[constraints_ls][0] for keyword_ls in keywords_ls]
                    # if constraint is empty then skip generation
                    non_empty_constraints = [c for c in constraints if c != []]
                    skipped_key = []
                    if len(constraints) != len(non_empty_constraints):
                        final_constraints = []
                        final_prefix = []
                        for i,constraint_list in enumerate(constraints):
                            if constraint_list == []:
                                skipped_key.append(current_indices[i])
                                print(f"Skipped {current_indices[i]} for {constraints_ls} due to no constraints")
                            else:
                                final_constraints.append(constraint_list)
                                final_prefix.append(prefix[i])
                        input_encoding = tokenizer_gpt2(final_prefix, padding=True, return_tensors="pt").to(args.device)
                        constraints = final_constraints
                    logging.info('Starting %s', constraints_ls)
                    output_sequences[keywords_name][constraints_ls] = {}
                    for do_sample in args.do_sample:
                        logging.info('Sampling: %s', do_sample)
                        generation_config = GenerationConfig(
                            max_new_tokens=max_new_length, num_return_sequences=args.num_return_sequences, pad_token_id=model_gpt2.config.pad_token_id,
                            num_beams=args.num_beams, no_repeat_ngram_size=args.no_repeat_ngram,
                            do_sample=do_sample,
                        )
                        for ordered in args.ordered:
                            logging.info('Ordered: %s', ordered)
                            for diversity in args.diversity:
                                logging.info('Diversity: %s', diversity)

# 2.2 Prepare neurologic decoding config
                                neurologic_config = NeuroLogicConfig(
                                    constraints=constraints,
                                    ordered=ordered, 
                                    grouping_strategy=args.grouping_strategy, 
                                    likelihood_prune_factor=args.likelihood_prune_factor,
                                    constraint_prune_factor = args.constraint_prune_factor,
                                    constraint_prune_number=int(len(constraints[0])*args.constraint_prune_factor), 
                                    diversity = diversity, 
                                    do_early_stopping = args.do_early_stopping 
                                )
                
# 2.3 Create Generations
                                wrapper = GenerationWrapper(model_gpt2)

                                outputs = wrapper.generate(
                                    **input_encoding,
                                    generation_config=generation_config,
                                    neurologic_config=neurologic_config,
                                    no_repeat_ngram_size=args.no_repeat_ngram,
                                    bad_words_ids=bad_words_ids,
                                    min_length=args.min_length,
                                    repetition_penalty=args.repetition_penalty,
                                )
                                generation_parameters = f"{keywords_name}_{constraints_ls}_{str(do_sample)}_{str(ordered)}_{str(diversity)}"
                                input_and_generation = tokenizer_gpt2.batch_decode(outputs, skip_special_tokens=True)
                                generations = [g[len(prefix):] for g in input_and_generation] # batch_size X number_return_sequences
                                for i,key in enumerate(sorted_indices[start_idx:end_idx]):
                                    # if we skip generation (no constraints given)
                                    if key in skipped_key:
                                        continue
                                    batch_generations = [g[len(prefix[i]):] for g in input_and_generation[i * args.num_return_sequences : (i+1) * args.num_return_sequences]] # number_return_sequences
                                    paragraph[key]["generations"][generation_parameters] = batch_generations
            end = time.time()
            print("Number of Generation before Filter: ", len(output_sequences))
            logging.info('End Generation')

# 3. Save data/generation 
        #     date_var = str(date.today().strftime("%b-%d-%Y"))
        #     torch.save(paragraph, save_filename + date_var)
        #     save_idx +=1
        # date_var = str(date.today().strftime("%b-%d-%Y"))
        # torch.save(paragraph, save_filename + date_var + "_final")
        processed_paragraphs.append(paragraph)
        
    # Save data
    torch.save(processed_paragraphs, args.output_path)
    exp_total_time = time.time() - exp_start_time
    print(f"Total Generation Time for: {exp_total_time/60} minutes")
    print(f"Results output to {args.output_path}")

