"""This file evaluated filtered generations using filtered generations extracted from main_filter.py
"""
import os
import sys
from argparse import ArgumentParser
import logging
import time
import numpy as np
import pickle
import evaluate
import torch
from src.utils import cola_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.stylometric_method import StylometricGeneration
import src.writeprintsStatic as ws

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    parser = ArgumentParser()
    
    # Directories and Experimental arguments
    parser.add_argument("--device_id", default =0, type=int)
    parser.add_argument("--data_path", type=str, help="Location of raw data", required=True)
    parser.add_argument("--output_path", type=str, help="Location of results", required=True)
    parser.add_argument("--word_embedding_dict_dir", type=str, help="Location of word embedding dictionary", required=True)
    parser.add_argument("--eval_only", default =False, type=bool, help="If final paragraph generations have been created")


    # Evaluation Arguments
    parser.add_argument("--eval_nli_threshold", default =0.8, type=float) 
    parser.add_argument("--eval_cola_threshold", default =0.8, type=float)
    
    # Stylometric Arguments (used if using basic stylometric for no-generation sent)
    parser.add_argument("--eval_use_stylo", default =True, type=bool, help="If True, will use basic stylometric method to obfuscate any sentence that does not have a suitable generation. If false, will use orig. sent.")
    parser.add_argument("--eval_stylo_cola_threshold", default =0.7, type=float, help="Threshold value that it must pass in CoLA")
    parser.add_argument("--eval_stylo_top_k", default =3, type=int, help="Top-k similar words to consider for replacement")
    parser.add_argument("--stylo_alpha", default = 0.75, type = float, help="Weight of CoLA versus similarity scores when choosing a replacement word")
    parser.add_argument("--stylo_verbose_threshold", default = 1.0, type = float, help="Use to reduce number of adjectives, if 1.0 than will keep all adjectives")
    parser.add_argument("--sentence_similarity", default =True, type=bool, help="Set sentences similarity by sentence instead of word")
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device_id}")


    return args


if __name__ == "__main__":
    # Set working directory to this file
    args = parse_args()
#0. Set Parameters and load models/data
    exp_start_time = time.time()
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',  stream=sys.stdout)
    logging.info('Arguments: %s', args)
    logging.info("START EVALUATING FOR %s", args.data_path)

    # Download necessary models for evaluation
    nli_model_name = "alisawuffles/roberta-large-wanli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, map_location = args.device)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(args.device)

    cola_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(args.device)
    cola_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA', map_location = args.device)

    # Create stylometric obfuscator if want to use on text which do not have suitable generations
    if args.eval_use_stylo:
        stylo_generator = StylometricGeneration(
            args=args,
            device = torch.device(f"cuda:{args.device_id}"),
            word_embedding_dict_dir=args.word_embedding_dict_dir
        )

# 1. Prepare inputs
    logging.info('Prepare Data')
    paragraphs = torch.load(args.data_path)
    
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
            

    # Cycle through all files in data_dir
    processed_paragraphs = []
    for num_text, data in enumerate(paragraphs):
        logging.info('Starting evaluation for %s / %s', num_text, len(paragraphs))

        original_text = []
        obfuscated_text = []
        mean_nli_ls = []
        num_orig_used = 0
        num_stylo_used = 0

#2.1  Create final obfuscated generation
        logging.info("Creating obfuscated text")
        # cycle through each chunk of sentences in a paragraph
        for key in data.keys():
            sent_start_time = time.time()
            y_orig = data[key]['y_orig']
            original_text.append(y_orig)
            prefix = data[key]['x_l']

            # if we skipped filtering than use original
            cola_options = []
            if data[key]['nli_values_topk'] == "skipped filtering":
                obfuscated_text.append(data[key]['y_orig'])
                mean_nli_ls.append(1)
                num_orig_used +=1
                continue

            # select final obfuscation using nli/cola threshold
            # find filtered generations that are above nli threshold
            for i, nli in enumerate(data[key]['nli_values_topk']):
                if nli > args.eval_nli_threshold:
                    cola_options.append(data[key]['cola_values_topk'][i])
            # find filtered generations that are above cola threshold
            for i in range(len(cola_options)):
                if cola_options[i] < args.eval_cola_threshold:
                    cola_options[i] = 0 
            # use cola to find optimal generations (can change this based on authorship qualities)  
            if (len(cola_options)>0) and (max(cola_options)>0):
                obfuscated_text.append(data[key]['filtered_topk'][cola_options.index(max(cola_options))])
                mean_nli_ls.append((data[key]['nli_values_topk'][cola_options.index(max(cola_options))]))
            else:
# 2.2 If no generation pass NLI/CoLA threshold than either use basic stylometric method or use original sentence
                if args.eval_use_stylo:
                    # create new obfuscation using stylometric method
                    stylo_obfuscation, stylo_method_data, stylo_obfuscation_cola = stylo_generator.generate(prompt = data[key]['y_orig'], 
                                    alpha = 0.75, 
                                    k = args.eval_stylo_top_k, 
                                    cola_threshold = args.eval_stylo_cola_threshold, 
                                    verbose_threshold = args.stylo_verbose_threshold,
                                    sentence_similarity = args.sentence_similarity) 
                    # cola threshold HERE! IF DOESN'T PASS --> USE ORIGINAL
                    if stylo_obfuscation_cola >= args.eval_stylo_cola_threshold:
                        num_stylo_used +=1
                        obfuscated_text.append(stylo_obfuscation[0])
                    else:
                        obfuscated_text.append(data[key]['y_orig'])
                        mean_nli_ls.append(1)
                        num_orig_used +=1
                else:
                    obfuscated_text.append(data[key]['y_orig'])
                    mean_nli_ls.append(1)
                    num_orig_used +=1
        obfuscated_text = [g.encode("ascii", "ignore").decode() for g in obfuscated_text]
        processed_paragraphs.append(' '.join(obfuscated_text))
        logging.info("Total number of sentences: %s ", len(data.keys()))
        logging.info("Number of original sentences used: %s", num_orig_used)
    
    torch.save(processed_paragraphs, args.output_path)
    
    
        
       