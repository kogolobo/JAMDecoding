"""This file creates filters generations using generations from main_generation.py
"""
import os
import sys
from argparse import ArgumentParser
import logging
import time
from datetime import date

import torch
from src.utils import create_complete_sentence, skip_filtering, create_valid_filter_keys
from src.filter import SummFilter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    parser = ArgumentParser()
    
    # Directories and Experimental arguments
    parser.add_argument("--device_id", default =0, type=int)
    parser.add_argument("--data_path", type=str, help="Location of raw data", required=True)
    parser.add_argument("--output_path", type=str, help="Location of results", required=True)

    # Filter Config
    parser.add_argument("--nli_threshold", default =0.8, type=float, help="Creates threshold value for probability of 'entailment'")  
    parser.add_argument("--nli_k", default =5, type=float, help="Top-k based on nli score to consider") 
    parser.add_argument("--nli_batch_size", default =5, type=float, help="Batch size for NLI model") 
    parser.add_argument("--cola_threshold", default =0.0, type=float, help="Creates threshold value for grammatically, if '0.0', then will take top cola_k cola values")  
    parser.add_argument("--cola_k", default =3, type=float, help="Must be smaller than nli_k") 
    
    # Filter which Generations (include in .ssh if you do NOT want to include)
    parser.add_argument("--filter_infill", action='store_false') 
    parser.add_argument("--filter_gpt2", action='store_false')  
    parser.add_argument("--filter_keybert", action='store_false') 
    parser.add_argument("--filter_content_words", action='store_false') 
    parser.add_argument("--filter_rake", action='store_false') 
    parser.add_argument("--filter_sampled", action='store_false') 
    parser.add_argument("--filter_ordered", action='store_false') 
    parser.add_argument("--filter_diversity", action='store_false') 

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
    logging.info("START FILTERING GENERATIONS FOR %s", args.data_path)

    # NLI/CoLA tokenizer and model
    nli_model_name = "alisawuffles/roberta-large-wanli"
    args.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    args.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(args.device)

    args.cola_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(args.device)
    args.cola_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')

    # Only consider certain types of generations based on user inputted arguments
    valid_generations = []
    # Will filter if only want to use certain generations
    if args.filter_infill == True:
        valid_generations.append("likelihood-infill")
    if args.filter_gpt2 == True:
        valid_generations.append("likelihood-gpt2")
    if args.filter_keybert == True:
        valid_generations.append("keybert")
    if args.filter_content_words == True:
        valid_generations.append("content_words")
    if args.filter_rake == True:
        valid_generations.append("rake")
    valid_generations.extend(create_valid_filter_keys(args.filter_sampled, args.filter_ordered, args.filter_diversity))
    
# 1. Prepare inputs
    paragraphs = torch.load(args.data_path)
    
    # Create a new directory if it does not exist
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    
# 2. Start Filtering
    # Cycle through all data in each file in the data_dir
    processed_paragraphs = []
    for paragraph_id, data in enumerate(paragraphs):
        logging.info('Start filtering for %s / %s', paragraph_id, len(paragraphs))
        
# 2.1 Filter through each kind of generation
        for key in data.keys():
            sent_start_time = time.time()
            y_orig = data[key]['y_orig']
            prefix = data[key]['x_l']
            all_generations = data[key]['generations']
            # If original sentence was less than min_length_generate than don't filter (i.e. if generation is a string instead of list)
            if isinstance(all_generations, str):
                data[key].update(skip_filtering(all_generations))
                continue
            
            generations_keys = [k for k in list(data[key]['generations'].keys()) if (k.split("_")[0] in valid_generations) and ("_".join(k.split("_")[-3:]) in valid_generations)]
            output_sequences = [item for g_key in generations_keys for item in data[key]['generations'][g_key]]
            # get rid of any ascii code in generations
            output_sequences = [g.encode("ascii", "ignore").decode() for g in output_sequences]

# 2.2 Pre-process generations
            # Filter to only use full sentences (will cut off extra non-full sentence text)
            final_generations = create_complete_sentence(output_sequences)
            # Filter to remove repeated generations
            final_generations_norepeat = []
            for gen in final_generations:
                if gen not in final_generations_norepeat:
                    final_generations_norepeat.append(gen)
            # Filter generations that are less than 1 word)
            generations_list = [[g] for g in final_generations_norepeat if len(g)>1]
            y_orig_list = [y_orig]*len(generations_list)
            
# 2.3 Filter and save them to out_samples 
            print("Number of Generation before Filter: ", len(generations_list))
            logging.info('Start Filtering group %s out of %s', key,len(data.keys()))
            summ_filter = SummFilter(args)

            out_samples_topk = []
            out_colas_topk = []
            out_nlis_topk = []
            out_samples_greedy = []
            out_colas_greedy = []
            out_samples_threshold = []
            out_colas_threshold = []
            if generations_list == []:
                continue
            out_sample_topk, out_cola_topk, out_nli_topk = summ_filter.filter_all(y_orig_list, generations_list, args.nli_threshold, args.nli_k, args.cola_k, "topk")
            out_samples_topk.extend(out_sample_topk)
            out_colas_topk.extend(out_cola_topk)
            out_nlis_topk.extend(out_nli_topk)
            
            # 2 End by deleting summ_filter
            del summ_filter
            torch.cuda.empty_cache()
            logging.info('End Filtering group %s out of %s', key,len(data.keys()))
            print("Number of Generation after Filter (topk): ", len(out_samples_topk))

# 3. Save data/generation 
            data[key]['raw_generations'] = output_sequences
            data[key]['processed_generations'] = [x[0] for x in generations_list]
            data[key]['filtered_topk'] =  [out['pair'][0] for out in out_samples_topk]
            data[key]['cola_values_topk']= out_colas_topk
            data[key]['nli_values_topk']= out_nlis_topk
        
        processed_paragraphs.append(data)
    
    torch.save(processed_paragraphs, args.output_path)
    print(f"Results output to {args.output_path}")

    exp_total_time = time.time() - exp_start_time
    print(f"Total Filtering Time for {paragraph_id}: {exp_total_time/60} minutes")

