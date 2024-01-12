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
    parser.add_argument("--data_dir", default =None, type=str)
    parser.add_argument("--cache_dir", default ="/cache/", type=str)
    parser.add_argument("--class_model_dir", default ="/classifier_models/", type=str)
    parser.add_argument("--dataset", default ="amt", type=str)
    parser.add_argument("--num_authors", default =3, type=int)
    parser.add_argument("--save_dir", default ="/results/", type=str)
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
    abspath = os.path.abspath(__file__)
    cwd = os.path.dirname(abspath)
    os.chdir(cwd)
    args = parse_args()
    args.class_model_dir = cwd + args.class_model_dir
    args.save_dir = cwd + args.save_dir
    args.cache_dir = cwd + args.cache_dir
#0. Set Parameters and load models/data
    exp_start_time = time.time()
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',  stream=sys.stdout)
    logging.info('Arguments: %s', args)
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    logging.info("START EVALUATING FOR %s - %s", args.dataset, args.num_authors)

    # Download necessary models for evaluation
    nli_model_name = "alisawuffles/roberta-large-wanli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, cache_dir = args.cache_dir ,map_location = args.device)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir = args.cache_dir).to(args.device)

    cola_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA', cache_dir = args.cache_dir).to(args.device)
    cola_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA', cache_dir = args.cache_dir, map_location = args.device)

    # Classifier model
    attributionClassifier_Ensemble = cwd + args.class_model_dir + args.dataset + "-" + str(args.num_authors) + "/trained_model_Ensemble.sav"
    attributionClassifier_RFC = cwd + args.class_model_dir + args.dataset + "-" + str(args.num_authors) + "/trained_model_RFC.sav"
    attributionClassifier_SVC = cwd + args.class_model_dir + args.dataset + "-" + str(args.num_authors) + "/trained_model_SVC.sav"

    # Create stylometric obfuscator if want to use on text which do not have suitable generations
    if args.eval_use_stylo:
        stylo_generator = StylometricGeneration(args=args,
            device = torch.device(f"cuda:{args.device_id}")
        )

# 1. Prepare inputs
    logging.info('Prepare Data')
    if args.data_dir == None:
        args.data_dir = cwd + args.save_dir  + args.dataset + str(args.num_authors) + "/filtered/"
    # using all the final results in a data directory
    if os.path.isdir(args.data_dir):
        # download all data files in directory (only used "final" which was created by "main_filtered.py")
        dir_list = [args.data_dir + file for file in os.listdir(args.data_dir)]
        dir_list = [d for d in dir_list if ("filtered" in d) and ("final" in d)]
    # or using a specific file
    elif os.path.isfile(args.data_dir):
        dir_list = args.data_dir
    else:
        print("Error in data directory inputted")
        quit()
    # Create a new directory if it does not exist
    if not os.path.exists(cwd + args.save_dir  + args.dataset + str(args.num_authors) + "/evaluation/"):
        os.makedirs(cwd + args.save_dir  + args.dataset + str(args.num_authors) + "/evaluation/")
    currently_saved_files = os.listdir(cwd + args.save_dir  + args.dataset + str(args.num_authors) + "/evaluation/")
    
    
# 2. Start Evaluation
    results = {}
    # Cycle through all files in data_dir
    for num_text, data_dir in enumerate(dir_list):
        logging.info('Starting evaluation for %s / %s', num_text, len(dir_list))
        text_name = "_".join(data_dir.split("_")[-4:-2])
        save_filename = cwd + args.save_dir  + args.dataset + str(args.num_authors) + "/evaluation/" + "evaluation_" + args.dataset + "_" + str(text_name) + "_"
        # check to see if this file exist already, if so skip 
        saved_files_for_author = [c for c in currently_saved_files if text_name in c]
        if len([s for s in saved_files_for_author if ("final" in s) and ("evaluation" in s)]):
            logging.info('Already evaluated %s / %s', num_text, len(dir_list))
            continue
        else:    
            data = torch.load(data_dir)
        # setup list/dict
        save_idx = 0
        results[text_name] = {}
        if args.eval_use_stylo:
            results[text_name]['stylo_info'] = {}
        original_text = []
        obfuscated_text = []
        mean_nli_ls = []
        num_orig_used = 0
        num_stylo_used = 0

#2.1  Create final obfuscated generation
        if args.eval_only == False:
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
                    mean_nli_ls.append((data[key]['nli_values_topk'][cola_options.index(max(cola_options))])
            )
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
                            results[text_name]["stylo_info"][key] = stylo_method_data
                        else:
                            obfuscated_text.append(data[key]['y_orig'])
                            mean_nli_ls.append(1)
                            num_orig_used +=1
                    else:
                        obfuscated_text.append(data[key]['y_orig'])
                        mean_nli_ls.append(1)
                        num_orig_used +=1
            obfuscated_text = [g.encode("ascii", "ignore").decode() for g in obfuscated_text]
            results[text_name]["obfuscated_text"] = "".join(obfuscated_text)
            results[text_name]["original_text"] = "".join(original_text)
            results[text_name]["ratio_stylo_used"] = num_stylo_used/len(list(data.keys()))
            results[text_name]["ratio_orig_used"] = num_orig_used/len(list(data.keys()))
            results[text_name]["num_sent"] = len(list(data.keys()))
            logging.info("Total number of sentences: %s ", len(data.keys()))
            logging.info("Number of original sentences used: %s", num_orig_used)
            if len(data.keys()) == 0:
                logging.info("No Data to Generate!!! Look into %s", text_name)
                continue
        # Start evaluation

        # Classification (by all 3 models used in Mutant-X)
        logging.info("Starting evaluation: Classification")
        for class_name, classifier in zip(["Ensemble", "RFC", "SVC"], [attributionClassifier_Ensemble, attributionClassifier_RFC, attributionClassifier_SVC]):
            clf = pickle.load(open(classifier, 'rb'))
            orig_pred = clf.predict(np.asarray([ws.calculateFeatures(results[text_name]["original_text"])]))[0]
            obf_pred = clf.predict(np.asarray([ws.calculateFeatures(results[text_name]["obfuscated_text"])]))[0]
            label = "classification_" + class_name
            results[text_name]["orig_" + label] = orig_pred
            results[text_name]["obf_" + label] = obf_pred
        logging.info("Original Classification: %s", orig_pred)
        logging.info("Obfuscated Classification: %s", obf_pred)
                
        # Meteor Score
        logging.info("Starting evaluation: Meteor Score")
        meteor = evaluate.load('meteor', cache_dir= args.cache_dir)
        meteor_score = round(meteor.compute(predictions = [results[text_name]["obfuscated_text"]], references = [results[text_name]["original_text"]])['meteor'],4)
        results[text_name]["meteor_score"] = meteor_score
        logging.info("Meteor Score: %s", meteor_score)

        # NLI Score
        logging.info("Starting evaluation: NLI Score")
        # split roughly by sentences
        orig_text_split = results[text_name]["original_text"].split(".")
        obf_text_split = results[text_name]["obfuscated_text"].split(".")
        for i, o in enumerate(orig_text_split):
            if len(o) > 800:
                print("The following original text is too big and will be broken up:", i)
                num_groups = int(np.ceil(len(o)/200))
                for n in range(num_groups):
                    # cut it in half and add both to list
                    o_split = o[n * int(len(o)/num_groups): (n+1) * int(len(o)/num_groups)]
                    # override original with first half and add other half to end
                    if n == 0:
                        orig_text_split[i] = o_split
                    else: 
                        orig_text_split.append(o_split)
        nli_model_prediction_ls = []
        nli_batch_size = 5
        for i in range(len(obf_text_split)):
            options = []
            for j in range(int(np.ceil(len(orig_text_split)/nli_batch_size))):
                orig_text_batch = orig_text_split[j*nli_batch_size: (j+1)*nli_batch_size]
                input_encoding = nli_tokenizer([obf_text_split[i] for x in range(len(orig_text_batch))], orig_text_batch, padding=True, return_tensors="pt").to(args.device)
                if input_encoding['input_ids'].shape[1] >= 514:
                    print("The following generation is too big to find NLI so will divide up the obf_text")
                    print(obf_text_split[i])
                    for k in range(10):
                        cut_off = int(len(obf_text_split[i].split())/10)
                        obf_text = " ".join(obf_text_split[i].split()[k*cut_off: (k+1)*cut_off])
                        input_encoding = nli_tokenizer([obf_text for x in range(len(orig_text_batch))], orig_text_batch, padding=True, return_tensors="pt").to(args.device)
                        input_encoding = {k: v if isinstance(v, torch.Tensor) else v
                                        for k, v in input_encoding.items()}
                        nli_model_prediction = nli_model(
                            input_ids=input_encoding["input_ids"],
                            attention_mask=input_encoding["attention_mask"]
                        ).logits.softmax(dim=1).squeeze(0)
                        # if only one prediction made then use this
                        if len(orig_text_batch) == 1:
                            options.append(nli_model_prediction[1].item())
                        else:
                            for pred in nli_model_prediction:
                                options.append(pred[1].item())
                else:
                    input_encoding = {k: v if isinstance(v, torch.Tensor) else v
                                    for k, v in input_encoding.items()}
                    nli_model_prediction = nli_model(
                        input_ids=input_encoding["input_ids"],
                        attention_mask=input_encoding["attention_mask"]
                    ).logits.softmax(dim=1).squeeze(0)
                    # if only one prediction made then use this
                    if len(orig_text_batch) == 1:
                        options.append(nli_model_prediction[1].item())
                    else:
                        for pred in nli_model_prediction:
                            options.append(pred[1].item())
            nli_model_prediction_ls.append(np.max(options))
        results[text_name]['nli_score'] = round(np.mean(nli_model_prediction_ls),4)
        logging.info("NLI Score: %s", round(np.mean(nli_model_prediction_ls),4))


    # Cola score
        cola_ls = []
        for i in range(len(obf_text_split)):
            if obf_text_split[i] == '':
                continue
            # if text is too big, break it up 
            if len(obf_text_split[i]) > 800:
                o = obf_text_split[i]
                print("The following original text is too big and will be broken up:", i)
                num_groups = int(np.ceil(len(o)/200))
                for n in range(num_groups):
                    # cut it in half and add both to list
                    o_split = o[n * int(len(o)/num_groups): (n+1) * int(len(o)/num_groups)]
                    cola_ls.append(cola_score(o_split, cola_tokenizer, cola_model, args.device))
            else:
                cola_ls.append(cola_score(obf_text_split[i], cola_tokenizer, cola_model, args.device))
        results[text_name]['cola_score'] = round(np.mean(cola_ls),4)
        logging.info("Cola Score: %s", round(np.mean(cola_ls),4))

        torch.save(results[text_name], save_filename + "final")
        exp_total_time = time.time() - exp_start_time

        print(f"Evaluation Time for this document: {exp_total_time/60} minutes")
        print(f"Results output to f{save_filename + 'final'}")
    exp_total_time = time.time() - exp_start_time
    print(f"Total Evaluation Time: {exp_total_time/60} minutes")


