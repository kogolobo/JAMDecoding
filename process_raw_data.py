"""This file processes raw data to be used by our authorship obfuscation method 
"""
import torch
import pickle
import re
import math
import os
import logging
import sys
import spacy
from typing import List, Tuple
from datasets import load_dataset
from argparse import ArgumentParser
from src.utils import uniform_paragraph_symbols

def split_sentences(text: str, nlp) -> List[Tuple[int, str]]:
    # Output format: (sentence_id, sentence_text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Location of raw data", required=True)
    parser.add_argument("--input_key", default="fullText", type=str, help="Key to extract text from")
    parser.add_argument("--output_path", type=str, help="Path to write to", required=True)
    parser.add_argument("--max_sentence_length", default=5, type=int, help="Max number of sentences in a paragraph")
    return parser.parse_args()

def main():
    # Set working directory to this file  
    abspath = os.path.abspath(__file__)
    cwd = os.path.dirname(abspath)
    os.chdir(cwd)
    args = parse_args()
    
    # Set logging 
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',  stream=sys.stdout)
    args = parse_args()
    logging.info('Arguments: %s', args)
    logging.info("START PROCESSING DATA FOR %s", args.data_path)
    
    # download all passages
    data = load_dataset('json', data_files=args.data_path)['train'][args.input_key]
    nlp = spacy.load("en_core_web_sm")
    
    # cycle through each passage  
    processed_data = []
    for i in range(len(data)):
        logging.info("Processing %s out of %s", i, len(data))
        # replace all combinations of "\n" into "\n\t"
        # text = uniform_paragraph_symbols(text)

        # Create paragraphs by splitting on "\n\t" or "\t\n"
        # if args.dataset in ["amt"]:
        #     paragraphs = text.split("\n\t")
        # else: # if blog dataset
        #     paragraphs = [p for p in text.split("   ") if p != ""]
        paragraphs = [data[i]]
        x_l = []
        y_orig = []
        j =0
        for p in paragraphs:
            p = p.replace("\'", "'")
            p = p.replace("”", "\"")
            p = p.replace("”", "\"")
            p = p.replace("“", "\"")
            p = re.sub(r'([.!?])\n\n', r'\1 ', p)
            p = p.replace("\n\n", ". ")
            # Split into sentences (using .,!, ?, .")
            # sentences = re.findall(r'(.*?[.!?](?:"|\s|$))', p)
            # if sentences == []:
            #     sentences = [s.strip() + "." for s in p.split(".")]
            sentences = split_sentences(p, nlp)
            
            sentences = [s for s in sentences if s not in ["", "/n"]]
            # make sure paragraphs are not too long, if they are split them up
            num_splits = math.ceil(len(sentences) / args.max_sentence_length)
            for n in range(num_splits):
                sentences_group = sentences[n*args.max_sentence_length:(n+1)*args.max_sentence_length]
                for i, s in enumerate(sentences_group):
                    # remove space before sentences
                    s = s.lstrip()
                    if i == 0:
                        # if it is the first sentence of the whole text, set x_l equal to y_orig
                        if (n == 0) and (j == 0):
                            x_l.append(s)
                        else:
                        # if it is the first sentence of a paragraph, set x_l equal to last sentence of previous paragraph
                            x_l.append(last_sentence_paragraph)
                        y_orig.append(s)
                    else:
                        # set x_l to cumulation of sentences in the paragraph so far
                        x_l.append(" ".join(sentences_group[0:i]))
                        y_orig.append(s)
                    if i+1 == len(sentences_group):
                        last_sentence_paragraph = s
                        j+=1        
        
        processed_data.append({'x_l': x_l, 'y_orig': y_orig})
    
    # Save data
    print("Saving processed data to ", args.output_path)
    torch.save(processed_data,args.output_path)
    logging.info("FINISHED PROCESSING DATA FOR %s", args.data_path)

main()


