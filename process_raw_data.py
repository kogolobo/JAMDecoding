"""This file processes raw data to be used by our authorship obfuscation method
"""
import torch
import pickle
import re
import math
import os
import logging
import sys
from argparse import ArgumentParser
from src.utils import uniform_paragraph_symbols


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="/datasets/", type=str)
    parser.add_argument("--num_authors", default=3, type=int, help="Total number of authors under observation")
    parser.add_argument("--dataset", default="amt", type=str, help="Name of dataset to test with")
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
    logging.info("START PROCESSING DATA FOR %s - %s", args.dataset, args.num_authors)
    
    # download all passages
    logging.info(f"Download Data from {cwd + args.data_dir + str(args.dataset) + '-' + str(args.num_authors) + '/X_test.pickle'}")
    data_filename = cwd + args.data_dir + str(args.dataset) + str(args.num_authors) + '/X_test.pickle'
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)
    
    # cycle through each passage
    for i in range(len(data)):
        logging.info("Processing %s out of %s", i, len(data))
        file_name = data[i][1][:-6]
        text = data[i][4]
        # replace all combinations of "\n" into "\n\t"
        text = uniform_paragraph_symbols(text)

        # Create paragraphs by splitting on "\n\t" or "\t\n"
        if args.dataset in ["amt"]:
            paragraphs = text.split("\n\t")
        else: # if blog dataset
            paragraphs = [p for p in text.split("   ") if p != ""]

        x_l = []
        y_orig = []
        j =0
        for p in paragraphs:
            p = p.replace("\'", "'")
            p = p.replace("”", "\"")
            p = p.replace("”", "\"")
            p = p.replace("“", "\"")
            # Split into sentences (using .,!, ?, .")
            sentences = re.findall(r'(.*?[.!?](?:"|\s|$))', p)
            if sentences == []:
                sentences = [s + "." for s in p.split(".")]
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
        # Save data
        print("Saving processed data to ", cwd + args.data_dir + str(args.dataset) + str(args.num_authors) + "/" + str(args.dataset) + str(args.num_authors)+ '_' + file_name + "_processed_data")
        torch.save({'x_l': x_l, 
                    'y_orig': y_orig},
                    cwd + args.data_dir + str(args.dataset) + str(args.num_authors) + "/" + str(args.dataset) + str(args.num_authors)+ '_' + file_name + "_processed_data")
    logging.info("FINISHED PROCESSING DATA FOR %s - %s", args.dataset, args.num_authors)

main()


