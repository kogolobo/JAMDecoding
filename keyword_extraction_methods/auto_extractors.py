"""This file support main_authorship_keyword_extraction.py
"""
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import nltk



# KeyBERT
def get_keybert_constraints(y_orig, args):
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0", cache_folder=args.cache_dir)
    keyword_model = KeyBERT(model=sentence_model)
    keywords = keyword_model.extract_keywords(y_orig, top_n=y_orig.count(" ") // args.keybert_length)
    # Can change this to have a threshold for keybert values
    constraint_list = [k[0] for k in keywords if k[1] >= 0.00]
    # Must have at least 1 keyword
    i=0
    divide_by = args.keybert_length
    while (len(constraint_list) <1) and (divide_by > 1):
        divide_by = (args.keybert_length-i)
        keywords = keyword_model.extract_keywords(y_orig, top_n=y_orig.count(" ") // divide_by)
        constraint_list = [k[0] for k in keywords if k[1] >= 0.00]
        i+=1
    return(constraint_list)

# RAKE
def get_rake_constraints(y_orig):
    nltk.download('stopwords')
    rake_nltk_var1 = Rake()
    rake_nltk_var1.extract_keywords_from_text(y_orig)
    constraint_words = rake_nltk_var1.get_ranked_phrases()
    return(constraint_words)


