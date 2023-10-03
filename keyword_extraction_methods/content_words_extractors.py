"""This file support main_authorship_keywordextraction.py
"""
import spacy
nlp = spacy.load("en_core_web_lg")

def get_contentwords_constraints(y_orig):
    content_pos = ["ADJ", "NOUN", "NUM", "PROPN", "VERB"]
    constraint_words = []
    doc = nlp(y_orig)
    current_words = []
    # Add all content words, keeping phrases together
    for i, token in enumerate(doc):
        if token.pos_ in content_pos:
            current_words.append(str(token))
            indicator = 1
        else:
            indicator = 0
        if indicator==0 and len(current_words)!=0: 
            constraint_words.append(" ".join(current_words))
            current_words = []
        if i == len(doc) and len(current_words) !=0:
            constraint_words.append(str(current_words))
    return(constraint_words)

