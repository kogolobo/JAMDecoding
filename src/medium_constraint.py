
"""This file support main_keyword_extraction.py
"""
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.sentiment import *
import nltk
nltk.data.path.append('/myapp/')

def remove_repeated_tokens(constraint_ls, tokenizer):
  phrasal_constraints = []
  disjunctive_constraints = []
  # seperate into phrasal (single word) or disjunctive (any of multple words can satisfy constraint)
  for i,constraint in enumerate(constraint_ls):
    if isinstance(constraint, str):
      phrasal_constraints.append([constraint,i])
    else:
      disjunctive_constraints.append([constraint,i])
  # Make sure there is no overlapping tokens (not allowed in current implementation of constraint decoding)
  phrasal_constraints_split = sum([constraint[0].split() for constraint in phrasal_constraints], [])
  phrasal_constraints_split_tokens = sum([tokenizer.encode(w) for w in phrasal_constraints_split],[])
  disjunctive_constraints_norepeat = []
  current_tokens = phrasal_constraints_split_tokens
  # For each word in the disjunctive constraint only include it, if the token does not overalp
  for disjunctive_constraint in disjunctive_constraints:
    constraint = []
    for d in disjunctive_constraint[0]:
      # add space in front of word (because this will be done when creating the phrasal/disjunctive constraints)
      d_space = " "+d
      disjunctive_constraint_token = tokenizer.encode(d_space)
      overlap = [word for word in disjunctive_constraint_token if word not in current_tokens]
      if len(overlap) == len(disjunctive_constraint_token):
        constraint.append(d)
        current_tokens.extend(disjunctive_constraint_token)
      else:
        continue
    if constraint == []:
      continue
    disjunctive_constraints_norepeat.append([constraint,disjunctive_constraint[1]])
  # put constraint back in order
  all_constraints = phrasal_constraints + disjunctive_constraints_norepeat
  final_constraint_ls = []
  final_indices = np.sort([a[1] for a in all_constraints])
  for i in final_indices:
    final_constraint_ls.append([constraint_ls[0] for constraint_ls in all_constraints if constraint_ls[1] == i][0] )
  return final_constraint_ls


# Code from https://stackoverflow.com/questions/45145020/with-nltk-how-can-i-generate-different-form-of-word-when-a-certain-word-is-giv
def get_lemmatized_words(text):
    lemmatized_ls = []
    for t in text:
        # if constraint is more than one words, use it as is
        if len(t.split()) > 1:
            lemmatized_ls.append([])
            continue
        # if constraint is upper case (propernoun), use it as is
        if t.isupper():
            lemmatized_ls.append([])
            continue  
        forms = set() #We'll store the derivational forms in a set to eliminate duplicates
        # forms.add(t)
        for happy_lemma in wn.lemmas(t): #for each word lemma in WordNet
            forms.add(happy_lemma.name()) #add the lemma itself
            for related_lemma in happy_lemma.derivationally_related_forms(): #for each related lemma
                forms.add(related_lemma.name()) #add the related lemma
        lemmatized_ls.append(list(forms))
    return(lemmatized_ls)

# All code below adapted from Karadjov et al., 2017 (https://arxiv.org/abs/1707.03736)
# Replace some of the nouns, verbs or adjectives with hypernims or synonims
def get_synon_words(text, y_orig):
    synon_ls = []
    y_orig_words_pos = nltk.pos_tag(y_orig.lower().split(" "), tagset = 'universal')
    # find pos from original sentence (to make sure we capture the correct pos)
    y_orig_words = [w[0] for w in y_orig_words_pos]
    y_orig_pos = [w[1] for w in y_orig_words_pos]
    pos_tagged = nltk.pos_tag(text, tagset = 'universal')
    for tagged_word in pos_tagged:
        word = tagged_word[0]
        pos_tag = y_orig_pos[y_orig_words.index(word.lower())]
        # if constraint is more than one words, use it as is
        if len(str(word).split()) > 1:
            synon_ls.append([])
            continue
        # if constraint is upper case (propernoun), use it as is
        if str(word).isupper():
            synon_ls.append([])
            continue  
        # find synonms with same pos and different lemma than original words
        synons = get_synonim(word, pos_tag)
        hypermins = get_hypernim(word, synons, pos_tag)
        synon_ls.append(synons + hypermins)
    return(synon_ls)

def get_synonim(word, posTag):
    difSyns = get_synsets_by_pos(word, posTag)
    if difSyns:
        return difSyns
    return []

def get_synsets_by_pos(word, posTag):
    if posTag not in ['ADJ', 'ADJ_SAT', 'ADV', 'NOUN', 'VERB']:
        return None
    if posTag == 'NOUN':
        posTag = 'n'
    if posTag == 'VERB':
        posTag = 'v'
    if posTag == 'ADJ_SAT':
        posTag = 's'
    if posTag == 'ADV':
        posTag = 'r'
    if posTag == 'ADJ':
        posTag = 'a'
    #todo temp solution

    result = list()
    t = wn.synsets(word)
    # make sure it is the same part of speech as original
    syns = [ el for el in t if str(el._pos) == posTag ]
    if syns and len(syns) > 0:
        # making sure it is not the same word (or lemma string) as the original
        filtr = [ fl.lemma_names()[0] for fl in syns if str(fl.lemma_names()[0]) != word and str(fl.lemma_names()[0]).find('_') == -1 ]
        result.extend(list(set(filtr)))
        return result
    return None # If not synsets are found

def get_hypernim(word, difsyn, posTag='NOUN'):
    wordSynset = wn.synsets(word)
    if wordSynset:
        # first word is original word
        hypernim = wordSynset[0].hypernyms()
        hypernim_ls = []
        for i in range(len(hypernim)):
            hypernim_ls.append(hypernim[i].lemma_names()[0])
        return hypernim_ls
    return []


