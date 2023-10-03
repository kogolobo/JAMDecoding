"""This file support main_authorship_keywordextraction.py
"""
import torch
import torch.nn.functional as F


# GPT2 Likelihood Method
def get_token_likelihood_gpt2(args, tokenizer,model, x_l, y_orig) :
    """
    Returns tokenized y_orig and its corresponding token-level log-likelihood.
    """
    # combine left context and the original sentence
    x_l_y_orig = [f"{x_l} {y_orig}" for x_l, y_orig in zip(x_l, y_orig)]
    # tokenize using gpt2
    input_encoding = tokenizer(x_l_y_orig, return_tensors="pt", padding = True).to(args.device)
    output = model(**input_encoding)
    y_orig_token_likelihood_ls = []
    y_orig_input_ids_ls = []
    # extract likelihoods of original token for each token in y_orig
    for i in range(output.logits.shape[0]):
        x_l_len = len(tokenizer(x_l[i]).input_ids)
        y_orig_input_ids = input_encoding.input_ids[i, x_l_len:]  # (y_orig_len)
        y_orig_vocab_distribution = F.log_softmax(output.logits[i, x_l_len:], dim=-1)  # (y_orig_len, vocab_size)
        y_orig_token_likelihood = torch.gather(
            y_orig_vocab_distribution, dim=-1, index=y_orig_input_ids.view(-1, 1)).view(-1)
        y_orig_input_ids_ls.append(y_orig_input_ids)
        y_orig_token_likelihood_ls.append(y_orig_token_likelihood)

    return y_orig_input_ids_ls, y_orig_token_likelihood_ls


def get_likelihood_constraints_gpt2(tokenizer, model, y_orig, prefix, args):
    """
    Returns list of constraing words with least token-level log-likelihood using GPT2 (autoregressive).
    """
    # Prepare constraint words by retrieving p_threshold of tokens in y_orig with least likelihood
    y_orig_input_ids_ls, y_orig_token_likelihood_ls = get_token_likelihood_gpt2(args, tokenizer, model, prefix, y_orig)
    constraint_words_ls = []
    i = 0
    for y_orig_input_ids, y_orig_token_likelihood in zip(y_orig_input_ids_ls, y_orig_token_likelihood_ls):
        tokens_with_least_likelihood = []
        least_likelihoods = sorted(y_orig_token_likelihood)[:int(len(y_orig_token_likelihood) * args.likelihood_p_threshold)]
        # select least likely tokens
        for token_id, token_likelihood in zip(y_orig_input_ids, y_orig_token_likelihood):
            if token_likelihood in least_likelihoods:
                tokens_with_least_likelihood.append([token_id])
        # decode
        tokens_with_least_likelihood = tokenizer.batch_decode(tokens_with_least_likelihood)

        constraint_words = []
        for word in tokens_with_least_likelihood:
            # for tokens without prefix space, add preceding tokens until we get prefix space
            if word[0] != " ":
                word_start_idx = y_orig[i].find(word)
                word_end_idx = word_start_idx + len(word)
                preceding_space_idx = y_orig[i][:word_start_idx].rfind(" ")
                new_word = y_orig[i][preceding_space_idx:word_end_idx]
                if all([(new_word not in w) and (w not in new_word) for w in constraint_words]):
                    constraint_words.append(y_orig[i][preceding_space_idx:word_end_idx])
            else:
                constraint_words.append(word)
        constraint_words_ls.append(constraint_words)
        i+=1
    return(constraint_words_ls)


# T5 Infill Method
def get_token_likelihood_fillin(y_orig, model, tokenizer):
    """
    Returns tokenized y_orig and its corresponding token-level log-likelihood.
    """
    y_orig_masked_ls = []
    orig_sentence_split = y_orig.split()
    # create a sentnece for each word that masks each word
    for i in range(len(orig_sentence_split)):
        masked_sentence = y_orig.split()
        masked_sentence[i] = "<extra_id_0>"
        y_orig_masked_ls.append(" ".join(masked_sentence))
    masked_matrix_tok = tokenizer(y_orig_masked_ls, return_tensors="pt", add_special_tokens = False, padding = True)['input_ids'] # MXN (M = num words, N = num max tokens)
    outputs = model.generate(masked_matrix_tok.squeeze(0), return_dict_in_generate=True, output_scores=True, max_length = 1)
    logits = outputs.scores
    probabilities_all = torch.softmax(logits[0], dim=-1).squeeze(1) # MXV V = vocab size
    probabilities_original = probabilities_all[torch.arange(len(probabilities_all)),range(len(y_orig.split()))] # extract probabilities of original tokens
    sorted_probability = torch.sort(probabilities_original) # sort them
    sorted_tokens = [orig_sentence_split[i] for i in sorted_probability.indices]
    return(sorted_tokens)
    

def get_likelihood_constraints_infill(y_orig, model, tokenizer, args):
    """
    Returns list of constraing words with least token-level log-likelihood using T5 in-filling.
    """
    y_orig_tokens_likelihood = get_token_likelihood_fillin(y_orig, model, tokenizer)
    words_with_least_likelihood = y_orig_tokens_likelihood[:int(len(y_orig_tokens_likelihood) * args.likelihood_p_threshold)]
    return(words_with_least_likelihood)


