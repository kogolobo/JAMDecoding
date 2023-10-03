from dataclasses import dataclass
from typing import List, Union, Iterable

from torch import Tensor
from transformers import PreTrainedTokenizer

from neurologic_super_fast.constraints import PhrasalConstraint, DisjunctiveConstraint, Constraint


def prepare_constraints(tokenizer: PreTrainedTokenizer, word_list: List[Union[str, List[str]]]):
    _constraints = []
    for constraint_idx, word in enumerate(word_list):
        if type(word) == str:
            _constraint = PhrasalConstraint(
                tokenizer(word, add_prefix_space=True, add_special_tokens=False).input_ids, constraint_idx
            )
        else:  # disjunctive constraints
            _constraint = DisjunctiveConstraint(
                tokenizer(word, add_prefix_space=True, add_special_tokens=False).input_ids, constraint_idx
            )
        _constraints.append(_constraint)

    return _constraints


@dataclass
class NeuroLogicConfig:
    ordered: bool = False
    sat_tolerance: int = -1
    grouping_strategy: str = "type"  # "type" or "number"

    #ADDED: diversity setting
    def __init__(self, constraints: List[List[Constraint]], ordered: bool = False,
                 constraint_prune_factor: float = -1, constraint_prune_number = 0, likelihood_prune_factor: float = -1, grouping_strategy: str = "type", diversity: bool = False, do_early_stopping: bool = True):
        self.ordered = ordered
        self.constraint_prune_factor = constraint_prune_factor
        self.constraint_prune_number = constraint_prune_number
        self.likelihood_prune_factor = likelihood_prune_factor
        self.grouping_strategy = grouping_strategy
        self.constraints = constraints
        self.diversity = diversity
        self.do_early_stopping = do_early_stopping

        self.validate_input()

    def validate_input(self):
        if -1 < self.sat_tolerance <= 1:
            raise ValueError("`sat_tolerance`, if set, should be larger than 1.")

        if self.grouping_strategy not in ["type", "number"]:
            raise ValueError("`grouping_strategy` should be either `type` or `number`")


def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores, batch_size, num_beams, input_ids, repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

            for i, banned_tokens in enumerate(banned_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores
def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens