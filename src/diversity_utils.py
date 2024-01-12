"""This file supports code in neurologic_super_fast/
"""
# import Levenshtein
import torch


# ADAPTED from: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html
class HammingDiversityLogitsProcessor():
    r"""
    :class:`transformers.LogitsProcessor` that enforces diverse beam search. Note that this logits processor is only
    effective for :meth:`transformers.PretrainedModel.group_beam_search`. See `Diverse Beam Search: Decoding Diverse
    Solutions from Neural Sequence Models <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_penalty (:obj:`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is enabled.
        num_beams (:obj:`int`):
            Number of beams used for group beam search. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for
            more details.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`")
        self._num_sub_beams = num_beams // num_beam_groups

    def generate(
        self,
        # input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ] # most popular tokens from previous group_start_idx beams
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            # scores - \lambda * token_frequency
            # scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency
            scores[(batch_idx * self._num_beams) + group_start_idx : (batch_idx * self._num_beams) + group_end_idx] -= self._diversity_penalty * token_frequency
        return scores
        
class HammingDiversityLogitsProcessor2():
    r"""
    :class:`transformers.LogitsProcessor` that enforces diverse beam search. Note that this logits processor is only
    effective for :meth:`transformers.PretrainedModel.group_beam_search`. See `Diverse Beam Search: Decoding Diverse
    Solutions from Neural Sequence Models <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_penalty (:obj:`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is enabled.
        num_beams (:obj:`int`):
            Number of beams used for group beam search. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for
            more details.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`")
        self._num_sub_beams = num_beams // num_beam_groups

    def generate(
        self,
        # input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            # scores - \lambda * token_frequency
            # scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency
            scores[(batch_idx+1) * group_start_idx : (batch_idx + 1) * group_end_idx] -= torch.abs(self._diversity_penalty * token_frequency * scores[(batch_idx+1) * group_start_idx : (batch_idx + 1) * group_end_idx])
        return scores

