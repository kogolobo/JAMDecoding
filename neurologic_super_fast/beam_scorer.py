import warnings
from collections import UserDict
from itertools import zip_longest
from typing import List, Optional, Tuple

# import ipdb
import numpy as np
import torch
from transformers import BeamScorer

from neurologic_super_fast.constraints import Constraint, ConstraintListState
from neurologic_super_fast.util import NeuroLogicConfig


class ConstrainedBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.
    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        max_length (`int`):
            The maximum length of the sequence to be generated.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[List[Constraint]]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """

    def __init__(
        self,
        neurologic_config: NeuroLogicConfig,
        batch_size: int,
        num_beams: int,
        constraints: List[List[Constraint]],
        prefix_length: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints

        self.neurologic_config = neurologic_config
        self.prefix_length = prefix_length

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to ConstrainedBeamSearchScorer is deprecated and has no effect. "
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ", or `group_beam_search(...)`."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def make_constraint_states(self, n, batch_idx):
        return [ConstraintListState([constraint.copy() for constraint in self.constraints[batch_idx]],
                                    self.neurologic_config.ordered) for _ in range(n)]

    def check_completes_constraints(self, sequence, batch_idx):
        new_state = self.make_constraint_states(1, batch_idx)[0]
        new_state.reset(sequence)
        return new_state.completed

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        scores_for_all_vocab: torch.FloatTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> UserDict[str, torch.FloatTensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            scores_for_all_vocab (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`):
                The scores of all tokens in the vocabulary for each of the beam hypotheses.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
        Return:
            `UserDict`: A dictionary composed of the fields as defined above:
                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of
                  all
                non-finished beams.
                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be
                  added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """

        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
        all_batch_banks = []
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence.
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue

                    completes_constraint = self.check_completes_constraints(
                        input_ids[batch_beam_idx][self.prefix_length:].cpu().tolist(), batch_idx
                    )

                    if completes_constraint:
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            next_score.item(),
                        )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            new_scores, new_tokens, new_indices, all_banks = self.step_sentence_constraint(
                batch_idx,
                input_ids,
                scores_for_all_vocab,
                next_beam_scores[batch_idx],
                next_beam_tokens[batch_idx],
                next_beam_indices[batch_idx],
            )
            all_batch_banks.extend(all_banks.tolist())

            if new_scores.size(-1) != next_beam_scores[batch_idx].size(-1):
                ipdb.set_trace()

            next_beam_scores[batch_idx] = new_scores
            next_beam_tokens[batch_idx] = new_tokens
            next_beam_indices[batch_idx] = new_indices

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
                "met_constraints": [int(c) for c in all_batch_banks]
            }
        )

    def step_sentence_constraint(
        self,
        batch_idx: int,
        input_ids: torch.LongTensor,
        vocab_scores: torch.FloatTensor,
        sent_beam_scores: torch.FloatTensor,
        sent_beam_tokens: torch.LongTensor,
        sent_beam_indices: torch.LongTensor,
        push_progress: bool = False,
    ):
        # sent_beam_tokens are the next {num_beams} number of tokens that are under consideration for this beam
        # (candidate next tokens)

        # 1. Adding "advance_tokens"
        #     using ConstraintStateList.advance(), we propose new tokens to be added into this "candidate list" that will
        #     advance us in fulfilling the constraints.

        # 2. Selecting best candidates such that we end up with highest probable candidates
        #     that fulfill our constraints.

        orig_len = sent_beam_indices.size(0)
        device = sent_beam_indices.device

        # initialize states
        topk_constraint_states = self.make_constraint_states(orig_len, batch_idx)
        advance_constraint_states = self.make_constraint_states(orig_len, batch_idx)

        sidx, eidx = batch_idx * orig_len, (batch_idx + 1) * orig_len
        this_batch_input_ids = input_ids[sidx:eidx]
        this_batch_token_scores = vocab_scores[sidx:eidx]
        full_hypotheses = torch.cat((input_ids[sent_beam_indices], sent_beam_tokens.unsqueeze(-1)), dim=-1)

        # need to make new hypothesis that advance the constraints
        track_new = {
            "new_seqs": full_hypotheses.tolist(),
            "new_states": [],
            "new_indices": [],
            "new_tokens": [],
            "new_scores": []}
        for seq_idx, pre_seq in enumerate(this_batch_input_ids):
            # pre_seq = ith sequence generated before this step.

            # input_ids -> (topk) generic beam search best model next tokens
            #           -> (advance) constraints forcing the next token
            # either way, we need to sort them into "banks" later, so store a "ConstraintListState" for all types of
            # hypotheses.

            topk_state = topk_constraint_states[seq_idx]
            topk_state.reset(full_hypotheses[seq_idx][self.prefix_length:].cpu().tolist())

            advance_state = advance_constraint_states[seq_idx]
            advance_state.reset(pre_seq[self.prefix_length:].cpu().tolist())

            if not advance_state.completed:
                advance_tokens = torch.LongTensor(advance_state.advance()).to(device)
                for advance_token in advance_tokens:
                    # since adding each `advance_token` leads to a different hypothesis, create new state instance.
                    new_state = advance_state.copy(stateful=True)
                    new_state.add(advance_token.cpu().tolist())

                    advance_seq = torch.cat((pre_seq, advance_token.unsqueeze(0)), -1).cpu().tolist()
                    if advance_seq not in track_new["new_seqs"]:
                        if not self.neurologic_config.ordered or new_state.is_ordered():
                            # prevent duplicates, which are basically bound to happen in this process.
                            track_new["new_seqs"].append(advance_seq)
                            track_new["new_indices"].append(sidx + seq_idx)  # idx -> global idx across all the batches
                            track_new["new_tokens"].append(advance_token)
                            track_new["new_scores"].append(this_batch_token_scores[seq_idx, advance_token])
                            track_new["new_states"].append(new_state)
            elif push_progress:
                # Basically, `sent_beam_indices` often chooses very little among `input_ids` the generated sequences that
                # actually fulfill our constraints. For example, let constraints == ["loves pies"] and

                #     pre_seq_1 = "The child loves pies and" pre_seq_2 = "The child plays in the playground and"

                # Without this step, if `sent_beam_indices` is something like [1,1], then
                #     1. `pre_seq_1` won't be added to the list of (topk) hypothesis since it's not in the indices and
                #     2.  it won't be added to the list of (advance) hypothesis since it's completed already. (this is
                #         the else part of `if constraints_completed[seq_idx]`)
                #     3. it ends up simply getting removed from consideration.

                # #3 might be fine and actually desired, since it's likely that it's a low-probability output anyways,
                # especially if it's not in the list of `sent_beam_indices`. But this often leads to lengthened beam
                # search times, since completed sequences keep getting removed after all this effort for constrained
                # generation.

                # Here, we basically take `pre_seq_1` and to "push" it into the considered list of hypotheses, by simply
                # appending the next likely token in the vocabulary and adding it to the list of hypotheses.

                new_score, new_token = torch.max(this_batch_token_scores[seq_idx], 0)  # some next probable token
                advance_seq = torch.cat((pre_seq, new_token.unsqueeze(0)), -1)

                advance_state = advance_constraint_states[seq_idx]

                advance_seq = advance_seq.cpu().tolist()

                advance_state.reset(advance_seq[self.prefix_length:])

                if advance_seq not in track_new["new_seqs"]:
                    if not self.neurologic_config.ordered or advance_state.is_ordered():
                        # but still don't want to have duplicates
                        track_new["new_seqs"].append(advance_seq)
                        track_new["new_indices"].append(seq_idx)
                        track_new["new_tokens"].append(new_token)
                        track_new["new_scores"].append(new_score)
                        track_new["new_states"].append(advance_state)
        # Save for later                
        all_num_satisfied_constraints = torch.Tensor([one.get_bank() for one in topk_constraint_states]).to(device)
        if len(track_new["new_indices"]) > 0:
            new_scores = torch.stack(track_new["new_scores"]).to(device)

            # ADDED: only add new constraints if they have a likelihood in the top-p of the current list of candidates
            keep_index = []
            for i, ns in enumerate(new_scores):
                seq = sorted(torch.cat((sent_beam_scores, ns.unsqueeze(0)), dim=-1))
                rank = (seq.index(ns)+1)/len(sent_beam_scores) # smaller number = lower likelihood percentile
                if rank > self.neurologic_config.likelihood_prune_factor:
                    keep_index.append(i)
            if keep_index == []:
                all_states = topk_constraint_states
                all_tokens = sent_beam_tokens
                all_scores = sent_beam_scores
                all_indices = sent_beam_indices
            else:
                keep_index = torch.tensor(keep_index)
                new_tokens = torch.index_select(torch.stack(track_new["new_tokens"]).to(device), 0, keep_index.to(device))
                new_indices = torch.index_select(torch.tensor(track_new["new_indices"]).to(device), 0,  keep_index.to(device))
                new_scores = torch.index_select(torch.stack(track_new["new_scores"]).to(device), 0,  keep_index.to(device))
                new_states = [track_new["new_states"][i] for i in keep_index.tolist()]


                all_states = topk_constraint_states + new_states
                all_tokens = torch.cat((sent_beam_tokens, new_tokens), dim=-1)
                all_scores = torch.cat((sent_beam_scores, new_scores), dim=-1)
                all_indices = torch.cat((sent_beam_indices, new_indices), dim=-1)

            # Store for later use
            all_num_satisfied_constraints = torch.Tensor([one.get_bank() for one in all_states]).to(device)

            # Grouping based on number of satisfied constraints
            if self.neurologic_config.grouping_strategy == "number":
                all_banks = torch.Tensor([one.get_bank() for one in all_states]).to(device)
                zipped = all_banks * 100 + all_scores
                indices = zipped.sort(descending=True).indices
                sorted_banks = all_banks[indices]

                # Rearrange index to add 1 best beam from each bank, then second best beam from each bank, ...
                counter = -1
                cur_bank = sorted_banks[0]
                increments = []
                for bank in sorted_banks:
                    if bank == cur_bank:
                        counter += 1
                    else:
                        counter = 0
                        cur_bank = bank
                    increments.append(counter)
                rearranger = torch.tensor(np.argsort(increments, kind="mergesort"))
                indices = indices[rearranger][:orig_len]

            # Grouping based on types of satisfied constraints
            elif self.neurologic_config.grouping_strategy == "type":
                # Filter out beams with satisfaction tolerance
                if self.neurologic_config.constraint_prune_factor < 1:
                    all_num_satisfied_constraints = torch.Tensor([one.get_bank() for one in all_states]).to(device)
                    satisfaction_threshold = all_num_satisfied_constraints.max() - self.neurologic_config.constraint_prune_number
                    satisfaction_filtered_results = all_num_satisfied_constraints >= satisfaction_threshold

                    all_states = [state for state, filtered_result in zip(all_states, satisfaction_filtered_results)
                                  if filtered_result]
                    all_tokens = all_tokens[satisfaction_filtered_results]
                    all_scores = all_scores[satisfaction_filtered_results]
                    all_indices = all_indices[satisfaction_filtered_results]

                # Sort based on scores
                sorted_indices = all_scores.sort(descending=True).indices
                all_states = [all_states[idx] for idx in sorted_indices]
                all_tokens = all_tokens[sorted_indices]
                all_scores = all_scores[sorted_indices]
                all_indices = all_indices[sorted_indices]

                types_of_satisfied_constraints = []
                for state in all_states:
                    if state.get_complete_constraints() not in types_of_satisfied_constraints:
                        types_of_satisfied_constraints.append(state.get_complete_constraints())
                state_indices_grouped_by_banks = [
                    [idx for idx in range(len(all_states)) if all_states[idx].get_complete_constraints() == t]
                    for t in types_of_satisfied_constraints
                ]

                # Rearrange index to add 1 best beam from each bank, then second best beam from each bank, ...
                indices = []
                for zipped_elements in zip_longest(*state_indices_grouped_by_banks):
                    indices.extend([state_idx for state_idx in zipped_elements if state_idx is not None])
                indices = torch.LongTensor(indices)[:orig_len]

            else:
                raise NotImplementedError
            

            sent_beam_scores = all_scores[indices]
            sent_beam_tokens = all_tokens[indices]
            sent_beam_indices = all_indices[indices]

            # Pad the outputs if less then num_beams have been left
            if sent_beam_scores.size(-1) < orig_len:
                pad_size = orig_len - sent_beam_scores.size(-1)
                score_pad = torch.Tensor([-1e9] * pad_size).to(device)
                token_pad = torch.Tensor([sent_beam_tokens[-1]] * pad_size).to(device)
                index_pad = torch.Tensor([sent_beam_indices[-1]] * pad_size).to(device)

                sent_beam_scores = torch.cat([sent_beam_scores, score_pad], dim=-1)
                sent_beam_tokens = torch.cat([sent_beam_tokens, token_pad], dim=-1)
                sent_beam_indices = torch.cat([sent_beam_indices, index_pad], dim=-1)

        return sent_beam_scores, sent_beam_tokens, sent_beam_indices, all_num_satisfied_constraints

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> UserDict[str, torch.FloatTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams

            ids_collect = []
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]

                completes_constraint = self.check_completes_constraints(
                    input_ids[batch_idx][self.prefix_length:].cpu().tolist(), batch_idx
                )

                if completes_constraint:
                    beam_hyp.add(final_tokens, final_score)
                    ids_collect.append(beam_id)

            # due to overly complex constraints or other factors, sometimes we can't guarantee a successful
            # generation. In these cases we simply return the highest scoring outputs.
            if len(ids_collect) < self.num_beam_hyps_to_keep:
                for beam_id in range(self.num_beams):
                    if beam_id not in ids_collect:
                        batch_beam_idx = batch_idx * self.num_beams + beam_id
                        final_score = final_beam_scores[batch_beam_idx].item()
                        final_tokens = input_ids[batch_beam_idx]
                        beam_hyp.add(final_tokens, final_score)
                    if len(ids_collect) >= self.num_beam_hyps_to_keep:
                        break

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1

        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret