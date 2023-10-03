import warnings
from typing import Optional, Union
from numpy import inf

import torch
import torch.distributed as dist
from torch import nn
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import BeamSearchOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput

from neurologic_super_fast.beam_scorer import ConstrainedBeamSearchScorer
from neurologic_super_fast.util import NeuroLogicConfig, postprocess_next_token_scores

from src.diversity_utils import HammingDiversityLogitsProcessor


def constrained_beam_sample(
        self,
        input_ids: torch.LongTensor,
        neurologic_config: NeuroLogicConfig,
        bad_words_ids,
        no_repeat_ngram_size,
        min_length,
        repetition_penalty,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **constrained beam search
    decoding**.
    Parameters:
        self:
            Model for constrained_beam_search.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        neurologic_config (`NeuroLogicConfig`):
            An instance of NeuroLogicConfig.
        constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
            A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation, while satisfying a list of positive constraints. For more information, the
            documentation of [`ConstrainedBeamSearchScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.
    Return:
        [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size = len(constrained_beam_scorer._beam_hyps)
    num_beams = constrained_beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    met_constraints = [0]* beam_scores.shape[0]
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        #  CHANGING THIS TO MATCH TRANSFORMER 4.2 --> JUST RETURNS LOGITS IT GETS
        # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        next_token_scores = postprocess_next_token_scores(
        self,
        next_token_scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
        )



    # ADDED: Incldue diversity/groups here
        if neurologic_config.diversity:
            current_tokens = torch.tensor([], device=input_ids.device, dtype=torch.long)
            # start with tokens processed above
            next_token_scores_processed = next_token_scores
            # Make it so the number of beam groups = number of beams (each beam is unique from others)
            num_beam_groups = num_beams
            # instantiate diversity logit processor
            diversity_logit_processor = HammingDiversityLogitsProcessor(diversity_penalty = 500.0, num_beams= num_beams, num_beam_groups = num_beam_groups)
            # next_token_scores_processed = diversity_logit_processor.generate(scores=next_token_scores_processed , current_tokens=current_tokens, beam_group_idx=num_beams)

            for i in range(num_beam_groups):
                next_token_scores_processed = diversity_logit_processor.generate(scores=next_token_scores_processed , current_tokens=current_tokens, beam_group_idx=i)
                if i < num_beam_groups:
                # set current_tokens of beam using greedy decoding (for future beam group)
                        current_tokens = torch.argmax(next_token_scores_processed, dim=1)

            next_token_scores_processed = postprocess_next_token_scores(
                self,
                next_token_scores_processed,
                input_ids,
                no_repeat_ngram_size,
                bad_words_ids,
                cur_len,
                min_length,
                eos_token_id,
                repetition_penalty,
                batch_size,
                num_beams,
                )
        else:
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        # -------------
    
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        if neurologic_config.ordered or neurologic_config.grouping_strategy == "number":
            scores_for_all_vocab = next_token_scores_processed.clone()

        elif neurologic_config.grouping_strategy == "type":
            scores_for_all_vocab = next_token_scores.clone()

        else:
            raise NotImplementedError

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        # if all constraints are not met, then do not end the sentence
        for i in range(next_token_scores.shape[0]):
            if met_constraints[i] != len(neurologic_config.constraints[0]):
                next_token_scores[i][13] = -inf
                next_token_scores[i][764] = -inf

        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        probs = nn.functional.softmax(next_token_scores, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
        next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, _indices)

        next_indices = (next_tokens / vocab_size).long()
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = constrained_beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            scores_for_all_vocab,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        met_constraints = beam_outputs['met_constraints']

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        # increase cur_len
        cur_len = cur_len + 1

        if constrained_beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = constrained_beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]