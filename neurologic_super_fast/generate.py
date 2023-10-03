import copy
import inspect
import warnings
from typing import Optional, Iterable, Union, Callable, List, Tuple


import torch
from transformers import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging

from neurologic_super_fast.beam_sample import constrained_beam_sample
from neurologic_super_fast.beam_scorer import ConstrainedBeamSearchScorer
from neurologic_super_fast.constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from neurologic_super_fast.beam_search import constrained_beam_search
from neurologic_super_fast.util import NeuroLogicConfig

logger = logging.get_logger(__name__)


class GenerationWrapper:
    def __init__(self, model):
        self.model = model
        self.config = model.config

    @torch.no_grad()
    def generate(
        self,
        generation_config: GenerationConfig,
        neurologic_config: NeuroLogicConfig,
        no_repeat_ngram_size: Optional[Iterable[int]] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        min_length: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        inputs: Optional[torch.Tensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head.
        """
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        if neurologic_config.constraints is None:
            raise ValueError("`neurologic_config.constraints` must be specified.")

        # 1. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if (
                generation_config.pad_token_id is not None
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self.model._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to"
                f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the"
                " config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif has_default_max_length and generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        elif not has_default_max_length and generation_config.max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 6. determine generation mode
        is_constraint_beam_search_mode = (
            not generation_config.do_sample
        )
        is_constraint_beam_sample_mode = (
            generation_config.do_sample
        )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if generation_config.num_return_sequences > generation_config.num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        if generation_config.num_beams <= 1:
            raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

        if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
            raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

        final_constraints = neurologic_config.constraints

        # 11. prepare beam search scorer
        prefix_length = model_kwargs["attention_mask"].size(-1)
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            neurologic_config=neurologic_config,
            constraints=final_constraints,
            prefix_length=prefix_length,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam search
        if is_constraint_beam_search_mode:
            return constrained_beam_search(
                self.model,
                input_ids,
                neurologic_config=neurologic_config,
                bad_words_ids=bad_words_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=False,
                **model_kwargs,
            )
        elif is_constraint_beam_sample_mode:
            return constrained_beam_sample(
                self.model,
                input_ids,
                neurologic_config=neurologic_config,
                bad_words_ids=bad_words_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=False,
                **model_kwargs,
            )
        else:
            raise NotImplementedError
