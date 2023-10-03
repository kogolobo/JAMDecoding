"""This file support main_filter.py
    This code was adapted from code provided by https://github.com/jaehunjung1
"""
from argparse import Namespace
from typing import List, Tuple, Dict
import itertools
import numpy as np
import torch
from src.utils import calc_cola_score


class SummFilter:
    def __init__(self, args: Namespace):
        self.args = args
        self.device = args.device
        self.cola_threshold = args.cola_threshold
        self.length_threshold = 1.5
        self.cola_tokenizer = args.cola_tokenizer
        self.cola_model = args.cola_model
        self.nli_tokenizer = args.nli_tokenizer
        self.nli_model = args.nli_model
        self.nli_batch_size = args.nli_batch_size
    
    def filter_all(self, y_orig_list: List[dict], generation_list: List[List[str]], nli_threshold, nli_k, cola_k, nli_filter) -> List[Dict]:
        # Find NLI values for each generation
        type1_type2_indices, nli_values = self.filter_nli(y_orig_list, generation_list, nli_threshold, nli_k, nli_filter, self.nli_batch_size)

        out_y_orig_list = []
        out_colas = []
        cola_ls = []
        generation_ls = []
        out_sample = None
        # Filter with NLI top-k (type 1 and 2)
        type1_2_generations_list = [generation_list[index[0]][0] for index in type1_type2_indices]

        # Filter each generation (which made it base nli fitler) with CoLA threshold
        for type1_2_generation in type1_2_generations_list:
            if type1_2_generation != "":
                if nli_filter != "topk":
                    # pass a cola threshold
                    cola = calc_cola_score(type1_2_generation, self.cola_model, self.cola_tokenizer, self.device)
                    if cola[0] > self.cola_threshold:
                        out_sample = {
                            "pair": (type1_2_generation, y_orig_list[0]),
                        }
                        out_cola = cola[0]
                else:
                    cola = calc_cola_score(type1_2_generation, self.cola_model, self.cola_tokenizer, self.device)
                    cola_ls.append(cola[0])
                    generation_ls.append(type1_2_generation)
        
        if nli_filter == "topk":
            # sort by cola and choose top-k
            sorted_cola = np.argsort(cola_ls)
            cola_index = sorted_cola[int(-1*cola_k):]
            out_cola = [cola_ls[i] for i in cola_index]
            out_sample = []
            out_nli_values = []
            for i in cola_index:
                out_sample.append({"pair": (generation_ls[i], y_orig_list[0])})
                out_nli_values.append(nli_values[i])

            out_y_orig_list.append(out_sample)
            out_colas.append(out_cola)
            return out_y_orig_list[0], out_colas[0], out_nli_values
        
        else:
            if out_sample is not None:
                out_y_orig_list.append(out_sample)
                out_colas.append(out_cola)
            return out_y_orig_list, out_colas    

    def filter_nli(self, y_orig_list: List[str], generation_list: List[List[str]], nli_threshold, nli_k, nli_filter, nli_batch_size) -> Tuple[List, List]:
        # Returns (list of y_summ that can serve as reference, list of y_summ that can serve as summary)
        assert all([len(generation) == len(generation_list[0]) for generation in generation_list]), \
            "All y_summs in `generation_list` should be in equal length."
        num_generations= len(generation_list[0])
        y_orig_list = [[y_orig] for y_orig in y_orig_list]
        
        # Find NLI scores (both ways) and filter based on NLI scores
        prediction1, nli_values1 = self.infer_nli(generation_list, y_orig_list, nli_threshold, nli_k, nli_filter, nli_batch_size)
        prediction1 = prediction1.view(-1, num_generations)
        prediction2, nli_values2 = self.infer_nli(y_orig_list, generation_list, nli_threshold, nli_k, nli_filter, nli_batch_size)
        prediction2 = prediction2.view(-1, num_generations)

        # Index of generations with highest entailment
        generations_as_reference1 = torch.ne(prediction1, 0).nonzero().tolist()
        generations_as_reference2 = torch.ne(prediction2, 0).nonzero().tolist()

        # Use any generation which has a nli_value > nli_threshold (from either direction)
        generations_as_reference = generations_as_reference1
        generations_as_reference.extend(p2 for p2 in generations_as_reference2 if p2 not in generations_as_reference)

        # Generations who are entailed from both sides (above the threshold) will hold more weight
        nli_values = [nli1 + nli2 for nli1, nli2 in zip(nli_values1, nli_values2)]
        nli_values = [nv for nv in nli_values if nv != 0]

        return generations_as_reference, nli_values

    def infer_nli(self, premise_list: List[List[str]], hypothesis_list: List[List[str]], nli_threshold, nli_k, nli_filter, nli_batch_size) -> torch.LongTensor:
        # Many-to-Many NLI inference
        batch_premise_hypothesis = []  # [(prem1, hypo1), (prem2, hypo2), ...]
        for premise, hypothesis in zip(premise_list, hypothesis_list):
            batch_premise_hypothesis.extend(itertools.product(premise, hypothesis))
        batch_premise, batch_hypothesis = list(zip(*batch_premise_hypothesis))

        prediction = []
        prediction_ls = []
        # Can change '5' to be any number needed to not run out of memory
        for i in range(int(np.ceil(len(batch_premise)/5))):
            input_encoding = self.nli_tokenizer(batch_premise[i*5:(i+1)*5], batch_hypothesis[i*5:(i+1)*5], padding=True, return_tensors="pt")
            input_encoding = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                            for k, v in input_encoding.items()}
            nli_values = []
            if nli_filter == 'greedy':
                model_prediction = torch.argmax(self.nli_model(
                    input_ids=input_encoding["input_ids"],
                    attention_mask=input_encoding["attention_mask"]
                ).logits, dim=-1)
                for t in model_prediction:
                    if t == 1:
                        prediction.append(1)
                    else:
                        prediction.append(0)
            elif nli_filter == 'threshold':
                model_prediction = self.nli_model(
                    input_ids=input_encoding["input_ids"],
                    attention_mask=input_encoding["attention_mask"]
                ).logits.softmax(dim=1).squeeze(0)
                # Change: if prediciton for entailment > threshold than include 
                prediction = []
                for t in model_prediction:
                    if t[1]>nli_threshold:
                        prediction.append(1)
                    else:
                        prediction.append(0)
            # 0: contradiction, 1: entailment, 2: neutral
            elif nli_filter == "topk":
                # Calculate NLI prediction
                model_prediction = self.nli_model(
                    input_ids=input_encoding["input_ids"],
                    attention_mask=input_encoding["attention_mask"]
                ).logits.softmax(dim=1).squeeze(0)
                # Single out the 'entailment' value
                if (len(list(model_prediction.size())) == 1):
                    prediction_ls.append(model_prediction[1].item())
                else:
                    for t in model_prediction:
                        prediction_ls.append(t[1].item())
        if nli_filter == "topk":
            sorted_prediction = np.argsort(prediction_ls)
            prediction_index = sorted_prediction[int(-1*nli_k):]
            for i in range(len(batch_premise)):
                if i in prediction_index:
                    prediction.append(1)
                    nli_values.append(prediction_ls[i])
                else:
                    prediction.append(0)
                    nli_values.append(0)
        return torch.tensor(prediction), nli_values

    def filter_length(self, y_orig: str, y_summ: str, y_summ_as_reference: bool) -> bool:
        if y_summ_as_reference:
            return len(y_orig) < self.length_threshold * len(y_summ)
        else:
            return len(y_summ) < self.length_threshold * len(y_orig)

def replace_below_threshold(list, threshold):
    """
    This function replaces values in a list with 0 if they are not above a threshold.
    """
    for i in range(len(list)):
        if list[i] < threshold:
            list[i] = 0
    return list
