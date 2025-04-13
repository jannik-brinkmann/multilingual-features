"""
This code has been copied and modified from [CausalGym](https://github.com/aryamanarora/causalgym).
"""

import json
import random
import re
import torch

from collections import namedtuple
from transformers import AutoModel, AutoTokenizer
from typing import Union


random.seed(42)
Tokenized = namedtuple("Tokenized", ["base", "src", "alignment_base", "alignment_src"])


class Pair:
    """
    A pair of sentences where all features except one are held constant.

    Each pair has a base sentence and a source sentence. These two sentences
    have different "types" (the value of the differing feature) and different
    "labels" (expected continuation of the sentence).
    """

    base: list[str]
    src: list[str]
    base_type: str
    src_type: str
    base_label: str
    src_label: str

    def __init__(self, base: list[str], src: list[str], base_type: str, src_type: str, base_label: str, src_label: str):
        self.base = base
        self.src = src
        self.base_type = base_type
        self.src_type = src_type
        self.base_label = base_label
        self.src_label = src_label
    

    def tokenize(self, tokenizer: AutoTokenizer, device: str="cpu") -> Tokenized:
        """Tokenize the pair and produce alignments."""
        alignment_base, alignment_src = [], []
        pos_base, pos_src = 0, 0
        for span_i in range(len(self.base)):

            # get span lengths in tokens
            tok_base = tokenizer.tokenize(self.base[span_i])
            tok_src = tokenizer.tokenize(self.src[span_i])
            alignment = [
                list(range(pos_base, pos_base + len(tok_base))),
                list(range(pos_src, pos_src + len(tok_src)))
            ]
            
            # update positions
            alignment_base.append(alignment[0])
            alignment_src.append(alignment[1])
            pos_base += len(tok_base)
            pos_src += len(tok_src)

        # tokenize full pair and return
        base_tok = tokenizer(''.join(self.base), return_tensors="pt", padding=True).to(device)
        src_tok = tokenizer(''.join(self.src), return_tensors="pt", padding=True).to(device)
        return Tokenized(base=base_tok, src=src_tok, alignment_base=alignment_base, alignment_src=alignment_src)
    

    def swap(self) -> "Pair":
        """Swap the base and src sentences."""
        return Pair(self.src, self.base, self.src_type, self.base_type, self.src_label, self.base_label)

    
    def __repr__(self):
        return f"Pair('{self.base}' > '{self.base_label}', '{self.src}' > '{self.src_label}', {self.base_type}, {self.src_type})"


class Batch:
    """
    A Batch is a collection of Pairs that have been tokenized and padded.
    The messy part is figuring out where to do interventions, so a Batch
    encapsulates the functions for computing pos_i for the intervention
    at inference time, using the tokenized pair and alignments.
    """

    def __init__(self, pairs: list[Pair], tokenizer: AutoTokenizer, device: str="cpu"):
        self.pairs = pairs

        # tokenize base and src
        tokenized = [pair.tokenize(tokenizer, device) for pair in pairs]
        max_len = max([max(x.base.input_ids.shape[-1], x.src.input_ids.shape[-1]) for x in tokenized])
        self.base = self._stack_and_pad([x.base for x in tokenized], max_len=max_len)
        self.src = self._stack_and_pad([x.src for x in tokenized], max_len=max_len)
        self.alignment_base = [x.alignment_base for x in tokenized]
        self.alignment_src = [x.alignment_src for x in tokenized]
        
        # labels
        self.base_labels = torch.LongTensor([tokenizer.encode(pair.base_label)[0] for pair in pairs]).to(device)
        self.src_labels = torch.LongTensor([tokenizer.encode(pair.src_label)[0] for pair in pairs]).to(device)
        self.base_types = [pair.base_type for pair in pairs]
        self.src_types = [pair.src_type for pair in pairs]
        self.cached_pos = {}
    
    
    def _pos_bounds(self, span1: list[int], span2: list[int]) -> list[int]:
        """Compute the bounds of a span."""
        if self.pos_strategy == "first":
            return span1[:1], span2[:1]
        elif self.pos_strategy == "last":
            return span1[-1:], span2[-1:]
        elif self.pos_strategy == "all":
            max_len = max(len(span1), len(span2))
            return span1 + [span1[-1]] * (max_len - len(span1)), span2 + [span2[-1]] * (max_len - len(span2))
    

    def compute_pos(self, strategy: str) -> torch.LongTensor:
        """Compute pos alignments as tensors."""
        # shape of alignment: [batch_size, 2, num_spans, tokens_in_span]
        # not a proper tensor though! tokens_in_span is variable, rest is constant
        if strategy in self.cached_pos:
            return self.cached_pos[strategy]
        self.pos_strategy = strategy
        assert self.pos_strategy in ["first", "last", "all"]
        rets_base, rets_src = [], []
        for batch_i in range(len(self.pairs)):
            ret_base, ret_src = [], []
            for span_i in range(len(self.alignment_src[batch_i])):
                # skip null alignments
                if len(self.alignment_base[batch_i][span_i]) == 0 or len(self.alignment_src[batch_i][span_i]) == 0:
                    ret_base.append([-1])
                    ret_src.append([-1])
                else:
                    bounds = self._pos_bounds(self.alignment_base[batch_i][span_i], self.alignment_src[batch_i][span_i])
                    ret_base.append(bounds[0])
                    ret_src.append(bounds[1])
            rets_base.append(ret_base)
            rets_src.append(ret_src)
        
        # shape: [2, batch_size, length, 1]
        # dim 0 -> src, base (the intervention code wants src first)
        ret = [rets_src, rets_base]
        self.cached_pos[strategy] = ret
        return ret


    def _stack_and_pad(self, input_list: dict, pad_token: int=0, max_len: int=100) -> dict:
        """Stack and pad a list of tensors outputs from a tokenizer."""
        input_ids = torch.stack([torch.nn.functional.pad(x.input_ids[0], (0, max_len - x.input_ids.shape[-1]), mode='constant', value=pad_token) for x in input_list])
        attention_mask = torch.stack([torch.nn.functional.pad(x.attention_mask[0], (0, max_len - x.attention_mask.shape[-1]), mode='constant', value=0) for x in input_list])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


    def __repr__(self):
        return f"Batch({len(self.pairs)} pairs)\n" + "\n".join([f"  {pair}" for pair in self.pairs])


class Dataset:
    """
    A Dataset is a template for generating minimal pairs that is loaded
    from a JSON specification.

    We importantly want examples generated from a dataset to include token-
    level alignments.
    """

    templates: list[str]
    label_vars: list[str]
    labels: dict[str, list[str]]
    variables: dict[str, Union[list[str], dict[str, list[str]]]]
    result_prepend_space: bool


    def __init__(self, data: dict):
        # load basics
        self.templates = data["templates"]
        self.template = [x for x in re.split(r"(?<=\})|(?= \{)|(?<! )(?=\{)", '<|endoftext|>' + random.choice(self.templates)) if x != '']
        self.label_vars = data["label"] if isinstance(data["label"], list) else [data["label"]]
        self.labels = data["labels"]
        self.types = list(self.labels.keys())
        self.variables = data["variables"]

        # split template into spans and find variables
        self.vars_per_span, self.span_names = [], []
        self.first_var_pos = None
        for token_i in range(len(self.template)):
            var = re.findall(r"\{(.+?)\}", self.template[token_i])
            if len(var) > 0 and self.first_var_pos is None:
                if var[0] in self.label_vars:
                    self.first_var_pos = token_i
            self.vars_per_span.append(var)
            self.span_names.append("{" + var[0] + "}" if len(var) == 1 else self.template[token_i].replace(' ', '_'))

        # other stuff
        length = {}
        for var in self.variables:
            if '.' in var:
                head_var = var.split('.')[0]
                if head_var not in length:
                    length[head_var] = len(self.variables[var])
                else:
                    assert length[head_var] == len(self.variables[var]), f"Variable {var} has length {len(self.variables[var])} but {head_var} has length {length[head_var]}"
        self.result_prepend_space = data["result_prepend_space"]


    @classmethod
    def load_from(self, template: str) -> "Dataset":
        """Load a Dataset from a json template."""
        template_file, template_name = template.split('/')
        if template_name.endswith("_inverted"):
            template_name = template_name[:-len("_inverted")]
        data = json.load(open(f"data/templates/{template_file}.json", "r"))
        return Dataset(data[template_name])
    
    
    @property
    def length(self) -> int:
        return len(self.template)    
    
    def sample_pair(self, random_order: bool=False) -> Pair:
        """Sample a minimal pair from the dataset."""
        # pick types (should differ)
        if random_order:
            base_type = random.choice(self.types)
            src_type = base_type
            while src_type == base_type:
                src_type = random.choice(self.types)
        else:
            base_type = self.types[0]
            src_type = self.types[1]

        # make template
        base, src = self.template[:], self.template[:]

        # go token by token
        stored_choices = {}
        for token_i in range(len(self.template)):
            var = self.vars_per_span[token_i]
            if len(var) == 0: continue
            var = var[0]
            var_temp = '{' + var + '}'

            # set label vars (different)
            if var in self.label_vars:
                base_choice = random.choice(self.variables[var][base_type])
                src_choice = random.choice(self.variables[var][src_type])
                base[token_i] = base[token_i].replace(var_temp, base_choice)
                src[token_i] = src[token_i].replace(var_temp, src_choice)
            # set other vars (same for both)
            elif '.' in var:
                head_var = var.split('.')[0]
                if head_var not in stored_choices:
                    stored_choices[head_var] = random.randint(0, len(self.variables[var]) - 1)
                base[token_i] = base[token_i].replace(var_temp, self.variables[var][stored_choices[head_var]])
                src[token_i] = src[token_i].replace(var_temp, self.variables[var][stored_choices[head_var]])
            else:
                choice = random.choice(self.variables[var])
                base[token_i] = base[token_i].replace(var_temp, choice)
                src[token_i] = src[token_i].replace(var_temp, choice)
        
        # get continuations
        base_label = random.choice(self.labels[base_type])
        src_label = random.choice(self.labels[src_type])
        if self.result_prepend_space:
            base_label = " " + base_label
            src_label = " " + src_label

        return Pair(base, src, base_type, src_type, base_label, src_label)
    
    
    @torch.no_grad()
    def _sample_doable_pair(self, model: AutoModel, tokenizer: AutoTokenizer, device: str="cpu", discard: set[str]=set()) -> Pair:
        """Sample a minimal pair from the dataset that is correctly labelled by a model."""

        # keep resampling until we get a pair that is correctly labelled
        correct, ct = False, 0
        while not correct:
            pair = self.sample_pair()
            if ''.join(pair.base) in discard:
                continue
            base = tokenizer(''.join(pair.base), return_tensors="pt").to(device)
            src = tokenizer(''.join(pair.src), return_tensors="pt").to(device)
            base_logits = model(**base).logits[0, -1]
            src_logits = model(**src).logits[0, -1]
            base_label = tokenizer.encode(pair.base_label)[0]
            src_label = tokenizer.encode(pair.src_label)[0]
            if base_logits[base_label] > base_logits[src_label] and src_logits[src_label] > src_logits[base_label]:
                correct = True
            ct += 1
            if ct == 20 and not correct:
                print("WARNING: could not find a doable pair after 20 iterations")
                print("Using a random pair instead")
                break
            
        return pair


    def sample_batch(
            self, tokenizer: AutoTokenizer, batch_size: int, device: str="cpu",
            model: Union[AutoModel, None]=None, discard: set[str]=set(),
            manipulate: Union[str, None]=None) -> Batch:
        """Sample a batch of minimal pairs from the dataset."""
        pairs = []

        # get the pairs
        if model is None:
            while len(pairs) < batch_size // 2:
                pair = self.sample_pair()
                ct = 0
                while ''.join(pair.base) in discard:
                    pair = self.sample_pair()
                    ct += 1
                    if ct == 20:
                        print("WARNING: could not find a pair not in the discard set after 20 iterations")
                        print("Using a random pair instead")
                        break
                pairs.append(pair)
        else:
            pairs = [
                self._sample_doable_pair(model, tokenizer, device, discard)
                for _ in range(batch_size // 2)
            ]

        # control tasks
        if manipulate == "invert":
            for i in range(len(pairs)):
                pairs[i].base_label, pairs[i].src_label = pairs[i].src_label, pairs[i].base_label
        elif manipulate == "dog-give":
            for i in range(len(pairs)):
                pairs[i].base_label = " dog" if pairs[i].base_type == self.types[0] else " give"
                pairs[i].src_label = " dog" if pairs[i].src_type == self.types[0] else " give"
        elif manipulate == "random":
            for i in range(len(pairs)):
                if random.random() < 0.5:
                    pairs[i].base_label, pairs[i].src_label = pairs[i].src_label, pairs[i].base_label
                    pairs[i].base_type, pairs[i].src_type = pairs[i].src_type, pairs[i].base_type
        
        # add flipped pairs
        for i in range(batch_size // 2):
            pairs.append(pairs[i].swap())
        
        # make batch
        return Batch(pairs, tokenizer, device)


    def sample_batches(
            self, tokenizer: AutoTokenizer, batch_size: int, num_batches: int,
            device: str="cpu", seed: int=42, model: Union[AutoModel, None]=None,
            discard: set[str]=set(), manipulate: Union[str, None]=None) -> list[Batch]:
        """Sample a list of batches of minimal pairs from the dataset."""
        random.seed(seed)
        return [self.sample_batch(tokenizer, batch_size, device, model, discard, manipulate) for _ in range(num_batches)]


class TranslationDataset(Dataset):
    """
    A TranslationDataset is a dataset of minimal pairs for translation tasks.

    This assumes that the source and target language templates use the same label variables.
    """

    def __init__(self, task: str, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_data = Dataset.load_from(task + "/" + source_lang)
        self.target_data = Dataset.load_from(task + "/" + target_lang)

    def sample_translation_pair(self, sample_type=0) -> Pair:
        """Sample a minimal translationpair from the dataset."""
        # pick types (should differ)
        base_type = self.source_data.types[sample_type]
        src_type = self.target_data.types[sample_type]

        # make templates
        base, src = self.source_data.template[:], self.target_data.template[:]

        # go token by token
        stored_choices = {}
        for token_i in range(len(self.source_data.template)):
            var = self.source_data.vars_per_span[token_i]
            if len(var) == 0: continue
            var = var[0]
            var_temp = '{' + var + '}'

            # set label vars (different)
            if var in self.source_data.label_vars:
                base_choice = random.choice(self.source_data.variables[var][base_type])
                base[token_i] = base[token_i].replace(var_temp, base_choice)
                src[token_i] = src[token_i].replace(var_temp, base_choice)
            # set other vars (same for both)
            elif '.' in var:
                head_var = var.split('.')[0]
                if head_var not in stored_choices:
                    stored_choices[head_var] = random.randint(0, len(self.source_data.variables[var]) - 1)
                base[token_i] = base[token_i].replace(var_temp, self.source_data.variables[var][stored_choices[head_var]])
                src[token_i] = src[token_i].replace(var_temp, self.target_data.variables[var][stored_choices[head_var]])
            elif var == "name":
                choice_index = random.randint(0, len(self.source_data.variables[var]) - 1)
                base_choice = self.source_data.variables[var][choice_index]
                base[token_i] = base[token_i].replace(var_temp, base_choice)
                src[token_i] = base[token_i].replace(var_temp, base_choice)
            else:
                choice_index = random.randint(0, len(self.source_data.variables[var]) - 1)
                base_choice = self.source_data.variables[var][choice_index]
                src_choice = self.target_data.variables[var][choice_index]
                base[token_i] = base[token_i].replace(var_temp, base_choice)
                src[token_i] = src[token_i].replace(var_temp, src_choice)
        
        # get continuations
        base_label = random.choice(self.source_data.labels[base_type])
        src_label = random.choice(self.target_data.labels[src_type])
        if self.source_data.result_prepend_space:
            base_label = " " + base_label
        if self.target_data.result_prepend_space:
            src_label = " " + src_label

        return Pair(base, src, base_type, src_type, base_label, src_label)
