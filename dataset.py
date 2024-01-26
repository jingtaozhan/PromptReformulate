import re
import os
import json
import torch
import random
import string
import logging
import numpy as np
import transformers
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from transformers.data.data_collator import default_data_collator

from .template import AdvancedString

logger = logging.getLogger(__name__)
    

class PromptRewriteDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 is_train, 
                 input_template, 
                 output_template,
                 delta_clip = None, 
                 delta_aesthetic = None, 
                 delta_overall = None,
                 delta_phrase_cnt = None,
                 length_compare = None,
                 ):
        super(PromptRewriteDataset, self).__init__()
        self.tokenizer = tokenizer

        self.data = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.delta_clip = delta_clip
        self.delta_aesthetic = delta_aesthetic
        self.delta_overall = delta_overall
        self.length_compare = length_compare
        self.delta_phrase_cnt = delta_phrase_cnt
        self.input_template = AdvancedString(self.tokenizer.bos_token + input_template)
        self.output_template = AdvancedString(output_template + self.tokenizer.eos_token)
        
        logger.info(f"Input Template: {self.input_template}")
        logger.info(f"Output Template: {self.output_template}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]

        input_values = {
            "prompt": item['prompt'],
            "init_clip": item['init_clip'],
            "init_aesthetic": item['init_aesthetic'],
            "init_overall": item['init_overall'],
            "new_clip": item['new_clip'] if self.delta_clip is None else item['init_clip'] + self.delta_clip,
            "new_aesthetic": item['new_aesthetic'] if self.delta_aesthetic is None else item['init_aesthetic'] + self.delta_aesthetic,
            "new_overall": item['new_overall'] if self.delta_overall is None else item['init_overall'] + self.delta_overall,
            "new_phrase_cnt": item['new_phrase_cnt'] if self.delta_phrase_cnt is None else item['init_phrase_cnt'] + self.delta_phrase_cnt,
            "length_compare": item['length_compare'] if self.length_compare is None else self.length_compare,
            "max_clip": item['max_clip'],
            "max_aesthetic": item['max_aesthetic'],
            "max_overall": item['max_overall'],
        }
        input_values['new_clip'] = max(min(input_values['new_clip'], item['max_clip']), item['min_clip'])
        input_values['new_aesthetic'] = max(min(input_values['new_aesthetic'], item["max_aesthetic"]), item['min_aesthetic'])
        input_values['new_overall'] = max(min(input_values['new_overall'], item["max_overall"]), item['min_overall'])
        # input_values = self._pad_num_with_zero(input_values)

        if self.is_train:
            input_values['rewritten_prompt'] = item['rewritten_prompt']

        input_text = self.input_template.format(**input_values)
        if self.is_train:
            output_text = self.output_template.format(**input_values)
            if isinstance(self.tokenizer, transformers.LlamaTokenizer) or isinstance(self.tokenizer, transformers.LlamaTokenizerFast):
                text = input_text + " " + output_text
            else:
                text = input_text + " " + output_text
                output_text = " " + output_text
        else:
            text = input_text
        tokenizer_output = self.tokenizer(
            text, 
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False
        )
        input_ids = tokenizer_output.input_ids[0]
        attention_mask = tokenizer_output.attention_mask[0]
        
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.is_train:
            annotation_input_len = len(self.tokenizer(input_text, add_special_tokens=False).input_ids)
            annotation_output_len = len(self.tokenizer(output_text, add_special_tokens=False).input_ids)
            if annotation_input_len + annotation_output_len != attention_mask.sum() and attention_mask.sum() != len(attention_mask):
                logger.warning(f"annotation_input_len + annotation_output_len != attention_mask.sum()  {annotation_input_len} {annotation_output_len} {attention_mask.sum()} \"{input_text}\"  \"{output_text}\" ")

            labels = input_ids.clone()
            labels[:annotation_input_len] = -100
            labels[labels==self.tokenizer.pad_token_id] = -100
                
            output["labels"] = labels

        return output

    def _pad_num_with_zero(self, item):
        """
            For init/new clip/aesthetic/overall, we pad these numbers according to max clip/aesthetic/overall respectively
            If max_clip is 99, we pad the init_clip and new_clip to 2 digits, e.g. 1 -> 01, 10 -> 10
            If max_overall is 999, we pad the init_overall and new_overall to 2 digits, e.g. 1 -> 001, 10 -> 010

            input item example: {"init_clip": 4, "max_clip": 99, ...}
            output example {"init_clip": 04, "max_clip": 99, ...}
        """
        for metric in ["clip", "aesthetic", "overall"]:
            for k in [f"init_{metric}", f"new_{metric}"]:
                item[k] = str(item[k]).zfill(len(str(item[f"max_{metric}"])))
        return item
        
    def show_example(self):
        idx = random.choice(range(len(self)))
        logger.info(f"Example-Raw-data: {self.data[idx]}")
        output = self[idx]
        logger.info(f"Example-Input-Text: {self.tokenizer.decode(output['input_ids'], skip_special_tokens=True)}")
        if self.is_train:
            logger.info(f"Example-Input-Label: {self.tokenizer.decode(torch.clip(output['labels'], min=0), skip_special_tokens=True)}")

