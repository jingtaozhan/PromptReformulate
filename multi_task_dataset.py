import json
import torch
import random
import logging
import transformers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import default_data_collator

from .dataset import PromptRewriteDataset
from .template import TRAIN_INPUT_TEMPLATE, TRAIN_OUTPUT_TEMPLATE

logger = logging.getLogger(__name__)


PREDICT_REVISED_INPUT = """A text-to-image generation system transforms text prompts into visual images. The original prompt is "{prompt}". Its generation quality is: prompt-image similarity of {init_clip}, aesthetic quality of {init_aesthetic}, and overall quality of {init_overall}. The prompt is revised to a new prompt. The revised prompt is: "{rewritten_prompt}". Predict the quality of the newly generated image for the original prompt:"""
PREDICT_REVISED_OUTPUT = """Prompt-image similarity is {new_clip}. Aesthetic quality is {new_aesthetic}. Overall quality is {new_overall}. """

PREDICT_LENGTH_INPUT = """A text-to-image generation system transforms text prompts into visual images. A prompt is "{prompt}". The prompt is revised to a new prompt. The revised prompt is: "{rewritten_prompt}". Compare their length and revision type:"""
PREDICT_LENGTH_OUTPUT = """The revised prompt, structured into {new_phrase_cnt} phrases, is {length_compare} the original prompt.""" + " "

TASKS = [
    (TRAIN_INPUT_TEMPLATE, TRAIN_OUTPUT_TEMPLATE),
    (PREDICT_LENGTH_INPUT, PREDICT_LENGTH_OUTPUT),
    (PREDICT_REVISED_INPUT, PREDICT_REVISED_OUTPUT),
]

class MultiTaskPromptRewriteDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 is_train=True, 
                 ):
        super(MultiTaskPromptRewriteDataset, self).__init__()

        assert is_train, "only support train data"
        self.datasets = [
            PromptRewriteDataset(
                data_path, 
                tokenizer=tokenizer,
                is_train=is_train,
                input_template=cur_input_template,
                output_template=cur_output_template,
            )
            for cur_input_template, cur_output_template in TASKS
        ]
        assert all(len(d) == len(self.datasets[0]) for d in self.datasets), "all datasets should have the same length"
        
    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return [d[i] for d in self.datasets]

    def show_example(self):
        for d in self.datasets:
            d.show_example()
