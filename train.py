import pathlib
import logging
import transformers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import is_main_process, set_seed
from transformers.data.data_collator import default_data_collator

from .flash_attn_patch import replace_llama_attn_with_flash_attn
from .modeling import AutoPromptSuggestor

from .dataset import PromptRewriteDataset
from .multi_task_dataset import MultiTaskPromptRewriteDataset
from .template import (
    TRAIN_INPUT_TEMPLATE, TRAIN_OUTPUT_TEMPLATE, 
    EVAL_INPUT_TEMPLATE, EVAL_OUTPUT_TEMPLATE
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    model_max_length: int = field(default=500)
    

@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to the training data."}
    )
    valid_data_path: str = field(
        metadata={"help": "Path to the training data."}
    )
    task_mode: str = field(metadata={"choices": ["single", "multi"]})


@dataclass
class MyTrainingArguments(TrainingArguments):
    pass
        
        
@dataclass
class TrainCollator:
    def __call__(self, features: List[str]) -> Dict[str, Any]:   
        if isinstance(features[0], list):
            features = sum(features, [])
        features = default_data_collator(features)
        max_len = features['attention_mask'].sum(-1).max().item()
        features['input_ids'] = features['input_ids'][:, :max_len]
        features['attention_mask'] = features['attention_mask'][:, :max_len]
        features['labels'] = features['labels'][:, :max_len]
        return features


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args: DataArguments
    model_args: ModelArguments
    training_args: MyTrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training parameters %s", training_args)
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    if "llama2" in model_args.model_name_or_path.lower():
        from .flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        logger.info("Replace llama attention with flash attention")
        
    model = AutoPromptSuggestor.from_pretrained(model_args.model_name_or_path)

    # Print how many parameters still requires grad
    logger.info(f"Number of parameters still requires grad: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    if data_args.task_mode == "multi":
        train_dataset = MultiTaskPromptRewriteDataset(
            data_path=data_args.train_data_path,
            tokenizer=tokenizer,
            is_train = True,
        ) 
    elif data_args.task_mode == "single":
        train_dataset = PromptRewriteDataset(
            data_path=data_args.train_data_path,
            tokenizer=tokenizer,
            is_train = True,
            input_template=TRAIN_INPUT_TEMPLATE,
            output_template=TRAIN_OUTPUT_TEMPLATE,
        ) 
    else:
        raise NotImplementedError(f"train_mode {data_args.train_mode} not implemented")
    valid_dataset = PromptRewriteDataset(
        data_path=data_args.valid_data_path,
        tokenizer=tokenizer,
        is_train = True,
        input_template=EVAL_INPUT_TEMPLATE,
        output_template=EVAL_OUTPUT_TEMPLATE,
    ) 
    
    train_dataset.show_example()

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=TrainCollator(),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    train()