import os
import sys
import json
import logging
import transformers
from dataclasses import dataclass, field

from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import is_main_process, set_seed
from dataclasses_json import dataclass_json

import torch
torch.multiprocessing.set_sharing_strategy("file_system")

from .dataset import PromptRewriteDataset
from .modeling import AutoPromptSuggestor
from .eval_utils import PromptSuggestTrainer, EvalCollator
from .template import EVAL_INPUT_TEMPLATE, EVAL_OUTPUT_TEMPLATE, EVAL_PREFIXED_INPUT_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    delta_phrase_cnt: int = field(default=None)
    delta_overall: int = field(default=None)
    delta_aesthetic: int = field(default=None)
    delta_clip: int = field(default=None)
    length_compare: str = field(default="longer than")
    model_max_length: int = field(default=500)
    no_repeat_ngram_size: int = field(default=3)
    add_prefix: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class GenerationArguments(Seq2SeqTrainingArguments):
    output_name: str = field(default="prompt.json")


def run_rewrite(model_args: ModelArguments, data_path, eval_args, model, tokenizer, save_path, show_dataset_example=True):

    dataset = PromptRewriteDataset(
        data_path = data_path,
        tokenizer=tokenizer,
        is_train=False,
        input_template=EVAL_INPUT_TEMPLATE if not model_args.add_prefix else EVAL_PREFIXED_INPUT_TEMPLATE,
        output_template="",
        delta_clip=model_args.delta_clip,
        delta_aesthetic=model_args.delta_aesthetic,
        delta_overall=model_args.delta_overall,
        delta_phrase_cnt=model_args.delta_phrase_cnt,
        length_compare=model_args.length_compare,
    )
    if show_dataset_example:
        dataset.show_example()

    trainer = PromptSuggestTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=eval_args,
        data_collator = EvalCollator(),
    )

    gen_kwargs = {
        "do_sample": False,
        "num_return_sequences": 1,
        "max_new_tokens": 100,
        "no_repeat_ngram_size": model_args.no_repeat_ngram_size,
    }
    outputs = trainer.predict(dataset, **gen_kwargs)

    if is_main_process(eval_args.local_rank):
        # clean prompts to remove repetition words
        all_prompts = [s.strip() for s in outputs.predictions]
        rewritten_prompts = []
        for prompt in all_prompts:
            prompt = prompt.strip()
            splits = prompt.split("the revised prompt should be:")
            if not len(splits) == 2:
                logger.error(f"Prompt should be in the format of `... the revised prompt should be: ...`, but got {prompt}")
            prompt = splits[1].strip()
            rewritten_prompts.append(prompt)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(all_prompts, open(save_path + ".log.json", 'w'), indent=4)
        json.dump(model_args.to_dict(), open(save_path + ".setup.json", 'w'), indent=4)
        try:
            json.dump(rewritten_prompts, open(save_path, 'w'), indent=4)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"Error saving {save_path}", file=sys.stderr)
            raise


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GenerationArguments)
    )
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataArguments
    eval_args: GenerationArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(eval_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {eval_args.local_rank}, device: {eval_args.device}, n_gpu: {eval_args.n_gpu}"
        + f"distributed training: {bool(eval_args.local_rank != -1)}, 16-bits training: {eval_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(eval_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training parameters %s", eval_args)
    
    save_path = os.path.join(eval_args.output_dir, eval_args.output_name)
    if os.path.exists(save_path):
        if eval_args.overwrite_output_dir:
            logger.warning(f"{save_path} already exists. Overwriting.")
        else:
            logger.warning(f"{save_path} already exists. Skipping.")
            return
        
    # Set seed before initializing model.
    set_seed(eval_args.seed)
   
    model = AutoPromptSuggestor.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.float32)
    model.config.use_cache = True
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    run_rewrite(model_args, data_args.data_path, eval_args, model, tokenizer, save_path)


if __name__ == "__main__":
    main()