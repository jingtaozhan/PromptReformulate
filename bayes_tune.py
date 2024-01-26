import os
import json
import torch
import pickle
import logging
import transformers
from tqdm import tqdm
import numpy as np
from typing import List
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from transformers.trainer_utils import is_main_process, set_seed
from transformers import Seq2SeqTrainingArguments

np.int = np.int_
torch.multiprocessing.set_sharing_strategy("file_system")
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

from .modeling import AutoPromptSuggestor
from .evaluate import run_rewrite
from .evaluate import ModelArguments as ModelArgumentsForRewritting
from .generate_image import run_generate, load_generate_model
from .evaluate_image import run_score, load_score_model

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class DataArguments:
    rewrite_input_data_path: str = field()
    score_input_data_path: str = field()

    def __post_init__(self):
        assert os.path.exists(self.rewrite_input_data_path), f"{self.rewrite_input_data_path} does not exist"
        assert os.path.exists(self.score_input_data_path), f"{self.score_input_data_path} does not exist"
        rewrite_input_data = json.load(open(self.rewrite_input_data_path))
        score_input_data = json.load(open(self.score_input_data_path))
        assert len(rewrite_input_data) == len(score_input_data), f"rewrite_input_data and score_input_data have different length"
        assert all(
            score_input == rewrite_input['prompt'] 
            for score_input, rewrite_input in zip(score_input_data, rewrite_input_data)
        )


@dataclass_json
@dataclass
class RewriteModelArguments:
    rewrite_model_name_or_path: str = field()
    rewrite_model_max_length: int = field(default=500)
    length_compare: str = field(default="longer than")
    no_repeat_ngram_size: int = field(default=3)
    add_prefix: bool = field(default=True)
    rewrite_output_name: str = field(default="prompt.json")


@dataclass_json
@dataclass
class GenerateModelArguments:
    generate_model_name_or_path: str = field()
    generate_seeds: List[int] = field()
    generate_inference_steps: int = field(default=50)


@dataclass_json
@dataclass
class WeightingArguments:
    clip: float = field()
    aesthetic: float = field()
    hpsv2: float = field()
    imagereward: float = field()
    threshold_clip: float = field() # weight * min(clip - 0.28, 0)


@dataclass_json
@dataclass
class TuneHyperParameter:
    log_output_dir: str = field()
    n_calls: int = field(default=50)
    random_state: int = field(default=0)
    add_history: bool = field(default=False)
    min_clip: int = field(default=0)
    max_clip: int = field(default=9)
    min_aesthetic: int = field(default=0)
    max_aesthetic: int = field(default=9)
    min_overall: int = field(default=0)
    max_overall: int = field(default=9)
    min_phrase_cnt: int = field(default=1)
    max_phrase_cnt: int = field(default=12)
    

def numpy_to_json_saveable(obj):
    """
    Convert a numpy object to a JSON-saveable object.
    Handles np.float64, np.int64, list of np.float64, list of list of np.int64, np.ndarray, etc.
    Converts numpy arrays to lists and dtype to int/float.
    
    Args:
    obj (numpy object): The numpy object to convert.
    
    Returns:
    Object that is JSON saveable.
    """
    
    # Check if the object is a numpy array
    if isinstance(obj, np.ndarray):
        # Convert ndarray to a list of lists and recursively process elements
        return [numpy_to_json_saveable(el) for el in obj]
    
    # Check if the object is a numpy float
    elif isinstance(obj, np.floating):
        # Convert numpy float to Python float
        return float(obj)
    
    # Check if the object is a numpy int
    elif isinstance(obj, np.integer):
        # Convert numpy int to Python int
        return int(obj)
    
    # Check if the object is a list
    elif isinstance(obj, list):
        # Recursively process each element in the list
        return [numpy_to_json_saveable(el) for el in obj]
    
    # If it's none of the above types, return the object as is
    return obj


def get_history_scores(root: str, generate_args: GenerateModelArguments, weight_args: WeightingArguments, x_param_names: List[str]):
    '''
        Search for directory in root that contains prompt.json and prompt.json.setup.json
        Then, check whether the sub-folder 'generate_args.generate_model_name_or_path/step_{generate_args.generate_inference_steps}' contains CLIP.json, Aesthetic.json, hpsv2.json and ImageReward.json
        The prompt.json.setup.json contains the x0 (delta_...)
        And the CLIP/....json files contain the  CLIP, Aesthetic, ... scores. y0 is their weighted sum.
    '''
    x0, y0 = [], []
    for cur_path, dirs, files in os.walk(root):
        if "prompt.json" in files and "prompt.json.setup.json" in files:
            setup = json.load(open(os.path.join(cur_path, "prompt.json.setup.json")))
            reward_directory = os.path.join(cur_path, generate_args.generate_model_name_or_path, f"step_{generate_args.generate_inference_steps}", "reward_generations")
            if os.path.exists(reward_directory) and all(
                f"{model_name}.json" in os.listdir(reward_directory)
                for model_name in ["CLIP", "Aesthetic", "hpsv2", "ImageReward"]
            ):
                x0.append({param_name: setup[param_name] for param_name in x_param_names})
                threshold_clip_score = np.mean([
                    min(item['avg_score'] - 0.28, 0) for item in json.load(open(os.path.join(reward_directory, "CLIP.json")))['data']
                ]).item() * weight_args.threshold_clip
                y0.append(
                    (threshold_clip_score + sum(
                        json.load(open(os.path.join(reward_directory, f"{model_name}.json")))['avg_score'] * getattr(weight_args, model_name.lower())
                        for model_name in ["CLIP", "Aesthetic", "hpsv2", "ImageReward"]
                    )) * (-1)
                )
    return x0, y0


class Objective:
    def __init__(self, 
                 weight_args: WeightingArguments,
                 data_args: DataArguments,
                 rewrite_model_args: RewriteModelArguments, 
                 generate_args: GenerateModelArguments,
                 eval_args: Seq2SeqTrainingArguments
                 ):
        self.weight_args = weight_args
        self.data_args = data_args
        self.rewrite_model_args = rewrite_model_args
        self.generate_args = generate_args
        self.eval_args = eval_args

        self.rewrite_model = AutoPromptSuggestor.from_pretrained(
            rewrite_model_args.rewrite_model_name_or_path, 
            torch_dtype=torch.float16,
        )
        self.rewrite_model.config.use_cache = True
        self.rewrite_model.eval()

        self.rewrite_tokenizer = transformers.AutoTokenizer.from_pretrained(
            rewrite_model_args.rewrite_model_name_or_path,
            model_max_length=rewrite_model_args.rewrite_model_max_length,
            padding_side="left",
            use_fast=True,
        )
        self.rewrite_tokenizer.pad_token = self.rewrite_tokenizer.unk_token

        self.generate_model = load_generate_model(
            generate_args.generate_model_name_or_path, 
            device=eval_args.device, 
            compile=False
        )
        self.score_models = {
            model_name: load_score_model(model_name, device=eval_args.device)
            for model_name in ["CLIP", "Aesthetic", "hpsv2", "ImageReward"]
        }

    def rewrite(self, delta_phrase_cnt: int, delta_overall: int, delta_aesthetic: int, delta_clip: int, rewrite_save_path: str):    
        model_args_for_rewritting = ModelArgumentsForRewritting(
            model_name_or_path=self.rewrite_model_args.rewrite_model_name_or_path,
            delta_clip=delta_clip,
            delta_aesthetic=delta_aesthetic,
            delta_overall=delta_overall,
            delta_phrase_cnt=delta_phrase_cnt,
            length_compare=self.rewrite_model_args.length_compare,
            add_prefix=self.rewrite_model_args.add_prefix,
        )
        logger.info("model_args_for_rewritting parameters %s", model_args_for_rewritting)
        if os.path.exists(rewrite_save_path) and not self.eval_args.overwrite_output_dir:
            logger.info(f"{rewrite_save_path} already exists. Skipping.")
        else:
            if os.path.exists(rewrite_save_path) and self.eval_args.overwrite_output_dir:
                logger.info(f"{rewrite_save_path} already exists. Overwriting.")
            self.eval_args.predict_with_generate = True
            torch.cuda.empty_cache()
            run_rewrite(
                model_args = model_args_for_rewritting,
                data_path=self.data_args.rewrite_input_data_path,
                eval_args=self.eval_args,
                model=self.rewrite_model,
                tokenizer=self.rewrite_tokenizer,
                save_path=rewrite_save_path,
                show_dataset_example=False
            )
        if self.eval_args.local_rank != -1:
            torch.distributed.barrier()

    def generate(self, rewrite_save_path, generate_output_dir):
        torch.cuda.empty_cache()
        run_generate(
            pipe=self.generate_model,
            input_file=rewrite_save_path,
            output_root=generate_output_dir,
            seeds=self.generate_args.generate_seeds,
            num_inference_steps=self.generate_args.generate_inference_steps,
            num_gpus=self.eval_args.world_size,
            local_rank=self.eval_args.local_rank,
            overwrite=self.eval_args.overwrite_output_dir,
        )
        if self.eval_args.local_rank != -1:
            torch.distributed.barrier()

    def score(self, original_prompt_path, generate_output_dir, score_output_dir):
        torch.cuda.empty_cache()
        run_score(
            model_dict=self.score_models,
            models=list(self.score_models.keys()),
            prompt_path=original_prompt_path,
            image_root=generate_output_dir,
            seeds=self.generate_args.generate_seeds,
            output_dir=score_output_dir,
            local_rank=self.eval_args.local_rank,
            overwrite=self.eval_args.overwrite_output_dir,
            num_gpus=self.eval_args.world_size,
        )
        if self.eval_args.local_rank != -1:
            torch.distributed.barrier()

    def __call__(self, delta_phrase_cnt, delta_overall, delta_aesthetic, delta_clip):
        delta_clip = int(delta_clip)
        delta_aesthetic = int(delta_aesthetic)
        delta_overall = int(delta_overall)
        delta_phrase_cnt = int(delta_phrase_cnt)
            
        save_dir = os.path.join(
            self.eval_args.output_dir, 
            f"phrase{delta_phrase_cnt}", 
            f"overall{delta_overall}", 
            f"aesthetic{delta_aesthetic}", 
            f"clip{delta_clip}", 
        )
        rewrite_save_path = os.path.join(save_dir, self.rewrite_model_args.rewrite_output_name)
        self.rewrite(
            delta_phrase_cnt=delta_phrase_cnt,
            delta_overall=delta_overall,
            delta_aesthetic=delta_aesthetic,
            delta_clip=delta_clip,
            rewrite_save_path=rewrite_save_path
        )
        eval_output_root = os.path.join(
            save_dir,
            self.generate_args.generate_model_name_or_path,
            f"step_{self.generate_args.generate_inference_steps}"
        )
        generate_output_dir = os.path.join(eval_output_root, "images")
        self.generate(rewrite_save_path, generate_output_dir)

        score_output_dir = os.path.join(eval_output_root, "reward_generations")
        self.score(
            original_prompt_path=self.data_args.score_input_data_path,
            generate_output_dir=generate_output_dir,
            score_output_dir=score_output_dir
        )
        
        clip_score = json.load(open(os.path.join(score_output_dir, "CLIP.json")))['avg_score']
        threshold_clip_score = np.mean([
            min(item['avg_score'] - 0.28, 0) for item in json.load(open(os.path.join(score_output_dir, "CLIP.json")))['data']
        ]).item()
        aesthetic_score = json.load(open(os.path.join(score_output_dir, "Aesthetic.json")))['avg_score']
        hpsv2_score = json.load(open(os.path.join(score_output_dir, "hpsv2.json")))['avg_score']
        imagereward_score = json.load(open(os.path.join(score_output_dir, "ImageReward.json")))['avg_score']

        output_score = (
            clip_score * self.weight_args.clip + 
            aesthetic_score * self.weight_args.aesthetic + 
            hpsv2_score * self.weight_args.hpsv2 + 
            imagereward_score * self.weight_args.imagereward +
            threshold_clip_score * self.weight_args.threshold_clip
        ) * (-1)
        logger.info(f"delta_phrase_cnt: {delta_phrase_cnt}, delta_overall: {delta_overall}, delta_aesthetic: {delta_aesthetic}, delta_clip: {delta_clip}, score: {output_score}")
        return output_score



def main():
    parser = transformers.HfArgumentParser(
        (WeightingArguments, RewriteModelArguments, GenerateModelArguments, TuneHyperParameter, DataArguments, Seq2SeqTrainingArguments)
    )
    weight_args, rewrite_model_args, generate_args, tune_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    
    weight_args: WeightingArguments
    rewrite_model_args: RewriteModelArguments
    generate_args: GenerateModelArguments
    tune_args: TuneHyperParameter
    data_args: DataArguments
    eval_args: Seq2SeqTrainingArguments

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
    logger.info("Weighting parameters %s", weight_args)
    logger.info("Rewrite model parameters %s", rewrite_model_args)
    logger.info("Generate parameters %s", generate_args)
    logger.info("Tune parameters %s", tune_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training parameters %s", eval_args)
        
    # Set seed before initializing model.
    set_seed(eval_args.seed)
   
    objective = Objective(
        weight_args=weight_args,
        data_args=data_args,
        rewrite_model_args=rewrite_model_args,
        generate_args=generate_args,
        eval_args=eval_args
    )
    # Perform Bayesian Optimization

    x_param_names=["delta_phrase_cnt", "delta_overall", "delta_aesthetic", "delta_clip"]
    if tune_args.add_history:
        x0, y0 = get_history_scores(
            root = eval_args.output_dir,
            generate_args=generate_args,
            weight_args=weight_args,
            x_param_names=x_param_names,
        )
        logger.info(f"Number of history points: {len(x0)}")
        if len(x0) == 0:
            x0, y0 = None, None
    else:
        x0, y0 = None, None

    logger.info(f"X0: {x0}")
    logger.info(f"Y0: {y0}")

    space = [
    ]
    fixed_params = {}
    for min_value, max_value, name in [
        (tune_args.min_phrase_cnt, tune_args.max_phrase_cnt, "delta_phrase_cnt"),
        (tune_args.min_overall, tune_args.max_overall, "delta_overall"),
        (tune_args.min_aesthetic, tune_args.max_aesthetic, "delta_aesthetic"),
        (tune_args.min_clip, tune_args.max_clip, "delta_clip"),
    ]:
        if min_value == max_value:
            fixed_params[name] = min_value
            x_param_names.remove(name)
        else:
            space.append(Integer(min_value, max_value, name=name))
        if x0 is not None:
            # should first process y0 and then x0 because otherwise len(x0) < len(y0)
            y0 = [yi for xi, yi in zip(x0, y0) if xi[name] <= max_value and xi[name] >= min_value]
            x0 = [xi for xi in x0 if xi[name] <= max_value and xi[name] >= min_value]
            
    # convert the dict in x0 to list
    if x0 is not None:
        if len(x0) > 0:
            x0 = [[xi[param_name] for param_name in x_param_names] for xi in x0]
            logger.info(f"Number of history points after filtering: {len(x0)}")
            assert len(x0) == len(y0), f"len(x0) != len(y0): {len(x0)} != {len(y0)}"
        else:
            x0, y0 = None, None
            logger.info(f"Number of history points after filtering: 0")
    logger.info(f"X0: {x0}")
    logger.info(f"Y0: {y0}")

    @use_named_args(space)
    def objective_wrapper(**params):
        return objective(**fixed_params, **params)

    result = gp_minimize(
        func=objective_wrapper, 
        dimensions=space,
        acq_func="gp_hedge", # "LCB", 
        n_calls=tune_args.n_calls, 
        n_initial_points=tune_args.n_calls // 5,
        random_state=tune_args.random_state,
        kappa=1.96,
        verbose=True,
        x0=x0,
        y0=y0,
    )

    jsonable_result = {
        k: numpy_to_json_saveable(result[k])
        for k in ["x", "fun", "func_vals", "x_iters"]
    }
    if is_main_process(eval_args.local_rank):
        # Print the result
        print(result)
        print("Best parameters: {}".format(result.x))
        print("Best score: {}".format(-result.fun))
        os.makedirs(tune_args.log_output_dir, exist_ok=True)
        json.dump(jsonable_result, open(os.path.join(tune_args.log_output_dir, "brief_result.json"), "w"), indent=2)
        json.dump(data_args.to_dict(), open(os.path.join(tune_args.log_output_dir, "data_args.json"), "w"), indent=2)
        json.dump(rewrite_model_args.to_dict(), open(os.path.join(tune_args.log_output_dir, "rewrite_model_args.json"), "w"), indent=2)
        json.dump(generate_args.to_dict(), open(os.path.join(tune_args.log_output_dir, "generate_args.json"), "w"), indent=2)
        json.dump(weight_args.to_dict(), open(os.path.join(tune_args.log_output_dir, "weight_args.json"), "w"), indent=2)
        json.dump(tune_args.to_dict(), open(os.path.join(tune_args.log_output_dir, "tune_args.json"), "w"), indent=2)
        json.dump(eval_args.to_dict(), open(os.path.join(tune_args.log_output_dir, "eval_args.json"), "w"), indent=2)
    

if __name__ == "__main__":
    main()
