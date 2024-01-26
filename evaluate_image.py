import os
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import sys
import json
import clip
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Union

import torch
import ImageReward as RM
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

from .score_models import (
    load_score_model,
    preprocess_images_for_scorers,
    preprocess_prompt_for_scorers
)


def create_temp_files(dir_path, prefix, n):
    os.makedirs(dir_path, exist_ok=True)
    temp_files = []
    for i in range(n):
        temp_file = os.path.join(dir_path, prefix + f".temp_{i}.json")
        temp_files.append(temp_file)
    return temp_files


def run_score(
        model_dict: Dict[str, callable],
        models: List[str], 
        prompt_path: str, 
        image_root: str, 
        seeds: List[int], 
        output_dir: str, 
        local_rank: int,
        overwrite: bool,
        num_gpus: int,
    ):
    # https://github.com/huggingface/diffusers/issues/3061
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    with open(prompt_path) as f:
        all_prompts = json.load(f)
    all_prompts = [{"id": i, "text": text} for i, text in enumerate(all_prompts)]

    if num_gpus > 1:
        chunk_size = len(all_prompts) / num_gpus
        process_prompts = all_prompts[int(chunk_size * local_rank): int(chunk_size * (local_rank + 1))]
    else:
        process_prompts = all_prompts
    os.makedirs(output_dir, exist_ok=True)

    get_image_scores(
        model_dict=model_dict,
        models=models,
        process_prompts=process_prompts,
        image_root=image_root,
        seeds=seeds,
        overwrite=overwrite,
    )
    if num_gpus > 1 and local_rank >= 0:
        dist.barrier()

    if local_rank in [0, -1] or num_gpus == 1:
        output_dict = {model: [] for model in models}
        output_prefix = "_".join(sorted(models)).lower()
        for prompt_data in tqdm(all_prompts):
            idx, prompt = prompt_data["id"], prompt_data["text"]
            all_metric_dict = {model: [] for model in models}
            for seed in seeds:
                load_data = json.load(open(os.path.join(image_root, f"{idx}/{idx}-{seed}.{output_prefix}.json")))
                for model in models:
                    all_metric_dict[model].append(load_data[model])
            for model in models:
                cur_output_data = deepcopy(prompt_data)
                cur_output_data["scores"] = all_metric_dict[model]
                cur_output_data["avg_score"] = np.mean(all_metric_dict[model]).item()
                output_dict[model].append(cur_output_data)

        for model in models:
            output_path = os.path.join(output_dir, f"{model}.json")
            json.dump({
                    "avg_score": np.mean([data["avg_score"] for data in output_dict[model]]).item(),
                    "data": output_dict[model],
                }, open(output_path, 'w'), indent=2
            )
        

class PromptEvaluationDataset(Dataset):
    def __init__(
            self, 
            prompts: List[Dict[str, str]], 
            image_seeds: List[int], 
            image_root: str, 
            preprocess_dict: Dict[str, callable],
            tokenizer_dict: Dict[str, callable],
            output_prefix: str,
            overwrite=False,
    ):
        self.prompts = prompts
        self.image_seeds = image_seeds
        self.image_root = image_root
        self.preprocess_dict = preprocess_dict
        self.tokenizer_dict = tokenizer_dict
        self.output_prefix = output_prefix
        self.overwrite = overwrite


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item_idx):
        idx, prompt = self.prompts[item_idx]["id"], self.prompts[item_idx]["text"]
        image_output_pairs = [
            (os.path.join(self.image_root, f"{idx}/{idx}-{seed}.webp"), 
             os.path.join(self.image_root, f"{idx}/{idx}-{seed}.{self.output_prefix}.json"),
             seed)
            for seed in self.image_seeds
        ]
        if not self.overwrite:
            # filter pairs from image_output_pairs that the output_path already exists
            image_output_pairs = [x for x in image_output_pairs if not os.path.exists(x[1])]

        if len(image_output_pairs) == 0:
            return None
        
        images = [Image.open(x[0]) for x in image_output_pairs]
        images_dict = preprocess_images_for_scorers(preprocess_dict=self.preprocess_dict, images=images)
        input_prompt_dict = preprocess_prompt_for_scorers(tokenizer_dict=self.tokenizer_dict, prompt=prompt)
        return {
            "prompt": self.prompts[item_idx],
            "prompt_tensor": input_prompt_dict,
            "images_tensor": images_dict,
            "image_output_pairs": image_output_pairs,
        }


class PromptCollator:
    def __call__(self, features):
        assert len(features) == 1, "Only support batch size 1"
        features: Dict[str, Union[torch.Tensor, List[str]]] = features[0]
        return features
    

@torch.no_grad()
def get_image_scores(
        model_dict: Dict[str, callable],
        models: List[str], 
        process_prompts: List[Dict[str, str]], 
        image_root: str, 
        seeds: List[int], 
        overwrite=False,
    ):

    preprocess_dict = {
        "hpsv2": model_dict["hpsv2"].preprocess_val,
        "ImageReward": model_dict["ImageReward"].preprocess,
        "CLIP": model_dict["CLIP"].preprocess,
        "Aesthetic": model_dict["Aesthetic"].preprocess,
    }
    tokenizer_dict = {
        "hpsv2": model_dict["hpsv2"].tokenizer,
        "ImageReward": model_dict["ImageReward"].blip.tokenizer,
        "CLIP": clip.tokenize,
        "Aesthetic": clip.tokenize, # fake
    }
    dataset = PromptEvaluationDataset(
        prompts=process_prompts,
        image_seeds=seeds,
        image_root=image_root,
        preprocess_dict=preprocess_dict,
        tokenizer_dict=tokenizer_dict,
        output_prefix="_".join(sorted(models)).lower(),
        overwrite=overwrite,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        collate_fn=PromptCollator(),
    )
    for data in tqdm(dataloader):
        if data is None:
            continue
        image_output_pairs = data['image_output_pairs']
        images_dict = data['images_tensor']
        input_prompt_dict = data['prompt_tensor']
        scores_dict = {
            model_name: model_dict[model_name](input_prompt_dict[model_name], images_dict[model_name])
            for model_name in models
        }
        for i, (_, output_path, seed) in enumerate(image_output_pairs):
            output_data_for_one_seed = deepcopy(data['prompt'])
            output_data_for_one_seed["seed"] = seed 
            output_data_for_one_seed.update({
                model_name: scores_dict[model_name][i]
                for model_name in models
            })
            try:
                json.dump(output_data_for_one_seed, open(output_path, "w"), indent=2)
            except:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"Error saving {output_path}", file=sys.stderr)
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()

    print(local_rank, file=sys.stderr)

    device=torch.device(f"cuda:{local_rank}")
    model_dict = {
        model_name: load_score_model(model_name, device=device)
        for model_name in args.models
    }
    run_score(
        model_dict=model_dict,
        models=args.models,
        prompt_path=args.prompt_path,
        image_root=args.image_root,
        seeds=args.seeds,
        output_dir=args.output_dir,
        local_rank=local_rank,
        overwrite=args.overwrite,
        num_gpus=dist.get_world_size(),
    )


if __name__ == "__main__":
    main()