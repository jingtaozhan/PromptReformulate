import os
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import sys
import json
import torch
import random
import argparse
import logging
import traceback
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from diffusers import StableDiffusionPipeline, DiffusionPipeline

from torch.utils.data import Dataset, DataLoader
from threading import Thread
from queue import Queue

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)


class ImageGenerationDataset(Dataset):
    def __init__(self, prompts, image_seeds, output_root, overwrite=False):
        self.prompts = prompts
        self.image_seeds = image_seeds
        self.output_root = output_root
        self.overwrite = overwrite


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item_idx):
        prompt_dict = self.prompts[item_idx]
        idx = prompt_dict["id"]
        if not self.overwrite:
            seeds = [s for s in self.image_seeds if not os.path.exists(os.path.join(self.output_root, str(idx), f"{idx}-{s}.webp"))]
        else:
            seeds = self.image_seeds
        prompt = prompt_dict["text"]
        return {
            "ids": [idx]*len(seeds),
            "prompts": [prompt]*len(seeds),
            "seeds": seeds,
        }


class PromptCollator:
    def __call__(self, features) -> Any:
        ids = sum([f["ids"] for f in features], [])
        prompts = sum([f["prompts"] for f in features], [])
        seeds = sum([f["seeds"] for f in features], [])
        return {
            "ids": ids,
            "prompts": prompts,
            "seeds": seeds,
        }


def run_generate(
    pipe: Union[StableDiffusionPipeline, DiffusionPipeline],
    input_file: str,
    output_root: str,
    seeds: List[int],
    num_inference_steps: int,
    num_gpus: int,
    local_rank: int,
    overwrite: bool, 
    batch_size: int = 1,
):
    # https://github.com/huggingface/diffusers/issues/3061
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # split question file into num_gpus files
    with open(input_file) as f:
        prompts = json.load(f)
    prompts = [
        {"id": i, "text": text} for i, text in enumerate(prompts)
    ]
    if num_gpus > 1:
        chunk_size = len(prompts) / num_gpus
        prompts = prompts[int(chunk_size * local_rank): int(chunk_size * (local_rank + 1))]
    generate_image(
        pipe=pipe, 
        prompts=prompts, 
        image_seeds=seeds, 
        num_inference_steps=num_inference_steps, 
        output_root=output_root, 
        overwrite=overwrite,
        batch_size=batch_size,
    )


def save_images(queue: Queue):
    try:
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                break
            image, output_path = item
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                image.save(output_path, lossless=True)
            except:
                if os.path.exists(output_path):
                    os.remove(output_path)
                logger.warning(f"Error saving {output_path}")
                raise
            queue.task_done()
        logger.warning("Saving thread finished")
    except:
        logger.error(f"Error in saving thread: {traceback.format_exc()}")
        raise


@torch.no_grad()
def generate_image(
        pipe: Union[StableDiffusionPipeline, DiffusionPipeline],
        prompts: List[str], 
        image_seeds: List[int], 
        num_inference_steps: int, 
        output_root: str, 
        overwrite: bool,
        batch_size: int = 1,
    ):

    # Usage
    dataset = ImageGenerationDataset(prompts, image_seeds, output_root, overwrite=overwrite)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=PromptCollator(), num_workers=0)

    # Initialize the queue and start the saving thread
    queue = Queue()
    try:
        saver_thread = Thread(target=save_images, args=(queue,))
        saver_thread.start()

        for batch in tqdm(dataloader):
            # Generate images
            if batch['prompts']: # if these images have already been generated, the batch is empty
                generators = [torch.Generator(device=pipe.device).manual_seed(s) for s in batch['seeds']]
                images = pipe(batch['prompts'], num_images_per_prompt=1, generator=generators, num_inference_steps=num_inference_steps).images
                assert len(images) == len(batch['ids'])
                # Queue images for saving
                for idx, image, s in zip(batch['ids'], images, batch['seeds']):
                    output_path = os.path.join(output_root, str(idx), f"{idx}-{s}.webp")
                    queue.put((image, output_path))

        # Signal the saving thread to finish and wait for it
        logger.info("Waiting for the saving thread to finish")
        queue.put(None)
        queue.join()
        saver_thread.join()
        logger.info("Done!")
    except:
        queue.put(None)
        logger.error(f"Error in generating images: {traceback.format_exc()}")
        raise
    

def load_generate_model(model_path: str, device: str, compile: bool = False):
    if model_path == "ReFL":
        model_path = "./data/baselines/refl/refl_model"
        print(f"Use ReFL model from {model_path}", file=sys.stderr)

    # device = torch.device('cuda', local_rank)
    if model_path == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
        print(f"Use stable diffusion XL")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to(device)
    if compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank <= 0 else logging.WARN,
    )

    num_gpus = dist.get_world_size()

    logger.warning(
        f"Process rank: {local_rank}, device: cuda:{local_rank}, n_gpu: {num_gpus}"
        + f"distributed training: {bool(local_rank != -1)}"
    )
    logger.info(f"Num-Inference-Steps {args.num_inference_steps}", )

    pipe = load_generate_model(args.model_path, device=f"cuda:{local_rank}")
    run_generate(
        pipe=pipe,
        input_file=args.input_file,
        output_root=args.output_root,
        seeds=args.seeds,
        num_inference_steps=args.num_inference_steps,
        num_gpus=num_gpus,
        local_rank=local_rank,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()