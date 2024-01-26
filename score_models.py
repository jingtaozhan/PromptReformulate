import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from typing import List, Dict, Union

import torch
import ImageReward as RM
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import clip


class HPSV2(nn.Module):
    def __init__(self, model, tokenizer, preprocess_val, device) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_val = preprocess_val
        self.device = device
    
    def forward(self, prompt: torch.Tensor, generations_list: torch.Tensor):  
        result = []
        generations_list = [img.unsqueeze(0).to(device=self.device, non_blocking=True) for img in generations_list]
        text = prompt.to(device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text, normalize=True)
            for image in generations_list:
                # Load your image and prompt
                # Process the image
                # image = self.preprocess_val(img).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                # text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                # outputs = self.model(image, text)
                # image_features, text_features = outputs["image_features"], outputs["text_features"]
                image_features = self.model.encode_image(image, normalize=True)
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0].item())
        # batch_image_features = self.model.encode_image(torch.cat(generations_list, 0), normalize=True)
        # for image_features in batch_image_features:
        #     image_features = image_features.unsqueeze(0)
        #     with torch.cuda.amp.autocast():
        #         logits_per_image = image_features @ text_features.T
        #         hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        #     result.append(hps_score[0].item())
        return result


class ImageReward(nn.Module):

    def __init__(self, med_config, device='cpu'):
        super().__init__()

        from ImageReward.ImageReward import BLIP_Pretrain, _transform, MLP

        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def forward(self, prompt: torch.Tensor, generations_list: List[torch.Tensor]):
        
        # text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_input = prompt.to(self.device)

        image_embeds = self.blip.visual_encoder(generations_list.to(self.device))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        txt_features = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask = text_input.attention_mask,
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_atts,
            return_dict = True,
        ).last_hidden_state[:,0,:].float()
        
        # txt_set = []
        # for generation in generations_list:
        #     # image encode
        #     pil_image = generation

        #     # image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        #     image = pil_image.unsqueeze(0).to(self.device)
        #     image_embeds = self.blip.visual_encoder(image)
            
        #     # text encode cross attention with image
        #     image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        #     text_output = self.blip.text_encoder(text_input.input_ids,
        #                                             attention_mask = text_input.attention_mask,
        #                                             encoder_hidden_states = image_embeds,
        #                                             encoder_attention_mask = image_atts,
        #                                             return_dict = True,
        #                                         )
        #     txt_set.append(text_output.last_hidden_state[:,0,:])
            
        # txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        rewards = self.mlp(txt_features) # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return rewards.detach().cpu().numpy().tolist()


class CLIP(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def forward(self, prompt: torch.Tensor, generations_list: torch.Tensor):
        
        # text = clip.tokenize(prompt, truncate=True).to(self.device)
        text = prompt.to(self.device)
        txt_feature = F.normalize(self.clip_model.encode_text(text))
        
        txt_set = [txt_feature for _ in range(len(generations_list))]
        img_features = F.normalize(self.clip_model.encode_image(generations_list.to(self.device))).float()
        # txt_set = []
        # img_set = []
        # for generations in generations_list:
        #     # image encode
        #     pil_image = generations
        #     # image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        #     image = pil_image.unsqueeze(0).to(self.device)
        #     image_features = F.normalize(self.clip_model.encode_image(image))
        #     img_set.append(image_features)
        #     txt_set.append(txt_feature)
        # img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]

        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return rewards.detach().cpu().numpy().tolist()
    

class Aesthetic(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        from ImageReward.models.AestheticScore import MLP
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        self.mlp = MLP(768)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def forward(self, prompt: torch.Tensor, generations_list: torch.Tensor):
        
        img_features = F.normalize(self.clip_model.encode_image(generations_list.to(self.device))).float()
        # img_set = []
        # for generations in generations_list:
        #     # image encode
        #     pil_image = generations
        #     # image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        #     image = pil_image.unsqueeze(0).to(self.device)
        #     image_features = F.normalize(self.clip_model.encode_image(image))
        #     img_set.append(image_features)
            
        # img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]
        rewards = self.mlp(img_features)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return rewards.detach().cpu().numpy().tolist()
    

@torch.no_grad()
def load_score_model(model_name, device) -> Union[HPSV2, ImageReward, CLIP, Aesthetic]:
    if model_name == "hpsv2":
        from hpsv2.score import model, tokenizer, preprocess_val
        model.to(device)
        model = HPSV2(model, tokenizer, preprocess_val, device)
    else:
        if model_name == "ImageReward":
            state_dict = torch.load("./data/ImageReward/ImageReward.pt", map_location='cpu')
            state_dict = {
                    k.replace('module.', ''): v for k, v in state_dict.items()
                }
            model = ImageReward(device=device, med_config="./data/ImageReward/med_config.json").to(device)
            msg = model.load_state_dict(state_dict,strict=True)
            model.eval()
        else:
            model_download_root = os.path.expanduser("~/.cache/ImageReward")
            from ImageReward.utils import _SCORES, _download
            if model_name in _SCORES:
                model_path = _download(_SCORES[model_name], model_download_root)

            if  model_name == "CLIP":
                model = CLIP(download_root=model_download_root, device=device).to(device)
            elif model_name == "Aesthetic":
                state_dict = torch.load(model_path, map_location='cpu')
                model = Aesthetic(download_root=model_download_root, device=device).to(device)
                model.mlp.load_state_dict(state_dict,strict=False)
            else:
                raise RuntimeError(f"Score {model_name} not found")
        
    model.eval()
    return model


def preprocess_images_for_scorers(
        preprocess_dict: Dict[str, callable],                           
        images: List[Image.Image]
    ) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess images for scorers
    We find for these four scorers, their preprocess functions are essentially the same.
    Therefore, we only preprocess one time and then copy the result for other scorers.
    """
    # preprocess images
    imgrwd_images = torch.vstack([preprocess_dict['ImageReward'](img).unsqueeze(0) for img in images])
    hpsv2_images = torch.vstack([preprocess_dict['hpsv2'](img).unsqueeze(0) for img in images])
    return {
        "hpsv2": hpsv2_images,
        "ImageReward": imgrwd_images,
        "CLIP": imgrwd_images,
        "Aesthetic": imgrwd_images,
    }


def preprocess_prompt_for_scorers(
        tokenizer_dict: Dict[str, callable],                           
        prompt: str
    ) -> Dict[str, str]:
    clip_prompt = tokenizer_dict["CLIP"](prompt, truncate=True)
    aesthetic_prompt = tokenizer_dict["Aesthetic"](prompt, truncate=True)
    imagereward_prompt = tokenizer_dict["ImageReward"](prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
    hpsv2_prompt = tokenizer_dict["hpsv2"]([prompt])
    return {
        "hpsv2": hpsv2_prompt,
        "ImageReward": imagereward_prompt,
        "CLIP": clip_prompt,
        "Aesthetic": aesthetic_prompt,
    }
