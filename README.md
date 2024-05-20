# PromptReformulate

This code base is to provide algorithms to reformulate image prompts for better generation.

We currently include code for the following two papers: 
- SIGIR 2024: Capability-aware Prompt Reformulation Learning for Text-to-Image Generation 
- ACL 2024: Prompt Refinement with Image Pivot for Text-to-Image Generation (We are currently working on the code base. Stay tuned!)

Capability-aware Prompt Reformulation (CAPR) innovatively integrates user capability into the reformulation process through two key components: the Conditional Reformulation Model (CRM) and Configurable Capability Features (CCF). CRM reformulates prompts according to a specified user capability, as represented by CCF. The CCF, in turn, offers the flexibility to tune and guide the CRM's behavior. This enables CAPR to effectively learn diverse reformulation strategies across various user capacities and to simulate high-capability user reformulation during inference. 


## Requirements

This repo is developed with PyTorch, clip, and skopt. They should be installed manually due to the requirement of platform-specific custom configuration. 

## Training


```bash
torchrun --nproc_per_node=4 \
    train.py \
    --task_mode single \
    --train_data_path ${train_data_path} \
    --valid_data_path ${valid_data_path} \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T" \
    --bf16 True \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --output_dir $output_dir \
    --num_train_epochs 2 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate $lr \
    --optim adamw_hf \
    --seed 2023 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --remove_unused_columns False
```

## Tuning CCF

```bash
torchrun --nproc_per_node=1 \
    bayes_tune.py \
    --rewrite_input_data_path ${rewrite_input_data_path} \
    --score_input_data_path ${score_input_data_path} \
    --rewrite_model_name_or_path $rewrite_model_name_or_path \
    --bf16 True \
    --no_repeat_ngram_size 3 \
    --generate_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --generate_seeds 0 1 \
    --generate_inference_steps 20 \
    --imagereward 1 \
    --log_output_dir $log_output_dir \
    --n_calls 50 \
    --min_clip 0 \
    --max_clip 9 \
    --min_aesthetic 0 \
    --max_aesthetic 9 \
    --min_overall 9 \
    --max_overall 9 \
    --min_phrase_cnt 1 \
    --max_phrase_cnt 10 \
    --dataloader_num_workers 0 \
    --output_dir $output_dir \
    --seed 2022 \
    --per_device_eval_batch_size 8 \
    --remove_unused_columns False \
    --overwrite_output_dir True
```

## Reformulation

```bash
torchrun --nproc_per_node=1 \
    evaluate.py \
    --predict_with_generate True \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --delta_clip $delta_clip \
    --delta_aesthetic $delta_aesthetic \
    --delta_overall $delta_overall \
    --delta_phrase_cnt $delta_phrase_cnt \
    --output_name "prompt.json" \
    --dataloader_num_workers 1 \
    --output_dir $output_dir \
    --seed 2022 \
    --per_device_eval_batch_size 8 \
    --logging_steps 20 \
    --remove_unused_columns False \
```

## Evaluation

```bash
torchrun --nproc_per_node=1 \
    generate_image.py \
    --model-path "CompVis/stable-diffusion-v1-4" \
    --input-file $reformulate_prompt_path \
    --output-root ${output_root}/images \
    --num-inference-steps 50 \
    --num-gpus 1

torchrun --nproc_per_node=1 \
    evaluate_image.py \
    --models "hpsv2" "ImageReward" \
    --prompt_path $user_input_prompt_path  \
    --image_root ${output_root}/images \
    --output_dir ${output_root}/scores

```

