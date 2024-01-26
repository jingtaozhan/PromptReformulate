import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from transformers import Seq2SeqTrainer
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import is_main_process, set_seed
from transformers.trainer import (
    EvalLoopOutput, 
    has_length, 
    find_batch_size, 
    is_torch_tpu_available,
    deepspeed_init,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalCollator:
    def __call__(self, features: List[str]) -> Dict[str, Any]:   
        features = default_data_collator(features)
        max_len = features['attention_mask'].sum(-1).max().item()
        features['input_ids'] = features['input_ids'][:, -max_len:]
        features['attention_mask'] = features['attention_mask'][:, -max_len:]
        return features


class PromptSuggestTrainer(Seq2SeqTrainer):

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
    
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        all_prompts = []
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
            if is_torch_tpu_available():
                from transformers.trainer import xm
                xm.mark_step()

            assert logits is not None
            logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=self.tokenizer.pad_token_id)
            if self.preprocess_logits_for_metrics is not None:
                logits = self.preprocess_logits_for_metrics(logits, labels)
            logits = self.accelerator.gather_for_metrics((logits))
            if is_main_process(self.args.local_rank):
                prompts = self.tokenizer.batch_decode(logits.detach().cpu().numpy().tolist(), skip_special_tokens=True)
                all_prompts.extend(prompts)

                if (step + 1) % self.args.logging_steps == 0:
                    logger.info(f"Step {step} / {len(dataloader)} Example output: {prompts}")

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        return EvalLoopOutput(predictions=all_prompts, label_ids=None, metrics={}, num_samples=len(all_prompts))