# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ...data.mixed_batch_sampler import MixedBatchSampler
import math


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, DataArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        data_args: Optional["DataArguments"] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self.data_args = data_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

        # Per-dataset gradient bookkeeping
        self._enable_per_dataset_grad = bool(
            getattr(self.finetuning_args, "record_per_dataset_grad_norm", False)
            or (getattr(self.finetuning_args, "per_dataset_grad_scale", None) is not None)
        )
        self._per_dataset_scales = {}
        if getattr(self.finetuning_args, "per_dataset_grad_scale", None) is not None:
            # finetuning_args should provide a dict after __post_init__ parsing if user configured it
            self._per_dataset_scales = self.finetuning_args.per_dataset_grad_scale  # type: ignore[attr-defined]

        self._accum_grads: Optional[list[torch.Tensor]] = None
        self._per_dataset_grad_norm_sum: dict[str, float] = {}
        self._per_dataset_grad_norm_cnt: dict[str, int] = {}

        # Per-dataset loss bookkeeping
        self._enable_per_dataset_loss = bool(getattr(self.finetuning_args, "record_per_dataset_loss", False))
        self._per_dataset_loss_sum: dict[str, float] = {}
        self._per_dataset_loss_cnt: dict[str, int] = {}

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_train_dataloader(self):
        # Use custom batch sampler when mixing probs are set or per-dataset grad feature is enabled
        if (
            self.train_dataset is None
            or self.data_args is None
            or (self.data_args.sft_batch_mix_probs is None and not self._enable_per_dataset_grad)
        ):
            return super().get_train_dataloader()

        # Must be map-style to access columns
        try:
            dataset_names = list(self.train_dataset["dataset"])  # type: ignore[index]
        except Exception:
            return super().get_train_dataloader()

        per_device_bsz = self.args.per_device_train_batch_size
        world_size = self.args.world_size
        rank = self.args.process_index
        shuffle = not self.finetuning_args.disable_shuffling
        mix_probs = (
            [float(x) for x in self.data_args.sft_batch_mix_probs]
            if self.data_args.sft_batch_mix_probs is not None
            else None
        )
        if mix_probs is None:
            # default uniform over provided datasets order
            uniq = list(dict.fromkeys(dataset_names))
            mix_probs = [1.0 / len(uniq)] * len(uniq)
        dataset_order = self.data_args.dataset or list(dict.fromkeys(dataset_names))
        mode = "pure" if self._enable_per_dataset_grad else "mixed"

        batch_sampler = MixedBatchSampler(
            dataset=self.train_dataset,
            dataset_field="dataset",
            per_device_batch_size=per_device_bsz,
            mix_probs=mix_probs,
            dataset_order=dataset_order,  # type: ignore[arg-type]
            mode=mode,
            shuffle=shuffle,
            drop_last=self.args.dataloader_drop_last,
            world_size=world_size,
            rank=rank,
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def training_step(self, model: "torch.nn.Module", inputs: dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # Extract dataset tags and avoid forwarding to model
        batch_dataset = inputs.pop("batch_dataset", None)
        dataset_name = None
        if isinstance(batch_dataset, list) and len(batch_dataset) > 0:
            dataset_name = str(batch_dataset[0])

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # Record per-dataset loss if enabled
        if self._enable_per_dataset_loss and dataset_name is not None:
            loss_value = float(loss.detach())
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    t = torch.tensor(loss_value, device=loss.device, dtype=torch.float32)
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                    loss_value = (t / self.args.world_size).item()
            except Exception:
                pass
            self._per_dataset_loss_sum[dataset_name] = self._per_dataset_loss_sum.get(dataset_name, 0.0) + loss_value
            self._per_dataset_loss_cnt[dataset_name] = self._per_dataset_loss_cnt.get(dataset_name, 0) + 1

        # Backward for this micro-batch
        self.accelerator.backward(loss)

        # If not enabled, use default accumulation behavior
        if not self._enable_per_dataset_grad:
            return loss.detach()

        # Compute grad norm (pre-scaling) and reduce across ranks
        if bool(getattr(self.finetuning_args, "record_per_dataset_grad_norm", False)) and dataset_name is not None:
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.float().norm(2)
                    total_norm_sq += float(param_norm.item() ** 2)
            total_norm = math.sqrt(max(total_norm_sq, 0.0))
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    t = torch.tensor(total_norm, device=loss.device, dtype=torch.float32)
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                    total_norm = (t / self.args.world_size).item()
            except Exception:
                pass
            self._per_dataset_grad_norm_sum[dataset_name] = (
                self._per_dataset_grad_norm_sum.get(dataset_name, 0.0) + total_norm
            )
            self._per_dataset_grad_norm_cnt[dataset_name] = self._per_dataset_grad_norm_cnt.get(dataset_name, 0) + 1

        # Scale grads per dataset if configured
        scale = 1.0
        if dataset_name is not None and dataset_name in self._per_dataset_scales:
            try:
                scale = float(self._per_dataset_scales[dataset_name])
            except Exception:
                scale = 1.0
        if abs(scale - 1.0) > 1e-8:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale)

        # Accumulate grads into buffer and clear param grads
        if self._accum_grads is None:
            self._accum_grads = [
                (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p, device=p.device))
                for p in model.parameters()
            ]
        else:
            for buf, p in zip(self._accum_grads, model.parameters()):
                if p.grad is not None:
                    buf.add_(p.grad)

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        # On update step, restore accumulated grads to parameters so HF can clip/step
        if self.accelerator.sync_gradients:
            if self._accum_grads is not None:
                for buf in self._accum_grads:
                    buf.div_(self.args.gradient_accumulation_steps)
                for buf, p in zip(self._accum_grads, model.parameters()):
                    p.grad = buf
            # Log per-dataset grad norm means and loss means
            if self.state.global_step > 0 and (self.state.global_step % self.args.logging_steps == 0):
                metrics = {}
                for name, s in self._per_dataset_grad_norm_sum.items():
                    cnt = max(1, self._per_dataset_grad_norm_cnt.get(name, 1))
                    metrics[f"grad_norm/{name}"] = s / cnt
                for name, s in self._per_dataset_loss_sum.items():
                    cnt = max(1, self._per_dataset_loss_cnt.get(name, 1))
                    metrics[f"loss/{name}"] = s / cnt
                if len(metrics):
                    self.log(metrics)
                # Reset counters
                self._per_dataset_grad_norm_sum.clear()
                self._per_dataset_grad_norm_cnt.clear()
                self._per_dataset_loss_sum.clear()
                self._per_dataset_loss_cnt.clear()
            # reset buffer; .grad will be cleared by HF later
            self._accum_grads = None

        return loss.detach()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
