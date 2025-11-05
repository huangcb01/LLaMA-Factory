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
from collections import defaultdict
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer, Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
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
        self.model_args = model_args
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

        # buffer for custom metrics (per split) to be reduced/logged in `log`
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

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
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        # Detect split by model mode
        split = "train" if model.training else "eval"
        gold_logits = inputs.pop("gold_router_logits", None)
        inputs["output_router_logits"] = True
        inputs["return_dict"] = True
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch
            loss_log_multiplier = self.args.gradient_accumulation_steps
        else:
            loss_log_multiplier = 1.0
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        self._stored_metrics[split]["lm_loss"].append(loss.detach().float().mean().item() * loss_log_multiplier)
        if gold_logits is not None and getattr(self.finetuning_args, "moe_router_loss_weight", 0.0) > 0.0:
            aux_loss, router_acc = self._compute_gold_router_aux_loss(
                outputs.router_logits,
                gold_logits,
                inputs["attention_mask"],
            )
            if split == "train" and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                aux_loss /= self.args.gradient_accumulation_steps
            loss = loss + self.finetuning_args.moe_router_loss_weight * aux_loss
            self._stored_metrics[split]["gold_router_aux_loss"].append(aux_loss.detach().float().mean().item() * loss_log_multiplier)
            self._stored_metrics[split]["total_loss"].append(loss.detach().float().mean().item() * loss_log_multiplier)
            self._stored_metrics[split]["router_topk_acc"].append(
                router_acc.detach().float().mean().item()
            )
        if return_outputs:
            return loss, outputs
        return loss

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """Log `logs` on the various objects watching training, including stored metrics.

        We aggregate and all-reduce custom losses (lm_loss, gold_router_aux_loss, total_loss)
        and inject them into the log history so that both trainer_state.json and
        trainer_log.jsonl contain these metrics for plotting.
        """
        # Decide split and prefix based on presence of loss keys
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""

        # Nothing to add
        if len(self._stored_metrics[train_eval]) == 0:
            return super().log(logs, *args, **kwargs)

        # Gather keys and values, pad to fixed length for safe all-reduce
        key_list, metric_list = [], []
        for key, values in self._stored_metrics[train_eval].items():
            key_list.append(key)
            # average within process first
            try:
                metric_list.append(torch.tensor(values, dtype=torch.float, device=self.accelerator.device).mean().item())
            except Exception:
                # fallback to python average
                metric_list.append(float(sum(values) / max(len(values), 1)))

        # clear stored metrics for this split
        del self._stored_metrics[train_eval]

        # pad to at least 4 to avoid potential collective issues
        while len(metric_list) < 4:
            key_list.append(f"dummy_{len(metric_list)}")
            metric_list.append(0.0)

        tensor_metrics = torch.tensor(metric_list, dtype=torch.float, device=self.accelerator.device)
        # all-reduce (mean across processes)
        reduced = self.accelerator.reduce(tensor_metrics, "mean").tolist()

        # write back to logs with prefix for eval
        for key, metric in zip(key_list, reduced):
            if not key.startswith("dummy_"):
                logs[f"{prefix}{key}"] = metric

        return Trainer.log(self, logs, *args, **kwargs)

    def _compute_gold_router_aux_loss(
        self,
        router_logits: tuple,
        gold_router_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute auxiliary loss and Top-K accuracy based on gold router activated expert indices.

        Supports two loss types controlled by `self.finetuning_args.moe_router_loss_type`:

        1) "set" (default): Original set-based symmetric softplus penalty encouraging overlap of Top-K sets.
           L_set = sum_{tokens,layers} [ softplus(-logits_e) for gold-but-missed experts
                                        + softplus(logits_e) for extra predicted experts ] / (#valid_tokens * L)

        2) "kl": KL divergence between a target distribution Q (uniform over gold Top-K experts) and
           model distribution P = softmax(logits). We minimize KL(Q||P) = -H(Q) + CE(Q,P).
           Since H(Q) is constant w.r.t. parameters, we implement CE(Q,P) = - sum_{e in gold} (1/K) log P_e.
           We mask padding tokens similarly and average over tokens*layers.

        Router Top-K accuracy (per token, per layer):
            acc = |TopK_model ∩ TopK_gold| / K

        Args:
            router_logits: Tuple of router logits from each layer, shape: (batch*seq_len, num_experts)
            gold_indices: Gold router indices, shape: (batch, num_layers, seq_len, top_k)
            attention_mask: Attention mask, shape: (batch, seq_len)

        Returns:
            A tuple of (auxiliary loss scalar, router top-k accuracy scalar)
        """
        device = router_logits[0].device
        # gold_router_logits: (B, num_layers, seq_len, num_experts)
        gold_router_logits = gold_router_logits.to(device)
        batch_size, num_layers, seq_len, num_experts_gold = gold_router_logits.shape
        # derive a pseudo top_k for accuracy if needed
        top_k = self._get_top_k(self.model)

        # Stack layers -> (B, S, L, E)
        logits = torch.stack([r.view(batch_size, seq_len, -1) for r in router_logits], dim=2)
        # Top-k from model per (B,S,L)
        _, topk_idx = torch.topk(logits.detach(), k=top_k, dim=-1)  # (B, S, L, K)

        # Build boolean masks (B, S, L, E)
        in_model_topk = torch.zeros_like(logits, dtype=torch.bool)
        in_model_topk.scatter_(-1, topk_idx, True)

        in_gold_topk = torch.zeros_like(logits, dtype=torch.bool)
        # derive gold top-k from gold logits for accuracy & set loss fallback
        gold_logits_bsl_e = gold_router_logits.permute(0, 2, 1, 3).contiguous()  # (B,S,L,E)
        _, gold_topk_idx = torch.topk(gold_logits_bsl_e.detach(), k=top_k, dim=-1)
        in_gold_topk.scatter_(-1, gold_topk_idx.long(), True)

        loss_type = getattr(self.finetuning_args, "moe_router_loss_type", "set")
        if loss_type == "kl":
            if gold_router_logits is not None:
                gold_logits_bsl_e = gold_router_logits.permute(0, 2, 1, 3).contiguous()  # (B,S,L,E)
                model_log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = torch.log_softmax(gold_logits_bsl_e, dim=-1)
                target_probs = target_log_probs.exp()
                kl = (target_probs * (target_log_probs - model_log_probs)).sum(dim=-1)  # (B,S,L)
                token_layer_loss = kl.clamp_min(0)
            else:
                # Fallback: uniform over gold top-k indices
                probs = torch.softmax(logits, dim=-1)
                gold_mask = in_gold_topk.float()
                denom = gold_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                uniform_q = gold_mask / denom
                ce = -(uniform_q * (probs.clamp_min(1e-12).log())).sum(dim=-1)
                token_layer_loss = ce
        else:
            # Original set-based loss
            loss_missing = ((in_gold_topk & ~in_model_topk).float() * F.softplus(-logits)).sum(dim=-1)  # (B, S, L)
            loss_extra = ((~in_gold_topk & in_model_topk).float() * F.softplus(logits)).sum(dim=-1)  # (B, S, L)
            token_layer_loss = loss_missing + loss_extra  # (B, S, L)

        mask2d = attention_mask.float()  # expected shape (B, S)
        token_layer_loss = token_layer_loss * mask2d.unsqueeze(-1)  # (B, S, L)
        num_valid_tokens = mask2d.sum().item() * num_layers
        total_loss = token_layer_loss.sum()
        if num_valid_tokens > 0:
            aux_loss = total_loss / num_valid_tokens
        else:
            aux_loss = torch.tensor(0.0, device=device)

        # Router Top-K accuracy using the same masks and shapes
        overlap = (in_model_topk & in_gold_topk).sum(dim=-1).float()  # (B, S, L)
        acc_bsl = overlap / max(top_k, 1)
        acc_bsl = acc_bsl * mask2d.unsqueeze(-1)
        total_acc = acc_bsl.sum()
        if num_valid_tokens > 0:
            router_acc = total_acc / num_valid_tokens
        else:
            router_acc = torch.tensor(0.0, device=device)

        return aux_loss, router_acc

    def _get_top_k(self, model) -> int:
        if hasattr(model, "module") and hasattr(model.module, "config"):
            return getattr(model.module.config, "num_experts_per_tok", 1)
        elif hasattr(model, "config"):
            return getattr(model.config, "num_experts_per_tok", 1)
        else:
            raise ValueError("Cannot determine num_experts_per_tok from model config.")

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
