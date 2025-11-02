# Copyright 2025 the LlamaFactory team.
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

import os
import gc
import datasets
import fire
import numpy as np
import torch
import torch.distributed as dist
from typing import Any, Optional, Literal, cast
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from typing_extensions import override

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


# Use a logger under the `llamafactory` namespace so it shares the configured handlers
# from `llamafactory.extras.logging` and actually prints to stdout.
logger = get_logger("llamafactory.scripts.extract_olmoe_routing")


class RouterExtractionTrainer(Seq2SeqTrainer):
    """Custom trainer for extracting activated expert indices from MoE models."""

    def __init__(self, output_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.output_dir_routing = output_dir
        # Infer top-k from model config field `num_experts_per_tok`
        model_config = self.model.config if hasattr(self.model, "config") else self.model.module.config
        self.top_k = int(getattr(model_config, "num_experts_per_tok"))
        logger.info(f"Using router top_k={self.top_k} (from model.config.num_experts_per_tok)")
        self.all_activated_indices: dict[str, np.ndarray] = {}

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Override prediction_step to extract activated expert indices (top-k)."""
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Forward pass with router outputs
            original_indices = inputs.pop("original_index")  # Extract original indices
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_router_logits=True,
                return_dict=True,
            )

            # Extract router logits and compute top-k indices
            batch_size = inputs["input_ids"].size(0)
            seq_len = inputs["input_ids"].size(1)
            num_layers = len(outputs.router_logits)
            num_experts = outputs.router_logits[0].size(-1)

            # Stack all layers: tuple of [batch*seq_len, num_experts] -> [num_layers, batch, seq_len, num_experts]
            router_logits = torch.stack(outputs.router_logits, dim=0).view(num_layers, batch_size, seq_len, num_experts)
            attention_mask = inputs["attention_mask"].to(torch.bool)

            # Compute top-k indices for each layer/batch/seq token
            # Shapes: values/indices -> [num_layers, batch, seq_len, top_k]
            _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)

            # Process each sample in the batch
            for i in range(batch_size):
                # Filter by attention mask: [num_layers, seq_len, top_k] -> [num_layers, valid_seq_len, top_k]
                sample_topk_idx = topk_indices[:, i, attention_mask[i], :].to(torch.int32).cpu().numpy()

                # Store with original index as key (already in final dict format)
                self.all_activated_indices[f"sample_{original_indices[i].item()}"] = sample_topk_idx

        # Return dummy values (we don't care about loss/predictions)
        return (None, None, None)

    def save_activated_indices(self, dataset_name: str):
        """Save collected activated expert indices in a DDP-safe way.

        - Each rank writes its own shard: {dataset}.rank{r}.npz
        - Rank 0 waits for all ranks, merges shards into {dataset}.npz, and cleans up.
        """
        dist_available = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_available else 0
        world_size = dist.get_world_size() if dist_available else 1

        # Save shard for this rank
        os.makedirs(self.output_dir_routing, exist_ok=True)
        shard_path = os.path.join(self.output_dir_routing, f"{dataset_name}.rank{rank}.npz")
        logger.info(f"Saving activated expert indices shard (rank {rank}/{world_size}) to {shard_path}")
        np.savez(shard_path, **self.all_activated_indices)
        logger.info(f"✓ Rank {rank}: saved {len(self.all_activated_indices)} samples")
        self.all_activated_indices = {}  # Free memory after writing this shard
        gc.collect()  # Proactively trigger GC to free tensor/array memory

        # Sync all ranks before merging
        if dist_available:
            dist.barrier()

        # Merge shards on rank 0
        if (not dist_available) or rank == 0:
            merged = {}
            shard_paths = []
            for r in range(world_size):
                p = os.path.join(self.output_dir_routing, f"{dataset_name}.rank{r}.npz")
                # Ensure the NPZ file is closed immediately after reading to release file/memory resources
                with np.load(p, allow_pickle=False) as data:
                    for k in data.files:
                        merged[k] = data[k]
                shard_paths.append(p)

            final_path = os.path.join(self.output_dir_routing, f"{dataset_name}.npz")
            logger.info(f"Merging {len(shard_paths)} shard(s) -> {final_path}")
            np.savez(final_path, **merged)
            logger.info(f"✓ Successfully saved {len(merged)} samples (activated indices) in original order to {final_path}")
            # Help the GC by dropping large temporary maps
            merged.clear()
            gc.collect()

            # Cleanup shard files
            for p in shard_paths:
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning(f"Failed to remove shard file {p}: {e}")


def main(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    output_dir: str = "routing_outputs",
    batch_size: int = 4,
    default_system: Optional[str] = None,
):
    """
    提取 OLMoE 模型的路由激活信息（仅保存被激活的专家索引），基于 Trainer 基础设施。

    Args:
        model_name_or_path: Path to the pretrained model
        adapter_name_or_path: Path to the LoRA adapter(s)
        dataset: Dataset name(s), comma-separated for multiple datasets
        dataset_dir: Directory containing dataset files
        template: Template name for processing
        cutoff_len: Maximum sequence length
        max_samples: Maximum number of samples to process
        output_dir: Output directory for routing results
        batch_size: Batch size for processing
        default_system: Default system message
    """
    model_args, data_args, finetuning_args, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            eval_dataset=dataset,
            eval_on_each_dataset=True,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=os.cpu_count(),
            default_system=default_system,
        )
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=1,
        remove_unused_columns=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        prediction_loss_only=False,
        disable_tqdm=False,
    )

    # Load tokenizer and model (once for all datasets)
    logger.info_rank0("Loading tokenizer and template...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    logger.info_rank0("Loading model...")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    # Check model type
    model_type = getattr(model.config, "model_type", None)
    if model_type != "olmoe":
        logger.warning(f"Model type is '{model_type}', not 'olmoe'. " "This script is designed for OLMoE models.")

    # Load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    eval_datasets = dataset_module.get("eval_dataset")
    assert isinstance(eval_datasets, dict), "eval_dataset should be a dict of datasets for multiple datasets."

    # Create data collator
    attn_impl = cast(Literal['eager', 'sdpa', 'flash_attention_2'], getattr(model.config, "_attn_implementation", "eager"))
    compute_dtype = cast(torch.dtype, model_args.compute_dtype if model_args.compute_dtype is not None else torch.float32)
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template_obj,
        model=None,
        pad_to_multiple_of=None,
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=attn_impl,
        compute_dtype=compute_dtype,
        **tokenizer_module,
    )

    # Create custom trainer
    trainer = RouterExtractionTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        output_dir=output_dir,
        tokenizer=tokenizer,
    )

    # Process each dataset
    for dataset_name, eval_dataset in eval_datasets.items():
        logger.info_rank0("=" * 80)
        logger.info_rank0(f"Processing dataset {dataset_name} with {len(eval_dataset)} samples...")
        logger.info_rank0("=" * 80)

        # Sort dataset by sequence length (longest first) to minimize padding
        with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
            logger.info_rank0("Sorting dataset by sequence length to minimize padding...")
            eval_dataset = eval_dataset.map(
                lambda x, idx: {"length": len(x["input_ids"]), "original_index": idx},
                with_indices=True,
                num_proc=os.cpu_count(),
            )
            eval_dataset = eval_dataset.sort("length", reverse=True)
            lengths = eval_dataset["length"]
            eval_dataset = eval_dataset.remove_columns("length")
            logger.info_rank0(f"  Longest sequence: {lengths[0]} tokens")
            logger.info_rank0(f"  Shortest sequence: {lengths[-1]} tokens")
            logger.info_rank0(f"  Average length: {sum(lengths) / len(lengths):.1f} tokens")

        # Run prediction to extract activated indices
        logger.info_rank0("Extracting activated expert indices (top-k)...")
        trainer.predict(eval_dataset)  # type: ignore

        # Save results
        trainer.save_activated_indices(dataset_name)


if __name__ == "__main__":
    fire.Fire(main)
