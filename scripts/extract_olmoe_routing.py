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
import fire
import numpy as np
import torch
from typing import Any, Optional
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from typing_extensions import override

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


logger = get_logger(__name__)


class RouterExtractionTrainer(Seq2SeqTrainer):
    """Custom trainer for extracting router logits from MoE models."""

    def __init__(self, output_dir: str, dataset_name: str, **kwargs):
        super().__init__(**kwargs)
        self.output_dir_routing = output_dir
        self.dataset_name = dataset_name
        self.all_router_logits = []
        self.all_sample_indices = []

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Override prediction_step to extract router logits."""
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Forward pass with router outputs
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_router_logits=True,
                return_dict=True,
            )

            # Extract and store router logits
            batch_size = inputs["input_ids"].size(0)
            seq_len = inputs["input_ids"].size(1)
            num_layers = len(outputs.router_logits)
            num_experts = outputs.router_logits[0].size(-1)

            # Stack all layers: tuple of [batch*seq_len, num_experts] -> [num_layers, batch, seq_len, num_experts]
            router_logits = torch.stack(outputs.router_logits, dim=0).view(num_layers, batch_size, seq_len, num_experts)
            attention_mask = inputs["attention_mask"].to(torch.bool)

            # Process each sample in the batch
            for i in range(batch_size):
                # Filter by attention mask: [num_layers, seq_len, num_experts] -> [num_layers, valid_seq_len, num_experts]
                sample_router_logits = router_logits[:, i, attention_mask[i], :].float().cpu().numpy()
                self.all_router_logits.append(sample_router_logits)

        # Return dummy values (we don't care about loss/predictions)
        return (None, None, None)

    def save_router_logits(self):
        """Save collected router logits to file."""
        if not self.is_world_process_zero():
            return

        os.makedirs(self.output_dir_routing, exist_ok=True)
        save_path = os.path.join(self.output_dir_routing, f"{self.dataset_name}.npz")

        logger.info(f"Saving router logits to {save_path}")

        # Prepare save dict
        save_dict = {}
        for idx, sample_logits in enumerate(self.all_router_logits):
            save_dict[f"sample_{idx}"] = sample_logits

        # Save as compressed NPZ
        np.savez_compressed(save_path, **save_dict)

        logger.info(f"✓ Successfully saved {len(save_dict)} samples")
        if save_dict:
            first_sample = list(save_dict.values())[0]
            logger.info(f"  Shape per sample: {first_sample.shape} (num_layers, valid_seq_len, num_experts)")


def extract_olmoe_routing(
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
    Extract routing outputs from OLMoE model using Trainer infrastructure.

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

    Usage:
        # Single GPU
        python extract_olmoe_routing_v2.py \\
            --model_name_or_path allenai/OLMoE-1B-7B-0924 \\
            --dataset alpaca_en_demo \\
            --batch_size 4

        # Multi-GPU (automatic with Trainer)
        CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_olmoe_routing_v2.py \\
            --model_name_or_path allenai/OLMoE-1B-7B-0924 \\
            --dataset alpaca_en_demo,alpaca_zh_demo \\
            --batch_size 4
    """
    # Parse dataset names
    dataset_names = [name.strip() for name in dataset.split(",")]

    logger.info("=" * 80)
    logger.info("OLMoE Routing Extraction Script")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Datasets: {', '.join(dataset_names)} ({len(dataset_names)} total)")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Load tokenizer and model (once for all datasets)
    logger.info("Loading tokenizer and template...")
    model_args, data_args, finetuning_args, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset_names[0],
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
        )
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    logger.info("Loading model...")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    # Check model type
    model_type = getattr(model.config, "model_type", None)
    if model_type != "olmoe":
        logger.warning(
            f"Model type is '{model_type}', not 'olmoe'. "
            "This script is designed for OLMoE models."
        )

    # Process each dataset
    for dataset_idx, current_dataset in enumerate(dataset_names):
        logger.info("=" * 80)
        logger.info(f"Processing dataset {dataset_idx + 1}/{len(dataset_names)}: {current_dataset}")
        logger.info("=" * 80)

        # Update data_args for current dataset
        _, data_args, _, _ = get_infer_args(
            dict(
                model_name_or_path=model_name_or_path,
                adapter_name_or_path=adapter_name_or_path,
                dataset=current_dataset,
                dataset_dir=dataset_dir,
                template=template,
                cutoff_len=cutoff_len,
                max_samples=max_samples,
                preprocessing_num_workers=16,
                default_system=default_system,
            )
        )

        # Create minimal training args for prediction
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            do_train=False,
            do_eval=False,
            do_predict=True,
            prediction_loss_only=False,
            disable_tqdm=False,
        )

        # Load dataset
        logger.info(f"Loading dataset: {current_dataset}...")
        dataset_module = get_dataset(
            template_obj,
            model_args,
            data_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )

        eval_dataset = dataset_module.get("eval_dataset")
        if eval_dataset is None:
            eval_dataset = dataset_module.get("train_dataset")

        if eval_dataset is None:
            logger.error(f"No dataset available for '{current_dataset}'")
            continue

        logger.info(f"Dataset loaded: {len(eval_dataset)} samples")

        # Create data collator
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=template_obj,
            model=None,
            pad_to_multiple_of=None,
            label_pad_token_id=IGNORE_INDEX,
            block_diag_attn=model_args.block_diag_attn,
            attn_implementation=getattr(model.config, "_attn_implementation", None),
            compute_dtype=model_args.compute_dtype,
            **tokenizer_module,
        )

        # Create custom trainer
        trainer = RouterExtractionTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            dataset_name=current_dataset,
            tokenizer=tokenizer,
        )

        # Run prediction to extract router logits
        logger.info("Extracting router logits...")
        trainer.predict(eval_dataset)

        # Save results
        trainer.save_router_logits()

    logger.info("=" * 80)
    logger.info(f"✓ All {len(dataset_names)} datasets processed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(extract_olmoe_routing)
