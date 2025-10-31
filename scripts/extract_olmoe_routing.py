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

import json
import os
from typing import TYPE_CHECKING, Optional, cast

import fire
import numpy as np
import torch
from tqdm import tqdm
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from accelerate import Accelerator, DistributedDataParallelKwargs

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


if TYPE_CHECKING:
    from datasets import Dataset


def extract_olmoe_routing(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    output_dir: str = "routing_outputs",
    batch_size: int = 1,
    default_system: Optional[str] = None,
):
    r"""
    Extract routing outputs from OLMoE model on training data with multi-GPU support.

    This script loads an OLMoE model and performs forward passes on the specified dataset,
    capturing the router logits from each MoE layer. The routing information is saved to NPZ file.

    Supports multi-GPU data parallel inference using Accelerate. Each GPU loads a copy of the model
    and processes a portion of the dataset in parallel.

    Args:
        model_name_or_path: Path to the pretrained model or model identifier from huggingface.co/models
        adapter_name_or_path: Path to the LoRA adapter(s) to merge with the model
        dataset: Name of the dataset to use (must be registered in dataset_info.json)
        dataset_dir: Directory containing the dataset files
        template: Template name for processing the data
        cutoff_len: Maximum sequence length for input
        max_samples: Maximum number of samples to process (None for all)
        output_dir: Output directory for saving routing results
        batch_size: Batch size for processing (recommend 1 for OLMoE due to memory)
        default_system: Default system message to use in the template

    Usage:
        # Single GPU
        python extract_olmoe_routing.py \
            --model_name_or_path allenai/OLMoE-1B-7B-0924 \
            --dataset alpaca_en_demo \
            --template default \
            --output_dir routing_outputs

        # Multi-GPU (4 GPUs)
        accelerate launch --num_processes 4 extract_olmoe_routing.py \
            --model_name_or_path allenai/OLMoE-1B-7B-0924 \
            --dataset alpaca_en_demo \
            --template default \
            --output_dir routing_outputs
    """
    # Initialize Accelerator for multi-GPU support
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # Get dataset file name from dataset_info.json
    dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError(f"dataset_info.json not found at {dataset_info_path}")

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if dataset not in dataset_info:
        raise ValueError(f"Dataset '{dataset}' not found in dataset_info.json")

    dataset_file_name = dataset_info[dataset].get("file_name")
    if not dataset_file_name:
        raise ValueError(f"No file_name specified for dataset '{dataset}' in dataset_info.json")

    # Create output file path with same name as dataset file (but .npz extension)
    base_name = os.path.splitext(dataset_file_name)[0]
    save_name = os.path.join(output_dir, f"{base_name}.npz")

    # Only print on main process
    if accelerator.is_main_process:
        print("=" * 80)
        print("OLMoE Routing Extraction Script (Multi-GPU)")
        print("=" * 80)
        print(f"Model: {model_name_or_path}")
        print(f"Dataset: {dataset}")
        print(f"Dataset file: {dataset_file_name}")
        print(f"Template: {template}")
        print(f"Output file: {save_name}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Current process: {accelerator.process_index}")
        print("=" * 80)

    # Initialize arguments
    model_args, data_args, finetuning_args, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
        )
    )

    # Create training args for dataset loading
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")

    # Load tokenizer and template
    if accelerator.is_main_process:
        print("\n[1/4] Loading tokenizer and template...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load model
    if accelerator.is_main_process:
        print("\n[2/4] Loading model...")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    # Check if model is OLMoE
    model_type = getattr(model.config, "model_type", None)
    if model_type != "olmoe" and accelerator.is_main_process:
        print(f"Warning: Model type is '{model_type}', not 'olmoe'. This script is designed for OLMoE models.")
        print("Continuing anyway, but routing outputs may not be available.")

    model.eval()

    # Prepare model with accelerator
    model = accelerator.prepare(model)
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Model loaded on device: {device}")
        print(f"Model config: num_hidden_layers={model.config.num_hidden_layers}, "
              f"num_experts={getattr(model.config, 'num_experts', 'N/A')}, "
              f"num_experts_per_tok={getattr(model.config, 'num_experts_per_tok', 'N/A')}")

    # Load dataset
    if accelerator.is_main_process:
        print("\n[3/4] Loading dataset...")
    dataset_module = get_dataset(
        template_obj,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module
    )
    eval_dataset = dataset_module.get("eval_dataset")
    if eval_dataset is None:
        eval_dataset = dataset_module.get("train_dataset")

    if eval_dataset is None:
        raise ValueError("No dataset available. Please check your dataset configuration.")

    # Cast to Dataset for type checking
    if TYPE_CHECKING:
        eval_dataset = cast("Dataset", eval_dataset)

    dataset_len = len(eval_dataset)  # type: ignore
    if accelerator.is_main_process:
        print(f"Dataset loaded: {dataset_len} samples")

    # Split dataset across processes
    # Each process will handle a subset of the data
    samples_per_process = (dataset_len + accelerator.num_processes - 1) // accelerator.num_processes
    start_idx = accelerator.process_index * samples_per_process
    end_idx = min(start_idx + samples_per_process, dataset_len)

    if accelerator.is_main_process:
        print(f"\nData distribution across {accelerator.num_processes} processes:")
        for i in range(accelerator.num_processes):
            proc_start = i * samples_per_process
            proc_end = min(proc_start + samples_per_process, dataset_len)
            print(f"  Process {i}: samples {proc_start} to {proc_end-1} ({proc_end - proc_start} samples)")

    accelerator.print(f"Process {accelerator.process_index}: Processing samples {start_idx} to {end_idx-1}")

    # Process data and extract routing
    if accelerator.is_main_process:
        print("\n[4/4] Processing data and extracting routing outputs...")

    all_samples_router_logits = []  # List to store router logits for each sample
    all_sample_indices = []  # Store original sample indices

    with torch.no_grad():
        # Only process this process's portion of the dataset
        iterator = range(start_idx, end_idx, batch_size)
        if accelerator.is_main_process:
            iterator = tqdm(iterator, desc=f"Process {accelerator.process_index} extracting routing")
        else:
            iterator = tqdm(iterator, desc=f"Process {accelerator.process_index}", disable=not accelerator.is_local_main_process)

        for idx in iterator:
            # Get batch samples
            batch_samples = []
            batch_indices = []
            for i in range(idx, min(idx + batch_size, end_idx)):
                batch_samples.append(eval_dataset[i])  # type: ignore
                batch_indices.append(i)

            # Prepare batch inputs
            input_ids_list = [sample["input_ids"] for sample in batch_samples]
            attention_mask_list = [sample["attention_mask"] for sample in batch_samples]

            input_ids = torch.tensor(input_ids_list).to(device)
            attention_mask = torch.tensor(attention_mask_list).to(device)

            # Forward pass with router outputs enabled
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_router_logits=True,
                return_dict=True,
            )

            # Extract router logits for each sample
            num_layers = len(outputs.router_logits)

            # Process each sample in the batch
            for i in range(len(batch_samples)):
                sample_router_logits = []

                # Get attention mask for this sample to identify non-padding tokens
                sample_attention_mask = attention_mask[i].cpu().numpy()  # shape: [seq_len]
                valid_positions = sample_attention_mask == 1  # True for non-padding positions

                # Collect all layers' router logits for this sample
                for layer_idx in range(num_layers):
                    # router_logits shape: [batch_size, seq_len, num_experts]
                    layer_routing = outputs.router_logits[layer_idx][i].cpu().numpy()

                    # Only keep router logits for non-padding positions
                    layer_routing_valid = layer_routing[valid_positions]  # shape: [valid_seq_len, num_experts]
                    sample_router_logits.append(layer_routing_valid)

                # Stack layers into a single array: [num_layers, valid_seq_len, num_experts]
                all_samples_router_logits.append(np.stack(sample_router_logits, axis=0))
                all_sample_indices.append(batch_indices[i])

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Gather results from all processes
    if accelerator.is_main_process:
        print(f"\n[5/6] Gathering results from all processes...")

    # Save local results first
    local_save_dict = {}
    for idx, sample_logits in zip(all_sample_indices, all_samples_router_logits):
        local_save_dict[f"sample_{idx}"] = sample_logits

    # Use object_list to gather dictionaries from all processes
    if accelerator.num_processes > 1:
        import torch.distributed as dist
        gathered_dicts = [None] * accelerator.num_processes
        dist.all_gather_object(gathered_dicts, local_save_dict)
    else:
        gathered_dicts = [local_save_dict]

    # Only main process saves the results
    if accelerator.is_main_process:
        print(f"\n[6/6] Saving results to {save_name}...")
        save_dir = os.path.dirname(save_name)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Prepare data for NPZ format - merge all dictionaries from all processes
        save_dict = {}
        for proc_dict in gathered_dicts:
            if proc_dict is not None:
                save_dict.update(proc_dict)

        # Save as compressed NPZ
        np.savez_compressed(save_name, **save_dict)

        print("=" * 80)
        print(f"✓ Successfully extracted routing outputs for {len(save_dict)} samples")
        print(f"✓ Results saved to: {save_name}")
        print("=" * 80)

        # Print sample statistics
        if save_dict:
            print("\nSample statistics:")
            first_sample = list(save_dict.values())[0]
            print(f"  - Total samples: {len(save_dict)}")
            print(f"  - Shape per sample: {first_sample.shape} (num_layers, seq_len, num_experts)")
            print(f"  - Data type: {first_sample.dtype}")
            print(f"\nTo load a specific sample:")
            print(f"  data = np.load('{save_name}')")
            print(f"  sample_0 = data['sample_0']  # Shape: {first_sample.shape}")

    # Wait for main process to finish saving
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(extract_olmoe_routing)
