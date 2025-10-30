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
    Extract routing outputs from OLMoE model on training data.

    This script loads an OLMoE model and performs forward passes on the specified dataset,
    capturing the router logits from each MoE layer. The routing information is saved to NPZ file.

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
        python extract_olmoe_routing.py \
            --model_name_or_path allenai/OLMoE-1B-7B-0924 \
            --dataset alpaca_en_demo \
            --template default \
            --output_dir routing_outputs
    """
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

    print("=" * 80)
    print("OLMoE Routing Extraction Script")
    print("=" * 80)
    print(f"Model: {model_name_or_path}")
    print(f"Dataset: {dataset}")
    print(f"Dataset file: {dataset_file_name}")
    print(f"Template: {template}")
    print(f"Output file: {save_name}")
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
    print("\n[1/4] Loading tokenizer and template...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load model
    print("\n[2/4] Loading model...")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    # Check if model is OLMoE
    model_type = getattr(model.config, "model_type", None)
    if model_type != "olmoe":
        print(f"Warning: Model type is '{model_type}', not 'olmoe'. This script is designed for OLMoE models.")
        print("Continuing anyway, but routing outputs may not be available.")

    model.eval()
    device = next(model.parameters()).device

    print(f"Model loaded on device: {device}")
    print(f"Model config: num_hidden_layers={model.config.num_hidden_layers}, "
          f"num_experts={getattr(model.config, 'num_experts', 'N/A')}, "
          f"num_experts_per_tok={getattr(model.config, 'num_experts_per_tok', 'N/A')}")

    # Load dataset
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
    print(f"Dataset loaded: {dataset_len} samples")

    # Process data and extract routing
    print("\n[4/4] Processing data and extracting routing outputs...")
    all_samples_router_logits = []  # List to store router logits for each sample

    with torch.no_grad():
        for idx in tqdm(range(0, dataset_len, batch_size), desc="Extracting routing"):
            # Get batch samples
            batch_samples = []
            for i in range(idx, min(idx + batch_size, dataset_len)):
                batch_samples.append(eval_dataset[i])  # type: ignore

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
            if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                num_layers = len(outputs.router_logits)

                # Process each sample in the batch
                for i in range(len(batch_samples)):
                    sample_router_logits = []

                    # Collect all layers' router logits for this sample
                    for layer_idx in range(num_layers):
                        # router_logits shape: [batch_size, seq_len, num_experts]
                        layer_routing = outputs.router_logits[layer_idx][i].cpu().numpy()
                        sample_router_logits.append(layer_routing)

                    # Stack layers into a single array: [num_layers, seq_len, num_experts]
                    all_samples_router_logits.append(np.stack(sample_router_logits, axis=0))

    # Save results
    print(f"\n[5/5] Saving results to {save_name}...")
    save_dir = os.path.dirname(save_name)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Prepare data for NPZ format - store by sample
    save_dict = {}
    for sample_idx, sample_logits in enumerate(all_samples_router_logits):
        # Each sample's data shape: [num_layers, seq_len, num_experts]
        save_dict[f"sample_{sample_idx}"] = sample_logits

    # Save as compressed NPZ
    np.savez_compressed(save_name, **save_dict)

    print("=" * 80)
    print(f"✓ Successfully extracted routing outputs for {len(all_samples_router_logits)} samples")
    print(f"✓ Results saved to: {save_name}")
    print("=" * 80)

    # Print sample statistics
    if all_samples_router_logits:
        print("\nSample statistics:")
        first_sample = all_samples_router_logits[0]
        print(f"  - Total samples: {len(all_samples_router_logits)}")
        print(f"  - Shape per sample: {first_sample.shape} (num_layers, seq_len, num_experts)")
        print(f"  - Data type: {first_sample.dtype}")
        print(f"\nTo load a specific sample:")
        print(f"  data = np.load('{save_name}')")
        print(f"  sample_0 = data['sample_0']  # Shape: {first_sample.shape}")


if __name__ == "__main__":
    fire.Fire(extract_olmoe_routing)
