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
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np
import torch

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


class GoldRouterIndicesLoader:
    r"""Loader for gold router activated expert indices used in MoE auxiliary loss.

    Note: The stored files are .npz with per-sample arrays shaped as
    [num_layers, valid_seq_len, top_k] containing integer expert indices.
    """

    def __init__(self, data_args: "DataArguments"):
        self.data_args = data_args
        self.logits_cache = {}  # Cache loaded indices files (npz)
        self.dataset_to_file = {}  # Map dataset name to indices file

        if data_args.gold_router_logits_dir is not None:
            self._initialize_mappings()

    def _initialize_mappings(self):
        """Initialize mappings between datasets and gold router indices files."""
        datasets = self.data_args.dataset if self.data_args.dataset is not None else []

        if len(datasets) == 0:
            return

        # Load dataset_info.json to get dataset file names
        dataset_info_path = os.path.join(self.data_args.dataset_dir, "dataset_info.json")
        if not os.path.exists(dataset_info_path):
            logger.warning_rank0(f"dataset_info.json not found at {dataset_info_path}")
            return

        try:
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception as e:
            logger.warning_rank0(f"Failed to load dataset_info.json: {e}")
            return

        # Map each dataset to its corresponding logits file
        for dataset_name in datasets:
            if dataset_name not in dataset_info:
                logger.warning_rank0(f"Dataset '{dataset_name}' not found in dataset_info.json")
                continue

            dataset_file_name = dataset_info[dataset_name].get("file_name")
            if not dataset_file_name:
                logger.warning_rank0(f"No file_name specified for dataset '{dataset_name}' in dataset_info.json")
                continue

            # Get base name without extension and add .npz
            base_name = os.path.splitext(dataset_file_name)[0]
            logits_file = f"{base_name}.npz"

            # Check if the file exists in the gold_router_logits_dir
            dir_path = cast(str, self.data_args.gold_router_logits_dir)
            logits_path = os.path.join(dir_path, logits_file)
            if os.path.exists(logits_path):
                self.dataset_to_file[dataset_name] = logits_file
                logger.info_rank0(f"Found gold router indices for dataset '{dataset_name}': {logits_file}")
            else:
                logger.warning_rank0(f"Gold router indices file not found for dataset '{dataset_name}': {logits_path}")

        logger.info_rank0(f"Gold router indices mappings: {self.dataset_to_file}")

    def load_indices_file(self, logits_path: str):
        """Load a gold router indices file. Returns NpzFile object or None."""
        if logits_path in self.logits_cache:
            return self.logits_cache[logits_path]
        dir_path = cast(str, self.data_args.gold_router_logits_dir)
        full_path = os.path.join(dir_path, logits_path)
        if not os.path.exists(full_path):
            logger.warning_rank0(f"Gold router indices file not found: {full_path}")
            return None

        try:
            logits_data = np.load(full_path)
            self.logits_cache[logits_path] = logits_data
            logger.info_rank0(f"Loaded gold router indices from {full_path}, contains {len(logits_data.files)} samples")
            return logits_data
        except Exception as e:
            logger.warning_rank0(f"Failed to load gold router indices from {full_path}: {e}")
            return None

    def get_sample_indices(self, dataset_name: str, sample_idx: int) -> Optional[torch.Tensor]:
        """Get gold router activated expert indices for a specific sample.

        Returns
        -------
        Optional[torch.Tensor]
            Tensor of shape [num_layers, seq_len, top_k] with dtype torch.int64
        """
        if dataset_name not in self.dataset_to_file:
            return None

        logits_path = self.dataset_to_file[dataset_name]
        logits_data = self.load_indices_file(logits_path)

        if logits_data is None:
            return None

        sample_key = f"sample_{sample_idx}"
        if sample_key not in logits_data:
            return None

        # Convert to torch tensor: [num_layers, seq_len, top_k] (indices)
        arr = logits_data[sample_key]
        return torch.from_numpy(arr).to(torch.long)

    def has_gold_indices(self, dataset_name: str) -> bool:
        """Check if gold router indices are available for a dataset."""
        return dataset_name in self.dataset_to_file


# Global instance
_global_loader: Optional[GoldRouterIndicesLoader] = None


def initialize_gold_router_loader(data_args: "DataArguments"):
    """Initialize the global gold router indices loader.

    Args:
        data_args: DataArguments object containing dataset and gold_router_logits_dir configuration.
    """
    global _global_loader
    _global_loader = GoldRouterIndicesLoader(data_args)


def get_gold_router_loader() -> Optional[GoldRouterIndicesLoader]:
    """Get the global gold router indices loader."""
    return _global_loader
