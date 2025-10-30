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
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


class GoldRouterLogitsLoader:
    r"""Loader for gold router logits used in MoE auxiliary loss."""

    def __init__(self, data_args: "DataArguments"):
        self.data_args = data_args
        self.logits_cache = {}  # Cache loaded logits files
        self.dataset_to_file = {}  # Map dataset name to logits file

        if data_args.gold_router_logits_path is not None:
            self._initialize_mappings()

    def _initialize_mappings(self):
        """Initialize mappings between datasets and gold router logits files."""
        datasets = self.data_args.dataset if self.data_args.dataset is not None else []
        logits_paths = self.data_args.gold_router_logits_path if self.data_args.gold_router_logits_path is not None else []

        if len(logits_paths) == 0:
            return

        if len(logits_paths) == 1 and len(datasets) > 1:
            # Use the same logits file for all datasets
            for dataset_name in datasets:
                self.dataset_to_file[dataset_name] = logits_paths[0]
        elif len(logits_paths) == len(datasets):
            # One-to-one mapping
            for dataset_name, logits_path in zip(datasets, logits_paths):
                if logits_path:  # Skip empty paths
                    self.dataset_to_file[dataset_name] = logits_path
        else:
            raise ValueError(
                f"Number of gold_router_logits_path ({len(logits_paths)}) must be 1 or equal to "
                f"number of datasets ({len(datasets)})."
            )

        logger.info_rank0(f"Gold router logits mappings: {self.dataset_to_file}")

    def load_logits_file(self, logits_path: str):
        """Load a gold router logits file. Returns NpzFile object or None."""
        if logits_path in self.logits_cache:
            return self.logits_cache[logits_path]

        full_path = os.path.join(self.data_args.dataset_dir, logits_path)
        if not os.path.exists(full_path):
            logger.warning_rank0(f"Gold router logits file not found: {full_path}")
            return None

        try:
            logits_data = np.load(full_path)
            self.logits_cache[logits_path] = logits_data
            logger.info_rank0(f"Loaded gold router logits from {full_path}, contains {len(logits_data.files)} samples")
            return logits_data
        except Exception as e:
            logger.warning_rank0(f"Failed to load gold router logits from {full_path}: {e}")
            return None

    def get_sample_logits(self, dataset_name: str, sample_idx: int) -> Optional[torch.Tensor]:
        """Get gold router logits for a specific sample."""
        if dataset_name not in self.dataset_to_file:
            return None

        logits_path = self.dataset_to_file[dataset_name]
        logits_data = self.load_logits_file(logits_path)

        if logits_data is None:
            return None

        sample_key = f"sample_{sample_idx}"
        if sample_key not in logits_data:
            return None

        # Convert to torch tensor: [num_layers, seq_len, num_experts]
        return torch.from_numpy(logits_data[sample_key])

    def has_gold_logits(self, dataset_name: str) -> bool:
        """Check if gold router logits are available for a dataset."""
        return dataset_name in self.dataset_to_file


# Global instance
_global_loader: Optional[GoldRouterLogitsLoader] = None


def initialize_gold_router_loader(data_args: Union["DataArguments", dict[str, str]]):
    """Initialize the global gold router logits loader.

    Args:
        data_args: Can be either DataArguments object or a dict mapping dataset names to logits paths.
    """
    global _global_loader
    if isinstance(data_args, dict):
        # Create a mock DataArguments with the provided mapping
        from types import SimpleNamespace
        mock_args = SimpleNamespace()
        mock_args.dataset = ",".join(data_args.keys())
        mock_args.gold_router_logits_path = ",".join(data_args.values())
        _global_loader = GoldRouterLogitsLoader(mock_args)  # type: ignore
    else:
        _global_loader = GoldRouterLogitsLoader(data_args)


def get_gold_router_loader() -> Optional[GoldRouterLogitsLoader]:
    """Get the global gold router logits loader."""
    return _global_loader
