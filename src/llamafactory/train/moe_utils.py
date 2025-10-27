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

"""Utilities for handling Mixture of Experts (MoE) models."""

from typing import TYPE_CHECKING, Optional, Tuple

from ..extras.logging import get_logger


if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig


logger = get_logger(__name__)


def get_moe_expert_info(config: "PretrainedConfig") -> Tuple[Optional[int], Optional[int]]:
    """
    Extract MoE expert information from model configuration.

    Args:
        config: The model configuration object.

    Returns:
        A tuple of (default_num_experts, num_experts):
        - default_num_experts: Default number of activated experts (num_experts_per_tok)
        - num_experts: Total number of experts in the model
        Returns (None, None) if the model is not a supported MoE model.
    """
    text_config = getattr(config, "text_config", None)

    # Check text_config first (for multimodal models)
    if text_config is not None:
        default_num_experts = getattr(text_config, "num_experts_per_tok", None)
        num_experts = getattr(text_config, "num_experts", None)
        if default_num_experts is not None and num_experts is not None:
            return default_num_experts, num_experts

    # Check main config
    default_num_experts = getattr(config, "num_experts_per_tok", None)
    num_experts = getattr(config, "num_experts", None)
    if default_num_experts is not None and num_experts is not None:
        return default_num_experts, num_experts

    return None, None


def update_moe_num_experts(model: "torch.nn.Module", num_experts: int) -> bool:
    """
    Update the number of activated experts in the MoE model.

    This function supports multiple MoE architectures by detecting the model type
    and updating the appropriate configuration parameter.

    Args:
        model: The model instance to update.
        num_experts: The new number of activated experts.

    Returns:
        True if the update was successful, False otherwise.
    """
    config = getattr(model, "config", None)
    if config is None:
        logger.warning_rank0("Model has no config attribute, cannot update MoE expert count")
        return False

    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", None)
    updated = False

    # For multimodal models with text_config
    if text_config is not None:
        text_model_type = getattr(text_config, "model_type", None)
        if hasattr(text_config, "num_experts_per_tok"):
            setattr(text_config, "num_experts_per_tok", num_experts)
            logger.info_rank0(f"Updated text_config.num_experts_per_tok to {num_experts}")
            updated = True

    # For standard MoE models
    if not updated:
        # Most MoE models use num_experts_per_tok
        if hasattr(config, "num_experts_per_tok"):
            setattr(config, "num_experts_per_tok", num_experts)
            logger.info_rank0(f"Updated config.num_experts_per_tok to {num_experts}")
            updated = True
        # DeepSeek models might use different parameter names
        elif hasattr(config, "n_routed_experts"):
            # For DeepSeek, we update the top_k parameter
            if hasattr(config, "num_experts_per_tok"):
                setattr(config, "num_experts_per_tok", num_experts)
                logger.info_rank0(f"Updated config.num_experts_per_tok to {num_experts}")
                updated = True
        elif hasattr(config, "topk_method"):
            # Some models might have topk_method instead
            logger.warning_rank0(f"Model type {model_type} uses topk_method, manual update may be needed")

    # Update the model layers if they have been instantiated
    # This ensures that already-instantiated layers also use the new configuration
    if updated and hasattr(model, "model"):
        base_model = model.model
        if hasattr(base_model, "layers"):
            for layer in base_model.layers:
                # Check for MoE block in the layer
                moe_block = None
                if hasattr(layer, "block_sparse_moe"):
                    moe_block = layer.block_sparse_moe
                elif hasattr(layer, "mlp"):
                    if hasattr(layer.mlp, "num_experts_per_tok"):
                        moe_block = layer.mlp

                if moe_block is not None and hasattr(moe_block, "num_experts_per_tok"):
                    moe_block.num_experts_per_tok = num_experts

    if not updated:
        logger.warning_rank0(
            f"Failed to update MoE expert count for model type '{model_type}'. "
            f"Model may not be a supported MoE architecture."
        )

    return updated
