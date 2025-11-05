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
import math
import os
from typing import Any

from transformers.trainer import TRAINER_STATE_NAME

from . import logging
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)


def smooth(scalars: list[float]) -> list[float]:
    r"""EMA implementation according to TensorBoard."""
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_loss_plot(trainer_log: list[dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plot loss curves (total, LM, gold_router_aux) in LlamaBoard on a single figure."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    steps_total, steps_lm, steps_aux = [], [], []
    total, lm, aux = [], [], []
    for log in trainer_log:
        step = log.get("current_steps")
        if step is None:
            continue
        # collect if present (track independent step axes)
        if log.get("loss") is not None:
            steps_total.append(step)
            total.append(log["loss"])
        if log.get("lm_loss") is not None:
            steps_lm.append(step)
            lm.append(log["lm_loss"])
        if log.get("gold_router_aux_loss") is not None:
            steps_aux.append(step)
            aux.append(log["gold_router_aux_loss"])

    # plot available series
    plotted = False
    if len(total) > 0:
        ax.plot(steps_total, total, color="#1f77b4", alpha=0.25, label="total (raw)")
        ax.plot(steps_total, smooth(total), color="#1f77b4", label="total")
        plotted = True
    if len(lm) > 0:
        ax.plot(steps_lm, lm, color="#2ca02c", alpha=0.25, label="lm (raw)")
        ax.plot(steps_lm, smooth(lm), color="#2ca02c", label="lm")
        plotted = True
    if len(aux) > 0:
        ax.plot(steps_aux, aux, color="#d62728", alpha=0.25, label="gold_router_aux (raw)")
        ax.plot(steps_aux, smooth(aux), color="#d62728", label="gold_router_aux")
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No loss metrics available", ha="center", va="center", transform=ax.transAxes)

    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses")
    return fig


def _collect_series_from_trainer_state(data: dict[str, Any], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    metrics: list[float] = []
    for i in range(len(data["log_history"])):
        if key in data["log_history"][i]:
            steps.append(data["log_history"][i]["step"])
            metrics.append(data["log_history"][i][key])
    return steps, metrics


def _plot_single_metric(save_dictionary: str, data: dict[str, Any], key: str) -> None:
    """Plot a single metric to its own image file.

    The figure is saved as metric_{key}.png under save_dictionary.
    """
    if not is_matplotlib_available():
        return

    steps, values = _collect_series_from_trainer_state(data, key)
    if len(values) == 0:
        logger.warning_rank0(f"No metric {key} to plot.")
        return

    plt.switch_backend("agg")
    plt.figure()
    plt.plot(steps, values, alpha=0.25, label=f"{key} (raw)")
    plt.plot(steps, smooth(values), label=f"{key}")
    plt.title(f"{key} of {save_dictionary}")
    plt.xlabel("step")
    plt.ylabel(key)
    plt.legend()
    figure_path = os.path.join(save_dictionary, f"metric_{key}.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)


def plot_loss(save_dictionary: str, keys: list[str] = ["loss", "lm_loss", "gold_router_aux_loss"]) -> None:
    r"""Plot only loss-related curves on one image; non-loss metrics are drawn separately.

    - Loss group (single figure): loss, lm_loss, gold_router_aux_loss, total_loss (if available)
    - Other requested keys: each plotted to its own figure as metric_{key}.png

    Falls back gracefully for missing keys.
    """
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    # Partition requested keys into loss vs non-loss
    loss_key_set = {"loss", "lm_loss", "gold_router_aux_loss", "total_loss"}
    requested_set = set(keys)
    non_loss_keys = sorted(list(requested_set - loss_key_set))
    # Always include total_loss if present in logs, even if not requested explicitly
    keys_for_loss_plot = [k for k in ["loss", "lm_loss", "gold_router_aux_loss", "total_loss"] if k in requested_set or k == "total_loss"]

    # collect loss metrics only
    series: dict[str, tuple[list[int], list[float]]] = {}
    for key in keys_for_loss_plot:
        steps, metrics = _collect_series_from_trainer_state(data, key)
        if len(metrics) > 0:
            series[key] = (steps, metrics)

    if len(series) == 0:
        logger.warning_rank0("No loss metrics available to plot.")
    else:
        # make a single figure with multiple lines (loss-only)
        plt.figure()
        color_map = {
            "loss": "#1f77b4",
            "lm_loss": "#2ca02c",
            "gold_router_aux_loss": "#d62728",
            "total_loss": "#ff7f0e",
        }
        for key, (steps, metrics) in series.items():
            color = color_map.get(key, None)
            plt.plot(steps, metrics, color=color, alpha=0.25, label=f"{key} (raw)")
            plt.plot(steps, smooth(metrics), color=color, label=f"{key}")

        plt.title(f"training losses of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_losses.png")
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)

    # Plot each non-loss metric separately (if requested)
    for key in non_loss_keys:
        _plot_single_metric(save_dictionary, data, key)
