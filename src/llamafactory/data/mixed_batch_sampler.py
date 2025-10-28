from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

import torch


class MixedBatchSampler(torch.utils.data.Sampler[List[int]]):
	r"""
	A batch sampler that mixes samples from multiple sub-datasets within a merged dataset
	according to given probabilities. It is rank-aware (DDP) by sharding indices per dataset
	using modulo on the global index space to avoid duplication across ranks.

	Two modes are supported:
	- mode='pure': each mini-batch (of size per_device_train_batch_size) is composed purely
	  of one dataset. The dataset for each batch is drawn according to mix_probs.
	- mode='mixed': each mini-batch contains a mix of samples from different datasets, with
	  counts per dataset determined by mix_probs per step (rounded per batch).
	"""

	def __init__(
		self,
		*,
		dataset,
		dataset_field: str,
		per_device_batch_size: int,
		mix_probs: Sequence[float],
		dataset_order: Sequence[str],
		mode: str = "pure",
		shuffle: bool = True,
		drop_last: bool = False,
		world_size: int = 1,
		rank: int = 0,
		generator: random.Random | None = None,
	) -> None:
		self.dataset = dataset
		self.dataset_field = dataset_field
		self.per_device_batch_size = per_device_batch_size
		self.mode = mode
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.world_size = world_size
		self.rank = rank
		self.rng = generator or random.Random()

		if abs(sum(mix_probs) - 1.0) > 1e-6:
			total = float(sum(mix_probs))
			mix_probs = [p / total for p in mix_probs]

		self.dataset_order = list(dataset_order)
		self.mix_probs = list(mix_probs)

		# Build per-dataset index pools and shard to each rank.
		name2indices: Dict[str, List[int]] = defaultdict(list)
		dataset_names = self.dataset[self.dataset_field]
		for idx, name in enumerate(dataset_names):
			# shard by global index to keep deterministic splitting and preserve dataset grouping
			if (idx % world_size) == rank:
				name2indices[str(name)].append(idx)

		# Ensure order follows user-provided dataset_order and filter missing datasets
		self.index_pools: Dict[str, List[int]] = {}
		for name in self.dataset_order:
			if name in name2indices:
				indices = name2indices[name]
				if self.shuffle:
					self.rng.shuffle(indices)
				self.index_pools[name] = indices

		# Pre-compute mixed-mode per-batch counts template (for one step)
		if self.mode == "mixed":
			base_counts = [int(round(self.per_device_batch_size * p)) for p in self.mix_probs]
			diff = self.per_device_batch_size - sum(base_counts)
			# Fix rounding by distributing the remainder to the largest frac parts deterministically
			if diff != 0:
				# build fractional parts priority (use probs to guide)
				frac_order = sorted(range(len(self.mix_probs)), key=lambda i: self.mix_probs[i], reverse=True)
				for i in range(abs(diff)):
					base_counts[frac_order[i % len(frac_order)]] += 1 if diff > 0 else -1
			# Avoid negative counts
			base_counts = [max(0, c) for c in base_counts]
			# As a last guard, ensure sum equals batch size
			s = sum(base_counts)
			if s == 0:
				base_counts[0] = self.per_device_batch_size
			elif s != self.per_device_batch_size:
				# allocate difference to the largest-prob dataset
				j = max(range(len(self.mix_probs)), key=lambda i: self.mix_probs[i])
				base_counts[j] += (self.per_device_batch_size - s)
			self.mixed_counts = base_counts

	def __iter__(self) -> Iterator[List[int]]:
		# Create working copies of pools
		pools = {k: v[:] for k, v in self.index_pools.items()}

		def can_form_pure_batch(name: str) -> bool:
			return len(pools.get(name, [])) >= self.per_device_batch_size

		def sample_pure_dataset_name() -> str | None:
			# Pick dataset per probs but also require availability
			available = [i for i, name in enumerate(self.dataset_order) if can_form_pure_batch(name)]
			if not available:
				return None
			# Renormalize probs over available
			probs = [self.mix_probs[i] for i in available]
			tot = sum(probs)
			probs = [p / tot for p in probs]
			choice = self.rng.choices(available, weights=probs, k=1)[0]
			return self.dataset_order[choice]

		while True:
			if self.mode == "pure":
				name = sample_pure_dataset_name()
				if name is None:
					break
				batch = [pools[name].pop() for _ in range(self.per_device_batch_size)]
				yield batch
			else:  # mixed mode
				batch: List[int] = []
				for name, count in zip(self.dataset_order, getattr(self, "mixed_counts", [])):
					for _ in range(count):
						if len(pools.get(name, [])) == 0:
							break
						batch.append(pools[name].pop())
				if len(batch) < self.per_device_batch_size:
					# try to fill from any pool
					for name in self.dataset_order:
						while len(batch) < self.per_device_batch_size and len(pools.get(name, [])) > 0:
							batch.append(pools[name].pop())
				if len(batch) == 0:
					break
				if len(batch) < self.per_device_batch_size:
					if self.drop_last:
						break
					else:
						# Allow smaller final batch
						yield batch
						break
				yield batch

	def __len__(self) -> int:
		# Number of batches limited by the bottleneck dataset in pure mode
		if len(self.index_pools) == 0:
			return 0
		if self.mode == "pure":
			# expected count per dataset ~ floor(len_i / bsz), weighted by probs; conservatively, use sum of floors
			return sum(len(v) // self.per_device_batch_size for v in self.index_pools.values())
		else:
			# mixed: constrained by total available samples
			total = sum(len(v) for v in self.index_pools.values())
			return total // self.per_device_batch_size if self.drop_last else math.ceil(total / self.per_device_batch_size)
