# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import copy
import heapq
from itertools import chain

import numpy as np
import torch
from torch import distributed as dist

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name


def calculate_workload(seqlen_list: torch.Tensor) -> torch.Tensor:
    """Calculate approximate computational workload for transformer attention.

    Estimates FLOPs for dense transformer blocks based on sequence length using
    the formula: FLOPs ≈ 12 * hidden_size² * seqlen + 2 * hidden_size * seqlen²

    The constants are calibrated for a 7B model (hidden_size=4096), yielding:
    workload ∝ 24576 * seqlen + seqlen²

    Args:
        seqlen_list: Sequence lengths as a tensor.

    Returns:
        torch.Tensor: Estimated workload values proportional to actual FLOPs.

    Note:
        The returned values are relative workloads, not actual FLOP counts.
        Useful for balancing computation across data parallel ranks.
    """
    return 24576 * seqlen_list + seqlen_list**2


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition items into k groups using the Karmarkar-Karp differencing method.

    Implements the Largest Differencing Method (LDM) algorithm for balanced
    multi-way number partitioning. This heuristic produces near-optimal partitions
    by iteratively combining the sets with the largest difference.

    Args:
        seqlen_list: Values to partition (typically sequence lengths or workloads).
        k_partitions: Number of partitions to create.
        equal_size: If True, each partition will have exactly len(seqlen_list) / k_partitions
            items. If False, partitions may have different sizes.

    Returns:
        list[list[int]]: List of k partitions, each containing indices into seqlen_list.

    See Also:
        https://en.wikipedia.org/wiki/Largest_differencing_method

    Note:
        When equal_size=True, len(seqlen_list) must be divisible by k_partitions.
    """

    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition items into k groups using a greedy assignment strategy.

    Assigns each item to the partition with the smallest current sum, iterating
    through items in order. Simpler but typically less optimal than Karmarkar-Karp.

    Args:
        seqlen_list: Values to partition (typically sequence lengths or workloads).
        k_partitions: Number of partitions to create.
        equal_size: If True, adds a bias to ensure equal partition sizes.
            Requires len(seqlen_list) to be divisible by k_partitions.

    Returns:
        list[list[int]]: List of k partitions, each containing indices into seqlen_list.

    Note:
        When equal_size=True, a large bias is added to encourage equal distribution
        of items before considering the actual values.
    """
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def multiway_greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition items into k groups using multiway greedy (LPT) assignment.

    Sort items by value descending, then assign each item to the partition with
    the smallest current sum. When equal_size=True, enforces a fixed number of
    items per partition.

    Args:
        seqlen_list: Values to partition (typically sequence lengths or workloads).
        k_partitions: Number of partitions to create.
        equal_size: If True, enforces len(seqlen_list) / k_partitions items per partition.

    Returns:
        list[list[int]]: List of k partitions, each containing indices into seqlen_list.
    """
    assert k_partitions > 0, "k_partitions must be > 0"
    assert len(seqlen_list) >= k_partitions, f"{len(seqlen_list)} < {k_partitions}"

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        target_size = len(seqlen_list) // k_partitions
    else:
        target_size = None

    sorted_indices = sorted(range(len(seqlen_list)), key=lambda i: seqlen_list[i], reverse=True)
    # heap item: (current_sum, current_count, partition_id, indices)
    heap = [(0, 0, p, []) for p in range(k_partitions)]
    heapq.heapify(heap)

    for idx in sorted_indices:
        temp = None
        if target_size is None:
            cur_sum, cur_count, p, items = heapq.heappop(heap)
        else:
            temp = []
            cur_sum = cur_count = p = None
            items = None
            while heap:
                cur_sum, cur_count, p, items = heapq.heappop(heap)
                if cur_count < target_size:
                    break
                temp.append((cur_sum, cur_count, p, items))
            if items is None or cur_count is None or cur_count >= target_size:
                raise RuntimeError("Failed to assign partition in multiway_greedy_partition")

        items.append(idx)
        cur_sum += seqlen_list[idx]
        cur_count += 1
        heapq.heappush(heap, (cur_sum, cur_count, p, items))

        if temp:
            for entry in temp:
                heapq.heappush(heap, entry)

    heap.sort(key=lambda x: x[2])
    partitions = [items for _, _, _, items in heap]

    if target_size is not None:
        for i, partition in enumerate(partitions):
            assert len(partition) == target_size, f"partition {i} size {len(partition)} != {target_size}"

    return partitions


def snake_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition indices using a snake (zig-zag) assignment over sorted values.

    Sort values descending, then assign in 0→k-1→0 zig-zag order so large and
    small elements interleave across partitions. When equal_size is True, each
    partition receives exactly len(seqlen_list) / k_partitions items.
    """

    assert k_partitions > 0, "k_partitions must be > 0"
    assert len(seqlen_list) >= k_partitions, f"{len(seqlen_list)} < {k_partitions}"
    if k_partitions == 1:
        return [list(range(len(seqlen_list)))]

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        target_size = len(seqlen_list) // k_partitions
    else:
        target_size = None

    # Sort by value descending, keep original indices
    sorted_pairs = sorted(enumerate(seqlen_list), key=lambda x: x[1], reverse=True)
    partitions: list[list[int]] = [[] for _ in range(k_partitions)]

    # Build snake pattern: 0->k-1->0 repeating (length 2k-2)
    pattern = list(range(k_partitions)) + list(range(k_partitions - 2, 0, -1))
    if not pattern:  # safety for k_partitions == 1 (should have returned earlier)
        pattern = [0]
    ptr = 0
    pattern_len = len(pattern)

    for orig_idx, _ in sorted_pairs:
        attempts = 0
        while True:
            p = pattern[ptr % pattern_len]
            ptr += 1
            if target_size is None or len(partitions[p]) < target_size:
                partitions[p].append(orig_idx)
                break

            attempts += 1
            if attempts > k_partitions * 2:
                raise RuntimeError("Failed to assign partition in snake_partition")

    if target_size is not None:
        for i, partition in enumerate(partitions):
            assert len(partition) == target_size, f"partition {i} size {len(partition)} != {target_size}"

    return partitions


def snake_partition_full(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition indices using a full symmetric snake pattern.

    Uses 0→k-1→k-1→0 repeating, i.e. pattern length 2k with endpoints included.
    """
    assert k_partitions > 0, "k_partitions must be > 0"
    assert len(seqlen_list) >= k_partitions, f"{len(seqlen_list)} < {k_partitions}"
    if k_partitions == 1:
        return [list(range(len(seqlen_list)))]

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        target_size = len(seqlen_list) // k_partitions
    else:
        target_size = None

    sorted_pairs = sorted(enumerate(seqlen_list), key=lambda x: x[1], reverse=True)
    partitions: list[list[int]] = [[] for _ in range(k_partitions)]

    pattern = list(range(k_partitions)) + list(range(k_partitions - 1, -1, -1))
    pattern_len = len(pattern)
    ptr = 0

    for orig_idx, _ in sorted_pairs:
        attempts = 0
        while True:
            p = pattern[ptr % pattern_len]
            ptr += 1
            if target_size is None or len(partitions[p]) < target_size:
                partitions[p].append(orig_idx)
                break
            attempts += 1
            if attempts > k_partitions * 2:
                raise RuntimeError("Failed to assign partition in snake_partition_full")

    if target_size is not None:
        for i, partition in enumerate(partitions):
            assert len(partition) == target_size, f"partition {i} size {len(partition)} != {target_size}"

    return partitions


def stratified_balanced_partition(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool,
    num_buckets: int | None = None,
) -> list[list[int]]:
    """Balanced stratified partition ensuring distribution similarity across partitions.

    This approach ensures each partition has SIMILAR workload distribution (not just total):
    1. Divides sequences into equal-frequency workload tiers (buckets)
    2. From each bucket, distributes items to partitions using workload-aware assignment
    3. Each partition receives proportional representation from ALL tiers

    Key insight: By ensuring each partition gets items from every workload tier,
    we guarantee similar workload DISTRIBUTIONS, not just similar totals. This means:
    - Similar kernel behavior across DP ranks (similar L² effects)
    - Better micro-batch alignment when partitioned later
    - More predictable execution times

    Note: This function does NOT sort items within partitions. The micro-batch
    ordering should be handled separately by rearrange_micro_batches() at runtime.

    Args:
        seqlen_list: Workload values to partition.
        k_partitions: Number of DP partitions.
        equal_size: Enforce equal number of items per partition.
        num_buckets: Number of workload tiers. Defaults to k_partitions.

    Returns:
        list[list[int]]: List of k partitions, each containing indices into seqlen_list.
        Items within each partition maintain their relative order from bucket assignment.
    """
    n = len(seqlen_list)
    assert n >= k_partitions, f"number of items:[{n}] < k_partitions:[{k_partitions}]"

    # ---------- Equal size constraint preprocessing ----------
    if equal_size:
        assert n % k_partitions == 0, f"{n} % {k_partitions} != 0"
        target_size = n // k_partitions
    else:
        target_size = None

    # ---------- Determine number of buckets ----------
    if num_buckets is None:
        num_buckets = k_partitions
    # Ensure we have at least k items per bucket for meaningful distribution
    num_buckets = max(1, min(num_buckets, n // k_partitions))

    # ---------- Phase 1: Sort by workload in descending order ----------
    sorted_indices = sorted(range(n), key=lambda i: seqlen_list[i], reverse=True)

    # ---------- Phase 2: Equal-frequency bucketing (split by sorted position) ----------
    bucket_size = n // num_buckets
    remainder = n % num_buckets
    buckets: list[list[int]] = []
    ptr = 0
    for b in range(num_buckets):
        cur_size = bucket_size + (1 if b < remainder else 0)
        buckets.append(sorted_indices[ptr:ptr + cur_size])
        ptr += cur_size

    # ---------- Initialize global state ----------
    partition_workloads = [0] * k_partitions
    partitions: list[list[int]] = [[] for _ in range(k_partitions)]
    remainder_start = 0

    # ---------- Phase 3: Bucket-by-bucket allocation ----------
    for bucket_idx, bucket in enumerate(buckets):
        # --- Calculate quotas (with rotation for fairness) ---
        items_per_partition = len(bucket) // k_partitions
        bucket_remainder = len(bucket) % k_partitions

        # Generate list of partitions that get extra quota this round (rotation)
        extra_partitions = [
            (remainder_start + i) % k_partitions for i in range(bucket_remainder)
        ]
        remainder_start = (remainder_start + bucket_remainder) % k_partitions

        # Quota array for each partition
        quotas = [items_per_partition] * k_partitions
        for p in extra_partitions:
            quotas[p] += 1

        # --- Initialize heap: only include partitions with remaining quota ---
        heap = []
        for p in range(k_partitions):
            if quotas[p] > 0:
                heapq.heappush(heap, (partition_workloads[p], 0, p))

        # --- Allocation within bucket ---
        for item_idx, idx in enumerate(bucket):
            if not heap:
               raise RuntimeError(
                    f"Bucket {bucket_idx}: No partition available to assign item {item_idx} "
                    f"(workload={seqlen_list[idx]})."
                )
            # Pop partition with smallest current global accumulated workload
            cur_workload, cur_count, p = heapq.heappop(heap)

            # Assign element
            partitions[p].append(idx)
            partition_workloads[p] += seqlen_list[idx]
            cur_count += 1

            # If partition still has remaining quota in this bucket, re-push to heap; otherwise it's full, don't re-push
            if cur_count < quotas[p]:
                heapq.heappush(heap, (partition_workloads[p], cur_count, p))
            # Quota exhausted -> don't re-push to heap

    # ---------- Phase 4: Validate equal size (only when equal_size=True) ----------
    # Note: Due to the quota system with rotation in Phase 3, all partitions should
    # already have exactly target_size elements. This phase only validates that invariant.
    if equal_size:
        # Validate that Phase 3 produced correctly sized partitions
        for p in range(k_partitions):
            actual_size = len(partitions[p])
            if actual_size != target_size:
                raise RuntimeError(
                    f"Phase 3 allocation error: Partition {p} has {actual_size} elements, "
                    f"expected {target_size}. This indicates a bug in the quota allocation logic."
                )

    # ---------- Phase 5: Comprehensive result validation ----------
    total_assigned = sum(len(p) for p in partitions)
    if total_assigned != n:
        raise RuntimeError(
            f"stratified_balanced_partition: Not all items were assigned. "
            f"Expected {n}, got {total_assigned}"
        )

    # Verify no duplicate indices
    all_indices = set()
    for partition in partitions:
        for idx in partition:
            if idx in all_indices:
                raise RuntimeError(f"stratified_balanced_partition: Duplicate index {idx} found")
            all_indices.add(idx)

    if len(all_indices) != n:
        raise RuntimeError(
            f"stratified_balanced_partition: Index set size mismatch. "
            f"Expected {n}, got {len(all_indices)}"
        )

    return partitions


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool,
    method: str = "balanced",
    num_buckets: int | None = None,
) -> list[list[int]]:
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced.

    This is the unified entry point for all partitioning operations, supporting multiple algorithms.
    Use this for both DP partitioning and micro-batch partitioning.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.
        method (str): Partitioning algorithm to use. Can be overridden by VERL_PARTITION_METHOD env var.
            Options:
            - "snake_full": Full symmetric snake pattern (default, good distribution)
            - "multiway": Multiway greedy (LPT). Best for workload balance
            - "balanced": Stratified balanced. Best overall (balance + distribution + sync)
            - "greedy": Simple greedy assignment
            - "karmarkar_karp": Karmarkar-Karp differencing method (best balance)
        num_buckets (int | None): Number of workload tiers for "balanced" method. Defaults to k_partitions.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
        ValueError: If an unknown method is specified.
    """
    import os
    
    # Allow environment variable to override method parameter
    env_method = os.environ.get('VERL_PARTITION_METHOD')
    if env_method:
        method = env_method
    
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    # Route to appropriate partitioning algorithm
    if method == "snake_full":
        partitions = snake_partition_full(seqlen_list, k_partitions, equal_size=equal_size)
    elif method == "multiway":
        partitions = multiway_greedy_partition(seqlen_list, k_partitions, equal_size=equal_size)
    elif method == "balanced":
        partitions = stratified_balanced_partition(
            seqlen_list, k_partitions, equal_size=equal_size, num_buckets=num_buckets
        )
    elif method == "greedy":
        partitions = greedy_partition(seqlen_list, k_partitions, equal_size=equal_size)
    elif method == "karmarkar_karp":
        partitions = karmarkar_karp(seqlen_list, k_partitions, equal_size=equal_size)
    else:
        raise ValueError(
            f"Unknown partition method: {method}. "
            f"Available methods: snake_full, multiway, balanced, greedy, karmarkar_karp"
        )

    return _check_and_sort_partitions(partitions)

def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix):
    """
    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    This function computes comprehensive metrics across three dimensions:
    1. Memory dimension: sum(L) - total sequence length per partition
    2. Compute dimension: sum(L²) - attention computational workload per partition
    3. Distribution similarity: variance within each partition

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
        prefix (str): A prefix to be added to each metric key in the returned dictionary.

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance, including:
            - Memory imbalance ratio: max(sum_L) / mean(sum_L) - 1
            - Compute imbalance ratio: max(sum_L²) / mean(sum_L²) - 1
            - Variance discrepancy: max(var) / mean(var) - 1 across partitions
    """
    # Get the number of partitions
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    
    # ========== Before partitioning (naive sequential split) ==========
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Iterate over each batch of sequence lengths
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    # ========== After partitioning (balanced split) ==========
    balanced_sum_seqlen_list = []  # Memory dimension: sum(L)
    balanced_sum_seqlen_sq_list = []  # Compute dimension: sum(L²)
    balanced_variance_list = []  # Distribution similarity: var(L) within partition
    
    for partition in partitions:
        partition_seqlens = [seqlen_list[i] for i in partition]
        
        # Memory metric: sum of L
        sum_L = sum(partition_seqlens)
        balanced_sum_seqlen_list.append(sum_L)
        
        # Compute metric: sum of L² (attention workload)
        sum_L_sq = sum(L**2 for L in partition_seqlens)
        balanced_sum_seqlen_sq_list.append(sum_L_sq)
        
        # Distribution metric: variance within partition
        if len(partition_seqlens) > 1:
            variance = float(np.var(partition_seqlens, ddof=1))  # Sample variance
        else:
            variance = 0.0
        balanced_variance_list.append(variance)
    
    # Extract statistics for memory dimension (sum L)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)
    mean_sum_seqlen_balanced = float(np.mean(balanced_sum_seqlen_list))
    
    # Extract statistics for compute dimension (sum L²)
    min_sum_seqlen_sq_balanced = min(balanced_sum_seqlen_sq_list)
    max_sum_seqlen_sq_balanced = max(balanced_sum_seqlen_sq_list)
    mean_sum_seqlen_sq_balanced = float(np.mean(balanced_sum_seqlen_sq_list))
    
    # ========== Key Evaluation Metrics ==========
    
    # 1. Memory Imbalance Ratio: max(sum_L) / mean(sum_L) - 1
    # Physical meaning: Risk of OOM on the most loaded GPU
    # Target: < 0.05 (5% imbalance) is excellent
    memory_imbalance_ratio = (
        (max_sum_seqlen_balanced / mean_sum_seqlen_balanced - 1) 
        if mean_sum_seqlen_balanced > 0 else 0.0
    )
    
    # 2. Compute Imbalance Ratio: max(sum_L²) / mean(sum_L²) - 1
    # Physical meaning: DP synchronization bottleneck (木桶效应)
    # Target: < 0.10 (10% imbalance) is excellent
    compute_imbalance_ratio = (
        (max_sum_seqlen_sq_balanced / mean_sum_seqlen_sq_balanced - 1) 
        if mean_sum_seqlen_sq_balanced > 0 else 0.0
    )
    
    # 3. Variance Discrepancy: max(var) / mean(var) - 1
    # Physical meaning: Proves "same distribution" property vs. Karmarkar-Karp
    # Target: < 0.20 (20% variance difference) proves distribution similarity
    mean_variance = float(np.mean(balanced_variance_list))
    max_variance = max(balanced_variance_list)
    if mean_variance > 1e-6:  # Avoid division by near-zero
        variance_discrepancy = (max_variance / mean_variance - 1)
    else:
        variance_discrepancy = 0.0
    
    # 4. Sequence Length Range Analysis (for OOM risk assessment)
    # Extract min/max sequence length in each partition
    partition_max_seqlens = []
    partition_min_seqlens = []
    partition_seqlen_ranges = []
    
    for partition in partitions:
        partition_seqlens = [seqlen_list[i] for i in partition]
        if partition_seqlens:
            pmax = max(partition_seqlens)
            pmin = min(partition_seqlens)
            partition_max_seqlens.append(pmax)
            partition_min_seqlens.append(pmin)
            partition_seqlen_ranges.append(pmax - pmin)
    
    # Range discrepancy: indicates if some partitions have much wider distributions
    max_range = max(partition_seqlen_ranges) if partition_seqlen_ranges else 0
    mean_range = float(np.mean(partition_seqlen_ranges)) if partition_seqlen_ranges else 0
    range_discrepancy = ((max_range / mean_range - 1) if mean_range > 0 else 0.0)
    
    # Global sequence length statistics
    # Convert to numpy array if it's a tensor to avoid type errors
    if isinstance(seqlen_list, torch.Tensor):
        seqlen_array = seqlen_list.cpu().numpy()
    else:
        seqlen_array = np.asarray(seqlen_list)
    
    global_max_seqlen = float(np.max(seqlen_array))
    global_min_seqlen = float(np.min(seqlen_array))
    global_mean_seqlen = float(np.mean(seqlen_array))
    
    # 5. Load Balance Quality Score (综合评分 0-100)
    # Combined score: penalize memory, compute, and variance imbalances
    # Score = 100 * (1 - weighted_avg_imbalance)
    # Weights: memory=0.3, compute=0.5, variance=0.2 (compute is most critical for DP sync)
    weighted_imbalance = (
        0.3 * min(memory_imbalance_ratio, 1.0) +  # Cap at 100% imbalance
        0.5 * min(compute_imbalance_ratio, 1.0) +
        0.2 * min(variance_discrepancy, 1.0)
    )
    balance_quality_score = max(0.0, 100.0 * (1.0 - weighted_imbalance))

    # Helper to convert tensor/numpy values to Python native types for logging
    def to_python(val):
        if hasattr(val, 'item'):
            return val.item()  # PyTorch tensor or numpy scalar
        return float(val) if isinstance(val, (np.number, np.ndarray)) else val

    return {
        # ===== Original metrics (before balancing) =====
        f"{prefix}/min": to_python(min_sum_seqlen),
        f"{prefix}/max": to_python(max_sum_seqlen),
        f"{prefix}/minmax_diff": to_python(max_sum_seqlen - min_sum_seqlen),
        
        # ===== Memory dimension (sum of L) =====
        f"{prefix}/balanced_min": to_python(min_sum_seqlen_balanced),
        f"{prefix}/balanced_max": to_python(max_sum_seqlen_balanced),
        f"{prefix}/mean": to_python(mean_sum_seqlen_balanced),
        f"{prefix}/memory_imbalance_ratio": to_python(memory_imbalance_ratio),
        
        # ===== Compute dimension (sum of L²) =====
        f"{prefix}/compute_min": to_python(min_sum_seqlen_sq_balanced),
        f"{prefix}/compute_max": to_python(max_sum_seqlen_sq_balanced),
        f"{prefix}/compute_mean": to_python(mean_sum_seqlen_sq_balanced),
        f"{prefix}/compute_imbalance_ratio": to_python(compute_imbalance_ratio),
        
        # ===== Distribution similarity (variance) =====
        f"{prefix}/variance_mean": to_python(mean_variance),
        f"{prefix}/variance_max": to_python(max_variance),
        f"{prefix}/variance_discrepancy": to_python(variance_discrepancy),
        
        # ===== Sequence length range analysis =====
        f"{prefix}/seqlen_global_max": to_python(global_max_seqlen),
        f"{prefix}/seqlen_global_min": to_python(global_min_seqlen),
        f"{prefix}/seqlen_global_mean": to_python(global_mean_seqlen),
        f"{prefix}/partition_max_range": to_python(max_range),
        f"{prefix}/partition_mean_range": to_python(mean_range),
        f"{prefix}/range_discrepancy": to_python(range_discrepancy),
        
        # ===== Overall quality score =====
        f"{prefix}/balance_quality_score": to_python(balance_quality_score),
    }


def analyze_cross_dp_micro_batch_balance(
    dp_partitions: list[list[int]],
    workload_list: list[float],
    micro_batch_size: int | None = None,
    prefix: str = "cross_dp_mb",
) -> dict:
    """
    Analyze workload balance across DP ranks for same-index micro-batches.
    
    This function simulates how micro-batches would be formed on each DP rank
    and checks if same-index micro-batches have similar workloads across ranks.
    This is critical for reducing synchronization overhead in data parallel training.
    
    Args:
        dp_partitions: List of DP partitions, each containing indices into workload_list.
        workload_list: Workload value for each sample.
        micro_batch_size: Fixed size for micro-batches. If None, analyzes variable-sized batches.
        prefix: Prefix for metric keys in returned dictionary.
        
    Returns:
        dict: Metrics including:
            - {prefix}/num_micro_batches: Number of micro-batches per DP rank
            - {prefix}/mb{i}_std: Standard deviation of workload for micro-batch i across DP ranks
            - {prefix}/mb{i}_cv: Coefficient of variation (std/mean) for micro-batch i
            - {prefix}/avg_std: Average std across all micro-batch indices
            - {prefix}/avg_cv: Average coefficient of variation
            - {prefix}/max_std: Maximum std (worst case synchronization delay)
            - {prefix}/sync_efficiency: 1 - (avg_cv), higher is better (closer to 1.0)
            
    Example:
        >>> # 8 DP ranks, each with 16 samples
        >>> dp_partitions = [list(range(i*16, (i+1)*16)) for i in range(8)]
        >>> workloads = [random.randint(100, 500) for _ in range(128)]
        >>> metrics = analyze_cross_dp_micro_batch_balance(dp_partitions, workloads, micro_batch_size=4)
        >>> print(f"Sync efficiency: {metrics['cross_dp_mb/sync_efficiency']:.2%}")
    """
    import numpy as np
    
    num_dp_ranks = len(dp_partitions)
    metrics = {}
    
    # Simulate micro-batch formation on each DP rank
    dp_micro_batches = []  # [dp_rank][mb_index] -> workload
    min_num_mbs = float('inf')
    max_num_mbs = 0
    
    for rank_idx, partition in enumerate(dp_partitions):
        rank_workloads = [workload_list[i] for i in partition]
        
        if micro_batch_size is not None:
            # Fixed-size micro-batches
            num_mbs = len(rank_workloads) // micro_batch_size
            mb_workloads = []
            for mb_idx in range(num_mbs):
                start = mb_idx * micro_batch_size
                end = start + micro_batch_size
                mb_total = sum(rank_workloads[start:end])
                mb_workloads.append(mb_total)
        else:
            # Variable-sized: each sample is its own micro-batch (for analysis)
            mb_workloads = rank_workloads
        
        dp_micro_batches.append(mb_workloads)
        min_num_mbs = min(min_num_mbs, len(mb_workloads))
        max_num_mbs = max(max_num_mbs, len(mb_workloads))
    
    metrics[f"{prefix}/num_micro_batches"] = min_num_mbs
    metrics[f"{prefix}/num_micro_batches_max"] = max_num_mbs
    
    if min_num_mbs == 0:
        metrics[f"{prefix}/warning"] = "No micro-batches formed"
        return metrics
    
    # Analyze each micro-batch index across DP ranks
    mb_stds = []
    mb_cvs = []  # Coefficient of variation: std/mean
    
    for mb_idx in range(min_num_mbs):
        # Collect workload for this micro-batch index across all DP ranks
        mb_workloads_across_ranks = [
            dp_micro_batches[rank_idx][mb_idx] 
            for rank_idx in range(num_dp_ranks)
            if mb_idx < len(dp_micro_batches[rank_idx])
        ]
        
        if len(mb_workloads_across_ranks) < 2:
            continue
            
        mean_workload = np.mean(mb_workloads_across_ranks)
        std_workload = np.std(mb_workloads_across_ranks)
        cv = std_workload / mean_workload if mean_workload > 0 else 0
        
        mb_stds.append(std_workload)
        mb_cvs.append(cv)
        
        # Store per-micro-batch metrics (limit to first 10 for brevity)
        if mb_idx < 10:
            metrics[f"{prefix}/mb{mb_idx}_mean"] = float(mean_workload)
            metrics[f"{prefix}/mb{mb_idx}_std"] = float(std_workload)
            metrics[f"{prefix}/mb{mb_idx}_cv"] = float(cv)
            metrics[f"{prefix}/mb{mb_idx}_min"] = float(min(mb_workloads_across_ranks))
            metrics[f"{prefix}/mb{mb_idx}_max"] = float(max(mb_workloads_across_ranks))
    
    # Aggregate metrics
    if mb_stds:
        metrics[f"{prefix}/avg_std"] = float(np.mean(mb_stds))
        metrics[f"{prefix}/max_std"] = float(np.max(mb_stds))
        metrics[f"{prefix}/min_std"] = float(np.min(mb_stds))
    
    if mb_cvs:
        avg_cv = float(np.mean(mb_cvs))
        metrics[f"{prefix}/avg_cv"] = avg_cv
        metrics[f"{prefix}/max_cv"] = float(np.max(mb_cvs))
        metrics[f"{prefix}/min_cv"] = float(np.min(mb_cvs))
        # Sync efficiency: higher is better, 1.0 means perfect balance
        metrics[f"{prefix}/sync_efficiency"] = max(0.0, 1.0 - avg_cv)
    
    return metrics


def ceildiv(a: int, b: int) -> int:
    """Compute ceiling division of a by b.

    Returns the smallest integer greater than or equal to a/b.
    Uses the identity: ceil(a/b) = floor((a + b - 1) / b) = -(-a // b)

    Args:
        a: Dividend (numerator).
        b: Divisor (denominator), must be non-zero.

    Returns:
        int: Ceiling of a divided by b.

    Example:
        >>> ceildiv(7, 3)  # ceil(7/3) = ceil(2.33) = 3
        3
        >>> ceildiv(6, 3)  # ceil(6/3) = ceil(2.0) = 2
        2
    """
    return -(a // -b)


def roundup_divisible(a: int, b: int) -> int:
    """Round up a to the nearest multiple of b.

    Returns the smallest multiple of b that is >= a.

    Args:
        a: Value to round up.
        b: Divisor to round to (must be positive).

    Returns:
        int: Smallest multiple of b that is >= a.

    Example:
        >>> roundup_divisible(7, 4)  # nearest multiple of 4 >= 7 is 8
        8
        >>> roundup_divisible(8, 4)  # 8 is already a multiple of 4
        8
    """
    return ((a + b - 1) // b) * b


def rearrange_micro_batches(
    batch,
    max_token_len,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Split a batch into micro-batches by total token count, with optional DP sync and padding.

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): max sum of attention_mask per micro-batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
        use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches.
    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
        
    Note:
        The partitioning algorithm is controlled by get_seqlen_balanced_partitions's default method parameter.
        To change the algorithm globally, modify the default value in get_seqlen_balanced_partitions.
    """
    # this is per local micro_bsz
    input_ids = batch["input_ids"]
    if input_ids.is_nested:
        seq_len_effective: torch.Tensor = input_ids.offsets().diff()
        max_seq_len = max(seq_len_effective)
    else:
        max_seq_len = batch["attention_mask"].shape[-1]
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)

    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    )
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    assert num_micro_batches <= len(seq_len_effective)

    # note that seq_len_effective is a GPU tensor. We need to make it a list to avoid D2H!
    workloads = calculate_workload(seq_len_effective).cpu().tolist()
    
    # Use get_seqlen_balanced_partitions with its default method
    micro_bsz_idx = get_seqlen_balanced_partitions(
        workloads, num_micro_batches, equal_size=False
    )

    if use_dynamic_bsz_balance:
        # Use the sum of squared sequence lengths to approximate attention computation workload
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(workloads[idx] for idx in partition),
                partition[0] if partition else 0,
            ),
            reverse=True,
        )
        # Place smaller micro-batches at both ends to reduce the bubbles exposed during the warm-up and cool-down.
        micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]

    micro_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = tu.index_select_tensor_dict(batch, partition)
        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map


def prepare_dynamic_batch(
    data: DataProto,
    max_token_len: int,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
) -> tuple[list[DataProto], list[list[int]]]:
    """
    Prepare a batch for dynamic batching.

    Args:
        data (DataProto): The input data.
        max_token_len (int): The maximum token length for dynamic batching.

    Returns:
        Tuple[List[DataProto], List[List[int]]]: A tuple containing a list of DataProto objects
        and a list of index lists.
        
    Note:
        The partitioning algorithm is controlled by get_seqlen_balanced_partitions's default method parameter.
    """
    batch, batch_idx_list = rearrange_micro_batches(
        data.batch,
        max_token_len=max_token_len,
        dp_group=dp_group,
        num_batches_divided_by=num_batches_divided_by,
        same_micro_num_in_dp=same_micro_num_in_dp,
        min_num_micro_batch=min_num_micro_batch,
        use_dynamic_bsz_balance=use_dynamic_bsz_balance,
    )
    micro_batches = []
    for i, batch_idx in enumerate(batch_idx_list):
        tensors = dict(batch[i])
        non_tensors = {key: value[batch_idx] for key, value in data.non_tensor_batch.items()}
        meta_info = copy.deepcopy(data.meta_info)
        micro_batches.append(DataProto.from_dict(tensors, non_tensors, meta_info=meta_info))

    return micro_batches, batch_idx_list


def restore_dynamic_batch(data: torch.Tensor, batch_idx_list: list[list[int]]) -> torch.Tensor:
    """
    Restore a batch from dynamic batching.

    Args:
        data (torch.Tensor): The input data.
        batch_idx_list (List[List[int]]): The list of index lists.

    Returns:
        torch.Tensor: The restored data.
    """
    indices = list(chain.from_iterable(batch_idx_list))
    batch_size = data.shape[0]
    assert len(indices) == batch_size, f"{len(indices)} vs. {batch_size}"
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)

    if data.is_nested:
        data_lst = data.unbind()
        tensors = [data_lst[i] for i in revert_indices]
        reverted_data = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
    else:
        reverted_data = data[revert_indices]

    return reverted_data
