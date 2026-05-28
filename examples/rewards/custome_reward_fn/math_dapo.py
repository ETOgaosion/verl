# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from typing import Any

from verl.utils.reward_score.math_dapo import compute_score, is_correct_minerva


def compute_score_math_dapo_boxed(
    solution_str: str,
    ground_truth: str,
    data_source: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = is_correct_minerva(
        solution_str,
        ground_truth,
        answer_pattern=r"(?i)\\boxed\{\s*([^\n]+)\s*\}",
    )

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }


def compute_score_math_dapo(
    solution_str: str,
    ground_truth: str,
    data_source: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    return compute_score(solution_str, ground_truth)
