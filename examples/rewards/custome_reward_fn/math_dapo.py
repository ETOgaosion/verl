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
