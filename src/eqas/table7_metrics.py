from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

DEFAULT_HUMAN_DIMENSIONS = (
    "correctness",
    "clarity",
    "sufficiency",
    "helpfulness",
)

def mean_latency_seconds(
    rows: Sequence[Mapping],
    latency_key: str = "latency_seconds",
) -> float:
    """
    Mean end-to-end response latency.

    L = (1/N) * sum_i (t_i_end - t_i_start)

    Each row stores the already measured difference in `latency_seconds`.
    """
    if not rows:
        return 0.0

    values = [float(row[latency_key]) for row in rows]
    if any(value < 0 for value in values):
        raise ValueError("latency_seconds cannot be negative")

    return sum(values) / len(values)


def fallback_rate(
    rows: Sequence[Mapping],
    fallback_key: str = "used_fallback",
) -> float:
    """
    Percentage of queries redirected away from the primary direct-QA pathway.

    FR = (N_fallback / N) * 100
    """
    if not rows:
        return 0.0

    n_fallback = sum(bool(row[fallback_key]) for row in rows)
    return 100.0 * n_fallback / len(rows)


def overall_human_evaluation(
    rows: Sequence[Mapping],
    dimensions: Sequence[str] = DEFAULT_HUMAN_DIMENSIONS,
    response_key: str = "response_id",
    evaluator_key: str = "evaluator_id",
    expected_evaluators: int | None = 3,
) -> float:
    """
    Overall human-evaluation score across responses, evaluators, and dimensions.

    H = (1 / (N * R * D)) * sum_i sum_r sum_d s_ird

    Input convention:
    one row = one evaluator's ratings for one response.
    """
    if not rows:
        return 0.0
    if not dimensions:
        raise ValueError("At least one evaluation dimension is required")

    if expected_evaluators is not None:
        counts = Counter(str(row[response_key]) for row in rows)
        invalid = {
            response_id: count
            for response_id, count in counts.items()
            if count != expected_evaluators
        }
        if invalid:
            sample = list(invalid.items())[:5]
            raise ValueError(
                "Each response must have exactly "
                f"{expected_evaluators} evaluator rows; examples: {sample}"
            )

        pairs = [
            (str(row[response_key]), str(row[evaluator_key]))
            for row in rows
        ]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate response_id/evaluator_id pair detected")

    scores = []
    for row in rows:
        for dimension in dimensions:
            score = float(row[dimension])
            if not 1.0 <= score <= 5.0:
                raise ValueError(
                    f"{dimension} score must be in [1,5], got {score}"
                )
            scores.append(score)

    return sum(scores) / len(scores)


def validate_same_query_ids(
    output_rows: Sequence[Mapping],
    annotation_rows: Sequence[Mapping],
    output_id_key: str = "id",
    annotation_id_key: str = "response_id",
) -> None:
    """Require metric annotations and system outputs to refer to the same responses."""
    output_ids = {str(row[output_id_key]) for row in output_rows}
    annotation_ids = {str(row[annotation_id_key]) for row in annotation_rows}
    if output_ids != annotation_ids:
        missing_ann = sorted(output_ids - annotation_ids)[:5]
        missing_out = sorted(annotation_ids - output_ids)[:5]
        raise ValueError(
            "Output/annotation ID mismatch. "
            f"Missing annotations: {missing_ann}; missing outputs: {missing_out}"
        )
