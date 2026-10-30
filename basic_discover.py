"""
Discover and validate category-specific candidate base formulas.

Candidates are generated only from training prescriptions by depth-first,
one-herb-at-a-time expansion. Held-out prescriptions are used afterwards for
reproduction and enrichment estimates.

Validation reports:
1. Wilson 95% confidence intervals for reproduction rates.
2. Stratified bootstrap 95% confidence intervals for enrichment scores.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from category import (
    CATEGORIES,
    DEFAULT_DATA_DIR,
    DEFAULT_SCORE_DIR,
    EPS,
    load_prescriptions,
    load_scores,
)


ROOT = Path(__file__).resolve().parent

DEFAULT_OUTPUT = (
    ROOT / "base_formula_outputs/base_formula_all_categories.csv"
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--score-dir",
        type=Path,
        default=DEFAULT_SCORE_DIR
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR
    )

    parser.add_argument(
        "--validation-top-k",
        type=int,
        default=20
    )

    parser.add_argument(
        "--retain-top-k",
        type=int,
        default=100,
        help="Retain this many ranked formulas per category in the final output (0 keeps all).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=400
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=5
    )

    parser.add_argument(
        "--bootstrap-times",
        type=int,
        default=2000
    )

    return parser.parse_args()


# ============================================================
# Wilson confidence interval
# ============================================================

def wilson_ci(
    success: int,
    total: int,
) -> tuple[float, float]:

    if total == 0:
        return 0.0, 0.0

    z = 1.96

    p = success / total

    denominator = (
        1 +
        z**2 / total
    )

    center = (
        p +
        z**2 / (2 * total)
    ) / denominator

    margin = (
        z *
        math.sqrt(
            p * (1 - p) / total
            +
            z**2 / (4 * total**2)
        )
        /
        denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


# ============================================================
# Candidate discovery
# ============================================================

def discover_category(
    category: str,
    score_dir: Path,
    data_dir: Path,
    candidate_top_k: int,
    min_size: int,
    max_size: int,
    min_count: int,
) -> pd.DataFrame:

    scores = load_scores(
        score_dir,
        category
    )

    if candidate_top_k > 0:
        scores = scores.head(candidate_top_k)

    herbs = scores["herb_name"].tolist()

    rank_score = {
        herb: 1.0 - (rank - 1) / max(len(herbs), 1)
        for herb, rank in zip(
            scores["herb_name"],
            scores["model_rank"]
        )
    }

    prescriptions = load_prescriptions(
        data_dir,
        category,
        "train"
    )

    herb_tids = {
        herb: {
            index
            for index, prescription in enumerate(prescriptions)
            if herb in prescription
        }
        for herb in herbs
    }

    candidates = []


    def extend(
        prefix: tuple[str, ...],
        tids: set[int] | None,
        start: int,
    ):

        for index in range(start, len(herbs)):

            herb = herbs[index]

            next_tids = (
                herb_tids[herb]
                if tids is None
                else tids & herb_tids[herb]
            )

            if len(next_tids) < min_count:
                continue

            formula = prefix + (herb,)


            if len(formula) >= min_size:

                candidates.append(
                    (
                        formula,
                        len(next_tids)
                    )
                )


            if len(formula) < max_size:

                extend(
                    formula,
                    next_tids,
                    index + 1
                )


    extend(
        tuple(),
        None,
        0
    )


    if not candidates:
        raise ValueError(
            f"No formulas found for {category}"
        )


    max_count = max(
        count
        for _, count in candidates
    )


    rows = []


    for formula, count in candidates:

        mean_rank = sum(
            rank_score[herb]
            for herb in formula
        ) / len(formula)


        relative_support = (
            math.log1p(count)
            /
            math.log1p(max_count)
        )


        composite = math.sqrt(
            mean_rank *
            relative_support
        )


        rows.append(
            {
                "category": category,
                "category_cn": CATEGORIES[category],
                "base_formula": "|".join(formula),
                "formula_size": len(formula),
                "formula_score": composite,
                "train_support": (
                    count / len(prescriptions)
                ),
                "_train_count": count,
            }
        )


    result = pd.DataFrame(rows)


    result = result.sort_values(
        [
            "formula_score",
            "_train_count",
            "formula_size",
            "base_formula",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    )


    result = result.reset_index(
        drop=True
    )


    result["rank"] = (
        range(
            1,
            len(result)+1
        )
    )


    return result


# ============================================================
# Validation utilities
# ============================================================

def build_tidsets(
    prescriptions: list[set[str]]
) -> dict[str, set[int]]:

    tidsets = {}

    for index, prescription in enumerate(prescriptions):

        for herb in prescription:

            tidsets.setdefault(
                herb,
                set()
            ).add(index)

    return tidsets


def count_from_tidsets(
    formula: frozenset[str],
    tidsets: dict[str, set[int]]
) -> int:

    herb_sets = [
        tidsets.get(
            herb,
            set()
        )
        for herb in formula
    ]

    if not herb_sets or any(
        not indices
        for indices in herb_sets
    ):
        return 0


    return len(
        set.intersection(
            *herb_sets
        )
    )


def bootstrap_enrichment_ci(
    formula: frozenset[str],
    target_category: str,
    tests: dict[str, list[set[str]]],
    n_bootstrap: int = 2000,
) -> tuple[float, float]:

    categories = list(
        tests.keys()
    )

    category_sizes = {category: len(tests[category]) for category in categories}
    category_counts = {
        category: np.random.binomial(
            category_sizes[category],
            sum(formula.issubset(record) for record in tests[category]) / category_sizes[category],
            size=n_bootstrap,
        )
        for category in categories
    }
    target_support = category_counts[target_category] / category_sizes[target_category]
    overall_support = sum(category_counts.values()) / sum(category_sizes.values())
    values = (target_support + EPS) / (overall_support + EPS)


    return (
        np.percentile(values, 2.5),
        np.percentile(values, 97.5),
    )

# ============================================================
# Validation
# ============================================================

def validate(
    formulas: pd.DataFrame,
    bootstrap_formulas: pd.DataFrame,
    data_dir: Path,
    bootstrap_times: int,
) -> pd.DataFrame:


    tests = {
        category:
        load_prescriptions(
            data_dir,
            category,
            "test"
        )
        for category in CATEGORIES
    }


    test_tidsets = {
        category:
        build_tidsets(records)
        for category, records
        in tests.items()
    }


    n_all = sum(
        len(records)
        for records in tests.values()
    )


    # --------------------------------------------------------
    # Store top-K formulas requiring bootstrap validation
    # --------------------------------------------------------

    bootstrap_keys = set(
        zip(
            bootstrap_formulas["category"],
            bootstrap_formulas["base_formula"]
        )
    )


    output = []


    for row in formulas.itertuples(
        index=False
    ):

        formula = frozenset(
            str(row.base_formula).split("|")
        )


        # -------------------------------
        # Held-out reproduction
        # -------------------------------

        target_count = count_from_tidsets(
            formula,
            test_tidsets[row.category]
        )


        target_total = len(
            tests[row.category]
        )


        target_support = (
            target_count
            /
            target_total
        )


        support_ci_lower, support_ci_upper = wilson_ci(
            target_count,
            target_total
        )


        # -------------------------------
        # Overall support
        # -------------------------------

        overall_count = sum(
            count_from_tidsets(
                formula,
                tidsets
            )
            for tidsets
            in test_tidsets.values()
        )


        overall_support = (
            overall_count
            /
            n_all
        )


        enrichment = (
            target_support + EPS
        ) / (
            overall_support + EPS
        )


        # -------------------------------
        # Bootstrap enrichment CI
        # Only top-K formulas
        # -------------------------------

        if (
            row.category,
            row.base_formula
        ) in bootstrap_keys:


            enrichment_ci_lower, enrichment_ci_upper = (
                bootstrap_enrichment_ci(
                    formula,
                    row.category,
                    tests,
                    bootstrap_times,
                )
            )

            bootstrap_validated = True


        else:

            enrichment_ci_lower = 0.0

            enrichment_ci_upper = 0.0

            bootstrap_validated = False



        output.append(
            {
                "category": row.category,

                "category_cn": row.category_cn,

                "rank": row.rank,

                "base_formula": row.base_formula,

                "formula_size": row.formula_size,

                "formula_score": row.formula_score,

                "train_support": row.train_support,


                "test_count": target_count,

                "test_total": target_total,

                "test_support": target_support,

                "test_support_ci_lower":
                    support_ci_lower,

                "test_support_ci_upper":
                    support_ci_upper,


                "formula_enrichment":
                    enrichment,


                "formula_enrichment_ci_lower":
                    enrichment_ci_lower,

                "formula_enrichment_ci_upper":
                    enrichment_ci_upper,


                "bootstrap_validated":
                    bootstrap_validated,
            }
        )


    return pd.DataFrame(output)



# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    np.random.seed(42)


    if not (
        2 <= args.min_size <= args.max_size
    ):
        raise ValueError(
            "Require 2 <= min-size <= max-size"
        )


    discovered = []


    # --------------------------------------------------------
    # Candidate discovery
    # --------------------------------------------------------

    for category in CATEGORIES:

        frame = discover_category(
            category,
            args.score_dir,
            args.data_dir,
            args.candidate_top_k,
            args.min_size,
            args.max_size,
            args.min_count,
        )


        discovered.append(frame)


        print(
            f"{category}: retained {len(frame)} formulas"
        )


    # All candidate formulas
    formulas = pd.concat(
        discovered,
        ignore_index=True
    )

    if args.retain_top_k > 0:
        formulas = (
            formulas.sort_values(["category", "rank"])
            .groupby("category", group_keys=False)
            .head(args.retain_top_k)
            .reset_index(drop=True)
        )


    # --------------------------------------------------------
    # Select top-K formulas for bootstrap only
    # --------------------------------------------------------

    bootstrap_formulas = (
        formulas
        .sort_values(
            [
                "category",
                "formula_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "category",
            group_keys=False
        )
        .head(
            args.validation_top_k
        )
        .reset_index(
            drop=True
        )
    )


    print(
        f"Bootstrap validation formulas: "
        f"{len(bootstrap_formulas)}"
    )


    # --------------------------------------------------------
    # Validate all formulas
    # --------------------------------------------------------

    validated = validate(
        formulas,
        bootstrap_formulas,
        args.data_dir,
        args.bootstrap_times,
    )


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    validated.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"Saved {len(validated)} formulas "
        f"with held-out validation results "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
