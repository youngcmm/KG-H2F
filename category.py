"""
Calculate paper-level held-out evidence for all category-herb scores.

The model scores are treated as fixed upstream inputs. Held-out records are
used only to calculate prescription prevalence and category enrichment;
they never change the model ranking.

Validation reports:
1. Wilson 95% confidence intervals for herb reproduction rates.
2. Stratified bootstrap 95% confidence intervals for enrichment scores.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DEFAULT_SCORE_DIR = (
    ROOT /
    "outputs/kg_link_prediction/result_v1/downstream_scores"
)

DEFAULT_DATA_DIR = (
    ROOT /
    "data/broad_category_splits_standardized_6"
)

DEFAULT_OUTPUT = (
    ROOT /
    "evaluation_outputs/herb_ranking_all_categories.csv"
)

PRESCRIPTION_KEY = "标准处方药材名称"

EPS = 1e-8


CATEGORIES = {
    "sleep_dermatologic": "睡眠皮肤病共病",
    "sleep_emotional": "睡眠情志共病",
    "sleep_fatigue_deficiency": "睡眠虚劳共病",
    "sleep_neurologic": "睡眠神经共病",
    "sleep_oncologic": "睡眠肿瘤共病",
    "sleep_urinary": "睡眠泌尿共病",
}


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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--bootstrap-times",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--bootstrap-top-k",
        type=int,
        default=20
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
            p * (1-p) / total
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
# Data loading
# ============================================================

def split_herbs(
    value: object
) -> set[str]:

    if value is None:
        return set()

    return {
        item.strip()
        for item in str(value)
        .replace(",", "、")
        .split("、")
        if item.strip()
    }


def load_prescriptions(
    data_dir: Path,
    category: str,
    split: str
) -> list[set[str]]:

    path = (
        data_dir /
        category /
        f"{split}.json"
    )

    records = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    prescriptions = [
        split_herbs(
            record.get(
                PRESCRIPTION_KEY
            )
        )
        for record in records
    ]

    return [
        prescription
        for prescription in prescriptions
        if prescription
    ]


def load_scores(
    score_dir: Path,
    category: str
) -> pd.DataFrame:

    selected_path = (
        score_dir /
        f"comorbidity_{category}_selected_herb_scores.csv"
    )

    new_path = (
        score_dir /
        f"comorbidity_{category}_conve_herb_scores.csv"
    )

    legacy_path = (
        score_dir /
        f"category_{category}_conve_herb_scores.csv"
    )

    path = (
        selected_path
        if selected_path.exists()
        else new_path
        if new_path.exists()
        else legacy_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing score file: {path}"
        )

    frame = pd.read_csv(path)

    if "is_oov_for_selected_model" in frame.columns:
        oov = frame["is_oov_for_selected_model"].astype(str).str.lower().isin({"true", "1"})
        frame = frame.loc[~oov].copy()

    required = {
        "herb_name",
        "score"
    }

    missing = (
        required -
        set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{path} missing {missing}"
        )

    frame = frame.copy()

    frame["herb_name"] = (
        frame["herb_name"]
        .astype(str)
        .str.strip()
    )

    frame["score"] = pd.to_numeric(
        frame["score"],
        errors="coerce"
    )

    frame = frame.dropna(
        subset=[
            "herb_name",
            "score"
        ]
    )

    frame = (
        frame
        .sort_values(
            "score",
            ascending=False
        )
        .drop_duplicates(
            "herb_name"
        )
    )

    frame = frame.reset_index(
        drop=True
    )

    frame["model_rank"] = (
        range(
            1,
            len(frame)+1
        )
    )

    return frame


# ============================================================
# Bootstrap enrichment CI
# ============================================================

def bootstrap_herb_enrichment_ci(
    herb: str,
    target_category: str,
    test_sets: dict[str, list[set[str]]],
    n_bootstrap: int = 2000,
) -> tuple[float, float]:

    categories = list(
        test_sets.keys()
    )

    category_sizes = {category: len(test_sets[category]) for category in categories}
    # Resampling a binary presence vector with replacement gives a binomial
    # count. Drawing those counts directly is exactly equivalent and avoids
    # repeatedly materialising tens of thousands of prescriptions.
    category_counts = {
        category: np.random.binomial(
            category_sizes[category],
            sum(herb in record for record in test_sets[category]) / category_sizes[category],
            size=n_bootstrap,
        )
        for category in categories
    }
    target_rate = category_counts[target_category] / category_sizes[target_category]
    overall_rate = sum(category_counts.values()) / sum(category_sizes.values())
    enrichment_values = (target_rate + EPS) / (overall_rate + EPS)


    return (
        np.percentile(
            enrichment_values,
            2.5
        ),
        np.percentile(
            enrichment_values,
            97.5
        )
    )
# ============================================================
# Main analysis
# ============================================================

def run(
    score_dir: Path,
    data_dir: Path,
    output: Path,
    bootstrap_times: int,
    bootstrap_top_k: int,
) -> pd.DataFrame:


    test_sets = {
        category:
        load_prescriptions(
            data_dir,
            category,
            "test"
        )
        for category in CATEGORIES
    }


    counts = {
        category:
        Counter(
            herb
            for prescription
            in prescriptions
            for herb in prescription
        )
        for category, prescriptions
        in test_sets.items()
    }


    all_prescriptions = [
        prescription
        for prescriptions
        in test_sets.values()
        for prescription
        in prescriptions
    ]


    all_counts = Counter(
        herb
        for prescription
        in all_prescriptions
        for herb
        in prescription
    )


    n_all = len(
        all_prescriptions
    )


    output_frames = []


    for category, category_cn in CATEGORIES.items():

        scores = load_scores(
            score_dir,
            category
        )


        # ----------------------------------------------------
        # Only top-K herbs calculate bootstrap CI
        # All herbs are still retained in output
        # ----------------------------------------------------

        bootstrap_herbs = set(
            scores
            .head(
                bootstrap_top_k
            )["herb_name"]
        )


        print(
            f"{category}: "
            f"bootstrap herbs = {len(bootstrap_herbs)}"
        )


        n_target = len(
            test_sets[category]
        )


        rows = []


        for row in scores.itertuples(
            index=False
        ):

            herb = row.herb_name


            # -------------------------------
            # Held-out reproduction
            # -------------------------------

            target_count = counts[category][herb]

            overall_count = all_counts[herb]


            target_rate = (
                target_count
                /
                n_target
            )


            overall_rate = (
                overall_count
                /
                n_all
            )


            enrichment = (
                target_rate + EPS
            ) / (
                overall_rate + EPS
            )


            rate_ci_lower, rate_ci_upper = wilson_ci(
                target_count,
                n_target
            )


            # -------------------------------
            # Bootstrap enrichment CI
            # -------------------------------

            if herb in bootstrap_herbs:

                enrichment_ci_lower, enrichment_ci_upper = (
                    bootstrap_herb_enrichment_ci(
                        herb,
                        category,
                        test_sets,
                        bootstrap_times
                    )
                )

            else:

                enrichment_ci_lower = 0.0

                enrichment_ci_upper = 0.0


            rows.append(
                {
                    "category": category,

                    "category_cn": category_cn,

                    "rank": row.model_rank,

                    "herb_name": herb,

                    "model_score": row.score,


                    "test_count": target_count,

                    "test_total": n_target,

                    "test_rate": target_rate,

                    "test_rate_ci_lower":
                        rate_ci_lower,

                    "test_rate_ci_upper":
                        rate_ci_upper,


                    "category_enrichment":
                        enrichment,

                    "category_enrichment_ci_lower":
                        enrichment_ci_lower,

                    "category_enrichment_ci_upper":
                        enrichment_ci_upper,

                    "bootstrap_validated":
                        herb in bootstrap_herbs,
                }
            )


        output_frames.append(
            pd.DataFrame(rows)
        )


    all_results = pd.concat(
        output_frames,
        ignore_index=True
    )


    all_results = (
        all_results
        .sort_values(
            [
                "category",
                "rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    all_results.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )


    return all_results



# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()


    np.random.seed(42)


    results = run(
        args.score_dir,
        args.data_dir,
        args.output,
        args.bootstrap_times,
        args.bootstrap_top_k,
    )


    print(
        f"Saved {len(results)} "
        f"category-herb rows to {args.output}"
    )


    print(
        results
        .groupby("category")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
