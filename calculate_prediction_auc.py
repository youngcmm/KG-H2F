"""
Calculate ROC-AUC and PR-AUC from link-prediction score CSV files.

Example:
    python calculate_prediction_auc.py

The default input directory is:
    outputs/kg_link_prediction/result_v1/predictions

The default output file is:
    outputs/kg_link_prediction/result_v1/prediction_auc_metrics.csv
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


# DEFAULT_PRED_DIR = Path("outputs/kg_link_prediction/result_v1_seed43/predictions")
# DEFAULT_OUT = Path("outputs/kg_link_prediction/result_v1_seed43/predictions/prediction_auc_metrics.csv")

DEFAULT_PRED_DIR = Path("outputs/kg_link_prediction/result_v1/predictions")
DEFAULT_OUT = Path("outputs/kg_link_prediction/result_v1/predictions/prediction_auc_metrics.csv")

# DEFAULT_PRED_DIR = Path("outputs/kg_link_prediction/result_v1_seed44/predictions")
# DEFAULT_OUT = Path("outputs/kg_link_prediction/result_v1_seed44/predictions/prediction_auc_metrics.csv")


METHOD_NAMES = {
    "complex": "ComplEx",
    "compgcn": "CompGCN",
    "conve": "ConvE",
    "distmult": "DistMult",
    "p_transe": "P-TransE",
    "pr_transe": "PR-TransE",
    "r_transe": "R-TransE",
    "rotate": "RotatE",
    "transe": "TransE",
    "tucker": "TuckER",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ROC-AUC and PR-AUC from prediction score CSV files."
    )
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pattern",
        default="*_scores.csv",
        help="Prediction filename pattern under --pred-dir.",
    )
    parser.add_argument(
        "--group-cols",
        nargs="*",
        default=["category", "edge_type"],
        help="Columns used for grouped metrics. Use an empty value to output only ALL.",
    )
    return parser.parse_args()


def is_missing(value):
    return value is None or str(value).strip() == ""


def to_float(value):
    if is_missing(value):
        return math.nan
    return float(str(value).strip())


def to_int_label(value):
    number = to_float(value)
    if math.isnan(number):
        raise ValueError("label contains an empty or non-numeric value")
    return int(number)


def find_score_column(fieldnames):
    score_cols = [name for name in fieldnames if name.endswith("_score")]
    if len(score_cols) != 1:
        raise ValueError(f"Expected exactly one *_score column, found: {score_cols}")
    return score_cols[0]


def infer_split(path):
    name = path.name
    if "_" not in name:
        return ""
    return name.split("_", 1)[0]


def infer_method(path, score_col):
    stem = path.stem
    if stem.endswith("_scores"):
        stem = stem[: -len("_scores")]
    split = infer_split(path)
    prefix = f"{split}_"
    raw_name = stem[len(prefix) :] if split and stem.startswith(prefix) else stem
    if raw_name.endswith("_score"):
        raw_name = raw_name[: -len("_score")]
    raw_name = score_col[: -len("_score")] if raw_name == stem else raw_name
    return METHOD_NAMES.get(raw_name, raw_name)


def average_ranks(values):
    sorted_items = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_items):
        end = index + 1
        while end < len(sorted_items) and sorted_items[end][1] == sorted_items[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for original_index, _ in sorted_items[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def roc_auc_score(labels, scores):
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return math.nan

    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc_score(labels, scores):
    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        return math.nan

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0

    while index < len(pairs):
        score = pairs[index][0]
        group_positive = 0
        group_negative = 0
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                group_positive += 1
            else:
                group_negative += 1
            index += 1

        true_positive += group_positive
        false_positive += group_negative
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall

    return average_precision


def format_metric(value):
    if math.isnan(value):
        return ""
    return f"{value:.12g}"


def metric_row(split, method, category, edge_type, rows):
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    return {
        "split": split,
        "method": method,
        "category": category,
        "edge_type": edge_type,
        "ROC-AUC": format_metric(roc_auc_score(labels, scores)),
        "PR-AUC": format_metric(pr_auc_score(labels, scores)),
        "num_samples": str(len(rows)),
        "num_positive": str(positives),
        "num_negative": str(negatives),
    }


def read_prediction_file(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header")
        if "label" not in reader.fieldnames:
            raise ValueError("CSV file must contain a label column")
        score_col = find_score_column(reader.fieldnames)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                label = to_int_label(row["label"])
                score = to_float(row[score_col])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if label not in {0, 1}:
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1, got {label}")
            if math.isnan(score):
                raise ValueError(f"{path}:{line_number}: score is empty")
            rows.append({"raw": row, "label": label, "score": score})
    return score_col, rows


def grouped_rows(rows, group_cols):
    if not group_cols:
        return []

    groups = defaultdict(list)
    for row in rows:
        raw = row["raw"]
        key = tuple(raw.get(col, "") for col in group_cols)
        groups[key].append(row)

    output = []
    for key, group in sorted(groups.items()):
        group_values = dict(zip(group_cols, key))
        output.append(
            (
                group_values.get("category", ""),
                group_values.get("edge_type", ""),
                group,
            )
        )
    return output


def calculate_metrics(pred_dir, pattern, group_cols):
    metric_rows = []
    prediction_files = sorted(pred_dir.glob(pattern))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files matched: {pred_dir / pattern}")

    for path in prediction_files:
        score_col, rows = read_prediction_file(path)
        split = infer_split(path)
        method = infer_method(path, score_col)

        metric_rows.append(metric_row(split, method, "ALL", "ALL", rows))
        for category, edge_type, group in grouped_rows(rows, group_cols):
            metric_rows.append(metric_row(split, method, category, edge_type, group))

    return metric_rows


def write_metrics(rows, out_path):
    fieldnames = [
        "split",
        "method",
        "category",
        "edge_type",
        "ROC-AUC",
        "PR-AUC",
        "num_samples",
        "num_positive",
        "num_negative",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = calculate_metrics(args.pred_dir, args.pattern, args.group_cols)
    write_metrics(rows, args.out)
    print(f"Saved AUC metrics to: {args.out}")


if __name__ == "__main__":
    main()
