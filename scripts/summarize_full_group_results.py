#!/usr/bin/env python3
"""Validate and summarize the five full-group baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

RUN_NAMES = (
    "transformer",
    "solar_park",
    "wind_park",
    "mv_feeder",
    "station_installation",
)
METHODS = (
    "independent",
    "static_gaussian",
    "conditional_low_rank",
    "set_aware_low_rank",
)
METHOD_LABELS = {
    "independent": "M0 independent",
    "static_gaussian": "M1 static",
    "conditional_low_rank": "M2 conditional low-rank",
    "set_aware_low_rank": "M3 set-aware",
}
GROUP_LABELS = {
    "transformer": "Transformer",
    "solar_park": "Solar park",
    "wind_park": "Wind park",
    "mv_feeder": "MV feeder",
    "station_installation": "Station installation",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_protocol(metadata: dict[str, Any], path: Path) -> None:
    protocol = metadata.get("experimental_protocol", {})
    expected = {
        "name": "full_group",
        "full_group_only": True,
        "subset_training": False,
        "entity_selection_augmentation_enabled": False,
    }
    if protocol != expected:
        raise ValueError(f"{path} is not a clean full-group run: {protocol!r}")


def _format(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return f"{value:.8g}"


def summarize(runs_root: Path, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for group_name in RUN_NAMES:
        run = runs_root / f"full_group_{group_name}_m0_m3"
        experiment_manifest = _read_json(run / "experiment_manifest.json")
        evaluation_manifest = _read_json(run / "evaluation" / "evaluation_manifest.json")
        _validate_protocol(experiment_manifest, run / "experiment_manifest.json")
        _validate_protocol(evaluation_manifest, run / "evaluation" / "evaluation_manifest.json")
        group = experiment_manifest["group"]
        if group != evaluation_manifest["group"] or group["name"] != group_name:
            raise ValueError(f"inconsistent group metadata in {run}")
        entity_ids = group["entity_ids"]
        if group["entity_count"] != len(entity_ids) or len(set(entity_ids)) != len(entity_ids):
            raise ValueError(f"invalid complete entity list in {run}")
        if tuple(experiment_manifest["methods"]) != METHODS:
            raise ValueError(f"unexpected method hierarchy in {run}")
        if any((run / "evaluation").glob("*variable*")):
            raise ValueError(f"reduced-cardinality output found in {run}")

        metrics = _read_json(run / "evaluation" / "metrics.json")
        if set(metrics) != set(METHODS):
            raise ValueError(f"incomplete metrics in {run}")
        valid_counts = {metrics[method]["valid_origin_lead_count"] for method in METHODS}
        if len(valid_counts) != 1:
            raise ValueError(f"methods use different valid cases in {run}")

        for method in METHODS:
            recorded_method_run = Path(experiment_manifest["method_runs"][method])
            if recorded_method_run.name != method:
                raise ValueError(f"unexpected method path in {run}")
            method_run = run / method
            run_metadata = _read_json(method_run / "run_metadata.json")
            _validate_protocol(run_metadata, method_run / "run_metadata.json")
            if run_metadata["group"] != group:
                raise ValueError(f"method group differs from experiment group in {method_run}")
            summary_path = method_run / "training_summary.json"
            training = _read_json(summary_path) if summary_path.exists() else {}
            result = metrics[method]
            rows.append(
                {
                    "group": group_name,
                    "entity_count": group["entity_count"],
                    "entity_ids": json.dumps(entity_ids, ensure_ascii=False),
                    "method": method,
                    "mean_pinball": result["mean_pinball"],
                    "crps": result["crps"],
                    "weighted_interval_score": result["weighted_interval_score"],
                    "coverage_0.9": result["coverage_0.9"],
                    "interval_width_0.9": result["interval_width_0.9"],
                    "energy_score": result["energy_score"],
                    "variogram_score": result["variogram_score"],
                    "valid_origin_lead_count": result["valid_origin_lead_count"],
                    "dropped_origin_lead_count": result["dropped_origin_lead_count"],
                    "best_epoch": training.get("best_epoch", ""),
                    "best_validation_nll": training.get("best_validation_nll", ""),
                }
            )
        groups.append(group)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "full_group_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Full-group baseline result summary",
        "",
        "This file is generated only after validating every experiment and method manifest as",
        "`full_group_only: true`, `subset_training: false`, with entity-selection augmentation",
        "disabled. M2 and M3 are newly trained full-group models; no legacy neural checkpoint is used.",
        "Each table describes the saved resolved configuration of its source run; regenerate after",
        "any input-protocol change before treating it as a current experimental result.",
        "",
        "## Headline ranking by group",
        "",
        "| Group | $K_g$ | Valid cases | Best mean-pinball method | Mean pinball | Change vs M0 | 90% coverage |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group["name"]]
        best = min(group_rows, key=lambda row: float(row["mean_pinball"]))
        m0 = next(row for row in group_rows if row["method"] == "independent")
        change = 100 * (float(best["mean_pinball"]) / float(m0["mean_pinball"]) - 1)
        markdown.append(
            f"| {GROUP_LABELS[group['name']]} | {group['entity_count']} | "
            f"{best['valid_origin_lead_count']:,} | {METHOD_LABELS[best['method']]} | "
            f"{_format(float(best['mean_pinball']))} | {change:+.3f}% | "
            f"{float(best['coverage_0.9']):.4f} |"
        )

    markdown.extend(
        [
            "",
            "## Complete fixed entity groups",
            "",
        ]
    )
    for group in groups:
        markdown.extend(
            [
                f"### {GROUP_LABELS[group['name']]} ($K_g={group['entity_count']}$)",
                "",
                *[f"- `{entity_id}`" for entity_id in group["entity_ids"]],
                "",
            ]
        )

    markdown.extend(
        [
            "## Complete metric table",
            "",
            "Energy Score is the empirical all-pairs estimator on the selected 512-member joint ensemble.",
            "",
            "| Group | Method | Mean pinball | WIS | 90% coverage | 90% width | Energy Score |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        markdown.append(
            f"| {GROUP_LABELS[row['group']]} | {METHOD_LABELS[row['method']]} | "
            f"{_format(float(row['mean_pinball']))} | "
            f"{_format(float(row['weighted_interval_score']))} | "
            f"{float(row['coverage_0.9']):.4f} | "
            f"{_format(float(row['interval_width_0.9']))} | "
            f"{_format(float(row['energy_score']))} |"
        )
    (output_dir / "full_group_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summarize(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()
