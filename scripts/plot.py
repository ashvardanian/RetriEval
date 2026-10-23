#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""
Generate benchmark charts from JSON result files, as SVG in the diagrams' red palette.

Usage:
    uv run scripts/plot.py results/ --output plots/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

THEME = {
    "field": "#C44536",
    "ink": "#FFE6A7",
    "frame": "#E5E5E5",
    "font": ["Poppins", "Helvetica", "Arial", "sans-serif"],
}
# USearch draws at full strength and every other engine at half, as on the USearch page.
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def load_reports(input_path: str) -> list[dict[str, Any]]:
    """Load all JSON report files from a directory or a single file."""
    path = Path(input_path)
    reports: list[dict[str, Any]] = []
    if path.is_dir():
        for filepath in sorted(path.glob("*.json")):
            with open(filepath) as fh:
                reports.append(json.load(fh))
    elif path.is_file():
        with open(path) as fh:
            reports.append(json.load(fh))
    return reports


def series_key(report: dict[str, Any]) -> str:
    """Derive a unique series name from a report's config."""
    config: dict[str, Any] = report.get("config", {})
    parts: list[str] = [config.get("backend", "?")]
    for field in ("data_type", "metric"):
        if field in config:
            parts.append(str(config[field]))
    if "connectivity" in config:
        parts.append(f"M={config['connectivity']}")
    if "expansion_add" in config and "expansion_search" in config:
        parts.append(f"ef={config['expansion_add']}/{config['expansion_search']}")
    elif "expansion_search" in config:
        parts.append(f"ef=?/{config['expansion_search']}")
    shards = config.get("shards", 1)
    if isinstance(shards, int) and shards > 1:
        parts.append(f"{shards}s")
    # k varies by dataset and by --top-k, and metrics taken at different
    # k are not comparable — without it two such runs draw as one series.
    if "top_k" in config:
        parts.append(f"@{config['top_k']}")
    # Self-search parameters take part in the config hash, so two runs differing
    # only in these write separate report files.
    if "self_search_top_k" in config:
        parts.append(
            f"self@{config['self_search_top_k']}×{config.get('self_search_sample', '?')}"
        )
    return " · ".join(parts)


def add_field(name: str) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    """Read a field out of a step's add phase; None when the step had none."""
    return lambda step, _report: (step.get("add") or {}).get(name)


def ground_truth_field(name: str) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    """Read a field out of a step's ground-truth search."""
    return lambda step, _report: (step.get("ground_truth_search") or {}).get(name)


def coverage(step: dict[str, Any], report: dict[str, Any]) -> float | None:
    """Share of the base file this step had indexed — the ceiling raw recall is
    bounded by, and the divisor a reader needs to interpret a capped run."""
    total = (report.get("dataset") or {}).get("vectors_count") or 0
    return int(step["vectors_indexed"]) / total if total else None


def make_plot(
    title: str,
    reports: list[dict[str, Any]],
    y_fn: Callable[[dict[str, Any], dict[str, Any]], Any],
    x_label: str,
    y_label: str,
    filename: str,
    output_dir: Path,
    machine_info: dict[str, Any],
    logarithmic: bool = False,
) -> None:
    """Draw one chart of `y_fn` against the vectors indexed, one line per report."""
    figure, axis = plt.subplots(figsize=(16, 9.2))
    for index, report in enumerate(reports):
        # Points are paired before filtering: dropping only the y-values would
        # shift every remaining point onto the wrong x.
        points = []
        for step in report.get("steps", []):
            value = y_fn(step, report)
            if value is not None:
                points.append((int(step["vectors_indexed"]), value))
        # A load run has no add phase, so it contributes nothing to some charts.
        if not points:
            continue
        leading = (report.get("config") or {}).get("backend") == "usearch"
        axis.plot(
            [x for x, _ in points],
            [y for _, y in points],
            marker=MARKERS[index % len(MARKERS)],
            color=THEME["ink"],
            alpha=1.0 if leading else 0.5,
            linewidth=3 if leading else 2,
            label=series_key(report),
        )
    if logarithmic:
        axis.set_yscale("log")
        low, high = axis.get_ylim()
        axis.set_ylim(
            10 ** math.floor(math.log10(low)), 10 ** math.ceil(math.log10(high))
        )
    else:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.4g}"))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    subtitle = machine_info.get("cpu_model", "") if machine_info else ""
    axis.set_title(f"{title}\n{subtitle}".strip(), loc="left")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.2)
    axis.legend(loc="best", frameon=False)
    figure.savefig(output_dir / filename, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from JSON files"
    )
    parser.add_argument(
        "input", help="Directory containing JSON result files, or a single file"
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory, <input>/plots by default"
    )
    args = parser.parse_args()

    reports = load_reports(args.input)
    if not reports:
        print("No reports found.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = (
        args.output
        or (input_path if input_path.is_dir() else input_path.parent) / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["field"],
            "axes.facecolor": THEME["field"],
            "axes.edgecolor": THEME["frame"],
            "axes.labelcolor": THEME["ink"],
            "axes.titlecolor": THEME["ink"],
            "text.color": THEME["ink"],
            "xtick.color": THEME["ink"],
            "ytick.color": THEME["ink"],
            "grid.color": THEME["ink"],
            "font.family": THEME["font"],
            "font.size": 14,
            "svg.fonttype": "none",
        }
    )

    machine_info: dict[str, Any] = reports[0].get("machine", {})
    series_names = [series_key(r) for r in reports]
    print(f"Found {len(reports)} reports: {', '.join(series_names)}", file=sys.stderr)

    # k is per-report now, so title it only when every report agrees.
    counts = {
        (r.get("config") or {}).get("top_k")
        for r in reports
        if (r.get("config") or {}).get("top_k") is not None
    }
    k_label = str(counts.pop()) if len(counts) == 1 else "K"

    make_plot(
        "Construction Speed",
        reports,
        add_field("throughput"),
        "Vectors Indexed",
        "Vectors / Second",
        "construction-speed.svg",
        output_dir,
        machine_info,
        logarithmic=True,
    )

    make_plot(
        "Index Memory",
        reports,
        lambda s, _r: s["memory_bytes"] / 2**30,
        "Vectors Indexed",
        "Memory (GB)",
        "construction-memory.svg",
        output_dir,
        machine_info,
    )

    make_plot(
        "Search Speed",
        reports,
        ground_truth_field("throughput"),
        "Vectors Indexed",
        "Queries / Second",
        "search-speed.svg",
        output_dir,
        machine_info,
        logarithmic=True,
    )

    make_plot(
        "Recall@1",
        reports,
        ground_truth_field("recall_at_1"),
        "Vectors Indexed",
        "Recall@1",
        "recall-at-1.svg",
        output_dir,
        machine_info,
    )

    make_plot(
        f"Recall@{k_label}",
        reports,
        ground_truth_field("recall_at_k"),
        "Vectors Indexed",
        f"Recall@{k_label}",
        "recall-at-k.svg",
        output_dir,
        machine_info,
    )

    make_plot(
        f"Intersection@{k_label}",
        reports,
        ground_truth_field("intersection_at_k"),
        "Vectors Indexed",
        f"Intersection@{k_label}",
        "intersection-at-k.svg",
        output_dir,
        machine_info,
    )

    make_plot(
        f"NDCG@{k_label}",
        reports,
        ground_truth_field("ndcg_at_k"),
        "Vectors Indexed",
        f"NDCG@{k_label}",
        "ndcg-at-k.svg",
        output_dir,
        machine_info,
    )

    make_plot(
        "Ground-Truth Coverage",
        reports,
        coverage,
        "Vectors Indexed",
        "Indexed / Base File",
        "coverage.svg",
        output_dir,
        machine_info,
    )

    print(f"Plots written to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
