#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly>=6.1", "kaleido>=1"]
# ///
"""
Generate benchmark plots from JSON result files.

Usage:
    uv run scripts/plot.py results/ --output-dir plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


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
) -> None:
    """Generate a single Plotly chart."""
    fig = go.Figure()
    for i, report in enumerate(reports):
        # Points are paired before filtering: dropping only the y-values would
        # shift every remaining point onto the wrong x.
        points = []
        for step in report.get("steps", []):
            value = y_fn(step, report)
            if value is None:
                continue
            points.append((int(step["vectors_indexed"]), value))
        # A load run has no add phase, so it contributes nothing to some charts.
        # Skip it rather than drawing an empty legend entry.
        if not points:
            continue
        fig.add_trace(
            go.Scatter(
                x=[x for x, _ in points],
                y=[y for _, y in points],
                mode="lines+markers",
                name=series_key(report),
                line={"color": COLORS[i % len(COLORS)]},
            )
        )

    subtitle = (
        f"<br><sub>{machine_info.get('cpu_model', '')}</sub>" if machine_info else ""
    )
    fig.update_layout(
        title=f"{title}{subtitle}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        legend={"x": 0.01, "y": 0.99},
    )
    fig.write_image(str(output_dir / filename), width=1200, height=600, scale=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from JSON files"
    )
    parser.add_argument(
        "input", help="Directory containing JSON result files, or a single file"
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Output directory for PNGs"
    )
    args = parser.parse_args()

    reports = load_reports(args.input)
    if not reports:
        print("No reports found.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "construction-speed.png",
        output_dir,
        machine_info,
    )

    make_plot(
        "Index Memory",
        reports,
        lambda s, _r: s["memory_bytes"] / 2**30,
        "Vectors Indexed",
        "Memory (GB)",
        "construction-memory.png",
        output_dir,
        machine_info,
    )

    make_plot(
        "Search Speed",
        reports,
        ground_truth_field("throughput"),
        "Vectors Indexed",
        "Queries / Second",
        "search-speed.png",
        output_dir,
        machine_info,
    )

    make_plot(
        "Recall@1",
        reports,
        ground_truth_field("recall_at_1"),
        "Vectors Indexed",
        "Recall@1",
        "recall-at-1.png",
        output_dir,
        machine_info,
    )

    make_plot(
        f"Recall@{k_label}",
        reports,
        ground_truth_field("recall_at_k"),
        "Vectors Indexed",
        f"Recall@{k_label}",
        "recall-at-k.png",
        output_dir,
        machine_info,
    )

    make_plot(
        f"Intersection@{k_label}",
        reports,
        ground_truth_field("intersection_at_k"),
        "Vectors Indexed",
        f"Intersection@{k_label}",
        "intersection-at-k.png",
        output_dir,
        machine_info,
    )

    make_plot(
        f"NDCG@{k_label}",
        reports,
        ground_truth_field("ndcg_at_k"),
        "Vectors Indexed",
        f"NDCG@{k_label}",
        "ndcg-at-k.png",
        output_dir,
        machine_info,
    )

    make_plot(
        "Ground-Truth Coverage",
        reports,
        coverage,
        "Vectors Indexed",
        "Indexed / Base File",
        "coverage.png",
        output_dir,
        machine_info,
    )

    print(f"Plots written to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
