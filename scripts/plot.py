#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly", "kaleido==0.2.*"]
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
from pathlib import Path
from typing import Any, Callable

import plotly.graph_objects as go

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
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
    for field in ("dtype", "metric"):
        if field in config:
            parts.append(str(config[field]))
    if "connectivity" in config:
        parts.append(f"M={config['connectivity']}")
    if "expansion_add" in config and "expansion_search" in config:
        parts.append(f"ef={config['expansion_add']}/{config['expansion_search']}")
    shards = config.get("shards", 1)
    if isinstance(shards, int) and shards > 1:
        parts.append(f"{shards}s")
    return " · ".join(parts)


def make_plot(
    title: str,
    reports: list[dict[str, Any]],
    x_fn: Callable[[dict[str, Any]], Any],
    y_fn: Callable[[dict[str, Any]], Any],
    x_label: str,
    y_label: str,
    filename: str,
    output_dir: Path,
    machine_info: dict[str, Any],
) -> None:
    """Generate a single Plotly chart."""
    fig = go.Figure()
    for i, report in enumerate(reports):
        name = series_key(report)
        steps: list[dict[str, Any]] = report.get("steps", [])
        x_values = [x_fn(s) for s in steps]
        y_values = [y_fn(s) for s in steps]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", name=name,
            line={"color": COLORS[i % len(COLORS)]},
        ))

    subtitle = f"<br><sub>{machine_info.get('cpu_model', '')}</sub>" if machine_info else ""
    fig.update_layout(
        title=f"{title}{subtitle}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        legend={"x": 0.01, "y": 0.99},
    )
    fig.write_image(str(output_dir / filename), width=1200, height=600, scale=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSON files")
    parser.add_argument("input", help="Directory containing JSON result files, or a single file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for PNGs")
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

    def vectors_indexed(step: dict[str, Any]) -> int:
        return int(step["vectors_indexed"])

    make_plot("Construction Speed", reports, vectors_indexed,
              lambda s: s["add_throughput"],
              "Vectors Indexed", "Vectors / Second",
              "construction-speed.png", output_dir, machine_info)

    make_plot("Index Memory", reports, vectors_indexed,
              lambda s: s["memory_bytes"] / 1e9,
              "Vectors Indexed", "Memory (GB)",
              "construction-memory.png", output_dir, machine_info)

    make_plot("Search Speed", reports, vectors_indexed,
              lambda s: s["search_throughput"],
              "Vectors Indexed", "Queries / Second",
              "search-speed.png", output_dir, machine_info)

    make_plot("Recall@1", reports, vectors_indexed,
              lambda s: s["recall_at_1"],
              "Vectors Indexed", "Recall@1",
              "recall-at-1.png", output_dir, machine_info)

    make_plot("Recall@10", reports, vectors_indexed,
              lambda s: s["recall_at_10"],
              "Vectors Indexed", "Recall@10",
              "recall-at-10.png", output_dir, machine_info)

    make_plot("Recall@1 (normalized)", reports, vectors_indexed,
              lambda s: s["recall_at_1_normalized"],
              "Vectors Indexed", "Recall@1 (normalized)",
              "recall-at-1-normalized.png", output_dir, machine_info)

    make_plot("NDCG@10", reports, vectors_indexed,
              lambda s: s["ndcg_at_10"],
              "Vectors Indexed", "NDCG@10",
              "ndcg-at-10.png", output_dir, machine_info)

    print(f"Plots written to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
