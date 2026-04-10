#!/usr/bin/env python3
"""
Generate benchmark plots from JSONL output.

Usage:
    python scripts/plot.py results.jsonl --output-dir plots/
    cat results.jsonl | python scripts/plot.py - --output-dir plots/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_records(path: str) -> list[dict]:
    """Load JSONL records from a file path or stdin ('-')."""
    records = []
    source = sys.stdin if path == "-" else open(path)
    try:
        for line in source:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    finally:
        if source is not sys.stdin:
            source.close()
    return records


def series_key(record: dict) -> str:
    """Derive a unique series name from a record."""
    parts = [record.get("backend", "?")]
    if record.get("dtype"):
        parts.append(record["dtype"])
    if record.get("shards", 1) > 1:
        parts.append(f"{record['shards']}shards")
    return "-".join(parts)


def group_by_series(records: list[dict]) -> dict[str, list[dict]]:
    """Group measurement records by series key."""
    groups = defaultdict(list)
    for r in records:
        if r.get("phase") in ("add", "search"):
            groups[series_key(r)].append(r)
    return dict(groups)


def plot_with_plotly(groups: dict, output_dir: Path, machine_info: dict | None):
    """Generate plots using Plotly."""
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    def make_plot(title, x_fn, y_fn, x_label, y_label, filter_phase, filename, log_y=False):
        fig = go.Figure()
        for i, (name, recs) in enumerate(sorted(groups.items())):
            phase_recs = [r for r in recs if r["phase"] == filter_phase]
            if not phase_recs:
                continue
            x = [x_fn(r) for r in phase_recs]
            y = [y_fn(r) for r in phase_recs]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers",
                name=name, line=dict(color=colors[i % len(colors)]),
            ))
        subtitle = f"<br><sub>{machine_info.get('cpu_model', '')}</sub>" if machine_info else ""
        fig.update_layout(
            title=f"{title}{subtitle}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            yaxis_type="log" if log_y else "linear",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99),
        )
        fig.write_image(str(output_dir / filename), width=1200, height=600, scale=2)

    # Construction speed
    make_plot(
        "Construction Speed",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("vectors_per_second", 0),
        "Vectors Indexed", "Vectors / Second",
        "add", "construction-speed.png",
    )

    # Construction memory
    make_plot(
        "Index Memory",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("memory_bytes", 0) / 1e9,
        "Vectors Indexed", "Memory (GB)",
        "add", "construction-memory.png",
    )

    # Search speed
    make_plot(
        "Search Speed",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("queries_per_second", 0),
        "Vectors Indexed", "Queries / Second",
        "search", "search-speed.png",
    )

    # Search recall
    make_plot(
        "Search Recall@1",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("recall_at_1", 0),
        "Vectors Indexed", "Recall@1",
        "search", "search-recall-at-1.png",
    )

    make_plot(
        "Search Recall@10",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("recall_at_10", 0),
        "Vectors Indexed", "Recall@10",
        "search", "search-recall-at-10.png",
    )


def plot_with_matplotlib(groups: dict, output_dir: Path, machine_info: dict | None):
    """Generate plots using matplotlib (fallback)."""

    def make_plot(title, x_fn, y_fn, x_label, y_label, filter_phase, filename, log_y=False):
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, recs in sorted(groups.items()):
            phase_recs = [r for r in recs if r["phase"] == filter_phase]
            if not phase_recs:
                continue
            x = [x_fn(r) for r in phase_recs]
            y = [y_fn(r) for r in phase_recs]
            ax.plot(x, y, "o-", label=name, markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if log_y:
            ax.set_yscale("log")
        subtitle = f"\n{machine_info.get('cpu_model', '')}" if machine_info else ""
        ax.set_title(f"{title}{subtitle}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / filename), dpi=150)
        plt.close(fig)

    make_plot(
        "Construction Speed",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("vectors_per_second", 0),
        "Vectors Indexed", "Vectors / Second",
        "add", "construction-speed.png",
    )
    make_plot(
        "Index Memory",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("memory_bytes", 0) / 1e9,
        "Vectors Indexed", "Memory (GB)",
        "add", "construction-memory.png",
    )
    make_plot(
        "Search Speed",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("queries_per_second", 0),
        "Vectors Indexed", "Queries / Second",
        "search", "search-speed.png",
    )
    make_plot(
        "Search Recall@1",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("recall_at_1", 0),
        "Vectors Indexed", "Recall@1",
        "search", "search-recall-at-1.png",
    )
    make_plot(
        "Search Recall@10",
        lambda r: r["vectors_indexed"],
        lambda r: r.get("recall_at_10", 0),
        "Vectors Indexed", "Recall@10",
        "search", "search-recall-at-10.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSONL")
    parser.add_argument("input", help="JSONL file path or '-' for stdin")
    parser.add_argument("--output-dir", default="plots", help="Output directory for PNGs")
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    machine_info = next((r for r in records if r.get("phase") == "machine"), None)
    groups = group_by_series(records)

    if not groups:
        print("No measurement records found.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_names = sorted(groups.keys())
    print(f"Found {len(series_names)} series: {', '.join(series_names)}", file=sys.stderr)

    if HAS_PLOTLY:
        print("Using Plotly for visualization", file=sys.stderr)
        plot_with_plotly(groups, output_dir, machine_info)
    elif HAS_MATPLOTLIB:
        print("Using matplotlib for visualization (install plotly+kaleido for better plots)", file=sys.stderr)
        plot_with_matplotlib(groups, output_dir, machine_info)
    else:
        print("ERROR: Neither plotly nor matplotlib is installed.", file=sys.stderr)
        print("Install one: pip install plotly kaleido  OR  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    print(f"Plots written to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
