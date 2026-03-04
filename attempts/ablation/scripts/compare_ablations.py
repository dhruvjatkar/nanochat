#!/usr/bin/env python3
"""
Compare ablation study results.

Parses ablations/*/result.json files and produces ranked tables showing
which features help or hurt relative to the baseline.

Usage:
    python scripts/compare_ablations.py --dir ablations/
    python scripts/compare_ablations.py --dir ablations/ --sort core_score
    python scripts/compare_ablations.py --dir ablations/ --format csv
    python scripts/compare_ablations.py --dir ablations/ --format summary

Programmatic usage:
    from compare_ablations import load_results
    results = load_results("ablations/")
"""

import argparse
import csv
import io
import json
import os
import sys


def load_results(directory):
    """Load all result.json files from ablation subdirectories.

    Args:
        directory: Path to the ablations directory. Each subdirectory is
            expected to contain a result.json file.

    Returns:
        List of dicts, each containing the parsed result.json contents.
        Subdirectories without a valid result.json are silently skipped.
    """
    results = []
    if not os.path.isdir(directory):
        print(f"Warning: directory '{directory}' does not exist.", file=sys.stderr)
        return results

    for entry in sorted(os.listdir(directory)):
        subdir = os.path.join(directory, entry)
        if not os.path.isdir(subdir):
            continue
        result_path = os.path.join(subdir, "result.json")
        if not os.path.isfile(result_path):
            continue
        try:
            with open(result_path, "r") as f:
                data = json.load(f)
            data["_dir"] = subdir
            # Load per-node delta if available (from SLURM same-node comparison)
            delta_path = os.path.join(subdir, "delta.json")
            if os.path.isfile(delta_path):
                with open(delta_path, "r") as f:
                    data["_node_delta"] = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Warning: skipping {result_path}: {e}",
                file=sys.stderr,
            )
    return results


def _find_baseline(results):
    """Find the baseline result entry.

    Looks for a result whose name contains 'baseline' (case-insensitive).
    Returns None if no baseline is found.
    """
    for r in results:
        if "baseline" in r.get("name", "").lower():
            return r
    return None


def _format_elapsed(seconds):
    """Format elapsed seconds as 'Xm Ys'."""
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}m {secs:02d}s"


def _format_params(num_params):
    """Format parameter count with M/B suffix."""
    if num_params is None:
        return "N/A"
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def _sort_results(results, sort_key):
    """Sort results by the given key.

    min_val_bpb: ascending (lower is better).
    core_score: descending (higher is better).
    """
    if sort_key == "core_score":
        return sorted(results, key=lambda r: r.get("core_score", 0), reverse=True)
    # Default: min_val_bpb ascending
    return sorted(results, key=lambda r: r.get("min_val_bpb", float("inf")))


def format_table(results, sort_key="min_val_bpb"):
    """Format results as a human-readable ranked table.

    Args:
        results: List of result dicts from load_results().
        sort_key: Field to sort by ('min_val_bpb' or 'core_score').

    Returns:
        String containing the formatted table.
    """
    if not results:
        return "No results found."

    sorted_results = _sort_results(results, sort_key)
    baseline = _find_baseline(sorted_results)
    baseline_bpb = baseline.get("min_val_bpb") if baseline else None

    # Column headers
    headers = ["Rank", "Name", "min_val_bpb", "delta_bpb", "core_score", "elapsed", "params"]

    # Build rows
    rows = []
    for i, r in enumerate(sorted_results, start=1):
        name = r.get("name", "unknown")
        bpb = r.get("min_val_bpb")
        core = r.get("core_score")
        elapsed = r.get("elapsed_seconds")
        params = r.get("num_params")

        bpb_str = f"{bpb:.4f}" if bpb is not None else "N/A"

        # Prefer per-node delta (same-node baseline comparison) if available
        node_delta = r.get("_node_delta", {})
        if node_delta.get("delta_bpb") is not None:
            delta = node_delta["delta_bpb"]
            delta_str = f"{delta:+.4f}*"  # * marks same-node comparison
        elif baseline_bpb is not None and bpb is not None:
            delta = bpb - baseline_bpb
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"

        core_str = f"{core:.4f}" if core is not None else "N/A"
        elapsed_str = _format_elapsed(elapsed)
        params_str = _format_params(params)

        rows.append([str(i), name, bpb_str, delta_str, core_str, elapsed_str, params_str])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    # Format output
    lines = []

    # Header line
    header_line = "  ".join(h.ljust(col_widths[j]) for j, h in enumerate(headers))
    lines.append(header_line)

    # Separator line
    sep_line = "  ".join("-" * col_widths[j] for j in range(len(headers)))
    lines.append(sep_line)

    # Data rows
    for row in rows:
        row_line = "  ".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row))
        lines.append(row_line)

    return "\n".join(lines)


def format_csv(results, sort_key="min_val_bpb"):
    """Format results as CSV.

    Args:
        results: List of result dicts from load_results().
        sort_key: Field to sort by ('min_val_bpb' or 'core_score').

    Returns:
        String containing CSV output.
    """
    if not results:
        return ""

    sorted_results = _sort_results(results, sort_key)
    baseline = _find_baseline(sorted_results)
    baseline_bpb = baseline.get("min_val_bpb") if baseline else None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["rank", "name", "flags", "min_val_bpb", "delta_bpb",
                      "core_score", "num_params", "elapsed_seconds", "depth"])

    for i, r in enumerate(sorted_results, start=1):
        bpb = r.get("min_val_bpb")
        node_delta = r.get("_node_delta", {})
        if node_delta.get("delta_bpb") is not None:
            delta = node_delta["delta_bpb"]
        elif baseline_bpb is not None and bpb is not None:
            delta = bpb - baseline_bpb
        else:
            delta = ""

        writer.writerow([
            i,
            r.get("name", ""),
            r.get("flags", ""),
            bpb if bpb is not None else "",
            delta,
            r.get("core_score", ""),
            r.get("num_params", ""),
            r.get("elapsed_seconds", ""),
            r.get("depth", ""),
        ])

    return output.getvalue()


def format_summary(results, sort_key="min_val_bpb"):
    """Format results as a summary of helping vs hurting features.

    Groups features into HELPING (negative delta_bpb, meaning better than
    baseline) and HURTING (positive delta_bpb, meaning worse than baseline),
    with recommendations.

    Args:
        results: List of result dicts from load_results().
        sort_key: Field to sort by ('min_val_bpb' or 'core_score').

    Returns:
        String containing the summary.
    """
    if not results:
        return "No results found."

    sorted_results = _sort_results(results, sort_key)
    baseline = _find_baseline(sorted_results)

    if baseline is None:
        lines = [
            "WARNING: No baseline found. Cannot compute deltas.",
            "",
            "Results sorted by " + sort_key + ":",
            "",
        ]
        for i, r in enumerate(sorted_results, start=1):
            name = r.get("name", "unknown")
            bpb = r.get("min_val_bpb")
            bpb_str = f"{bpb:.4f}" if bpb is not None else "N/A"
            lines.append(f"  {i}. {name}: min_val_bpb={bpb_str}")
        return "\n".join(lines)

    baseline_bpb = baseline.get("min_val_bpb")
    if baseline_bpb is None:
        return "ERROR: Baseline has no min_val_bpb value."

    helping = []
    hurting = []
    neutral = []

    for r in sorted_results:
        name = r.get("name", "unknown")
        bpb = r.get("min_val_bpb")
        if bpb is None:
            continue
        node_delta_data = r.get("_node_delta", {})
        if node_delta_data.get("delta_bpb") is not None:
            delta = node_delta_data["delta_bpb"]
        else:
            delta = bpb - baseline_bpb

        entry = {
            "name": name,
            "bpb": bpb,
            "delta": delta,
            "flags": r.get("flags", ""),
        }

        if "baseline" in name.lower():
            neutral.append(entry)
        elif delta < -1e-6:
            helping.append(entry)
        elif delta > 1e-6:
            hurting.append(entry)
        else:
            neutral.append(entry)

    # Sort helping by delta ascending (most helpful first)
    helping.sort(key=lambda e: e["delta"])
    # Sort hurting by delta descending (most harmful first)
    hurting.sort(key=lambda e: e["delta"], reverse=True)

    lines = []
    lines.append(f"Baseline: {baseline.get('name', 'unknown')} "
                 f"(min_val_bpb={baseline_bpb:.4f})")
    lines.append("")

    # Helping
    lines.append(f"HELPING ({len(helping)} features with negative delta -- better than baseline):")
    if helping:
        for e in helping:
            lines.append(f"  {e['delta']:+.4f}  {e['name']}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Hurting
    lines.append(f"HURTING ({len(hurting)} features with positive delta -- worse than baseline):")
    if hurting:
        for e in hurting:
            lines.append(f"  {e['delta']:+.4f}  {e['name']}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Neutral
    if neutral:
        lines.append(f"NEUTRAL ({len(neutral)} features with no significant delta):")
        for e in neutral:
            lines.append(f"  {e['delta']:+.4f}  {e['name']}")
        lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS:")
    if helping:
        best = helping[0]
        lines.append(f"  - Best feature: {best['name']} ({best['delta']:+.4f} bpb)")
        lines.append(f"    Keep this feature. It provides the largest improvement.")
    if hurting:
        worst = hurting[0]
        lines.append(f"  - Worst feature: {worst['name']} ({worst['delta']:+.4f} bpb)")
        lines.append(f"    Consider removing this feature. It hurts performance the most.")
    if helping and len(helping) > 1:
        flags = [e["flags"] for e in helping if e["flags"]]
        if flags:
            lines.append(f"  - All helping flags: {' '.join(flags)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ablation study results from result.json files.",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the ablations directory containing subdirectories with result.json files.",
    )
    parser.add_argument(
        "--sort",
        choices=["min_val_bpb", "core_score"],
        default="min_val_bpb",
        help="Field to sort by (default: min_val_bpb ascending, core_score descending).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "summary"],
        default="table",
        dest="output_format",
        help="Output format (default: table).",
    )

    args = parser.parse_args()

    results = load_results(args.dir)
    if not results:
        print("No result.json files found.", file=sys.stderr)
        sys.exit(1)

    if args.output_format == "table":
        print(format_table(results, sort_key=args.sort))
    elif args.output_format == "csv":
        print(format_csv(results, sort_key=args.sort), end="")
    elif args.output_format == "summary":
        print(format_summary(results, sort_key=args.sort))


if __name__ == "__main__":
    main()
