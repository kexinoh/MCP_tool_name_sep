from __future__ import annotations

"""Generate heatmaps and tables of API gateway and call success rates.

This utility reads ``name_probe.csv`` results for multiple models and produces
two heatmaps and their corresponding Markdown tables:

* ``gateway_success_heatmap.png`` – fraction of probes where the API gateway
  accepted the tool name.
* ``call_success_heatmap.png`` – fraction of successful tool calls after the
  gateway accepted the tool.
* ``gateway_success_table.md`` – table of API gateway success rates.
* ``call_success_table.md`` – table of tool call success rates.

The resulting images place categories across the top and model names on the
left. Color gradients run from red (low) to green (high)."""

from pathlib import Path
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load results and compute gateway and call success rates.

    Parameters
    ----------
    results_dir: Path
        Directory containing model subdirectories with ``name_probe.csv`` files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two dataframes with categories as columns and models as rows. The first
        captures API gateway success rates and the second captures tool call
        success rates conditioned on gateway success.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Results directory {results_dir} does not exist. "
            "Run experiments or specify --results-dir."
        )

    gateway_rates: dict[str, pd.Series] = {}
    call_rates: dict[str, pd.Series] = {}
    for model_dir in results_dir.iterdir():
        csv_path = model_dir / "name_probe.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        total = df.groupby("category")["api_status"].count()
        gateway_ok = df[df["api_status"] == "ok"]
        if total.empty:
            continue
        gateway_rate = gateway_ok.groupby("category")["api_status"].count() / total
        call_rate = (
            gateway_ok.groupby("category")["can_call"].mean()
            if not gateway_ok.empty
            else pd.Series(dtype=float)
        )
        gateway_rates[model_dir.name] = gateway_rate
        call_rates[model_dir.name] = call_rate
    if not gateway_rates:
        raise ValueError(f"No name_probe.csv files found in {results_dir}")

    gateway_df = pd.DataFrame(gateway_rates).T.fillna(0.0).sort_index()
    call_df = pd.DataFrame(call_rates).T.fillna(0.0).sort_index()
    gateway_df["Total"] = gateway_df.mean(axis=1)
    call_df["Total"] = call_df.mean(axis=1)
    gateway_df = gateway_df.sort_values("Total", ascending=False)
    call_df = call_df.sort_values("Total", ascending=False)
    return gateway_df, call_df


def plot_heatmap(data: pd.DataFrame, output: Path, title: str) -> None:
    """Plot and save a heatmap for the aggregated data."""
    plt.figure(figsize=(1 + 1.2 * data.shape[1], 1 + 0.6 * data.shape[0]))
    ax = sns.heatmap(data, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("Category")
    ax.set_ylabel("Model")
    ax.set_title(title)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def save_table(data: pd.DataFrame, output: Path) -> None:
    """Save the dataframe to a Markdown table with percentage values."""
    display_df = data.map(lambda x: f"{x:.0%}")
    output.write_text(display_df.to_markdown(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE_DIR / "experiments" / "results" / "exp0",
        help="Path to the directory containing model result folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "report",
        help="Directory for the generated heatmap images",
    )
    args = parser.parse_args()
    gateway_df, call_df = load_results(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_table(gateway_df, args.output_dir / "gateway_success_table.md")
    save_table(call_df, args.output_dir / "call_success_table.md")
    plot_heatmap(gateway_df, args.output_dir / "gateway_success_heatmap.png", "API Gateway Success Rate")
    plot_heatmap(call_df, args.output_dir / "call_success_heatmap.png", "Tool Call Success Rate")


if __name__ == "__main__":
    main()
