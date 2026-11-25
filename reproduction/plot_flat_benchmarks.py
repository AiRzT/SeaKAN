"""Render the numerical curves of manuscript Figures 3–4 into eight PDF panels."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

LEGACY = Path(__file__).resolve().parent / "legacy"
sys.path.insert(0, str(LEGACY / "src"))
from mpm_kan_fig7.extracted import load_extracted_figure7


def depth_panels(output):
    data = pd.read_csv(LEGACY / "src/outputs/figure7/figure7_profiles.csv")
    references, _ = load_extracted_figure7(LEGACY / "figure7_data")
    cases = ["saturated_fine", "saturated_coarse", "unsaturated_fine", "unsaturated_coarse"]
    colors = ["#2766A0", "#AD5B20", "#258160", "#7B59A7"]
    for field, slug, label in [
        ("pore_pressure", "pressure", r"$|p|/p_0$"),
        ("sigma_x_eff", "sigma_x", r"$|\sigma'_x|/p_0$"),
        ("sigma_z_eff", "sigma_z", r"$|\sigma'_z|/p_0$"),
        ("tau_xz", "tau_xz", r"$|\tau_{xz}|/p_0$"),
    ]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for case, color in zip(cases, colors):
            profile = data[(data.case == case) & (data.source == "mpm_kan_pinn")]
            ax.plot(profile[field], profile.z_over_h, color=color)
            for key, style in (("analytical", "o"), ("numerical", "--")):
                series = references[case][field][key]
                idx = series["z_over_h"].argsort()
                ax.plot(series["value"][idx], series["z_over_h"][idx], style,
                        color=color, linewidth=1, markersize=2.4, fillstyle="none")
        ax.set(xlabel=label, ylabel=r"$z/h$", ylim=(-1, 0))
        ax.grid(alpha=0.18)
        fig.savefig(output / f"benchmark_figure7_{slug}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 0.8))
    ax.set_axis_off()
    handles = [Line2D([], [], color=c, label=s.replace("_", " ").capitalize()) for s, c in zip(cases, colors)]
    handles += [Line2D([], [], color="black", linestyle=style, marker=marker, label=label)
                for label, style, marker in [("SeaKAN", "-", ""), ("Zhu et al. numerical", "--", ""), ("Analytical", "none", "o")]]
    ax.legend(handles=handles, loc="center", ncol=4, frameon=False)
    fig.savefig(output / "benchmark_figure7_legend.pdf", bbox_inches="tight")
    plt.close(fig)


def additional_panels(output):
    columns = [
        (8, None, "normalized_pore_pressure", "z_over_h", r"$|p|/p_0$", "$z/h$", "benchmark_figure8"),
        (13, "pressure", "normalized_amplitude", "z_m", r"$|p|/p_0$", "$z$ (m)", "benchmark_figure13_pressure"),
        (13, "vertical_effective_stress", "normalized_amplitude", "z_m", r"$|\sigma'_z|/p_0$", "$z$ (m)", "benchmark_figure13_stress"),
    ]
    styles = {"analytical": ("-", "Analytical solution", "0.45"),
              "numerical": ("--", "Zhu et al. numerical", "#B3541D"),
              "experimental": ("o", "Experiment", "black"),
              "mpm_kan_pinn": ("-", "SeaKAN", "#24639C")}
    for n, quantity, xcol, ycol, xlabel, ylabel, filename in columns:
        data = pd.read_csv(LEGACY / f"figure{n}_mpm_kan/artifacts/figure{n}_profiles.csv")
        if quantity:
            data = data[data.quantity == quantity]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for source, series in data.groupby("source"):
            name = source.removeprefix("digitized_").removeprefix("paper_")
            if name in styles:
                style, label, color = styles[name]
                series = series.sort_values(ycol)
                ax.plot(series[xcol], series[ycol], style, color=color, label=label, markersize=4)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(frameon=False)
        ax.grid(alpha=0.18)
        fig.savefig(output / (filename + ".pdf"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=Path("runs/flat_panels"))
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    depth_panels(args.output)
    additional_panels(args.output)
