#!/usr/bin/env python3
"""Plot compact temporal mean-response arrows and distributional shifts."""

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def distribution(value, support, context):
    """Accept probabilities or percentages and normalize to unit mass."""
    d = json.loads(value) if isinstance(value, str) else value
    if not isinstance(d, dict):
        raise ValueError(f"{context}: distribution must be a JSON object")

    numeric = {float(k): float(v) for k, v in d.items()}
    if len(numeric) != len(d) or set(numeric) != set(support):
        raise ValueError(
            f"{context}: response categories do not match valid_options"
        )

    p = np.array([numeric[x] for x in support])
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{context}: invalid probability")

    total = p.sum()
    if not (
        np.isclose(total, 1, atol=0.005, rtol=0)
        or np.isclose(total, 100, atol=0.5, rtol=0)
    ):
        raise ValueError(
            f"{context}: total {total:g} is not approximately 1 or 100"
        )
    return p / total


def normalized_w1(p, q, x):
    return float(
        np.sum(np.abs(np.cumsum(p - q)[:-1]) * np.diff(x))
        / (x[-1] - x[0])
    )


def read_inputs(model_path, gold_path):
    model, gold = {}, {}

    with open(model_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["pair_id"], r["wave_type"])
            if key in model:
                raise ValueError(
                    f"Duplicate model key: {key}; "
                    "use one model/condition per CSV"
                )
            model[key] = r

    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["pair_id"], r["wave_type"])
            if key in gold:
                raise ValueError(f"Duplicate gold key: {key}")
            gold[key] = r

    missing = set(model) - set(gold)
    if missing:
        raise ValueError(
            f"{len(missing)} model keys lack gold; "
            f"first: {sorted(missing)[0]}"
        )
    return model, gold


def prepare_pairs(model, gold, args):
    candidates = sorted({k[0] for k in model})
    for pid in candidates:
        for wave in ("older", "current"):
            if (pid, wave) not in model:
                raise ValueError(f"{pid}: missing model {wave} row")

    if args.pair_id:
        candidates = list(dict.fromkeys(args.pair_id))
    else:
        candidates = [
            p for p in candidates
            if gold[(p, "older")]["variable"] == args.variable
        ]

        if args.countries:
            candidates = [
                p for p in candidates
                if gold[(p, "older")]["country_name"] in args.countries
            ]
            candidates.sort(
                key=lambda p: (
                    args.countries.index(
                        gold[(p, "older")]["country_name"]
                    ),
                    p,
                )
            )

            absent = set(args.countries) - {
                gold[(p, "older")]["country_name"] for p in candidates
            }
            if absent:
                raise ValueError(
                    f"No matching pairs for countries: {sorted(absent)}"
                )
        else:
            candidates.sort(
                key=lambda p: (gold[(p, "older")]["country_name"], p)
            )
            candidates = candidates[:args.n]

    if not candidates:
        raise ValueError("No pairs match the selected variable/countries")

    pairs = []
    for pid in candidates:
        for wave in ("older", "current"):
            if (pid, wave) not in model or (pid, wave) not in gold:
                raise ValueError(f"{pid}: missing {wave} row")

            status = model[(pid, wave)].get("status", "ok").lower()
            if status not in ("ok", "success", "successful"):
                raise ValueError(f"{pid}/{wave}: status={status}")

        go, gn = [gold[(pid, w)] for w in ("older", "current")]
        x = np.array(sorted(float(v) for v in go["valid_options"]))

        if (
            len(x) < 2
            or not np.isfinite(x).all()
            or (np.diff(x) <= 0).any()
        ):
            raise ValueError(f"{pid}: invalid ordered support")

        if list(x) != sorted(float(v) for v in gn["valid_options"]):
            raise ValueError(f"{pid}: response scales differ across periods")

        if go["response_options"] != gn["response_options"]:
            raise ValueError(f"{pid}: response labels differ across periods")

        ds = [
            distribution(r["gold_distribution"], x, pid)
            for r in (go, gn)
        ]
        ds += [
            distribution(
                model[(pid, w)]["predicted_distribution_json"], x, pid
            )
            for w in ("older", "current")
        ]

        pairs.append({
            "pair_id": pid,
            "metadata": go,
            "current": gn,
            "x": x,
            "means": [float(p @ x) for p in ds],
            "human_w1": normalized_w1(ds[0], ds[1], x),
            "model_w1": normalized_w1(ds[2], ds[3], x),
        })

    first = pairs[0]
    if any(
        p["metadata"]["variable"] != first["metadata"]["variable"]
        or p["metadata"]["response_options"]
        != first["metadata"]["response_options"]
        or not np.array_equal(p["x"], first["x"])
        for p in pairs
    ):
        raise ValueError("Use one question and common response coding per plot")

    return pairs


def draw(pairs, args):
    human = "#087F8C"
    model_color = "#C65D21"
    ink = "#202C3C"
    secondary = "#485463"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": "#C8D1D9",
        "text.color": ink,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    n = len(pairs)
    first = pairs[0]["metadata"]

    # Shared period labels are used only when all rows have the same periods.
    period_pairs = {
        (
            str(p["metadata"]["time_period"]),
            str(p["current"]["time_period"]),
        )
        for p in pairs
    }
    shared_periods = len(period_pairs) == 1

    if shared_periods:
        older_period, current_period = next(iter(period_pairs))
        older_label = f"Older: {older_period}"
        current_label = f"Current: {current_period}"
    else:
        older_label = "Older"
        current_label = "Current"

    fig = plt.figure(
        figsize=(12.8, 1.8 + 1.15 * n),
        facecolor="white",
    )
    gs = fig.add_gridspec(
        n, 2,
        width_ratios=[4.7, 1.25],
        left=0.215,
        right=0.97,
        top=0.78,
        bottom=0.19,
        hspace=0.12,
        wspace=0.14,
    )

    # Keep the survey question as the only title.
    question = first["question_text"].strip()
    question = question[:1].upper() + question[1:]
    fig.text(
        0.055, 0.965,
        "\n".join(textwrap.wrap(question, 105)),
        fontsize=16,
        weight="bold",
        va="top",
    )

    fig.legend(
        handles=[
            Line2D(
                [], [], color=human, lw=3,
                label="Human survey",
            ),
            Line2D(
                [], [], color=model_color, lw=3,
                label="Model",
            ),
            Line2D(
                [], [],
                color=ink,
                marker="o",
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.5,
                linestyle="none",
                label=older_label,
            ),
            Line2D(
                [], [],
                color=ink,
                marker="o",
                markersize=7,
                linestyle="none",
                label=current_label,
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.055, 0.895),
        frameon=False,
        ncol=4,
        fontsize=12,
        columnspacing=1.5,
        handlelength=1.6,
    )

    for i, p in enumerate(pairs):
        ax = fig.add_subplot(gs[i, 0])
        stats = fig.add_subplot(gs[i, 1])
        x = p["x"]
        span = x[-1] - x[0]

        ax.set_xlim(x[0] - 0.025 * span, x[-1] + 0.025 * span)
        ax.set_ylim(-0.48, 1.65)
        ax.set_xticks(x)
        ax.grid(axis="x", color="#E8EDF1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks([])
        ax.tick_params(
            axis="x",
            length=0,
            labelbottom=i == n - 1,
            pad=8,
            colors=secondary,
        )
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)

        for j, (y, color) in enumerate(
            ((1, human), (0, model_color))
        ):
            old, new = p["means"][2 * j:2 * j + 2]

            if abs(new - old) > 1e-10:
                ax.annotate(
                    "",
                    xy=(new, y),
                    xytext=(old, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "lw": 2.8,
                        "color": color,
                        "mutation_scale": 17,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )

            ax.plot(
                old, y, "o",
                ms=8, mfc="white", mec=color,
                mew=1.8, zorder=4,
            )
            ax.plot(
                new, y, "o",
                ms=4.5, color=color, zorder=5,
            )

            if abs(new - old) < 0.12 * span:
                label = f"{old:.2f} → {new:.2f}"
                if abs(new - old) < 1e-10:
                    label += " (same mean)"
                ax.text(
                    (old + new) / 2, y + 0.24,
                    label,
                    ha="center",
                    color=color,
                    fontsize=12,
                )
            else:
                ax.text(
                    old, y + 0.24, f"{old:.2f}",
                    ha="center", color=color, fontsize=12,
                )
                ax.text(
                    new, y + 0.24, f"{new:.2f}",
                    ha="center", color=color, fontsize=12,
                )

        # Country names only when periods are shared across rows.
        country_y = 0.50 if shared_periods else 0.70
        ax.text(
            -0.035, country_y,
            p["metadata"]["country_name"],
            transform=ax.transAxes,
            fontsize=14,
            weight="bold",
            ha="right",
            va="center",
        )

        if not shared_periods:
            ax.text(
                -0.035, 0.30,
                f"{p['metadata']['time_period']}\n"
                f"→ {p['current']['time_period']}",
                transform=ax.transAxes,
                fontsize=10,
                color=secondary,
                ha="right",
                va="center",
            )

        stats.axis("off")
        if i == 0:
            stats.text(
                0, 1.20, "Distribution shift",
                fontsize=13, weight="bold",
            )
            stats.text(
                0, 1.01, "Normalized Wasserstein",
                fontsize=10, color=secondary,
            )

        # Align shift values with their corresponding arrow rows.
        human_y = (1 + 0.48) / (1.65 + 0.48)
        model_y = 0.48 / (1.65 + 0.48)

        stats.text(
            0, human_y,
            f"Human   {p['human_w1']:.3f}",
            fontsize=13, color=human, va="center",
        )
        stats.text(
            0, model_y,
            f"Model    {p['model_w1']:.3f}",
            fontsize=13, color=model_color, va="center",
        )

        if i == n - 1:
            opts = first["response_options"]
            if len(x) <= 5:
                ax.set_xticklabels(
                    [
                        f"{v:g}\n" + textwrap.fill(
                            opts.get(str(int(v)), opts.get(str(v), "")),
                            17,
                        )
                        for v in x
                    ],
                    fontsize=11,
                )
            ax.set_xlabel(
                "Mean response on the original answer scale",
                labelpad=10,
                fontsize=13,
            )

    # No footer: the paper caption explains the arrows.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf"):
        fig.savefig(
            str(out) + "." + ext,
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)

    summary = {
        "model_file": str(args.model),
        "gold_file": str(args.gold),
        "selection": "Selected examples; no knowledge gate",
        "rows": [
            {
                k: v for k, v in p.items()
                if k not in ("metadata", "current", "x")
            }
            for p in pairs
        ],
    }
    Path(str(out) + "_values.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)

    # Retained for compatibility; the figure always uses "Model".
    parser.add_argument("--model-label", default="Model")

    parser.add_argument("--variable")
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--pair-id", action="append")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument(
        "--out",
        default="temporal_arrows",
        help="Output prefix without extension",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")

    if not args.variable and not args.pair_id:
        args.variable = "A006"
        if not args.countries:
            args.countries = [
                "United States", "Australia", "India", "South Korea"
            ]

    model, gold = read_inputs(args.model, args.gold)
    draw(prepare_pairs(model, gold, args), args)


if __name__ == "__main__":
    main()