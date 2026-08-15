"""Schematic of the PPO/GRPO clipped surrogate in the (A_t, log rho_t) plane.

This is the qualitative companion to gradient_weight_heatmaps.py: no continuous
field and no colorbar, just the four regions the clip carves out, each labelled
with what J looks like there and which way gradient ascent pushes rho_t.

    A_t > 0:  J = min(rho_t, 1 + eps) A_t
              rho_t < 1 + eps  ->  J = rho_t A_t,      dJ/drho_t = A_t > 0, push rho up
              rho_t > 1 + eps  ->  J = (1 + eps) A_t,  gradient 0, clipped

    A_t < 0:  J = max(rho_t, 1 - eps) A_t
              rho_t > 1 - eps  ->  J = rho_t A_t,      dJ/drho_t = A_t < 0, push rho down
              rho_t < 1 - eps  ->  J = (1 - eps) A_t,  gradient 0, clipped
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A_LIM = 1.0
LOG_RHO_LIM = 0.6
EPS = 0.2

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"

RED_FILL = "#f6b3ad"
RED_DEEP = "#8f2020"
BLUE_FILL = "#9ec5f4"
BLUE_DEEP = "#184f95"
NEUTRAL = "#f0efec"

DASH = (0, (5, 4))
UP = np.log1p(EPS)
DOWN = np.log1p(-EPS)


def apply_theme():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.edgecolor": AXIS,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.color": AXIS,
            "hatch.linewidth": 0.7,
        }
    )


def region(ax, x0, x1, y0, y1, color, hatch=None):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=color,
            edgecolor=AXIS if hatch else "none",
            linewidth=0.0,
            hatch=hatch,
            zorder=0,
        )
    )


def block(ax, x, y, lines, color=INK_PRIMARY, ha="center"):
    """A stacked label: title line in the region colour, detail lines muted."""
    ax.text(
        x,
        y,
        lines,
        ha=ha,
        va="center",
        fontsize=12.5,
        color=color,
        linespacing=1.6,
        zorder=4,
    )


def draw(ax):
    # --- the four regions -------------------------------------------------
    region(ax, 0, A_LIM, -LOG_RHO_LIM, UP, RED_FILL)  # A>0, live
    region(ax, 0, A_LIM, UP, LOG_RHO_LIM, NEUTRAL, hatch="//////")  # A>0, clipped
    region(ax, -A_LIM, 0, DOWN, LOG_RHO_LIM, BLUE_FILL)  # A<0, live
    region(ax, -A_LIM, 0, -LOG_RHO_LIM, DOWN, NEUTRAL, hatch="//////")  # A<0, clipped

    # --- clip boundaries, each only over the half-plane where it bites ----
    ax.plot([0, A_LIM], [UP, UP], color=RED_DEEP, linestyle=DASH, linewidth=1.8, zorder=3)
    ax.plot([-A_LIM, 0], [DOWN, DOWN], color=BLUE_DEEP, linestyle=DASH, linewidth=1.8, zorder=3)
    ax.text(
        A_LIM * 0.985,
        UP + 0.022,
        r"$\log(1+\epsilon)$",
        ha="right",
        va="bottom",
        fontsize=11.5,
        color=RED_DEEP,
        zorder=4,
    )
    ax.text(
        -A_LIM * 0.985,
        DOWN - 0.022,
        r"$\log(1-\epsilon)$",
        ha="left",
        va="top",
        fontsize=11.5,
        color=BLUE_DEEP,
        zorder=4,
    )

    # --- what J is, region by region --------------------------------------
    block(
        ax,
        A_LIM * 0.42,
        (UP - LOG_RHO_LIM) / 2 + 0.06,
        "$J \\sim \\rho_t A_t$\n"
        "$\\partial J/\\partial\\rho_t = A_t > 0$\n"
        "$\\rho_t\\uparrow \\Rightarrow J\\uparrow$",
        color=RED_DEEP,
    )
    block(
        ax,
        A_LIM * 0.46,
        (UP + LOG_RHO_LIM) / 2,
        "$J \\sim (1+\\epsilon)\\,A_t$\ngradient $= 0$",
        color=INK_SECONDARY,
    )
    block(
        ax,
        -A_LIM * 0.42,
        (DOWN + LOG_RHO_LIM) / 2 - 0.06,
        "$J \\sim \\rho_t A_t$\n"
        "$\\partial J/\\partial\\rho_t = A_t < 0$\n"
        "$\\rho_t\\downarrow \\Rightarrow J\\uparrow$",
        color=BLUE_DEEP,
    )
    block(
        ax,
        -A_LIM * 0.46,
        (DOWN - LOG_RHO_LIM) / 2,
        "$J \\sim (1-\\epsilon)\\,A_t$\ngradient $= 0$",
        color=INK_SECONDARY,
    )

    # --- axes -------------------------------------------------------------
    ax.axhline(0.0, color=AXIS, linewidth=1.0, zorder=2)
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.2, zorder=2)
    ax.set_xlim(-A_LIM, A_LIM)
    ax.set_ylim(-LOG_RHO_LIM, LOG_RHO_LIM)
    ax.set_xticks(np.linspace(-A_LIM, A_LIM, 5))
    ax.set_yticks([-0.4, DOWN, 0.0, UP, 0.4])
    ax.set_yticklabels(["-0.4", "", "0", "", "0.4"])
    ax.set_xlabel(r"advantage  $A_t$")
    ax.set_ylabel(r"$\log \rho_t$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.set_title(
        f"PPO/GRPO   ($\\epsilon={EPS}$)",
        fontsize=14,
        color=INK_PRIMARY,
        pad=14,
    )

def main():
    apply_theme()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    draw(ax)
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ppo-clip-schematic.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/ppo-clip-schematic.png")


if __name__ == "__main__":
    main()
