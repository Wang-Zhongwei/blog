"""Per-token gradient gate for the clip-family surrogates, on the (A_t, log rho_t) plane.

All of these objectives share the shape J = E[f(rho_t) A_t]. What is plotted here
is the gate alone,

    f'(rho_t; A_t),

the factor each method puts in front of the advantage. Since three of the four
switch on sign(A_t), the gate itself is a function of both coordinates -- hence
the semicolon -- but it carries no factor of A_t. Dropping that factor is what
makes the panels comparable: the shape of the trust region is the whole story,
and multiplying by A_t only tilts every panel the same way.

NOTE: neither this nor f' * A_t is the coefficient on grad log pi. Since
drho_t/dtheta = rho_t * grad log pi, that coefficient carries an extra factor of
rho_t:

    grad J = f'(rho_t; A_t) * rho_t * A_t * grad log pi.

The rho_t factor is common to every method, so it does not affect the comparison
between panels, but it does matter for the tail: f' -> 0 does not by itself mean
the update vanishes, and f' bounded away from 0 does not mean it diverges.

Three of the four define f' as an indicator, so g is just A_t masked to whatever
trust region the method allows:

    PPO/GRPO   f'(rho) = 1[A > 0, rho < 1 + eps]  or  1[A < 0, rho > 1 - eps]
    DAPO       same, with a wider upper bound: eps_h > eps_l
    SAO        f'(rho) = 1[1 - eps_l < rho < 1 + eps_h]   -- two-sided, sign-blind

PPO and DAPO clip one sign per side, so each dead zone is a quadrant wedge. SAO
clips on the ratio alone, so its dead zones are full-width horizontal bands.

SAPO breaks the pattern: f(rho) = sigma(tau(rho - 1)) * 4/tau, hence

    SAPO       f'(rho) = 4 sigma(z) (1 - sigma(z)),  z = tau(rho - 1)

which is 1 at rho = 1 (matching the unclipped slope of the others) and decays
smoothly, never reaching zero. There is no boundary and no dead zone -- only
attenuation -- so that panel is drawn with f' contours instead of a hatched mask.

GSPO is the same f as PPO under the substitution (rho_t, A_t) -> (rho_s, A_s):
the token ratio becomes the length-normalized sequence ratio, the token
advantage becomes the sequence advantage, and f is untouched. Its panel is
therefore the same picture on relabelled axes -- the reason it is drawn
separately (render_gspo) rather than added to METHODS is purely scale. Published
clip ranges are 3e-4 / 4e-4 against GRPO's 0.2, so a shared ratio axis would
either blank GSPO's panel or crush the other four into a hairline;
render_gspo_scale_contrast makes that ~600x gap the subject of its own figure.

Writes to this experiment's local figures/ directory. Promote to
assets/figures/<post-slug>/ by hand once a figure is final.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch

A_LIM = 1.0
LOG_RHO_LIM = 0.6

# Hyperparameters below are ILLUSTRATIVE, not tuned values. Both papers pick
# settings whose effect is real but visually microscopic on this plane, so the
# figures exaggerate the asymmetric parameter and hold the other fixed. Published
# values are kept alongside for reference.
EPS = 0.2

EPS_LOW, EPS_HIGH = 0.2, 0.5
PAPER_EPS_LOW, PAPER_EPS_HIGH = 0.2, 0.28  # DAPO, arXiv:2503.14476

# SAPO reports tau_pos = 1.0, tau_neg = 1.05, at which f' only falls to ~0.85 at
# the edge of this window and the pos/neg gap is under 0.02 -- indistinguishable.
# Scaling both up, at a 2x ratio, keeps the reported ordering tau_neg > tau_pos
# while making both the decay and the pos/neg asymmetry legible.
TAU_POS, TAU_NEG = 2.0, 4.0
PAPER_TAU_POS, PAPER_TAU_NEG = 1.0, 1.05  # SAPO, arXiv:2511.20347, Sec. 5.1

# GSPO is the one method here whose published clip range is used verbatim rather
# than exaggerated, because the magnitude IS the finding. Its ratio is the
# length-normalized sequence likelihood ratio s_i = (pi_theta(y_i|x) /
# pi_old(y_i|x))^(1/|y_i|), a geometric mean over tokens, so it concentrates far
# more tightly than any single rho_t -- and the clip range shrinks by three
# orders of magnitude to match. Exaggerating these to 0.2-scale would draw a
# panel identical to DAPO's and delete the only thing worth showing.
GSPO_EPS_LOW, GSPO_EPS_HIGH = 3e-4, 4e-4  # GSPO, arXiv:2507.18071, Sec. 5.1
GSPO_LOG_S_LIM = 1.0e-3

# Chart chrome and ink (light surface), from the reference palette.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"

# The gate is non-negative and lives in [0, 1], so this is a sequential ramp, not
# a diverging one: neutral gray at 0 so "no gradient" reads as nothing rather
# than as a hue, deepening blue toward the unclipped slope of 1.
BLUE_ARM = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
NEUTRAL = "#f0efec"
GATE = LinearSegmentedColormap.from_list("gate_sequential", [NEUTRAL] + BLUE_ARM)

LABEL_BOX = dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.92)
DASH = (0, (5, 4))
# f' iso-levels drawn on the SAPO panel. Each level appears as up to four
# segments -- one per quadrant -- since tau switches on sign(A_t) and the
# sigmoid has an upper and a lower branch.
SAPO_LEVELS = [0.9, 0.8, 0.7]

# The effective-coefficient figure needs a much wider ratio window than the
# heatmaps: SAPO's f' * rho does not break away from the clip family's shared
# rho ramp until |log rho| ~ 1.5, well outside LOG_RHO_LIM.
EFF_LOG_RHO_LIM = 2.0

# Distinct linestyles matter more than distinct hues here, because several of
# these curves coincide exactly over parts of the range (PPO and DAPO for
# A_t < 0; DAPO and SAO above the shared upper bound for A_t > 0).
METHOD_STYLE = {
    "ppo": dict(color=INK_PRIMARY, linestyle="-", linewidth=3.2),
    "dapo": dict(color="#5598e7", linestyle=(0, (6, 3)), linewidth=2.0),
    "sao": dict(color="#d03b3b", linestyle=(0, (2, 2.5)), linewidth=2.0),
    "sapo": dict(color="#184f95", linestyle="-", linewidth=2.6),
}


# --- f'(rho) per method -------------------------------------------------------


def ppo_fprime(adv, rho, eps=EPS):
    live = ((adv > 0) & (rho < 1.0 + eps)) | ((adv < 0) & (rho > 1.0 - eps))
    return live.astype(float)


def dapo_fprime(adv, rho, eps_l=EPS_LOW, eps_h=EPS_HIGH):
    live = ((adv > 0) & (rho < 1.0 + eps_h)) | ((adv < 0) & (rho > 1.0 - eps_l))
    return live.astype(float)


def gspo_fprime(adv, rho_s, eps_l=GSPO_EPS_LOW, eps_h=GSPO_EPS_HIGH):
    """dapo_fprime under (rho_t, A_t) -> (rho_s, A_s). The argument is the change.

    GSPO's objective (Eq. 5 of arXiv:2507.18071) is PPO's min/clip verbatim with
    the token pair replaced by the sequence pair, so the gate is algebraically
    identical -- this body is dapo_fprime's, and is kept separate only to carry
    GSPO's own epsilons. The paper writes rho_s as s_i and A_s as A-hat_i, and
    its left and right ranges differ (3e-4 vs 4e-4), so the wedges come out
    asymmetric about rho_s = 1 the way DAPO's are rather than PPO's.
    """
    live = ((adv > 0) & (rho_s < 1.0 + eps_h)) | ((adv < 0) & (rho_s > 1.0 - eps_l))
    return live.astype(float)


def sao_fprime(adv, rho, eps_l=EPS_LOW, eps_h=EPS_HIGH):
    del adv  # sign-blind: the ratio alone decides
    return ((rho > 1.0 - eps_l) & (rho < 1.0 + eps_h)).astype(float)


def sapo_fprime(adv, rho, tau_pos=TAU_POS, tau_neg=TAU_NEG):
    tau = np.where(adv > 0, tau_pos, tau_neg)
    sig = 1.0 / (1.0 + np.exp(-tau * (rho - 1.0)))
    return 4.0 * sig * (1.0 - sig)


def sapo_level_log_rho(level, tau):
    """log rho on each branch where f' equals `level`, or None if off the plane.

    4 s (1 - s) = L  with  s = sigma(z)  gives  s = (1 +/- sqrt(1 - L)) / 2, so
    z = +/- log((1 + r) / (1 - r)) with r = sqrt(1 - L), and rho = 1 + z / tau.
    The lower branch can fall at rho <= 0 for small tau, which has no log.
    """
    root = np.sqrt(1.0 - level)
    z = np.log((1.0 + root) / (1.0 - root))
    out = []
    for rho in (1.0 + z / tau, 1.0 - z / tau):
        out.append(np.log(rho) if rho > 0 else None)
    return out


METHODS = [
    {
        "key": "ppo",
        "name": "PPO / GRPO",
        "kind": "clip",
        "fprime": ppo_fprime,
        "params": rf"$\epsilon={EPS}$",
        "boundaries": [
            (np.log1p(EPS), r"$\log(1+\epsilon)$", "bottom", 0.02),
            (np.log1p(-EPS), r"$\log(1-\epsilon)$", "top", -0.02),
        ],
        "regions": [
            (A_LIM * 0.55, LOG_RHO_LIM * 0.72, "clipped\n$A_t>0,\\ \\rho_t>1+\\epsilon$"),
            (-A_LIM * 0.55, -LOG_RHO_LIM * 0.72, "clipped\n$A_t<0,\\ \\rho_t<1-\\epsilon$"),
        ],
        "short_regions": [
            (A_LIM * 0.55, LOG_RHO_LIM * 0.75, "clipped"),
            (-A_LIM * 0.55, -LOG_RHO_LIM * 0.75, "clipped"),
        ],
    },
    {
        "key": "dapo",
        "name": "DAPO",
        "kind": "clip",
        "fprime": dapo_fprime,
        "params": rf"$\epsilon_l={EPS_LOW},\ \epsilon_h={EPS_HIGH}$",
        "boundaries": [
            (np.log1p(EPS_HIGH), r"$\log(1+\epsilon_h)$", "bottom", 0.02),
            (np.log1p(-EPS_LOW), r"$\log(1-\epsilon_l)$", "top", -0.02),
        ],
        "regions": [
            (A_LIM * 0.55, LOG_RHO_LIM * 0.83, "clipped\n$A_t>0,\\ \\rho_t>1+\\epsilon_h$"),
            (-A_LIM * 0.55, -LOG_RHO_LIM * 0.72, "clipped\n$A_t<0,\\ \\rho_t<1-\\epsilon_l$"),
        ],
        "short_regions": [
            (A_LIM * 0.55, LOG_RHO_LIM * 0.85, "clipped"),
            (-A_LIM * 0.55, -LOG_RHO_LIM * 0.75, "clipped"),
        ],
    },
    {
        "key": "sao",
        "name": "SAO",
        "kind": "clip",
        "fprime": sao_fprime,
        "params": rf"$\epsilon_l={EPS_LOW},\ \epsilon_h={EPS_HIGH}$",
        "boundaries": [
            (np.log1p(EPS_HIGH), r"$\log(1+\epsilon_h)$", "bottom", 0.02),
            (np.log1p(-EPS_LOW), r"$\log(1-\epsilon_l)$", "top", -0.02),
        ],
        "regions": [
            (A_LIM * 0.42, LOG_RHO_LIM * 0.83, "clipped for both signs"),
            (A_LIM * 0.42, -LOG_RHO_LIM * 0.72, "clipped for both signs"),
        ],
        "short_regions": [
            (A_LIM * 0.45, LOG_RHO_LIM * 0.85, "clipped (both signs)"),
            (A_LIM * 0.45, -LOG_RHO_LIM * 0.75, "clipped (both signs)"),
        ],
    },
    {
        "key": "sapo",
        "name": "SAPO",
        "kind": "smooth",
        "fprime": sapo_fprime,
        "levels": SAPO_LEVELS,
        "tau_pos": TAU_POS,
        "tau_neg": TAU_NEG,
        "params": rf"$\tau_{{pos}}={TAU_POS:g},\ \tau_{{neg}}={TAU_NEG:g}$",
        "boundaries": [],
        "regions": [
            (-A_LIM * 0.5, LOG_RHO_LIM * 0.92, "no boundary — $f'$ decays, never 0"),
        ],
        "short_regions": [
            (-A_LIM * 0.42, LOG_RHO_LIM * 0.85, "no boundary"),
        ],
    },
]


# GSPO is kept out of METHODS on purpose. The four-panel strip and the
# effective-coefficient curves both share a single ratio axis, and GSPO's axis is
# ~600x narrower; dropping it into either would either blank the panel or force
# every other method into a hairline. It gets its own figure, on its own scale,
# plus an explicit side-by-side scale contrast.
GSPO = {
    "key": "gspo",
    "name": "GSPO",
    "kind": "clip",
    "fprime": gspo_fprime,
    "log_lim": GSPO_LOG_S_LIM,
    "params": r"$\epsilon_l=3\times10^{-4},\ \epsilon_h=4\times10^{-4}$",
    "xlabel": r"sequence advantage  $A_s$",
    "ylabel": r"log sequence ratio  $\log \rho_s$",
    "cbar_label": r"gate  $f'(\rho_s;\, A_s)$",
    "sci_y": True,
    "note": (
        "Paper values, not exaggerated. Note the vertical scale: "
        r"$\pm 10^{-3}$, against $\pm 0.6$ in Figures 2–5."
    ),
    "boundaries": [
        (np.log1p(GSPO_EPS_HIGH), r"$\log(1+\epsilon_h)$", "bottom", GSPO_LOG_S_LIM * 0.033),
        (np.log1p(-GSPO_EPS_LOW), r"$\log(1-\epsilon_l)$", "top", -GSPO_LOG_S_LIM * 0.033),
    ],
    "regions": [
        (
            A_LIM * 0.55,
            GSPO_LOG_S_LIM * 0.78,
            "clipped\n$A_s>0,\\ \\rho_s>1+\\epsilon_h$",
        ),
        (
            -A_LIM * 0.55,
            -GSPO_LOG_S_LIM * 0.68,
            "clipped\n$A_s<0,\\ \\rho_s<1-\\epsilon_l$",
        ),
    ],
    "short_regions": [
        (A_LIM * 0.55, GSPO_LOG_S_LIM * 0.8, "clipped"),
        (-A_LIM * 0.55, -GSPO_LOG_S_LIM * 0.7, "clipped"),
    ],
}


# --- rendering ----------------------------------------------------------------


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
            # Hatch style is read at artist-construction time, so it has to be set
            # here rather than passed to contourf.
            "hatch.color": AXIS,
            "hatch.linewidth": 0.7,
        }
    )


def grids(log_lim=LOG_RHO_LIM):
    adv_grid = np.linspace(-A_LIM, A_LIM, 801)
    log_rho_grid = np.linspace(-log_lim, log_lim, 801)
    adv, log_rho = np.meshgrid(adv_grid, log_rho_grid)
    return adv, log_rho, np.exp(log_rho)


def draw_panel(ax, method, adv, log_rho, rho, *, short_labels=False, boundary_labels=True):
    fprime = method["fprime"](adv, rho)

    mesh = ax.pcolormesh(
        adv,
        log_rho,
        fprime,
        cmap=GATE,
        norm=Normalize(vmin=0.0, vmax=1.0),
        shading="auto",
        rasterized=True,
    )

    if method["kind"] == "clip":
        # Secondary encoding: hatch the zero-gradient region so "nothing happens
        # here" is not carried by color alone.
        dead = np.ma.masked_where(fprime > 0, np.ones_like(fprime))
        ax.contourf(adv, log_rho, dead, levels=[0.5, 1.5], colors="none", hatches=["//////"])
        for level, label, va, offset in method["boundaries"]:
            ax.axhline(level, color=INK_SECONDARY, linestyle=DASH, linewidth=1.6)
            if boundary_labels:
                ax.text(
                    -A_LIM * 0.97,
                    level + offset,
                    label,
                    ha="left",
                    va=va,
                    fontsize=11,
                    color=INK_SECONDARY,
                    bbox=LABEL_BOX,
                )
    else:
        # No boundary to draw -- show the decay itself as f' iso-levels. Each is
        # drawn per half-plane, since tau switches on sign(A_t), and labelled on
        # the line so every visible segment is identified.
        for level in method["levels"]:
            for x0, x1, tau in ((0.0, A_LIM, method["tau_pos"]), (-A_LIM, 0.0, method["tau_neg"])):
                for value in sapo_level_log_rho(level, tau):
                    if value is None or abs(value) > LOG_RHO_LIM:
                        continue
                    ax.plot(
                        [x0, x1],
                        [value, value],
                        color=INK_SECONDARY,
                        linestyle=DASH,
                        linewidth=1.3,
                    )
                    if boundary_labels:
                        ax.text(
                            0.5 * (x0 + x1),
                            value,
                            rf"$f'={level:g}$",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color=INK_SECONDARY,
                            bbox=LABEL_BOX,
                        )

    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.axvline(0.0, color=AXIS, linewidth=1.0)

    for x, y, text in method["short_regions" if short_labels else "regions"]:
        ax.annotate(
            text,
            xy=(x, y),
            ha="center",
            va="center",
            fontsize=10.5 if short_labels else 11,
            color=INK_SECONDARY,
            bbox=LABEL_BOX,
        )

    ax.set_xlim(-A_LIM, A_LIM)
    ax.set_ylim(-method.get("log_lim", LOG_RHO_LIM), method.get("log_lim", LOG_RHO_LIM))
    ax.set_xticks(np.linspace(-A_LIM, A_LIM, 5))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return mesh


def style_colorbar(cbar, label=r"gate  $f'(\rho_t;\, A_t)$"):
    cbar.set_label(label, color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=INK_MUTED, labelcolor=INK_MUTED)


def render_single(method, adv, log_rho, rho, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    mesh = draw_panel(ax, method, adv, log_rho, rho)
    ax.set_xlabel(method.get("xlabel", r"advantage  $A_t$"))
    ax.set_ylabel(method.get("ylabel", r"log importance ratio  $\log \rho_t$"))
    if method.get("sci_y"):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_color(INK_MUTED)
    if "note" in method:
        ax.text(
            0.5,
            -0.205,
            method["note"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            color=INK_MUTED,
        )
    ax.set_title(
        rf"{method['name']}   ({method['params']})",
        color=INK_PRIMARY,
        fontsize=14,
        pad=12,
    )
    style_colorbar(
        fig.colorbar(mesh, ax=ax, pad=0.02),
        method.get("cbar_label", r"gate  $f'(\rho_t;\, A_t)$"),
    )
    fig.tight_layout()
    save(fig, out_dir / f"{method['key']}-gradient-weight-heatmap")
    plt.close(fig)


def render_comparison(adv, log_rho, rho, out_dir):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(17.5, 4.4), sharey=True)
    for ax, method in zip(axes, METHODS):
        mesh = draw_panel(ax, method, adv, log_rho, rho, short_labels=True, boundary_labels=False)
        ax.set_xlabel(r"advantage  $A_t$")
        ax.set_title(
            f"{method['name']}   ({method['params']})",
            color=INK_PRIMARY,
            fontsize=13,
            pad=10,
        )
    axes[0].set_ylabel(r"log importance ratio  $\log \rho_t$")
    style_colorbar(fig.colorbar(mesh, ax=axes, pad=0.012, fraction=0.025))
    save(fig, out_dir / "clip-family-gradient-weight-heatmaps")
    plt.close(fig)


def render_effective_coefficient(out_dir):
    """f'(rho; A) * rho over a wide ratio window, at |A_t| = 1.

    The heatmaps deliberately strip the shared rho_t factor. This figure puts it
    back, because for SAPO the product is where the real behaviour lives: the
    sech^2 decay and the linear growth in rho nearly cancel over the heatmaps'
    +/-0.6 window (f' rho falls only from 1.00 to 0.99), so SAPO's attenuation is
    invisible at that zoom. Widened to +/-2 the cancellation breaks and the
    coefficient collapses to zero -- smoothly, where the clip family falls off a
    cliff.

    Two panels because f' switches on sign(A_t). Note that PPO and DAPO coincide
    exactly on the A_t < 0 panel: DAPO only moves the upper bound.
    """
    log_rho = np.linspace(-EFF_LOG_RHO_LIM, EFF_LOG_RHO_LIM, 4001)
    rho = np.exp(log_rho)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, sign, title in (
        (axes[0], 1.0, r"$A_t > 0$"),
        (axes[1], -1.0, r"$A_t < 0$"),
    ):
        adv = np.full_like(rho, sign)
        for method in METHODS:
            style = METHOD_STYLE[method["key"]]
            ax.plot(
                log_rho,
                method["fprime"](adv, rho) * rho,
                label=method["name"],
                **style,
            )
        ax.axvline(0.0, color=AXIS, linewidth=1.0)
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        # Mark the heatmaps' window so the two figures can be read together.
        ax.axvspan(-LOG_RHO_LIM, LOG_RHO_LIM, color=NEUTRAL, zorder=0)
        ax.set_xlabel(r"log importance ratio  $\log \rho_t$")
        ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=10)
        ax.set_xlim(-EFF_LOG_RHO_LIM, EFF_LOG_RHO_LIM)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(r"effective weight  $f'(\rho_t;A_t)\,\rho_t$")
    axes[0].set_ylim(0.0, 2.1)
    # Top-right of the A_t > 0 panel is empty -- every curve has decayed by then.
    axes[0].text(
        1.25,
        1.95,
        "shaded: window\nof Figures 2–5",
        ha="center",
        va="top",
        fontsize=9.5,
        color=INK_MUTED,
    )
    # The headline of this figure: for A_t < 0 neither PPO nor DAPO imposes an
    # upper bound, so past SAO's cliff their weight just keeps growing with rho.
    axes[1].annotate(
        "PPO and DAPO coincide, and keep growing:\n"
        r"no upper bound when $A_t<0$",
        xy=(0.78, 2.02),
        xytext=(-1.92, 1.62),
        fontsize=10,
        color=INK_SECONDARY,
        bbox=LABEL_BOX,
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.1),
    )
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout()
    save(fig, out_dir / "effective-coefficient-curves")
    plt.close(fig)


def render_gspo(out_dir):
    adv, log_s, s = grids(GSPO_LOG_S_LIM)
    render_single(GSPO, adv, log_s, s, out_dir)


def render_gspo_scale_contrast(out_dir):
    """PPO/GRPO and GSPO side by side, each on the ratio scale its epsilon implies.

    The gates are the same function -- both panels are the two-wedge picture of
    Figure 2/3. Putting them next to each other is only worth a figure because
    the axes differ by ~600x, and that difference is not cosmetic: s_i is a
    geometric mean over |y_i| token ratios, so it sits far closer to 1 than any
    individual rho_t, and a 0.2-wide window around it would never bind. The
    connector marks where the right panel lives inside the left one, which is a
    band roughly 7e-4 tall -- thinner than the line drawn to point at it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    ppo = METHODS[0]
    adv, log_rho, rho = grids()
    draw_panel(axes[0], ppo, adv, log_rho, rho, boundary_labels=False)
    axes[0].set_ylabel(r"log importance ratio  $\log \rho_t$")
    axes[0].set_xlabel(r"advantage  $A_t$")
    axes[0].set_title(
        rf"PPO / GRPO — token ratio   ({ppo['params']})",
        color=INK_PRIMARY,
        fontsize=13,
        pad=10,
    )

    adv_s, log_s, s = grids(GSPO_LOG_S_LIM)
    mesh = draw_panel(axes[1], GSPO, adv_s, log_s, s, boundary_labels=False)
    axes[1].set_ylabel(r"log sequence ratio  $\log \rho_s$")
    axes[1].set_xlabel(r"sequence advantage  $A_s$")
    axes[1].set_title(
        rf"GSPO — sequence ratio   ({GSPO['params']})",
        color=INK_PRIMARY,
        fontsize=13,
        pad=10,
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[1].yaxis.get_offset_text().set_color(INK_MUTED)

    # The right panel's full extent, drawn to scale inside the left panel. At
    # +/-1e-3 against +/-0.6 this is 1/600 of the axis height, so it is a line.
    axes[0].axhspan(
        -GSPO_LOG_S_LIM,
        GSPO_LOG_S_LIM,
        color="#d03b3b",
        linewidth=0,
        zorder=5,
    )
    axes[0].annotate(
        "the whole right panel,\ndrawn to scale here",
        xy=(-A_LIM * 0.45, 0.0),
        xytext=(-A_LIM * 0.5, LOG_RHO_LIM * 0.55),
        fontsize=10,
        color="#8f2020",
        ha="center",
        bbox=LABEL_BOX,
        arrowprops=dict(arrowstyle="->", color="#d03b3b", linewidth=1.2),
        zorder=6,
    )

    for x in (-GSPO_LOG_S_LIM, GSPO_LOG_S_LIM):
        fig.add_artist(
            ConnectionPatch(
                xyA=(A_LIM, x),
                coordsA=axes[0].transData,
                xyB=(-A_LIM, np.sign(x) * GSPO_LOG_S_LIM),
                coordsB=axes[1].transData,
                color=AXIS,
                linewidth=1.0,
                linestyle=DASH,
            )
        )

    fig.tight_layout()
    style_colorbar(fig.colorbar(mesh, ax=axes, pad=0.012, fraction=0.022), r"gate  $f'$")
    save(fig, out_dir / "ppo-vs-gspo-scale-contrast")
    plt.close(fig)


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    print(f"wrote {stem.name}.{{png,pdf}}")


def main():
    apply_theme()
    adv, log_rho, rho = grids()
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        render_single(method, adv, log_rho, rho, out_dir)
    render_comparison(adv, log_rho, rho, out_dir)
    render_effective_coefficient(out_dir)
    render_gspo(out_dir)
    render_gspo_scale_contrast(out_dir)


if __name__ == "__main__":
    main()
