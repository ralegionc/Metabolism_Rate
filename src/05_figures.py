"""
05_figures.py -- Publication-quality dark-theme figures.

Six figures, in the order the argument runs:
  1. The clocks work            -- predicted vs chronological age, real dolphin blood
  2. The allometry is a mirage  -- aging rate vs body mass, then vs sampled age span
  3. Outliers mostly dissolve   -- naive vs corrected residuals, with the famous species
  4. The data is real           -- lifespan-mass allometry and longevity quotient
  5. Signal does not generalise -- state importances vs leave-one-Order-out failure
  6. Where the clock CpGs live  -- genomic context enrichment and top genes
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BG = "#0d0e1a"
PANEL = "#151729"
FG = "#e6e6f0"
MUTED = "#8b8ba7"
GRID = "#252842"
INDIGO = "#6366f1"
VIOLET = "#a78bfa"
CYAN = "#22d3ee"
AMBER = "#fbbf24"
ROSE = "#fb7185"
EMERALD = "#34d399"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": ["DejaVu Sans"], "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.4, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 130,
})


def style(ax, title=None, sub=None):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, color=FG, pad=14 if sub else 8, loc="left")
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5,
                va="bottom")
    return ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    wrote figures/{name}")


# ---------------------------------------------------------------------------
def fig1_clocks():
    pub = pd.read_csv(PROC / "clock_predictions_published.csv")
    scr = pd.read_csv(PROC / "clock_predictions_scratch.csv")
    stats_j = json.loads((RES / "clock_validation.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    lim = [0, max(pub.Age.max(), pub.DNAmAge_clock2.max(),
                  scr.PredAge_scratch.max()) * 1.08]

    for ax, (x, y, lbl, col, st) in zip(axes, [
        (pub.Age, pub.DNAmAge_clock2, "Published universal clock 2", VIOLET,
         stats_j["published_clocks"]["clock2"]),
        (scr.Age, scr.PredAge_scratch, "Trained here, leave-one-out CV", CYAN,
         stats_j["from_scratch_clock"]),
    ]):
        ax.plot(lim, lim, "--", color=MUTED, lw=1, alpha=0.7, zorder=1)
        ax.scatter(x, y, s=46, c=col, alpha=0.85, edgecolors=BG, lw=0.6, zorder=3)
        r = st.get("pearson_r", st.get("loo_pearson_r"))
        mae = st.get("median_AE", st.get("loo_median_AE_years"))
        ncg = st.get("n_cpgs_used", st.get("n_cpgs_selected"))
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Chronological age (years)")
        ax.set_ylabel("Predicted age (years)")
        style(ax, lbl, f"r = {r:.3f}   median error = {mae:.2f} y   {ncg} CpGs")

    fig.suptitle("Methylation clocks on real bottlenose-dolphin blood (n = 50)",
                 color=FG, fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02)
    save(fig, "fig1_clock_validation.png")


# ---------------------------------------------------------------------------
def fig2_artifact():
    d = pd.read_csv(PROC / "species_residuals.csv")
    m = json.loads((RES / "aging_rate_models.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    ax = axes[0]
    ax.scatter(d.log_mass, d.log_arocm, s=34, c=INDIGO, alpha=0.75,
               edgecolors=BG, lw=0.5)
    xs = np.linspace(d.log_mass.min(), d.log_mass.max(), 50)
    b = m["models"]["M1"]
    fit = np.polyfit(d.log_mass, d.log_arocm, 1)
    ax.plot(xs, np.polyval(fit, xs), color=AMBER, lw=2)
    ax.set_xlabel("log adult body mass (g)")
    ax.set_ylabel("log epigenetic aging rate  |AROCM|")
    style(ax, "What it looks like",
          f"slope = {b['params']['log_mass']:+.3f}   R² = {b['r2']:.2f}   "
          f"p = {b['pvalues']['log_mass']:.1e}")

    ax = axes[1]
    ax.scatter(d.log_span, d.log_arocm, s=34, c=ROSE, alpha=0.75,
               edgecolors=BG, lw=0.5)
    xs = np.linspace(d.log_span.min(), d.log_span.max(), 50)
    fit2 = np.polyfit(d.log_span, d.log_arocm, 1)
    ax.plot(xs, np.polyval(fit2, xs), color=AMBER, lw=2, label="observed")
    c = np.mean(d.log_arocm + d.log_span)
    ax.plot(xs, c - xs, ":", color=EMERALD, lw=2,
            label="slope = −1 (pure artifact)")
    ax.legend(loc="upper right", labelcolor=FG, fontsize=9)
    sp = m["models"]["M2"]
    ax.set_xlabel("log sampled age span (years)")
    ax.set_ylabel("log epigenetic aging rate  |AROCM|")
    style(ax, "What it actually is",
          f"span coefficient = {sp['params']['log_span']:.3f} ± "
          f"{sp['se']['log_span']:.3f};  H₀: = −1 not rejected "
          f"(p = {sp['span_coef_vs_minus1']['p']:.2f})")

    fig.suptitle("The body-size scaling of epigenetic aging rate is mostly a "
                 "measurement artifact", color=FG, fontsize=13,
                 fontweight="bold", x=0.02, ha="left", y=1.03)
    save(fig, "fig2_sampling_artifact.png")


# ---------------------------------------------------------------------------
def fig3_residuals():
    d = pd.read_csv(PROC / "species_residuals.csv")
    d = d.dropna(subset=["resid_M3"])
    HERO = {"Heterocephalus glaber": "Naked mole rat",
            "Balaena mysticetus": "Bowhead whale",
            "Homo sapiens": "Human", "Mus musculus": "Mouse"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=1, ls="--", alpha=0.6)
    ax.axvline(0, color=MUTED, lw=1, ls="--", alpha=0.6)
    ax.scatter(d.resid_M1, d.resid_M3, s=30, c=MUTED, alpha=0.5,
               edgecolors="none")
    for latin, label in HERO.items():
        r = d[d.SpeciesLatinName == latin]
        if not len(r):
            continue
        r = r.iloc[0]
        col = AMBER if latin == "Heterocephalus glaber" else (
            CYAN if latin == "Balaena mysticetus" else VIOLET)
        ax.annotate("", xy=(r.resid_M1, r.resid_M3), xytext=(r.resid_M1, r.resid_M1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.6, alpha=0.9))
        ax.scatter([r.resid_M1], [r.resid_M3], s=140, c=col, zorder=5,
                   edgecolors=BG, lw=1.2)
        # keep labels inside the axes: flip to the left for far-left points
        left = r.resid_M1 < -1.2
        ax.annotate(label, (r.resid_M1, r.resid_M3), textcoords="offset points",
                    xytext=(-9 if left else 9, 8), color=col, fontsize=9.5,
                    fontweight="bold", ha="right" if left else "left")
    lim = [min(d.resid_M1.min(), d.resid_M3.min()) - .55,
           max(d.resid_M1.max(), d.resid_M3.max()) + .35]
    ax.plot(lim, lim, ":", color=GRID, lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Residual, naive model (mass only)")
    ax.set_ylabel("Residual, corrected (mass + span + Order)")
    style(ax, "Correcting the artifact moves the famous outliers",
          "arrows show each species shifting from the naive to the corrected fit")

    ax = axes[1]
    dd = d.sort_values("resid_M3")
    show = pd.concat([dd.head(10), dd.tail(10)])
    cols = [ROSE if v > 0 else INDIGO for v in show.resid_M3]
    ypos = np.arange(len(show))
    ax.barh(ypos, show.resid_M3, color=cols, alpha=0.9, height=0.75)
    for i, x in enumerate(show.itertuples()):
        if x.signif:
            ax.text(x.resid_M3 + (0.03 if x.resid_M3 > 0 else -0.03), i, "*",
                    va="center", ha="left" if x.resid_M3 > 0 else "right",
                    color=AMBER, fontsize=13, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels(show.SpeciesCommonName, fontsize=8.5, color=FG)
    ax.set_ylim(-0.9, len(show) - 0.1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Corrected residual  (← slower   faster →)")
    nsig = int(d.signif.sum())
    style(ax, "Ranked deviation from the body-size expectation",
          f"* = significant after BH correction across {len(d)} species "
          f"({nsig} of {len(d)} survive)")

    fig.suptitle("Once the artifact is removed, almost nobody is an outlier",
                 color=FG, fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02)
    save(fig, "fig3_residuals.png")


# ---------------------------------------------------------------------------
def fig4_validation():
    d = pd.read_csv(PROC / "species_residuals.csv")
    m = json.loads((RES / "aging_rate_models.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    ax = axes[0]
    orders = d.Order.value_counts().head(6).index.tolist()
    pal = [INDIGO, VIOLET, CYAN, AMBER, EMERALD, ROSE]
    for o, c in zip(orders, pal):
        s = d[d.Order == o]
        ax.scatter(s.log_mass, s.log_lifespan, s=34, c=c, alpha=0.85,
                   edgecolors=BG, lw=0.4, label=o)
    s = d[~d.Order.isin(orders)]
    ax.scatter(s.log_mass, s.log_lifespan, s=26, c=MUTED, alpha=0.5,
               edgecolors="none", label="other")
    xs = np.linspace(d.log_mass.min(), d.log_mass.max(), 50)
    fit = np.polyfit(d.log_mass, d.log_lifespan, 1)
    ax.plot(xs, np.polyval(fit, xs), color=FG, lw=1.8, ls="--")
    ax.legend(fontsize=7.5, labelcolor=FG, ncol=2, loc="lower right")
    ax.set_xlabel("log adult body mass (g)")
    ax.set_ylabel("log maximum lifespan (years)")
    lq = m["lifespan_mass_allometry"]
    style(ax, "Lifespan–mass allometry",
          f"lifespan ∝ mass^{lq['exponent']:.3f}   R² = {lq['r2']:.2f}   "
          f"n = {lq['n']} species")

    ax = axes[1]
    top = d.nlargest(12, "longevity_quotient").sort_values("longevity_quotient")
    ax.barh(np.arange(len(top)), top.longevity_quotient, color=VIOLET,
            alpha=0.9, height=0.75)
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.SpeciesCommonName, fontsize=8.5, color=FG)
    ax.set_xlabel("Longevity quotient (observed / mass-predicted lifespan)")
    style(ax, "The known longevity outliers come out on top",
          "bats and the naked mole rat recover their textbook positions")
    fig.suptitle("Sanity checks: the dataset behaves the way comparative biology says "
                 "it should", color=FG, fontsize=13, fontweight="bold",
                 x=0.02, ha="left", y=1.02)
    save(fig, "fig4_allometry_validation.png")


# ---------------------------------------------------------------------------
def fig5_interpretability():
    imp = pd.read_csv(PROC / "state_rf_importance.csv")
    uni = pd.read_csv(PROC / "state_lifespan_association.csv")
    j = json.loads((RES / "interpretability.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    t = imp.head(12).sort_values("importance")
    fam_col = {"BivProm": VIOLET, "ReprPC": INDIGO, "EnhA": AMBER, "TxEnh": AMBER,
               "PromF": CYAN, "TxEx": EMERALD, "Tx": EMERALD, "EnhWk": ROSE}
    cols = [fam_col.get(f, MUTED) for f in t.family]
    ax.barh(np.arange(len(t)), t.importance, xerr=t.importance_sd,
            color=cols, alpha=0.9, height=0.72,
            error_kw=dict(ecolor=MUTED, lw=1))
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels(t.state, fontsize=8.5, color=FG)
    ax.set_xlabel("Permutation importance (in-sample)")
    handles = [Line2D([], [], marker="s", ls="", color=c, label=k)
               for k, c in list(fam_col.items())[:6]]
    ax.legend(handles=handles, fontsize=7.5, labelcolor=FG, ncol=2, loc="lower right")
    style(ax, "Which chromatin states look informative",
          "fitting all 126 species at once")

    ax = axes[1]
    ml = j["ml"]
    names = ["Elastic net", "Random forest"]
    vals = [ml["elastic_net"]["r"], ml["random_forest"]["r"]]
    ax.axhline(0, color=MUTED, lw=1)
    ax.bar(names, vals, color=[INDIGO, VIOLET], alpha=0.9, width=0.5)
    p95 = ml["permutation"]["null_p95"]
    ax.axhspan(-p95, p95, color=GRID, alpha=0.55, zorder=0)
    ax.text(1.45, p95, "95% of the\nchance null", color=MUTED, fontsize=8,
            va="bottom", ha="right")
    ax.set_ylabel("Out-of-sample r  (leave-one-Order-out)")
    ax.set_ylim(-0.45, 0.45)
    style(ax, "… and how they do on an unseen Order",
          f"permutation p = {ml['permutation']['empirical_p']:.2f}  —  "
          "no better than chance")

    fig.suptitle("The chromatin-state signature does not survive a phylogenetic "
                 "holdout", color=FG, fontsize=13, fontweight="bold",
                 x=0.02, ha="left", y=1.02)
    save(fig, "fig5_interpretability.png")


# ---------------------------------------------------------------------------
def fig6_cpg_context():
    e = pd.read_csv(PROC / "clock_cpg_enrichment.csv")
    ann = pd.read_csv(PROC / "clock_cpg_annotation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    ax = axes[0]
    e = e.sort_values("odds_ratio")
    cols = [EMERALD if o > 1 else ROSE for o in e.odds_ratio]
    ax.barh(np.arange(len(e)), np.log2(e.odds_ratio), color=cols, alpha=0.9,
            height=0.7)
    for i, x in enumerate(e.itertuples()):
        if x.q < 0.05:
            off = 0.02 if x.odds_ratio > 1 else -0.02
            ax.text(np.log2(x.odds_ratio) + off, i, "*", va="center",
                    ha="left" if x.odds_ratio > 1 else "right",
                    color=AMBER, fontsize=13, fontweight="bold")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(np.arange(len(e)))
    ax.set_yticklabels([c.replace("_", " ") for c in e.category], fontsize=8.5,
                       color=FG)
    ax.set_xlabel("log₂ odds ratio vs the full array")
    style(ax, "Clock CpGs are pulled toward gene bodies and 5′ UTRs",
          "1,303 CpGs from the three universal clocks;  * = BH q < 0.05")

    ax = axes[1]
    top = ann.SYMBOL.value_counts().head(14).sort_values()
    ax.barh(np.arange(len(top)), top.values, color=VIOLET, alpha=0.9, height=0.72)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=8.5, color=FG, style="italic")
    ax.set_xlabel("Number of clock CpGs in the gene")
    style(ax, "and they concentrate in developmental regulators",
          "PAX2, HOXB7, EVX2, NPAS3 — the polycomb-target signature")

    fig.suptitle("What the clocks are actually reading",
                 color=FG, fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02)
    save(fig, "fig6_cpg_context.png")


def main():
    print("Rendering figures ...")
    fig1_clocks()
    fig2_artifact()
    fig3_residuals()
    fig4_validation()
    fig5_interpretability()
    fig6_cpg_context()
    print("done")


if __name__ == "__main__":
    main()
