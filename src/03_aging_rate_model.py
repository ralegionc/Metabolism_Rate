"""
03_aging_rate_model.py -- Pan-mammalian epigenetic aging rate vs body size,
and the residuals that fall out of it.

The question: which mammals age faster or slower than their body size predicts?

The complication, established empirically in this script before any modelling:
AROCM as published is the slope of *z-scored* mean methylation on age, so it is
mechanically bounded above by ~1/SD(sampled ages). Long-lived species get
sampled across wider spans in years, which drags AROCM down for reasons that
have nothing to do with biology. Any residual analysis that ignores this is
partly measuring study design.

So four models, in increasing order of how much I trust them:

  M1  log|AROCM| ~ log(mass)                          naive; replicates the
                                                      published allometry
  M2  log|AROCM| ~ log(mass) + log(sampled age span)  removes the sampling
                                                      artifact
  M3  M2 + random intercept per taxonomic Order       mammal Orders are not
                                                      independent draws
  M4  state-contrast ~ log(mass)                      ratio of one chromatin
                                                      state's rate to the
                                                      species' own genome-wide
                                                      mean; the 1/SD(age) term
                                                      cancels exactly

M3 residuals are the headline. M4 is the cross-check that does not depend on
the sampling correction being right.

Outputs -> results/aging_rate_models.json, data/processed/species_residuals.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260810)

META = ["SpeciesLatinName", "SpeciesCommonName", "Tissue", "MammalNumberHorvath",
        "Order", "adult_mass_g", "gestation_y", "maturity_y", "max_lifespan_y",
        "age_range_lo", "age_range_hi", "n_samples"]
PRIMARY_STATE = "BivProm2+"   # bivalent/PRC2 promoters: strongest age signal


def build_species_table():
    w = pd.read_csv(PROC / "species_state_arocm_wide.csv")
    states = [c for c in w.columns if c not in META]
    w["span_y"] = (w.age_range_hi - w.age_range_lo) * w.max_lifespan_y

    # Genome-wide mean rate across all 54 chromatin states, and the contrast of
    # the primary state against it. The contrast divides out any per-species
    # scaling (including the 1/SD(age) sampling term), leaving a pure shape.
    S = w[states].abs()
    w["arocm_mean_states"] = S.mean(axis=1)
    w["arocm_primary"] = w[PRIMARY_STATE].abs()
    w["contrast_primary"] = w["arocm_primary"] / w["arocm_mean_states"]

    ok = (w.span_y > 0) & (w.arocm_primary > 1e-6) & w.adult_mass_g.notna()
    dropped = (~ok).sum()
    w = w[ok].copy()
    print(f"    {dropped} species-tissue records dropped (zero span or rate)")

    # Aggregate tissues to one row per species, weighting by sample count.
    def wavg(g, col):
        return np.average(g[col], weights=g.n_samples)

    rows = []
    for sp, g in w.groupby("SpeciesLatinName"):
        rows.append({
            "SpeciesLatinName": sp,
            "SpeciesCommonName": g.SpeciesCommonName.iloc[0],
            "Order": g.Order.iloc[0],
            "n_tissues": len(g),
            "n_samples": int(g.n_samples.sum()),
            "adult_mass_g": g.adult_mass_g.iloc[0],
            "max_lifespan_y": g.max_lifespan_y.iloc[0],
            "maturity_y": g.maturity_y.iloc[0],
            "gestation_y": g.gestation_y.iloc[0],
            "span_y": wavg(g, "span_y"),
            # geometric mean across tissues, weighted by array count
            "arocm": np.exp(np.average(np.log(g.arocm_primary), weights=g.n_samples)),
            "arocm_mean_states": np.exp(np.average(np.log(g.arocm_mean_states),
                                                   weights=g.n_samples)),
            "contrast": np.average(g.contrast_primary, weights=g.n_samples),
            "tissues": "|".join(sorted(g.Tissue.unique())),
        })
    sp = pd.DataFrame(rows)
    for c, n in [("adult_mass_g", "log_mass"), ("max_lifespan_y", "log_lifespan"),
                 ("span_y", "log_span"), ("arocm", "log_arocm"),
                 ("contrast", "log_contrast")]:
        sp[n] = np.log(sp[c])

    # AROCM = cor(methylation, age) / SD(age) by construction, so multiplying the
    # rate back by the sampled SD recovers an implied correlation -- a bounded,
    # directly interpretable measure of how tightly the epigenome tracks age.
    # SD is approximated as span/sqrt(12) (uniform sampling over the span).
    sp["implied_cor"] = sp.arocm * sp.span_y / np.sqrt(12)
    print(f"    implied correlation: median {sp.implied_cor.median():.2f}, "
          f"IQR {sp.implied_cor.quantile(.25):.2f}-{sp.implied_cor.quantile(.75):.2f}, "
          f"{(sp.implied_cor > 1).sum()} above 1.0 (approximation error)")
    print(f"    species-level table: {len(sp)} species, {sp.Order.nunique()} orders, "
          f"{sp.n_samples.sum():,} underlying arrays")
    return sp, w, states


def longevity_quotient(sp):
    """Classic comparative-biology check: lifespan vs body mass allometry."""
    m = smf.ols("log_lifespan ~ log_mass", data=sp).fit()
    sp["lq_resid"] = m.resid
    sp["longevity_quotient"] = np.exp(m.resid)
    b, se = m.params["log_mass"], m.bse["log_mass"]
    print(f"    max lifespan ~ mass^{b:.3f} (SE {se:.3f}), R2={m.rsquared:.3f}")
    print("      literature places this exponent near 0.15-0.25 for mammals")
    top = sp.nlargest(5, "longevity_quotient")[["SpeciesCommonName", "longevity_quotient"]]
    print("      highest longevity quotient: " +
          ", ".join(f"{r.SpeciesCommonName} {r.longevity_quotient:.2f}x"
                    for r in top.itertuples()))
    return {"exponent": float(b), "se": float(se), "r2": float(m.rsquared),
            "n": int(m.nobs)}


def fit_models(sp):
    out = {}

    print("\n  M1  log|AROCM| ~ log(mass)   [naive]")
    m1 = smf.ols("log_arocm ~ log_mass", data=sp).fit()
    sp["resid_M1"] = m1.resid
    print(f"      slope={m1.params['log_mass']:+.3f} (SE {m1.bse['log_mass']:.3f}), "
          f"R2={m1.rsquared:.3f}, p={m1.pvalues['log_mass']:.2e}")
    out["M1"] = _summ(m1, ["log_mass"])

    print("\n  M2  log|AROCM| ~ log(mass) + log(sampled age span)   [sampling-controlled]")
    m2 = smf.ols("log_arocm ~ log_mass + log_span", data=sp).fit()
    sp["resid_M2"] = m2.resid
    for k in ["log_mass", "log_span"]:
        print(f"      {k:9s} {m2.params[k]:+.3f} (SE {m2.bse[k]:.3f})  p={m2.pvalues[k]:.2e}")
    print(f"      R2={m2.rsquared:.3f}")
    vif = 1 / (1 - smf.ols("log_mass ~ log_span", data=sp).fit().rsquared)
    print(f"      VIF(mass, span) = {vif:.2f}")
    out["M2"] = _summ(m2, ["log_mass", "log_span"])
    out["M2"]["vif"] = float(vif)

    # The pure-artifact prediction. If AROCM is the slope of z-scored methylation
    # on age, then AROCM = cor / SD(age) exactly, so log|AROCM| must carry a
    # log(span) coefficient of -1 with nothing biological attached. Test it.
    t = (m2.params["log_span"] + 1.0) / m2.bse["log_span"]
    p_vs_minus1 = float(2 * stats.t.sf(abs(t), df=int(m2.df_resid)))
    print(f"      H0: span coefficient = -1 (pure sampling artifact) -> "
          f"t={t:+.2f}, p={p_vs_minus1:.3f}"
          f"  {'CANNOT REJECT' if p_vs_minus1 > 0.05 else 'rejected'}")
    out["M2"]["span_coef_vs_minus1"] = {"t": float(t), "p": p_vs_minus1}

    print("\n  M3  M2 + random intercept by Order   [taxonomic non-independence]")
    keep = sp.Order.map(sp.Order.value_counts()) >= 2
    d3 = sp[keep].copy()
    m3 = smf.mixedlm("log_arocm ~ log_mass + log_span", data=d3,
                     groups=d3["Order"]).fit()
    d3["resid_M3"] = m3.resid
    sp["resid_M3"] = np.nan
    sp.loc[d3.index, "resid_M3"] = d3.resid_M3
    for k in ["log_mass", "log_span"]:
        print(f"      {k:9s} {m3.params[k]:+.3f} (SE {m3.bse[k]:.3f})  p={m3.pvalues[k]:.2e}")
    icc = float(m3.cov_re.iloc[0, 0] / (m3.cov_re.iloc[0, 0] + m3.scale))
    print(f"      Order variance={m3.cov_re.iloc[0,0]:.4f}, residual={m3.scale:.4f}, "
          f"ICC={icc:.3f}  (n={int(m3.nobs)} in {d3.Order.nunique()} orders)")
    out["M3"] = {"params": {k: float(m3.params[k]) for k in ["log_mass", "log_span"]},
                 "se": {k: float(m3.bse[k]) for k in ["log_mass", "log_span"]},
                 "pvalues": {k: float(m3.pvalues[k]) for k in ["log_mass", "log_span"]},
                 "order_variance": float(m3.cov_re.iloc[0, 0]),
                 "residual_variance": float(m3.scale), "icc": icc,
                 "n": int(m3.nobs), "n_orders": int(d3.Order.nunique())}

    print(f"\n  M4  log(state contrast {PRIMARY_STATE}/genome-wide) ~ log(mass)   "
          "[sampling-robust]")
    m4 = smf.ols("log_contrast ~ log_mass", data=sp).fit()
    sp["resid_M4"] = m4.resid
    print(f"      slope={m4.params['log_mass']:+.4f} (SE {m4.bse['log_mass']:.4f}), "
          f"R2={m4.rsquared:.4f}, p={m4.pvalues['log_mass']:.3f}")
    out["M4"] = _summ(m4, ["log_mass"])

    print("\n  M5  implied correlation(methylation, age) ~ log(mass)   "
          "[sampling-robust, bounded]")
    m5 = smf.ols("implied_cor ~ log_mass", data=sp).fit()
    sp["resid_M5"] = m5.resid
    print(f"      slope={m5.params['log_mass']:+.4f} (SE {m5.bse['log_mass']:.4f}), "
          f"R2={m5.rsquared:.4f}, p={m5.pvalues['log_mass']:.3f}")
    out["M5"] = _summ(m5, ["log_mass"])

    # How much of the naive allometry survives the sampling control?
    shrink = 1 - abs(m2.params["log_mass"] / m1.params["log_mass"])
    print(f"\n      mass coefficient shrinks {shrink*100:.0f}% once sampled age "
          "span is controlled")
    out["mass_coefficient_shrinkage"] = float(shrink)
    return sp, out


def _summ(m, keys):
    return {"params": {k: float(m.params[k]) for k in keys},
            "se": {k: float(m.bse[k]) for k in keys},
            "pvalues": {k: float(m.pvalues[k]) for k in keys},
            "r2": float(m.rsquared), "n": int(m.nobs)}


def flag_outliers(sp, n_boot=2000):
    """Which species are genuine outliers, not just ranked at the extremes?

    Externally studentized residuals from M2 -- each species' residual scaled by
    an error variance estimated with that species held out -- then Benjamini-
    Hochberg across all 126 tests. Bootstrap resampling is used separately for
    a confidence band on the fitted line, not for per-species significance
    (resampling species estimates uncertainty in the line, which is not the
    same question).
    """
    d = sp.reset_index(drop=True).copy()
    m = smf.ols("log_arocm ~ log_mass + log_span", data=d).fit()
    infl = m.get_influence()
    d["resid_student"] = infl.resid_studentized_external
    d["cooks_d"] = infl.cooks_distance[0]
    df = int(m.df_resid) - 1
    d["p_outlier"] = 2 * stats.t.sf(np.abs(d.resid_student), df=df)
    order = np.argsort(d.p_outlier.values)
    ranked = d.p_outlier.values[order]
    n = len(d)
    bh = ranked * n / (np.arange(1, n + 1))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(bh, 0, 1)
    d["q_outlier"] = q
    d["signif"] = d.q_outlier < 0.05
    print(f"    externally studentized residuals, BH-FDR across {n} species: "
          f"{int(d.signif.sum())} significant at q<0.05")
    if d.signif.any():
        for x in d[d.signif].sort_values("resid_student").itertuples():
            print(f"      {x.SpeciesCommonName:30s} t={x.resid_student:+.2f} "
                  f"q={x.q_outlier:.4f}")

    # Bootstrap band on the fitted line
    preds = []
    for _ in range(n_boot):
        b = d.iloc[RNG.integers(0, n, n)]
        try:
            preds.append(smf.ols("log_arocm ~ log_mass + log_span", data=b)
                         .fit().predict(d).values)
        except Exception:
            continue
    P = np.vstack(preds)
    d["fit_lo"] = np.percentile(P, 2.5, axis=0)
    d["fit_hi"] = np.percentile(P, 97.5, axis=0)
    print(f"    bootstrap band from {len(P)} resamples")
    return d


def rate_of_living(sp):
    """Does mass-specific metabolic rate predict epigenetic aging rate?"""
    an = pd.read_csv(PROC / "anage_traits.csv")
    an = an[["SpeciesLatinName", "Metabolic.rate..W.", "Body.mass..g.",
             "Temperature..K."]].dropna(subset=["Metabolic.rate..W."])
    an = an.drop_duplicates("SpeciesLatinName")
    d = sp.merge(an, on="SpeciesLatinName", how="inner")
    d = d[(d["Metabolic.rate..W."] > 0) & (d["Body.mass..g."] > 0)]
    if len(d) < 15:
        print(f"    only {len(d)} species with metabolic rate; skipping")
        return None
    d["log_bmr"] = np.log(d["Metabolic.rate..W."])
    d["log_msmr"] = np.log(d["Metabolic.rate..W."] / d["Body.mass..g."])
    kle = smf.ols("log_bmr ~ np.log(d['Body.mass..g.'])", data=d).fit()
    b = kle.params.iloc[1]
    print(f"    n={len(d)} species with basal metabolic rate")
    print(f"    Kleiber check: BMR ~ mass^{b:.3f} (Kleiber's law predicts 0.75)")
    m = smf.ols("log_arocm ~ log_msmr + log_span", data=d).fit()
    print(f"    log|AROCM| ~ mass-specific BMR (+span): "
          f"beta={m.params['log_msmr']:+.3f} (SE {m.bse['log_msmr']:.3f}), "
          f"p={m.pvalues['log_msmr']:.3f}")
    rho, prho = stats.spearmanr(d.log_msmr, d.resid_M2)
    print(f"    Spearman(mass-specific BMR, M2 residual) = {rho:+.3f} (p={prho:.3f})")
    return {"n": int(len(d)), "kleiber_exponent": float(b),
            "beta_massspecific_bmr": float(m.params["log_msmr"]),
            "se": float(m.bse["log_msmr"]), "p": float(m.pvalues["log_msmr"]),
            "spearman_rho_vs_M2resid": float(rho), "spearman_p": float(prho)}


def main():
    print("=== Building species-level table ===")
    sp, w, states = build_species_table()

    print("\n=== Sanity check: lifespan-mass allometry ===")
    lq = longevity_quotient(sp)

    print("\n=== Aging-rate models ===")
    sp, models = fit_models(sp)

    print("\n=== Outlier detection ===")
    d = flag_outliers(sp)

    print("\n=== Rate-of-living hypothesis ===")
    rol = rate_of_living(sp)

    print("\n=== Headline: who deviates most (M3, sampling + taxonomy controlled) ===")
    r = d.dropna(subset=["resid_M3"]).sort_values("resid_M3")
    print("\n  SLOWER than body size predicts (negative residual):")
    for x in r.head(8).itertuples():
        print(f"    {x.SpeciesCommonName:28s} {x.resid_M3:+.3f}  "
              f"mass={x.adult_mass_g:>12,.0f}g  LS={x.max_lifespan_y:>6.1f}y  "
              f"{'*' if x.signif else ''}")
    print("\n  FASTER than body size predicts (positive residual):")
    for x in r.tail(8).iloc[::-1].itertuples():
        print(f"    {x.SpeciesCommonName:28s} {x.resid_M3:+.3f}  "
              f"mass={x.adult_mass_g:>12,.0f}g  LS={x.max_lifespan_y:>6.1f}y  "
              f"{'*' if x.signif else ''}")

    for name in ["Heterocephalus glaber", "Balaena mysticetus", "Homo sapiens",
                 "Mus musculus"]:
        row = d[d.SpeciesLatinName == name]
        if len(row):
            x = row.iloc[0]
            pct = float((d.resid_M3 < x.resid_M3).mean() * 100)
            print(f"\n  {x.SpeciesCommonName}: M3 residual {x.resid_M3:+.3f} "
                  f"({pct:.0f}th percentile), naive M1 residual {x.resid_M1:+.3f}, "
                  f"longevity quotient {x.longevity_quotient:.2f}x")

    d.to_csv(PROC / "species_residuals.csv", index=False)
    payload = {"n_species": int(len(sp)), "n_orders": int(sp.Order.nunique()),
               "primary_state": PRIMARY_STATE,
               "lifespan_mass_allometry": lq, "models": models,
               "rate_of_living": rol,
               "n_significant_residuals": int(d.signif.sum())}
    (RES / "aging_rate_models.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[done] -> {RES/'aging_rate_models.json'}")


if __name__ == "__main__":
    main()
