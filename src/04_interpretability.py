"""
04_interpretability.py -- Which parts of the genome carry the cross-species
differences in epigenetic aging?

Model 3 showed that body mass explains almost nothing about the *magnitude* of
the aging rate once the sampling artifact is removed. That leaves the question
of shape: within a species, some chromatin states change with age much faster
than others, and the profile of which-states-change-fastest is a genuine
per-species phenotype. It is also immune to the 1/SD(age) artifact, because
every state in a species is divided by that same factor.

Four analyses:
  1. Which chromatin states vary most across species (the raw signal)
  2. Univariate association of each state's contrast with maximum lifespan,
     BH-corrected across the 54 states
  3. Interpretable ML: predict log maximum lifespan from the 54-state contrast
     vector, cross-validated by leaving out whole taxonomic Orders so the model
     cannot memorise close relatives. Elastic-net coefficients and permutation
     importance give the ranking.
  4. Genomic context of the clock CpGs: are the CpGs the published clocks chose
     enriched for promoters and CpG islands relative to the array background?

Outputs -> results/interpretability.json, data/processed/state_*.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260810)
CONSORTIUM_REPO = Path("/tmp/mmc")

META = ["SpeciesLatinName", "SpeciesCommonName", "Tissue", "MammalNumberHorvath",
        "Order", "adult_mass_g", "gestation_y", "maturity_y", "max_lifespan_y",
        "age_range_lo", "age_range_hi", "n_samples"]

STATE_BIOLOGY = {
    "BivProm": "Bivalent promoter (PRC2 / H3K27me3 + H3K4me3)",
    "ReprPC": "Polycomb-repressed",
    "PromF": "Promoter flanking",
    "TSS": "Transcription start site",
    "Tx": "Transcribed",
    "TxWk": "Weakly transcribed",
    "TxEx": "Transcribed exon",
    "TxEnh": "Transcribed enhancer",
    "EnhA": "Active enhancer",
    "EnhWk": "Weak enhancer",
    "Quies": "Quiescent / low signal",
    "Acet": "Acetylation-only",
    "HET": "Heterochromatin",
    "znf": "ZNF genes / repeats",
}


def state_family(s):
    base = s.rstrip("+-")
    for k in sorted(STATE_BIOLOGY, key=len, reverse=True):
        if base.startswith(k):
            return k
    return base


def build_contrast_matrix():
    """Per-species profile of state rate relative to that species' own mean."""
    w = pd.read_csv(PROC / "species_state_arocm_wide.csv")
    states = [c for c in w.columns if c not in META]
    w["span_y"] = (w.age_range_hi - w.age_range_lo) * w.max_lifespan_y
    w = w[(w.span_y > 0) & w.adult_mass_g.notna()].copy()

    S = w[states].abs().clip(lower=1e-6)
    contrast = S.div(S.mean(axis=1), axis=0)      # divides out 1/SD(age) exactly
    contrast[META[:5] + ["n_samples", "adult_mass_g", "max_lifespan_y"]] = \
        w[META[:5] + ["n_samples", "adult_mass_g", "max_lifespan_y"]]

    rows = []
    for spname, g in contrast.groupby("SpeciesLatinName"):
        rec = {"SpeciesLatinName": spname,
               "SpeciesCommonName": g.SpeciesCommonName.iloc[0],
               "Order": g.Order.iloc[0],
               "n_samples": int(g.n_samples.sum()),
               "adult_mass_g": g.adult_mass_g.iloc[0],
               "max_lifespan_y": g.max_lifespan_y.iloc[0]}
        for s in states:
            rec[s] = np.average(g[s], weights=g.n_samples)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"    contrast matrix: {len(df)} species x {len(states)} chromatin states")
    return df, states


def variation_across_species(df, states):
    print("\n=== 1. Which states differ most across species? ===")
    rec = []
    for s in states:
        v = df[s]
        rec.append({"state": s, "family": state_family(s),
                    "mean_contrast": float(v.mean()),
                    "cv_across_species": float(v.std() / v.mean()),
                    "min": float(v.min()), "max": float(v.max())})
    r = pd.DataFrame(rec).sort_values("cv_across_species", ascending=False)
    print("    most variable across species:")
    for x in r.head(6).itertuples():
        print(f"      {x.state:14s} {STATE_BIOLOGY.get(x.family,x.family)[:38]:40s} "
              f"CV={x.cv_across_species:.3f} mean={x.mean_contrast:.2f}")
    print("    least variable:")
    for x in r.tail(3).itertuples():
        print(f"      {x.state:14s} {STATE_BIOLOGY.get(x.family,x.family)[:38]:40s} "
              f"CV={x.cv_across_species:.3f} mean={x.mean_contrast:.2f}")
    return r


def univariate_lifespan(df, states):
    print("\n=== 2. State contrast vs maximum lifespan (BH-corrected) ===")
    y = np.log(df.max_lifespan_y)
    rec = []
    for s in states:
        rho, p = stats.spearmanr(df[s], y)
        r_p, p_p = stats.pearsonr(np.log(df[s]), y)
        rec.append({"state": s, "family": state_family(s), "spearman_rho": rho,
                    "spearman_p": p, "pearson_logr": r_p, "pearson_p": p_p})
    r = pd.DataFrame(rec)
    order = np.argsort(r.spearman_p.values)
    ranked = r.spearman_p.values[order]
    n = len(r)
    bh = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(bh, 0, 1)
    r["q"] = q
    r = r.sort_values("spearman_rho")
    sig = r[r.q < 0.05]
    print(f"    {len(sig)} / {n} states significant at q<0.05")
    for x in pd.concat([r.head(5), r.tail(5)]).itertuples():
        star = "*" if x.q < 0.05 else " "
        print(f"      {star} {x.state:14s} rho={x.spearman_rho:+.3f} q={x.q:.4f}  "
              f"{STATE_BIOLOGY.get(x.family,x.family)[:40]}")
    return r


def interpretable_ml(df, states, n_perm=200):
    """Predict log lifespan from the state profile, holding out whole Orders."""
    print("\n=== 3. Interpretable ML: state profile -> maximum lifespan ===")
    counts = df.Order.value_counts()
    big = counts[counts >= 5].index.tolist()
    d = df[df.Order.isin(big)].reset_index(drop=True)
    print(f"    orders with >=5 species: {big}")
    print(f"    {len(d)} species used; leave-one-Order-out CV "
          f"(a fold never shares an Order with training)")

    X = np.log(d[states].values)
    y = np.log(d.max_lifespan_y.values)
    groups = d.Order.values

    def loo_order(model_fn):
        pred = np.empty(len(d))
        for o in big:
            te = groups == o
            tr = ~te
            sc = StandardScaler().fit(X[tr])
            m = model_fn()
            m.fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        return pred

    en = lambda: ElasticNetCV(l1_ratio=[0.3, 0.6, 0.9], alphas=25, cv=5,
                              max_iter=5000, tol=1e-3, random_state=0)
    rf = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                       random_state=0, n_jobs=-1)

    results = {}
    for name, fn in [("elastic_net", en), ("random_forest", rf)]:
        pred = loo_order(fn)
        r = float(np.corrcoef(y, pred)[0, 1])
        r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        mae = float(np.mean(np.abs(np.exp(y) - np.exp(pred))))
        print(f"    {name:14s} leave-one-Order-out: r={r:+.3f}  R2={r2:+.3f}  "
              f"MAE={mae:.1f} y", flush=True)
        results[name] = {"r": r, "r2_oos": r2, "mae_years": mae}

    # Permutation null. A fixed penalty (median of the real per-fold choices)
    # keeps this affordable; re-tuning inside each shuffle changes the null
    # negligibly but multiplies the cost by ~40x.
    alpha_fix = float(np.median([
        ElasticNetCV(l1_ratio=[0.6], alphas=15, cv=3, max_iter=3000, tol=1e-3,
                     random_state=0)
        .fit(StandardScaler().fit_transform(X[groups != o]), y[groups != o]).alpha_
        for o in big]))
    print(f"    permutation null: {n_perm} shuffles at fixed alpha={alpha_fix:.4f}",
          flush=True)
    from sklearn.linear_model import ElasticNet
    null = []
    for b in range(n_perm):
        yp = RNG.permutation(y)
        pr = np.empty(len(d))
        for o in big:
            te = groups == o
            tr = ~te
            sc = StandardScaler().fit(X[tr])
            m = ElasticNet(alpha=alpha_fix, l1_ratio=0.6, max_iter=3000, tol=1e-3)
            m.fit(sc.transform(X[tr]), yp[tr])
            pr[te] = m.predict(sc.transform(X[te]))
        null.append(np.corrcoef(yp, pr)[0, 1] if np.std(pr) > 1e-12 else 0.0)
    null = np.array(null)
    p_emp = float((np.sum(null >= results["elastic_net"]["r"]) + 1) / (n_perm + 1))
    print(f"    permutation null: mean r={null.mean():+.3f}, 95th pct="
          f"{np.percentile(null,95):+.3f} -> empirical p={p_emp:.4f}")
    results["permutation"] = {"n": n_perm, "null_mean_r": float(null.mean()),
                              "null_p95": float(np.percentile(null, 95)),
                              "empirical_p": p_emp}

    # Fit on everything for interpretation
    sc = StandardScaler().fit(X)
    fin = ElasticNetCV(l1_ratio=[0.3, 0.6, 0.9], alphas=25, cv=5,
                       max_iter=5000, tol=1e-3, random_state=0).fit(sc.transform(X), y)
    coef = pd.DataFrame({"state": states, "family": [state_family(s) for s in states],
                         "coef": fin.coef_})
    coef["abs"] = coef.coef.abs()
    coef = coef.sort_values("abs", ascending=False)
    nz = int((fin.coef_ != 0).sum())
    print(f"    elastic net kept {nz}/{len(states)} states "
          f"(alpha={fin.alpha_:.4f}, l1_ratio={fin.l1_ratio_})")
    for x in coef[coef.coef != 0].head(8).itertuples():
        print(f"      {x.state:14s} {x.coef:+.4f}  "
              f"{STATE_BIOLOGY.get(x.family,x.family)[:42]}")

    rfm = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                random_state=0, n_jobs=-1).fit(X, y)
    pi = permutation_importance(rfm, X, y, n_repeats=20, random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"state": states,
                        "family": [state_family(s) for s in states],
                        "importance": pi.importances_mean,
                        "importance_sd": pi.importances_std}
                       ).sort_values("importance", ascending=False)
    print("    random-forest permutation importance, top 5:")
    for x in imp.head(5).itertuples():
        print(f"      {x.state:14s} {x.importance:.4f} +/- {x.importance_sd:.4f}")

    results["n_states_selected"] = nz
    results["alpha"] = float(fin.alpha_)
    return results, coef, imp, d


def clock_cpg_context():
    """Genomic context of the CpGs the published clocks selected."""
    print("\n=== 4. Where do the clock CpGs sit? ===")
    slim = ROOT / "data" / "raw" / "array_cpg_annotation_slim.csv.gz"
    full = (CONSORTIUM_REPO / "Annotations, Amin Haghani" / "Mammals" /
            "Homo_sapiens.hg38.HorvathMammalMethylChip40.v1.csv")
    ann_path = slim if slim.exists() else full
    clock_ann = pd.read_csv(PROC / "clock_cpg_annotation.csv")
    if not ann_path.exists():
        print("    background annotation unavailable; skipping enrichment")
        return None, clock_ann

    bg = pd.read_csv(ann_path, usecols=["CGid", "main_Categories", "SYMBOL"],
                     low_memory=False).drop_duplicates("CGid")
    print(f"    background: {len(bg):,} array CpGs; clock CpGs: {len(clock_ann):,}")

    cats = sorted(set(bg.main_Categories.dropna()) | set(clock_ann.main_Categories.dropna()))
    rec = []
    for c in cats:
        a = int((clock_ann.main_Categories == c).sum())
        b = int(len(clock_ann) - a)
        cc = int((bg.main_Categories == c).sum())
        dd = int(len(bg) - cc)
        odds, p = stats.fisher_exact([[a, b], [cc, dd]])
        rec.append({"category": c, "clock_n": a, "clock_pct": 100 * a / len(clock_ann),
                    "bg_pct": 100 * cc / len(bg), "odds_ratio": odds, "p": p})
    e = pd.DataFrame(rec)
    order = np.argsort(e.p.values)
    ranked = e.p.values[order]
    n = len(e)
    bh = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(bh, 0, 1)
    e["q"] = q
    e = e.sort_values("odds_ratio", ascending=False)
    for x in e.itertuples():
        star = "*" if x.q < 0.05 else " "
        print(f"      {star} {x.category:24s} clock {x.clock_pct:5.1f}% vs "
              f"array {x.bg_pct:5.1f}%  OR={x.odds_ratio:.2f}  q={x.q:.2e}")

    top = (clock_ann.SYMBOL.value_counts().head(12))
    print("    genes hit by the most clock CpGs: " +
          ", ".join(f"{g} ({c})" for g, c in top.items()))
    return e, clock_ann


def main():
    print("=== Building sampling-robust contrast matrix ===")
    df, states = build_contrast_matrix()

    var = variation_across_species(df, states)
    uni = univariate_lifespan(df, states)
    ml, coef, imp, used = interpretable_ml(df, states)
    enrich, clock_ann = clock_cpg_context()

    df.to_csv(PROC / "species_state_contrast.csv", index=False)
    var.to_csv(PROC / "state_variation.csv", index=False)
    uni.to_csv(PROC / "state_lifespan_association.csv", index=False)
    coef.to_csv(PROC / "state_elasticnet_coefficients.csv", index=False)
    imp.to_csv(PROC / "state_rf_importance.csv", index=False)
    if enrich is not None:
        enrich.to_csv(PROC / "clock_cpg_enrichment.csv", index=False)

    payload = {
        "n_species": int(len(df)), "n_states": len(states),
        "most_variable_states": var.head(10).to_dict("records"),
        "n_states_lifespan_significant": int((uni.q < 0.05).sum()),
        "top_lifespan_states": uni.reindex(uni.spearman_rho.abs()
                                           .sort_values(ascending=False).index)
                                  .head(10).to_dict("records"),
        "ml": ml,
        "top_elasticnet_states": coef[coef.coef != 0].head(10).to_dict("records"),
        "top_rf_states": imp.head(10).to_dict("records"),
        "clock_cpg_enrichment": enrich.to_dict("records") if enrich is not None else None,
        "top_clock_genes": clock_ann.SYMBOL.value_counts().head(20).to_dict(),
    }
    (RES / "interpretability.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[done] -> {RES/'interpretability.json'}")


if __name__ == "__main__":
    main()
