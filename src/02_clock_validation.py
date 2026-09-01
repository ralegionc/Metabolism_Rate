"""
02_clock_validation.py -- Two things, both on real methylation data.

(A) Apply the three *published* universal pan-mammalian clocks to the 50 real
    bottlenose-dolphin blood samples, using the consortium's own inverse
    transformations (transcribed from R_pgm1_threeUnversalClocks.R), and
    measure agreement with chronological age.

(B) Train an age clock from scratch on the same 37,554-CpG matrix with nested
    cross-validation -- inner 5-fold to pick the elastic-net penalty, outer
    leave-one-out for an unbiased error estimate -- then compare it to the
    published clock and check how much the two agree on which CpGs matter.

A permutation null is included because n=50 with p=37,554 is exactly the regime
where a cross-validated correlation can look impressive by accident.

Outputs -> data/processed/clock_*.csv, results/clock_validation.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260810)

# Trait constants for Tursiops truncatus, taken from the consortium's anAge table
MYMAX = 1.3  # consortium inflates non-human/mouse max lifespan by 30%


# ---------------------------------------------------------------------------
# Published-clock inverse transformations (verbatim from the consortium R code)
# ---------------------------------------------------------------------------
def f2_antitrans_clock2(y, y_max_age, y_gestation, const=1.0):
    x0 = const * np.exp(-np.exp(-y))
    return x0 * (y_max_age + y_gestation) - y_gestation


def f2_revtrsf_clock3(y_pred, m1, m2=None, c1=1.0):
    m2 = m1 if m2 is None else m2
    return np.where(y_pred < 0, (np.exp(y_pred / c1) - 1) * m2 * c1 + m1, y_pred * m2 + m1)


def a_logli(gestation, maturity, c1=5.0, c2=0.38, c0=0.0):
    """m1 tuning point: m = 5*(G/ASM)^0.38, consortium formula (7)."""
    return c1 * ((gestation + c0) / maturity) ** c2


def load_dolphin_traits():
    anage = pd.read_csv(PROC / "anage_traits.csv")
    row = anage[anage.SpeciesLatinName == "Tursiops truncatus"].iloc[0]
    traits = {
        "max_age": float(row["maxAge"]),
        "gestation_y": float(row["GestationTimeInYears"]),
        "maturity_y": float(row["averagedMaturity.yrs"]),
    }
    traits["high_max_age"] = MYMAX * traits["max_age"]
    return traits


def apply_published_clocks(betas, samples, traits):
    print("\n=== (A) Published universal clocks on real dolphin blood ===")
    print(f"    Tursiops truncatus: maxAge={traits['max_age']} y "
          f"(inflated to {traits['high_max_age']:.1f}), "
          f"gestation={traits['gestation_y']:.3f} y, maturity={traits['maturity_y']:.2f} y")

    out = samples.copy()
    linear = {}
    for k in [1, 2, 3]:
        clock = pd.read_csv(PROC / f"universal_clock{k}.csv")
        icept = float(clock.loc[clock.CGid == "Intercept", "coef"].iloc[0])
        cg = clock[clock.CGid != "Intercept"].set_index("CGid")["coef"]
        missing = [c for c in cg.index if c not in betas.index]
        if missing:
            print(f"    clock{k}: {len(missing)} CpGs absent from array -- dropped")
        cg = cg[[c for c in cg.index if c in betas.index]]
        linear[k] = icept + betas.loc[cg.index].T.values @ cg.values

    # Clock 1: log(age+2) scale
    out["DNAmAge_clock1"] = np.exp(linear[1]) - 2
    # Clock 2: relative age via a Gompertz-style double exponential
    out["DNAmRelativeAge"] = np.exp(-np.exp(-linear[2]))
    out["DNAmAge_clock2"] = f2_antitrans_clock2(
        linear[2], traits["high_max_age"], traits["gestation_y"])
    # Clock 3: log-linear relative adult age
    m1 = a_logli(traits["gestation_y"], traits["maturity_y"])
    out["DNAmRelativeAdultAge"] = f2_revtrsf_clock3(linear[3], m1)
    out["DNAmAge_clock3"] = (out["DNAmRelativeAdultAge"]
                             * (traits["maturity_y"] + traits["gestation_y"])
                             - traits["gestation_y"])
    print(f"    clock3 tuning point m1 = {m1:.4f}")

    stats_out = {}
    for k in [1, 2, 3]:
        col = f"DNAmAge_clock{k}"
        r = float(np.corrcoef(out.Age, out[col])[0, 1])
        stats_out[f"clock{k}"] = {
            "pearson_r": r,
            "median_AE": float(np.median(np.abs(out.Age - out[col]))),
            "mean_AE": float(np.mean(np.abs(out.Age - out[col]))),
            "n_cpgs_used": int(len(pd.read_csv(PROC / f"universal_clock{k}.csv")) - 1),
        }
        print(f"    clock{k}: r={r:.3f}  medianAE={stats_out[f'clock{k}']['median_AE']:.2f} y")
    out.to_csv(PROC / "clock_predictions_published.csv", index=False)
    return out, stats_out


# ---------------------------------------------------------------------------
# (B) From-scratch clock with nested CV
# ---------------------------------------------------------------------------
def transform_age(age, adult=1.0):
    """Horvath-style log-linear age transform (single species)."""
    return np.log(age + adult)


def inv_transform_age(y, adult=1.0):
    return np.exp(y) - adult


def train_from_scratch(betas, samples, n_perm=100):
    print("\n=== (B) From-scratch elastic-net clock, nested CV ===")
    X = betas.T.values.astype(np.float64)  # samples x CpGs
    cg_ids = betas.index.to_numpy()
    age = samples.Age.to_numpy()
    y = transform_age(age)
    n, p = X.shape
    TOPK = 5000
    print(f"    design: {n} samples x {p:,} CpGs  (p/n = {p/n:,.0f})")
    print(f"    per fold: rank CpGs by training-set variance, keep top {TOPK:,}, "
          f"then elastic net")
    print("    (the filter is refit inside every fold and never sees age, so no leakage)")

    l1_ratios = [0.5, 0.9]

    def fit_fold(Xtr, ytr, seed):
        sd = Xtr.std(axis=0)
        idx = np.argpartition(sd, -TOPK)[-TOPK:]
        m = ElasticNetCV(l1_ratio=l1_ratios, alphas=25,
                         cv=KFold(3, shuffle=True, random_state=seed),
                         max_iter=3000, tol=1e-3, n_jobs=-1,
                         selection="random", random_state=seed)
        m.fit(Xtr[:, idx], ytr)
        return m, idx

    preds = np.empty(n)
    chosen = []
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        m, idx = fit_fold(X[tr], y[tr], i)
        preds[i] = m.predict(X[i:i + 1, idx])[0]
        chosen.append({"alpha": m.alpha_, "l1_ratio": m.l1_ratio_,
                       "n_nonzero": int((m.coef_ != 0).sum())})
        if (i + 1) % 10 == 0:
            print(f"      fold {i+1}/{n} ... alpha={m.alpha_:.4f} "
                  f"nonzero={chosen[-1]['n_nonzero']}", flush=True)

    pred_age = inv_transform_age(preds)
    r = float(np.corrcoef(age, pred_age)[0, 1])
    r_t = float(np.corrcoef(y, preds)[0, 1])
    med_ae = float(np.median(np.abs(age - pred_age)))
    print(f"    LOO-CV: r={r:.3f} (transformed scale r={r_t:.3f}), "
          f"medianAE={med_ae:.2f} y")

    # Permutation null on the transformed scale (cheaper: 5-fold, not LOO)
    print(f"    permutation null ({n_perm} shuffles, 5-fold CV) ...")
    null_r = []
    kf = KFold(5, shuffle=True, random_state=0)
    alpha_med = float(np.median([c["alpha"] for c in chosen]))
    sd_all = X.std(axis=0)
    idx_all = np.argpartition(sd_all, -TOPK)[-TOPK:]
    Xs = X[:, idx_all]
    for b in range(n_perm):
        yp = RNG.permutation(y)
        pr = np.empty(n)
        for tr, te in kf.split(Xs):
            m = ElasticNet(alpha=alpha_med, l1_ratio=0.5, max_iter=2000,
                           tol=1e-3, selection="random", random_state=b)
            m.fit(Xs[tr], yp[tr])
            pr[te] = m.predict(Xs[te])
        null_r.append(np.corrcoef(yp, pr)[0, 1] if np.std(pr) > 1e-12 else 0.0)
        if (b + 1) % 25 == 0:
            print(f"      perm {b+1}/{n_perm}", flush=True)
    null_r = np.array(null_r)
    p_emp = float((np.sum(null_r >= r_t) + 1) / (n_perm + 1))
    print(f"    null r: mean={null_r.mean():.3f} sd={null_r.std():.3f} "
          f"95th pct={np.percentile(null_r,95):.3f} -> empirical p={p_emp:.4f}")

    # Final model on all data, for coefficient inspection
    final = ElasticNetCV(l1_ratio=l1_ratios, alphas=25,
                         cv=KFold(5, shuffle=True, random_state=42),
                         max_iter=5000, tol=1e-4, n_jobs=-1,
                         selection="random", random_state=42)
    final.fit(Xs, y)
    nz = final.coef_ != 0
    coefs = pd.DataFrame({"CGid": cg_ids[idx_all][nz], "coef": final.coef_[nz]})
    coefs = coefs.reindex(coefs.coef.abs().sort_values(ascending=False).index)
    print(f"    final model: {nz.sum()} CpGs selected "
          f"(alpha={final.alpha_:.4f}, l1_ratio={final.l1_ratio_})")

    # Overlap with the published universal clocks
    pub = set()
    for k in [1, 2, 3]:
        c = pd.read_csv(PROC / f"universal_clock{k}.csv")
        pub |= set(c.CGid) - {"Intercept"}
    sel = set(coefs.CGid)
    ov = sel & pub
    # hypergeometric test: is the overlap more than chance?
    M, K, N_ = len(cg_ids), len(pub & set(cg_ids)), len(sel)
    p_hyper = float(stats.hypergeom.sf(len(ov) - 1, M, K, N_))
    exp_ov = N_ * K / M
    print(f"    overlap with published clock CpGs: {len(ov)}/{len(sel)} "
          f"(expected {exp_ov:.1f} by chance, hypergeometric p={p_hyper:.2e})")

    res = pd.DataFrame({"Basename": samples.Basename, "Age": age,
                        "PredAge_scratch": pred_age})
    res.to_csv(PROC / "clock_predictions_scratch.csv", index=False)
    coefs.to_csv(PROC / "clock_scratch_coefficients.csv", index=False)

    return {
        "loo_pearson_r": r,
        "loo_pearson_r_transformed": r_t,
        "loo_median_AE_years": med_ae,
        "loo_mean_AE_years": float(np.mean(np.abs(age - pred_age))),
        "n_cpgs_selected": int(nz.sum()),
        "alpha": float(final.alpha_),
        "l1_ratio": float(final.l1_ratio_),
        "permutation": {"n": n_perm, "null_mean_r": float(null_r.mean()),
                        "null_sd_r": float(null_r.std()),
                        "null_p95_r": float(np.percentile(null_r, 95)),
                        "empirical_p": p_emp},
        "overlap_with_published": {
            "n_selected": len(sel), "n_overlap": len(ov),
            "expected_by_chance": float(exp_ov), "hypergeom_p": p_hyper,
            "cpgs": sorted(ov)[:50]},
        "median_nonzero_across_folds": float(np.median([c["n_nonzero"] for c in chosen])),
    }


def require_betas():
    """The beta matrix is derived, not versioned. Fail with instructions."""
    p = PROC / "dolphin_betas.parquet"
    if not p.exists():
        raise SystemExit(
            "\ndolphin_betas.parquet is missing.\n"
            "It is regenerated from data/raw/mydata_GitHub.Rds and is not kept in\n"
            "git because it is 18 MB of derived data. Build it first:\n\n"
            "    python src/01_ingest.py\n")
    return pd.read_parquet(p)


def main():
    betas = require_betas()
    samples = pd.read_csv(PROC / "dolphin_samples.csv")
    traits = load_dolphin_traits()

    _, pub_stats = apply_published_clocks(betas, samples, traits)
    scratch = train_from_scratch(betas, samples)

    out = {"species": "Tursiops truncatus (bottlenose dolphin)",
           "tissue": "Blood", "n_samples": int(len(samples)),
           "n_cpgs": int(betas.shape[0]),
           "age_range_years": [float(samples.Age.min()), float(samples.Age.max())],
           "traits": traits,
           "published_clocks": pub_stats,
           "from_scratch_clock": scratch}
    (RES / "clock_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {RES/'clock_validation.json'}")


if __name__ == "__main__":
    main()
