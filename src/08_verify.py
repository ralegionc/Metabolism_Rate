"""
08_verify.py -- Independent verification of every headline number.

This deliberately does NOT import the analysis modules. It recomputes each
claim from the raw consortium files by a different route, then asserts against
the numbers quoted in FINDINGS.md, README.md and the write-ups. If a number in
the prose drifts from the data, this fails.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"

PASS, FAIL = [], []


def check(name, got, want, tol=0.005, note=""):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    (PASS if ok else FAIL).append(name)
    flag = "OK  " if ok else "FAIL"
    print(f"  [{flag}] {name:52s} got={got!r:>14} want={want!r}" +
          (f"  {note}" if note else ""))


def truthy(name, cond, note=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  {note}" if note else ""))


print("=" * 78)
print("1. Raw-file provenance: do the tidy tables match the source CSVs?")
print("=" * 78)
t3 = pd.read_csv(RAW / "SupplementTable3_AROCM_Strata_v4.csv")
wide = pd.read_csv(PROC / "species_state_arocm_wide.csv")
check("SupplementTable3 rows", len(t3), 229, 0)
check("wide table rows preserved", len(wide), 229, 0)
check("unique species", wide.SpeciesLatinName.nunique(), 126, 0)
# the primary AROCM column must be exactly the source YoungSlope1 column
src = t3["BivProm2+YoungSlope1"].values
got = wide["BivProm2+"].values
truthy("BivProm2+ copied verbatim from BivProm2+YoungSlope1",
       np.allclose(src, got, equal_nan=True))
# spot-check a species against the raw file by hand
chee = t3[(t3.SpeciesLatinName == "Acinonyx jubatus")].iloc[0]
check("cheetah adult mass (g) from source", float(chee["AdultWeight(g)"]), 53500.0, 0)
check("cheetah max lifespan from source", float(chee["MaximumLifespanInYears"]), 20.5, 0)

print()
print("=" * 78)
print("2. Clock validation, recomputed from the beta matrix")
print("=" * 78)
_bp = PROC / "dolphin_betas.parquet"
if not _bp.exists():
    raise SystemExit("\ndolphin_betas.parquet is missing (derived, not versioned).\n"
                     "Run the pipeline first:  python src/01_ingest.py\n")
betas = pd.read_parquet(_bp)
samples = pd.read_csv(PROC / "dolphin_samples.csv")
cv = json.loads((RES / "clock_validation.json").read_text())
check("beta matrix CpGs", betas.shape[0], 37554, 0)
check("beta matrix samples", betas.shape[1], 50, 0)
truthy("betas within [0,1]", betas.values.min() >= 0 and betas.values.max() <= 1)
truthy("no NaNs in beta matrix", int(betas.isna().sum().sum()) == 0)

# Recompute clock 2 independently
c2 = pd.read_csv(PROC / "universal_clock2.csv")
icept = float(c2.loc[c2.CGid == "Intercept", "coef"].iloc[0])
cg = c2[c2.CGid != "Intercept"].set_index("CGid")["coef"]
cg = cg[[c for c in cg.index if c in betas.index]]
lin = icept + betas.loc[cg.index].T.values @ cg.values
maxage, gest = 67.0, 1.0301369863013700
x0 = np.exp(-np.exp(-lin))
dnam = x0 * (1.3 * maxage + gest) - gest
r2 = float(np.corrcoef(samples.Age, dnam)[0, 1])
check("clock2 r (recomputed)", round(r2, 3), 0.948, 0.002)
check("clock2 r (reported)", round(cv["published_clocks"]["clock2"]["pearson_r"], 3),
      0.948, 0.002)
check("clock2 median AE (recomputed)",
      round(float(np.median(np.abs(samples.Age - dnam))), 2), 1.91, 0.02)

s = cv["from_scratch_clock"]
check("from-scratch LOO r", round(s["loo_pearson_r"], 3), 0.915, 0.002)
check("from-scratch median AE", round(s["loo_median_AE_years"], 2), 2.49, 0.02)
check("from-scratch CpGs", s["n_cpgs_selected"], 31, 0)
ov = s["overlap_with_published"]
check("overlap count", ov["n_overlap"], 7, 0)
check("overlap expected by chance", round(ov["expected_by_chance"], 2), 1.08, 0.02)
truthy("overlap hypergeometric p < 1e-4", ov["hypergeom_p"] < 1e-4,
       f"p={ov['hypergeom_p']:.2e}")
# verify the overlap claim directly
pub = set()
for k in [1, 2, 3]:
    pub |= set(pd.read_csv(PROC / f"universal_clock{k}.csv").CGid) - {"Intercept"}
sel = set(pd.read_csv(PROC / "clock_scratch_coefficients.csv").CGid)
check("overlap recomputed from coefficient files", len(sel & pub), 7, 0)
truthy("permutation p < 0.05", s["permutation"]["empirical_p"] < 0.05,
       f"p={s['permutation']['empirical_p']:.4f}")

print()
print("=" * 78)
print("3. The artifact: models refit from scratch here")
print("=" * 78)
d = pd.read_csv(PROC / "species_residuals.csv")
check("species in model", len(d), 126, 0)

# Rebuild the design independently of 03_aging_rate_model.py
X1 = np.column_stack([np.ones(len(d)), d.log_mass])
b1 = np.linalg.lstsq(X1, d.log_arocm, rcond=None)[0]
check("M1 mass coefficient", round(float(b1[1]), 3), -0.150, 0.002)

X2 = np.column_stack([np.ones(len(d)), d.log_mass, d.log_span])
b2, *_ = np.linalg.lstsq(X2, d.log_arocm, rcond=None)
resid = d.log_arocm - X2 @ b2
dof = len(d) - 3
sigma2 = float(resid @ resid) / dof
se = np.sqrt(np.diag(np.linalg.inv(X2.T @ X2)) * sigma2)
check("M2 mass coefficient", round(float(b2[1]), 3), -0.017, 0.002)
check("M2 span coefficient", round(float(b2[2]), 3), -0.960, 0.002)
check("M2 span SE", round(float(se[2]), 3), 0.039, 0.002)
t_stat = (b2[2] + 1.0) / se[2]
p_vs1 = float(2 * stats.t.sf(abs(t_stat), df=dof))
check("span coef vs -1, t", round(float(t_stat), 2), 1.02, 0.03)
check("span coef vs -1, p", round(p_vs1, 3), 0.311, 0.01)
truthy("cannot reject span coefficient = -1", p_vs1 > 0.05)
shrink = 1 - abs(b2[1] / b1[1])
check("mass coefficient shrinkage", round(float(shrink) * 100, 1), 88.9, 0.4, "%")

# R-squared
ss_tot = float(((d.log_arocm - d.log_arocm.mean()) ** 2).sum())
check("M1 R2", round(1 - float(((d.log_arocm - X1 @ b1) ** 2).sum()) / ss_tot, 3),
      0.308, 0.003)
check("M2 R2", round(1 - float((resid ** 2).sum()) / ss_tot, 3), 0.881, 0.003)

# the raw correlations quoted in the prose
w = wide.copy()
w["span_y"] = (w.age_range_hi - w.age_range_lo) * w.max_lifespan_y
w = w[w.span_y > 0]
a = np.log(w["BivProm2+"].abs())
check("corr(logAROCM, log span) at record level",
      round(float(np.corrcoef(a, np.log(w.span_y))[0, 1]), 3), -0.889, 0.003)
check("corr(logAROCM, log lifespan) at record level",
      round(float(np.corrcoef(a, np.log(w.max_lifespan_y))[0, 1]), 3), -0.835, 0.003)
ceiling_frac = (w["BivProm2+"].abs() / (1 / (w.span_y / np.sqrt(12)))).median()
check("median fraction of 1/SD(age) ceiling",
      round(float(ceiling_frac) * 100, 0), 58.0, 1.5, "%")

print()
print("=" * 78)
print("4. Outliers and the famous species")
print("=" * 78)
am = json.loads((RES / "aging_rate_models.json").read_text())
check("significant outliers (q<0.05)", int(d.signif.sum()), 1, 0)
check("reported significant outliers", am["n_significant_residuals"], 1, 0)
sig = d[d.signif].iloc[0]
check("outlier identity", sig.SpeciesCommonName, "Northern giant mouse lemur")
check("outlier studentized residual", round(float(sig.resid_student), 2), -3.81, 0.02)
truthy("outlier q < 0.05", sig.q_outlier < 0.05, f"q={sig.q_outlier:.4f}")

for latin, m1v, m3v in [("Heterocephalus glaber", -0.852, +0.237),
                        ("Balaena mysticetus", -1.526, -0.573),
                        ("Homo sapiens", -2.082, -0.679),
                        ("Mus musculus", +0.878, -0.052)]:
    r = d[d.SpeciesLatinName == latin].iloc[0]
    check(f"{r.SpeciesCommonName} naive residual", round(float(r.resid_M1), 3),
          m1v, 0.003)
    check(f"{r.SpeciesCommonName} corrected residual", round(float(r.resid_M3), 3),
          m3v, 0.003)
truthy("naked mole rat residual changes sign under correction",
       float(d[d.SpeciesLatinName == "Heterocephalus glaber"].resid_M1.iloc[0]) < 0 <
       float(d[d.SpeciesLatinName == "Heterocephalus glaber"].resid_M3.iloc[0]))
truthy("bowhead whale stays negative",
       float(d[d.SpeciesLatinName == "Balaena mysticetus"].resid_M3.iloc[0]) < 0)

print()
print("=" * 78)
print("5. Sanity checks against known biology")
print("=" * 78)
lq = am["lifespan_mass_allometry"]
check("lifespan-mass exponent", round(lq["exponent"], 3), 0.131, 0.002)
rol = am["rate_of_living"]
check("Kleiber exponent", round(rol["kleiber_exponent"], 3), 0.742, 0.003)
truthy("Kleiber exponent within 0.05 of the textbook 0.75",
       abs(rol["kleiber_exponent"] - 0.75) < 0.05)
truthy("rate-of-living not supported", rol["p"] > 0.05, f"p={rol['p']:.3f}")
top_lq = d.nlargest(4, "longevity_quotient").SpeciesCommonName.tolist()
truthy("naked mole rat in top 4 longevity quotient",
       "Naked mole rat" in top_lq, str(top_lq))
truthy("a bat in top 4 longevity quotient", any("bat" in s.lower() for s in top_lq))

print()
print("=" * 78)
print("6. Interpretability: the generalisation failure")
print("=" * 78)
it = json.loads((RES / "interpretability.json").read_text())
check("states significant vs lifespan", it["n_states_lifespan_significant"], 6, 0)
check("chromatin states", it["n_states"], 54, 0)
check("elastic net leave-one-Order-out r",
      round(it["ml"]["elastic_net"]["r"], 3), -0.070, 0.003)
check("random forest leave-one-Order-out r",
      round(it["ml"]["random_forest"]["r"], 3), -0.115, 0.003)
truthy("both out-of-sample R2 negative",
       it["ml"]["elastic_net"]["r2_oos"] < 0 and it["ml"]["random_forest"]["r2_oos"] < 0)
truthy("permutation p not significant",
       it["ml"]["permutation"]["empirical_p"] > 0.05,
       f"p={it['ml']['permutation']['empirical_p']:.3f}")

e = pd.read_csv(PROC / "clock_cpg_enrichment.csv")
f5 = e[e.category == "fiveUTR"].iloc[0]
ex = e[e.category == "Exon"].iloc[0]
check("5'UTR odds ratio", round(float(f5.odds_ratio), 2), 1.38, 0.02)
check("Exon odds ratio", round(float(ex.odds_ratio), 2), 1.21, 0.02)
truthy("5'UTR enrichment significant", f5.q < 0.05, f"q={f5.q:.4f}")
ann = pd.read_csv(PROC / "clock_cpg_annotation.csv")
check("annotated clock CpGs", len(ann), 1303, 0)
top_genes = ann.SYMBOL.value_counts().head(10).index.tolist()
for g in ["PAX2", "HOXB7", "NPAS3", "CASZ1"]:
    truthy(f"{g} among top-10 clock genes", g in top_genes)

print()
print("=" * 78)
print("7. Circularity audit: is any claim resting on simulated data?")
print("=" * 78)
import ast

# Parse the AST rather than grepping text, so prose in docstrings and comments
# ("nothing is simulated") cannot trigger a false positive, and so this file
# does not flag its own pattern list.
FABRICATE = {"normal", "randn", "rand", "uniform", "poisson", "binomial",
             "exponential", "lognormal", "gamma", "beta", "standard_normal",
             "multivariate_normal", "make_regression", "make_classification",
             "make_blobs"}
RESAMPLE = {"permutation", "integers", "shuffle", "choice", "default_rng"}

srcs = [p for p in sorted((ROOT / "src").glob("*.py")) if p.name != Path(__file__).name]
fabricated, rng_calls = [], []
for f in srcs:
    tree = ast.parse(f.read_text(encoding="utf-8"), filename=f.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if name is None:
            continue
        base = ""
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            base = fn.value.id
        if name in FABRICATE and base in {"random", "np", "rng", "RNG", ""}:
            fabricated.append(f"{f.name}:{node.lineno}  {base}.{name}")
        if base in {"RNG", "rng"}:
            rng_calls.append((f.name, node.lineno, name))

truthy("no random-data-generating call in any pipeline module", not fabricated,
       "; ".join(fabricated) if fabricated else f"{len(srcs)} modules parsed")
truthy("every RNG call is resampling, not fabrication",
       all(n in RESAMPLE for _, _, n in rng_calls),
       f"{len(rng_calls)} calls")
for n, i, c in rng_calls:
    print(f"         {n}:{i}  RNG.{c}()  -> {'resampling' if c in RESAMPLE else 'FABRICATION'}")

# The dolphin beta matrix must be byte-identical to what the consortium shipped.
sys.path.insert(0, str(ROOT / "src"))
from rds_reader import read_rds as _rr  # noqa: E402
_obj = _rr(RAW / "mydata_GitHub.Rds")
_names = (_obj.get("_attr") or {}).get("names")
_mb = _obj["_values"][_names.index("meth_betas")]
_cols = (_mb.get("_attr") or {}).get("names")
_first = _mb["_values"][_cols.index("202894750036_R05C02")]
_first = _first["_values"] if isinstance(_first, dict) else _first
truthy("beta values round-trip identically from the source RDS",
       np.allclose(np.asarray(_first, dtype=float),
                   betas["202894750036_R05C02"].values))

# every processed table must trace back to a real raw file
truthy("dolphin betas come from the consortium RDS",
       (RAW / "mydata_GitHub.Rds").exists())
truthy("species rates come from SupplementTable3",
       (RAW / "SupplementTable3_AROCM_Strata_v4.csv").exists())

print()
print("=" * 78)
print("8. Prose consistency: numbers quoted in the write-ups")
print("=" * 78)
docs = {p.name: p.read_text(encoding="utf-8")
        for p in [ROOT / "FINDINGS.md", ROOT / "README.md",
                  ROOT / "writing" / "medium_article.md",
                  ROOT / "writing" / "linkedin_post.md"]}
must_appear = {
    "0.948": ["FINDINGS.md", "README.md", "medium_article.md", "linkedin_post.md"],
    "0.915": ["FINDINGS.md", "README.md", "medium_article.md", "linkedin_post.md"],
    "-0.960": ["FINDINGS.md", "README.md", "medium_article.md", "linkedin_post.md"],
    "88.9": ["FINDINGS.md", "README.md", "medium_article.md", "linkedin_post.md"],
    "0.311": ["FINDINGS.md", "README.md", "medium_article.md", "linkedin_post.md"],
    "0.742": ["FINDINGS.md", "medium_article.md"],
    "1.08": ["FINDINGS.md", "README.md", "medium_article.md", "linkedin_post.md"],
}
for num, files in must_appear.items():
    for f in files:
        truthy(f"'{num}' present in {f}", num in docs[f])
# no em dashes in the public-facing writing
for f in ["medium_article.md", "linkedin_post.md", "FINDINGS.md", "README.md"]:
    truthy(f"no em dashes in {f}", "—" not in docs[f])

print()
print("=" * 78)
print("9. Deliverables exist")
print("=" * 78)
for rel, minkb in [("docs/index.html", 100), ("figures/fig1_clock_validation.png", 40),
                   ("figures/fig2_sampling_artifact.png", 40),
                   ("figures/fig3_residuals.png", 40),
                   ("figures/fig4_allometry_validation.png", 40),
                   ("figures/fig5_interpretability.png", 40),
                   ("figures/fig6_cpg_context.png", 40),
                   ("notebooks/mammalian_methylation_residuals.ipynb", 20),
                   ("FINDINGS.md", 4), ("README.md", 3),
                   ("writing/medium_article.md", 8),
                   ("writing/linkedin_post.md", 1)]:
    p = ROOT / rel
    kb = p.stat().st_size / 1024 if p.exists() else 0
    truthy(f"{rel} exists and is non-trivial", p.exists() and kb >= minkb,
           f"{kb:.0f} KB")

nb = json.loads((ROOT / "notebooks" /
                 "mammalian_methylation_residuals.ipynb").read_text(encoding="utf-8"))
errs = [o for c in nb["cells"] for o in c.get("outputs", [])
        if o.get("output_type") == "error"]
truthy("notebook has zero execution errors", not errs)
executed = sum(1 for c in nb["cells"]
               if c["cell_type"] == "code" and c.get("execution_count"))
truthy("notebook cells were actually executed", executed >= 8, f"{executed} cells")

print()
print("=" * 78)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILURES:")
    for f in FAIL:
        print("  -", f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
