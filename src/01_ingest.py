"""
01_ingest.py -- Extract and tidy the real Mammalian Methylation Consortium data.

Sources (all real, all from the consortium's public GitHub release):
  * mydata_GitHub.Rds ......... 37,554 CpG x 50 bottlenose-dolphin beta matrix,
                                sample ages, and the three published universal
                                pan-mammalian clock coefficient sets
  * SupplementTable3 .......... per species-tissue methylation slopes (AROCM)
                                across 55 chromatin states, plus life-history
                                traits (adult weight, max lifespan, maturity)
  * SupplementTable4 .......... headline AROCM per species with lifespan
  * SupplementTable2 .......... chromatin-state metadata
  * Homo_sapiens annotation ... CpG -> gene symbol / genomic context

AROCM semantics, read off the consortium's own R code (0_fns_v2.R::calslope2):
for each species-tissue and each chromatin state, mean methylation across the
state's CpGs is z-scored across samples, then regressed on chronological age.
"YoungSlope{p}" restricts to samples with age <= p * maxLifespan. Units are
therefore SD-of-methylation per year. Because the z-scoring happens *within*
each subset, slope magnitude depends on the age range sampled -- handled
explicitly downstream.

Outputs land in data/processed/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rds_reader import read_rds  # noqa: E402

# Path to the cloned consortium repo (only needed for the large annotation file)
CONSORTIUM_REPO = Path("/tmp/mmc")


def _rcol(col):
    """Unwrap a parsed R column (handles factors and attributed vectors)."""
    if isinstance(col, dict) and "_values" in col:
        attr = col.get("_attr") or {}
        levels = attr.get("levels")
        if levels is not None:
            lv = levels["_values"] if isinstance(levels, dict) else levels
            return [None if v is None else lv[v - 1] for v in col["_values"]]
        return col["_values"]
    return col


def rdf(node):
    """Parsed R data.frame -> pandas.DataFrame."""
    names = (node.get("_attr") or {}).get("names")
    return pd.DataFrame({n: pd.Series(_rcol(c)) for n, c in zip(names, node["_values"])})


def sub(node, key):
    """Fetch a named element out of a parsed R list."""
    names = (node.get("_attr") or {}).get("names")
    return node["_values"][names.index(key)]


# ---------------------------------------------------------------------------
# 1. Dolphin methylation matrix + published clocks
# ---------------------------------------------------------------------------
def ingest_rds():
    print("[1] Parsing mydata_GitHub.Rds ...")
    obj = read_rds(RAW / "mydata_GitHub.Rds")

    samples = rdf(sub(obj, "SampleAnnotation"))
    betas = rdf(sub(obj, "meth_betas"))
    betas = betas.set_index("CGid")
    betas.index.name = "CGid"

    # Align sample order to the beta-matrix column order
    samples = (samples.set_index("Basename")
               .reindex(list(betas.columns))
               .rename_axis("Basename")
               .reset_index())
    samples["Age"] = pd.to_numeric(samples["Age"])
    samples["Female"] = pd.to_numeric(samples["Female"]).astype(int)

    print(f"    betas: {betas.shape[0]:,} CpGs x {betas.shape[1]} samples")
    print(f"    species: {samples.SpeciesCommonName.unique().tolist()}")
    print(f"    tissues: {samples.Tissue.value_counts().to_dict()}")
    print(f"    age range: {samples.Age.min():.2f} - {samples.Age.max():.2f} y "
          f"(median {samples.Age.median():.2f})")
    assert betas.isna().sum().sum() == 0, "unexpected NaNs in beta matrix"
    assert betas.values.min() >= 0 and betas.values.max() <= 1, "betas outside [0,1]"

    clocks = sub(obj, "ThreeUniversalPanMammalianClocks")
    clock_frames = {}
    for name in ["clock1", "clock2", "clock3"]:
        c = rdf(sub(clocks, name))
        c.columns = ["index", "CGid", "coef"]
        clock_frames[name] = c[["CGid", "coef"]]
        print(f"    {name}: {len(c) - 1} CpGs + intercept")

    anage = rdf(sub(obj, "anAge"))
    print(f"    anAge: {anage.shape[0]:,} species x {anage.shape[1]} traits")

    betas.to_parquet(OUT / "dolphin_betas.parquet")
    samples.to_csv(OUT / "dolphin_samples.csv", index=False)
    for name, c in clock_frames.items():
        c.to_csv(OUT / f"universal_{name}.csv", index=False)
    keep = ["SpeciesLatinName", "Common.name", "Order", "Family", "Genus",
            "maxAgeCaesar" if "maxAgeCaesar" in anage.columns else "maxAge",
            "weightCaesar", "averagedMaturity.yrs", "GestationTimeInYears",
            "Metabolic.rate..W.", "Body.mass..g.", "Temperature..K."]
    keep = [k for k in keep if k in anage.columns]
    anage[keep].to_csv(OUT / "anage_traits.csv", index=False)
    return betas, samples, clock_frames


# ---------------------------------------------------------------------------
# 2. Per-species aging rates across chromatin states
# ---------------------------------------------------------------------------
def ingest_aging_rates():
    print("\n[2] Tidying per-species aging rates (SupplementTable3) ...")
    t3 = pd.read_csv(RAW / "SupplementTable3_AROCM_Strata_v4.csv")
    states_meta = pd.read_csv(RAW / "SupplementTable2_ChromatinStates_v2.csv", index_col=0)

    meta_cols = ["SpeciesLatinName", "SpeciesCommonName", "Tissue",
                 "MammalNumberHorvath", "Order", "AdultWeight(g)",
                 "GestationTimeInYears", "AgeAtSexualMaturityInYearsAveragedOverSexes",
                 "MaximumLifespanInYears", "AgeRangeLowerLimit", "AgeRangeUpperLimit",
                 "Freq"]
    meta = t3[meta_cols].copy()
    meta = meta.rename(columns={
        "AdultWeight(g)": "adult_mass_g",
        "MaximumLifespanInYears": "max_lifespan_y",
        "AgeAtSexualMaturityInYearsAveragedOverSexes": "maturity_y",
        "GestationTimeInYears": "gestation_y",
        "Freq": "n_samples",
        "AgeRangeLowerLimit": "age_range_lo",
        "AgeRangeUpperLimit": "age_range_hi",
    })

    # Slope columns look like "<State><Young|Old>Slope<prop>"
    slope_cols = [c for c in t3.columns if "Slope" in c]
    recs = []
    for c in slope_cols:
        if "YoungSlope" in c:
            state, prop = c.split("YoungSlope")
            window = "young"
        elif "OldSlope" in c:
            state, prop = c.split("OldSlope")
            window = "old"
        else:
            continue
        recs.append((c, state, window, float(prop)))
    colmap = pd.DataFrame(recs, columns=["col", "state", "window", "prop"])
    print(f"    {len(colmap)} slope columns = {colmap.state.nunique()} states "
          f"x {len(colmap[['window','prop']].drop_duplicates())} age windows")

    # Primary AROCM: the full-range young window (age <= 1.0 * max lifespan)
    primary = colmap[(colmap.window == "young") & (colmap.prop == 1.0)]
    wide = meta.copy()
    for _, r in primary.iterrows():
        wide[r.state] = t3[r.col]

    state_cols = primary.state.tolist()
    long = wide.melt(id_vars=meta.columns.tolist(), value_vars=state_cols,
                     var_name="state", value_name="arocm")
    long["state_base"] = long.state.str.replace(r"[+-]$", "", regex=True)
    long["direction"] = np.where(long.state.str.endswith("+"), "hyper",
                                 np.where(long.state.str.endswith("-"), "hypo", "none"))

    print(f"    tidy long table: {len(long):,} rows "
          f"({wide.SpeciesLatinName.nunique()} species, {len(state_cols)} states)")

    # Headline table from SupplementTable4 (species level, with lifespan)
    t4 = pd.read_csv(RAW / "SupplementTable4_SlopesBySpecies_BivProm2+.csv", index_col=0)
    t4 = t4.rename(columns={
        "AROCM_BivProm2+": "arocm_bivprom2",
        "AdjAROCM_BivProm2+": "adj_arocm_bivprom2",
        "Lifespan": "max_lifespan_y",
    })
    print(f"    SupplementTable4: {len(t4)} species-level AROCM records")

    wide.to_csv(OUT / "species_state_arocm_wide.csv", index=False)
    long.to_csv(OUT / "species_state_arocm_long.csv", index=False)
    t4.to_csv(OUT / "species_arocm_headline.csv", index=False)
    states_meta.to_csv(OUT / "chromatin_states_meta.csv", index=False)

    # Also keep every age-window slope for the BivProm2+ state, so the
    # age-range sensitivity of AROCM can be checked downstream.
    biv = colmap[colmap.state == "BivProm2+"]
    biv_windows = meta.copy()
    for _, r in biv.iterrows():
        biv_windows[f"{r.window}_{r.prop}"] = t3[r.col]
    biv_windows.to_csv(OUT / "bivprom2_age_windows.csv", index=False)

    return wide, long, t4


# ---------------------------------------------------------------------------
# 3. CpG -> gene annotation, restricted to the union of clock CpGs
# ---------------------------------------------------------------------------
def annotation_source():
    """Prefer the vendored slim annotation; fall back to the full clone.

    The consortium's human alignment is a 20 MB column-rich CSV inside a 3.6 GB
    repository. data/raw/array_cpg_annotation_slim.csv.gz holds the same 38,608
    CpGs with only the columns this analysis uses, at 0.8 MB, so the pipeline
    runs from a clone of this repo alone.
    """
    slim = RAW / "array_cpg_annotation_slim.csv.gz"
    if slim.exists():
        return slim, "vendored slim annotation"
    full = (CONSORTIUM_REPO / "Annotations, Amin Haghani" / "Mammals" /
            "Homo_sapiens.hg38.HorvathMammalMethylChip40.v1.csv")
    if full.exists():
        return full, "full consortium annotation"
    return None, None


def ingest_annotation(clock_frames):
    print("\n[3] Annotating clock CpGs with genes ...")
    ann_path, which = annotation_source()
    if ann_path is None:
        print("    !! no CpG annotation available; skipping")
        return None
    print(f"    source: {which}")

    wanted = set()
    for c in clock_frames.values():
        wanted |= set(c.CGid) - {"Intercept"}
    print(f"    {len(wanted):,} unique clock CpGs to annotate")

    cols = ["CGid", "SYMBOL", "GENENAME", "annotation", "distanceToTSS",
            "main_Categories", "seqnames", "start", "genic", "Promoter"]
    ann = pd.read_csv(ann_path, usecols=lambda c: c in cols, low_memory=False)
    ann = ann[ann.CGid.isin(wanted)].drop_duplicates("CGid")
    print(f"    matched {len(ann):,} / {len(wanted):,} CpGs in the human alignment")
    ann.to_csv(OUT / "clock_cpg_annotation.csv", index=False)
    return ann


def main():
    betas, samples, clocks = ingest_rds()
    wide, long, t4 = ingest_aging_rates()
    ann = ingest_annotation(clocks)

    manifest = {
        "dolphin_betas": {"cpgs": int(betas.shape[0]), "samples": int(betas.shape[1])},
        "dolphin_age_range_y": [float(samples.Age.min()), float(samples.Age.max())],
        "universal_clocks": {k: int(len(v) - 1) for k, v in clocks.items()},
        "species_with_arocm": int(wide.SpeciesLatinName.nunique()),
        "species_tissue_records": int(len(wide)),
        "chromatin_states": int(long.state.nunique()),
        "orders": int(wide.Order.nunique()),
        "annotated_clock_cpgs": int(len(ann)) if ann is not None else 0,
    }
    (OUT / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n[done] manifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
