# Findings

Pan-mammalian methylation clocks, the body-size question, and a measurement
artifact that consumes most of the answer.

All results come from the public Mammalian Methylation Consortium release.
Nothing on this page is simulated.

## Scope of the data

| Asset | Size | Use |
|---|---|---|
| Bottlenose dolphin beta matrix | 37,554 CpGs x 50 blood samples | Clock training and validation |
| Universal clock coefficients | 335, 816 and 760 CpGs | Published benchmark |
| Per-species methylation slopes | 126 species, 229 species-tissue records, 54 chromatin states | Cross-species rates |
| Life-history traits | 126 species with body mass and maximum lifespan | Allometry |
| CpG to gene annotation | 38,608 array CpGs aligned to human | Interpretation |

The species-level rates rest on 11,112 underlying arrays across 15 taxonomic
orders.

## 1. The clocks work, and they work honestly

Applying the three published universal clocks to the 50 dolphin samples, using
the consortium's own inverse transformations:

| Clock | Pearson r | Median absolute error | CpGs |
|---|---|---|---|
| Universal clock 1 | 0.847 | 2.84 y | 335 |
| Universal clock 2 | 0.948 | 1.91 y | 816 |
| Universal clock 3 | 0.911 | 2.11 y | 760 |

Discarding those coefficients and training an elastic net from scratch on the
same matrix, scored leave-one-out with the variance prefilter refit inside every
fold, gives **r = 0.915** and a median error of **2.49 years** from **31 CpGs**.
A 100-shuffle permutation test puts the null mean at r = -0.035 and returns an
empirical p of **0.0099**.

The convergence is the interesting part. Of the 31 CpGs the from-scratch model
selected, **7 also appear in the published universal clocks**, against **1.08
expected by chance** (hypergeometric p = 7.5e-05). Two independent fits land on
the same sites.

## 2. The rate measure is mostly a sampling artifact

The consortium's cross-species ageing rate, AROCM, is the slope of z-scored mean
methylation regressed on age, computed separately within each species. Reading
their `0_fns_v2.R::calslope2`, the standardisation happens inside each species
subset. For a signal of the form `meth = a + b*age + noise`, that makes the
fitted slope

```
AROCM = b / sqrt(b^2 * Var(age) + sigma^2)
```

which collapses to `AROCM ~ 1/SD(age)` whenever the age signal dominates the
noise. The measure is bounded above by the inverse of the sampled age spread.
Long-lived species are necessarily sampled across wider spans in years, so their
rate is pushed down for reasons unrelated to biology.

Empirically, the observed rate correlates **-0.889** with log sampled age span
against **-0.835** with log lifespan, and sits at a median of **58%** of its
mechanical ceiling.

### The four models

| Model | Controls | Mass coefficient | p |
|---|---|---|---|
| M1 | mass only | **-0.150** (SE 0.020) | 1.5e-11 |
| M2 | + sampled age span | -0.017 (SE 0.010) | 0.099 |
| M3 | + random intercept per Order | -0.023 (SE 0.016) | 0.168 |
| M5 | implied correlation, artifact-free | -0.004 (SE 0.004) | 0.361 |

The mass coefficient **shrinks by 88.9%** and stops being significant as soon as
the sampled span is included. M2 reaches R-squared 0.881, almost all of it from
the span term.

The decisive test: if the relationship were purely mechanical, the coefficient
on log span would be exactly -1. Observed it is **-0.960 (SE 0.039)**. Testing
against -1 gives t = +1.02, **p = 0.311**. The null that the span term is a pure
artifact cannot be rejected.

M3 also shows that taxonomic order accounts for a non-trivial share of what is
left (ICC = 0.164 across 9 orders, n = 120).

## 3. Almost nobody is a real outlier

Ranking residuals guarantees something sits at the extremes. The test is whether
a species deviates further than the spread of the others justifies, corrected
for running 126 tests at once. Using externally studentized residuals and
Benjamini-Hochberg:

**1 species out of 126** survives at q < 0.05: the northern giant mouse lemur
(*Mirza zaza*), t = -3.81, q = 0.028.

The famous candidates, before and after correction:

| Species | Naive residual | Corrected residual | q | Longevity quotient |
|---|---|---|---|---|
| Naked mole rat | -0.852 | **+0.237** | 0.976 | 2.93x |
| Bowhead whale | -1.526 | -0.573 | 0.976 | 2.39x |
| Human | -2.082 | -0.679 | 0.497 | 3.65x |
| Mouse | +0.878 | -0.052 | 0.995 | 0.34x |

The naked mole rat is the clearest casualty. Naively it looks like a slow ager,
which is the expected result. After correction it **changes sign** and sits
slightly on the fast side of average. Its apparent slowness came from being
sampled across 22.3 years, wide relative to a 35 gram animal, not from unusual
epigenetic stability.

The bowhead whale moves from -1.526 to -0.573. It stays on the slow side but is
nowhere near significance.

### Interpretation of the corrected residual

Because `AROCM = cor(methylation, age) / SD(age)`, removing the span term leaves
a quantity proportional to the correlation. The corrected residual therefore
measures **how tightly the epigenome tracks age**, that is clock fidelity, and
not how fast ageing proceeds. This distinction matters for any downstream claim.

## 4. Sanity checks pass

Two independent checks confirm the dataset itself is sound.

- **Kleiber's law.** Basal metabolic rate scales as mass^**0.742** across the 55
  species with metabolic data. The textbook exponent is 0.75.
- **Longevity quotient.** Observed lifespan divided by mass-predicted lifespan
  ranks human (3.65x), little brown bat (3.17x), greater mouse-eared bat (3.02x)
  and naked mole rat (2.93x) at the top, exactly where comparative biology puts
  them.

Lifespan scales as mass^0.131 (SE 0.012, R-squared 0.479), a little below the
usual 0.15 to 0.25 range, consistent with a captive and opportunistically
sampled species set.

The **rate-of-living hypothesis**, which holds that animals burning energy
faster should age faster, finds no support: mass-specific metabolic rate against
ageing rate gives beta = +0.027 (SE 0.049), p = 0.588.

## 5. Interpretable machine learning, and an honest failure

Dividing each chromatin state's rate by the species' own genome-wide mean
cancels the `1/SD(age)` term exactly, since it is common to every state in that
animal. What survives is the shape of the ageing signature rather than its
magnitude, and that shape is artifact-free.

Six of 54 chromatin states correlate with maximum lifespan at q < 0.05:

| State | Spearman rho | q | Biology |
|---|---|---|---|
| PromF5- | -0.322 | 0.013 | Promoter flanking |
| BivProm1+ | -0.285 | 0.019 | Bivalent promoter (polycomb) |
| BivProm1- | -0.281 | 0.019 | Bivalent promoter (polycomb) |
| PromF5+ | -0.274 | 0.019 | Promoter flanking |
| TxEnh5- | +0.272 | 0.019 | Transcribed enhancer |
| EnhA2- | +0.285 | 0.019 | Active enhancer |

**None of it generalises.** Holding out whole taxonomic orders, so the model must
predict a lineage it has never seen, out-of-sample performance is:

| Model | r | R-squared |
|---|---|---|
| Elastic net | -0.070 | -0.407 |
| Random forest | -0.115 | -0.264 |

The permutation test returns **p = 0.627**. The within-sample associations are
phylogenetic structure. Closely related species resemble each other in both
chromatin dynamics and lifespan, and a model fitted across orders absorbs that
without learning anything portable.

This is the result that is easiest to miss. Reporting the six significant states
and stopping would have produced a far more exciting and far less true story.

## 6. What does hold up: the CpGs themselves

The 1,303 CpGs used by the three universal clocks, tested against the full array
background:

| Context | Clock | Array | Odds ratio | q |
|---|---|---|---|---|
| 5' UTR | 9.4% | 7.0% | **1.38** | 0.003 |
| Exon | 32.8% | 28.6% | **1.21** | 0.003 |
| Promoter | 8.0% | 6.8% | 1.20 | 0.116 |
| Intergenic downstream | 4.6% | 6.9% | **0.65** | 0.003 |

The genes carrying the most clock CpGs are **PAX2** (9), **HOXB7** (9),
**NPAS3** (8), **CASZ1** (8), **ZFHX3** (7), **BCOR** (7), **EVX2** (6) and
**ZNF521** (6). These are developmental transcription factors and polycomb
targets, the same family every methylation clock ever built has converged on.
Polycomb is a protein complex that keeps developmental genes switched off in
adult tissue, and the sites it guards drift steadily with age across mammals.

## Summary

| Claim | Verdict |
|---|---|
| Methylation clocks predict age across mammals | **Holds.** r = 0.948 published, 0.915 retrained |
| Independent fits recover the same CpGs | **Holds.** 7 of 31, 1.08 expected, p = 7.5e-05 |
| Clock CpGs sit in developmental regulators | **Holds.** Enriched in 5' UTRs and exons |
| Epigenetic ageing rate scales with body size | **Fails.** 88.9% of it is a sampling artifact |
| Naked mole rats age slowly by this measure | **Fails.** Residual reverses sign once corrected |
| Bowhead whales age slowly | **Weak.** Survives correction, not significant |
| Chromatin-state signature predicts lifespan | **Fails.** No generalisation across orders |
| Rate of living predicts epigenetic ageing | **Fails.** p = 0.588 |

## What would fix it

The artifact is a property of the published summary tables, not of the
underlying biology. Estimating rates on a common relative-age grid, with the
sampled age range held constant across species, would remove it. That requires
the raw arrays in GEO accession GSE223748 rather than the per-species slopes
released on GitHub. The pipeline here is structured so that swapping in a real
multi-species beta matrix changes only `01_ingest.py`.

A second improvement would be a proper phylogenetic comparative method. Taxonomic
order as a random intercept is a coarse proxy for shared ancestry; a
time-calibrated mammal phylogeny with Pagel's lambda or a phylogenetic
generalised least squares fit would be the correct treatment.
