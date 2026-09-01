# The clock is real. The speedometer is not.

## What happened when I tried to find out which mammals age faster than their body size predicts

---

There is a very good idea in ageing research, and there is a very tempting idea
sitting immediately next to it. They are easy to confuse. I spent a while
confusing them, and this is a description of how I stopped.

The very good idea is the DNA methylation clock. **DNA methylation** is a
chemical tag, a methyl group, attached to the DNA molecule at specific positions
called **CpG sites**, which are places where a cytosine base sits next to a
guanine base. These tags do not change the genetic code. They change how
readable a stretch of DNA is. Some of them change in remarkably predictable ways
as an organism gets older, and if you measure a few hundred of them and fit a
regression, you can estimate someone's age from a blood sample to within a few
years.

Steve Horvath built the first of these for humans in 2013. The Mammalian
Methylation Consortium then did something more ambitious: they built an array
that measures 37,554 CpG sites conserved across mammals, profiled thousands of
animals from 185 species, and fitted a single clock that works on all of them.
Not one clock per species. One clock, for mice and elephants and bats and
whales.

The tempting idea is what you do next. If a clock tells you how fast an animal
is ageing, and you know that big animals live longer than small ones, then the
residual, meaning the part of the ageing rate that body size does not explain,
should tell you which species have unusual biology. Naked mole rats live thirty
years at a body weight of thirty five grams. Bowhead whales reach two hundred.
Those two should light up.

I set out to build that analysis on the consortium's real data. What I found
instead was that the residual was mostly measuring how the study was designed.

---

## Getting to actual data

The full array data lives in the Gene Expression Omnibus under accession
GSE223748. I could not reach it from where I was working, which turned out to be
useful, because it forced me to look at what the consortium publishes on GitHub
instead. That repository is three and a half gigabytes and contains more than I
expected:

- A real methylation matrix: 37,554 CpG sites by 50 bottlenose dolphin blood
  samples, with ages from 0.57 to 57.9 years.
- The exact coefficients of all three published universal clocks.
- Per-species rates of methylation change for 126 species, broken down by 54
  chromatin states, alongside body mass and maximum lifespan.
- An annotation mapping every array CpG to a human gene.

The first obstacle was that several of these files are R `.RDS` files, and the
standard Python reader for them, `pyreadr`, returns an empty dictionary without
raising an error when the top-level object is a plain list rather than a data
frame. Which is exactly how the consortium ships the important one.

R's serialization format is documented, so I wrote a reader. It handles the
subset of the grammar those files actually use: attributed vectors, pairlists
for attributes, the reference table for repeated symbols, and the compact
integer sequences R uses for row names.

```python
from rds_reader import read_rds
obj = read_rds("mydata_GitHub.Rds")
# -> SampleAnnotation, meth_betas, anAge, ThreeUniversalPanMammalianClocks
```

That opened everything. It is about 200 lines and is in the repository if
anybody else hits the same wall.

---

## First, check the clock actually works

Before questioning anything, reproduce it.

I transcribed the consortium's own inverse transformations from their R code.
Clock 2, for instance, predicts a quantity that has to be pushed back through a
double exponential and then rescaled by the species' maximum lifespan and
gestation time:

```python
def f2_antitrans_clock2(y, max_age, gestation):
    x0 = np.exp(-np.exp(-y))
    return x0 * (max_age + gestation) - gestation
```

Applied to the 50 dolphins, universal clock 2 gives a correlation of **0.948**
with true age and a median error of **1.91 years**. That matches the figure in
the published paper. The transcription is correct.

Then I threw the published coefficients away and trained a clock from scratch on
the same matrix. Fifty samples against 37,554 sites is precisely the situation
where cross-validation flatters you, so two precautions mattered. First, scoring
is leave-one-out, meaning each animal is predicted by a model that never saw it.
Second, the step that filters down to the most variable sites is **refit inside
every fold** and never sees the ages. A filter fitted once on all the data, using
the labels, is the single most common way methylation papers overstate their
accuracy.

The result: **r = 0.915**, median error **2.49 years**, from 31 CpG sites. A
permutation test, where the ages are shuffled 100 times and the whole procedure
rerun, puts the null correlation at -0.035 and returns p = 0.0099.

The part I did not expect: of those 31 sites, **7 also appear in the published
universal clocks**. By chance you would expect 1.08. The hypergeometric p is
7.5e-05. Two entirely independent fits, one by the consortium across thousands
of animals and one by me on fifty dolphins, converge on the same specific
positions in the genome.

That is a real result and it is worth holding onto, because everything after
this gets more sceptical.

---

## The measure that ate the analysis

To compare ageing rates across species, the consortium uses a quantity called
AROCM, the average rate of change of methylation. Reading their function
`calslope2` in `0_fns_v2.R`, it is computed like this: take the average
methylation across all the CpG sites in one chromatin state, standardise that
average across the samples of a given species, then regress it on age.

**Chromatin state** means the functional category of a region of genome,
inferred from the protein marks around it: bivalent promoter, active enhancer,
quiescent, and so on. **Standardise** means subtract the mean and divide by the
standard deviation, so the values have mean zero and variance one.

That standardisation is where the trouble is. Suppose methylation really behaves
as `meth = a + b*age + noise`. After standardising, the slope you fit is not
`b`. It is

```
AROCM = b / sqrt(b^2 * Var(age) + sigma^2)
```

When the age signal is strong relative to the noise, the `b` terms dominate and
this collapses to roughly `1 / SD(age)`. The measure is bounded above by the
inverse of how spread out the sampled ages are. It is not a rate of biological
change. It is close to being a restatement of the sampling design.

And the sampling design is not random with respect to lifespan. To study a
bowhead whale across its life you need animals spanning a century. To study a
mouse you need animals spanning three years. Long-lived species are sampled
across wider spans in years, so they get lower AROCM values automatically.

Before fitting anything, I checked this directly. Across the 126 species, log
AROCM correlates **-0.889** with the log of the sampled age span, and **-0.835**
with log maximum lifespan. It correlates more strongly with the study design
than with the biology. And the observed values sit at a median of 58 per cent of
their mechanical ceiling, which is where you would expect a measure that is
mostly saturated by the sampling.

---

## Watching the answer disappear

So I fitted four models, each adding one control.

**Model 1** is the naive analysis: log ageing rate against log body mass. It
looks great. The slope is -0.150, R-squared is 0.308, and the p value is 1.5e-11.
Bigger mammals age more slowly, apparently.

**Model 2** adds the log sampled age span. The mass coefficient collapses from
-0.150 to **-0.017**, and p rises to 0.099. It stops being significant. The
model's R-squared jumps to 0.881, essentially all of it coming from the span
term.

**Model 3** adds a random intercept per taxonomic order, because 126 species are
not 126 independent observations. Primates share ancestry with primates. The
mass coefficient stays non-significant at -0.023, p = 0.168. Order accounts for
about 16 per cent of the remaining variance.

**Model 5** uses a measure that is artifact-free by construction. Since
`AROCM = correlation / SD(age)`, multiplying the rate back by the sampled
standard deviation recovers something proportional to the correlation itself,
which is bounded and does not depend on the span. Against body mass: coefficient
-0.004, p = 0.361.

The body-size effect **shrinks by 88.9 per cent** the moment you control for how
the animals were sampled.

Then the test that settled it for me. If the span relationship were purely
mechanical, the coefficient on log span would be exactly -1, because that is
what the algebra predicts. Observed, it is **-0.960 with a standard error of
0.039**. Testing the observed value against -1:

```python
t = (m2.params["log_span"] + 1.0) / m2.bse["log_span"]   # +1.02
p = 2 * stats.t.sf(abs(t), df=int(m2.df_resid))          # 0.311
```

The null hypothesis that this is a pure artifact **cannot be rejected**. The
data is entirely consistent with the span term carrying no biology whatsoever.

---

## So who is actually an outlier?

Something always sits at the extremes of a ranking. The real question is whether
a species deviates further than the spread of the other 125 justifies, and
whether it still does so after accounting for the fact that you are running 126
tests simultaneously.

The right tool is the **externally studentized residual**, which rescales each
species' residual by an error estimate computed with that species held out, so
an extreme point cannot inflate the yardstick used to judge it. Then
Benjamini-Hochberg to control the false discovery rate.

**One species out of 126** survives at q < 0.05. It is the northern giant mouse
lemur, which is not a species anybody had a prior hypothesis about.

Here is what happened to the ones people do have hypotheses about:

| Species | Naive residual | Corrected | q |
|---|---|---|---|
| Naked mole rat | -0.852 | **+0.237** | 0.976 |
| Bowhead whale | -1.526 | -0.573 | 0.976 |
| Human | -2.082 | -0.679 | 0.497 |
| Mouse | +0.878 | -0.052 | 0.995 |

The naked mole rat is the clean casualty. Naively it looks like a slow ager,
exactly as the folklore predicts. After correction its residual **changes sign**.
The animals were sampled across 22.3 years, which is an enormous span for a
thirty five gram rodent, and that alone accounts for its apparently slow clock.

The bowhead whale moves from -1.526 to -0.573. It stays on the slow side, which
is mildly reassuring, but it is nowhere near statistical significance.

There is a further subtlety worth stating plainly. Once the span term is
removed, what remains is proportional to the *correlation* between methylation
and age. So the corrected residual measures how tightly a species' epigenome
tracks its age, which is clock fidelity, and not how fast that species ages. Any
claim built on it needs to say so.

---

## Checking that the data itself is fine

A negative result is only interesting if the dataset is sound, so I ran two
checks against things that are already known.

**Kleiber's law** states that basal metabolic rate scales with body mass to the
power 0.75. Across the 55 species with metabolic data here, the fitted exponent
is **0.742**.

**The longevity quotient** is observed lifespan divided by the lifespan you would
predict from body mass. The top of that ranking comes out as human (3.65x),
little brown bat (3.17x), greater mouse-eared bat (3.02x), naked mole rat
(2.93x). Bats and naked mole rats are the canonical longevity outliers in
comparative biology, and they land exactly where they should.

The dataset is fine. The rate measure is the problem.

While I was there I tested the **rate-of-living hypothesis**, the old idea that
animals burning energy faster should age faster. Mass-specific metabolic rate
against epigenetic ageing rate gives a coefficient of +0.027 with a standard
error of 0.049 and p = 0.588. Nothing.

---

## The interpretable ML, and the failure that mattered most

There is one measure in this data that is genuinely immune to the sampling
artifact. If you divide each chromatin state's rate by that species' own average
across all 54 states, the `1/SD(age)` term cancels exactly, because it is common
to every state in the same animal. What is left is the *shape* of the ageing
signature: which parts of the genome drift fastest relative to everything else.

So I asked whether that shape predicts maximum lifespan.

Six of the 54 states correlate with lifespan at q < 0.05. Bivalent promoters,
which are regions held in a poised state by the polycomb protein complex, come
out negative. Active enhancers come out positive. This is a clean, publishable
looking result and it fits the existing literature.

Then I ran the test that matters. Instead of splitting species randomly, I held
out **whole taxonomic orders**. Train on primates, rodents, bats and
artiodactyls; predict carnivores. If the signature is real biology, it should
transfer to a lineage the model has never seen.

| Model | Out-of-sample r | R-squared |
|---|---|---|
| Elastic net | -0.070 | -0.407 |
| Random forest | -0.115 | -0.264 |

Both are worse than predicting the mean. The permutation test gives p = 0.627.

The six significant associations are phylogenetic structure. Closely related
species resemble each other in chromatin dynamics *and* in lifespan, and a model
fitted across orders absorbs that resemblance without learning anything
transferable. A random cross-validation split would have hidden this completely,
because random splits scatter close relatives across training and test folds.

This is the finding I am most glad I checked. Reporting the six states and
stopping would have produced a much more exciting article than this one.

---

## What does survive

The clock CpG sites themselves, and where they sit in the genome.

Testing the 1,303 sites used by the three universal clocks against the full
array background, they are enriched in 5' untranslated regions (odds ratio 1.38,
q = 0.003) and in exons (1.21, q = 0.003), and depleted in intergenic regions
downstream of genes (0.65, q = 0.003). They are pulled toward the functional
parts of genes.

And the genes carrying the most clock sites are these: **PAX2**, **HOXB7**,
**NPAS3**, **CASZ1**, **ZFHX3**, **BCOR**, **EVX2**, **ZNF521**.

Every one of those is a developmental transcription factor or a polycomb target.
**Polycomb** is a protein complex that keeps developmental genes switched off in
adult tissue, and the sites it guards accumulate methylation steadily with age
across essentially every mammal anybody has measured. The clock is reading a
conserved developmental programme slowly losing its silencing.

That is the durable finding. It replicates, it is mechanistically interpretable,
and it does not depend on any of the cross-species rate machinery.

---

## What I would tell someone starting this

Three things.

**Know how your outcome variable is constructed.** The single most consequential
line in this entire project was a call to `scale()` buried inside somebody else's
R function. Everything downstream inherited it. I only found it because I went
looking for the definition rather than accepting the column name.

**Rank orderings are not findings.** Every dataset has a top and a bottom. If
your result is that species X has the most extreme residual, you have not yet
tested anything. One of my 126 species survived a proper correction, and it was
not one of the interesting ones.

**Cross-validate against the structure you are worried about.** Random splits
answer the question "can this model interpolate among species like the ones it
has seen". Leave-one-order-out answers "has this model learned something about
mammals". Those are different questions and here they gave opposite answers.

The pan-mammalian clock is a genuine achievement. It predicts age across two
hundred million years of divergence from a few hundred chemical marks, and when
you rebuild it from scratch you land on the same marks. What it does not
straightforwardly hand you is a cross-species ageing speedometer, and the gap
between those two things is where I spent most of this project.

Fixing it is tractable. The artifact lives in the published summary tables, not
in the biology. Estimating rates on a common relative-age grid, with the sampled
age range held constant across species, would remove it, and that needs the raw
arrays in GSE223748 rather than the per-species slopes on GitHub. The pipeline
is structured so only the ingest step would change.

---

*Code, data, figures and an interactive dashboard:
[github.com/ralegionc/Metabolism_Rate](https://github.com/ralegionc/Metabolism_Rate).
All analysis uses the public Mammalian Methylation Consortium release. Nothing
is simulated.*
