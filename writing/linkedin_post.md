# LinkedIn post

I set out to find which mammals age faster than their body size predicts. The
answer turned out to be a lesson in reading someone else's variable definition.

DNA methylation clocks estimate an animal's age from chemical tags on its
genome, and the pan-mammalian version works across 185 species. Rebuilding it on
real Mammalian Methylation Consortium data, the published clock hits r = 0.948
on 50 bottlenose dolphin blood samples. I then trained one from scratch on the
same array and got r = 0.915 from 31 CpG sites, 7 of which also appear in the
published clock against 1.08 expected by chance. Two independent fits, same
positions in the genome. The clock is real.

Then the interesting question: study the residuals. Which species age slower
than body size predicts? Naked mole rats and bowhead whales are the obvious
candidates.

Fitting log ageing rate against log body mass gives a clean result. Slope
-0.150, p = 1.5e-11. Bigger mammals age more slowly.

Except the rate measure is the slope of z-scored methylation on age. Standardising
inside each species makes that slope mechanically proportional to 1/SD(sampled
ages). Long-lived species get sampled across wider age ranges, so their rate
drops for reasons that have nothing to do with biology.

Add the sampled age span as a covariate and the body-mass effect shrinks 88.9%
and stops being significant. The span coefficient comes out at -0.960 ± 0.039.
A pure artifact predicts exactly -1, and that null cannot be rejected (p = 0.311).

After correcting properly, using studentized residuals and Benjamini-Hochberg
across all 126 species, exactly one species remains a defensible outlier, and it
is a lemur nobody had a hypothesis about. The naked mole rat's residual reverses
sign.

The part I nearly missed: six chromatin states correlated with lifespan at
q < 0.05, which looked like a finding. Holding out whole taxonomic orders instead
of splitting species randomly, out-of-sample r was -0.07 and the permutation test
gave p = 0.63. The associations were phylogeny, not biology. A random split would
have hidden that completely.

What survives is the clock and its sites. They concentrate in developmental
transcription factors, PAX2, HOXB7, EVX2, NPAS3, the polycomb targets every
methylation clock converges on.

Three takeaways I would keep:

Know how your outcome variable was built. The most consequential line in this
project was a call to scale() inside someone else's R function.

A rank ordering is not a finding. Every dataset has a top and a bottom.

Cross-validate against the structure you are actually worried about. Random
splits and leave-one-order-out gave opposite answers here.

Full write-up, code and an interactive dashboard in the comments. All real
consortium data, nothing simulated.

#DataScience #Bioinformatics #MachineLearning #Ageing #Epigenetics
