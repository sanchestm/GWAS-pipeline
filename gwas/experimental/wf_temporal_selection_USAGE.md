# Wright-Fisher temporal SNP-frequency selection scan

This module implements a mass-conservative Wright-Fisher Fokker-Planck solver plus a genome-wide temporal allele-frequency scan for estimating drift, effective population size, and per-SNP selection.

Files:

- `wf_temporal_selection.py`: final Python module.
- This document: usage notes and examples.

## Conceptual model

The full allele-frequency distribution at time `t` is represented as a mixed distribution:

```text
P_t = M0(t) * delta_0 + f(p,t) dp + M1(t) * delta_1
```

where:

- `M0(t)` is the sticky/absorbing probability mass at allele frequency `p=0`.
- `M1(t)` is the sticky/absorbing probability mass at allele frequency `p=1`.
- `f(p,t)` is the continuous density over the open interval `(0,1)`.

The Fokker-Planck equation is implemented in conservative flux form:

```text
df/dt = -dJ/dp
J    = mu(p) f - d[D(p) f]/dp
mu(p) = s p(1-p)
D(p)  = p(1-p)/(diffusion_scale * Ne)
```

The default `diffusion_scale=4.0` corresponds to the classical diploid Wright-Fisher diffusion convention. Use `diffusion_scale=2.0` if you want to reproduce the earlier prototype scaling.

The per-SNP selection coefficient in all functions has the same interpretation:

```text
s > 0: tracked allele increased more than expected under drift
s < 0: tracked allele decreased more than expected under drift
s = 0: neutral drift null
```

## Installation / import

Copy `wf_temporal_selection.py` into your analysis directory, then import it:

```python
import sys
sys.path.insert(0, "/path/to/module")

from wf_temporal_selection import (
    wf_fp_solve_many,
    temporal_selection_scan_dataframe,
    estimate_ne_from_temporal_freqs,
    fit_s_pde_mle,
    simulate_wright_fisher_binomial,
    heat_kernel_absorbing_distribution,
    validate_module,
)
```

If working directly from `/mnt/data` in this environment:

```python
import sys
sys.path.insert(0, "/mnt/data")
from wf_temporal_selection import temporal_selection_scan_dataframe
```

## Validate the module

Run from the command line:

```bash
python wf_temporal_selection.py
```

Expected validation output should look approximately like:

```text
Validation checks
-----------------
max |mass_total - 1|: 1.679e-13
temporal Ne check:     {'true_Ne': 190.0, 'estimated_Ne': 190.36, 'F': 0.02596}
selection scan check:  {'true_Ne': 200.0, 'estimated_Ne': 207.56, 'top10_contains_selected': 10, ...}
```

The first check confirms that the Fokker-Planck solver is conserving probability. The temporal-Ne check simulates neutral drift and recovers the true effective population size. The selection-scan check simulates selected SNPs and verifies that they rank among the strongest hits.

## Main genome-wide function

Use this for the 7M-SNP analysis:

```python
temporal_selection_scan_dataframe(
    df,
    generations,
    snp_col="SNP_ID",
    freq_a_col="frequency generation a",
    freq_b_col="frequency generation b",
)
```

Input DataFrame must contain at least:

```text
SNP_ID
frequency generation a
frequency generation b
```

The two frequency columns must refer to the same allele orientation. Since you confirmed there is no allele flipping, the function does not attempt to reorient alleles.

### Minimal example

```python
import pandas as pd
from wf_temporal_selection import temporal_selection_scan_dataframe

df = pd.DataFrame({
    "SNP_ID": ["rs1", "rs2", "rs3", "rs4"],
    "frequency generation a": [0.20, 0.50, 0.80, 0.10],
    "frequency generation b": [0.21, 0.65, 0.78, 0.02],
})

out = temporal_selection_scan_dataframe(
    df,
    generations=10,
    Ne=190,              # supply known Ne, or omit to estimate from data
    test_min_maf=0.01,
)

print(out["Ne"])
print(out["Ne_summary"])
print(out["results"])
```

Important output columns:

```text
SNP_ID
p_generation_a
p_generation_b
delta_p
testable
Ne_used
F_drift_used
var_delta_used
selection_coefficient_s
selection_direction
expected_p_b_under_s_hat
se_s
z_s
z_s_delta_method
loglike_null_s0
loglike_alt_s_hat
lr_stat
p_value
lod_score
q_value_bh
significant_direction
```

`z_s` is the signed square root of `lr_stat`, so it is exactly consistent with `p_value` and `lod_score`. `z_s_delta_method` is a coefficient-scale Wald diagnostic from the logit-scale delta method; it is useful for orientation but should not be converted into an alternate p-value for strong signals.

### Typical large-scale HS-rat use

```python
import pandas as pd
from wf_temporal_selection import temporal_selection_scan_dataframe

freq_df = pd.read_parquet("snp_frequencies_generation_A_B.parquet")

out = temporal_selection_scan_dataframe(
    freq_df,
    generations=20,
    snp_col="SNP_ID",
    freq_a_col="frequency generation a",
    freq_b_col="frequency generation b",
    Ne=None,                     # estimate Ne genome-wide
    ne_model="discrete_diploid",
    ne_method="normal_mle",
    ne_trim_quantile=0.995,      # robustly trims strongest outliers before Ne estimation
    ne_min_maf=0.01,
    test_min_maf=0.01,
    logit_eps=1e-6,
    s_bounds=(-5, 5),
    add_fdr=True,
    alpha=0.05,
)

results = out["results"]
summary = out["Ne_summary"]

print(summary)
results.to_parquet("temporal_selection_scan_results.parquet", index=False)
```

### With sample sizes

If the frequency estimates come from finite sample sizes, include them. Sample sizes can be scalar or columns.

```python
out = temporal_selection_scan_dataframe(
    freq_df,
    generations=20,
    n_a_col="n_rats_generation_a",
    n_b_col="n_rats_generation_b",
    sample_sizes_are="diploid_individuals",
    test_min_maf=0.01,
)
```

or:

```python
out = temporal_selection_scan_dataframe(
    freq_df,
    generations=20,
    n_a=450,
    n_b=600,
    sample_sizes_are="diploid_individuals",
)
```

The function converts diploid individuals to chromosome counts internally by multiplying by 2.

## How the genome-wide scan works

### Step 1: Estimate genome-wide drift

For each SNP, the standardized temporal allele-frequency change is approximately:

```text
(delta_p)^2 / [p_a(1-p_a)]
```

Across mostly neutral SNPs, this estimates the temporal drift variance:

```text
F_t ≈ E[(p_b-p_a)^2 / {p_a(1-p_a)}]
```

If sample sizes are provided, expected binomial sampling variance is incorporated.

### Step 2: Convert temporal drift to Ne

The default discrete diploid model is:

```text
F_t = 1 - (1 - 1/(2Ne))^t
```

which gives:

```text
Ne = 1 / {2[1 - (1-F_t)^(1/t)]}
```

This matches the usual heterozygosity decay equation:

```text
H_t = H_0 (1 - 1/(2Ne))^t
```

### Step 3: Estimate per-SNP selection coefficient

Ignoring drift, selection follows:

```text
dp/dt = s p(1-p)
```

with closed-form solution:

```text
logit(p_b) = logit(p_a) + s * generations
```

Therefore:

```text
s_hat = [logit(p_b) - logit(p_a)] / generations
```

This is vectorized and suitable for millions of SNPs.

### Step 4: Test against neutral drift

For each SNP, the neutral variance of `delta_p` is approximated as:

```text
Var(delta_p) = F_t * p_a(1-p_a) + sampling_variance
```

The null model is:

```text
p_b ~ Normal(p_a, Var(delta_p))
```

The alternative model uses the expected selected frequency under `s_hat`:

```text
p_b ~ Normal(expected_p_b_under_s_hat, Var(delta_p))
```

The likelihood-ratio statistic is converted to:

```text
p_value = chi2.sf(lr_stat, df=1)
lod_score = lr_stat / (2 log(10))
```

This produces a fast genome-wide ranking. For final biological interpretation, top loci should be inspected for LD, batch effects, local ancestry/haplotype effects, and genotyping artifacts.

## Exact/PDE validation for top SNPs

The genome-wide scan is fast and approximate. For a small number of top SNPs, you can validate with the Fokker-Planck likelihood.

Example: observed 130 tracked alleles out of 200 chromosomes after 10 generations, starting at frequency 0.5:

```python
from wf_temporal_selection import fit_s_pde_mle

fit = fit_s_pde_mle(
    k=130,
    n=200,
    p0=0.5,
    Ne=190,
    T=10,
    s_bounds=(-0.5, 0.5),
    M=400,
    diffusion_scale=4.0,
)

print(fit)
```

Output includes:

```text
s_hat
loglike_alt
loglike_null
lr_stat
p_value
lod_score
optimizer_success
```

This path uses `scipy.optimize.minimize_scalar` and should be used for hundreds or thousands of selected SNPs, not for 7M SNPs.

## Fokker-Planck distribution example

```python
from wf_temporal_selection import wf_fp_solve_many, wf_density_cdf

out = wf_fp_solve_many(
    p0s=[0.1, 0.5, 0.9],
    Ne=190,
    s=0.0,
    T=20,
    M=400,
    diffusion_scale=4.0,
    flux_form="ito",
)

print(out["mass_total"])
print(out["M0"])
print(out["M1"])

p_grid, cdf = wf_density_cdf(out)
```

`mass_total` should be essentially 1 for every starting frequency. `M0` and `M1` are the accumulated sticky boundary masses.

## Recommended analysis workflow for HS-rat temporal SNP frequencies

1. Use allele frequencies with fixed allele orientation across generations.
2. Filter poorly measured SNPs and use a MAF threshold such as 0.01 or 0.05 for stable inference.
3. Estimate genome-wide Ne using `temporal_selection_scan_dataframe(..., Ne=None)`.
4. Use robust trimming, for example `ne_trim_quantile=0.995`, so selected loci or genotyping outliers do not inflate drift and bias Ne downward.
5. Rank SNPs by `p_value`, `lod_score`, or `q_value_bh`.
6. Validate top loci by checking local LD, generation-specific batch effects, platform changes, founder-haplotype frequencies, and missingness.
7. For selected top SNPs with allele counts, run `fit_s_pde_mle` as a distributional validation.

## Interpretation caveats

The fast genome-wide p-values are approximate. They assume:

- most SNPs used for Ne estimation are neutral;
- allele frequencies are on the same allele orientation;
- the estimated Ne applies genome-wide;
- SNPs are treated as independent for p-value ranking, although LD creates dependence;
- unmodeled batch/platform effects do not dominate temporal frequency shifts.

For manuscript use, report genome-wide temporal Ne with confidence intervals or sensitivity analyses, and validate top selection signals against LD-pruned SNP sets, founder haplotypes, platform covariates, and local ancestry.

---

## Optional diagnostic helpers added for the sine-series / simulation prototypes

The module includes corrected versions of the constant-diffusion sine-series idea as **diagnostic tools**, not as the main Wright-Fisher model:

```python
from wf_temporal_selection import heat_kernel_absorbing_distribution
import numpy as np

p = np.linspace(0, 1, 1001)
out = heat_kernel_absorbing_distribution(
    p,
    p0=0.5,
    Ne=100,
    t=10,
    diffusion_scale=4.0,
    n_terms=2000,
)

print(out["M0"], out["M1"], out["mass_total_grid_trapz"])
```

This solves the constant-coefficient heat equation with absorbing boundaries. It is useful for mathematical sanity checks, but it does not replace the Wright-Fisher Fokker-Planck solver because Wright-Fisher drift has frequency-dependent diffusion, `D(p)=p(1-p)/(diffusion_scale*Ne)`.

The module also includes a vectorized Wright-Fisher binomial simulator that replaces the slower Dask/list-comprehension prototype:

```python
from wf_temporal_selection import simulate_wright_fisher_binomial

traj = simulate_wright_fisher_binomial(
    p0=0.5,
    Ne=190,
    generations=20,
    nsim=100_000,
    s=0.0,
    random_state=1,
)

print(traj.shape)  # (100000, 21)
```

Use this simulator to validate temporal-Ne estimates and false-positive rates under neutral drift.


## Review-patch notes

This version resolves the issues identified in the independent review:

- `simulate_wright_fisher_binomial` and `heat_kernel_absorbing_distribution` are now public functions in the module.
- `wf_fixation_prob` now defaults to `diffusion_scale=4.0`, matching the module-wide classical diploid convention. Use `diffusion_scale=2.0` only for reproducing the original prototype scaling.
- `z_s` is now the signed square root of the LR statistic and is therefore consistent with `p_value`. The older delta-method coefficient statistic is retained as `z_s_delta_method`.
- A small NumPy compatibility wrapper allows trapezoidal integration to work with NumPy 1.x (`trapz`) or NumPy 2.x (`trapezoid`).
