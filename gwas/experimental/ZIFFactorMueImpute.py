#!/usr/bin/env python
"""zip_factor_imputation_pipeline.py

End‑to‑end pipeline for low‑coverage genotype imputation using a
Zero‑Inflated Poisson (ZIP) Factor model with sparse, recombination‑aware
smoothness regularisation.

Pipeline steps
--------------
1.  **BAM/CRAM ➜ allele‑count matrix** via pysam     (`extract_counts_from_bam`)
2.  **ZIP Factor training + variational inference**  (`ZipFactorImputer`)
3.  **Expected counts ➜ genotype probabilities**      (`counts_to_genotype_probs`)

Dependencies
------------
- python≥3.9
- jax & jaxlib (cuda or cpu)
- optax
- numpy, scipy, matplotlib (optional, for hotspot plots)
- pysam (for BAM/CRAM reading)

Usage (example)
---------------
```bash
python zip_factor_imputation_pipeline.py \
       --bam_list bams.txt \
       --vcf sites.vcf.gz \
       --reference ref.fa \
       --output imputed_genotypes.npz
```
`bams.txt` is a text file containing one path per line.

The script writes an `.npz` file with:
- `genotype_probs`  (shape = n_individuals x n_sites x 3)
- `imputed_counts` (shape = 2 x n_sites x n_individuals)
- `inferred_theta` (vector length n_sites)
- `positions`       (bp positions)
"""
from __future__ import annotations

import argparse
import pathlib
import time
from typing import List, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random, jit, value_and_grad
from scipy.sparse import coo_matrix
import pysam

# --------------------------------------------------------------------------------------
# 1.  BAM ➜ COUNT MATRIX
# --------------------------------------------------------------------------------------

Site = Tuple[str, int, str, str]  # (chrom, pos, ref, alt)

def read_sites_from_vcf(vcf_path: str) -> List[Site]:
    """Read biallelic SNP sites from a (bgzipped) VCF using pysam."""
    sites: List[Site] = []
    with pysam.VariantFile(vcf_path) as vcf:
        for rec in vcf.fetch():
            if rec.alts and len(rec.alts) == 1 and len(rec.ref) == 1 and len(rec.alts[0]) == 1:
                sites.append((rec.chrom, rec.pos, rec.ref.upper(), rec.alts[0].upper()))
    if not sites:
        raise ValueError("VCF contained no biallelic SNPs.")
    return sites

def extract_counts_from_bam(bam_paths: Sequence[str], sites: Sequence[Site], ref_fasta: str) -> np.ndarray:
    """Return counts matrix of shape (n_individuals, n_sites, 2)."""
    n_sites = len(sites)
    n_individuals = len(bam_paths)
    counts = np.zeros((n_individuals, n_sites, 2), dtype=np.int32)

    for ind_idx, bam_path in enumerate(bam_paths):
        bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=ref_fasta)
        for site_idx, (chrom, pos, ref, alt) in enumerate(sites):
            # pysam uses 0‑based positions internally
            for pileup in bam.pileup(chrom, pos - 1, pos, truncate=True, stepper="samtools"):
                if pileup.pos != pos - 1:
                    continue
                base_calls = pileup.get_query_sequences(add_indels=False)
                counts[ind_idx, site_idx, 0] = base_calls.count(ref)
                counts[ind_idx, site_idx, 1] = base_calls.count(alt)
        bam.close()
    return counts

def extract_variable_counts_from_bam(bam_paths: Sequence[str], sites: Sequence[Site], ref_fasta: str) -> np.ndarray:
    """Return counts matrix of shape (n_individuals, n_variable_sites, 2) by scanning sites first.

    For each site, counts the reference and alternate allele across individuals.
    If the site is fixed (no alternate allele observed in any individual) it is skipped.
    """
    # Open all BAM files once
    bams = [pysam.AlignmentFile(bam_path, "rb", reference_filename=ref_fasta) for bam_path in bam_paths]
    variable_site_counts = []  # each element will be a list of counts per individual for a variable site

    # Iterate over sites
    for chrom, pos, ref, alt in sites:
        site_counts = []
        is_variable = False
        for bam in bams:
            count_ref = 0
            count_alt = 0
            # pysam uses 0-based positions internally
            for pileup in bam.pileup(chrom, pos - 1, pos, truncate=True, stepper="samtools"):
                if pileup.pos != pos - 1:
                    continue
                base_calls = pileup.get_query_sequences(add_indels=False)
                count_ref = base_calls.count(ref)
                count_alt = base_calls.count(alt)
                # If this individual shows the alternate allele, mark the site as variable
                if count_alt > 0:
                    is_variable = True
            site_counts.append([count_ref, count_alt])
        if is_variable:
            variable_site_counts.append(site_counts)
    
    # Close all BAMs once done
    for bam in bams:
        bam.close()
    
    # Convert to array.
    # variable_site_counts is (n_variable_sites, n_individuals, 2)
    # We need to transpose to (n_individuals, n_variable_sites, 2)
    counts = np.array(variable_site_counts, dtype=np.int32).transpose(1, 0, 2)
    return counts

# --------------------------------------------------------------------------------------
# 2.  ZIP FACTOR IMPUTER (sparse, recombination‑aware, variational)
# --------------------------------------------------------------------------------------

def vectorised_sparse_graph(pos: jnp.ndarray, theta: jnp.ndarray, *, gamma: float = 10.0, window: int = 2_000_000):
    """Create sparse affinity graph within `window` bp using Haldane map + theta modifiers."""
    diffs = jnp.abs(pos[:, None] - pos[None, :])
    valid = (jnp.triu(diffs, 1) <= window) & (jnp.triu(diffs, 1) > 0)
    rows, cols = jnp.where(valid)
    dists = diffs[rows, cols]
    r = 0.5 * (1.0 - jnp.exp(dists * -2e-6))
    r_ij = 0.5 * (theta[rows] + theta[cols]) * r
    w = jnp.exp(-gamma * r_ij)
    # symmetrise
    rows = jnp.concatenate([rows, cols])
    cols = jnp.concatenate([cols, rows[: rows.shape[0]]])
    vals = jnp.concatenate([w, w])
    return np.array(rows), np.array(cols), np.array(vals), np.array(dists)

# ---------- utility functions ----------

def kl_normal(mu, logvar):
    return 0.5 * jnp.sum(jnp.exp(logvar) + mu ** 2 - 1.0 - logvar)

def zip_log_prob(x, lam, pi):
    eps = 1e-8
    zero = x == 0
    lp0 = jnp.log(pi + (1.0 - pi) * jnp.exp(-lam) + eps)
    lp1 = jnp.log(1.0 - pi + eps) - lam + x * jnp.log(lam + eps)
    return jnp.where(zero, lp0, lp1)

def laplacian_penalty_sparse(W_f, rows, cols, w_vals):
    return jnp.sum(w_vals * jnp.sum(W_f[rows] * W_f[cols], axis=1))

# ---------- main imputer class ----------

class ZipFactorImputer:
    """Variational ZIP factor model with sparse Laplacian smoothing."""

    def __init__(
        self,
        n_factors: int = 20,
        gamma: float = 10.0,
        alpha: float = 1.0,
        batch_size: int = 256,
        lr: float = 1e-2,
        seed: int = 0,
    ):
        self.k = n_factors
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.lr = lr
        self.key = random.PRNGKey(seed)
        self.params = None  # (W, mu, pi_logit)
        self.var_params = None  # (muZ, logvarZ, muTheta, logvarTheta)

    # ----- initialisation helpers -----
    def _init_params(self, n_vars: int, n_samples: int):
        key_w, key_mu, key_pi, self.key = random.split(self.key, 4)
        W = random.normal(key_w, (n_vars, self.k)) * 0.01
        mu = jnp.zeros((n_vars,))
        pi_logit = jnp.zeros((n_vars,))
        return W, mu, pi_logit

    def _init_var_params(self, n_vars: int, n_samples: int):
        k = self.k
        key_mz, key_lvz, key_mt, key_lvt, self.key = random.split(self.key, 5)
        muZ = random.normal(key_mz, (k, n_samples)) * 0.01
        logvarZ = random.normal(key_lvz, (k, n_samples)) - 5.0
        muTheta = jnp.zeros((n_vars,))
        logvarTheta = jnp.zeros((n_vars,)) - 5.0
        return muZ, logvarZ, muTheta, logvarTheta

    # ----- forward components -----
    def _batched_elbo(
        self,
        params,
        var_params,
        X,
        batch_idx,
        rows,
        cols,
        dists,
    ):
        W, mu, pi_log = params
        muZ, lvZ, muTh, lvTh = var_params
        subkey1, subkey2, self.key = random.split(self.key, 3)
        Z = muZ[:, batch_idx] + jnp.exp(0.5 * lvZ[:, batch_idx]) * random.normal(subkey1, (self.k, batch_idx.shape[0]))
        theta = jnp.exp(muTh + jnp.exp(0.5 * lvTh) * random.normal(subkey2, muTh.shape))

        lam = jnp.exp(W @ Z + mu[:, None])
        pi = jax.nn.sigmoid(pi_log)[:, None]
        ll = jnp.sum(zip_log_prob(X[:, batch_idx], lam, pi))

        kl_z = kl_normal(muZ[:, batch_idx], lvZ[:, batch_idx])
        kl_t = kl_normal(muTh, lvTh)

        r = 0.5 * (1.0 - jnp.exp(dists * -2e-6))
        r_ij = 0.5 * (theta[rows] + theta[cols]) * r
        w_vals = jnp.exp(-self.gamma * r_ij)
        smooth = laplacian_penalty_sparse(W, rows, cols, w_vals)

        return -(ll - (kl_z + kl_t) - self.alpha * smooth)

    # ----- training method -----
    def fit(self, X: np.ndarray, positions: np.ndarray, epochs: int = 5000):
        """Train the model. X expected shape = (variants*alleles, individuals)"""
        n_vars, n_samples = X.shape
        self.params = self._init_params(n_vars, n_samples)
        self.var_params = self._init_var_params(n_vars, n_samples)

        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init((self.params, self.var_params))

        # initial theta=1 for graph building
        rows, cols, _, dists = vectorised_sparse_graph(positions, jnp.ones(n_vars), gamma=self.gamma)

        @jit
        def _update(params, var_params, opt_state, X, batch_idx):
            loss, grads = value_and_grad(self._batched_elbo, argnums=(0, 1))(
                params, var_params, X, batch_idx, rows, cols, dists
            )
            updates, opt_state_ = optimizer.update(grads, opt_state, (params, var_params))
            params_, var_params_ = optax.apply_updates((params, var_params), updates)
            return params_, var_params_, opt_state_, loss

        n_batches = max(1, n_samples // self.batch_size)
        for ep in range(epochs):
            batch_idx = np.random.choice(n_samples, self.batch_size, replace=False)
            self.params, self.var_params, opt_state, loss = _update(
                self.params, self.var_params, opt_state, X, batch_idx
            )
            if ep % 500 == 0:
                print(f"[ep {ep:5d}] loss={loss:.3f}")
        print("Training complete.")

    # ----- imputation -----
    def impute_counts(self, mc_samples: int = 100) -> np.ndarray:
        """Return expected count tensor shape = (alleles, sites, individuals)."""
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        W, mu, _ = self.params
        muZ, lvZ, _, _ = self.var_params
        keys = random.split(random.PRNGKey(0), mc_samples)
        eps = jax.vmap(lambda k: random.normal(k, muZ.shape))(keys)
        Zs = muZ[None, :, :] + jnp.exp(0.5 * lvZ[None, :, :]) * eps
        lam_samples = jnp.exp(jnp.einsum("vf,mfs->mvs", W, Zs) + mu[:, None, None])
        return jnp.mean(lam_samples, axis=0)  # (variants, individuals)

    # accessors
    @property
    def inferred_theta(self):
        _, _, muTh, _ = self.var_params
        return jnp.exp(muTh)

# --------------------------------------------------------------------------------------
# 3. EXPECTED COUNTS ➜ GENOTYPE PROBABILITIES
# --------------------------------------------------------------------------------------

def counts_to_genotype_probs(expected_counts: jnp.ndarray, n_sites: int):
    """Convert expected allele counts to genotype probabilities.

    expected_counts shape = (features, individuals) where features = 2 × n_sites.
    Returns array (individuals, n_sites, 3) with probabilities for genotypes (0, 1, 2).
    """
    exp = expected_counts.reshape(2, n_sites, -1)  # (allele, site, ind)
    ref_c = exp[0].T  # individuals × sites
    alt_c = exp[1].T
    tot = ref_c + alt_c + 1e-8
    p_alt = alt_c / tot
    p0 = (1 - p_alt) ** 2
    p1 = 2 * p_alt * (1 - p_alt)
    p2 = p_alt ** 2
    return jnp.stack([p0, p1, p2], axis=-1)  # (ind, site, 3)

# --------------------------------------------------------------------------------------
# CLI ENTRY
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Low‑coverage genotype imputation with ZIP factor model")
    ap.add_argument("--bam_list", required=True, help="Text file with one BAM/CRAM path per line")
    ap.add_argument("--vcf", required=True, help="VCF of polymorphic sites (bgzipped + tabix)")
    ap.add_argument("--reference", required=True, help="Reference FASTA")
    ap.add_argument("--output", required=True, help="Output .npz file")
    ap.add_argument("--factors", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    bam_paths = [p.strip() for p in pathlib.Path(args.bam_list).read_text().splitlines() if p.strip()]
    print(f"Found {len(bam_paths)} BAM/CRAM files …")

    print("Reading VCF sites …")
    sites = read_sites_from_vcf(args.vcf)
    positions = np.array([pos for _, pos, _, _ in sites])
    print(f"Loaded {len(sites)} biallelic SNPs")

    print("Counting reads in BAMs … (this may take a while)")
    t0 = time.time()
    counts = extract_counts_from_bam(bam_paths, sites, args.reference)
    print(f"Done in {time.time() - t0:.1f}s")

    # reshape counts to (features, individuals)
    X = counts.reshape(counts.shape[0], -1).T.astype(np.float32)

    # train model
    imputer = ZipFactorImputer(n_factors=args.factors, batch_size=args.batch_size)
    imputer.fit(X, positions, epochs=args.epochs)

    # impute expected counts
    exp_counts = imputer.impute_counts(mc_samples=100)  # (features, individuals)
    geno_probs = counts_to_genotype_probs(exp_counts, len(sites))  # (ind, site, 3)

    np.savez_compressed(
        args.output,
        genotype_probs=np.array(geno_probs, dtype=np.float32),
        imputed_counts=np.array(exp_counts.reshape(2, len(sites), len(bam_paths)), dtype=np.float32),
        inferred_theta=np.array(imputer.inferred_theta),
        positions=positions,
    )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
