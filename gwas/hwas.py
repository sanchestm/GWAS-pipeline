import gc
import os
from glob import glob
from itertools import pairwise

import dask
import dask.array as da
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from scipy.special import erfc
from scipy.stats import chi2
from scipy.stats import f as scipyf
from scipy.stats import t as scipyt
from tqdm import tqdm
import pysam
from . import npplink
from . import genome_decomposition_v2 as gdv2

__all__ = [
    "detect_stitch_haplotype_field",
    "scan_stitch_bcf",
    "read_stitch_bcf_chunk",
    "load_stitch_bcf",
    "load_stitch_bcf_xarray",
    "stitch2array",
    "prepare_haplotype_design",
    "regression_hwas_ols",
    "HWAS",
    "mvHWAS",
]

def _normalize_alt_tuple(alts):
    if alts is None:
        return "."
    if isinstance(alts, str):
        return alts
    return ",".join(map(str, alts))


def _infer_n_haplotypes_from_value(value, collapse_phased=True):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1)
    if collapse_phased and (arr.size > 2) and (arr.size % 2 == 0):
        return arr.size // 2
    return arr.size


def _value_to_haplotype_vector(value, n_haplotypes=None, collapse_phased=True, dtype=np.float32):
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1)
    if n_haplotypes is None:
        n_haplotypes = _infer_n_haplotypes_from_value(arr, collapse_phased=collapse_phased)
    if n_haplotypes is None:
        return None
    if collapse_phased and (arr.size == 2 * n_haplotypes):
        return arr[:n_haplotypes] + arr[n_haplotypes:]
    if arr.size == n_haplotypes:
        return arr
    if arr.size == 2 and n_haplotypes == 1:
        return np.asarray([arr.sum()], dtype=dtype)
    if arr.size > n_haplotypes:
        return arr[:n_haplotypes]
    out = np.full(n_haplotypes, np.nan, dtype=dtype)
    out[: arr.size] = arr
    return out


def detect_stitch_haplotype_field(bcf_path, haplotype_field="auto", scan_variants=64, collapse_phased=True):
    """Infer which FORMAT field contains haplotype probabilities/dosages."""
    if haplotype_field not in [None, "auto"]: return haplotype_field
    preferred = ["AP", "HP", "HAP", "HAPROB", "HAPPROB", "HDS", "DS", "GP"]
    with pysam.VariantFile(bcf_path) as vf:
        sample_ids = list(vf.header.samples)
        probes = sample_ids[: min(4, len(sample_ids))]
        best_field = None
        best_score = (-1, -1)
        for rec_idx, rec in enumerate(vf):
            if rec_idx >= scan_variants:
                break
            for field in preferred:
                if field not in rec.format:
                    continue
                lengths = []
                for sid in probes:
                    value = rec.samples[sid].get(field)
                    if value is None:
                        continue
                    inferred = _infer_n_haplotypes_from_value(value, collapse_phased=collapse_phased)
                    if inferred is not None:
                        lengths.append(int(inferred))
                if not lengths:
                    continue
                score = (max(lengths), sum(l > 2 for l in lengths))
                if score > best_score:
                    best_score = score
                    best_field = field
            if best_field is not None and best_score[0] > 1:
                break
    if best_field is None:
        raise ValueError(
            "Could not infer a haplotype FORMAT field from the BCF. "
            "Pass `haplotype_field=...` explicitly."
        )
    return best_field


def scan_stitch_bcf(bcf_path, haplotype_field="auto", collapse_phased=True, max_variants=None):
    """Scan sample and variant metadata without loading all haplotype probabilities."""
    haplotype_field = detect_stitch_haplotype_field(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
    )
    records = []
    n_haplotypes = None
    with pysam.VariantFile(bcf_path) as vf:
        sample_ids = list(vf.header.samples)
        probes = sample_ids[: min(4, len(sample_ids))]
        for i, rec in enumerate(vf):
            if (max_variants is not None) and (i >= max_variants):
                break
            if n_haplotypes is None and haplotype_field in rec.format:
                for sid in probes:
                    value = rec.samples[sid].get(haplotype_field)
                    n_haplotypes = _infer_n_haplotypes_from_value(value, collapse_phased=collapse_phased)
                    if n_haplotypes is not None:
                        break
            records.append(
                {
                    "chrom": str(rec.contig),
                    "pos": int(rec.pos),
                    "snp": rec.id if rec.id not in [None, "."] else f"{rec.contig}:{rec.pos}:{rec.ref}:{_normalize_alt_tuple(rec.alts)}",
                    "id": "." if rec.id is None else str(rec.id),
                    "ref": str(rec.ref),
                    "alt": _normalize_alt_tuple(rec.alts),
                    "i": i,
                }
            )
    if n_haplotypes is None:
        raise ValueError(f"Could not infer the number of haplotypes from FORMAT/{haplotype_field}.")
    variants = pd.DataFrame.from_records(records)
    variants.attrs["haplotype_field"] = haplotype_field
    variants.attrs["n_haplotypes"] = int(n_haplotypes)
    samples = pd.DataFrame({"iid": sample_ids, "i": np.arange(len(sample_ids), dtype=int)})
    samples.attrs["haplotype_field"] = haplotype_field
    samples.attrs["n_haplotypes"] = int(n_haplotypes)
    return variants, samples


def read_stitch_bcf_chunk(
    bcf_path,
    variant_chunk,
    sample_ids,
    haplotype_field,
    n_haplotypes,
    collapse_phased=True,
    dtype=np.float32,
):
    """Read one BCF chunk into an array with shape (sample, snp, haplotype)."""
    variant_chunk = variant_chunk.reset_index(drop=True)
    sample_ids = list(sample_ids)
    out = np.full((len(sample_ids), len(variant_chunk), n_haplotypes), np.nan, dtype=dtype)
    wanted = {}
    for j, row in enumerate(variant_chunk.itertuples(index=False)):
        key = (str(row.chrom), int(row.pos), str(row.ref), str(row.alt), str(row.id))
        wanted[key] = j

    def _fill_from_record(rec, out_array):
        key = (str(rec.contig), int(rec.pos), str(rec.ref), _normalize_alt_tuple(rec.alts), "." if rec.id is None else str(rec.id))
        j = wanted.get(key)
        if j is None:
            key = (str(rec.contig), int(rec.pos), str(rec.ref), _normalize_alt_tuple(rec.alts), str(rec.id))
            j = wanted.get(key)
        if j is None:
            return
        for i, sid in enumerate(sample_ids):
            vec = _value_to_haplotype_vector(
                rec.samples[sid].get(haplotype_field),
                n_haplotypes=n_haplotypes,
                collapse_phased=collapse_phased,
                dtype=dtype,
            )
            if vec is None:
                continue
            out_array[i, j, : len(vec)] = vec

    grouped = variant_chunk.groupby("chrom", sort=False)
    try:
        with pysam.VariantFile(bcf_path) as vf:
            vf.subset_samples(sample_ids)
            for chrom, grp in grouped:
                start = max(int(grp["pos"].min()) - 1, 0)
                stop = int(grp["pos"].max())
                for rec in vf.fetch(str(chrom), start, stop):
                    _fill_from_record(rec, out)
        return out
    except (ValueError, OSError):
        with pysam.VariantFile(bcf_path) as vf:
            vf.subset_samples(sample_ids)
            for rec in vf:
                _fill_from_record(rec, out)
        return out


def load_stitch_bcf(bcf_path, chunk_variants=1000, dtype=np.float32, haplotype_field="auto", collapse_phased=True, max_variants=None):
    """Load STITCH haplotype probabilities lazily with Dask."""
    variants, samples = scan_stitch_bcf(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
    )
    haplotype_field = variants.attrs["haplotype_field"]
    n_haplotypes = int(variants.attrs["n_haplotypes"])
    delayed_chunks = []
    for offset in range(0, len(variants), int(chunk_variants)):
        chunk = variants.iloc[offset : offset + int(chunk_variants)].copy()
        delayed = dask.delayed(read_stitch_bcf_chunk)(
            bcf_path,
            chunk,
            samples["iid"].tolist(),
            haplotype_field,
            n_haplotypes,
            collapse_phased,
            dtype,
        )
        delayed_chunks.append(da.from_delayed(delayed, shape=(len(samples), len(chunk), n_haplotypes), dtype=dtype))
    haplotypes_dask = da.concatenate(delayed_chunks, axis=1) if delayed_chunks else da.empty((len(samples), 0, n_haplotypes), dtype=dtype)
    variants.attrs["haplotype_field"] = haplotype_field
    variants.attrs["n_haplotypes"] = n_haplotypes
    samples.attrs["haplotype_field"] = haplotype_field
    samples.attrs["n_haplotypes"] = n_haplotypes
    return variants, samples, haplotypes_dask


def load_stitch_bcf_xarray(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    haplotype_names=None,
):
    """Load STITCH haplotype probabilities and wrap them in an xarray.Dataset."""
    variants, samples, haplotypes_dask = load_stitch_bcf(
        bcf_path,
        chunk_variants=chunk_variants,
        dtype=dtype,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
    )
    n_haplotypes = int(variants.attrs["n_haplotypes"])
    if haplotype_names is None:
        haplotype_names = [f"h{h + 1}" for h in range(n_haplotypes)]
    return xr.Dataset(
        data_vars={"haplotype_probs": (("iid", "snp", "haplotype"), haplotypes_dask)},
        coords={
            "iid": samples["iid"].to_numpy(),
            "snp": variants["snp"].to_numpy(),
            "haplotype": np.asarray(haplotype_names, dtype=object),
            "chrom": ("snp", variants["chrom"].to_numpy()),
            "pos": ("snp", variants["pos"].to_numpy()),
            "ref": ("snp", variants["ref"].to_numpy()),
            "alt": ("snp", variants["alt"].to_numpy()),
            "variant_id": ("snp", variants["id"].to_numpy()),
        },
        attrs={
            "bcf_path": str(bcf_path),
            "haplotype_field": variants.attrs.get("haplotype_field"),
            "n_haplotypes": n_haplotypes,
            "collapse_phased": bool(collapse_phased),
        },
    )


def stitch2array(bcf_data, rfids=None, snplist=None, c=None, pos_start=None, pos_end=None, haplotypes=None, compute=True):
    """Subset lazy STITCH haplotype probabilities similarly to npplink.plink2df."""
    if isinstance(bcf_data, str):
        variants, samples, hap = load_stitch_bcf(bcf_data)
    else:
        variants, samples, hap = bcf_data
    vset = variants.copy()
    query_sentence = []
    if c is not None:
        c = str(c)
        query_sentence += ["chrom == @c"]
    if (pos_start is not None) and (pos_end is not None):
        query_sentence += ["pos.between(@pos_start, @pos_end)"]
    elif pos_start is not None:
        query_sentence += ["pos >= @pos_start"]
    elif pos_end is not None:
        query_sentence += ["pos <= @pos_end"]
    if snplist is not None:
        query_sentence += ["snp.isin(@snplist)"]
    query_sentence = " and ".join(query_sentence)
    if len(query_sentence):
        vset = vset.query(query_sentence)

    sset = samples.assign(_row_i=samples.index)
    if rfids is not None:
        rfids = list(rfids)
        sset_by_iid = sset.set_index("iid")
        keep = [x for x in rfids if x in sset_by_iid.index]
        missing = sorted(set(rfids) - set(keep))
        if missing:
            print(f'iids {"|".join(missing)} not in BCF file, sample will be dropped')
        sset = sset_by_iid.loc[keep].reset_index()
    else:
        keep = sset["iid"].tolist()

    hnames = [f"h{h + 1}" for h in range(hap.shape[2])]
    if haplotypes is None:
        hap_idx = np.arange(hap.shape[2], dtype=int)
        hcoords = hnames
    else:
        if isinstance(haplotypes, (str, int)):
            haplotypes = [haplotypes]
        hap_lookup = {name: i for i, name in enumerate(hnames)}
        hap_idx = np.array([hap_lookup[h] if isinstance(h, str) else int(h) for h in haplotypes], dtype=int)
        hcoords = [hnames[i] for i in hap_idx]

    arr = hap[sset["i"].to_numpy(), :, :][:, vset["i"].to_numpy(), :][:, :, hap_idx]
    if compute and hasattr(arr, "compute"):
        arr = arr.compute()

    return xr.DataArray(
        arr,
        dims=("iid", "snp", "haplotype"),
        coords={
            "iid": np.asarray(keep, dtype=object),
            "snp": vset["snp"].to_numpy(),
            "haplotype": np.asarray(hcoords, dtype=object),
            "chrom": ("snp", vset["chrom"].to_numpy()),
            "pos": ("snp", vset["pos"].to_numpy()),
            "ref": ("snp", vset["ref"].to_numpy()),
            "alt": ("snp", vset["alt"].to_numpy()),
            "variant_id": ("snp", vset["id"].to_numpy()),
        },
        name="haplotype_probs",
    )


def _resolve_reference_haplotype(haplotype_names, reference="last"):
    haplotype_names = list(haplotype_names)
    if reference == "last":
        return len(haplotype_names) - 1
    if reference == "first":
        return 0
    if isinstance(reference, str):
        return haplotype_names.index(reference)
    return int(reference)


def prepare_haplotype_design(X, precision=np.float32, center=False, scale=False, reference="last", add_intercept=True):
    """Prepare a HWAS design tensor from haplotype probabilities."""
    if isinstance(X, xr.Dataset):
        X = X["haplotype_probs"]
    haplotype_names = list(X.coords["haplotype"].to_numpy()) if isinstance(X, xr.DataArray) else [f"h{h + 1}" for h in range(np.asarray(X).shape[2])]
    ref_idx = _resolve_reference_haplotype(haplotype_names, reference=reference)
    keep_idx = [i for i in range(len(haplotype_names)) if i != ref_idx]
    kept_haplotypes = [haplotype_names[i] for i in keep_idx]
    X = np.asarray(X, dtype=precision)
    snp_mask = np.isfinite(X).all(axis=2)
    X = X[:, :, keep_idx].copy()
    X[~np.isfinite(X)] = 0.0
    if center:
        counts = snp_mask.sum(axis=0).astype(precision)
        sums = X.sum(axis=0)
        means = np.divide(sums, np.maximum(counts[:, None], 1.0), out=np.zeros_like(sums), where=counts[:, None] > 0)
        X -= means[None, :, :]
        X[~snp_mask, :] = 0.0
    if scale:
        sumsq = np.einsum("nmp,nmp->mp", X, X, optimize=True)
        denom = np.maximum(snp_mask.sum(axis=0, keepdims=False)[:, None] - 1.0, 1.0)
        design_std = np.sqrt(np.divide(sumsq, denom, out=np.ones_like(sumsq), where=denom > 0))
        design_std[design_std == 0] = 1.0
        X /= design_std[None, :, :]
    else:
        design_std = np.ones((X.shape[1], X.shape[2]), dtype=precision)
    if add_intercept:
        intercept = snp_mask.astype(precision)[:, :, None]
        X = np.concatenate([intercept, X], axis=2)
        design_std = np.concatenate([np.ones((design_std.shape[0], 1), dtype=precision), design_std], axis=1)
    return X, design_std, snp_mask.astype(precision), kept_haplotypes


def regression_hwas_ols(design, straits, snp_mask, traits_mask, dof="correct", stat="ttest", sided="two-sided", center=True, regression_type=None):
    """Blockwise multivariate HWAS using per-locus multiple regression.

    All modes solve the same vectorized normal-equation system on haplotype/founder
    probabilities. ``regression_type`` controls how the fit is summarized:
    ``'ttest'``/``'f'`` for F-based locus tests, ``'wald'`` for Wald tests,
    and ``'hk'``/``'lod'`` for Haley–Knott-style LOD summaries.
    """
    del center  # kept for API symmetry with npplink regression helpers
    if sided not in ["two-sided", "one-sided"]:
        raise ValueError("sided must be 'two-sided' or 'one-sided'")
    regression_type = stat if regression_type is None else regression_type
    if regression_type not in ["ttest", "f", "wald", "hk", "lod"]:
        raise ValueError("regression_type must be one of ['ttest', 'f', 'wald', 'hk', 'lod']")
    coeff_test = "wald" if regression_type == "wald" else "ttest"
    locus_test = {"ttest": "f", "f": "f", "wald": "wald", "hk": "lod", "lod": "lod"}[regression_type]

    design = np.asarray(design, dtype=np.float64)
    straits = np.asarray(straits, dtype=np.float64)
    snp_mask = np.asarray(snp_mask, dtype=np.float64)
    traits_mask = np.asarray(traits_mask, dtype=np.float64)
    p = design.shape[2]

    XtY = np.einsum("nmp,nt->mtp", design, straits, optimize=True)
    XtX = np.einsum("nmp,nt,nmq->mtpq", design, traits_mask, design, optimize=True)
    ysq = straits * traits_mask
    np.square(ysq, out=ysq)
    yty = np.einsum("nm,nt->mt", snp_mask, ysq, optimize=True)
    sumy = np.einsum("nm,nt->mt", snp_mask, straits, optimize=True)
    nobs = np.einsum("nm,nt->mt", snp_mask, traits_mask, optimize=True)

    xtx_inv = np.linalg.pinv(XtX)
    beta_full = np.matmul(xtx_inv, XtY[..., None])[..., 0]
    sse = yty - np.einsum("mtp,mtp->mt", beta_full, XtY, optimize=True)
    sse = np.clip(sse, 0.0, None)

    if dof == "incorrect":
        df_denom = np.broadcast_to(traits_mask.sum(axis=0) - p, nobs.shape).astype(np.float64, copy=True)
    else:
        df_denom = nobs - p
    df_denom[df_denom <= 0] = np.nan
    sigma2 = np.divide(sse, df_denom, out=np.full_like(sse, np.nan), where=np.isfinite(df_denom) & (df_denom > 0))
    beta_var = xtx_inv * sigma2[:, :, None, None]
    beta_se_full = np.sqrt(np.clip(np.diagonal(beta_var, axis1=-2, axis2=-1), 0.0, None))

    sse0 = yty - np.divide(sumy * sumy, nobs, out=np.full_like(yty, np.nan), where=nobs > 0)
    sse0 = np.clip(sse0, 0.0, None)
    ss_model = np.clip(sse0 - sse, 0.0, None)
    df_num = p - 1

    omnibus_f_stat = np.full_like(sse, np.nan)
    omnibus_f_p = np.full_like(sse, np.nan)
    omnibus_wald_stat = np.full_like(sse, np.nan)
    omnibus_wald_p = np.full_like(sse, np.nan)
    omnibus_lod = np.full_like(sse, np.nan)

    if df_num > 0:
        valid_f = (ss_model >= 0) & np.isfinite(df_denom) & (df_denom > 0) & np.isfinite(sse) & (sse >= 0)
        omnibus_f_stat = np.divide(ss_model / df_num, sse / df_denom, out=np.full_like(sse, np.nan), where=valid_f)
        omnibus_f_p = scipyf.sf(omnibus_f_stat, df_num, df_denom)

        valid_w = np.isfinite(sigma2) & (sigma2 > 0)
        omnibus_wald_stat = np.divide(ss_model, sigma2, out=np.full_like(sse, np.nan), where=valid_w)
        omnibus_wald_p = chi2.sf(omnibus_wald_stat, df=df_num)

        valid_lod = np.isfinite(sse0) & (sse0 > 0) & np.isfinite(sse) & (sse > 0)
        omnibus_lod = 0.5 * nobs * np.log10(np.divide(sse0, sse, out=np.full_like(sse, np.nan), where=valid_lod))

    beta = beta_full[:, :, 1:]
    beta_se = beta_se_full[:, :, 1:]
    stats = np.divide(beta, beta_se, out=np.full_like(beta, np.nan), where=beta_se > 0)
    if coeff_test == "ttest":
        p_values = scipyt.sf(np.abs(stats), df=df_denom[:, :, None])
        if sided == "two-sided":
            p_values *= 2.0
    else:
        zstats = np.abs(stats)
        p_values = erfc(zstats / np.sqrt(2.0))
        if sided == "one-sided":
            p_values *= 0.5
        np.square(zstats, out=zstats)
        stats = zstats

    if locus_test == "f":
        omnibus_stat = omnibus_f_stat
        omnibus_p = omnibus_f_p
        omnibus_stat_name = "F"
    elif locus_test == "wald":
        omnibus_stat = omnibus_wald_stat
        omnibus_p = omnibus_wald_p
        omnibus_stat_name = "Wald_chisq"
    else:
        omnibus_stat = omnibus_lod
        omnibus_p = np.power(10.0, -omnibus_lod, where=np.isfinite(omnibus_lod), out=np.full_like(omnibus_lod, np.nan))
        omnibus_stat_name = "LOD"

    with np.errstate(divide="ignore", invalid="ignore"):
        neglog_p = -np.log10(p_values)
        omnibus_neglog_p = -np.log10(omnibus_p)
        omnibus_f_neglog_p = -np.log10(omnibus_f_p)
        omnibus_wald_neglog_p = -np.log10(omnibus_wald_p)

    beta = np.transpose(beta, (0, 2, 1))
    beta_se = np.transpose(beta_se, (0, 2, 1))
    stats = np.transpose(stats, (0, 2, 1))
    neglog_p = np.transpose(neglog_p, (0, 2, 1))
    return {
        "beta": beta,
        "beta_se": beta_se,
        "stat": stats,
        "neglog_p": neglog_p,
        "dof": df_denom,
        "omnibus_stat": omnibus_stat,
        "omnibus_neglog_p": omnibus_neglog_p,
        "omnibus_df_num": np.full_like(df_denom, df_num, dtype=np.float64),
        "omnibus_df_denom": df_denom,
        "omnibus_f_stat": omnibus_f_stat,
        "omnibus_f_neglog_p": omnibus_f_neglog_p,
        "omnibus_wald_stat": omnibus_wald_stat,
        "omnibus_wald_neglog_p": omnibus_wald_neglog_p,
        "omnibus_lod": omnibus_lod,
        "regression_type": regression_type,
        "coef_test": coeff_test,
        "locus_test": locus_test,
        "omnibus_stat_name": omnibus_stat_name,
    }


def HWAS(
    traits,
    haplotypes,
    dtype="pandas_highmem",
    precision=np.float32,
    stat="ttest",
    sided="two-sided",
    dof="correct",
    center=True,
    hap_center=False,
    hap_scale=False,
    reference="last",
    regression_type=None,
):
    """Haplotype-based association scan analogous to npplink.GWA."""
    if isinstance(haplotypes, tuple) and len(haplotypes) == 4:
        design, design_std, snp_mask, kept_haplotypes = haplotypes
        hap_da = None
    else:
        hap_da = haplotypes if isinstance(haplotypes, xr.DataArray) else None
        design, design_std, snp_mask, kept_haplotypes = prepare_haplotype_design(
            haplotypes,
            precision=precision,
            center=hap_center,
            scale=hap_scale,
            reference=reference,
            add_intercept=True,
        )

    if isinstance(traits, tuple) and len(traits) == 3:
        straits, traits_std, traits_mask = traits
    else:
        straits, traits_std, traits_mask = npplink.scale_with_mask(traits, precision=precision, center=center, scale=False)

    del design_std, traits_std
    res = regression_hwas_ols(
        design,
        straits,
        snp_mask,
        traits_mask,
        dof=dof,
        stat=stat,
        sided=sided,
        center=center,
        regression_type=regression_type,
    )

    if hap_da is not None:
        snp_names = list(hap_da.coords["snp"].to_numpy())
        chrom = hap_da.coords["chrom"].to_numpy() if "chrom" in hap_da.coords else np.repeat(np.nan, len(snp_names))
        pos = hap_da.coords["pos"].to_numpy() if "pos" in hap_da.coords else np.repeat(np.nan, len(snp_names))
        ref = hap_da.coords["ref"].to_numpy() if "ref" in hap_da.coords else np.repeat(np.nan, len(snp_names))
        alt = hap_da.coords["alt"].to_numpy() if "alt" in hap_da.coords else np.repeat(np.nan, len(snp_names))
    else:
        snp_names = [f"snp_{i}" for i in range(design.shape[1])]
        chrom = np.repeat(np.nan, len(snp_names))
        pos = np.repeat(np.nan, len(snp_names))
        ref = np.repeat(np.nan, len(snp_names))
        alt = np.repeat(np.nan, len(snp_names))

    trait_names = list(traits.columns) if isinstance(traits, pd.DataFrame) else [f"trait_{i}" for i in range(straits.shape[1])]

    ds = xr.Dataset(
        data_vars={
            "beta": (("snp", "haplotype", "trait"), res["beta"]),
            "beta_se": (("snp", "haplotype", "trait"), res["beta_se"]),
            "stat": (("snp", "haplotype", "trait"), res["stat"]),
            "neglog_p": (("snp", "haplotype", "trait"), res["neglog_p"]),
            "dof": (("snp", "trait"), res["dof"]),
            "omnibus_stat": (("snp", "trait"), res["omnibus_stat"]),
            "omnibus_neglog_p": (("snp", "trait"), res["omnibus_neglog_p"]),
            "omnibus_df_num": (("snp", "trait"), res["omnibus_df_num"]),
            "omnibus_df_denom": (("snp", "trait"), res["omnibus_df_denom"]),
            "omnibus_f_stat": (("snp", "trait"), res["omnibus_f_stat"]),
            "omnibus_f_neglog_p": (("snp", "trait"), res["omnibus_f_neglog_p"]),
            "omnibus_wald_stat": (("snp", "trait"), res["omnibus_wald_stat"]),
            "omnibus_wald_neglog_p": (("snp", "trait"), res["omnibus_wald_neglog_p"]),
            "omnibus_lod": (("snp", "trait"), res["omnibus_lod"]),
        },
        coords={
            "snp": np.asarray(snp_names, dtype=object),
            "haplotype": np.asarray(kept_haplotypes, dtype=object),
            "trait": np.asarray(trait_names, dtype=object),
            "chrom": ("snp", np.asarray(chrom, dtype=object)),
            "pos": ("snp", np.asarray(pos)),
            "ref": ("snp", np.asarray(ref, dtype=object)),
            "alt": ("snp", np.asarray(alt, dtype=object)),
        },
        attrs={
            "reference_haplotype": reference,
            "haplotype_centered": bool(hap_center),
            "haplotype_scaled": bool(hap_scale),
            "regression_type": res["regression_type"],
            "coef_test": res["coef_test"],
            "locus_test": res["locus_test"],
            "omnibus_stat_name": res["omnibus_stat_name"],
        },
    )

    if dtype == "tuple":
        return res
    if dtype == "xarray_dataset":
        return ds
    if dtype == "xarray":
        return ds.to_array("metric")
    if dtype in ["pandas_highmem", "pandas"]:
        founder = ds[["beta", "beta_se", "stat", "neglog_p"]].to_dataframe().reset_index()
        omni = ds[[
            "dof",
            "omnibus_stat",
            "omnibus_neglog_p",
            "omnibus_df_num",
            "omnibus_df_denom",
            "omnibus_f_stat",
            "omnibus_f_neglog_p",
            "omnibus_wald_stat",
            "omnibus_wald_neglog_p",
            "omnibus_lod",
        ]].to_dataframe().reset_index()
        out = founder.merge(omni, on=["snp", "trait"], how="left")
        out["regression_type"] = res["regression_type"]
        out["coef_test"] = res["coef_test"]
        out["locus_test"] = res["locus_test"]
        return out
    raise ValueError("dtype has to be in ['tuple', 'xarray', 'xarray_dataset', 'pandas', 'pandas_highmem']")


def mvHWAS(
    traitdf,
    haplotypes,
    grms_folder="grm",
    save_path="results/hwas_parquet/",
    save=False,
    y_correction="blup_resid",
    y_correction_multivariate=False,
    return_table=True,
    stat="ttest",
    chrset=None,
    dtype="pandas",
    dof="correct",
    snp_block_size=50000000.0,
    gwa_center=True,
    hap_center=False,
    hap_scale=False,
    regression_mode="ols",
    haplotype_field="auto",
    chunk_variants=1000,
    collapse_phased=True,
    reference_haplotype="last",
    regression_type=None,
):
    """Multivariate haplotype scan mirroring genome_decomposition_v2.mvGWAS."""
    current_mem = lambda: str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 2)) + "Gb"

    if isinstance(grms_folder, str):
        grms_folder = glob(f"{grms_folder}/*.grm.bin")
    if regression_mode != "ols":
        raise ValueError("mvHWAS currently supports regression_mode='ols' only")

    read_hap = load_stitch_bcf(
        haplotypes,
        chunk_variants=chunk_variants,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
    ) if isinstance(haplotypes, str) else haplotypes

    chrunique = [str(x) for x in read_hap[0].chrom.astype(str).unique()]
    if chrset is not None:
        chrset = {str(x) for x in chrset}
        chrunique = [x for x in chrunique if x in chrset]

    grms_folder = pd.DataFrame(grms_folder, columns=["path"])
    grms_folder.index = grms_folder["path"].str.extract(r"([\d\w_]+)chrGRM.", expand=False)
    grms_folder = grms_folder.sort_index(
        key=lambda idx: idx.str.lower().map(
            {str(i): int(i) for i in range(1000)}
            | {i: int(i) for i in range(1000)}
            | {"all": -1000, "x": 1001, "y": 1002, "mt": 1003, "m": 1003}
        )
    )
    grms_folder = grms_folder[~grms_folder.index.isna()]

    allGRM = npplink.read_grm(grms_folder.loc["All", "path"].replace(".bin", ""))
    grms_folder["in_chrunique"] = grms_folder.index.isin(chrunique)
    grms_folder["isnum"] = grms_folder.index.str.isnumeric()
    max_grm_chr = grms_folder.query("isnum").index.astype(int).max()
    if grms_folder.in_chrunique.eq(False).sum() > 1:
        print("some chromosomes in the grms folder does not align with the haplotypes, trying to convert x,y,mt to +1, +2, +4")
        grms_folder = grms_folder.rename(
            {
                "x": str(max_grm_chr + 1),
                "y": str(max_grm_chr + 2),
                "mt": str(max_grm_chr + 4),
                "X": str(max_grm_chr + 1),
                "Y": str(max_grm_chr + 2),
                "MT": str(max_grm_chr + 4),
            }
        )
        grms_folder["in_chrunique"] = grms_folder.index.isin(chrunique)
        if grms_folder.in_chrunique.eq(False).sum() > 1:
            raise ValueError("cannot match chromosomes in grm folder and haplotype file")

    chrom_sizes = read_hap[0].astype({"chrom": str}).groupby("chrom").pos.max()
    sumstats = []
    if save:
        os.makedirs(save_path, exist_ok=True)

    for c, row in (pbar := tqdm(list(grms_folder.drop(["All"]).iterrows()))):
        if str(c) not in chrunique:
            continue
        pbar.set_description(f" HWAS-Chr{c}-reading {c}GRM")
        c_grm = npplink.read_grm(row.path.replace(".bin", ""))
        subgrm = allGRM["grm"].to_pandas() if not row.isnum else ((allGRM["grm"] * allGRM["w"] - c_grm["grm"] * c_grm["w"]) / (allGRM["w"] - c_grm["w"])).to_pandas()

        pbar.set_description(f"HWAS-Chr{c}-estimating var/covarmatrix")
        GErG = gdv2.genetic_varcov_PSD(subgrm, traitdf, n_boot=0)
        if y_correction == "blup_resid":
            pbar.set_description(f"HWAS-Chr{c}-BLUPresiduals-MEM:{current_mem()}")
            traits = traitdf - gdv2.mvBLUP(traitdf, subgrm, GErG["G"], E=GErG["E"], missing="exact", diag_only=not y_correction_multivariate)
        elif y_correction in ["ystar"]:
            pbar.set_description(f"HWAS-Chr{c}-whitenmatrix-MEM:{current_mem()}")
            traits = gdv2.mvWhiten(traitdf, subgrm, GErG["G"], E=GErG["E"], only_observed=False, diag_only=not y_correction_multivariate)
        elif y_correction is None:
            traits = traitdf.copy()
        else:
            raise ValueError("y_correction has to be in ['blup_resid', 'ystar', None]")

        if snp_block_size < chrom_sizes[str(c)]:
            snp_blocks = list(pairwise(np.arange(0, chrom_sizes[str(c)] + snp_block_size, snp_block_size)))
        else:
            snp_blocks = [(None, None)]

        for start, stop in snp_blocks:
            if start is None:
                pbar.set_description(f"HWAS-Chr{c}-HWAS-MEM:{current_mem()}")
                hblock = stitch2array(read_hap, c=c, rfids=traits.index, compute=True)
                suffix = ""
            else:
                pbar.set_description(f"HWAS-Chr{c}-HWAS[{start / 1000000.0:.0f}-{stop / 1000000.0:.0f}]Mb-MEM:{current_mem()}")
                hblock = stitch2array(read_hap, c=c, rfids=traits.index, pos_start=start, pos_end=stop, compute=True)
                suffix = f"_{int(start)}_{int(stop)}"
            if hblock.sizes.get("snp", 0) == 0:
                continue

            block = HWAS(
                traits,
                hblock,
                dtype=dtype,
                stat=stat,
                dof=dof,
                center=gwa_center,
                hap_center=hap_center,
                hap_scale=hap_scale,
                reference=reference_haplotype,
                regression_type=regression_type,
            )
            if save:
                block.to_parquet(
                    f"{save_path}hwas{c}{suffix}.parquet.gz",
                    compression="gzip",
                    engine="pyarrow",
                    compression_level=1,
                    use_dictionary=True,
                )
            if return_table:
                sumstats.append(block)
            del hblock, block
            gc.collect()

        del c_grm, subgrm, traits
        gc.collect()

    if return_table:
        if "pandas" not in dtype:
            return sumstats
        return pd.concat(sumstats, ignore_index=True) if len(sumstats) else pd.DataFrame()
    return None
