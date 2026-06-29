from __future__ import annotations

import gc
import math
import os
from glob import glob
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import psutil
import xarray as xr
from scipy.special import erfc
from scipy.stats import chi2
from scipy.stats import f as scipyf
from scipy.stats import t as scipyt
from tqdm import tqdm

try:
    import dask
    import dask.array as da
    from dask import delayed
except Exception:
    dask = None
    da = None
    delayed = None



def _require_npplink():
    try:
        import npplink
    except ImportError as exc:
        raise ImportError("npplink.py must be importable for this function.") from exc
    return npplink


def _require_gdv2():
    try:
        import genome_decomposition_v2 as gdv2
    except ImportError as exc:
        raise ImportError("genome_decomposition_v2.py must be importable for this function.") from exc
    return gdv2


def _center_traits_with_mask(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.to_numpy(dtype=np.float32, copy=True)
    mask = np.isfinite(vals)
    counts = mask.sum(axis=0, keepdims=True)
    sums = np.where(mask, vals, 0.0).sum(axis=0, keepdims=True)
    means = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums), where=counts > 0)
    vals = vals - means
    vals[~mask] = np.nan
    return pd.DataFrame(vals, index=df.index, columns=df.columns)

__all__ = [
    "detect_stitch_haplotype_field",
    "scan_stitch_bcf",
    "read_stitch_bcf_chunk",
    "load_stitch_bcf",
    "load_stitch_bcf_xarray",
    "stitch2array",
    "HGRM",
    "HGRM_lowmem",
    "scale_kinship",
    "calc_haplotype_kinship",
    "prepare_haplotype_design",
    "regression_hwas_ols",
    "warmup_jax_hwas",
    "warmup_jax_hgrm",
    "HWAS",
    "mvHWAS",
]


# -----------------------------------------------------------------------------
# Optional backends
# -----------------------------------------------------------------------------

def _require_pysam():
    try:
        import pysam
    except ImportError as exc:
        raise ImportError(
            "The BCF/VCF loader requires pysam (HTSlib-backed). Install it with `conda install -c bioconda pysam`."
        ) from exc
    return pysam


def _require_dask():
    if da is None or delayed is None:
        raise ImportError("This function requires dask[array].")
    return da


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "The JAX backend requires jax and jaxlib. Install a CPU or CUDA build before using backend='jax'."
        ) from exc
    return jax, jnp


def _is_dask_array(x: Any) -> bool:
    return da is not None and isinstance(x, da.Array)


def _is_xarray_with_dask(x: Any) -> bool:
    return isinstance(x, xr.DataArray) and _is_dask_array(x.data)


def _is_jax_array(x: Any) -> bool:
    try:
        import jax
        return isinstance(x, jax.Array)
    except Exception:
        return False


def _backend_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "dask":
        return _require_dask()
    if backend == "jax":
        _, jnp = _require_jax()
        return jnp
    raise ValueError("backend must be one of {'numpy','dask','jax'}")


def _infer_backend(data: Any, backend: str = "auto") -> str:
    if backend != "auto":
        if backend not in {"numpy", "dask", "jax"}:
            raise ValueError("backend must be one of {'auto','numpy','dask','jax'}")
        return backend
    if isinstance(data, xr.DataArray):
        data = data.data
    if _is_dask_array(data):
        return "dask"
    if _is_jax_array(data):
        return "jax"
    return "numpy"


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, xr.DataArray):
        x = x.data
    if _is_dask_array(x):
        return np.asarray(x.compute())
    if _is_jax_array(x):
        return np.asarray(x)
    return np.asarray(x)


def _as_backend_array(x: Any, backend: str, dtype=np.float32):
    if isinstance(x, xr.DataArray):
        x = x.data
    if backend == "numpy":
        return np.asarray(_to_numpy(x), dtype=dtype)
    if backend == "dask":
        da_mod = _require_dask()
        if isinstance(x, da_mod.Array):
            return x.astype(dtype)
        return da_mod.asarray(np.asarray(x, dtype=dtype))
    if backend == "jax":
        _, jnp = _require_jax()
        return jnp.asarray(_to_numpy(x), dtype=dtype)
    raise ValueError("Unknown backend")


def _maybe_materialize_dataset(ds: xr.Dataset, backend: str) -> xr.Dataset:
    if backend == "dask":
        return ds
    if backend == "numpy":
        return ds.compute() if any(_is_dask_array(v.data) for v in ds.data_vars.values()) else ds
    if backend == "jax":
        _, jnp = _require_jax()
        base = ds.compute() if any(_is_dask_array(v.data) for v in ds.data_vars.values()) else ds
        new_vars = {}
        for k, v in base.data_vars.items():
            arr = np.asarray(v.data)
            if arr.dtype.kind in "biufc":
                arr = jnp.asarray(arr)
            new_vars[k] = (v.dims, arr, dict(v.attrs))
        return xr.Dataset(data_vars=new_vars, coords=base.coords, attrs=base.attrs)
    raise ValueError("Unknown backend")


# -----------------------------------------------------------------------------
# BCF/VCF metadata helpers
# -----------------------------------------------------------------------------

def _normalize_alt_tuple(alts: Any) -> str:
    if alts is None:
        return "."
    if isinstance(alts, str):
        return alts
    return ",".join(map(str, alts))


def _normalize_filter_value(filt: Any) -> str:
    if filt in (None, (), [], set()):
        return "."
    if isinstance(filt, str):
        return filt
    return ";".join(map(str, filt))


def _field_dtype(field: str, field_type: str | None):
    if field == "GT":
        return np.int16
    if field_type is None:
        return np.float32
    t = str(field_type).lower()
    if t == "integer":
        return np.int32
    if t == "float":
        return np.float32
    if t == "flag":
        return np.bool_
    return object


def _missing_fill(dtype):
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return np.nan
    if np.issubdtype(dt, np.integer):
        return -1
    if np.issubdtype(dt, np.bool_):
        return False
    return None


def _flatten_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.reshape(-1).tolist()


def _infer_components(value: Any, collapse_phased: bool = False) -> int:
    vals = _flatten_value(value)
    n = len(vals)
    if n == 0:
        return 0
    if collapse_phased and n > 2 and (n % 2 == 0):
        return n // 2
    return n


def _convert_value(value: Any, *, ncomp: int, dtype, collapse_phased: bool = False):
    fill = _missing_fill(dtype)
    dt = np.dtype(dtype)
    if ncomp <= 1:
        if value is None:
            return fill
        vals = _flatten_value(value)
        if not vals:
            return fill
        if dt == np.dtype(object):
            return None if vals[0] is None else str(vals[0])
        return np.asarray(vals[:1], dtype=dt)[0]

    if dt == np.dtype(object):
        out = np.empty((ncomp,), dtype=object)
        out[:] = fill
    else:
        out = np.full((ncomp,), fill, dtype=dt)
    vals = _flatten_value(value)
    if not vals:
        return out
    if collapse_phased and len(vals) == 2 * ncomp and len(vals) > 2:
        left = np.asarray(vals[:ncomp], dtype=np.float32)
        right = np.asarray(vals[ncomp:], dtype=np.float32)
        vals = (left + right).tolist()
    vals = vals[:ncomp]
    if dt == np.dtype(object):
        out[:len(vals)] = [None if x is None else str(x) for x in vals]
    else:
        out[:len(vals)] = np.asarray(vals, dtype=dt)
    return out


def detect_stitch_haplotype_field(bcf_path, haplotype_field="auto", scan_variants=64, collapse_phased=True):
    if haplotype_field not in [None, "auto"]:
        return str(haplotype_field)
    pysam = _require_pysam()
    preferred = ["AP", "HP", "HAP", "HAPROB", "HAPPROB", "HDS", "DS", "GP"]
    with pysam.VariantFile(bcf_path) as vf:
        sample_ids = list(vf.header.samples)
        probes = sample_ids[: min(4, len(sample_ids))]
        best_field = None
        best_score = (-1, -1)
        for rec_i, rec in enumerate(vf):
            if rec_i >= scan_variants:
                break
            for field in preferred:
                if field not in rec.format:
                    continue
                lengths = []
                for sid in probes:
                    value = rec.samples[sid].get(field)
                    inferred = _infer_components(value, collapse_phased=collapse_phased)
                    if inferred > 1:
                        lengths.append(inferred)
                if not lengths:
                    continue
                score = (max(lengths), sum(l > 2 for l in lengths))
                if score > best_score:
                    best_score = score
                    best_field = field
            if best_field is not None and best_score[0] > 1:
                break
    if best_field is None:
        raise ValueError("Could not infer the haplotype FORMAT field. Pass haplotype_field explicitly.")
    return best_field


def scan_stitch_bcf(
    bcf_path,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    schema_scan_variants=128,
):
    """Scan variant metadata, INFO fields, and FORMAT schemas.

    Returns
    -------
    variants : pandas.DataFrame
    samples : pandas.DataFrame

    Side effects
    ------------
    The returned DataFrames contain attrs used by the lazy loaders:
    - variants.attrs['format_specs']
    - variants.attrs['info_specs']
    - variants.attrs['info_arrays']
    - variants.attrs['haplotype_field']
    - variants.attrs['n_haplotypes']
    """
    pysam = _require_pysam()
    haplotype_field = detect_stitch_haplotype_field(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
    )

    records = []
    info_raw = None
    info_specs = None
    format_specs = None
    n_haplotypes = None

    with pysam.VariantFile(bcf_path) as vf:
        sample_ids = list(vf.header.samples)
        probes = sample_ids[: min(4, len(sample_ids))]
        info_specs = {}
        for field, meta in vf.header.info.items():
            info_specs[str(field)] = {
                "dtype": _field_dtype(str(field), getattr(meta, "type", None)),
                "ncomp": 1,
                "type": getattr(meta, "type", None),
                "number": getattr(meta, "number", None),
            }
        format_specs = {}
        for field, meta in vf.header.formats.items():
            format_specs[str(field)] = {
                "dtype": _field_dtype(str(field), getattr(meta, "type", None)),
                "ncomp": 1,
                "type": getattr(meta, "type", None),
                "number": getattr(meta, "number", None),
                "collapse_phased": bool(str(field) == str(haplotype_field) and collapse_phased),
            }
        info_raw = {field: [] for field in info_specs}

        for i, rec in enumerate(vf):
            if max_variants is not None and i >= max_variants:
                break
            records.append(
                {
                    "chrom": str(rec.contig),
                    "pos": int(rec.pos),
                    "snp": rec.id if rec.id not in [None, "."] else f"{rec.contig}:{rec.pos}:{rec.ref}:{_normalize_alt_tuple(rec.alts)}",
                    "id": "." if rec.id is None else str(rec.id),
                    "ref": str(rec.ref),
                    "alt": _normalize_alt_tuple(rec.alts),
                    "qual": np.nan if rec.qual is None else float(rec.qual),
                    "filter": _normalize_filter_value(rec.filter.keys() if hasattr(rec.filter, "keys") else rec.filter),
                    "i": i,
                }
            )
            for field in info_specs:
                val = rec.info.get(field)
                info_raw[field].append(val)
                info_specs[field]["ncomp"] = max(info_specs[field]["ncomp"], max(1, _infer_components(val, collapse_phased=False)))

            if i < schema_scan_variants:
                for sid in probes:
                    s = rec.samples[sid]
                    for field, spec in format_specs.items():
                        if field not in rec.format:
                            continue
                        val = s.get(field)
                        spec["ncomp"] = max(spec["ncomp"], max(1, _infer_components(val, collapse_phased=spec["collapse_phased"])))
                        if field == haplotype_field and n_haplotypes is None:
                            inferred = _infer_components(val, collapse_phased=collapse_phased)
                            if inferred > 0:
                                n_haplotypes = int(inferred)

    if n_haplotypes is None:
        n_haplotypes = int(format_specs[haplotype_field]["ncomp"])
    if n_haplotypes <= 0:
        raise ValueError("Could not infer the number of haplotypes from the chosen FORMAT field.")

    variants = pd.DataFrame.from_records(records)
    samples = pd.DataFrame({"iid": sample_ids, "i": np.arange(len(sample_ids), dtype=int)})

    info_arrays = {}
    n_variants = len(variants)
    for field, spec in info_specs.items():
        ncomp = max(1, int(spec["ncomp"]))
        arr = []
        for val in info_raw[field]:
            arr.append(_convert_value(val, ncomp=ncomp, dtype=spec["dtype"], collapse_phased=False))
        if ncomp == 1:
            info_arrays[field] = np.asarray(arr, dtype=spec["dtype"] if spec["dtype"] is not object else object)
        else:
            info_arrays[field] = np.asarray(arr, dtype=spec["dtype"] if spec["dtype"] is not object else object).reshape(n_variants, ncomp)

    variants.attrs["haplotype_field"] = str(haplotype_field)
    variants.attrs["n_haplotypes"] = int(n_haplotypes)
    variants.attrs["format_specs"] = format_specs
    variants.attrs["info_specs"] = info_specs
    variants.attrs["info_arrays"] = info_arrays
    samples.attrs["haplotype_field"] = str(haplotype_field)
    samples.attrs["n_haplotypes"] = int(n_haplotypes)
    return variants, samples


# -----------------------------------------------------------------------------
# Lazy FORMAT readers
# -----------------------------------------------------------------------------

def _chunk_reader_all_formats(
    bcf_path,
    variant_chunk: pd.DataFrame,
    sample_ids,
    format_specs,
):
    pysam = _require_pysam()
    variant_chunk = variant_chunk.reset_index(drop=True)
    sample_ids = list(sample_ids)
    out = {}
    for field, spec in format_specs.items():
        shape = (len(sample_ids), len(variant_chunk)) if int(spec["ncomp"]) == 1 else (len(sample_ids), len(variant_chunk), int(spec["ncomp"]))
        fill = _missing_fill(spec["dtype"])
        if spec["dtype"] is object:
            arr = np.empty(shape, dtype=object)
            arr[:] = fill
        else:
            arr = np.full(shape, fill, dtype=spec["dtype"])
        out[field] = arr

    wanted = {}
    for j, row in enumerate(variant_chunk.itertuples(index=False)):
        wanted[(str(row.chrom), int(row.pos), str(row.ref), str(row.alt), str(row.id))] = j
        wanted[(str(row.chrom), int(row.pos), str(row.ref), str(row.alt), "." if pd.isna(row.id) else str(row.id))] = j

    def _fill_record(rec):
        key = (str(rec.contig), int(rec.pos), str(rec.ref), _normalize_alt_tuple(rec.alts), "." if rec.id is None else str(rec.id))
        j = wanted.get(key)
        if j is None:
            key = (str(rec.contig), int(rec.pos), str(rec.ref), _normalize_alt_tuple(rec.alts), str(rec.id))
            j = wanted.get(key)
        if j is None:
            return
        for i, sid in enumerate(sample_ids):
            sample = rec.samples[sid]
            for field, spec in format_specs.items():
                if field not in rec.format:
                    continue
                val = _convert_value(
                    sample.get(field),
                    ncomp=int(spec["ncomp"]),
                    dtype=spec["dtype"],
                    collapse_phased=bool(spec.get("collapse_phased", False)),
                )
                if int(spec["ncomp"]) == 1:
                    out[field][i, j] = val
                else:
                    out[field][i, j, :] = val

    grouped = variant_chunk.groupby("chrom", sort=False)
    try:
        with pysam.VariantFile(bcf_path) as vf:
            vf.subset_samples(sample_ids)
            for chrom, grp in grouped:
                start = max(int(grp["pos"].min()) - 1, 0)
                stop = int(grp["pos"].max())
                for rec in vf.fetch(str(chrom), start, stop):
                    _fill_record(rec)
    except Exception:
        with pysam.VariantFile(bcf_path) as vf:
            vf.subset_samples(sample_ids)
            for rec in vf:
                _fill_record(rec)
    return out


def read_stitch_bcf_chunk(
    bcf_path,
    variant_chunk,
    sample_ids,
    haplotype_field,
    n_haplotypes,
    collapse_phased=True,
    dtype=np.float32,
):
    """Read one BCF chunk for the haplotype field with shape (sample, snp, haplotype)."""
    specs = {
        str(haplotype_field): {
            "dtype": dtype,
            "ncomp": int(n_haplotypes),
            "collapse_phased": bool(collapse_phased),
        }
    }
    return _chunk_reader_all_formats(bcf_path, variant_chunk, sample_ids, specs)[str(haplotype_field)]


# -----------------------------------------------------------------------------
# Public lazy loaders
# -----------------------------------------------------------------------------

def load_stitch_bcf(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    backend="dask",
):
    """Load only the haplotype field lazily.

    Returns
    -------
    variants, samples, haplotypes
        `haplotypes` has shape (sample, snp, haplotype).
    """
    variants, samples = scan_stitch_bcf(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
    )
    n_haplotypes = int(variants.attrs["n_haplotypes"])
    haplotype_field = variants.attrs["haplotype_field"]
    _require_dask()

    delayed_chunks = []
    for offset in range(0, len(variants), chunk_variants):
        vchunk = variants.iloc[offset: offset + chunk_variants].copy()
        nchunk = len(vchunk)
        delayed_arr = delayed(read_stitch_bcf_chunk)(
            bcf_path,
            vchunk,
            samples["iid"].tolist(),
            haplotype_field,
            n_haplotypes,
            collapse_phased=collapse_phased,
            dtype=dtype,
        )
        delayed_chunks.append(da.from_delayed(delayed_arr, shape=(len(samples), nchunk, n_haplotypes), dtype=dtype))
    hap = da.concatenate(delayed_chunks, axis=1) if delayed_chunks else da.empty((len(samples), 0, n_haplotypes), dtype=dtype)

    if backend == "numpy":
        hap = np.asarray(hap.compute(), dtype=dtype)
    elif backend == "jax":
        _, jnp = _require_jax()
        hap = jnp.asarray(hap.compute(), dtype=dtype)
    elif backend != "dask":
        raise ValueError("backend must be one of {'dask','numpy','jax'}")

    return variants, samples, hap


def load_stitch_bcf_xarray(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    backend="dask",
):
    """Lazily parse the BCF into an xarray.Dataset.

    FORMAT fields keep their original names (e.g. GT, GP, DS, AP).
    INFO fields are stored as variables prefixed with ``INFO_``. Vector-valued
    fields gain a component dimension named ``<field>_component`` or
    ``INFO_<field>_component``.
    """
    _require_dask()
    variants, samples = scan_stitch_bcf(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
    )
    format_specs = variants.attrs["format_specs"]
    info_specs = variants.attrs["info_specs"]
    info_arrays = variants.attrs["info_arrays"]

    chunk_readers = []
    chunk_slices = []
    for offset in range(0, len(variants), chunk_variants):
        vchunk = variants.iloc[offset: offset + chunk_variants].copy()
        chunk_slices.append((offset, offset + len(vchunk)))
        chunk_readers.append(
            delayed(_chunk_reader_all_formats)(
                bcf_path,
                vchunk,
                samples["iid"].tolist(),
                format_specs,
            )
        )

    data_vars = {}
    coords = {
        "iid": samples["iid"].to_numpy(),
        "snp": variants["snp"].to_numpy(),
        "chrom": ("snp", variants["chrom"].to_numpy()),
        "pos": ("snp", variants["pos"].to_numpy()),
        "variant_id": ("snp", variants["id"].to_numpy()),
        "ref": ("snp", variants["ref"].to_numpy()),
        "alt": ("snp", variants["alt"].to_numpy()),
        "qual": ("snp", variants["qual"].to_numpy()),
        "filter": ("snp", variants["filter"].to_numpy()),
    }

    for field, spec in format_specs.items():
        arr_chunks = []
        ndim = 2 if int(spec["ncomp"]) == 1 else 3
        for reader, (start, stop) in zip(chunk_readers, chunk_slices):
            nchunk = stop - start
            arr = delayed(lambda d, k: d[k])(reader, field)
            shape = (len(samples), nchunk) if ndim == 2 else (len(samples), nchunk, int(spec["ncomp"]))
            arr_chunks.append(da.from_delayed(arr, shape=shape, dtype=spec["dtype"]))
        full = da.concatenate(arr_chunks, axis=1) if arr_chunks else da.empty((len(samples), 0), dtype=spec["dtype"])
        if ndim == 2:
            data_vars[field] = (("iid", "snp"), full)
        else:
            comp_dim = f"{field}_component"
            coords[comp_dim] = np.arange(int(spec["ncomp"]), dtype=int)
            data_vars[field] = (("iid", "snp", comp_dim), full)

    for field, spec in info_specs.items():
        arr = info_arrays[field]
        var_name = f"INFO_{field}"
        if arr.ndim == 1:
            data_vars[var_name] = (("snp",), arr)
        else:
            comp_dim = f"INFO_{field}_component"
            coords[comp_dim] = np.arange(arr.shape[1], dtype=int)
            data_vars[var_name] = (("snp", comp_dim), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "bcf_path": str(bcf_path),
            "haplotype_field": variants.attrs["haplotype_field"],
            "n_haplotypes": int(variants.attrs["n_haplotypes"]),
            "collapse_phased": bool(collapse_phased),
        },
    )
    return _maybe_materialize_dataset(ds, backend=backend)


# -----------------------------------------------------------------------------
# Subsetting helpers
# -----------------------------------------------------------------------------

def _default_haplotype_field_from_dataset(ds: xr.Dataset, field: str | None = None) -> str:
    if field not in [None, "auto"]:
        return str(field)
    hfield = ds.attrs.get("haplotype_field")
    if hfield is None:
        raise ValueError("Could not infer the haplotype field from the Dataset attrs. Pass field=... explicitly.")
    return str(hfield)


def stitch2array(
    bcf_data,
    rfids=None,
    snplist=None,
    c=None,
    pos_start=None,
    pos_end=None,
    haplotypes=None,
    field=None,
    compute=False,
    backend="auto",
):
    """Subset a lazy STITCH dataset and return a haplotype DataArray.

    Parameters
    ----------
    bcf_data : str | tuple | xarray.Dataset | xarray.DataArray
        - str: BCF path
        - tuple: (variants, samples, haplotypes) from load_stitch_bcf
        - Dataset: usually from load_stitch_bcf_xarray
        - DataArray: already a haplotype tensor
    field : str | None
        FORMAT field to use as the haplotype tensor when `bcf_data` is a Dataset.
    compute : bool, default=False
        Materialize the subset if it is Dask-backed.
    """
    if isinstance(bcf_data, str):
        bcf_data = load_stitch_bcf_xarray(bcf_data, backend="dask")

    if isinstance(bcf_data, tuple) and len(bcf_data) == 3:
        variants, samples, hap = bcf_data
        n_h = hap.shape[2]
        da_out = xr.DataArray(
            hap,
            dims=("iid", "snp", "haplotype"),
            coords={
                "iid": samples["iid"].to_numpy(),
                "snp": variants["snp"].to_numpy(),
                "haplotype": np.asarray([f"h{i+1}" for i in range(n_h)], dtype=object),
                "chrom": ("snp", variants["chrom"].to_numpy()),
                "pos": ("snp", variants["pos"].to_numpy()),
                "ref": ("snp", variants["ref"].to_numpy()),
                "alt": ("snp", variants["alt"].to_numpy()),
                "variant_id": ("snp", variants["id"].to_numpy()),
            },
            name=variants.attrs.get("haplotype_field", "haplotypes"),
            attrs={
                "haplotype_field": variants.attrs.get("haplotype_field", "haplotypes"),
                "n_haplotypes": int(variants.attrs.get("n_haplotypes", n_h)),
            },
        )
    elif isinstance(bcf_data, xr.Dataset):
        field = _default_haplotype_field_from_dataset(bcf_data, field)
        var = bcf_data[field]
        if var.ndim == 2:
            raise ValueError(f"Selected field {field!r} is 2D. HWAS requires a 3D haplotype/founder-probability field.")
        if var.ndim != 3:
            raise ValueError(f"Selected field {field!r} has ndim={var.ndim}; expected 3 dimensions.")
        comp_dim = var.dims[2]
        da_out = xr.DataArray(
            var.data,
            dims=("iid", "snp", "haplotype"),
            coords={
                "iid": bcf_data["iid"].to_numpy(),
                "snp": bcf_data["snp"].to_numpy(),
                "haplotype": bcf_data[comp_dim].to_numpy() if comp_dim in bcf_data.coords else np.arange(var.shape[2]),
                "chrom": ("snp", bcf_data["chrom"].to_numpy()),
                "pos": ("snp", bcf_data["pos"].to_numpy()),
                "ref": ("snp", bcf_data["ref"].to_numpy()),
                "alt": ("snp", bcf_data["alt"].to_numpy()),
                "variant_id": ("snp", bcf_data["variant_id"].to_numpy()),
            },
            name=field,
            attrs={
                "haplotype_field": field,
                "n_haplotypes": int(var.shape[2]),
                **dict(bcf_data.attrs),
            },
        )
    elif isinstance(bcf_data, xr.DataArray):
        da_out = bcf_data
    else:
        raise TypeError("bcf_data must be a path, (variants,samples,haplotypes) tuple, xarray.Dataset, or xarray.DataArray")

    if rfids is not None:
        rfids = [x for x in rfids if x in set(map(str, da_out["iid"].to_numpy()))]
        da_out = da_out.sel(iid=rfids)
    if snplist is not None:
        snplist = [x for x in snplist if x in set(map(str, da_out["snp"].to_numpy()))]
        da_out = da_out.sel(snp=snplist)
    if c is not None:
        mask = xr.DataArray(da_out["chrom"].astype(str).to_numpy() == str(c), dims=("snp",), coords={"snp": da_out["snp"]})
        da_out = da_out.sel(snp=mask)
    if pos_start is not None:
        da_out = da_out.sel(snp=da_out["pos"] >= int(pos_start))
    if pos_end is not None:
        da_out = da_out.sel(snp=da_out["pos"] <= int(pos_end))
    if haplotypes is not None:
        da_out = da_out.sel(haplotype=haplotypes)

    backend = _infer_backend(da_out, backend=backend)
    if compute and _is_dask_array(da_out.data):
        if backend == "jax":
            _, jnp = _require_jax()
            return xr.DataArray(jnp.asarray(da_out.data.compute()), dims=da_out.dims, coords=da_out.coords, attrs=da_out.attrs, name=da_out.name)
        return da_out.compute()
    if backend == "numpy" and _is_dask_array(da_out.data):
        return da_out.compute()
    return da_out


# -----------------------------------------------------------------------------
# Haplotype GRM
# -----------------------------------------------------------------------------

def _resolve_reference_haplotype(haplotype_names, reference="last"):
    haplotype_names = list(haplotype_names)
    if reference in [None, False, "none", "all"]:
        return None
    if reference == "last":
        return len(haplotype_names) - 1
    if reference == "first":
        return 0
    if isinstance(reference, str):
        return haplotype_names.index(reference)
    return int(reference)


def _unwrap_haplotype_array(X, field=None):
    if isinstance(X, str):
        X = load_stitch_bcf_xarray(X, backend="dask")
    if isinstance(X, xr.Dataset):
        field = _default_haplotype_field_from_dataset(X, field)
        X = stitch2array(X, field=field, compute=False)
    elif isinstance(X, tuple) and len(X) == 3:
        X = stitch2array(X, compute=False)
    if not isinstance(X, xr.DataArray):
        raise TypeError("Expected a haplotype xarray.DataArray, xarray.Dataset, path, or (variants,samples,haplotypes) tuple.")
    return X


def _drop_reference(Xnp, hap_names, reference="last"):
    ref_idx = _resolve_reference_haplotype(hap_names, reference=reference)
    if ref_idx is None:
        return Xnp, list(hap_names), None
    keep = [i for i in range(len(hap_names)) if i != ref_idx]
    return Xnp[:, :, keep], [hap_names[i] for i in keep], ref_idx


def _standardize_haplotype_block_numpy(block, *, center=True, scale=True, reference="last", hap_names=None):
    X = np.asarray(block, dtype=np.float32)
    if hap_names is None:
        hap_names = [f"h{i+1}" for i in range(X.shape[2])]
    X, kept, ref_idx = _drop_reference(X, hap_names, reference=reference)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if scale:
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X = X / sd
    return X, kept, ref_idx


def _qtl2_haplotype_block_numpy(block, *, normalize_ploidy=True):
    X = np.asarray(block, dtype=np.float32)
    if normalize_ploidy:
        denom = X.sum(axis=2, keepdims=True)
        denom[denom == 0] = 1.0
        X = X / denom
    return X


def _jax_prepare_hgrm_block(block, center=True, scale=True):
    _, jnp = _require_jax()
    Z = block
    if center:
        Z = Z - jnp.mean(Z, axis=0, keepdims=True)
    if scale:
        sd = jnp.std(Z, axis=0, keepdims=True)
        sd = jnp.where(sd == 0, 1.0, sd)
        Z = Z / sd
    Z = Z.reshape(Z.shape[0], -1)
    return Z @ Z.T, Z.shape[1]


def _jax_prepare_hgrm_block_qtl2(block, normalize_ploidy=True):
    _, jnp = _require_jax()
    Z = block
    if normalize_ploidy:
        denom = jnp.sum(Z, axis=2, keepdims=True)
        denom = jnp.where(denom == 0, 1.0, denom)
        Z = Z / denom
    Z = Z.reshape(Z.shape[0], -1)
    return Z @ Z.T, block.shape[1]


def warmup_jax_hgrm(n_samples, block_variants, n_haplotypes_kept, *, center=True, scale=True, method="standardized", qtl2_normalize_ploidy=True, dtype=np.float32):
    jax, jnp = _require_jax()
    if method == "qtl2":
        kernel = jax.jit(lambda x: _jax_prepare_hgrm_block_qtl2(x, normalize_ploidy=qtl2_normalize_ploidy))
        nh = int(n_haplotypes_kept)
    else:
        kernel = jax.jit(lambda x: _jax_prepare_hgrm_block(x, center=center, scale=scale))
        nh = int(n_haplotypes_kept)
    dummy = jnp.zeros((int(n_samples), int(block_variants), nh), dtype=dtype)
    gram, feats = kernel(dummy)
    gram.block_until_ready()
    return kernel


def scale_kinship(K):
    """Scale a kinship matrix to be correlation-like, matching qtl2's scale_kinship()."""
    if isinstance(K, xr.DataArray):
        arr = K.data
        attrs = dict(K.attrs)
        ids0 = K.coords.get("sample_0")
        ids1 = K.coords.get("sample_1")
        if _is_dask_array(arr):
            d = da.sqrt(da.diagonal(arr))
            d = da.where(d == 0, 1.0, d)
            out = arr / d[:, None] / d[None, :]
        elif _is_jax_array(arr):
            _, jnp = _require_jax()
            d = jnp.sqrt(jnp.diag(arr))
            d = jnp.where(d == 0, 1.0, d)
            out = arr / d[:, None] / d[None, :]
        else:
            arr = np.asarray(arr)
            d = np.sqrt(np.diag(arr))
            d[d == 0] = 1.0
            out = arr / d[:, None] / d[None, :]
        return xr.DataArray(out, dims=K.dims, coords=K.coords, attrs=attrs, name=K.name)
    arr = K
    if _is_dask_array(arr):
        d = da.sqrt(da.diagonal(arr))
        d = da.where(d == 0, 1.0, d)
        return arr / d[:, None] / d[None, :]
    if _is_jax_array(arr):
        _, jnp = _require_jax()
        d = jnp.sqrt(jnp.diag(arr))
        d = jnp.where(d == 0, 1.0, d)
        return arr / d[:, None] / d[None, :]
    arr = np.asarray(arr)
    d = np.sqrt(np.diag(arr))
    d[d == 0] = 1.0
    return arr / d[:, None] / d[None, :]


def _normalize_grm_output(K, ids=None, attrs=None):
    if ids is None:
        return K
    return xr.DataArray(
        K,
        dims=("sample_0", "sample_1"),
        coords={"sample_0": np.asarray(ids, dtype=object), "sample_1": np.asarray(ids, dtype=object)},
        attrs={} if attrs is None else dict(attrs),
        name="grm",
    )


def _save_grm_simple(savefile, K, ids=None, n_features=None):
    arr = _to_numpy(K)
    savefile = str(savefile)
    folder = os.path.dirname(savefile)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if ids is not None:
        ids = np.asarray(ids, dtype=object)
        pd.DataFrame({"fid": ids, "iid": ids}).to_csv(f"{savefile}.grm.id", index=False, header=False, sep="\t")
    tri = np.tril_indices_from(arr)
    arr[tri].astype(np.float32).tofile(f"{savefile}.grm.bin")
    if n_features is not None:
        with open(f"{savefile}.n_features.txt", "w", encoding="utf-8") as fh:
            fh.write(str(int(n_features)) + "\n")


def HGRM(
    X,
    *,
    center=True,
    scale=True,
    correlation_matrix=False,
    savefile=None,
    backend="auto",
    dtype=np.float32,
    reference="last",
    field=None,
    method="standardized",
    qtl2_normalize_ploidy=True,
):
    da_x = _unwrap_haplotype_array(X, field=field)
    hap_names = list(map(str, da_x["haplotype"].to_numpy()))
    ids = da_x["iid"].to_numpy()
    backend = _infer_backend(da_x, backend)
    method = str(method).lower()
    if method not in {"standardized", "qtl2"}:
        raise ValueError("method must be one of {'standardized','qtl2'}")

    if backend == "dask":
        darr = da_x.data if _is_dask_array(da_x.data) else _require_dask().asarray(np.asarray(da_x.data, dtype=dtype))
        if method == "standardized":
            ref_idx = _resolve_reference_haplotype(hap_names, reference=reference)
            if ref_idx is not None:
                keep = [i for i in range(len(hap_names)) if i != ref_idx]
                darr = darr[:, :, keep]
                kept = [hap_names[i] for i in keep]
            else:
                kept = hap_names
            if center:
                darr = darr - darr.mean(axis=0, keepdims=True)
            if scale:
                sd = darr.std(axis=0, keepdims=True)
                sd = da.where(sd == 0, 1.0, sd)
                darr = darr / sd
            Z = darr.reshape((darr.shape[0], darr.shape[1] * darr.shape[2]))
            K = Z @ Z.T
            denom = max(int(len(kept) * int(da_x.sizes["snp"])) - 1, 1)
            attrs = {"backend": "dask", "n_features": int(len(kept) * int(da_x.sizes["snp"])), "kinship_method": method, "kept_haplotypes": kept, "reference_haplotype_index": ref_idx}
        else:
            kept = hap_names
            ref_idx = None
            if qtl2_normalize_ploidy:
                s = darr.sum(axis=2, keepdims=True)
                s = da.where(s == 0, 1.0, s)
                darr = darr / s
            Z = darr.reshape((darr.shape[0], darr.shape[1] * darr.shape[2]))
            K = Z @ Z.T
            denom = max(int(da_x.sizes["snp"]), 1)
            attrs = {"backend": "dask", "n_features": int(da_x.sizes["snp"]), "n_haplotypes_used": int(da_x.sizes["haplotype"]), "kinship_method": method, "qtl2_normalize_ploidy": bool(qtl2_normalize_ploidy), "kept_haplotypes": kept, "reference_haplotype_index": ref_idx}
        K = K / denom
        if correlation_matrix:
            K = scale_kinship(K)
        out = _normalize_grm_output(K, ids=ids, attrs=attrs)
        if savefile is not None:
            _save_grm_simple(savefile, out, ids=ids, n_features=int(attrs["n_features"]))
        return out

    Xnp = _to_numpy(da_x).astype(dtype, copy=False)
    if method == "standardized":
        Xnp, kept, ref_idx = _drop_reference(Xnp, hap_names, reference=reference)
        if backend == "jax":
            _, jnp = _require_jax()
            Xb = jnp.asarray(Xnp)
            if center:
                Xb = Xb - jnp.mean(Xb, axis=0, keepdims=True)
            if scale:
                sd = jnp.std(Xb, axis=0, keepdims=True)
                sd = jnp.where(sd == 0, 1.0, sd)
                Xb = Xb / sd
            Z = Xb.reshape((Xb.shape[0], Xb.shape[1] * Xb.shape[2]))
            K = Z @ Z.T
            n_features = int(Xnp.shape[1] * Xnp.shape[2])
            K = K / max(n_features - 1, 1)
            if correlation_matrix:
                K = scale_kinship(K)
            out = _normalize_grm_output(K, ids=ids, attrs={"backend": "jax", "n_features": n_features, "kinship_method": method, "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})
        else:
            if center:
                Xnp = Xnp - Xnp.mean(axis=0, keepdims=True)
            if scale:
                sd = Xnp.std(axis=0, keepdims=True)
                sd[sd == 0] = 1.0
                Xnp = Xnp / sd
            Z = Xnp.reshape((Xnp.shape[0], Xnp.shape[1] * Xnp.shape[2]))
            K = Z @ Z.T
            n_features = int(Xnp.shape[1] * Xnp.shape[2])
            K = K / max(n_features - 1, 1)
            if correlation_matrix:
                K = scale_kinship(K)
            out = _normalize_grm_output(K, ids=ids, attrs={"backend": "numpy", "n_features": n_features, "kinship_method": method, "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})
    else:
        kept = hap_names
        ref_idx = None
        if backend == "jax":
            _, jnp = _require_jax()
            Xb = jnp.asarray(Xnp)
            if qtl2_normalize_ploidy:
                s = jnp.sum(Xb, axis=2, keepdims=True)
                s = jnp.where(s == 0, 1.0, s)
                Xb = Xb / s
            Z = Xb.reshape((Xb.shape[0], Xb.shape[1] * Xb.shape[2]))
            K = Z @ Z.T
            n_features = int(Xnp.shape[1])
            K = K / max(n_features, 1)
            if correlation_matrix:
                K = scale_kinship(K)
            out = _normalize_grm_output(K, ids=ids, attrs={"backend": "jax", "n_features": n_features, "n_haplotypes_used": int(Xnp.shape[2]), "kinship_method": method, "qtl2_normalize_ploidy": bool(qtl2_normalize_ploidy), "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})
        else:
            if qtl2_normalize_ploidy:
                s = Xnp.sum(axis=2, keepdims=True)
                s[s == 0] = 1.0
                Xnp = Xnp / s
            Z = Xnp.reshape((Xnp.shape[0], Xnp.shape[1] * Xnp.shape[2]))
            K = Z @ Z.T
            n_features = int(Xnp.shape[1])
            K = K / max(n_features, 1)
            if correlation_matrix:
                K = scale_kinship(K)
            out = _normalize_grm_output(K, ids=ids, attrs={"backend": "numpy", "n_features": n_features, "n_haplotypes_used": int(Xnp.shape[2]), "kinship_method": method, "qtl2_normalize_ploidy": bool(qtl2_normalize_ploidy), "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})

    if savefile is not None:
        _save_grm_simple(savefile, out, ids=ids, n_features=int(out.attrs.get("n_features", 0)))
    return out


def HGRM_lowmem(
    X,
    *,
    block_size=1024,
    center=True,
    scale=True,
    correlation_matrix=False,
    savefile=None,
    backend="auto",
    dtype=np.float32,
    reference="last",
    field=None,
    warmup_jax=True,
    method="standardized",
    qtl2_normalize_ploidy=True,
):
    da_x = _unwrap_haplotype_array(X, field=field)
    ids = da_x["iid"].to_numpy()
    hap_names = list(map(str, da_x["haplotype"].to_numpy()))
    Xd = da_x.data
    source_is_dask = _is_dask_array(Xd)
    method = str(method).lower()
    if method not in {"standardized", "qtl2"}:
        raise ValueError("method must be one of {'standardized','qtl2'}")

    if method == "standardized":
        ref_idx = _resolve_reference_haplotype(hap_names, reference=reference)
        if ref_idx is not None:
            keep = [i for i in range(len(hap_names)) if i != ref_idx]
            kept = [hap_names[i] for i in keep]
            Xd = Xd[:, :, keep]
        else:
            kept = hap_names
    else:
        ref_idx = None
        kept = hap_names

    n_samples, n_snps, n_haps = map(int, Xd.shape)
    chunk_len = int(block_size)
    block_bounds = [(s, min(s + chunk_len, n_snps)) for s in range(0, n_snps, chunk_len)]
    n_features = int(n_snps * n_haps) if method == "standardized" else int(n_snps)

    def _get_block(start, end):
        if source_is_dask:
            return np.asarray(Xd[:, start:end, :].compute(), dtype=dtype)
        return np.asarray(_to_numpy(Xd[:, start:end, :]), dtype=dtype)

    if backend == "dask":
        _require_dask()
        if not source_is_dask:
            Xd = da.asarray(np.asarray(_to_numpy(Xd), dtype=dtype), chunks=(n_samples, min(chunk_len, n_snps), n_haps))
            source_is_dask = True
        partials = []
        for s, e in block_bounds:
            block = Xd[:, s:e, :]
            if method == "standardized":
                def _partial(arr):
                    Z, _, _ = _standardize_haplotype_block_numpy(arr, center=center, scale=scale, reference=None, hap_names=kept)
                    Z = Z.reshape(Z.shape[0], -1)
                    return Z @ Z.T
            else:
                def _partial(arr):
                    Z = _qtl2_haplotype_block_numpy(arr, normalize_ploidy=qtl2_normalize_ploidy)
                    Z = Z.reshape(Z.shape[0], -1)
                    return Z @ Z.T
            part = delayed(_partial)(block)
            partials.append(da.from_delayed(part, shape=(n_samples, n_samples), dtype=np.float32))
        K = da.stack(partials, axis=0).sum(axis=0) if partials else da.zeros((n_samples, n_samples), dtype=np.float32)
        denom = max(n_features - 1, 1) if method == "standardized" else max(n_features, 1)
        K = K / denom
        if correlation_matrix:
            K = scale_kinship(K)
        out = _normalize_grm_output(K, ids=ids, attrs={"backend": "dask", "n_features": n_features, "kinship_method": method, "qtl2_normalize_ploidy": bool(qtl2_normalize_ploidy), "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})
        if savefile is not None:
            _save_grm_simple(savefile, out, ids=ids, n_features=n_features)
        return out

    if backend == "auto":
        backend = "jax" if _is_jax_array(Xd) else "numpy"
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be one of {'auto','numpy','dask','jax'}")

    if backend == "jax":
        jax, jnp = _require_jax()
        if method == "standardized":
            kernel = jax.jit(lambda block: _jax_prepare_hgrm_block(block, center=center, scale=scale))
        else:
            kernel = jax.jit(lambda block: _jax_prepare_hgrm_block_qtl2(block, normalize_ploidy=qtl2_normalize_ploidy))
        if warmup_jax and block_bounds:
            warmup_jax_hgrm(n_samples, chunk_len, n_haps, center=center, scale=scale, method=method, qtl2_normalize_ploidy=qtl2_normalize_ploidy, dtype=dtype)
        K = jnp.zeros((n_samples, n_samples), dtype=jnp.float32)
        for s, e in block_bounds:
            block = _get_block(s, e)
            if block.shape[1] < chunk_len:
                pad = np.zeros((n_samples, chunk_len - block.shape[1], n_haps), dtype=dtype)
                block = np.concatenate([block, pad], axis=1)
            gram, _ = kernel(jnp.asarray(block))
            if method == "qtl2" and block.shape[1] > (e - s):
                gram = gram  # padding is zero after qtl2 normalization on zero rows
            K = K + gram
        denom = max(n_features - 1, 1) if method == "standardized" else max(n_features, 1)
        K = K / denom
        if correlation_matrix:
            K = scale_kinship(K)
        out = _normalize_grm_output(K, ids=ids, attrs={"backend": "jax", "n_features": n_features, "kinship_method": method, "qtl2_normalize_ploidy": bool(qtl2_normalize_ploidy), "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})
    else:
        K = np.zeros((n_samples, n_samples), dtype=np.float64)
        for s, e in block_bounds:
            block = _get_block(s, e)
            if method == "standardized":
                Z, _, _ = _standardize_haplotype_block_numpy(block, center=center, scale=scale, reference=None, hap_names=kept)
            else:
                Z = _qtl2_haplotype_block_numpy(block, normalize_ploidy=qtl2_normalize_ploidy)
            Z = Z.reshape(Z.shape[0], -1)
            K += Z @ Z.T
        denom = max(n_features - 1, 1) if method == "standardized" else max(n_features, 1)
        K = K / denom
        if correlation_matrix:
            K = scale_kinship(K)
        out = _normalize_grm_output(K, ids=ids, attrs={"backend": "numpy", "n_features": n_features, "kinship_method": method, "qtl2_normalize_ploidy": bool(qtl2_normalize_ploidy), "kept_haplotypes": kept, "reference_haplotype_index": ref_idx})

    if savefile is not None:
        _save_grm_simple(savefile, out, ids=ids, n_features=n_features)
    return out


def calc_haplotype_kinship(
    X,
    *,
    type="overall",
    method="qtl2",
    omit_x=False,
    chromosomes=None,
    field=None,
    backend="auto",
    block_size=1024,
    center=True,
    scale=True,
    reference="last",
    correlation_matrix=False,
    warmup_jax=True,
    qtl2_normalize_ploidy=True,
    dtype=np.float32,
):
    """Calculate overall, per-chromosome, or LOCO kinship from haplotype probabilities.

    For `method="qtl2"`, this mirrors qtl2's kinship construction more closely:
    use all haplotypes, convert additive dosages to per-locus allele probabilities by
    normalizing each sample-locus vector to sum to 1, and average `sum_l p_ikl p_jkl`
    over loci. `correlation_matrix=True` then applies qtl2's `scale_kinship()` rule.
    """
    da_x = _unwrap_haplotype_array(X, field=field)
    chrom = pd.Index(da_x["chrom"].to_numpy()).astype(str)
    chroms = list(pd.unique(chrom))
    if omit_x:
        drop = {"x", "X"}
        chroms = [c for c in chroms if c not in drop]
    if chromosomes is not None:
        wanted = {str(c) for c in chromosomes}
        chroms = [c for c in chroms if c in wanted]
    type = str(type).lower()
    if type not in {"overall", "chr", "loco"}:
        raise ValueError("type must be one of {'overall','chr','loco'}")

    if type == "overall":
        mask = np.isin(chrom.to_numpy(), chroms)
        sub = da_x.isel(snp=np.where(mask)[0])
        return HGRM_lowmem(sub, block_size=block_size, center=center, scale=scale, correlation_matrix=correlation_matrix, backend=backend, dtype=dtype, reference=reference, warmup_jax=warmup_jax, method=method, qtl2_normalize_ploidy=qtl2_normalize_ploidy)

    per_chr = {}
    for c in chroms:
        mask = (chrom.to_numpy() == c)
        sub = da_x.isel(snp=np.where(mask)[0])
        per_chr[str(c)] = HGRM_lowmem(sub, block_size=block_size, center=center, scale=scale, correlation_matrix=False, backend=backend, dtype=dtype, reference=reference, warmup_jax=warmup_jax, method=method, qtl2_normalize_ploidy=qtl2_normalize_ploidy)
    if type == "chr":
        if correlation_matrix:
            return {c: scale_kinship(K) for c, K in per_chr.items()}
        return per_chr

    overall = HGRM_lowmem(da_x.isel(snp=np.where(np.isin(chrom.to_numpy(), chroms))[0]), block_size=block_size, center=center, scale=scale, correlation_matrix=False, backend=backend, dtype=dtype, reference=reference, warmup_jax=warmup_jax, method=method, qtl2_normalize_ploidy=qtl2_normalize_ploidy)
    out = {}
    overall_arr = _to_numpy(overall)
    overall_denom = float(overall.attrs.get("n_features", int(da_x.sizes.get("snp", 0))))
    for c, Kc in per_chr.items():
        arr_c = _to_numpy(Kc)
        denom_c = float(Kc.attrs.get("n_features", int((chrom == c).sum())))
        if method == "standardized":
            raw_all = overall_arr * max(overall_denom - 1.0, 1.0)
            raw_c = arr_c * max(denom_c - 1.0, 1.0)
            denom_sub = max(overall_denom - denom_c - 1.0, 1.0)
        else:
            raw_all = overall_arr * max(overall_denom, 1.0)
            raw_c = arr_c * max(denom_c, 1.0)
            denom_sub = max(overall_denom - denom_c, 1.0)
        sub = (raw_all - raw_c) / denom_sub
        Ka = xr.DataArray(sub, dims=overall.dims, coords=overall.coords, attrs=dict(overall.attrs), name="grm")
        Ka.attrs["type"] = "loco"
        Ka.attrs["left_out_chromosome"] = str(c)
        Ka.attrs["n_features"] = int(max(overall_denom - denom_c, 1.0))
        if correlation_matrix:
            Ka = scale_kinship(Ka)
        out[str(c)] = Ka
    return out


# -----------------------------------------------------------------------------
# HWAS design + kernels
# -----------------------------------------------------------------------------

def prepare_haplotype_design(
    X,
    precision=np.float32,
    center=False,
    scale=False,
    reference="last",
    add_intercept=True,
    backend="auto",
):
    """Prepare a per-locus HWAS design tensor.

    This assumes there is no genotype missingness mask. Trait missingness is handled
    later inside the regression step.
    """
    da_x = _unwrap_haplotype_array(X) if not isinstance(X, xr.DataArray) else X
    hap_names = list(map(str, da_x["haplotype"].to_numpy())) if isinstance(da_x, xr.DataArray) else None
    backend = _infer_backend(da_x, backend)
    Xnp = _to_numpy(da_x).astype(precision, copy=False)
    Xnp, kept, ref_idx = _drop_reference(Xnp, hap_names, reference=reference)
    if center:
        Xnp = Xnp - Xnp.mean(axis=0, keepdims=True)
    if scale:
        sd = Xnp.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Xnp = Xnp / sd
    if add_intercept:
        intercept = np.ones((Xnp.shape[0], Xnp.shape[1], 1), dtype=precision)
        Xnp = np.concatenate([intercept, Xnp], axis=2)
    if backend == "jax":
        _, jnp = _require_jax()
        Xout = jnp.asarray(Xnp)
    elif backend == "dask":
        Xout = _require_dask().asarray(Xnp)
    else:
        Xout = Xnp
    return Xout, kept, ref_idx


def _numpy_regression_core(design, Y, mask, *, dof="correct"):
    design = np.asarray(design, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    M = design.shape[1]
    p = design.shape[2]
    XtY = np.einsum("nmp,nt->mtp", design, Y, optimize=True)
    XtX = np.einsum("nmp,nt,nmq->mtpq", design, mask, design, optimize=True)
    ysq = Y * Y
    yty0 = np.einsum("nt,nt->t", ysq, mask, optimize=True)
    sumy0 = np.einsum("nt,nt->t", Y, mask, optimize=True)
    nobs0 = mask.sum(axis=0)
    yty = np.broadcast_to(yty0[None, :], (M, mask.shape[1])).copy()
    sumy = np.broadcast_to(sumy0[None, :], (M, mask.shape[1])).copy()
    nobs = np.broadcast_to(nobs0[None, :], (M, mask.shape[1])).astype(np.float64, copy=True)

    xtx_inv = np.linalg.pinv(XtX)
    beta_full = np.matmul(xtx_inv, XtY[..., None])[..., 0]
    sse = yty - np.einsum("mtp,mtp->mt", beta_full, XtY, optimize=True)
    sse = np.clip(sse, 0.0, None)

    if dof == "incorrect":
        df_denom = np.broadcast_to(mask.sum(axis=0) - p, nobs.shape).astype(np.float64, copy=True)
    else:
        df_denom = nobs - p
    df_denom[df_denom <= 0] = np.nan
    sigma2 = np.divide(sse, df_denom, out=np.full_like(sse, np.nan), where=np.isfinite(df_denom) & (df_denom > 0))
    beta_var = xtx_inv * sigma2[:, :, None, None]
    beta_se_full = np.sqrt(np.clip(np.diagonal(beta_var, axis1=-2, axis2=-1), 0.0, None))

    sse0 = yty - np.divide(sumy * sumy, nobs, out=np.full_like(yty, np.nan), where=nobs > 0)
    sse0 = np.clip(sse0, 0.0, None)
    ss_model = np.clip(sse0 - sse, 0.0, None)
    return beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom


def _jax_regression_core(design, Y, mask, *, dof="correct"):
    _, jnp = _require_jax()
    design = jnp.asarray(design, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    M = design.shape[1]
    p = design.shape[2]
    XtY = jnp.einsum("nmp,nt->mtp", design, Y, optimize=True)
    XtX = jnp.einsum("nmp,nt,nmq->mtpq", design, mask, design, optimize=True)
    ysq = Y * Y
    yty0 = jnp.einsum("nt,nt->t", ysq, mask, optimize=True)
    sumy0 = jnp.einsum("nt,nt->t", Y, mask, optimize=True)
    nobs0 = mask.sum(axis=0)
    yty = jnp.broadcast_to(yty0[None, :], (M, mask.shape[1]))
    sumy = jnp.broadcast_to(sumy0[None, :], (M, mask.shape[1]))
    nobs = jnp.broadcast_to(nobs0[None, :], (M, mask.shape[1]))

    xtx_inv = jnp.linalg.pinv(XtX)
    beta_full = jnp.matmul(xtx_inv, XtY[..., None])[..., 0]
    sse = yty - jnp.einsum("mtp,mtp->mt", beta_full, XtY, optimize=True)
    sse = jnp.clip(sse, 0.0, None)

    if dof == "incorrect":
        df_denom = jnp.broadcast_to(mask.sum(axis=0) - p, nobs.shape)
    else:
        df_denom = nobs - p
    df_denom = jnp.where(df_denom <= 0, jnp.nan, df_denom)
    sigma2 = sse / df_denom
    beta_var = xtx_inv * sigma2[:, :, None, None]
    beta_se_full = jnp.sqrt(jnp.clip(jnp.diagonal(beta_var, axis1=-2, axis2=-1), 0.0, None))

    sse0 = yty - (sumy * sumy) / nobs
    sse0 = jnp.clip(sse0, 0.0, None)
    ss_model = jnp.clip(sse0 - sse, 0.0, None)
    return beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom


def warmup_jax_hwas(n_samples, block_variants, n_predictors, n_traits, *, dtype=np.float32, dof="correct"):
    jax, jnp = _require_jax()
    kernel = jax.jit(lambda X, Y, M: _jax_regression_core(X, Y, M, dof=dof))
    X = jnp.zeros((int(n_samples), int(block_variants), int(n_predictors)), dtype=dtype)
    Y = jnp.zeros((int(n_samples), int(n_traits)), dtype=dtype)
    M = jnp.ones((int(n_samples), int(n_traits)), dtype=dtype)
    out = kernel(X, Y, M)
    out[0].block_until_ready()
    return kernel


def regression_hwas_ols(
    design,
    straits,
    traits_mask=None,
    dof="correct",
    stat="ttest",
    sided="two-sided",
    regression_type=None,
    backend="numpy",
    warmup_jax=False,
):
    """Vectorized per-locus multi-haplotype regression.

    `regression_type` controls the locus summary:
    - 'ttest' or 'f'  -> coefficient t-tests + omnibus F test
    - 'wald'          -> coefficient Wald/z tests + omnibus Wald chi-square
    - 'hk' or 'lod'   -> coefficient t-tests + omnibus LOD
    """
    if sided not in {"two-sided", "one-sided"}:
        raise ValueError("sided must be 'two-sided' or 'one-sided'")
    regression_type = stat if regression_type is None else regression_type
    if regression_type not in {"ttest", "f", "wald", "hk", "lod"}:
        raise ValueError("regression_type must be one of ['ttest','f','wald','hk','lod']")

    if traits_mask is None:
        traits_mask = np.isfinite(np.asarray(straits, dtype=np.float64)).astype(np.float32)
    Y = np.nan_to_num(np.asarray(straits, dtype=np.float32), nan=0.0)
    mask = np.asarray(traits_mask, dtype=np.float32)
    coeff_test = "wald" if regression_type == "wald" else "ttest"
    locus_test = {"ttest": "f", "f": "f", "wald": "wald", "hk": "lod", "lod": "lod"}[regression_type]

    if backend == "jax":
        jax, jnp = _require_jax()
        if warmup_jax:
            warmup_jax_hwas(design.shape[0], design.shape[1], design.shape[2], Y.shape[1], dtype=np.float32, dof=dof)
        kernel = jax.jit(lambda X, Yv, Mv: _jax_regression_core(X, Yv, Mv, dof=dof))
        beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom = kernel(jnp.asarray(design), jnp.asarray(Y), jnp.asarray(mask))
        beta_full = np.asarray(beta_full)
        beta_se_full = np.asarray(beta_se_full)
        sse = np.asarray(sse)
        sse0 = np.asarray(sse0)
        ss_model = np.asarray(ss_model)
        nobs = np.asarray(nobs)
        df_denom = np.asarray(df_denom)
    else:
        beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom = _numpy_regression_core(design, Y, mask, dof=dof)

    p = int(beta_full.shape[2])
    df_num = p - 1
    sigma2 = np.divide(sse, df_denom, out=np.full_like(sse, np.nan), where=np.isfinite(df_denom) & (df_denom > 0))

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
        stats = zstats * zstats

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

    return {
        "beta": np.transpose(beta, (0, 2, 1)),
        "beta_se": np.transpose(beta_se, (0, 2, 1)),
        "stat": np.transpose(stats, (0, 2, 1)),
        "neglog_p": np.transpose(neglog_p, (0, 2, 1)),
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


# -----------------------------------------------------------------------------
# User-facing HWAS wrappers
# -----------------------------------------------------------------------------

def _hwas_block_numpy(
    block_arr,
    trait_values,
    trait_mask,
    *,
    hap_center=False,
    hap_scale=False,
    reference="last",
    regression_type=None,
    stat="ttest",
    sided="two-sided",
    dof="correct",
    backend="numpy",
    warmup_jax=False,
):
    design, kept_haplotypes, ref_idx = prepare_haplotype_design(
        xr.DataArray(block_arr, dims=("iid", "snp", "haplotype"), coords={"haplotype": np.arange(block_arr.shape[2])}),
        center=hap_center,
        scale=hap_scale,
        reference=reference,
        add_intercept=True,
        backend="numpy" if backend != "jax" else "jax",
    )
    res = regression_hwas_ols(
        design,
        trait_values,
        traits_mask=trait_mask,
        dof=dof,
        stat=stat,
        sided=sided,
        regression_type=regression_type,
        backend=backend,
        warmup_jax=warmup_jax,
    )
    return res, kept_haplotypes, ref_idx


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
    backend="auto",
    block_variants=None,
    warmup_jax=True,
    pad_last_chunk=True,
    field=None,
):
    """Haplotype-based association scan.

    Execution model
    ---------------
    - Dask handles out-of-core chunking and scheduling.
    - NumPy or JAX handles the actual block math.
    - For JAX, fixed-shape block kernels are warmed up once by default and the
      last block is optionally padded so the compiled executable is reused.
    """
    traitdf = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
    if center:
        traitdf = _center_traits_with_mask(traitdf)
    trait_values = np.nan_to_num(traitdf.to_numpy(dtype=precision), nan=0.0)
    trait_mask = np.isfinite(traits.to_numpy(dtype=float) if isinstance(traits, pd.DataFrame) else np.asarray(traits, dtype=float)).astype(np.float32)
    trait_names = traitdf.columns.astype(str).tolist()

    hda = _unwrap_haplotype_array(haplotypes, field=field)
    if isinstance(traits, pd.DataFrame) and "iid" in hda.coords:
        common = [x for x in traitdf.index.astype(str) if x in set(map(str, hda["iid"].to_numpy()))]
        hda = hda.sel(iid=common)
        traitdf = traitdf.loc[common]
        trait_values = np.nan_to_num(traitdf.to_numpy(dtype=precision), nan=0.0)
        trait_mask = np.isfinite(traitdf.to_numpy(dtype=float)).astype(np.float32)
        trait_names = traitdf.columns.astype(str).tolist()

    backend = _infer_backend(hda, backend=backend)
    source_is_dask = _is_dask_array(hda.data)
    if block_variants is None:
        if source_is_dask:
            block_variants = int(hda.data.chunks[1][0])
        else:
            block_variants = int(hda.sizes["snp"])
    block_variants = max(1, int(block_variants))

    hap_names = list(map(str, hda["haplotype"].to_numpy()))
    kept_example = [h for i, h in enumerate(hap_names) if i != (_resolve_reference_haplotype(hap_names, reference=reference) if _resolve_reference_haplotype(hap_names, reference=reference) is not None else -1)]
    if _resolve_reference_haplotype(hap_names, reference=reference) is None:
        kept_example = hap_names
    p_fixed = len(kept_example) + 1

    if backend == "jax" and warmup_jax:
        warmup_jax_hwas(int(hda.sizes["iid"]), block_variants, p_fixed, len(trait_names), dtype=precision, dof=dof)

    block_results = []
    snp_names_all = []
    chrom_all = []
    pos_all = []
    ref_all = []
    alt_all = []
    kept_haplotypes = None

    n_snps = int(hda.sizes["snp"])
    for start in range(0, n_snps, block_variants):
        stop = min(start + block_variants, n_snps)
        block = hda.isel(snp=slice(start, stop))
        if source_is_dask:
            block_arr = np.asarray(block.data.compute(), dtype=precision)
        else:
            block_arr = np.asarray(_to_numpy(block), dtype=precision)
        true_len = block_arr.shape[1]
        if backend == "jax" and pad_last_chunk and true_len < block_variants:
            pad = np.zeros((block_arr.shape[0], block_variants - true_len, block_arr.shape[2]), dtype=precision)
            block_arr = np.concatenate([block_arr, pad], axis=1)

        res, kept_haplotypes, _ = _hwas_block_numpy(
            block_arr,
            trait_values,
            trait_mask,
            hap_center=hap_center,
            hap_scale=hap_scale,
            reference=reference,
            regression_type=regression_type,
            stat=stat,
            sided=sided,
            dof=dof,
            backend="jax" if backend == "jax" else "numpy",
            warmup_jax=False,
        )
        if true_len < block_variants:
            for key in [
                "beta", "beta_se", "stat", "neglog_p",
                "dof", "omnibus_stat", "omnibus_neglog_p",
                "omnibus_df_num", "omnibus_df_denom",
                "omnibus_f_stat", "omnibus_f_neglog_p",
                "omnibus_wald_stat", "omnibus_wald_neglog_p", "omnibus_lod",
            ]:
                res[key] = res[key][:true_len]
        block_results.append(res)
        snp_names_all.extend(block["snp"].to_numpy().tolist())
        chrom_all.extend(block["chrom"].to_numpy().tolist())
        pos_all.extend(block["pos"].to_numpy().tolist())
        ref_all.extend(block["ref"].to_numpy().tolist())
        alt_all.extend(block["alt"].to_numpy().tolist())

    if not block_results:
        return pd.DataFrame() if "pandas" in dtype else xr.Dataset()

    def _cat(key):
        return np.concatenate([b[key] for b in block_results], axis=0)

    res = {key: _cat(key) for key in [
        "beta", "beta_se", "stat", "neglog_p",
        "dof", "omnibus_stat", "omnibus_neglog_p",
        "omnibus_df_num", "omnibus_df_denom",
        "omnibus_f_stat", "omnibus_f_neglog_p",
        "omnibus_wald_stat", "omnibus_wald_neglog_p", "omnibus_lod",
    ]}
    res.update({k: block_results[0][k] for k in ["regression_type", "coef_test", "locus_test", "omnibus_stat_name"]})

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
            "snp": np.asarray(snp_names_all, dtype=object),
            "haplotype": np.asarray(kept_haplotypes, dtype=object),
            "trait": np.asarray(trait_names, dtype=object),
            "chrom": ("snp", np.asarray(chrom_all, dtype=object)),
            "pos": ("snp", np.asarray(pos_all)),
            "ref": ("snp", np.asarray(ref_all, dtype=object)),
            "alt": ("snp", np.asarray(alt_all, dtype=object)),
        },
        attrs={
            "reference_haplotype": reference,
            "haplotype_centered": bool(hap_center),
            "haplotype_scaled": bool(hap_scale),
            "regression_type": res["regression_type"],
            "coef_test": res["coef_test"],
            "locus_test": res["locus_test"],
            "omnibus_stat_name": res["omnibus_stat_name"],
            "backend": backend,
            "block_variants": int(block_variants),
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
            "dof", "omnibus_stat", "omnibus_neglog_p",
            "omnibus_df_num", "omnibus_df_denom",
            "omnibus_f_stat", "omnibus_f_neglog_p",
            "omnibus_wald_stat", "omnibus_wald_neglog_p", "omnibus_lod",
        ]].to_dataframe().reset_index()
        out = founder.merge(omni, on=["snp", "trait"], how="left")
        out["regression_type"] = res["regression_type"]
        out["coef_test"] = res["coef_test"]
        out["locus_test"] = res["locus_test"]
        return out
    raise ValueError("dtype must be one of ['tuple','xarray','xarray_dataset','pandas','pandas_highmem']")


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
    backend="auto",
    jax_warmup=True,
    jax_pad_last_chunk=True,
):
    """Multivariate HWAS mirroring the mvGWAS workflow.

    Execution pattern
    -----------------
    - The BCF is held as a Dask-backed xarray Dataset for out-of-core block loading.
    - Each genomic block is passed to HWAS.
    - HWAS uses NumPy or JAX as the hot math backend.
    """
    current_mem = lambda: f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f}Gb"

    if isinstance(grms_folder, str):
        grms_folder = glob(f"{grms_folder}/*.grm.bin")
    in_memory_kinship = isinstance(grms_folder, dict)
    if regression_mode != "ols":
        raise ValueError("mvHWAS currently supports regression_mode='ols' only")

    read_hap = load_stitch_bcf_xarray(
        haplotypes,
        chunk_variants=chunk_variants,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        backend="dask",
    ) if isinstance(haplotypes, str) else haplotypes

    if isinstance(read_hap, xr.DataArray):
        read_hap = read_hap.to_dataset(name=read_hap.name or "haplotypes")

    hap_field = _default_haplotype_field_from_dataset(read_hap, None)
    chrunique = [str(x) for x in pd.Index(read_hap["chrom"].to_numpy()).astype(str).unique().tolist()]
    if chrset is not None:
        chrset = {str(x) for x in chrset}
        chrunique = [x for x in chrunique if x in chrset]

    npplink = None if in_memory_kinship else _require_npplink()
    gdv2 = _require_gdv2()
    if in_memory_kinship:
        kinship_map = {str(k): (v.to_pandas() if isinstance(v, xr.DataArray) else pd.DataFrame(np.asarray(v), index=traitdf.index, columns=traitdf.index)) for k, v in grms_folder.items()}
        grm_index = pd.Index(list(kinship_map.keys()), dtype="object")
        if not grm_index.isin(chrunique).any():
            raise ValueError("Provided kinship dict does not contain chromosome keys matching the haplotype data.")
    else:
        grms_folder = pd.DataFrame(grms_folder, columns=["path"])
        grms_folder.index = grms_folder["path"].str.extract(r"([\d\w_]+)chrGRM.", expand=False)
        grms_folder = grms_folder.sort_index(
            key=lambda idx: idx.str.lower().map(
                {str(i): int(i) for i in range(1000)} | {i: int(i) for i in range(1000)} | {"all": -1000, "x": 1001, "y": 1002, "mt": 1003, "m": 1003}
            )
        )
        grms_folder = grms_folder[~grms_folder.index.isna()]
        allGRM = npplink.read_grm(grms_folder.loc["All", "path"].replace(".bin", ""))
        grms_folder["in_chrunique"] = grms_folder.index.isin(chrunique)
        grms_folder["isnum"] = grms_folder.index.str.isnumeric()
        max_grm_chr = grms_folder.query("isnum").index.astype(int).max()
        if grms_folder.in_chrunique.eq(False).sum() > 1:
            grms_folder = grms_folder.rename({
                "x": str(max_grm_chr + 1), "y": str(max_grm_chr + 2), "mt": str(max_grm_chr + 4),
                "X": str(max_grm_chr + 1), "Y": str(max_grm_chr + 2), "MT": str(max_grm_chr + 4),
            })
            grms_folder["in_chrunique"] = grms_folder.index.isin(chrunique)
            if grms_folder.in_chrunique.eq(False).sum() > 1:
                raise ValueError("cannot match chromosomes in grm folder and haplotype file")

    chrom_sizes = pd.DataFrame({"chrom": read_hap["chrom"].to_numpy().astype(str), "pos": read_hap["pos"].to_numpy()}).groupby("chrom").pos.max()
    sumstats = []
    if save:
        os.makedirs(save_path, exist_ok=True)

    grm_iter = list(grms_folder.drop(["All"]).iterrows()) if not in_memory_kinship else [(c, None) for c in chrunique if str(c) in kinship_map]
    for c, row in (pbar := tqdm(grm_iter)):
        if str(c) not in chrunique:
            continue
        pbar.set_description(f"HWAS-Chr{c}-reading GRM")
        if in_memory_kinship:
            subgrm = kinship_map[str(c)].copy()
            c_grm = None
        else:
            c_grm = npplink.read_grm(row.path.replace(".bin", ""))
            subgrm = allGRM["grm"].to_pandas() if not row.isnum else ((allGRM["grm"] * allGRM["w"] - c_grm["grm"] * c_grm["w"]) / (allGRM["w"] - c_grm["w"])).to_pandas()

        pbar.set_description(f"HWAS-Chr{c}-estimating var/covar")
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
                pbar.set_description(f"HWAS-Chr{c}-scan-MEM:{current_mem()}")
                hblock = stitch2array(read_hap, c=c, rfids=traits.index.astype(str).tolist(), field=hap_field, compute=False)
                suffix = ""
            else:
                pbar.set_description(f"HWAS-Chr{c}-scan[{start / 1e6:.0f}-{stop / 1e6:.0f}]Mb-MEM:{current_mem()}")
                hblock = stitch2array(read_hap, c=c, rfids=traits.index.astype(str).tolist(), pos_start=start, pos_end=stop, field=hap_field, compute=False)
                suffix = f"_{int(start)}_{int(stop)}"
            if int(hblock.sizes.get("snp", 0)) == 0:
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
                backend=backend,
                block_variants=chunk_variants,
                warmup_jax=jax_warmup,
                pad_last_chunk=jax_pad_last_chunk,
            )
            if save and hasattr(block, "to_parquet"):
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

# =============================================================================
# Optimized production overrides
# =============================================================================
# These definitions intentionally override the earlier prototype definitions above.
# They keep the same public API while adding:
#   * single-pass BCF schema/haplotype detection inside scan_stitch_bcf()
#   * production BCF parsing modes that read only the haplotype FORMAT field
#   * reusable JAX kernels and trait-only precomputation
#   * Dask prefetching of lazy variant blocks
#   * block streaming to parquet/csv/pickle without accumulating a full chromosome
#   * valid HK/LOD p-value handling through the equivalent F-test by default


def _selected_format_fields(format_specs, format_fields=None, *, haplotype_field=None, parse_mode="full"):
    """Return a filtered FORMAT-spec dictionary for BCF loading."""
    if format_fields is None:
        format_fields = "haplotype" if str(parse_mode).lower() in {"hwas", "haplotype", "haplotypes", "production"} else "all"
    if isinstance(format_fields, str):
        key = format_fields.lower()
        if key in {"all", "full", "*"}:
            fields = list(format_specs)
        elif key in {"haplotype", "haplotypes", "hwas", "selected"}:
            if haplotype_field is None:
                raise ValueError("haplotype_field must be known when format_fields='haplotype'.")
            fields = [str(haplotype_field)]
        else:
            fields = [format_fields]
    else:
        fields = [str(f) for f in format_fields]
    missing = [f for f in fields if f not in format_specs]
    if missing:
        raise KeyError(f"Requested FORMAT fields are not in the BCF header/schema: {missing}")
    return {f: dict(format_specs[f]) for f in fields}


def scan_stitch_bcf(
    bcf_path,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    schema_scan_variants=128,
):
    """Scan variant metadata, INFO fields, and FORMAT schemas in one BCF pass.

    This optimized version folds haplotype-field auto-detection into the metadata
    scan. The older implementation opened the BCF once for auto-detection and a
    second time for full metadata/schema scanning.
    """
    pysam = _require_pysam()
    preferred = ["AP", "HP", "HAP", "HAPROB", "HAPPROB", "HDS", "DS", "GP"]
    fixed_hap_field = None if haplotype_field in [None, "auto"] else str(haplotype_field)

    records = []
    info_raw = None
    selected_hap_field = fixed_hap_field
    best_field = None
    best_score = (-1, -1, -1)
    observed_components = {}

    with pysam.VariantFile(bcf_path) as vf:
        sample_ids = list(vf.header.samples)
        probes = sample_ids[: min(4, len(sample_ids))]

        info_specs = {}
        for field, meta in vf.header.info.items():
            info_specs[str(field)] = {
                "dtype": _field_dtype(str(field), getattr(meta, "type", None)),
                "ncomp": 1,
                "type": getattr(meta, "type", None),
                "number": getattr(meta, "number", None),
            }
        format_specs = {}
        for field, meta in vf.header.formats.items():
            format_specs[str(field)] = {
                "dtype": _field_dtype(str(field), getattr(meta, "type", None)),
                "ncomp": 1,
                "type": getattr(meta, "type", None),
                "number": getattr(meta, "number", None),
                "collapse_phased": False,
            }
        info_raw = {field: [] for field in info_specs}

        if fixed_hap_field is not None and fixed_hap_field not in format_specs:
            raise KeyError(f"Requested haplotype FORMAT field {fixed_hap_field!r} is not in the BCF header.")

        for i, rec in enumerate(vf):
            if max_variants is not None and i >= max_variants:
                break
            records.append(
                {
                    "chrom": str(rec.contig),
                    "pos": int(rec.pos),
                    "snp": rec.id if rec.id not in [None, "."] else f"{rec.contig}:{rec.pos}:{rec.ref}:{_normalize_alt_tuple(rec.alts)}",
                    "id": "." if rec.id is None else str(rec.id),
                    "ref": str(rec.ref),
                    "alt": _normalize_alt_tuple(rec.alts),
                    "qual": np.nan if rec.qual is None else float(rec.qual),
                    "filter": _normalize_filter_value(rec.filter.keys() if hasattr(rec.filter, "keys") else rec.filter),
                    "i": i,
                }
            )

            for field in info_specs:
                val = rec.info.get(field)
                info_raw[field].append(val)
                info_specs[field]["ncomp"] = max(info_specs[field]["ncomp"], max(1, _infer_components(val, collapse_phased=False)))

            if i < schema_scan_variants:
                for sid in probes:
                    s = rec.samples[sid]
                    for field, spec in format_specs.items():
                        if field not in rec.format:
                            continue
                        val = s.get(field)
                        raw_n = max(1, _infer_components(val, collapse_phased=False))
                        observed_components[field] = max(observed_components.get(field, 1), raw_n)
                        spec["ncomp"] = max(spec["ncomp"], raw_n)

                        if fixed_hap_field is None and raw_n > 1:
                            # Prefer known STITCH/founder-probability names, then larger vector fields.
                            pref_rank = (len(preferred) - preferred.index(field)) if field in preferred else 0
                            score = (pref_rank, raw_n, int(raw_n > 2))
                            if score > best_score:
                                best_score = score
                                best_field = field

        if fixed_hap_field is None:
            selected_hap_field = best_field
        if selected_hap_field is None:
            raise ValueError("Could not infer the haplotype FORMAT field. Pass haplotype_field explicitly.")

        raw_hap_components = int(observed_components.get(selected_hap_field, format_specs[selected_hap_field]["ncomp"]))
        n_haplotypes = int(_infer_components([0] * raw_hap_components, collapse_phased=collapse_phased))
        format_specs[selected_hap_field]["collapse_phased"] = bool(collapse_phased)
        format_specs[selected_hap_field]["ncomp"] = max(1, n_haplotypes)

    if n_haplotypes <= 0:
        raise ValueError("Could not infer the number of haplotypes from the chosen FORMAT field.")

    variants = pd.DataFrame.from_records(records)
    samples = pd.DataFrame({"iid": sample_ids, "i": np.arange(len(sample_ids), dtype=int)})

    info_arrays = {}
    n_variants = len(variants)
    for field, spec in info_specs.items():
        ncomp = max(1, int(spec["ncomp"]))
        arr = []
        for val in info_raw[field]:
            arr.append(_convert_value(val, ncomp=ncomp, dtype=spec["dtype"], collapse_phased=False))
        if ncomp == 1:
            info_arrays[field] = np.asarray(arr, dtype=spec["dtype"] if spec["dtype"] is not object else object)
        else:
            info_arrays[field] = np.asarray(arr, dtype=spec["dtype"] if spec["dtype"] is not object else object).reshape(n_variants, ncomp)

    variants.attrs["haplotype_field"] = str(selected_hap_field)
    variants.attrs["n_haplotypes"] = int(n_haplotypes)
    variants.attrs["format_specs"] = format_specs
    variants.attrs["info_specs"] = info_specs
    variants.attrs["info_arrays"] = info_arrays
    samples.attrs["haplotype_field"] = str(selected_hap_field)
    samples.attrs["n_haplotypes"] = int(n_haplotypes)
    return variants, samples


def load_stitch_bcf_xarray(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    backend="dask",
    *,
    parse_mode="full",
    format_fields=None,
    include_info=True,
):
    """Lazily parse a BCF into an xarray.Dataset.

    Parameters
    ----------
    parse_mode : {'full', 'hwas', 'haplotype'}, default='full'
        'full' preserves all FORMAT fields. 'hwas'/'haplotype' only creates the
        selected haplotype FORMAT field, reducing BCF parsing and Dask graph size.
    format_fields : str | sequence | None
        Explicit FORMAT fields to parse. Use 'all' for every FORMAT field or
        'haplotype' for only the selected haplotype field.
    include_info : bool, default=True
        If False, skip INFO_* variables in the xarray Dataset. Coordinates still
        include chrom/pos/id/ref/alt/qual/filter.
    """
    _require_dask()
    variants, samples = scan_stitch_bcf(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
    )
    hap_field = variants.attrs["haplotype_field"]
    format_specs_all = variants.attrs["format_specs"]
    format_specs = _selected_format_fields(format_specs_all, format_fields, haplotype_field=hap_field, parse_mode=parse_mode)
    # Use requested numeric dtype for the selected haplotype field unless the header is non-numeric.
    if hap_field in format_specs and np.dtype(format_specs[hap_field]["dtype"]).kind in "iuifc":
        format_specs[hap_field] = dict(format_specs[hap_field])
        format_specs[hap_field]["dtype"] = dtype

    info_specs = variants.attrs["info_specs"]
    info_arrays = variants.attrs["info_arrays"]

    chunk_readers = []
    chunk_slices = []
    for offset in range(0, len(variants), chunk_variants):
        vchunk = variants.iloc[offset: offset + chunk_variants].copy()
        chunk_slices.append((offset, offset + len(vchunk)))
        chunk_readers.append(
            delayed(_chunk_reader_all_formats)(
                bcf_path,
                vchunk,
                samples["iid"].tolist(),
                format_specs,
            )
        )

    data_vars = {}
    coords = {
        "iid": samples["iid"].to_numpy(),
        "snp": variants["snp"].to_numpy(),
        "chrom": ("snp", variants["chrom"].to_numpy()),
        "pos": ("snp", variants["pos"].to_numpy()),
        "variant_id": ("snp", variants["id"].to_numpy()),
        "ref": ("snp", variants["ref"].to_numpy()),
        "alt": ("snp", variants["alt"].to_numpy()),
        "qual": ("snp", variants["qual"].to_numpy()),
        "filter": ("snp", variants["filter"].to_numpy()),
    }

    for field, spec in format_specs.items():
        arr_chunks = []
        ndim = 2 if int(spec["ncomp"]) == 1 else 3
        for reader, (start, stop) in zip(chunk_readers, chunk_slices):
            nchunk = stop - start
            arr = delayed(lambda d, k: d[k])(reader, field)
            shape = (len(samples), nchunk) if ndim == 2 else (len(samples), nchunk, int(spec["ncomp"]))
            arr_chunks.append(da.from_delayed(arr, shape=shape, dtype=spec["dtype"]))
        full = da.concatenate(arr_chunks, axis=1) if arr_chunks else da.empty((len(samples), 0), dtype=spec["dtype"])
        if ndim == 2:
            data_vars[field] = (("iid", "snp"), full)
        else:
            comp_dim = f"{field}_component"
            coords[comp_dim] = np.arange(int(spec["ncomp"]), dtype=int)
            data_vars[field] = (("iid", "snp", comp_dim), full)

    if include_info:
        for field, spec in info_specs.items():
            arr = info_arrays[field]
            var_name = f"INFO_{field}"
            if arr.ndim == 1:
                data_vars[var_name] = (("snp",), arr)
            else:
                comp_dim = f"INFO_{field}_component"
                coords[comp_dim] = np.arange(arr.shape[1], dtype=int)
                data_vars[var_name] = (("snp", comp_dim), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "bcf_path": str(bcf_path),
            "haplotype_field": hap_field,
            "n_haplotypes": int(variants.attrs["n_haplotypes"]),
            "collapse_phased": bool(collapse_phased),
            "parse_mode": str(parse_mode),
            "format_fields": tuple(format_specs.keys()),
            "include_info": bool(include_info),
        },
    )
    return _maybe_materialize_dataset(ds, backend=backend)


def load_stitch_haplotypes_xarray(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased=True,
    max_variants=None,
    backend="dask",
    include_info=False,
):
    """Production loader: lazily parse only the selected haplotype FORMAT field."""
    return load_stitch_bcf_xarray(
        bcf_path,
        chunk_variants=chunk_variants,
        dtype=dtype,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
        backend=backend,
        parse_mode="hwas",
        format_fields="haplotype",
        include_info=include_info,
    )


# expose optimized helper if users inspect __all__
if "load_stitch_haplotypes_xarray" not in __all__:
    __all__.append("load_stitch_haplotypes_xarray")


def _precompute_trait_stats(Y, mask):
    """Precompute trait-only constants reused across all variant blocks."""
    Y = np.asarray(Y, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    ysq = Y * Y
    return {
        "yty0": np.einsum("nt,nt->t", ysq, mask, optimize=True),
        "sumy0": np.einsum("nt,nt->t", Y, mask, optimize=True),
        "nobs0": mask.sum(axis=0),
    }


def _numpy_regression_core(design, Y, mask, *, dof="correct", trait_stats=None):
    design = np.asarray(design, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    M = design.shape[1]
    p = design.shape[2]
    XtY = np.einsum("nmp,nt->mtp", design, Y, optimize=True)
    XtX = np.einsum("nmp,nt,nmq->mtpq", design, mask, design, optimize=True)

    if trait_stats is None:
        ysq = Y * Y
        yty0 = np.einsum("nt,nt->t", ysq, mask, optimize=True)
        sumy0 = np.einsum("nt,nt->t", Y, mask, optimize=True)
        nobs0 = mask.sum(axis=0)
    else:
        yty0 = np.asarray(trait_stats["yty0"], dtype=np.float64)
        sumy0 = np.asarray(trait_stats["sumy0"], dtype=np.float64)
        nobs0 = np.asarray(trait_stats["nobs0"], dtype=np.float64)

    yty = np.broadcast_to(yty0[None, :], (M, mask.shape[1])).copy()
    sumy = np.broadcast_to(sumy0[None, :], (M, mask.shape[1])).copy()
    nobs = np.broadcast_to(nobs0[None, :], (M, mask.shape[1])).astype(np.float64, copy=True)

    xtx_inv = np.linalg.pinv(XtX)
    beta_full = np.matmul(xtx_inv, XtY[..., None])[..., 0]
    sse = yty - np.einsum("mtp,mtp->mt", beta_full, XtY, optimize=True)
    sse = np.clip(sse, 0.0, None)

    if dof == "incorrect":
        df_denom = np.broadcast_to(mask.sum(axis=0) - p, nobs.shape).astype(np.float64, copy=True)
    else:
        df_denom = nobs - p
    df_denom[df_denom <= 0] = np.nan
    sigma2 = np.divide(sse, df_denom, out=np.full_like(sse, np.nan), where=np.isfinite(df_denom) & (df_denom > 0))
    beta_var = xtx_inv * sigma2[:, :, None, None]
    beta_se_full = np.sqrt(np.clip(np.diagonal(beta_var, axis1=-2, axis2=-1), 0.0, None))

    sse0 = yty - np.divide(sumy * sumy, nobs, out=np.full_like(yty, np.nan), where=nobs > 0)
    sse0 = np.clip(sse0, 0.0, None)
    ss_model = np.clip(sse0 - sse, 0.0, None)
    return beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom, xtx_inv


def _jax_regression_core(design, Y, mask, *, dof="correct", trait_stats=None):
    _, jnp = _require_jax()
    design = jnp.asarray(design, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    M = design.shape[1]
    p = design.shape[2]
    XtY = jnp.einsum("nmp,nt->mtp", design, Y, optimize=True)
    XtX = jnp.einsum("nmp,nt,nmq->mtpq", design, mask, design, optimize=True)

    if trait_stats is None:
        ysq = Y * Y
        yty0 = jnp.einsum("nt,nt->t", ysq, mask, optimize=True)
        sumy0 = jnp.einsum("nt,nt->t", Y, mask, optimize=True)
        nobs0 = mask.sum(axis=0)
    else:
        yty0 = jnp.asarray(trait_stats["yty0"], dtype=jnp.float32)
        sumy0 = jnp.asarray(trait_stats["sumy0"], dtype=jnp.float32)
        nobs0 = jnp.asarray(trait_stats["nobs0"], dtype=jnp.float32)

    yty = jnp.broadcast_to(yty0[None, :], (M, mask.shape[1]))
    sumy = jnp.broadcast_to(sumy0[None, :], (M, mask.shape[1]))
    nobs = jnp.broadcast_to(nobs0[None, :], (M, mask.shape[1]))

    xtx_inv = jnp.linalg.pinv(XtX)
    beta_full = jnp.matmul(xtx_inv, XtY[..., None])[..., 0]
    sse = yty - jnp.einsum("mtp,mtp->mt", beta_full, XtY, optimize=True)
    sse = jnp.clip(sse, 0.0, None)

    if dof == "incorrect":
        df_denom = jnp.broadcast_to(mask.sum(axis=0) - p, nobs.shape)
    else:
        df_denom = nobs - p
    df_denom = jnp.where(df_denom <= 0, jnp.nan, df_denom)
    sigma2 = sse / df_denom
    beta_var = xtx_inv * sigma2[:, :, None, None]
    beta_se_full = jnp.sqrt(jnp.clip(jnp.diagonal(beta_var, axis1=-2, axis2=-1), 0.0, None))

    sse0 = yty - (sumy * sumy) / nobs
    sse0 = jnp.clip(sse0, 0.0, None)
    ss_model = jnp.clip(sse0 - sse, 0.0, None)
    return beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom, xtx_inv


def warmup_jax_hwas(n_samples, block_variants, n_predictors, n_traits, *, dtype=np.float32, dof="correct", trait_stats=None):
    """Compile and return the fixed-shape JAX HWAS kernel."""
    jax, jnp = _require_jax()
    if trait_stats is None:
        kernel = jax.jit(lambda X, Y, M: _jax_regression_core(X, Y, M, dof=dof, trait_stats=None))
        X = jnp.zeros((int(n_samples), int(block_variants), int(n_predictors)), dtype=dtype)
        Y = jnp.zeros((int(n_samples), int(n_traits)), dtype=dtype)
        M = jnp.ones((int(n_samples), int(n_traits)), dtype=dtype)
        out = kernel(X, Y, M)
    else:
        const = {k: jnp.asarray(v, dtype=dtype) for k, v in trait_stats.items()}
        kernel = jax.jit(lambda X, Y, M: _jax_regression_core(X, Y, M, dof=dof, trait_stats=const))
        X = jnp.zeros((int(n_samples), int(block_variants), int(n_predictors)), dtype=dtype)
        Y = jnp.zeros((int(n_samples), int(n_traits)), dtype=dtype)
        M = jnp.ones((int(n_samples), int(n_traits)), dtype=dtype)
        out = kernel(X, Y, M)
    out[0].block_until_ready()
    return kernel


def _true_wald_from_beta(beta_full, xtx_inv, sigma2):
    """Coefficient-covariance Wald statistic for non-intercept haplotype terms."""
    if beta_full.shape[2] <= 1:
        return np.full(beta_full.shape[:2], np.nan, dtype=np.float64)
    bh = beta_full[:, :, 1:]
    cov = xtx_inv[:, :, 1:, 1:] * sigma2[:, :, None, None]
    cov_inv = np.linalg.pinv(cov)
    return np.einsum("mth,mthk,mtk->mt", bh, cov_inv, bh, optimize=True)


def regression_hwas_ols(
    design,
    straits,
    traits_mask=None,
    dof="correct",
    stat="ttest",
    sided="two-sided",
    regression_type=None,
    backend="numpy",
    warmup_jax=False,
    *,
    jax_kernel=None,
    trait_stats=None,
    lod_pvalue="f",
    compute_wald=False,
):
    """Vectorized per-locus multi-haplotype regression.

    Optimized changes relative to the prototype:
    - `trait_stats` reuses trait-only constants across blocks.
    - `jax_kernel` reuses a precompiled fixed-shape JAX executable.
    - HK/LOD p-values default to the equivalent full-vs-null F-test rather than
      the invalid monotone transform 10**(-LOD).
    - true coefficient-covariance Wald omnibus statistics are computed only when
      requested or when regression_type='wald'.
    """
    if sided not in {"two-sided", "one-sided"}:
        raise ValueError("sided must be 'two-sided' or 'one-sided'")
    regression_type = stat if regression_type is None else regression_type
    if regression_type not in {"ttest", "f", "wald", "hk", "lod"}:
        raise ValueError("regression_type must be one of ['ttest','f','wald','hk','lod']")
    if lod_pvalue not in {"f", "nan", "lod_transform"}:
        raise ValueError("lod_pvalue must be one of {'f','nan','lod_transform'}")

    if traits_mask is None:
        traits_mask = np.isfinite(np.asarray(straits, dtype=np.float64)).astype(np.float32)
    Y = np.nan_to_num(np.asarray(straits, dtype=np.float32), nan=0.0)
    mask = np.asarray(traits_mask, dtype=np.float32)
    coeff_test = "wald" if regression_type == "wald" else "ttest"
    locus_test = {"ttest": "f", "f": "f", "wald": "wald", "hk": "lod", "lod": "lod"}[regression_type]

    if backend == "jax":
        jax, jnp = _require_jax()
        if jax_kernel is None:
            if warmup_jax:
                jax_kernel = warmup_jax_hwas(design.shape[0], design.shape[1], design.shape[2], Y.shape[1], dtype=np.float32, dof=dof, trait_stats=trait_stats)
            else:
                if trait_stats is None:
                    jax_kernel = jax.jit(lambda X, Yv, Mv: _jax_regression_core(X, Yv, Mv, dof=dof, trait_stats=None))
                else:
                    const = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in trait_stats.items()}
                    jax_kernel = jax.jit(lambda X, Yv, Mv: _jax_regression_core(X, Yv, Mv, dof=dof, trait_stats=const))
        beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom, xtx_inv = jax_kernel(jnp.asarray(design), jnp.asarray(Y), jnp.asarray(mask))
        beta_full = np.asarray(beta_full)
        beta_se_full = np.asarray(beta_se_full)
        sse = np.asarray(sse)
        sse0 = np.asarray(sse0)
        ss_model = np.asarray(ss_model)
        nobs = np.asarray(nobs)
        df_denom = np.asarray(df_denom)
        xtx_inv = np.asarray(xtx_inv)
    else:
        beta_full, beta_se_full, sse, sse0, ss_model, nobs, df_denom, xtx_inv = _numpy_regression_core(design, Y, mask, dof=dof, trait_stats=trait_stats)

    p = int(beta_full.shape[2])
    df_num = p - 1
    sigma2 = np.divide(sse, df_denom, out=np.full_like(sse, np.nan), where=np.isfinite(df_denom) & (df_denom > 0))

    omnibus_f_stat = np.full_like(sse, np.nan)
    omnibus_f_p = np.full_like(sse, np.nan)
    omnibus_wald_stat = np.full_like(sse, np.nan)
    omnibus_wald_p = np.full_like(sse, np.nan)
    omnibus_lod = np.full_like(sse, np.nan)
    if df_num > 0:
        valid_f = (ss_model >= 0) & np.isfinite(df_denom) & (df_denom > 0) & np.isfinite(sse) & (sse >= 0)
        omnibus_f_stat = np.divide(ss_model / df_num, sse / df_denom, out=np.full_like(sse, np.nan), where=valid_f)
        omnibus_f_p = scipyf.sf(omnibus_f_stat, df_num, df_denom)

        if compute_wald or regression_type == "wald":
            valid_w = np.isfinite(sigma2) & (sigma2 > 0)
            omnibus_wald_stat = _true_wald_from_beta(beta_full, xtx_inv, sigma2)
            omnibus_wald_p = np.where(valid_w, chi2.sf(omnibus_wald_stat, df=df_num), np.nan)

        valid_lod = np.isfinite(sse0) & (sse0 > 0) & np.isfinite(sse) & (sse > 0)
        ratio = np.divide(sse0, sse, out=np.full_like(sse, np.nan), where=valid_lod)
        omnibus_lod = 0.5 * nobs * np.log10(ratio)

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
        stats = zstats * zstats

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
        if lod_pvalue == "f":
            omnibus_p = omnibus_f_p
        elif lod_pvalue == "nan":
            omnibus_p = np.full_like(omnibus_lod, np.nan)
        else:
            omnibus_p = np.power(10.0, -omnibus_lod, where=np.isfinite(omnibus_lod), out=np.full_like(omnibus_lod, np.nan))
        omnibus_stat_name = "LOD"

    with np.errstate(divide="ignore", invalid="ignore"):
        neglog_p = -np.log10(p_values)
        omnibus_neglog_p = -np.log10(omnibus_p)
        omnibus_f_neglog_p = -np.log10(omnibus_f_p)
        omnibus_wald_neglog_p = -np.log10(omnibus_wald_p)

    return {
        "beta": np.transpose(beta, (0, 2, 1)),
        "beta_se": np.transpose(beta_se, (0, 2, 1)),
        "stat": np.transpose(stats, (0, 2, 1)),
        "neglog_p": np.transpose(neglog_p, (0, 2, 1)),
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


def _hwas_block_numpy(
    block_arr,
    trait_values,
    trait_mask,
    *,
    hap_center=False,
    hap_scale=False,
    reference="last",
    regression_type=None,
    stat="ttest",
    sided="two-sided",
    dof="correct",
    backend="numpy",
    warmup_jax=False,
    jax_kernel=None,
    trait_stats=None,
    lod_pvalue="f",
    compute_wald=False,
):
    design, kept_haplotypes, ref_idx = prepare_haplotype_design(
        xr.DataArray(block_arr, dims=("iid", "snp", "haplotype"), coords={"haplotype": np.arange(block_arr.shape[2])}),
        center=hap_center,
        scale=hap_scale,
        reference=reference,
        add_intercept=True,
        backend="numpy" if backend != "jax" else "jax",
    )
    res = regression_hwas_ols(
        design,
        trait_values,
        traits_mask=trait_mask,
        dof=dof,
        stat=stat,
        sided=sided,
        regression_type=regression_type,
        backend=backend,
        warmup_jax=warmup_jax,
        jax_kernel=jax_kernel,
        trait_stats=trait_stats,
        lod_pvalue=lod_pvalue,
        compute_wald=compute_wald,
    )
    return res, kept_haplotypes, ref_idx


def _hwas_result_to_xarray(res, *, snp_names, chrom, pos, ref, alt, kept_haplotypes, trait_names, attrs=None):
    attrs = {} if attrs is None else dict(attrs)
    return xr.Dataset(
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
            "regression_type": res["regression_type"],
            "coef_test": res["coef_test"],
            "locus_test": res["locus_test"],
            "omnibus_stat_name": res["omnibus_stat_name"],
            **attrs,
        },
    )


def _hwas_dataset_to_pandas(ds):
    founder = ds[["beta", "beta_se", "stat", "neglog_p"]].to_dataframe().reset_index()
    omni = ds[[
        "dof", "omnibus_stat", "omnibus_neglog_p",
        "omnibus_df_num", "omnibus_df_denom",
        "omnibus_f_stat", "omnibus_f_neglog_p",
        "omnibus_wald_stat", "omnibus_wald_neglog_p", "omnibus_lod",
    ]].to_dataframe().reset_index()
    out = founder.merge(omni, on=["snp", "trait"], how="left")
    out["regression_type"] = ds.attrs.get("regression_type")
    out["coef_test"] = ds.attrs.get("coef_test")
    out["locus_test"] = ds.attrs.get("locus_test")
    return out


def _write_hwas_block(ds, output_path, output_prefix, block_index, output_format="parquet"):
    os.makedirs(output_path, exist_ok=True)
    output_format = str(output_format).lower()
    stem = os.path.join(output_path, f"{output_prefix}.block{int(block_index):06d}")
    if output_format == "parquet":
        df = _hwas_dataset_to_pandas(ds)
        df.to_parquet(f"{stem}.parquet", compression="gzip", engine="pyarrow", use_dictionary=True)
        return f"{stem}.parquet"
    if output_format == "csv":
        df = _hwas_dataset_to_pandas(ds)
        df.to_csv(f"{stem}.csv.gz", index=False, compression="gzip")
        return f"{stem}.csv.gz"
    if output_format == "pickle":
        ds.to_pickle(f"{stem}.pkl")
        return f"{stem}.pkl"
    if output_format == "zarr":
        ds.to_zarr(f"{stem}.zarr", mode="w")
        return f"{stem}.zarr"
    raise ValueError("output_format must be one of {'parquet','csv','pickle','zarr'}")


def _materialize_dask_blocks(hda, block_slices, precision=np.float32, scheduler=None):
    arrays = [hda.isel(snp=slice(s, e)).data for s, e in block_slices]
    if not arrays:
        return []
    if any(_is_dask_array(a) for a in arrays):
        if scheduler is None:
            mats = dask.compute(*arrays)
        else:
            mats = dask.compute(*arrays, scheduler=scheduler)
        return [np.asarray(a, dtype=precision) for a in mats]
    return [np.asarray(_to_numpy(a), dtype=precision) for a in arrays]


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
    backend="auto",
    block_variants=None,
    warmup_jax=True,
    pad_last_chunk=True,
    field=None,
    *,
    dask_prefetch=1,
    dask_scheduler=None,
    output_path=None,
    output_prefix="hwas",
    output_format="parquet",
    return_table=True,
    lod_pvalue="f",
    compute_wald=False,
):
    """Haplotype-based association scan with lazy block materialization.

    Dask is used only to schedule/stream BCF blocks. NumPy or a reusable JAX
    kernel performs the dense per-block regression. If `output_path` is given,
    each result block is written immediately; set `return_table=False` to avoid
    accumulating all blocks in memory.
    """
    traitdf = traits.copy() if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
    if center:
        traitdf = _center_traits_with_mask(traitdf)
    trait_values = np.nan_to_num(traitdf.to_numpy(dtype=precision), nan=0.0)
    trait_mask = np.isfinite(traits.to_numpy(dtype=float) if isinstance(traits, pd.DataFrame) else np.asarray(traits, dtype=float)).astype(np.float32)
    trait_names = traitdf.columns.astype(str).tolist()

    hda = _unwrap_haplotype_array(haplotypes, field=field)
    if isinstance(traits, pd.DataFrame) and "iid" in hda.coords:
        iid_set = set(map(str, hda["iid"].to_numpy()))
        common = [x for x in traitdf.index.astype(str) if x in iid_set]
        hda = hda.sel(iid=common)
        traitdf = traitdf.loc[common]
        trait_values = np.nan_to_num(traitdf.to_numpy(dtype=precision), nan=0.0)
        trait_mask = np.isfinite(traitdf.to_numpy(dtype=float)).astype(np.float32)
        trait_names = traitdf.columns.astype(str).tolist()

    backend = _infer_backend(hda, backend=backend)
    source_is_dask = _is_dask_array(hda.data)
    if block_variants is None:
        if source_is_dask:
            block_variants = int(hda.data.chunks[1][0])
        else:
            block_variants = int(hda.sizes["snp"])
    block_variants = max(1, int(block_variants))
    dask_prefetch = max(1, int(dask_prefetch))

    hap_names = list(map(str, hda["haplotype"].to_numpy()))
    ref_idx = _resolve_reference_haplotype(hap_names, reference=reference)
    kept_example = hap_names if ref_idx is None else [h for i, h in enumerate(hap_names) if i != ref_idx]
    p_fixed = len(kept_example) + 1
    trait_stats = _precompute_trait_stats(trait_values, trait_mask)

    jax_kernel = None
    if backend == "jax" and warmup_jax:
        jax_kernel = warmup_jax_hwas(
            int(hda.sizes["iid"]),
            block_variants,
            p_fixed,
            len(trait_names),
            dtype=precision,
            dof=dof,
            trait_stats=trait_stats,
        )

    block_results = []
    written_files = []
    snp_names_all = []
    chrom_all = []
    pos_all = []
    ref_all = []
    alt_all = []
    kept_haplotypes = None

    n_snps = int(hda.sizes["snp"])
    block_slices = [(s, min(s + block_variants, n_snps)) for s in range(0, n_snps, block_variants)]
    if not block_slices:
        return pd.DataFrame() if "pandas" in dtype else xr.Dataset()

    for batch_start in range(0, len(block_slices), dask_prefetch):
        batch_slices = block_slices[batch_start: batch_start + dask_prefetch]
        if source_is_dask:
            batch_arrays = _materialize_dask_blocks(hda, batch_slices, precision=precision, scheduler=dask_scheduler)
        else:
            batch_arrays = [np.asarray(_to_numpy(hda.isel(snp=slice(s, e))), dtype=precision) for s, e in batch_slices]

        for local_i, ((start, stop), block_arr) in enumerate(zip(batch_slices, batch_arrays)):
            block = hda.isel(snp=slice(start, stop))
            true_len = block_arr.shape[1]
            if backend == "jax" and pad_last_chunk and true_len < block_variants:
                pad = np.zeros((block_arr.shape[0], block_variants - true_len, block_arr.shape[2]), dtype=precision)
                block_arr = np.concatenate([block_arr, pad], axis=1)

            res, kept_haplotypes, _ = _hwas_block_numpy(
                block_arr,
                trait_values,
                trait_mask,
                hap_center=hap_center,
                hap_scale=hap_scale,
                reference=reference,
                regression_type=regression_type,
                stat=stat,
                sided=sided,
                dof=dof,
                backend="jax" if backend == "jax" else "numpy",
                warmup_jax=False,
                jax_kernel=jax_kernel,
                trait_stats=trait_stats,
                lod_pvalue=lod_pvalue,
                compute_wald=compute_wald,
            )
            if true_len < block_variants:
                for key in [
                    "beta", "beta_se", "stat", "neglog_p",
                    "dof", "omnibus_stat", "omnibus_neglog_p",
                    "omnibus_df_num", "omnibus_df_denom",
                    "omnibus_f_stat", "omnibus_f_neglog_p",
                    "omnibus_wald_stat", "omnibus_wald_neglog_p", "omnibus_lod",
                ]:
                    res[key] = res[key][:true_len]

            block_snp = block["snp"].to_numpy().tolist()
            block_chrom = block["chrom"].to_numpy().tolist()
            block_pos = block["pos"].to_numpy().tolist()
            block_ref = block["ref"].to_numpy().tolist()
            block_alt = block["alt"].to_numpy().tolist()
            attrs = {
                "reference_haplotype": reference,
                "haplotype_centered": bool(hap_center),
                "haplotype_scaled": bool(hap_scale),
                "backend": backend,
                "block_variants": int(block_variants),
                "lod_pvalue": lod_pvalue,
            }
            ds_block = _hwas_result_to_xarray(
                res,
                snp_names=block_snp,
                chrom=block_chrom,
                pos=block_pos,
                ref=block_ref,
                alt=block_alt,
                kept_haplotypes=kept_haplotypes,
                trait_names=trait_names,
                attrs=attrs,
            )

            if output_path is not None:
                written_files.append(_write_hwas_block(ds_block, output_path, output_prefix, batch_start + local_i, output_format=output_format))

            if return_table:
                block_results.append(res)
                snp_names_all.extend(block_snp)
                chrom_all.extend(block_chrom)
                pos_all.extend(block_pos)
                ref_all.extend(block_ref)
                alt_all.extend(block_alt)

            del block_arr, res, ds_block
        gc.collect()

    if not return_table:
        return written_files if output_path is not None else None
    if not block_results:
        return pd.DataFrame() if "pandas" in dtype else xr.Dataset()

    def _cat(key):
        return np.concatenate([b[key] for b in block_results], axis=0)

    res = {key: _cat(key) for key in [
        "beta", "beta_se", "stat", "neglog_p",
        "dof", "omnibus_stat", "omnibus_neglog_p",
        "omnibus_df_num", "omnibus_df_denom",
        "omnibus_f_stat", "omnibus_f_neglog_p",
        "omnibus_wald_stat", "omnibus_wald_neglog_p", "omnibus_lod",
    ]}
    res.update({k: block_results[0][k] for k in ["regression_type", "coef_test", "locus_test", "omnibus_stat_name"]})

    ds = _hwas_result_to_xarray(
        res,
        snp_names=snp_names_all,
        chrom=chrom_all,
        pos=pos_all,
        ref=ref_all,
        alt=alt_all,
        kept_haplotypes=kept_haplotypes,
        trait_names=trait_names,
        attrs={
            "reference_haplotype": reference,
            "haplotype_centered": bool(hap_center),
            "haplotype_scaled": bool(hap_scale),
            "backend": backend,
            "block_variants": int(block_variants),
            "lod_pvalue": lod_pvalue,
            "written_files": tuple(written_files),
        },
    )

    if dtype == "tuple":
        return res
    if dtype == "xarray_dataset":
        return ds
    if dtype == "xarray":
        return ds.to_array("metric")
    if dtype in ["pandas_highmem", "pandas"]:
        return _hwas_dataset_to_pandas(ds)
    raise ValueError("dtype must be one of ['tuple','xarray','xarray_dataset','pandas','pandas_highmem']")


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
    backend="auto",
    jax_warmup=True,
    jax_pad_last_chunk=True,
    *,
    parse_mode="hwas",
    format_fields="haplotype",
    include_info=False,
    persist_haplotype=False,
    dask_prefetch=1,
    dask_scheduler=None,
    hwas_output_format="parquet",
    lod_pvalue="f",
    compute_wald=False,
):
    """Multivariate HWAS mirroring mvGWAS with optimized lazy BCF streaming.

    For production, the default BCF loader parses only the selected haplotype
    FORMAT field (`parse_mode='hwas'`, `format_fields='haplotype'`). Use
    parse_mode='full' or format_fields='all' for QC/exploration.
    """
    current_mem = lambda: f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f}Gb"

    if isinstance(grms_folder, str):
        grms_folder = glob(f"{grms_folder}/*.grm.bin")
    in_memory_kinship = isinstance(grms_folder, dict)
    if regression_mode != "ols":
        raise ValueError("mvHWAS currently supports regression_mode='ols' only")

    read_hap = load_stitch_bcf_xarray(
        haplotypes,
        chunk_variants=chunk_variants,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        backend="dask",
        parse_mode=parse_mode,
        format_fields=format_fields,
        include_info=include_info,
    ) if isinstance(haplotypes, str) else haplotypes

    if isinstance(read_hap, xr.DataArray):
        read_hap = read_hap.to_dataset(name=read_hap.name or "haplotypes")

    hap_field = _default_haplotype_field_from_dataset(read_hap, None)
    if persist_haplotype and _is_dask_array(read_hap[hap_field].data):
        persisted = read_hap[hap_field].data.persist()
        read_hap = read_hap.copy()
        read_hap[hap_field] = (read_hap[hap_field].dims, persisted, dict(read_hap[hap_field].attrs))

    chrunique = [str(x) for x in pd.Index(read_hap["chrom"].to_numpy()).astype(str).unique().tolist()]
    if chrset is not None:
        chrset = {str(x) for x in chrset}
        chrunique = [x for x in chrunique if x in chrset]

    npplink = None if in_memory_kinship else _require_npplink()
    gdv2 = _require_gdv2()
    if in_memory_kinship:
        kinship_map = {str(k): (v.to_pandas() if isinstance(v, xr.DataArray) else pd.DataFrame(np.asarray(v), index=traitdf.index, columns=traitdf.index)) for k, v in grms_folder.items()}
        grm_index = pd.Index(list(kinship_map.keys()), dtype="object")
        if not grm_index.isin(chrunique).any():
            raise ValueError("Provided kinship dict does not contain chromosome keys matching the haplotype data.")
    else:
        grms_folder = pd.DataFrame(grms_folder, columns=["path"])
        grms_folder.index = grms_folder["path"].str.extract(r"([\d\w_]+)chrGRM.", expand=False)
        grms_folder = grms_folder.sort_index(
            key=lambda idx: idx.str.lower().map(
                {str(i): int(i) for i in range(1000)} | {i: int(i) for i in range(1000)} | {"all": -1000, "x": 1001, "y": 1002, "mt": 1003, "m": 1003}
            )
        )
        grms_folder = grms_folder[~grms_folder.index.isna()]
        allGRM = npplink.read_grm(grms_folder.loc["All", "path"].replace(".bin", ""))
        grms_folder["in_chrunique"] = grms_folder.index.isin(chrunique)
        grms_folder["isnum"] = grms_folder.index.str.isnumeric()
        max_grm_chr = grms_folder.query("isnum").index.astype(int).max()
        if grms_folder.in_chrunique.eq(False).sum() > 1:
            grms_folder = grms_folder.rename({
                "x": str(max_grm_chr + 1), "y": str(max_grm_chr + 2), "mt": str(max_grm_chr + 4),
                "X": str(max_grm_chr + 1), "Y": str(max_grm_chr + 2), "MT": str(max_grm_chr + 4),
            })
            grms_folder["in_chrunique"] = grms_folder.index.isin(chrunique)
            if grms_folder.in_chrunique.eq(False).sum() > 1:
                raise ValueError("cannot match chromosomes in grm folder and haplotype file")

    chrom_sizes = pd.DataFrame({"chrom": read_hap["chrom"].to_numpy().astype(str), "pos": read_hap["pos"].to_numpy()}).groupby("chrom").pos.max()
    sumstats = []
    if save:
        os.makedirs(save_path, exist_ok=True)

    grm_iter = list(grms_folder.drop(["All"]).iterrows()) if not in_memory_kinship else [(c, None) for c in chrunique if str(c) in kinship_map]
    for c, row in (pbar := tqdm(grm_iter)):
        if str(c) not in chrunique:
            continue
        pbar.set_description(f"HWAS-Chr{c}-reading GRM")
        if in_memory_kinship:
            subgrm = kinship_map[str(c)].copy()
            c_grm = None
        else:
            c_grm = npplink.read_grm(row.path.replace(".bin", ""))
            subgrm = allGRM["grm"].to_pandas() if not row.isnum else ((allGRM["grm"] * allGRM["w"] - c_grm["grm"] * c_grm["w"]) / (allGRM["w"] - c_grm["w"])).to_pandas()

        pbar.set_description(f"HWAS-Chr{c}-estimating var/covar")
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
                pbar.set_description(f"HWAS-Chr{c}-scan-MEM:{current_mem()}")
                hblock = stitch2array(read_hap, c=c, rfids=traits.index.astype(str).tolist(), field=hap_field, compute=False)
                suffix = ""
            else:
                pbar.set_description(f"HWAS-Chr{c}-scan[{start / 1e6:.0f}-{stop / 1e6:.0f}]Mb-MEM:{current_mem()}")
                hblock = stitch2array(read_hap, c=c, rfids=traits.index.astype(str).tolist(), pos_start=start, pos_end=stop, field=hap_field, compute=False)
                suffix = f"_{int(start)}_{int(stop)}"
            if int(hblock.sizes.get("snp", 0)) == 0:
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
                backend=backend,
                block_variants=chunk_variants,
                warmup_jax=jax_warmup,
                pad_last_chunk=jax_pad_last_chunk,
                dask_prefetch=dask_prefetch,
                dask_scheduler=dask_scheduler,
                output_path=save_path if save else None,
                output_prefix=f"hwas{c}{suffix}",
                output_format=hwas_output_format,
                return_table=return_table,
                lod_pvalue=lod_pvalue,
                compute_wald=compute_wald,
            )
            if return_table and block is not None:
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

# =============================================================================
# Fast FORMAT reader engine overrides (single-file patch)
# =============================================================================
# These definitions intentionally override the optimized loader definitions above.
# They add reader_engine={'auto','cyvcf2','htslib','bcftools','pysam'} and fix the
# default STITCH HD behavior so an 8-component founder/haplotype dosage vector is
# read as 8 components, not collapsed to 4.

import subprocess as _subprocess
import tempfile as _tempfile
import shutil as _shutil


def _collapse_phased_flag(collapse_phased="auto", *, field: str | None = None) -> bool:
    """Normalize collapse_phased.

    The default is intentionally False. STITCH HD/HDS/AP-style founder dosage
    vectors are already additive founder/haplotype vectors; an 8-founder HD field
    must stay length 8. Set collapse_phased=True only for fields explicitly known
    to be encoded as two phased vectors of length H each.
    """
    if collapse_phased is None or collapse_phased == "auto":
        return False
    return bool(collapse_phased)


def _flatten_value(value: Any) -> list[Any]:
    """Flatten a pysam/cyvcf2 scalar/vector value and preserve missing elements."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value.decode() if isinstance(value, bytes) else value]
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.reshape(-1).tolist()


def _is_missing_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, bytes):
        x = x.decode()
    if isinstance(x, str):
        return x in {"", ".", "NA", "NaN", "nan", "None", "none"}
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _safe_cast_vector(vals, *, dtype, fill):
    """Cast VCF scalar/vector values while replacing None/'.' with dtype fill."""
    dt = np.dtype(dtype)
    out_vals = []
    for x in vals:
        if _is_missing_scalar(x):
            out_vals.append(fill)
        else:
            if isinstance(x, bytes):
                x = x.decode()
            out_vals.append(x)
    if dt == np.dtype(object):
        return np.asarray([None if _is_missing_scalar(x) else str(x) for x in vals], dtype=object)
    if np.issubdtype(dt, np.bool_):
        return np.asarray(out_vals, dtype=dt)
    if np.issubdtype(dt, np.integer):
        # Float-looking integer fields can occur in VCF text paths; cast safely.
        return np.asarray([fill if _is_missing_scalar(x) else int(float(x)) for x in out_vals], dtype=dt)
    return np.asarray(out_vals, dtype=dt)


def _infer_components(value: Any, collapse_phased: bool = False) -> int:
    vals = _flatten_value(value)
    n = len(vals)
    if n == 0:
        return 0
    if _collapse_phased_flag(collapse_phased) and n > 2 and (n % 2 == 0):
        return n // 2
    return n


def _convert_value(value: Any, *, ncomp: int, dtype, collapse_phased: bool = False):
    """Convert one VCF scalar/vector value to a fixed-size array/scalar.

    Missing numeric elements are converted to the dtype-specific fill value
    (NaN for floats, -1 for integers). This fixes crashes when vector FORMAT
    fields contain partial missing entries such as (0.1, None, 0.2, ...).
    """
    fill = _missing_fill(dtype)
    dt = np.dtype(dtype)
    vals = _flatten_value(value)

    if ncomp <= 1:
        if not vals:
            return fill
        if dt == np.dtype(object):
            return None if _is_missing_scalar(vals[0]) else str(vals[0])
        return _safe_cast_vector(vals[:1], dtype=dt, fill=fill)[0]

    if dt == np.dtype(object):
        out = np.empty((int(ncomp),), dtype=object)
        out[:] = None
    else:
        out = np.full((int(ncomp),), fill, dtype=dt)
    if not vals:
        return out

    if _collapse_phased_flag(collapse_phased) and len(vals) == 2 * int(ncomp) and len(vals) > 2:
        left = _safe_cast_vector(vals[:ncomp], dtype=np.float32, fill=np.nan)
        right = _safe_cast_vector(vals[ncomp:], dtype=np.float32, fill=np.nan)
        vals = (left + right).tolist()

    vals = vals[: int(ncomp)]
    out[: len(vals)] = _safe_cast_vector(vals, dtype=dt, fill=fill)
    return out


def _require_cyvcf2():
    try:
        from cyvcf2 import VCF
    except ImportError as exc:
        raise ImportError(
            "reader_engine='cyvcf2' requires cyvcf2. Install with `conda install -c bioconda cyvcf2`."
        ) from exc
    return VCF


def _require_bcftools():
    exe = _shutil.which("bcftools")
    if exe is None:
        raise ImportError(
            "reader_engine='bcftools'/'htslib' requires the bcftools executable on PATH. "
            "Install with `conda install -c bioconda bcftools`."
        )
    return exe


def _resolve_reader_engine(reader_engine="auto", *, n_fields=None, require_numeric=False):
    key = "auto" if reader_engine is None else str(reader_engine).lower()
    if key in {"htslib", "bcftools-query", "bcftools_query"}:
        key = "bcftools"
    if key not in {"auto", "cyvcf2", "bcftools", "pysam"}:
        raise ValueError("reader_engine must be one of {'auto','cyvcf2','bcftools','htslib','pysam'}")
    if key != "auto":
        return key

    # cyvcf2 is the preferred fast path for vector FORMAT fields. It is much
    # faster than looping over rec.samples with pysam.
    try:
        _require_cyvcf2()
        return "cyvcf2"
    except Exception:
        pass

    # bcftools query is a useful HTSlib-backed fallback for one numeric field.
    if (n_fields is None or int(n_fields) == 1) and require_numeric:
        try:
            _require_bcftools()
            return "bcftools"
        except Exception:
            pass

    return "pysam"


def _variant_key_common(chrom, pos, ref, alt, vid=None):
    alt_s = _normalize_alt_tuple(alt)
    vid_s = "." if vid in [None, ""] else str(vid)
    return (str(chrom), int(pos), str(ref), str(alt_s), vid_s)


def _variant_key_from_cyvcf2(v):
    return _variant_key_common(v.CHROM, int(v.POS), v.REF, v.ALT, getattr(v, "ID", None))


def _coerce_numeric_array(arr, *, dtype, fill):
    """Convert cyvcf2/bcftools FORMAT output to dtype and normalize missing."""
    dt = np.dtype(dtype)
    if dt == np.dtype(object):
        return np.asarray(arr, dtype=object)
    a = np.asarray(arr)
    if a.dtype.kind in "OSU":
        flat = _safe_cast_vector(a.reshape(-1).tolist(), dtype=dt, fill=fill)
        return flat.reshape(a.shape)
    a = a.astype(dt, copy=False)
    if np.issubdtype(dt, np.floating):
        # cyvcf2 may use large negative sentinels for missing values.
        a = np.where(a < -1e20, np.nan, a)
    elif np.issubdtype(dt, np.integer):
        a = np.where(a < -1000000000, fill, a)
    return a


def _reshape_format_array(arr, *, n_samples, ncomp, dtype, fill, collapse_phased=False):
    """Return FORMAT array with shape (sample,) or (sample, component)."""
    dt = np.dtype(dtype)
    ncomp = int(ncomp)
    if arr is None:
        shape = (n_samples,) if ncomp == 1 else (n_samples, ncomp)
        return np.full(shape, fill, dtype=dt if dt != np.dtype(object) else object)

    a = np.asarray(arr, dtype=object if dt == np.dtype(object) else None)
    if a.ndim == 0:
        a = np.repeat(a.reshape(1), n_samples, axis=0)
    if a.ndim == 1:
        if a.shape[0] == n_samples:
            a = a.reshape(n_samples, 1)
        else:
            # One-sample vector or malformed scalar-vector; flatten as components.
            a = np.resize(a.reshape(1, -1), (n_samples, a.size))
    else:
        a = a.reshape(a.shape[0], -1)

    if a.shape[0] != n_samples:
        # Keep the function robust rather than silently reordering samples.
        raise ValueError(f"FORMAT array sample dimension mismatch: observed {a.shape[0]}, expected {n_samples}")

    if _collapse_phased_flag(collapse_phased) and a.shape[1] == 2 * ncomp and a.shape[1] > 2:
        left = _coerce_numeric_array(a[:, :ncomp], dtype=np.float32, fill=np.nan)
        right = _coerce_numeric_array(a[:, ncomp:], dtype=np.float32, fill=np.nan)
        a = left + right

    if ncomp == 1:
        col = a[:, 0] if a.shape[1] else np.full((n_samples,), fill, dtype=dt)
        return _coerce_numeric_array(col, dtype=dt, fill=fill)

    out = np.full((n_samples, ncomp), fill, dtype=dt if dt != np.dtype(object) else object)
    m = min(ncomp, a.shape[1])
    if m:
        out[:, :m] = _coerce_numeric_array(a[:, :m], dtype=dt, fill=fill)
    return out


def _make_chunk_output(sample_ids, variant_chunk, format_specs):
    out = {}
    n_samples = len(sample_ids)
    n_vars = len(variant_chunk)
    for field, spec in format_specs.items():
        ncomp = int(spec["ncomp"])
        shape = (n_samples, n_vars) if ncomp == 1 else (n_samples, n_vars, ncomp)
        fill = _missing_fill(spec["dtype"])
        if np.dtype(spec["dtype"]) == np.dtype(object):
            arr = np.empty(shape, dtype=object)
            arr[:] = fill
        else:
            arr = np.full(shape, fill, dtype=spec["dtype"])
        out[field] = arr
    return out


def _wanted_variant_map(variant_chunk: pd.DataFrame):
    wanted = {}
    for j, row in enumerate(variant_chunk.reset_index(drop=True).itertuples(index=False)):
        chrom = str(row.chrom)
        pos = int(row.pos)
        ref = str(row.ref)
        alt = str(row.alt)
        vid = "." if pd.isna(row.id) else str(row.id)
        # Primary key with ID plus an ID-agnostic key. The latter handles readers
        # that do not expose '.' exactly the same way as pysam/cyvcf2.
        wanted[(chrom, pos, ref, alt, vid)] = j
        wanted[(chrom, pos, ref, alt, ".")] = j
        wanted[(chrom, pos, ref, alt, "*")] = j
    return wanted


def _lookup_variant_index(wanted, chrom, pos, ref, alt, vid=None):
    alt_s = _normalize_alt_tuple(alt)
    vid_s = "." if vid in [None, ""] else str(vid)
    return wanted.get((str(chrom), int(pos), str(ref), str(alt_s), vid_s),
           wanted.get((str(chrom), int(pos), str(ref), str(alt_s), "."),
           wanted.get((str(chrom), int(pos), str(ref), str(alt_s), "*"))))


def _chunk_reader_all_formats_pysam(bcf_path, variant_chunk: pd.DataFrame, sample_ids, format_specs):
    """Robust fallback FORMAT reader using pysam sample loops."""
    pysam = _require_pysam()
    variant_chunk = variant_chunk.reset_index(drop=True)
    sample_ids = list(sample_ids)
    out = _make_chunk_output(sample_ids, variant_chunk, format_specs)
    wanted = _wanted_variant_map(variant_chunk)

    def _fill_record(rec):
        j = _lookup_variant_index(wanted, rec.contig, rec.pos, rec.ref, _normalize_alt_tuple(rec.alts), rec.id)
        if j is None:
            return
        for i, sid in enumerate(sample_ids):
            sample = rec.samples[sid]
            for field, spec in format_specs.items():
                if field not in rec.format:
                    continue
                val = _convert_value(
                    sample.get(field),
                    ncomp=int(spec["ncomp"]),
                    dtype=spec["dtype"],
                    collapse_phased=bool(spec.get("collapse_phased", False)),
                )
                if int(spec["ncomp"]) == 1:
                    out[field][i, j] = val
                else:
                    out[field][i, j, :] = val

    with pysam.VariantFile(bcf_path) as vf:
        try:
            vf.subset_samples(sample_ids)
        except Exception:
            pass
        try:
            for chrom, grp in variant_chunk.groupby("chrom", sort=False):
                start = max(int(grp["pos"].min()) - 1, 0)
                stop = int(grp["pos"].max())
                for rec in vf.fetch(str(chrom), start, stop):
                    _fill_record(rec)
        except (ValueError, OSError, RuntimeError):
            # Usually unindexed input. Fall back to linear scan. Conversion errors
            # are intentionally not swallowed here.
            vf.close()
            with pysam.VariantFile(bcf_path) as vf2:
                try:
                    vf2.subset_samples(sample_ids)
                except Exception:
                    pass
                for rec in vf2:
                    _fill_record(rec)
    return out


def _chunk_reader_all_formats_cyvcf2(bcf_path, variant_chunk: pd.DataFrame, sample_ids, format_specs):
    """Fast FORMAT reader using cyvcf2.Variant.format(field)."""
    VCF = _require_cyvcf2()
    variant_chunk = variant_chunk.reset_index(drop=True)
    sample_ids = list(sample_ids)
    out = _make_chunk_output(sample_ids, variant_chunk, format_specs)
    wanted = _wanted_variant_map(variant_chunk)

    vcf = VCF(str(bcf_path))
    try:
        try:
            vcf.set_samples(sample_ids)
        except Exception:
            # If set_samples fails, continue and validate sample dimension below.
            pass

        def _fill_variant(v):
            j = _lookup_variant_index(wanted, v.CHROM, int(v.POS), v.REF, _normalize_alt_tuple(v.ALT), getattr(v, "ID", None))
            if j is None:
                return
            for field, spec in format_specs.items():
                ncomp = int(spec["ncomp"])
                fill = _missing_fill(spec["dtype"])
                try:
                    raw = v.format(field)
                except Exception:
                    # GT and arbitrary strings are less reliable through Variant.format;
                    # use pysam for these rare cases.
                    raise
                arr = _reshape_format_array(
                    raw,
                    n_samples=len(sample_ids),
                    ncomp=ncomp,
                    dtype=spec["dtype"],
                    fill=fill,
                    collapse_phased=bool(spec.get("collapse_phased", False)),
                )
                if ncomp == 1:
                    out[field][:, j] = arr
                else:
                    out[field][:, j, :] = arr

        try:
            for chrom, grp in variant_chunk.groupby("chrom", sort=False):
                start = int(grp["pos"].min())
                stop = int(grp["pos"].max())
                region = f"{chrom}:{start}-{stop}"
                for v in vcf(region):
                    _fill_variant(v)
        except Exception:
            # Unindexed input or cyvcf2 region issue. Linear iteration is still
            # cyvcf2/C-backed and usually faster than per-sample pysam parsing.
            vcf.close()
            vcf = VCF(str(bcf_path))
            try:
                vcf.set_samples(sample_ids)
            except Exception:
                pass
            for v in vcf:
                _fill_variant(v)
    finally:
        try:
            vcf.close()
        except Exception:
            pass
    return out


def _parse_bcftools_value_token(token: str):
    if token is None:
        return []
    token = str(token)
    if token in {"", "."}:
        return []
    # FORMAT vectors are usually comma-separated. Preserve phased/bar-separated
    # strings by splitting on common vector delimiters only after replacing '|'.
    token = token.replace("|", ",").replace("/", ",")
    return token.split(",")


def _chunk_reader_all_formats_bcftools(bcf_path, variant_chunk: pd.DataFrame, sample_ids, format_specs):
    """Fast HTSlib-backed reader using `bcftools query` for one numeric FORMAT field.

    This is intended for production HWAS reads of a single numeric vector field
    such as HD/HDS/AP. For multiple fields or object fields use cyvcf2/pysam.
    """
    if len(format_specs) != 1:
        raise ValueError("reader_engine='bcftools' currently supports exactly one FORMAT field per call.")
    field, spec = next(iter(format_specs.items()))
    if np.dtype(spec["dtype"]) == np.dtype(object):
        raise ValueError("reader_engine='bcftools' is only supported for numeric FORMAT fields.")

    exe = _require_bcftools()
    variant_chunk = variant_chunk.reset_index(drop=True)
    sample_ids = list(sample_ids)
    out = _make_chunk_output(sample_ids, variant_chunk, format_specs)
    wanted = _wanted_variant_map(variant_chunk)
    ncomp = int(spec["ncomp"])
    fill = _missing_fill(spec["dtype"])

    with _tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as sfh:
        sample_file = sfh.name
        for sid in sample_ids:
            sfh.write(str(sid) + "\n")
    try:
        for chrom, grp in variant_chunk.groupby("chrom", sort=False):
            start = int(grp["pos"].min())
            stop = int(grp["pos"].max())
            region = f"{chrom}:{start}-{stop}"
            # One line per variant, then one SAMPLE:VALUE token per selected sample.
            fmt = f"%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE:%{field}]\n"
            cmd = [exe, "query", "-r", region, "-S", sample_file, "-f", fmt, str(bcf_path)]
            proc = _subprocess.run(cmd, check=True, text=True, capture_output=True)
            for line in proc.stdout.splitlines():
                if not line:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                chrom_o, pos_o, ref_o, alt_o = parts[:4]
                j = _lookup_variant_index(wanted, chrom_o, int(pos_o), ref_o, alt_o, ".")
                if j is None:
                    continue
                # Fill by sample order from the -S file. The output bracket follows
                # that order; we still parse SAMPLE:VALUE to tolerate absent samples.
                sample_to_row = {str(sid): i for i, sid in enumerate(sample_ids)}
                for tok_i, token in enumerate(parts[4:]):
                    if ":" in token:
                        sid, val = token.split(":", 1)
                        row_i = sample_to_row.get(sid, tok_i)
                    else:
                        val = token
                        row_i = tok_i
                    if row_i is None or row_i >= len(sample_ids):
                        continue
                    vals = _parse_bcftools_value_token(val)
                    conv = _convert_value(vals, ncomp=ncomp, dtype=spec["dtype"], collapse_phased=bool(spec.get("collapse_phased", False)))
                    if ncomp == 1:
                        out[field][row_i, j] = conv
                    else:
                        out[field][row_i, j, :] = conv
    finally:
        try:
            os.unlink(sample_file)
        except Exception:
            pass
    return out


def _chunk_reader_all_formats(
    bcf_path,
    variant_chunk: pd.DataFrame,
    sample_ids,
    format_specs,
    reader_engine="auto",
):
    """Dispatching chunk FORMAT reader.

    `cyvcf2` and `bcftools` avoid the slow nested Python loop over samples used by
    the robust pysam fallback. `htslib` is an alias for the bcftools-query path.
    """
    n_fields = len(format_specs)
    require_numeric = all(np.dtype(spec["dtype"]) != np.dtype(object) for spec in format_specs.values())
    engine = _resolve_reader_engine(reader_engine, n_fields=n_fields, require_numeric=require_numeric)
    if engine == "cyvcf2":
        try:
            return _chunk_reader_all_formats_cyvcf2(bcf_path, variant_chunk, sample_ids, format_specs)
        except Exception:
            if str(reader_engine).lower() == "cyvcf2":
                raise
            # Auto fallback for unsupported fields / local cyvcf2 edge cases.
            if n_fields == 1 and require_numeric:
                try:
                    return _chunk_reader_all_formats_bcftools(bcf_path, variant_chunk, sample_ids, format_specs)
                except Exception:
                    pass
            return _chunk_reader_all_formats_pysam(bcf_path, variant_chunk, sample_ids, format_specs)
    if engine == "bcftools":
        return _chunk_reader_all_formats_bcftools(bcf_path, variant_chunk, sample_ids, format_specs)
    return _chunk_reader_all_formats_pysam(bcf_path, variant_chunk, sample_ids, format_specs)


def scan_stitch_bcf(
    bcf_path,
    haplotype_field="auto",
    collapse_phased="auto",
    max_variants=None,
    schema_scan_variants=128,
    include_info=True,
):
    """Scan variant metadata and FORMAT schemas in one pass.

    `collapse_phased` defaults to 'auto', which does not collapse vector fields.
    This is the correct default for STITCH HD because an 8-founder HD field should
    remain length 8. Pass collapse_phased=True only for true 2H phased vectors.
    """
    pysam = _require_pysam()
    collapse_flag = _collapse_phased_flag(collapse_phased, field=None)
    preferred = ["HD", "AP", "HP", "HAP", "HAPROB", "HAPPROB", "HDS", "DS", "GP"]
    fixed_hap_field = None if haplotype_field in [None, "auto"] else str(haplotype_field)

    records = []
    selected_hap_field = fixed_hap_field
    best_field = None
    best_score = (-1, -1, -1)
    observed_components = {}

    with pysam.VariantFile(bcf_path) as vf:
        sample_ids = list(vf.header.samples)
        probes = sample_ids[: min(4, len(sample_ids))]

        info_specs = {}
        if include_info:
            for field, meta in vf.header.info.items():
                info_specs[str(field)] = {
                    "dtype": _field_dtype(str(field), getattr(meta, "type", None)),
                    "ncomp": 1,
                    "type": getattr(meta, "type", None),
                    "number": getattr(meta, "number", None),
                }
        format_specs = {}
        for field, meta in vf.header.formats.items():
            format_specs[str(field)] = {
                "dtype": _field_dtype(str(field), getattr(meta, "type", None)),
                "ncomp": 1,
                "type": getattr(meta, "type", None),
                "number": getattr(meta, "number", None),
                "collapse_phased": False,
            }
        info_raw = {field: [] for field in info_specs}

        if fixed_hap_field is not None and fixed_hap_field not in format_specs:
            raise KeyError(f"Requested haplotype FORMAT field {fixed_hap_field!r} is not in the BCF header.")

        for i, rec in enumerate(vf):
            if max_variants is not None and i >= max_variants:
                break
            records.append(
                {
                    "chrom": str(rec.contig),
                    "pos": int(rec.pos),
                    "snp": rec.id if rec.id not in [None, "."] else f"{rec.contig}:{rec.pos}:{rec.ref}:{_normalize_alt_tuple(rec.alts)}",
                    "id": "." if rec.id is None else str(rec.id),
                    "ref": str(rec.ref),
                    "alt": _normalize_alt_tuple(rec.alts),
                    "qual": np.nan if rec.qual is None else float(rec.qual),
                    "filter": _normalize_filter_value(rec.filter.keys() if hasattr(rec.filter, "keys") else rec.filter),
                    "i": i,
                }
            )

            if include_info:
                for field in info_specs:
                    val = rec.info.get(field)
                    info_raw[field].append(val)
                    info_specs[field]["ncomp"] = max(info_specs[field]["ncomp"], max(1, _infer_components(val, collapse_phased=False)))

            if i < schema_scan_variants:
                for sid in probes:
                    s = rec.samples[sid]
                    for field, spec in format_specs.items():
                        if field not in rec.format:
                            continue
                        val = s.get(field)
                        raw_n = max(1, _infer_components(val, collapse_phased=False))
                        observed_components[field] = max(observed_components.get(field, 1), raw_n)
                        spec["ncomp"] = max(spec["ncomp"], raw_n)

                        if fixed_hap_field is None and raw_n > 1:
                            pref_rank = (len(preferred) - preferred.index(field)) if field in preferred else 0
                            score = (pref_rank, raw_n, int(raw_n > 2))
                            if score > best_score:
                                best_score = score
                                best_field = field

        if fixed_hap_field is None:
            selected_hap_field = best_field
        if selected_hap_field is None:
            raise ValueError("Could not infer the haplotype FORMAT field. Pass haplotype_field explicitly.")

        raw_hap_components = int(observed_components.get(selected_hap_field, format_specs[selected_hap_field]["ncomp"]))
        n_haplotypes = int(_infer_components([0] * raw_hap_components, collapse_phased=collapse_flag))
        format_specs[selected_hap_field]["collapse_phased"] = bool(collapse_flag)
        format_specs[selected_hap_field]["ncomp"] = max(1, n_haplotypes)

    if n_haplotypes <= 0:
        raise ValueError("Could not infer the number of haplotypes from the chosen FORMAT field.")

    variants = pd.DataFrame.from_records(records)
    samples = pd.DataFrame({"iid": sample_ids, "i": np.arange(len(sample_ids), dtype=int)})

    info_arrays = {}
    n_variants = len(variants)
    if include_info:
        for field, spec in info_specs.items():
            ncomp = max(1, int(spec["ncomp"]))
            arr = []
            for val in info_raw[field]:
                arr.append(_convert_value(val, ncomp=ncomp, dtype=spec["dtype"], collapse_phased=False))
            if ncomp == 1:
                info_arrays[field] = np.asarray(arr, dtype=spec["dtype"] if spec["dtype"] is not object else object)
            else:
                info_arrays[field] = np.asarray(arr, dtype=spec["dtype"] if spec["dtype"] is not object else object).reshape(n_variants, ncomp)

    variants.attrs["haplotype_field"] = str(selected_hap_field)
    variants.attrs["n_haplotypes"] = int(n_haplotypes)
    variants.attrs["format_specs"] = format_specs
    variants.attrs["info_specs"] = info_specs
    variants.attrs["info_arrays"] = info_arrays
    samples.attrs["haplotype_field"] = str(selected_hap_field)
    samples.attrs["n_haplotypes"] = int(n_haplotypes)
    return variants, samples


def load_stitch_bcf(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased="auto",
    max_variants=None,
    backend="dask",
    *,
    reader_engine="auto",
):
    """Load only the selected haplotype field lazily.

    Returns (variants, samples, haplotypes), where haplotypes has shape
    (sample, snp, haplotype). `reader_engine='cyvcf2'` is usually fastest.
    """
    variants, samples = scan_stitch_bcf(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
        include_info=False,
    )
    n_haplotypes = int(variants.attrs["n_haplotypes"])
    haplotype_field = variants.attrs["haplotype_field"]
    _require_dask()

    spec = dict(variants.attrs["format_specs"][haplotype_field])
    spec["dtype"] = dtype
    format_specs = {haplotype_field: spec}

    delayed_chunks = []
    for offset in range(0, len(variants), chunk_variants):
        vchunk = variants.iloc[offset: offset + chunk_variants].copy()
        nchunk = len(vchunk)
        delayed_arr = delayed(_chunk_reader_all_formats)(
            bcf_path,
            vchunk,
            samples["iid"].tolist(),
            format_specs,
            reader_engine,
        )
        delayed_field = delayed(lambda d, k: d[k])(delayed_arr, haplotype_field)
        delayed_chunks.append(da.from_delayed(delayed_field, shape=(len(samples), nchunk, n_haplotypes), dtype=dtype))
    hap = da.concatenate(delayed_chunks, axis=1) if delayed_chunks else da.empty((len(samples), 0, n_haplotypes), dtype=dtype)

    if backend == "numpy":
        hap = np.asarray(hap.compute(), dtype=dtype)
    elif backend == "jax":
        _, jnp = _require_jax()
        hap = jnp.asarray(hap.compute(), dtype=dtype)
    elif backend != "dask":
        raise ValueError("backend must be one of {'dask','numpy','jax'}")
    return variants, samples, hap


def load_stitch_bcf_xarray(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased="auto",
    max_variants=None,
    backend="dask",
    *,
    parse_mode="full",
    format_fields=None,
    include_info=True,
    reader_engine="auto",
):
    """Lazily parse a BCF into an xarray.Dataset with selectable reader engines.

    Parameters
    ----------
    reader_engine : {'auto','cyvcf2','htslib','bcftools','pysam'}, default='auto'
        cyvcf2 uses C-backed Variant.format(); htslib/bcftools uses bcftools
        query for one numeric FORMAT field; pysam is the robust fallback.
    collapse_phased : {'auto', bool}, default='auto'
        Default 'auto' does not collapse vector fields. This preserves STITCH HD
        as 8 components when the file has 8 founder/haplotype dosages.
    """
    _require_dask()
    variants, samples = scan_stitch_bcf(
        bcf_path,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
        include_info=include_info,
    )
    hap_field = variants.attrs["haplotype_field"]
    format_specs_all = variants.attrs["format_specs"]
    format_specs = _selected_format_fields(format_specs_all, format_fields, haplotype_field=hap_field, parse_mode=parse_mode)
    if hap_field in format_specs and np.dtype(format_specs[hap_field]["dtype"]).kind in "iuifc":
        format_specs[hap_field] = dict(format_specs[hap_field])
        format_specs[hap_field]["dtype"] = dtype

    info_specs = variants.attrs.get("info_specs", {})
    info_arrays = variants.attrs.get("info_arrays", {})

    chunk_readers = []
    chunk_slices = []
    for offset in range(0, len(variants), chunk_variants):
        vchunk = variants.iloc[offset: offset + chunk_variants].copy()
        chunk_slices.append((offset, offset + len(vchunk)))
        chunk_readers.append(
            delayed(_chunk_reader_all_formats)(
                bcf_path,
                vchunk,
                samples["iid"].tolist(),
                format_specs,
                reader_engine,
            )
        )

    data_vars = {}
    coords = {
        "iid": samples["iid"].to_numpy(),
        "snp": variants["snp"].to_numpy(),
        "chrom": ("snp", variants["chrom"].to_numpy()),
        "pos": ("snp", variants["pos"].to_numpy()),
        "variant_id": ("snp", variants["id"].to_numpy()),
        "ref": ("snp", variants["ref"].to_numpy()),
        "alt": ("snp", variants["alt"].to_numpy()),
        "qual": ("snp", variants["qual"].to_numpy()),
        "filter": ("snp", variants["filter"].to_numpy()),
    }

    for field, spec in format_specs.items():
        arr_chunks = []
        ndim = 2 if int(spec["ncomp"]) == 1 else 3
        for reader, (start, stop) in zip(chunk_readers, chunk_slices):
            nchunk = stop - start
            arr = delayed(lambda d, k: d[k])(reader, field)
            shape = (len(samples), nchunk) if ndim == 2 else (len(samples), nchunk, int(spec["ncomp"]))
            arr_chunks.append(da.from_delayed(arr, shape=shape, dtype=spec["dtype"]))
        full = da.concatenate(arr_chunks, axis=1) if arr_chunks else da.empty((len(samples), 0), dtype=spec["dtype"])
        if ndim == 2:
            data_vars[field] = (("iid", "snp"), full)
        else:
            comp_dim = f"{field}_component"
            coords[comp_dim] = np.arange(int(spec["ncomp"]), dtype=int)
            data_vars[field] = (("iid", "snp", comp_dim), full)

    if include_info:
        for field, spec in info_specs.items():
            arr = info_arrays[field]
            var_name = f"INFO_{field}"
            if arr.ndim == 1:
                data_vars[var_name] = (("snp",), arr)
            else:
                comp_dim = f"INFO_{field}_component"
                coords[comp_dim] = np.arange(arr.shape[1], dtype=int)
                data_vars[var_name] = (("snp", comp_dim), arr)

    collapse_flag = _collapse_phased_flag(collapse_phased, field=hap_field)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "bcf_path": str(bcf_path),
            "haplotype_field": hap_field,
            "n_haplotypes": int(variants.attrs["n_haplotypes"]),
            "collapse_phased": bool(collapse_flag),
            "parse_mode": str(parse_mode),
            "format_fields": tuple(format_specs.keys()),
            "include_info": bool(include_info),
            "reader_engine": str(reader_engine),
        },
    )
    return _maybe_materialize_dataset(ds, backend=backend)


def load_stitch_haplotypes_xarray(
    bcf_path,
    chunk_variants=1000,
    dtype=np.float32,
    haplotype_field="auto",
    collapse_phased="auto",
    max_variants=None,
    backend="dask",
    include_info=False,
    reader_engine="auto",
):
    """Production loader: lazily parse only the selected haplotype FORMAT field."""
    return load_stitch_bcf_xarray(
        bcf_path,
        chunk_variants=chunk_variants,
        dtype=dtype,
        haplotype_field=haplotype_field,
        collapse_phased=collapse_phased,
        max_variants=max_variants,
        backend=backend,
        parse_mode="hwas",
        format_fields="haplotype",
        include_info=include_info,
        reader_engine=reader_engine,
    )


if "load_stitch_haplotypes_xarray" not in __all__:
    __all__.append("load_stitch_haplotypes_xarray")
