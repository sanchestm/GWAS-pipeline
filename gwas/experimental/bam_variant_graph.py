from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

try:
    import pysam  # type: ignore
except ImportError:  # pragma: no cover
    pysam = None

try:
    import dask.array as da  # type: ignore
    from dask import delayed  # type: ignore
except ImportError:  # pragma: no cover
    da = None
    delayed = None

try:
    from lgbmi import LGBMImputer_optuna
except ImportError:  # pragma: no cover
    LGBMImputer_optuna = None


_RE_HAS_LETTER = re.compile(r"[A-Za-z]")
_BASE_TO_COLOR = {"ref": "lightgray", "alt": "tomato"}


@dataclass
class Region:
    contig: str
    start: Optional[int] = None
    end: Optional[int] = None


class BAMVariantGraphDataset:
    """
    Region-oriented BAM -> count tensor -> imputation -> genotype/graph workflow.

    The design follows two stages when a variant catalog is not provided:
    1) discover a biallelic catalog for a region
    2) build lazy count tensors and downstream objects against that fixed catalog

    Parameters
    ----------
    inputdf
        DataFrame with one row per sample. Must contain either a ``path`` or ``url``
        column unless ``path_col`` is supplied.
    path_col
        Column holding BAM paths/URLs.
    sample_col
        Optional sample identifier column. If omitted, the DataFrame index is used.
    variant_catalog
        Optional predefined variant table with columns ``contig``, ``pos`` and
        optionally ``ref`` and ``alt``.
    metadata_cols
        Optional default metadata columns to expose to the imputer.
    baseq_threshold
        Minimum base quality for counting or read-path extraction.
    min_mapq
        Minimum mapping quality.
    chunk_size
        Convenience size for helper chunk iterators.
    """

    def __init__(
        self,
        inputdf: pd.DataFrame,
        path_col: Optional[str] = None,
        sample_col: Optional[str] = None,
        variant_catalog: Optional[pd.DataFrame] = None,
        metadata_cols: Optional[Sequence[str]] = None,
        baseq_threshold: int = 30,
        min_mapq: int = 10,
        chunk_size: int = 1_000_000,
    ) -> None:
        self.inputdf = inputdf.copy()
        self.path_col = self._resolve_path_col(path_col)
        self.sample_col = sample_col
        self.sample_ids = self._resolve_sample_ids(sample_col)
        self.metadata_cols = list(metadata_cols or [])
        self.baseq_threshold = int(baseq_threshold)
        self.min_mapq = int(min_mapq)
        self.chunk_size = int(chunk_size)
        self.variant_catalog = self._normalize_variant_catalog(variant_catalog)

        self._validate_inputdf()

    # ------------------------------------------------------------------
    # basic validation / normalization
    # ------------------------------------------------------------------
    def _require_pysam(self) -> None:
        if pysam is None:  # pragma: no cover
            raise ImportError("pysam is required for BAM reading but is not installed.")

    def _resolve_path_col(self, path_col: Optional[str]) -> str:
        if path_col is not None:
            return path_col
        for candidate in ("path", "url"):
            if candidate in self.inputdf.columns:
                return candidate
        raise ValueError("Provide path_col or add a 'path'/'url' column to inputdf.")

    def _resolve_sample_ids(self, sample_col: Optional[str]) -> pd.Index:
        if sample_col is None:
            return pd.Index(self.inputdf.index.astype(str), name=self.inputdf.index.name or "sample")
        if sample_col not in self.inputdf.columns:
            raise KeyError(f"sample_col '{sample_col}' was not found in inputdf")
        return pd.Index(self.inputdf[sample_col].astype(str), name=sample_col)

    def _validate_inputdf(self) -> None:
        if self.path_col not in self.inputdf.columns:
            raise KeyError(f"{self.path_col!r} column not found in inputdf")
        if self.sample_ids.has_duplicates:
            dup = self.sample_ids[self.sample_ids.duplicated()].tolist()[:10]
            raise ValueError(f"Sample identifiers must be unique. Example duplicates: {dup}")
        self.inputdf = self.inputdf.copy()
        self.inputdf.index = self.sample_ids

    @staticmethod
    def _normalize_variant_catalog(variant_catalog: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if variant_catalog is None:
            return None
        vc = variant_catalog.copy()
        rename = {c: c.lower() for c in vc.columns}
        vc = vc.rename(columns=rename)
        required = {"contig", "pos"}
        missing = required - set(vc.columns)
        if missing:
            raise ValueError(f"variant_catalog is missing required columns: {sorted(missing)}")
        if "ref" not in vc.columns:
            vc["ref"] = pd.NA
        if "alt" not in vc.columns:
            vc["alt"] = pd.NA
        vc["contig"] = vc["contig"].astype(str)
        vc["pos"] = vc["pos"].astype(int)
        vc["ref"] = vc["ref"].astype("string")
        vc["alt"] = vc["alt"].astype("string")
        vc = vc.sort_values(["contig", "pos", "ref", "alt"]).drop_duplicates(["contig", "pos", "ref", "alt"])
        vc = vc.reset_index(drop=True)
        return vc

    def set_variant_catalog(self, variant_catalog: pd.DataFrame) -> pd.DataFrame:
        self.variant_catalog = self._normalize_variant_catalog(variant_catalog)
        return self.variant_catalog

    @staticmethod
    def iter_chunks(contig: str, start: int, end: int, chunk_size: int) -> Iterable[Region]:
        for s in range(int(start), int(end), int(chunk_size)):
            e = min(int(end), s + int(chunk_size))
            yield Region(contig=contig, start=s, end=e)

    def _subset_catalog(
        self,
        contig: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        variant_catalog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        vc = self.variant_catalog if variant_catalog is None else self._normalize_variant_catalog(variant_catalog)
        if vc is None:
            raise ValueError("No variant catalog is available. Run discover_variant_catalog() or provide one.")
        out = vc.loc[vc["contig"].eq(str(contig))].copy()
        if start is not None:
            out = out.loc[out["pos"] >= int(start)]
        if end is not None:
            out = out.loc[out["pos"] < int(end)]
        out = out.sort_values(["pos", "ref", "alt"]).reset_index(drop=True)
        if out.empty:
            raise ValueError("No variants found in the requested region")
        if out[["ref", "alt"]].isna().any().any():
            raise ValueError("Variant catalog must contain ref and alt for counting/graph construction")
        if out["pos"].duplicated().any():
            dup = out.loc[out["pos"].duplicated(), "pos"].tolist()[:10]
            raise ValueError(
                "Current tensor/graph methods expect a single biallelic alt per position. Duplicate positions: "
                + str(dup)
            )
        return out

    def _sample_paths(self, sample_subset: Optional[Sequence[str]] = None) -> pd.Series:
        if sample_subset is None:
            return self.inputdf[self.path_col]
        idx = pd.Index(sample_subset).astype(str)
        return self.inputdf.loc[idx, self.path_col]

    # ------------------------------------------------------------------
    # low-level BAM helpers adapted from bamreader.py
    # ------------------------------------------------------------------
    @staticmethod
    def _md_mismatches_fast(md: str) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        ref_off = 0
        i = 0
        n = len(md)
        while i < n:
            c = md[i]
            oc = ord(c)
            if 48 <= oc <= 57:
                num = 0
                while i < n:
                    oc2 = ord(md[i])
                    if 48 <= oc2 <= 57:
                        num = num * 10 + (oc2 - 48)
                        i += 1
                    else:
                        break
                ref_off += num
                continue
            if c == "^":
                i += 1
                start = i
                while i < n:
                    oc2 = ord(md[i])
                    if (65 <= oc2 <= 90) or (97 <= oc2 <= 122):
                        i += 1
                    else:
                        break
                ref_off += (i - start)
                continue
            if (65 <= oc <= 90) or (97 <= oc <= 122):
                out.append((ref_off, c))
                ref_off += 1
                i += 1
                continue
            i += 1
        return out

    @staticmethod
    def _map_ref_offsets_to_qpos(
        cigartuples: Sequence[Tuple[int, int]],
        ref_offsets: Sequence[int],
    ) -> List[int]:
        qpos = 0
        roff = 0
        j = 0
        m = len(ref_offsets)
        out = [-1] * m
        for op, length in cigartuples:
            if j >= m:
                break
            if op in (0, 7, 8):
                block_end = roff + length
                while j < m and ref_offsets[j] < block_end:
                    out[j] = qpos + (ref_offsets[j] - roff)
                    j += 1
                qpos += length
                roff = block_end
            elif op in (2, 3):
                roff += length
            elif op in (1, 4):
                qpos += length
        return out

    def _ensure_index(self, bam_path: str) -> None:
        bai1 = bam_path + ".bai"
        bai2 = os.path.splitext(bam_path)[0] + ".bai"
        if not (os.path.exists(bai1) or os.path.exists(bai2)):
            pysam.index(bam_path)  # type: ignore[arg-type]

    def _open_bam(self, bam_path: str):
        self._require_pysam()
        self._ensure_index(bam_path)
        return pysam.AlignmentFile(bam_path, "rb")

    def extract_region_reads(
        self,
        bam_path: str,
        contig: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        include_secondary: bool = False,
        include_supplementary: bool = False,
        include_duplicates: bool = False,
        include_qcfail: bool = False,
        max_reads: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generalized version of chrY_extract_fast for any contig and optional range.
        Returns one row per read with start/end and a dict of SNV mismatches.
        """
        pos0_list: List[int] = []
        end0_list: List[int] = []
        mate_pos0_list: List[Optional[int]] = []
        mate_end0_list: List[Optional[int]] = []
        variants_list: List[Dict[int, str]] = []
        lane_list: List[int] = []
        qual_list: List[int] = []
        contig_list: List[str] = []
        qname_list: List[str] = []
        pending: Dict[str, int] = {}
        lane_heap: List[Tuple[int, int]] = []
        next_lane_id = 0
        thr = self.baseq_threshold
        has_letter = _RE_HAS_LETTER.search

        with self._open_bam(bam_path) as bam:
            refs = set(bam.references)
            if contig not in refs:
                raise ValueError(f"Contig {contig!r} not found in BAM. First refs: {list(bam.references)[:10]}")

            iterator = bam.fetch(contig, start, end)
            for i, read in enumerate(iterator):
                if max_reads is not None and i >= max_reads:
                    break
                if read.is_unmapped:
                    continue
                if read.is_qcfail and not include_qcfail:
                    continue
                if read.is_duplicate and not include_duplicates:
                    continue
                if read.is_secondary and not include_secondary:
                    continue
                if read.is_supplementary and not include_supplementary:
                    continue
                if read.mapping_quality < self.min_mapq:
                    continue

                pos0 = read.reference_start
                end0 = read.reference_end
                if pos0 is None or end0 is None or end0 <= pos0:
                    continue

                if lane_heap and lane_heap[0][0] <= pos0:
                    _, lane_id = lane_heap.pop(0)
                else:
                    lane_id = next_lane_id
                    next_lane_id += 1
                lane_heap.append((end0, lane_id))
                lane_heap.sort(key=lambda x: x[0])

                mate_pos0 = read.next_reference_start if (read.is_paired and not read.mate_is_unmapped) else None
                mate_end0 = None
                variants: Optional[Dict[int, str]] = None

                if not (read.has_tag("NM") and read.get_tag("NM") == 0):
                    if read.has_tag("MD") and read.cigartuples is not None:
                        md = read.get_tag("MD")
                        if has_letter(md) is not None:
                            mm = self._md_mismatches_fast(md)
                            if mm:
                                ref_offsets = [x[0] for x in mm]
                                qposes = self._map_ref_offsets_to_qpos(read.cigartuples, ref_offsets)
                                qseq = read.query_sequence
                                quals = read.query_qualities
                                if qseq is not None and quals is not None:
                                    ref_start = pos0
                                    for (ref_off, ref_base), qpos in zip(mm, qposes):
                                        if qpos < 0:
                                            continue
                                        if quals[qpos] <= thr:
                                            continue
                                        refb = ref_base.upper()
                                        alt = qseq[qpos].upper()
                                        if refb in "ACGT" and alt in "ACGT" and alt != refb:
                                            if variants is None:
                                                variants = {}
                                            gpos = ref_start + ref_off
                                            if start is not None and gpos < start:
                                                continue
                                            if end is not None and gpos >= end:
                                                continue
                                            variants[gpos] = alt
                if variants is None:
                    variants = {}

                row_idx = len(pos0_list)
                qname_list.append(read.query_name)
                contig_list.append(read.reference_name)
                pos0_list.append(pos0)
                end0_list.append(end0)
                mate_pos0_list.append(mate_pos0)
                mate_end0_list.append(mate_end0)
                variants_list.append(variants)
                lane_list.append(lane_id)
                qual_list.append(read.mapping_quality)

                if read.is_paired and not read.mate_is_unmapped:
                    qn = read.query_name
                    prev = pending.get(qn)
                    if prev is None:
                        pending[qn] = row_idx
                    else:
                        mate_end0_list[row_idx] = end0_list[prev]
                        mate_end0_list[prev] = end0
                        del pending[qn]

        res = pd.DataFrame(
            {
                "sample_path": bam_path,
                "contig": contig_list,
                "query_name": qname_list,
                "pos0": pos0_list,
                "end0": end0_list,
                "mate_pos0": mate_pos0_list,
                "mate_end0": mate_end0_list,
                "variants": variants_list,
                "lane": lane_list,
                "mapping_qual": qual_list,
            }
        )
        if res.empty: return res
        res.loc[res["mate_end0"].isna(), "mate_pos0"] = np.nan
        ref_cols = ["pos0", "end0", "mate_pos0", "mate_end0"]
        res["read_start"] = res[ref_cols].min(axis=1, skipna=True)
        res["read_end"] = res[ref_cols].max(axis=1, skipna=True)
        res["fw_start"] = res[["pos0", "end0"]].min(axis=1)
        res["fw_end"] = res[["pos0", "end0"]].max(axis=1)
        res["mate_start"] = res[["mate_pos0", "mate_end0"]].min(axis=1, skipna=True)
        res["mate_end"] = res[["mate_pos0", "mate_end0"]].max(axis=1, skipna=True)
        return res

    # ------------------------------------------------------------------
    # variant discovery
    # ------------------------------------------------------------------
    def discover_variant_catalog(
        self,
        contig: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        sample_subset: Optional[Sequence[str]] = None,
        min_alt_reads: int = 2,
        min_samples_alt: int = 1,
        max_reads_per_sample: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Discover a biallelic catalog from BAM mismatch evidence in a region.
        """
        records: List[Dict[str, Any]] = []
        sample_paths = self._sample_paths(sample_subset)
        for sample_id, bam_path in sample_paths.items():
            reads = self.extract_region_reads(
                bam_path=str(bam_path),
                contig=contig,
                start=start,
                end=end,
                max_reads=max_reads_per_sample,
            )
            if reads.empty:
                continue
            for _, row in reads.iterrows():
                for pos, alt in row["variants"].items():
                    records.append(
                        {
                            "sample": str(sample_id),
                            "contig": str(contig),
                            "pos": int(pos),
                            "alt": str(alt),
                            "read_count": 1,
                        }
                    )
        if not records:
            out = pd.DataFrame(columns=["contig", "pos", "ref", "alt", "n_alt_reads", "n_samples_alt"])
            self.variant_catalog = out
            return out

        tab = pd.DataFrame(records)
        tab = (
            tab.groupby(["contig", "pos", "alt"], as_index=False)
            .agg(n_alt_reads=("read_count", "sum"), n_samples_alt=("sample", "nunique"))
        )
        tab = tab.loc[tab["n_alt_reads"] >= int(min_alt_reads)]
        tab = tab.loc[tab["n_samples_alt"] >= int(min_samples_alt)]
        if tab.empty:
            out = pd.DataFrame(columns=["contig", "pos", "ref", "alt", "n_alt_reads", "n_samples_alt"])
            self.variant_catalog = out
            return out

        ref_lookup: Dict[Tuple[str, int, str], str] = {}
        sample_paths = self._sample_paths(sample_subset)
        needed = {(r.contig, int(r.pos), str(r.alt)) for r in tab.itertuples(index=False)}
        for _, bam_path in sample_paths.items():
            still_needed = [k for k in needed if k not in ref_lookup]
            if not still_needed:
                break
            with self._open_bam(str(bam_path)) as bam:
                for contig0, pos0, alt0 in still_needed:
                    try:
                        base = bam.count_coverage(contig0, pos0, pos0 + 1, quality_threshold=self.baseq_threshold)
                    except ValueError:
                        continue
                    counts = {"A": int(base[0][0]), "C": int(base[1][0]), "G": int(base[2][0]), "T": int(base[3][0])}
                    if counts:
                        ref = max(counts, key=counts.get)
                        if ref != alt0:
                            ref_lookup[(contig0, pos0, alt0)] = ref
        tab["ref"] = [ref_lookup.get((c, p, a), pd.NA) for c, p, a in zip(tab["contig"], tab["pos"], tab["alt"])]
        tab = tab.dropna(subset=["ref"]).reset_index(drop=True)
        tab = tab.loc[tab["ref"].ne(tab["alt"])].copy()
        tab = tab.sort_values(["contig", "pos", "n_samples_alt", "n_alt_reads", "alt"], ascending=[True, True, False, False, True])
        tab = tab.drop_duplicates(["contig", "pos"], keep="first")
        tab = tab[["contig", "pos", "ref", "alt", "n_alt_reads", "n_samples_alt"]]
        tab = tab.sort_values(["contig", "pos", "alt"]).reset_index(drop=True)
        self.variant_catalog = tab
        return tab

    # ------------------------------------------------------------------
    # count tensor construction
    # ------------------------------------------------------------------
    def _count_sample_region(
        self,
        bam_path: str,
        contig: str,
        start: Optional[int],
        end: Optional[int],
        catalog: pd.DataFrame,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> np.ndarray:
        out = np.full((len(catalog), 2), np.nan, dtype=dtype)
        if catalog.empty: return out

        pos_to_idx = {int(p): i for i, p in enumerate(catalog["pos"].tolist())}
        ref_lookup = dict(zip(catalog["pos"].astype(int), catalog["ref"].astype(str)))
        alt_lookup = dict(zip(catalog["pos"].astype(int), catalog["alt"].astype(str)))

        with self._open_bam(str(bam_path)) as bam:
            for pileup_col in bam.pileup(
                contig,
                start,
                end,
                truncate=True,
                min_base_quality=self.baseq_threshold,
                min_mapping_quality=self.min_mapq,
                stepper="samtools",
            ):
                pos = int(pileup_col.reference_pos)
                idx = pos_to_idx.get(pos)
                if idx is None:
                    continue
                ref = ref_lookup[pos]
                alt = alt_lookup[pos]
                ref_count = 0
                alt_count = 0
                for pileup_read in pileup_col.pileups:
                    if pileup_read.is_del or pileup_read.is_refskip:
                        continue
                    read = pileup_read.alignment
                    qpos = pileup_read.query_position
                    if qpos is None:
                        continue
                    if read.mapping_quality < self.min_mapq:
                        continue
                    quals = read.query_qualities
                    if quals is not None and quals[qpos] < self.baseq_threshold:
                        continue
                    base = read.query_sequence[qpos].upper()
                    if base == ref: ref_count += 1
                    elif base == alt: alt_count += 1
                out[idx, 0] = ref_count
                out[idx, 1] = alt_count
        return out

    def count_region(
        self,
        contig: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        variant_catalog: Optional[pd.DataFrame] = None,
        sample_subset: Optional[Sequence[str]] = None,
        lazy: bool = True,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> Tuple[Union[np.ndarray, "da.Array"], pd.DataFrame, pd.Index]:
        """
        Return a sample x variant x {ref_count, alt_count} tensor for one region.
        """
        catalog = self._subset_catalog(contig=contig, start=start, end=end, variant_catalog=variant_catalog)
        sample_paths = self._sample_paths(sample_subset)
        samples = pd.Index(sample_paths.index.astype(str), name=self.sample_ids.name or "sample")
        n_var = len(catalog)

        if lazy and da is not None and delayed is not None:
            blocks = []
            for bam_path in sample_paths.tolist():
                block = delayed(self._count_sample_region)(
                    bam_path=str(bam_path),
                    contig=contig,
                    start=start,
                    end=end,
                    catalog=catalog,
                    dtype=dtype,
                )
                blocks.append(da.from_delayed(block, shape=(n_var, 2), dtype=dtype))
            counts = da.stack(blocks, axis=0)
        else:
            counts = np.stack(
                [
                    self._count_sample_region(
                        bam_path=str(bam_path),
                        contig=contig,
                        start=start,
                        end=end,
                        catalog=catalog,
                        dtype=dtype,
                    )
                    for bam_path in sample_paths.tolist()
                ],
                axis=0,
            )
        return counts, catalog, samples

    def region_to_xarray(
        self,
        contig: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        variant_catalog: Optional[pd.DataFrame] = None,
        sample_subset: Optional[Sequence[str]] = None,
        lazy: bool = True,
        dtype: Union[str, np.dtype] = np.float32,
    ) -> xr.Dataset:
        counts, catalog, samples = self.count_region(
            contig=contig,
            start=start,
            end=end,
            variant_catalog=variant_catalog,
            sample_subset=sample_subset,
            lazy=lazy,
            dtype=dtype,
        )
        ds = xr.Dataset(
            data_vars={
                "counts": (("sample", "variant", "count_type"), counts),
            },
            coords={
                "sample": samples.to_numpy(),
                "variant": np.arange(len(catalog), dtype=int),
                "count_type": np.array(["ref", "alt"], dtype=object),
                "contig": ("variant", catalog["contig"].astype(str).to_numpy()),
                "pos": ("variant", catalog["pos"].astype(int).to_numpy()),
                "ref": ("variant", catalog["ref"].astype(str).to_numpy()),
                "alt": ("variant", catalog["alt"].astype(str).to_numpy()),
            },
            attrs={"region_contig": contig, "region_start": start, "region_end": end},
        )
        return ds

    # ------------------------------------------------------------------
    # xarray <-> imputation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _variant_labels_from_ds(ds: xr.Dataset) -> List[str]:
        return [
            f"{c}:{p}:{r}>{a}"
            for c, p, r, a in zip(
                ds["contig"].values.tolist(),
                ds["pos"].values.tolist(),
                ds["ref"].values.tolist(),
                ds["alt"].values.tolist(),
            )
        ]

    def flatten_counts_for_imputation(self, ds: xr.Dataset, count_var: str = "counts") -> pd.DataFrame:
        arr = ds[count_var].data
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.asarray(arr)
        labels = self._variant_labels_from_ds(ds)
        cols = []
        for label in labels:
            cols.append(f"{label}:ref_count")
            cols.append(f"{label}:alt_count")
        flat = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
        return pd.DataFrame(flat, index=ds["sample"].values.astype(str), columns=cols)

    @staticmethod
    def _prepare_metadata(meta: pd.DataFrame) -> pd.DataFrame:
        out = meta.copy()
        for col in out.columns:
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(int)
            elif pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
                out[col] = out[col].astype("category")
        return out

    def impute_counts(
        self,
        ds: xr.Dataset,
        metadata_cols: Optional[Sequence[str]] = None,
        imputer: Optional[Any] = None,
        count_var: str = "counts",
        output_var: str = "counts_imputed",
        clip_min: float = 0.0,
    ) -> xr.Dataset:
        """
        Flatten [variant, ref/alt] into columns, append metadata, and impute only the
        count columns using the LightGBM imputer from lgbmi.py.
        """
        meta_cols = list(self.metadata_cols if metadata_cols is None else metadata_cols)
        flat = self.flatten_counts_for_imputation(ds, count_var=count_var)
        model_df = flat.copy()
        if meta_cols:
            meta = self._prepare_metadata(self.inputdf.loc[flat.index, meta_cols])
            model_df = pd.concat([model_df, meta], axis=1)

        count_cols = flat.columns.tolist()
        if imputer is None:
            if LGBMImputer_optuna is None:  # pragma: no cover
                raise ImportError("lgbmi.LGBMImputer_optuna could not be imported")
            imputer = LGBMImputer_optuna(window=100, qc=False, silent=True, max_iter=1)

        imputed = imputer.fit_transform(model_df, columns_subset=count_cols)
        imputed_counts = imputed[count_cols].to_numpy(dtype=float)
        imputed_counts = np.clip(imputed_counts, clip_min, None)
        reshaped = imputed_counts.reshape(len(ds.sample), len(ds.variant), 2)
        return ds.assign({output_var: (("sample", "variant", "count_type"), reshaped)})

    def counts_to_genotypes(
        self,
        ds: xr.Dataset,
        count_var: str = "counts_imputed",
        output_prefix: str = "geno",
        ploidy: int = 2,
    ) -> xr.Dataset:
        """
        Convert ref/alt counts to allele probabilities, dosage, hardcalls, and genotype
        probabilities under a simple binomial model.
        """
        if ploidy < 1:
            raise ValueError("ploidy must be >= 1")
        arr = ds[count_var].data
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.asarray(arr, dtype=float)
        ref = arr[:, :, 0]
        alt = arr[:, :, 1]
        total = ref + alt

        p_alt = np.divide(alt, total, out=np.full_like(alt, np.nan), where=total > 0)
        p_ref = np.divide(ref, total, out=np.full_like(ref, np.nan), where=total > 0)
        dosage = p_alt * ploidy
        hardcall = np.rint(dosage)
        hardcall[~np.isfinite(dosage)] = np.nan
        hardcall = np.clip(hardcall, 0, ploidy)

        g_probs = np.full((arr.shape[0], arr.shape[1], ploidy + 1), np.nan, dtype=float)
        for k in range(ploidy + 1):
            coef = math.comb(ploidy, k)
            g_probs[:, :, k] = coef * np.power(p_alt, k) * np.power(p_ref, ploidy - k)
        norm = np.nansum(g_probs, axis=2, keepdims=True)
        g_probs = np.divide(g_probs, norm, out=np.full_like(g_probs, np.nan), where=norm > 0)

        out = ds.assign(
            {
                f"{output_prefix}_p_ref": (("sample", "variant"), p_ref),
                f"{output_prefix}_p_alt": (("sample", "variant"), p_alt),
                f"{output_prefix}_dosage": (("sample", "variant"), dosage),
                f"{output_prefix}_hardcall": (("sample", "variant"), hardcall),
                f"{output_prefix}_genotype_probability": (("sample", "variant", "genotype"), g_probs),
            }
        )
        out = out.assign_coords(genotype=np.arange(ploidy + 1, dtype=int))
        return out

    # ------------------------------------------------------------------
    # graph construction from read paths
    # ------------------------------------------------------------------
    @staticmethod
    def _node_id(contig: str, pos: int, state: str) -> str:
        return f"{contig}:{int(pos)}:{state}"

    def _covered_read_states(
        self,
        read,
        catalog_slice: pd.DataFrame,
    ) -> List[Tuple[int, str]]:
        if catalog_slice.empty:
            return []
        if read.cigartuples is None or read.query_sequence is None:
            return []
        positions = catalog_slice["pos"].astype(int).to_numpy()
        ref_offsets = (positions - int(read.reference_start)).tolist()
        qposes = self._map_ref_offsets_to_qpos(read.cigartuples, ref_offsets)
        quals = read.query_qualities
        out: List[Tuple[int, str]] = []
        for (pos, ref, alt), qpos in zip(
            catalog_slice[["pos", "ref", "alt"]].itertuples(index=False, name=None),
            qposes,
        ):
            if qpos is None or qpos < 0:
                continue
            if quals is not None and quals[qpos] < self.baseq_threshold:
                continue
            base = read.query_sequence[qpos].upper()
            ref = str(ref).upper()
            alt = str(alt).upper()
            if base == ref:
                out.append((int(pos), "ref"))
            elif base == alt:
                out.append((int(pos), "alt"))
        return out

    def build_read_graph(
        self,
        contig: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        variant_catalog: Optional[pd.DataFrame] = None,
        sample_subset: Optional[Sequence[str]] = None,
        max_reads_per_sample: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Build a proximal-edge read graph where nodes are ``contig:pos:ref`` or
        ``contig:pos:alt`` and edges connect adjacent observed variant states on a read.
        """
        catalog = self._subset_catalog(contig=contig, start=start, end=end, variant_catalog=variant_catalog)
        graph = nx.DiGraph(contig=contig, start=start, end=end)

        sample_paths = self._sample_paths(sample_subset)
        for sample_id, bam_path in sample_paths.items():
            with self._open_bam(str(bam_path)) as bam:
                it = bam.fetch(contig, start, end)
                for i, read in enumerate(it):
                    if max_reads_per_sample is not None and i >= max_reads_per_sample:
                        break
                    if read.is_unmapped or read.mapping_quality < self.min_mapq:
                        continue
                    if read.reference_start is None or read.reference_end is None:
                        continue
                    mask = (catalog["pos"] >= int(read.reference_start)) & (catalog["pos"] < int(read.reference_end))
                    local_catalog = catalog.loc[mask]
                    if local_catalog.empty:
                        continue
                    states = self._covered_read_states(read, local_catalog)
                    if not states:
                        continue
                    node_path = [self._node_id(contig, pos, state) for pos, state in states]
                    for (pos, state), node in zip(states, node_path):
                        if node not in graph:
                            row = catalog.loc[catalog["pos"].eq(pos)].iloc[0]
                            graph.add_node(
                                node,
                                contig=contig,
                                pos=int(pos),
                                state=state,
                                ref=str(row["ref"]),
                                alt=str(row["alt"]),
                                color=_BASE_TO_COLOR[state],
                                samples=set(),
                                n_visits=0,
                            )
                        graph.nodes[node]["samples"].add(str(sample_id))
                        graph.nodes[node]["n_visits"] += 1
                    for u, v in zip(node_path[:-1], node_path[1:]):
                        if graph.has_edge(u, v):
                            graph[u][v]["samples"].add(str(sample_id))
                            graph[u][v]["weight"] += 1
                        else:
                            graph.add_edge(u, v, samples={str(sample_id)}, weight=1)
        return graph

    # ------------------------------------------------------------------
    # graph-based path completion from imputed hardcalls
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_score(graph: nx.DiGraph, u: str, v: str, pseudocount: float = 0.5) -> float:
        if graph.has_edge(u, v):
            return math.log(float(graph[u][v].get("weight", 0.0)) + pseudocount)
        return math.log(pseudocount)

    def _phase_diploid_sample(
        self,
        hardcalls: np.ndarray,
        contig: str,
        positions: np.ndarray,
        graph: nx.DiGraph,
    ) -> np.ndarray:
        """
        Phase diploid hardcalls using a simple dynamic program against the graph edge weights.
        Returns shape (2, n_variant) with 0=ref and 1=alt.
        """
        state_options: List[List[Tuple[int, int]]] = []
        for g in hardcalls:
            if not np.isfinite(g):
                state_options.append([(0, 0), (1, 1), (0, 1), (1, 0)])
            elif int(g) <= 0:
                state_options.append([(0, 0)])
            elif int(g) >= 2:
                state_options.append([(1, 1)])
            else:
                state_options.append([(0, 1), (1, 0)])

        dp: List[np.ndarray] = []
        back: List[np.ndarray] = []
        dp.append(np.zeros(len(state_options[0]), dtype=float))
        back.append(np.full(len(state_options[0]), -1, dtype=int))

        for i in range(1, len(state_options)):
            prev_opts = state_options[i - 1]
            curr_opts = state_options[i]
            pos_prev = int(positions[i - 1])
            pos_curr = int(positions[i])
            scores = np.full(len(curr_opts), -np.inf, dtype=float)
            ptr = np.full(len(curr_opts), -1, dtype=int)
            for j, curr in enumerate(curr_opts):
                curr_nodes = (
                    self._node_id(contig, pos_curr, "alt" if curr[0] else "ref"),
                    self._node_id(contig, pos_curr, "alt" if curr[1] else "ref"),
                )
                for k, prev in enumerate(prev_opts):
                    prev_nodes = (
                        self._node_id(contig, pos_prev, "alt" if prev[0] else "ref"),
                        self._node_id(contig, pos_prev, "alt" if prev[1] else "ref"),
                    )
                    s = (
                        dp[i - 1][k]
                        + self._edge_score(graph, prev_nodes[0], curr_nodes[0])
                        + self._edge_score(graph, prev_nodes[1], curr_nodes[1])
                    )
                    if s > scores[j]:
                        scores[j] = s
                        ptr[j] = k
            dp.append(scores)
            back.append(ptr)

        last_idx = int(np.argmax(dp[-1]))
        path_indices = [last_idx]
        for i in range(len(state_options) - 1, 0, -1):
            last_idx = int(back[i][last_idx])
            path_indices.append(last_idx)
        path_indices = path_indices[::-1]

        phased = np.zeros((2, len(state_options)), dtype=np.int8)
        for i, opt_idx in enumerate(path_indices):
            phased[:, i] = np.asarray(state_options[i][opt_idx], dtype=np.int8)
        return phased

    def impute_graph_paths(
        self,
        ds: xr.Dataset,
        graph: nx.DiGraph,
        hardcall_var: str = "geno_hardcall",
        output_prefix: str = "graph",
    ) -> Tuple[nx.DiGraph, xr.Dataset]:
        """
        Use imputed diploid hardcalls to create a second graph where every sample has two
        completed paths across all positions. Returns the new graph and a genotype dataset.
        """
        if hardcall_var not in ds:
            raise KeyError(f"{hardcall_var!r} is not present in the dataset")

        hard = ds[hardcall_var].data
        if hasattr(hard, "compute"):
            hard = hard.compute()
        hard = np.asarray(hard, dtype=float)
        contigs = np.asarray(ds["contig"].values)
        uniq_contigs = pd.unique(contigs)
        if len(uniq_contigs) != 1:
            raise ValueError("impute_graph_paths currently expects a single-contig region dataset")
        contig = str(uniq_contigs[0])
        positions = np.asarray(ds["pos"].values, dtype=int)

        sample_ids = ds["sample"].values.astype(str)
        hap_state = np.zeros((len(sample_ids), 2, len(positions)), dtype=np.int8)
        out_graph = nx.DiGraph(contig=contig, imputed=True)

        for sidx, sample in enumerate(sample_ids):
            phased = self._phase_diploid_sample(hard[sidx], contig=contig, positions=positions, graph=graph)
            hap_state[sidx] = phased
            for h in range(2):
                node_path = [self._node_id(contig, int(pos), "alt" if int(state) else "ref") for pos, state in zip(positions, phased[h])]
                for pos, state, node in zip(positions, phased[h], node_path):
                    allele_state = "alt" if int(state) else "ref"
                    if node not in out_graph:
                        row = ds.sel(variant=np.where(positions == int(pos))[0][0])
                        out_graph.add_node(
                            node,
                            contig=contig,
                            pos=int(pos),
                            state=allele_state,
                            ref=str(row["ref"].item()),
                            alt=str(row["alt"].item()),
                            color=_BASE_TO_COLOR[allele_state],
                            samples=set(),
                            n_visits=0,
                        )
                    out_graph.nodes[node]["samples"].add(str(sample))
                    out_graph.nodes[node]["n_visits"] += 1
                for u, v in zip(node_path[:-1], node_path[1:]):
                    if out_graph.has_edge(u, v):
                        out_graph[u][v]["samples"].add(str(sample))
                        out_graph[u][v]["weight"] += 1
                    else:
                        out_graph.add_edge(u, v, samples={str(sample)}, weight=1)

        alt_count = hap_state.sum(axis=1).astype(float)
        ref_count = 2.0 - alt_count
        geno_counts = np.stack([ref_count, alt_count], axis=2)

        geno_ds = ds.assign(
            {
                f"{output_prefix}_haplotype_state": (("sample", "haplotype", "variant"), hap_state),
                f"{output_prefix}_counts": (("sample", "variant", "count_type"), geno_counts),
                f"{output_prefix}_hardcall": (("sample", "variant"), alt_count),
                f"{output_prefix}_dosage": (("sample", "variant"), alt_count),
            }
        ).assign_coords(haplotype=np.arange(2, dtype=int))
        return out_graph, geno_ds

    # ------------------------------------------------------------------
    # convenience export
    # ------------------------------------------------------------------
    def to_pandas_genotypes(
        self,
        ds: xr.Dataset,
        hardcall_var: str = "geno_hardcall",
    ) -> pd.DataFrame:
        labels = self._variant_labels_from_ds(ds)
        arr = ds[hardcall_var].data
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.asarray(arr)
        return pd.DataFrame(arr, index=ds["sample"].values.astype(str), columns=labels)


__all__ = ["BAMVariantGraphDataset", "Region"]
