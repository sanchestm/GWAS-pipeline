from __future__ import annotations
import os
import numpy as np
import scipy.sparse as sps
import pysam
from numba import njit, prange
from numba.typed import Dict as nbDict
from numba import types
from collections import OrderedDict
from typing import Iterable, Sequence, Optional, List, Tuple, Union, Any, Dict
from tqdm import tqdm

# import numpy as np
# import scipy.sparse as sps
# from pyroaring import BitMap
# try:
#     from pyroaring import BitMap  # fast compressed integer set
# except ImportError as e:
#     raise ImportError("Please install pyroaring: πp∈stallpyroar∈gpip install pyroaring") from e

# ---------- JIT hot loop: IN-PLACE (no allocations per read) ----------
@njit(cache=True, fastmath=True)
def _count_inplace(encoded: np.ndarray, k: int, stride: int, merge_rc: bool, total: np.ndarray) -> None:
    """
    Update →taltotal (int64[4**k]) with k-mer counts from encodedencoded (int8 array: 0..3 or -1).
    No return, no intermediate allocations.
    """
    L = encoded.shape[0]
    if L < k: return

    # Slide with given stride; build forward and optionally RC codes
    for i in range(0, L - k + 1, stride):
        valid = True
        idx = 0
        if merge_rc: rc_idx = 0
        for j in range(k):
            b = encoded[i + j]
            if b < 0:
                valid = False
                break
            idx = (idx << 2) + b  # multiply by 4 via shift
            if merge_rc:
                rb = 3 - encoded[i + (k - 1 - j)]
                if rb < 0:
                    valid = False
                    break
                rc_idx = (rc_idx << 2) + rb
        if not valid:  continue
        if merge_rc and rc_idx < idx: idx = rc_idx
        total[idx] += np.uint32(1)


# class KmerEncoder:
#     """
#     Fast k-mer counter with optional reverse-complement merging.
#     - Cached ASCII→base mapping (A,C,G,T -> 0,1,2,3; else -1)
#     - Numba JIT kernel that updates a shared counts vector in-place
#     - Simple per-file loop (fast for millions of ~300 bp reads)
#     """

#     def __init__(self, k: int, stride: int = 1, merge_rc: bool = False, warmup: bool = True, silent = False):
#         self.k = int(k)
#         self.stride = int(stride)
#         self.merge_rc = bool(merge_rc)
#         self.n_features = 4 ** self.k
#         self.silent = silent

#         # mapping table (cached once)
#         mapping = np.full(256, -1, dtype=np.int8)
#         mapping[ord("A")] = mapping[ord("a")] = 0
#         mapping[ord("C")] = mapping[ord("c")] = 1
#         mapping[ord("G")] = mapping[ord("g")] = 2
#         mapping[ord("T")] = mapping[ord("t")] = 3
#         self.mapping = mapping

#         if warmup:
#             # one-time JIT compile on a tiny dummy read (amortize compile cost)
#             dummy = np.array([0, 1, 2, 3] * max(1, self.k), dtype=np.int8)
#             tmp = np.zeros(self.n_features, dtype=np.uint32)
#             _count_inplace(dummy, self.k, self.stride, self.merge_rc, tmp)

#     # ---- internal ----
#     def _encode(self, seq: str) -> np.ndarray:
#         # Fast ASCII→uint8 view→int8 mapping
#         return self.mapping[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

#     def count(self, seq: str) -> np.ndarray:
#         """
#         Return a *new* dense vector of counts (allocates once).
#         Prefer using sample_kmer_counts() to aggregate into a single vector.
#         """
#         total = np.zeros(self.n_features, dtype=np.uint32)
#         _count_inplace(self._encode(seq), self.k, self.stride, self.merge_rc, total)
#         return total

#     def count_to_csr(self, seq: str) -> sps.csr_matrix:
#         total = self.count(seq)
#         nz = total.nonzero()[0]
#         return sps.csr_matrix((total[nz], (np.zeros_like(nz), nz)),
#                               shape=(1, self.n_features), dtype=np.uint32)

#     def sample_kmer_counts(self, path: str, max_reads: int | None = None, reference_fasta: str | None = None) -> sps.csr_matrix:
#         """
#         Aggregate k-mers across a file (FASTA/FASTQ/BAM/CRAM/BED) into one CSR row.
#         Drops unmapped/duplicate for BAM/CRAM. No batching: minimal overhead per read.
#         """
#         total = np.zeros(self.n_features, dtype=np.uint32)
#         ext = os.path.splitext(path)[1].lower()

#         # FASTA/FASTQ
#         if ext in [".fa", ".fasta", ".fq", ".fastq"] or path.endswith((".fa.gz", ".fasta.gz", ".fq.gz", ".fastq.gz")):
#             with pysam.FastxFile(path) as fh:
#                 for i, entry in tqdm(enumerate(fh),  disable = self.silent):
#                     if max_reads and i >= max_reads: break
#                     _count_inplace(self._encode(entry.sequence), self.k, self.stride, self.merge_rc, total)

#         # BAM/CRAM
#         elif ext in [".bam", ".cram"]:
#             with pysam.AlignmentFile(path, "rb") as bam:
#                 for i, read in tqdm(enumerate(bam.fetch(until_eof=True)),  disable = self.silent):
#                     if max_reads and i >= max_reads: break
#                     if read.is_unmapped or read.is_duplicate: continue
#                     if read.query_sequence:
#                         _count_inplace(self._encode(read.query_sequence), self.k, self.stride, self.merge_rc, total)

#         # BED (requires reference)
#         elif ext == ".bed" or path.endswith(".bed.gz"):
#             if reference_fasta is None:
#                 raise ValueError("BED input requires reference_fasta path.")
#             ref = pysam.FastaFile(reference_fasta)
#             with (pysam.TabixFile(path) if path.endswith(".gz") else open(path)) as bed:
#                 for i, line in tqdm(enumerate(bed),  disable = self.silent):
#                     if max_reads and i >= max_reads: break
#                     if isinstance(line, bytes): line = line.decode("utf-8")
#                     if line.startswith("#") or not line.strip(): continue
#                     chrom, start, end, *_ = line.strip().split("\t")
#                     seq = ref.fetch(chrom, int(start), int(end))
#                     if seq:
#                         _count_inplace(self._encode(seq), self.k, self.stride, self.merge_rc, total)
#         else:
#             raise ValueError(f"Unsupported file type: {path}")

#         nz = total.nonzero()[0]
#         return sps.csr_matrix((total[nz], (np.zeros_like(nz), nz)),
#                               shape=(1, self.n_features), dtype=np.uint32)




# ---------- 1) Column filtering by document frequency ----------
def filter_kmer_columns_df(
    X_csr: sps.csr_matrix,
    remove_empty: bool = True,         # drop columns with DF == 0
    remove_universal: bool = True,     # drop columns with DF == n_samples
    min_df: int | float | None = None, # optional extra threshold; int or fraction (0,1]
    max_df: int | float | None = None, # optional extra threshold; int or fraction (0,1]
    return_mapping: bool = True
):
    """
    Filter k-mers by document frequency (presence across samples).
    - Uses binary DF (presence/absence). Counts are ignored for DF.
    - If min_df/max_df are given:
        * int: absolute DF thresholds
        * float in (0,1]: fraction of samples

    Returns: (X_filtered, kept_cols) if return_mapping else X_filtered
    """
    n_samples = X_csr.shape[0]
    # compute DF in CSC for fast column counts
    X_csc = X_csr.tocsc(copy=False)
    df = np.diff(X_csc.indptr)  # number of nonzeros per column (binary DF)

    # start with all columns kept
    keep = np.ones(X_csr.shape[1], dtype=bool)

    if remove_empty: keep &= (df > 0)
    if remove_universal: keep &= (df < n_samples)

    if min_df is not None:
        thr = int(np.ceil(min_df * n_samples)) if (isinstance(min_df, float) and 0 < min_df <= 1) else int(min_df)
        keep &= (df >= thr)

    if max_df is not None:
        thr = int(np.floor(max_df * n_samples)) if (isinstance(max_df, float) and 0 < max_df <= 1) else int(max_df)
        keep &= (df <= thr)

    kept_cols = np.where(keep)[0]
    Xf = X_csr[:, kept_cols] if kept_cols.size else X_csr[:, []]
    return (Xf, kept_cols) if return_mapping else Xf


# ---------- 2) MinHash on CSR (binary presence) ----------
def _make_hash_params(num_perm: int, prime: int):
    rng = np.random.default_rng()
    a = rng.integers(1, prime, size=num_perm, dtype=np.int64)
    b = rng.integers(0, prime, size=num_perm, dtype=np.int64)
    return a, b

@njit(parallel=True, cache=True, fastmath=True)
def _minhash_rows_csr(n_rows, indptr, indices, a, b, prime, num_perm):
    """
    For each row, compute min_j ( (a[j]*idx + b[j]) mod prime ) over its column indices 'idx'.
    Returns sketches: int64[n_rows, num_perm]
    """
    sketches = np.full((n_rows, num_perm), np.iinfo(np.int64).max, dtype=np.int64)
    for i in prange(n_rows):
        start, end = indptr[i], indptr[i+1]
        if end <= start:
            continue
        # iterate nonzero column indices in this row
        for pos in range(start, end):
            idx = np.int64(indices[pos])
            # update all hash functions' minima
            for j in range(num_perm):
                hj = (a[j] * idx + b[j]) % prime
                if hj < sketches[i, j]:
                    sketches[i, j] = hj
    return sketches

def minhash_csr(
    X_csr: sps.csr_matrix,
    num_perm: int = 128,
    prime: int = (1 << 61) - 1,
    filter_empty: bool = True,
    filter_universal: bool = True,
    min_df: int | float | None = None,
    max_df: int | float | None = None,
):
    """
    Build MinHash sketches from CSR after optional DF filtering.
    - X_csr: samples x kmers (counts or binary; presence is used)
    - Returns (sketches, kept_cols)
    """
    # 1) DF filtering (presence/absence semantics)
    Xf, kept_cols = filter_kmer_columns_df(
        X_csr,
        remove_empty=filter_empty,
        remove_universal=filter_universal,
        min_df=min_df,
        max_df=max_df,
        return_mapping=True,
    )

    # 2) binarize (presence only) — keep CSR structure
    Xf = (Xf != 0).tocsr(copy=False)

    # 3) hash params & sketches
    a, b = _make_hash_params(num_perm, prime)
    sketches = _minhash_rows_csr(Xf.shape[0], Xf.indptr, Xf.indices, a, b, np.int64(prime), num_perm)
    return sketches, kept_cols


# ---------- 3) Compare sketches (approx. Jaccard) ----------
def jaccard_from_sketches(sketches: np.ndarray) -> np.ndarray:
    """
    All-vs-all Jaccard similarity from MinHash sketches.
    (For large n, prefer nearest-neighbor search instead of full matrix.)
    """
    n, m = sketches.shape
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        S[i, i] = 1.0
        si = sketches[i]
        for j in range(i+1, n):
            eq = np.sum(si == sketches[j])
            val = eq / m
            S[i, j] = S[j, i] = val
    return S





# ============================================================
#  JIT KERNELS — EDGES (rolling; float32 weights; in-place)
# ============================================================

@njit(cache=True)
def _dict_to_arrays_f32(edge_counts):
    """
    Convert numba.typed.Dict[int64 -> float32] to (keys, vals) arrays.
    """
    n = len(edge_counts)
    keys = np.empty(n, dtype=np.int64)
    vals = np.empty(n, dtype=np.float32)
    i = 0
    for k, v in edge_counts.items():
        keys[i] = k
        vals[i] = v
        i += 1
    return keys, vals

@njit(cache=True)
def _add_edge(edge_counts, u: np.int64, v: np.int64, w: np.float32, directed: bool, n_features: int):
    """
    Add (u,v) with weight w (float32) into nbDict, optionally folding if undirected.
    """
    if not directed and v < u:
        u, v = v, u
    key = u * np.int64(n_features) + v
    if key in edge_counts:
        edge_counts[key] += w
    else:
        edge_counts[key] = w

@njit(cache=True, fastmath=True)
def _rolling_first_last_kmer(encoded: np.ndarray, k: int):
    """
    Return (first_idx, first_rc, has_first, last_idx, last_rc, has_last) using 2-bit packing.
    Resets across invalid bases (N).
    """
    L = encoded.shape[0]
    if L < k:
        return -1, -1, False, -1, -1, False

    shift_msb = np.int64(2 * (k - 1))
    mask_km1 = (np.int64(1) << (2 * (k - 1))) - np.int64(1)

    idx = np.int64(0)
    rc  = np.int64(0)
    run = 0

    have_first = False
    f_idx = np.int64(-1); f_rc = np.int64(-1)
    l_idx = np.int64(-1); l_rc = np.int64(-1)
    have_last = False

    for pos in range(L):
        b = encoded[pos]
        if b < 0:
            run = 0
            idx = 0
            rc  = 0
            continue

        idx = ((idx & mask_km1) << 2) | np.int64(b)
        rc  = ( (np.int64(3 - b) << shift_msb) | (rc >> 2) )

        run += 1
        if run >= k:
            if not have_first:
                f_idx, f_rc, have_first = idx, rc, True
            l_idx, l_rc, have_last = idx, rc, True

    return f_idx, f_rc, have_first, l_idx, l_rc, have_last

@njit(cache=True, fastmath=True)
def _edges_inplace_update_rolling(encoded: np.ndarray,
                                  k: int,
                                  edge_stride: int,
                                  merge_rc: bool,
                                  directed: bool,
                                  n_features: int,
                                  edge_counts) -> None:
    """
    Rolling O(1)/step k-mer → k-mer edges into a Dict[int64 -> float32].
    Semantics:
      - Always add forward edges u_fwd -> v_fwd with +1.0
      - If merge_rc=True, ALSO add reverse-strand edges rc(v) -> rc(u) with +1.0
      - If directed=False, fold each edge by sorting (u,v)
    """
    if edge_stride < 1:
        return

    L = encoded.shape[0]
    if L < k + edge_stride:
        # not enough span to form at least one edge
        return

    shift_msb = np.int64(2 * (k - 1))
    mask_km1 = (np.int64(1) << (2 * (k - 1))) - np.int64(1)

    idx = np.int64(0)  # forward k-mer
    rc  = np.int64(0)  # reverse-comp k-mer
    run = 0

    s = edge_stride
    # ring buffer for k-mers at distance s
    buf_idx = np.full(s, -1, dtype=np.int64)
    buf_rc  = np.full(s, -1, dtype=np.int64)
    buf_ok  = np.zeros(s, dtype=np.uint8)

    for pos in range(L):
        b = encoded[pos]
        if b < 0:
            run = 0
            idx = 0
            rc  = 0
            continue

        idx = ((idx & mask_km1) << 2) | np.int64(b)
        rc  = ( (np.int64(3 - b) << shift_msb) | (rc >> 2) )

        run += 1
        if run < k:
            continue

        start_i = pos - k + 1

        # If we have a window exactly 's' before, add edge between them
        if start_i >= s and buf_ok[(start_i - s) % s] == 1:
            u = buf_idx[(start_i - s) % s]
            v = idx
            _add_edge(edge_counts, u, v, np.float32(1.0), directed, n_features)

            if merge_rc:
                urc = buf_rc[(start_i - s) % s]
                vrc = rc
                # reverse-strand: rc(v) -> rc(u)
                _add_edge(edge_counts, vrc, urc, np.float32(1.0), directed, n_features)

        # store current window for future edges (distance s ahead)
        slot = start_i % s
        buf_idx[slot] = idx
        buf_rc[slot]  = rc
        buf_ok[slot]  = 1


# ============================================================
#  HELPERS — edge dict factory & conversion
# ============================================================

def new_edge_dict():
    """Create a numba.typed.Dict[int64 -> float32] for edge weights."""
    return nbDict.empty(key_type=types.int64, value_type=types.float32)

def edges_dict_to_csr(edge_counts, n_features: int) -> sps.csr_matrix:
    """Convert the Dict to CSR (float32)."""
    keys, vals = _dict_to_arrays_f32(edge_counts)
    if keys.size == 0:
        return sps.csr_matrix((n_features, n_features), dtype=np.float32)
    rows = (keys // np.int64(n_features)).astype(np.int64)
    cols = (keys %  np.int64(n_features)).astype(np.int64)
    adj = sps.coo_matrix((vals, (rows, cols)),
                         shape=(n_features, n_features),
                         dtype=np.float32).tocsr()
    adj.eliminate_zeros()
    return adj


# ============================================================
#  CLASS — organized, warm-started, in-place APIs
# ============================================================

class KmerEncoder:
    """
    Fast k-mer counter + edge builder (rolling) with optional reverse-strand merging.
    - Counts: merge_rc canonicalizes nodes (min(fwd,rc)) as in your original.
    - Edges: merge_rc ADDS both directions (u_fwd->v_fwd and rc(v)->rc(u)).
    """

    def __init__(self, k: int, stride: int = 1, merge_rc: bool = False, warmup: bool = True, silent: bool = False):
        if k <= 0 or k > 31:
            raise ValueError("k must be in [1, 31] for 2-bit packing in int64.")
        if stride <= 0:
            raise ValueError("stride must be >= 1.")

        self.k = int(k)
        self.stride = int(stride)
        self.merge_rc = bool(merge_rc)
        self.n_features = 4 ** self.k
        self.silent = bool(silent)

        # ASCII→base mapping
        mapping = np.full(256, -1, dtype=np.int8)
        mapping[ord("A")] = mapping[ord("a")] = 0
        mapping[ord("C")] = mapping[ord("c")] = 1
        mapping[ord("G")] = mapping[ord("g")] = 2
        mapping[ord("T")] = mapping[ord("t")] = 3
        self.mapping = mapping

        # Warm compile the JIT kernels on tiny dummy inputs (amortize compile)
        if warmup:
            dummy = np.array([0, 1, 2, 3] * max(1, self.k), dtype=np.int8)
            tmp_counts = np.zeros(self.n_features, dtype=np.uint32)
            _count_inplace(dummy, self.k, self.stride, self.merge_rc, tmp_counts)

            ed = new_edge_dict()
            _edges_inplace_update_rolling(dummy, self.k, self.stride, self.merge_rc, True, self.n_features, ed)
            _rolling_first_last_kmer(dummy, self.k)
            _add_edge(ed, np.int64(0), np.int64(0), np.float32(1.0), True, self.n_features)

    # ---------- internal ----------
    def _encode(self, seq: str) -> np.ndarray:
        return self.mapping[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

    # ---------- COUNTS API (unchanged) ----------
    def count(self, seq: str) -> np.ndarray:
        total = np.zeros(self.n_features, dtype=np.uint32)
        _count_inplace(self._encode(seq), self.k, self.stride, self.merge_rc, total)
        return total

    def count_to_csr(self, seq: str) -> sps.csr_matrix:
        total = self.count(seq)
        nz = total.nonzero()[0]
        return sps.csr_matrix((total[nz], (np.zeros_like(nz), nz)),
                              shape=(1, self.n_features), dtype=np.uint32)

    def sample_kmer_counts(self, path: str, max_reads: int | None = None, reference_fasta: str | None = None) -> sps.csr_matrix:
        total = np.zeros(self.n_features, dtype=np.uint32)
        ext = os.path.splitext(path)[1].lower()

        # FASTA/FASTQ
        if ext in [".fa", ".fasta", ".fq", ".fastq"] or path.endswith((".fa.gz", ".fasta.gz", ".fq.gz", ".fastq.gz")):
            with pysam.FastxFile(path) as fh:
                for i, entry in tqdm(enumerate(fh), disable=self.silent):
                    if max_reads and i >= max_reads: break
                    _count_inplace(self._encode(entry.sequence), self.k, self.stride, self.merge_rc, total)

        # BAM/CRAM
        elif ext in [".bam", ".cram"]:
            with pysam.AlignmentFile(path, "rb") as bam:
                for i, read in tqdm(enumerate(bam.fetch(until_eof=True)), disable=self.silent):
                    if max_reads and i >= max_reads: break
                    if read.is_unmapped or read.is_duplicate: continue
                    if read.query_sequence:
                        _count_inplace(self._encode(read.query_sequence), self.k, self.stride, self.merge_rc, total)

        # BED
        elif ext == ".bed" or path.endswith(".bed.gz"):
            if reference_fasta is None:
                raise ValueError("BED input requires reference_fasta path.")
            ref = pysam.FastaFile(reference_fasta)
            with (pysam.TabixFile(path) if path.endswith(".gz") else open(path)) as bed:
                for i, line in tqdm(enumerate(bed), disable=self.silent):
                    if max_reads and i >= max_reads: break
                    if isinstance(line, bytes): line = line.decode("utf-8")
                    if line.startswith("#") or not line.strip(): continue
                    chrom, start, end, *_ = line.strip().split("\t")
                    seq = ref.fetch(chrom, int(start), int(end))
                    if seq:
                        _count_inplace(self._encode(seq), self.k, self.stride, self.merge_rc, total)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        nz = total.nonzero()[0]
        return sps.csr_matrix((total[nz], (np.zeros_like(nz), nz)),
                              shape=(1, self.n_features), dtype=np.uint32)

    # ---------- EDGES API (all IN-PLACE) ----------
    def count_edges_inplace(self, seq: str, edge_counts, directed: bool = True, edge_stride: int | None = None) -> None:
        """
        Update 'edge_counts' (Dict[int64->float32]) with within-read edges from a single sequence.
        """
        if edge_stride is None: edge_stride = self.stride
        _edges_inplace_update_rolling(self._encode(seq), self.k, edge_stride,
                                      self.merge_rc, directed, self.n_features, edge_counts)

    def sample_kmer_edges(self,
                          path: str | tuple[str, str] | list[str],
                          max_reads: int | None = None,
                          reference_fasta: str | None = None,
                          directed: bool = True,
                          edge_stride: int | None = None,
                          # paired FASTQ options
                          path2: str | None = None,
                          mate_link_weight: float = 0.5,
                          mate_link_symmetric: bool = False,
                          # NEW: BAM/CRAM mate-link options
                          bam_use_mates: bool = True,
                          bam_pair_buffer_limit: int = 500_000) -> sps.csr_matrix:
        """
        Build a NEW sparse k-mer adjacency from input and return it as CSR (float32).
    
        Within-read edges: +1.0 each (rolling O(1)).
        Paired FASTQ (path & path2, or path=(r1,r2)): add mate link(s) with 'mate_link_weight':
            last_kmer(R1) -> first_kmer(R2)
            and if merge_rc: rc(first_R2) -> rc(last_R1)
        If mate_link_symmetric=True, also add last(R2) -> first(R1) (+ reverse-strand if merge_rc).
    
        BAM/CRAM:
          - Counts within-read edges as usual.
          - If bam_use_mates=True, links read pairs using flags. A small FIFO buffer
            (size = bam_pair_buffer_limit) stores first/last k-mers per read name
            until the mate appears; then adds the same mate links as above.
          - Skips unmapped/duplicate/secondary/supplementary reads for both edges and mate links.
        """
        if edge_stride is None:
            edge_stride = self.stride
    
        def _is_fastx(p: str):
            pe = os.path.splitext(p)[1].lower()
            return (pe in [".fa", ".fasta", ".fq", ".fastq"] or
                    p.endswith((".fa.gz", ".fasta.gz", ".fq.gz", ".fastq.gz")))
    
        # fresh accumulator
        edge_counts = new_edge_dict()
    
        # Normalize paired FASTQ inputs
        r1_path = None
        r2_path = None
        if isinstance(path, (tuple, list)):
            if len(path) != 2:
                raise ValueError("If 'path' is a tuple/list, it must have exactly two FASTQ filenames (R1, R2).")
            r1_path, r2_path = path[0], path[1]
        elif path2 is not None:
            r1_path, r2_path = path, path2
    
        # --------- Paired FASTA/FASTQ ----------
        if r1_path is not None and r2_path is not None:
            if not (_is_fastx(r1_path) and _is_fastx(r2_path)):
                raise ValueError("Paired FASTQ mode requires FASTA/FASTQ inputs.")
            with pysam.FastxFile(r1_path) as fh1, pysam.FastxFile(r2_path) as fh2:
                for i, (e1, e2) in tqdm(enumerate(zip(fh1, fh2)), disable=self.silent):
                    if max_reads and i >= max_reads: break
                    s1 = self._encode(e1.sequence)
                    s2 = self._encode(e2.sequence)
    
                    # within-read edges (R1, R2)
                    _edges_inplace_update_rolling(s1, self.k, edge_stride, self.merge_rc, directed, self.n_features, edge_counts)
                    _edges_inplace_update_rolling(s2, self.k, edge_stride, self.merge_rc, directed, self.n_features, edge_counts)
    
                    # mate links
                    f1, frc1, okf1, l1, lrc1, okl1 = _rolling_first_last_kmer(s1, self.k)
                    f2, frc2, okf2, l2, lrc2, okl2 = _rolling_first_last_kmer(s2, self.k)
    
                    # R1 -> R2
                    if okl1 and okf2:
                        _add_edge(edge_counts, np.int64(l1), np.int64(f2),
                                  np.float32(mate_link_weight), directed, self.n_features)
                        if self.merge_rc:
                            _add_edge(edge_counts, np.int64(frc2), np.int64(lrc1),
                                      np.float32(mate_link_weight), directed, self.n_features)
    
                    # optional symmetric R2 -> R1
                    if mate_link_symmetric and okl2 and okf1:
                        _add_edge(edge_counts, np.int64(l2), np.int64(f1),
                                  np.float32(mate_link_weight), directed, self.n_features)
                        if self.merge_rc:
                            _add_edge(edge_counts, np.int64(frc1), np.int64(lrc2),
                                      np.float32(mate_link_weight), directed, self.n_features)
    
        # --------- Single FASTA/FASTQ ----------
        elif isinstance(path, str) and _is_fastx(path):
            with pysam.FastxFile(path) as fh:
                for i, entry in tqdm(enumerate(fh), disable=self.silent):
                    if max_reads and i >= max_reads: break
                    _edges_inplace_update_rolling(self._encode(entry.sequence), self.k, edge_stride,
                                                  self.merge_rc, directed, self.n_features, edge_counts)
    
        # --------- BAM/CRAM ----------
        elif isinstance(path, str) and os.path.splitext(path)[1].lower() in [".bam", ".cram"]:
            # FIFO buffer: read_name -> (is_read1, first_idx, first_rc, last_idx, last_rc)
            mate_buf: OrderedDict[str, tuple] = OrderedDict()
    
            def _push_buffer(qname: str, payload: tuple):
                if qname in mate_buf:
                    mate_buf.pop(qname, None)  # refresh position
                mate_buf[qname] = payload
                if len(mate_buf) > bam_pair_buffer_limit:
                    mate_buf.popitem(last=False)
    
            with pysam.AlignmentFile(path, "rb") as bam:
                for i, read in tqdm(enumerate(bam.fetch(until_eof=True)), disable=self.silent):
                    if max_reads and i >= max_reads: break
    
                    # skip low-quality contexts for both edges and mate links
                    if read.is_unmapped or read.is_duplicate or read.is_secondary or read.is_supplementary:
                        continue
    
                    if read.query_sequence:
                        s_enc = self._encode(read.query_sequence)
                        # within-read edges
                        _edges_inplace_update_rolling(s_enc, self.k, edge_stride,
                                                      self.merge_rc, directed, self.n_features, edge_counts)
    
                        # mate links (if enabled and paired)
                        if bam_use_mates and read.is_paired and not read.mate_is_unmapped:
                            f, frc, okf, l, lrc, okl = _rolling_first_last_kmer(s_enc, self.k)
                            if okf and okl:
                                qn = read.query_name
                                is_r1 = bool(read.is_read1)
    
                                prev = mate_buf.pop(qn, None)
                                if prev is None:
                                    # store current; wait for mate
                                    _push_buffer(qn, (is_r1, f, frc, l, lrc))
                                else:
                                    # we have both mates: identify R1 vs R2
                                    prev_is_r1, f_prev, frc_prev, l_prev, lrc_prev = prev
                                    if is_r1 == prev_is_r1:
                                        # Both halves claim same end (rare; supplementary handled above). Skip linking.
                                        continue
    
                                    if is_r1:
                                        # current is R1, prev is R2
                                        l1, lrc1 = l, lrc
                                        f2, frc2 = f_prev, frc_prev
                                        l2, lrc2 = l_prev, lrc_prev
                                        f1, frc1 = f, frc
                                    else:
                                        # current is R2, prev is R1
                                        l1, lrc1 = l_prev, lrc_prev
                                        f2, frc2 = f, frc
                                        l2, lrc2 = l, lrc
                                        f1, frc1 = f_prev, frc_prev
    
                                    # R1 -> R2
                                    _add_edge(edge_counts, np.int64(l1), np.int64(f2),
                                              np.float32(mate_link_weight), directed, self.n_features)
                                    if self.merge_rc:
                                        _add_edge(edge_counts, np.int64(frc2), np.int64(lrc1),
                                                  np.float32(mate_link_weight), directed, self.n_features)
    
                                    # (optional) R2 -> R1 symmetric
                                    if mate_link_symmetric:
                                        _add_edge(edge_counts, np.int64(l2), np.int64(f1),
                                                  np.float32(mate_link_weight), directed, self.n_features)
                                        if self.merge_rc:
                                            _add_edge(edge_counts, np.int64(frc1), np.int64(lrc2),
                                                      np.float32(mate_link_weight), directed, self.n_features)
    
            # On EOF: any unmatched reads in mate_buf are ignored (no link possible)
    
        # --------- BED ----------
        elif isinstance(path, str) and (path.endswith(".bed") or path.endswith(".bed.gz")):
            if reference_fasta is None:
                raise ValueError("BED input requires reference_fasta path.")
            ref = pysam.FastaFile(reference_fasta)
            with (pysam.TabixFile(path) if path.endswith(".gz") else open(path)) as bed:
                for i, line in tqdm(enumerate(bed), disable=self.silent):
                    if max_reads and i >= max_reads: break
                    if isinstance(line, bytes): line = line.decode("utf-8")
                    if line.startswith("#") or not line.strip(): continue
                    chrom, start, end, *_ = line.strip().split("\t")
                    seq = ref.fetch(chrom, int(start), int(end))
                    if seq:
                        _edges_inplace_update_rolling(self._encode(seq), self.k, edge_stride,
                                                      self.merge_rc, directed, self.n_features, edge_counts)
        else:
            raise ValueError(f"Unsupported file type or input combination: {path}")
    
        # pack & return
        return edges_dict_to_csr(edge_counts, self.n_features)




class ColoredGraphRBM:
    """
    Colored de Bruijn graph with Roaring bitmaps per edge.

    Attributes
    ----------
    shape : (n_nodes, n_nodes)
        Must match the k-mer index space (n_nodes = 4**k).
    rows, cols : int64 arrays
        Edge endpoints (u -> v) for each nonzero edge.
    colors : list[BitMap]
        Per-edge set of animal IDs (compressed).
    weight_sum : float32 array or None
        Sum of weights across animals for each edge (optional).
    """
    def __init__(self,
                 shape: Tuple[int, int],
                 rows: np.ndarray,
                 cols: np.ndarray,
                 colors: List[BitMap],
                 weight_sum: Optional[np.ndarray] = None):
        self.shape = shape
        self.rows = rows.astype(np.int64, copy=False)
        self.cols = cols.astype(np.int64, copy=False)
        self.colors = colors
        self.weight_sum = (weight_sum.astype(np.float32, copy=False)
                           if weight_sum is not None else None)

    # ---------- Queries / Projections ----------
    def color_counts(self) -> np.ndarray:
        """Number of animals per edge."""
        return np.fromiter((len(bm) for bm in self.colors), count=len(self.colors), dtype=np.int32)

    def edge_mask_has_any(self, animals: Sequence[int]) -> np.ndarray:
        """Boolean mask: edges present in ANY of the given animals."""
        target = BitMap(animals)
        return np.fromiter((len(bm & target) > 0 for bm in self.colors), count=len(self.colors), dtype=bool)

    def edge_mask_has_all(self, animals: Sequence[int]) -> np.ndarray:
        """Boolean mask: edges present in ALL of the given animals."""
        target = BitMap(animals)
        return np.fromiter((bm.issuperset(target) for bm in self.colors), count=len(self.colors), dtype=bool)

    def to_presence_csr(self, animals: Optional[Sequence[int]] = None) -> sps.csr_matrix:
        """
        Return a boolean presence CSR.
        - If animalsanimals is None: union across all animals (edge present if any color set nonempty)
        - Else: union across the specified animals (edge present if any of those animals have it)
        """
        if animals is None:
            keep = np.fromiter((len(bm) > 0 for bm in self.colors), count=len(self.colors), dtype=bool)
        else:
            keep = self.edge_mask_has_any(animals)
        if not keep.any():
            return sps.csr_matrix(self.shape, dtype=np.uint8)
        data = np.ones(int(keep.sum()), dtype=np.uint8)
        return sps.csr_matrix((data, (self.rows[keep], self.cols[keep])), shape=self.shape)

    def to_weight_csr(self, animals: Optional[Sequence[int]] = None) -> sps.csr_matrix:
        """
        Return a weight CSR.
        NOTE: If animalsanimals is provided, weights are still the *total* across all animals.
              If you need subset-specific weights, see notes in the code comments below.
        """
        if self.weight_sum is None:
            raise ValueError("This ColoredGraphRBM was built without weights (keep_weight_sum=False).")
        if animals is None:
            keep = np.ones_like(self.rows, dtype=bool)
        else:
            keep = self.edge_mask_has_any(animals)
        if not keep.any():
            return sps.csr_matrix(self.shape, dtype=np.float32)
        return sps.csr_matrix((self.weight_sum[keep], (self.rows[keep], self.cols[keep])), shape=self.shape)

    def keep_edges_with_min_colors(self, t: int) -> "ColoredGraphRBM":
        """Filter to edges present in at least t animals."""
        cnt = self.color_counts()
        keep = (cnt >= t)
        return ColoredGraphRBM(self.shape, self.rows[keep], self.cols[keep],
                               [self.colors[i] for i in np.flatnonzero(keep)],
                               self.weight_sum[keep] if self.weight_sum is not None else None)

    # ---------- Serialization (compact, columnar) ----------
    def to_parquet(self, path: str) -> None:
        """
        Save edge table with columns:
          row:int64, col:int64, weight_sum:float32 (optional), colors:bytes
        Colors are serialized roaring bitmaps (compact).
        """
        try:
            import pyarrow as pa, pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("πp∈stallpyarrowpip install pyarrow to save parquet.") from e

        rows = pa.array(self.rows, type=pa.int64())
        cols = pa.array(self.cols, type=pa.int64())
        cols_bytes = pa.array([bytes(bm) for bm in self.colors], type=pa.binary())
        arrays = [rows, cols, cols_bytes]
        names = ["row", "col", "colors"]
        if self.weight_sum is not None:
            arrays.insert(2, pa.array(self.weight_sum, type=pa.float32()))
            names.insert(2, "weight_sum")
        table = pa.table(arrays, names=names)
        pq.write_table(table, path, compression="zstd")

    @staticmethod
    def from_parquet(path: str, shape: Tuple[int, int]) -> "ColoredGraphRBM":
        """Load a graph saved with →parquetto_parquet."""
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        row = tbl["row"].to_numpy()
        col = tbl["col"].to_numpy()
        weight_sum = tbl["weight_sum"].to_numpy() if "weight_sum" in tbl.column_names else None
        colors_bytes = tbl["colors"].to_pylist()
        cols_rb = [BitMap(b) for b in colors_bytes]
        return ColoredGraphRBM(shape, row, col, cols_rb, weight_sum)


def build_colored_union_roaring(
    adj_list: Iterable[sps.csr_matrix],
    keep_weight_sum: bool = True,
    animal_ids: Optional[Sequence[int]] = None,
) -> ColoredGraphRBM:
    """
    Fuse many per-animal edge CSRs into a roaring-colored graph.
    - adj_list: iterable of CSR matrices (same shape/index space).
    - animal_ids: optional IDs; default is 0..N-1 in iteration order.
    - keep_weight_sum: if True, accumulate float32 total weights.

    Returns
    -------
    ColoredGraphRBM
    """
    # Peek first to get shape; also materialize as list if needed
    adj_list = list(adj_list)
    if not adj_list:
        raise ValueError("adj_list is empty.")
    shape0 = adj_list[0].shape
    if any(A.shape != shape0 for A in adj_list):
        raise ValueError("All matrices must share the same shape.")

    if animal_ids is None:
        animal_ids = list(range(len(adj_list)))
    if len(animal_ids) != len(adj_list):
        raise ValueError("animal_ids length must match number of matrices.")

    nrows, ncols = shape0
    # Edge maps keyed by linearized edge ID = u*ncols + v
    colors: dict[int, BitMap] = {}
    weight_sum: Optional[dict[int, float]] = {} if keep_weight_sum else None

    for aid, A in zip(animal_ids, adj_list):
        if not sps.isspmatrix_csr(A):
            A = A.tocsr()
        A = A.tocoo()  # iterate efficiently
        u = A.row
        v = A.col
        w = A.data
        if w.dtype != np.float32:
            w = w.astype(np.float32, copy=False)

        # Merge edges
        for uu, vv, ww in zip(u, v, w):
            key = int(uu) * ncols + int(vv)
            bm = colors.get(key)
            if bm is None:
                bm = BitMap([aid])
                colors[key] = bm
                if keep_weight_sum:
                    weight_sum[key] = float(ww)
            else:
                bm.add(aid)
                if keep_weight_sum:
                    weight_sum[key] += float(ww)

    # Finalize to arrays (stable order)
    keys = np.fromiter(colors.keys(), dtype=np.int64, count=len(colors))
    keys.sort()
    rows = (keys // ncols).astype(np.int64)
    cols = (keys %  ncols).astype(np.int64)
    cols_rb = [colors[int(k)] for k in keys]
    wsum_arr = (np.fromiter((weight_sum[int(k)] for k in keys), dtype=np.float32, count=len(keys))
                if keep_weight_sum else None)

    return ColoredGraphRBM((nrows, ncols), rows, cols, cols_rb, wsum_arr)



# # 1) Build per-animal adjacencies (float32 weights) using your KmerEncoder
# #    e.g., Adj_i = enc.sample_kmer_edges(("R1_i.fastq.gz","R2_i.fastq.gz"), edge_stride=1, mate_link_weight=0.25)

# per_animal = [Adj_0, Adj_1, Adj_2, ...]  # list of CSR matrices, same shape
# animal_ids = [1001, 1002, 1003, ...]      # optional external IDs (ints)

# # 2) Fuse into a colored graph with roaring bitmaps
# C = build_colored_union_roaring(per_animal, keep_weight_sum=True, animal_ids=animal_ids)

# # 3) Basic queries
# print("Edges:", len(C.rows))
# print("Median sharing:", np.median(C.color_counts()))
# G_presence = C.to_presence_csr()                # union presence across all animals
# G_weight    = C.to_weight_csr()                 # total weight across animals
# G_subset    = C.to_presence_csr([1001, 1005])   # presence restricted to animals {1001,1005}

# # 4) Filter to edges seen in ≥ 10 animals
# C10 = C.keep_edges_with_min_colors(10)
# G10 = C10.to_presence_csr()

# # 5) Save / load (compact)
# C.to_parquet("colored_dbg.parquet")             # stores row, col, weight_sum (opt), and serialized bitmaps
# C2 = ColoredGraphRBM.from_parquet("colored_dbg.parquet", shape=C.shape)




# ---------------------------------------------------------------------
# Wrapper: build roaring-colored graph directly from many file inputs
# ---------------------------------------------------------------------
AnimalInput = Union[str, Tuple[str, str], Sequence[str], Dict[str, Any]]

def build_colored_from_files(
    enc: "KmerEncoder",
    inputs: Sequence[AnimalInput],
    animal_ids: Optional[Sequence[int]] = None,
    *,
    # Global defaults (can be overridden per animal via dict entries)
    directed: bool = True,
    edge_stride: Optional[int] = None,
    mate_link_weight: float = 0.5,
    mate_link_symmetric: bool = False,
    bam_use_mates: bool = True,
    bam_pair_buffer_limit: int = 500_000,
    reference_fasta: Optional[str] = None,   # for BED inputs (global default)
    max_reads: Optional[int] = None,         # global cap per animal
    keep_weight_sum: bool = True,
) -> "ColoredGraphRBM":
    """
    Build a roaring-colored de Bruijn graph from many animals' inputs.

    Each element of ∈putsinputs can be one of:
      - str: single path (FASTA/FASTQ/BAM/CRAM/BED)
      - (str, str): paired FASTQ (R1, R2)
      - dict: {
            "path": str | (str,str),
            "path2": Optional[str],             # optional second FASTQ if not a tuple
            "reference_fasta": Optional[str],   # override global for BED
            "directed": Optional[bool],
            "edge_stride": Optional[int],
            "mate_link_weight": Optional[float],
            "mate_link_symmetric": Optional[bool],
            "bam_use_mates": Optional[bool],
            "bam_pair_buffer_limit": Optional[int],
            "max_reads": Optional[int],
        }

    Parameters
    ----------
    enc : KmerEncoder
        Your configured encoder (k, stride, merge_rc, etc.). All animals must
        share the same k/encoding so the node space matches.
    animal_ids : Optional sequence of ints
        External IDs to store in the color sets. Defaults to 0..N-1.

    Returns
    -------
    ColoredGraphRBM
        rows/cols arrays with per-edge roaring bitmap colors and optional weight_sum.
    """
    n = len(inputs)
    if n == 0: raise ValueError("∈putsinputs is empty.")

    if animal_ids is None: animal_ids = list(range(n))
    if len(animal_ids) != n: raise ValueError("animalanimal_ids length must match number of inputs.")
    n_nodes = enc.n_features
    shape = (n_nodes, n_nodes)

    # Accumulators keyed by edge key = u * n_nodes + v
    colors: Dict[int, BitMap] = {}
    wsum: Optional[Dict[int, float]] = {} if keep_weight_sum else None

    def _normalize_one(inp: AnimalInput) -> Dict[str, Any]:
        """Return a standardized dict with resolved fields for this animal."""
        if isinstance(inp, dict):
            d = dict(inp)  # copy
        elif isinstance(inp, (tuple, list)):
            if len(inp) != 2:
                raise ValueError("Tuple/list animal input must be (R1, R2).")
            d = {"path": (inp[0], inp[1])}
        elif isinstance(inp, str):
            d = {"path": inp}
        else:
            raise TypeError(f"Unsupported input type: {type(inp)}")

        # Fill defaults (only if missing)
        d.setdefault("directed", directed)
        d.setdefault("edge_stride", edge_stride)
        d.setdefault("mate_link_weight", mate_link_weight)
        d.setdefault("mate_link_symmetric", mate_link_symmetric)
        d.setdefault("bam_use_mates", bam_use_mates)
        d.setdefault("bam_pair_buffer_limit", bam_pair_buffer_limit)
        d.setdefault("reference_fasta", reference_fasta)
        d.setdefault("max_reads", max_reads)

        # Normalize (path, path2)
        if "path2" in d and d["path2"] is not None and isinstance(d["path"], str):
            d["path"] = (d["path"], d["path2"])
        return d

    for idx, (inp, aid) in enumerate(zip(inputs, animal_ids)):
        cfg = _normalize_one(inp)
        # Build this animal's adjacency (float32 CSR) with your encoder
        A: sps.csr_matrix = enc.sample_kmer_edges(
            path=cfg["path"],
            max_reads=cfg["max_reads"],
            reference_fasta=cfg["reference_fasta"],
            directed=cfg["directed"],
            edge_stride=cfg["edge_stride"],
            path2=None,  # handled by tuple in path if needed
            mate_link_weight=cfg["mate_link_weight"],
            mate_link_symmetric=cfg["mate_link_symmetric"],
            bam_use_mates=cfg["bam_use_mates"],
            bam_pair_buffer_limit=cfg["bam_pair_buffer_limit"] )
        if A.shape != shape: raise ValueError(f"Animal {aid}: adjacency shape mismatch {A.shape} != {shape}")

        # Stream-merge edges into roaring colors (+ weight sums)
        A = A.tocoo()
        u = A.row
        v = A.col
        w = A.data.astype(np.float32, copy=False)

        # Tight loop in Python is ok; we touch each nonzero once per animal
        for uu, vv, ww in zip(u, v, w):
            key = int(uu) * n_nodes + int(vv)
            bm = colors.get(key)
            if bm is None:
                colors[key] = BitMap([aid])
                if keep_weight_sum: wsum[key] = float(ww)
            else:
                bm.add(aid)
                if keep_weight_sum: wsum[key] += float(ww)

    # Finalize to arrays (stable edge order)
    keys = np.fromiter(colors.keys(), dtype=np.int64, count=len(colors))
    keys.sort()
    rows = (keys // n_nodes).astype(np.int64)
    cols = (keys %  n_nodes).astype(np.int64)
    colsets = [colors[int(k)] for k in keys]
    wsum_arr = (np.fromiter((wsum[int(k)] for k in keys), dtype=np.float32, count=len(keys))
                if keep_weight_sum else None)

    return ColoredGraphRBM(shape, rows, cols, colsets, wsum_arr)








