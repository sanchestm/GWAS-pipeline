import os
import numpy as np
import scipy.sparse as sps
import pysam
from numba import njit
from tqdm import tqdm

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


class KmerEncoder:
    """
    Fast k-mer counter with optional reverse-complement merging.
    - Cached ASCII→base mapping (A,C,G,T -> 0,1,2,3; else -1)
    - Numba JIT kernel that updates a shared counts vector in-place
    - Simple per-file loop (fast for millions of ~300 bp reads)
    """

    def __init__(self, k: int, stride: int = 1, merge_rc: bool = False, warmup: bool = True, silent = False):
        self.k = int(k)
        self.stride = int(stride)
        self.merge_rc = bool(merge_rc)
        self.n_features = 4 ** self.k
        self.silent = silent

        # mapping table (cached once)
        mapping = np.full(256, -1, dtype=np.int8)
        mapping[ord("A")] = mapping[ord("a")] = 0
        mapping[ord("C")] = mapping[ord("c")] = 1
        mapping[ord("G")] = mapping[ord("g")] = 2
        mapping[ord("T")] = mapping[ord("t")] = 3
        self.mapping = mapping

        if warmup:
            # one-time JIT compile on a tiny dummy read (amortize compile cost)
            dummy = np.array([0, 1, 2, 3] * max(1, self.k), dtype=np.int8)
            tmp = np.zeros(self.n_features, dtype=np.uint32)
            _count_inplace(dummy, self.k, self.stride, self.merge_rc, tmp)

    # ---- internal ----
    def _encode(self, seq: str) -> np.ndarray:
        # Fast ASCII→uint8 view→int8 mapping
        return self.mapping[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

    def count(self, seq: str) -> np.ndarray:
        """
        Return a *new* dense vector of counts (allocates once).
        Prefer using sample_kmer_counts() to aggregate into a single vector.
        """
        total = np.zeros(self.n_features, dtype=np.uint32)
        _count_inplace(self._encode(seq), self.k, self.stride, self.merge_rc, total)
        return total

    def count_to_csr(self, seq: str) -> sps.csr_matrix:
        total = self.count(seq)
        nz = total.nonzero()[0]
        return sps.csr_matrix((total[nz], (np.zeros_like(nz), nz)),
                              shape=(1, self.n_features), dtype=np.uint32)

    def sample_kmer_counts(self, path: str, max_reads: int | None = None, reference_fasta: str | None = None) -> sps.csr_matrix:
        """
        Aggregate k-mers across a file (FASTA/FASTQ/BAM/CRAM/BED) into one CSR row.
        Drops unmapped/duplicate for BAM/CRAM. No batching: minimal overhead per read.
        """
        total = np.zeros(self.n_features, dtype=np.uint32)
        ext = os.path.splitext(path)[1].lower()

        # FASTA/FASTQ
        if ext in [".fa", ".fasta", ".fq", ".fastq"] or path.endswith((".fa.gz", ".fasta.gz", ".fq.gz", ".fastq.gz")):
            with pysam.FastxFile(path) as fh:
                for i, entry in tqdm(enumerate(fh),  disable = self.silent):
                    if max_reads and i >= max_reads: break
                    _count_inplace(self._encode(entry.sequence), self.k, self.stride, self.merge_rc, total)

        # BAM/CRAM
        elif ext in [".bam", ".cram"]:
            with pysam.AlignmentFile(path, "rb") as bam:
                for i, read in tqdm(enumerate(bam.fetch(until_eof=True)),  disable = self.silent):
                    if max_reads and i >= max_reads: break
                    if read.is_unmapped or read.is_duplicate: continue
                    if read.query_sequence:
                        _count_inplace(self._encode(read.query_sequence), self.k, self.stride, self.merge_rc, total)

        # BED (requires reference)
        elif ext == ".bed" or path.endswith(".bed.gz"):
            if reference_fasta is None:
                raise ValueError("BED input requires reference_fasta path.")
            ref = pysam.FastaFile(reference_fasta)
            with (pysam.TabixFile(path) if path.endswith(".gz") else open(path)) as bed:
                for i, line in tqdm(enumerate(bed),  disable = self.silent):
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


import numpy as np
import scipy.sparse as sps
from numba import njit, prange

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
