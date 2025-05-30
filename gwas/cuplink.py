import cupy as cp
import cudf
import dask.array as da
import dask
import dask_cudf
import kvikio
import xarray as xr
import pysam
import numpy as np
import cupyx.scipy.sparse as cusparse
import cupy.linalg as cplinalg

__genotype_map = cp.array([0, cp.nan, 1, 2], dtype=cp.float16)
__b = cp.arange(256, dtype=cp.uint8)
__groups = cp.stack((( __b >> 0) & 3,
                     ( __b >> 2) & 3,
                     ( __b >> 4) & 3,
                     ( __b >> 6) & 3), axis=1)
__lookuptable = __genotype_map[__groups]

def read_fam_bim(prefix):
    fam = cudf.read_csv(f'{prefix}.fam',sep =' ', header=None,
                        names=['fid','iid','father','mother','gender','trait'],
                       dtype={'fid':str,'iid':str, 'father': str, 'mother': str })
    bim = cudf.read_csv(f'{prefix}.bim', sep ='\t',header=None,
                        names=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1'], 
                        dtype={'chrom':str,'snp': str, 'cm': float,  'pos': int,'a0': str,'a1': str})
    bim['i'] = bim.index.copy()
    return fam, bim


def decode_bed_chunk(buffer, num_samples, variants_per_chunk, dtype=cp.float16):
    decoded = __lookuptable[buffer].reshape(-1)
    bytes_per_variant = (num_samples + 3) // 4
    calls_per_variant = bytes_per_variant * 4 
    return decoded[: variants_per_chunk * calls_per_variant].reshape(variants_per_chunk, calls_per_variant)[:, :num_samples].T

def read_bed_chunk(bed_file, num_samples, variants_chunk_start, variants_per_chunk, dtype = cp.float16):
    bytes_per_variant = (num_samples + 3) // 4
    offset = 3 + variants_chunk_start * bytes_per_variant
    bed_size = variants_per_chunk * bytes_per_variant
    buffer = cp.empty(bed_size, dtype=cp.uint8)
    with kvikio.CuFile(bed_file, 'r') as f: f.read(buffer, file_offset=offset)
    return decode_bed_chunk(buffer, num_samples, variants_per_chunk,dtype =dtype)

def load_plink(prefix, chunk_variants=100000, dtype = cp.float16):
    fam, bim = read_fam_bim(prefix)
    bed_file = f'{prefix}.bed'
    num_samples = len(fam)
    num_variants = len(bim)
    delayed_chunks = []
    for offset in range(0, num_variants, chunk_variants):
        variants_per_chunk = min(chunk_variants, num_variants - offset)
        chunk = dask.delayed(read_bed_chunk)(bed_file, num_samples, offset, variants_per_chunk, dtype = dtype)
        da_chunk = da.from_delayed(chunk, shape=(num_samples, variants_per_chunk), dtype=dtype)
        delayed_chunks.append(da_chunk)
    genotype_dask = da.concatenate(delayed_chunks, axis=1)
    return bim, fam, genotype_dask

def load_plink_xarray(prefix, chunk_variants=100000, dtype=cp.float16):
    """
    Loads PLINK genotype data and wraps it in an xarray.DataArray for easier indexing.
    
    The xarray will have dimensions 'samples' and 'variants' with coordinates based
    on the .fam and .bim files.
    
    Parameters
    ----------
    prefix : str
         The prefix for PLINK files (.bed, .bim, .fam).
    chunk_variants : int, default 10000
         Number of variants per chunk.
    dtype : data-type, default np.float16
         Data type for the genotype matrix.
    use_memmap : bool, default False
         If True, uses memory mapping in reading the .bed file.
    
    Returns
    -------
    bim : pandas.DataFrame
         BIM DataFrame.
    fam : pandas.DataFrame
         FAM DataFrame.
    geno_xr : xarray.DataArray
         An xarray DataArray wrapping the Dask array with dimensions 'samples' and 'variants'
         and associated coordinates.
    """
    
    bim, fam, genotype_dask = load_plink('genotypes/genotypes')
    geno_xr = xr.DataArray(
        genotype_dask,dims=("iid", "snp"),
        coords={"iid": fam['iid'],
            "snp": bim['snp'],
            "sex": ("iid", fam['gender']),
            "fid": ("iid", fam['gender']),
            "father": ("iid", fam['father']),
            "mother": ("iid", fam['mother']),
            "chrom": ("snp", bim['chrom']),
            "pos": ("snp", bim['pos']),
            "a0": ("snp", bim['a0']),
            "a1": ("snp", bim['a1']),},
        name="genotypes")
    return geno_xr

def read_grm(path):
    bin_path = f'{path}.grm.bin'
    id_path = f'{path}.grm.id'
    ids = cudf.read_csv(id_path, sep=r'\t', header=None, names=['fid', 'iid'])
    n = ids.shape[0]
    n_elements = (n * (n + 1)) // 2
    bin_data = kvikio.read(bin_path)
    grm_values = cp.frombuffer(bin_data, dtype=cp.float16, count=n_elements)
    grm_matrix = cp.zeros((ids.shape[0], ids.shape[0]), dtype=cp.float16)
    triu_indices = cp.tril_indices(ids.shape[0])
    grm_matrix = cp.zeros((ids.shape[0], ids.shape[0]), dtype=cp.float16)
    grm_matrix[triu_indices] = grm_values
    grm_matrix = grm_matrix + grm_matrix.T - cp.diag(cp.diag(grm_matrix))
    return { 'grm': grm_matrix,'ids': ids}


def process_region(bam_file: str, region: str) -> cudf.DataFrame:
    """
    Process a single region (chromosome) from the BAM file.
    Opens the file using kvikio and parses alignments from the specified region.
    Returns a cudf.DataFrame containing selected alignment fields.
    
    Note: The BAM file must be indexed.
    """
    # Open the file via kvikio (which may offer GPU-direct I/O)
    with kvikio.open(bam_file, mode="rb") as f:
        # Pass the kvikio file handle to pysam
        with pysam.AlignmentFile(f, "rb") as bam:
            rows = []
            # Iterate over alignments in the given region
            for aln in bam.fetch(region=region):
                rows.append({
                    "qname": aln.query_name,
                    "flag": aln.flag,
                    "rname": bam.get_reference_name(aln.reference_id) if aln.reference_id != -1 else None,
                    "pos": aln.reference_start,
                    "mapq": aln.mapping_quality,
                    "cigar": aln.cigarstring
                })
            # Create a cudf DataFrame from the list of dictionaries
            return cudf.DataFrame(rows)

def read_bam_to_dask_cudf(bam_file: str, regions: list = None) -> dask_cudf.DataFrame:
    """
    Reads a BAM file into a lazy dask_cudf DataFrame on the GPU.
    
    Parameters:
      bam_file: Path to the BAM file (should be indexed).
      regions: List of regions (e.g., chromosomes) to partition the file.
               If None, all references in the BAM header are used.
    
    Returns:
      A lazy dask_cudf.DataFrame whose partitions are processed via dask.delayed.
    """
    # Open the BAM file once (via kvikio) to retrieve the list of references if needed.
    if regions is None:
        with kvikio.open(bam_file, mode="rb") as f:
            with pysam.AlignmentFile(f, "rb") as bam:
                regions = list(bam.references)
    
    # Create a delayed task for each region.
    delayed_parts = [dask.delayed(process_region)(bam_file, region) for region in regions]
    
    # Build a lazy dask_cudf DataFrame from the delayed partitions.
    ddf = dask_cudf.from_delayed(delayed_parts)
    return ddf

def load_sparse_matrix_from_files(prefix):
    """
    Load a sparse CSR matrix from files saved as NumPy arrays.
    Assumes files: <prefix>_data.npy, <prefix>_indices.npy, <prefix>_indptr.npy, <prefix>_shape.npy.
    The arrays are memory-mapped into GPU memory via cupy.
    """
    # Cupy’s load supports mmap_mode much like NumPy.
    # (Files must have been saved in a way that supports memory mapping.)
    data = cp.load(f"{prefix}_data.npy", mmap_mode='r')
    indices = cp.load(f"{prefix}_indices.npy", mmap_mode='r')
    indptr = cp.load(f"{prefix}_indptr.npy", mmap_mode='r')
    # Load shape using NumPy (shape is small) and then convert to tuple.
    shape = tuple(cp.load(f"{prefix}_shape.npy", mmap_mode='r'))
    # Create the CSR sparse matrix.
    A = cusparse.csr_matrix((data, indices, indptr), shape=shape)
    return A



def randomized_svd_sparse(A, k, n_iter=2, random_state=1234):
    """
    Perform a two-sided randomized SVD on a sparse matrix A.
    Parameters:
      A : cupyx.scipy.sparse.csr_matrix
          Input sparse matrix (m x n) in CSR format.
      k : int
          Target rank.
      n_iter : int, optional
          Number of power iterations (to improve the subspace approximation).
      random_state : int, optional
          Seed for the random number generator.
    Returns:
      U : cupy.ndarray
          Approximate left singular vectors (m x k).
      S : cupy.ndarray
          Approximate singular values (k,).
      V : cupy.ndarray
          Approximate right singular vectors (n x k).
    """
    cp.random.seed(random_state)
    m, n = A.shape
    # Generate two independent random Gaussian matrices.
    Omega = cp.random.standard_normal((n, k), dtype=cp.float16)
    Psi   = cp.random.standard_normal((m, k), dtype=cp.float16)
    # Form the sketches: Y = A * Omega and Z = A.T * Psi.
    Y = A @ Omega         # shape (m, k)
    Z = A.transpose() @ Psi  # shape (n, k)
    # Perform power iterations to improve accuracy.
    for _ in range(n_iter):
        Y = A @ (A.transpose() @ Y)
        Z = A.transpose() @ (A @ Z)
    # Compute the (thin) QR factorizations.
    Q_y, _ = cp.linalg.qr(Y, mode='reduced')
    Q_z, _ = cp.linalg.qr(Z, mode='reduced')
    # Form the small matrix B = Q_y^T * A * Q_z.
    B = Q_y.T @ (A @ Q_z)
    # Compute the SVD of the small dense matrix.
    U_small, S, VT_small = cp.linalg.svd(B, full_matrices=False)
    # Recover the approximate singular vectors of A.
    U = Q_y @ U_small
    V = Q_z @ VT_small.T
    return U, S, V


def randomized_svd_sparse_wrapper(k, use_dask=False, matrix_file_prefix=None,
                                  m=1000, n=800, density=0.01, n_iter=2, random_state=1234):
    """
    Wrapper to perform randomized SVD on a sparse matrix.
    If matrix_file_prefix is provided, the CSR matrix is loaded via memmap.
    Otherwise, a random sparse matrix is generated.
    If use_dask is True, the computation is wrapped in a Dask delayed object for lazy evaluation.
    Parameters:
      k : int
          Target rank.
      use_dask : bool, optional
          If True, wraps the computation as a lazy Dask task.
      matrix_file_prefix : str or None, optional
          Prefix for the memmap files (if available).
      m, n : int, optional
          Dimensions for the generated matrix if not loaded from file.
      density : float, optional
          Density for the random sparse matrix (if generated).
      n_iter : int, optional
          Number of power iterations.
      random_state : int, optional
          Random seed.
    Returns:
      If use_dask is False: (U, S, V) as cupy.ndarrays.
      If use_dask is True: a Dask delayed object which computes (U, S, V) upon calling .compute()
    """
    if matrix_file_prefix is not None:
        # Load from file via memory mapping.
        A = load_sparse_matrix_from_files(matrix_file_prefix)
    else:
        # Generate a random sparse matrix in CSR format using CuPy.
        A = cusparse.random(m, n, density=density, format='csr', dtype=cp.float16)
    if use_dask:
        # Lazy evaluation using Dask.
        from dask import delayed
        lazy_svd = delayed(randomized_svd_sparse)(A, k, n_iter, random_state)
        return lazy_svd
    else:
        # Immediate computation.
        return randomized_svd_sparse(A, k, n_iter, random_state)

def R2(X, Y= None, return_named = True, return_square = True):
    x =cp.array(X).astype(cp.float32)
    xna = (~cp.isnan(x)).astype(cp.float32) ##get all nas
    xnaax0 = xna.sum(axis = 0)
    x -= (cp.nansum(x, axis = 0)/xnaax0) #subtract mean
    cp.nan_to_num(x, copy=False, nan=0.0, posinf=None, neginf=None)
    xstd = cp.sqrt(cp.sum(x**2, axis = 0)/xnaax0) #estimate std
    xstd[xstd == 0] = cp.nan
    if Y is None:  y, yna, ystd = x, xna, xstd 
    else:
        y =cp.array(Y).astype(cp.float32)
        yna = (~cp.isnan(y)).astype(cp.float32) ##get all nas
        ynaax0 = yna.sum(axis = 0)
        y -= (cp.nansum(y, axis = 0)/ynaax0) #subtract mean
        cp.nan_to_num(y, copy = False,  nan=0.0, posinf=None, neginf=None ) ### will not affect sums 
        ystd = cp.sqrt(cp.sum(y**2, axis = 0)/ynaax0) #estimate std
        ystd[ystd == 0] = cp.nan
    xty_w = cp.dot(xna.T,yna)
    xty_w[xty_w == 0] = cp.nan
    cov = cp.dot(x.T,y) / xty_w
    res = cp.clip(cp.power(cov/cp.outer(xstd, ystd), 2), \
                  a_min = 0, a_max = 1)
    rindex = X.columns if isinstance(X, cudf.DataFrame) else list(range(x.shape[1]))
    if (Y is None) and isinstance(X, cudf.DataFrame): rcolumns = X.columns
    elif isinstance(Y, cudf.DataFrame): rcolumns = Y.columns
    else: rcolumns = list(range(y.shape[1]))
    if return_named: 
        res = cudf.DataFrame(res, index = rindex, columns = rcolumns)  
        if not return_square:
            res = res.reset_index(names = 'bp1').melt(id_vars = 'bp1', var_name='bp2')
            chrom = res['bp1'].iloc[0].split(':')[0]
            pos = len(chrom)+ 1
            res['c'] = chrom
            res['distance'] = (res.bp1.str.slice(start=pos).astype(int)  - res.bp2.str.slice(start=pos).astype(int)).abs()
    return res


def king_robust_kinship(G):
    """
    Compute KING-robust kinship matrix using GPU (CuPy).
    
    Parameters
    ----------
    G : cupy.ndarray, shape (n_individuals, m_snps)
        Genotype matrix with values 0, 1, or 2, and missing as cp.nan.
        
    Returns
    -------
    kin : cupy.ndarray, shape (n_individuals, n_individuals)
        KING robust kinship estimates.
    """
    n, m = G.shape
    # Create valid mask: 1 where genotype is not nan, 0 otherwise.
    M = (~cp.isnan(G)).astype(cp.float16)
    
    # Create indicator matrices for genotypes 0, 1, 2
    A = ((G == 0) & (~cp.isnan(G))).astype(cp.float16)
    B = ((G == 2) & (~cp.isnan(G))).astype(cp.float16)
    H = ((G == 1) & (~cp.isnan(G))).astype(cp.float16)  # heterozygotes
    
    # Compute IBS0 counts:
    # For each pair, IBS0 = dot( A[i], B[j] ) + dot( B[i], A[j] )
    IBS0 = cp.dot(A, B.T) + cp.dot(B, A.T)
    
    # For each pair, compute heterozygote counts for i (restricted to SNPs where j is non-missing)
    het_i = cp.dot(H, M.T)
    het_j = cp.dot(M, H.T)
    
    # Robust denominator: 2 * min(het_i, het_j)
    denom = 2 * cp.minimum(het_i, het_j)
    # Avoid division by zero: set denominator to nan where it is zero.
    denom[denom == 0] = cp.nan
    
    kin = 0.5 * (1 - IBS0/denom)
    
    # Set diagonal to 0.5
    cp.fill_diagonal(kin, 0.5)
    
    return kin




def king_robust_kinship_dask(G_dask):
    """
    Compute the KING-robust kinship matrix using a dask array with GPU (CuPy) blocks.
    
    Parameters
    ----------
    G_dask : dask.array.Array
        Genotype matrix of shape (n_individuals, m_snps), with values 0, 1, or 2 
        and missing values as cp.nan. It is assumed that the underlying blocks are 
        CuPy arrays (e.g. obtained via dask_cudf or by converting from a cupy array).
        
    Returns
    -------
    kin : dask.array.Array
        A dask array representing the kinship matrix of shape (n_individuals, n_individuals)
        computed with the robust KING estimator.
    """
    # Create a valid mask: 1 if genotype is not nan, else 0.
    M = G_dask.map_blocks(lambda x: (~cp.isnan(x)).astype(cp.float16), dtype=cp.float16)
    
    # Create indicator arrays for genotype 0, 1, and 2.
    A = G_dask.map_blocks(lambda x: (((x == 0) & (~cp.isnan(x))).astype(cp.float16)), dtype=cp.float16)
    H = G_dask.map_blocks(lambda x: (((x == 1) & (~cp.isnan(x))).astype(cp.float16)), dtype=cp.float16)
    B = G_dask.map_blocks(lambda x: (((x == 2) & (~cp.isnan(x))).astype(cp.float16)), dtype=cp.float16)
    
    # Compute pairwise IBS0 counts: for each pair (i,j):
    # IBS0 = dot( A[i], B[j] ) + dot( B[i], A[j] )
    IBS0 = da.dot(A, B.T) + da.dot(B, A.T)
    
    # Compute heterozygote counts for each pair.
    # For individual i (restricted to SNPs where j is valid) and vice versa.
    het_i = da.dot(H, M.T)  # (n_ind, n_ind)
    het_j = da.dot(M, H.T)
    
    # Robust denominator: 2 * min(het_i, het_j)
    denom = 2 * da.minimum(het_i, het_j)
    # Replace zeros in denom with cp.nan to avoid division by zero.
    denom = denom.map_blocks(lambda x: cp.where(x==0, cp.nan, x), dtype=cp.float16)
    
    # Compute kinship: phi = 0.5 * (1 - IBS0 / denom)
    kin = 0.5 * (1 - IBS0 / denom)
    
    # Set the diagonal to 0.5.
    def fill_diag(x):
        cp.fill_diagonal(x, 0.5)
        return x
    kin = kin.map_blocks(fill_diag, dtype=cp.float16)
    
    return kin


def plink2df(plinkpath, rfids=None, c=None, pos_start=None, pos_end=None, snplist=None):
    if isinstance(plinkpath, str):
        snps, iid, gen = load_plink(plinkpath)
    else:
        snps, iid, gen = plinkpath

    # Ensure chromosome column is consistent
    try:
        snps.chrom = snps.chrom.astype(int)
        if c is not None:  c = int(c)
    except:
        c = str(c)
        print('non-numeric chromosomes: using string comparison')

    # Prepare sample metadata
    iid = iid.reset_index(drop=True)
    iiid = iid.assign(i=cp.arange(len(iid))).set_index('iid')
    query_sentence = []
    if c is not None: query_sentence += ['chrom == @c']
    if (pos_start is not None) and (pos_end is not None): query_sentence += ['pos.between(@pos_start,@pos_end )']
    elif pos_start is not None: query_sentence += ['pos >= @pos_start']
    elif pos_end is not None: query_sentence += ['pos <= @pos_end']
    if snplist is not None: query_sentence += ['snp.isin(@snplist)']
    query_sentence = ' and '.join(query_sentence)
    sset = snps.query(query_sentence) if len(query_sentence) else snps
    col = iiid if rfids is None else iiid.loc[rfids]
    # Create a GPU dataframe from the CuPy array
    return cudf.DataFrame(gen[col.i.values ][:,  sset.i.values].compute().astype(cp.float32).get(),
                          index=cudf.Series(col.index),
                          columns=cudf.Series(sset.snp))


