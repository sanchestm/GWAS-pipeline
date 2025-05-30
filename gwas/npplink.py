import numpy as np
import dask.array as da
import dask
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import numba
from sklearn.utils.extmath import randomized_svd
from scipy.stats import t as scipyt
from scipy.stats import chi2


# try:
#     import cupy as np
#     _ = np.cuda.runtime.getDeviceCount() > 0
#     GPU_AVAILABLE = True
#     print('running npplink on the GPU')
# except Exception:
#     import numpy as np
#     GPU_AVAILABLE = False

# Precompute the lookup table once
__genotype_map = np.array([0, np.nan, 1, 2], dtype=np.float32)
__b = np.arange(256, dtype=np.uint8)
__groups = np.stack((( __b >> 0) & 3,
                     ( __b >> 2) & 3,
                     ( __b >> 4) & 3,
                     ( __b >> 6) & 3), axis=1)
__lookuptable = __genotype_map[__groups]

def read_fam_bim(prefix):
    """
    Reads the .fam and .bim files using pandas.
    """
    fam = pd.read_csv(f'{prefix}.fam', sep='\s+', header=None, low_memory=True, engine='c',
                      names=['fid', 'iid', 'father', 'mother', 'gender', 'trait'],
                    dtype={'fid':str,'iid':str, 'father': str, 'mother': str }
                     )
    bim = pd.read_csv(f'{prefix}.bim',  sep= '\s+', header=None, low_memory=True, 
                      names=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1'], engine='c',
                      dtype={'chrom':str,'snp': str, 'cm': float,  'pos': int,'a0': str,'a1': str})
    bim['i'] = bim.index.copy()
    return fam, bim



def memmap_bed_file(bed_file):
    """
    Memory-maps the entire .bed file (including the 3-byte header).
    Returns a memmap array of type np.uint8.
    """
    return np.memmap(bed_file, mode='r', dtype=np.uint8)

def decode_bed_chunk(buffer, num_samples, variants_per_chunk, dtype=np.float32):
    """
    Decodes a chunk from a PLINK .bed file in SNP-major format.
    
    Parameters
    ----------
    buffer : 1D numpy.ndarray of np.uint8
        The raw bytes (excluding the 3 magic bytes) for a block of variants.
    num_samples : int
        Number of samples.
    variants_per_chunk : int
        Number of variants in this chunk.
    dtype : data-type, default np.float32
        Data type for the genotype map.
    
    Returns
    -------
    genotypes : numpy.ndarray of shape (num_samples, variants_per_chunk)
        Genotype codes (0, 1, 2, or np.nan).
    """
    # Use the precomputed lookup table: each byte becomes 4 genotype calls.
    decoded = __lookuptable.astype(dtype)[buffer].reshape(-1)
    
    # Each variant uses ceil(num_samples/4) bytes; each such byte gives 4 calls.
    bytes_per_variant = (num_samples + 3) // 4
    calls_per_variant = bytes_per_variant * 4
    
    # Reshape so that each row corresponds to one variant, then select only the first num_samples calls.
    decoded = decoded[: variants_per_chunk * calls_per_variant].reshape(variants_per_chunk, calls_per_variant)
    genotypes = decoded[:, :num_samples].T  # shape becomes (num_samples, variants_per_chunk)
    return genotypes

def read_bed_chunk(bed_mem, num_samples, variants_chunk_start, variants_per_chunk, dtype=np.float32):
    """
    Reads a chunk from a memory-mapped PLINK .bed file.
    
    Parameters
    ----------
    bed_mem : np.memmap
         Memory-mapped array of the entire .bed file.
    num_samples : int
         Number of samples (from the .fam file).
    variants_chunk_start : int
         Starting variant index (0-indexed) for this chunk.
    variants_per_chunk : int
         Number of variants to read in this chunk.
    dtype : data-type, default np.float32
         Data type for the genotype map.
    
    Returns
    -------
    genotypes : numpy.ndarray of shape (num_samples, variants_per_chunk)
         The decoded genotype matrix.
    """
    bytes_per_variant = (num_samples + 3) // 4
    # Skip the 3 magic bytes at the beginning; then calculate offset for this chunk.
    offset = 3 + variants_chunk_start * bytes_per_variant
    bed_size = variants_per_chunk * bytes_per_variant
    # Slice the memmapped array
    buffer_np = bed_mem[offset: offset+bed_size]
    return decode_bed_chunk(buffer_np, num_samples, variants_per_chunk, dtype=dtype)

def load_plink(prefix, chunk_variants=10000, dtype=np.float32, use_memmap=True):
    """
    Loads PLINK genotype data using Dask, with lazy computation.
    
    Parameters
    ----------
    prefix : str
         The prefix for PLINK files (.bed, .bim, .fam).
    chunk_variants : int, default 10000
         Number of variants per chunk.
    dtype : data-type, default np.float32
         Data type for the genotype matrix.
    use_memmap : bool, default True
         If True, memory-map the entire .bed file ahead of time.
    
    Returns
    -------
    bim : pandas.DataFrame
         BIM DataFrame.
    fam : pandas.DataFrame
         FAM DataFrame.
    genotype_dask : dask.array.Array
         A Dask array of shape (num_samples, num_variants) with lazy evaluation.
    """
    fam, bim = read_fam_bim(prefix)
    bed_file = f'{prefix}.bed'
    num_samples = len(fam)
    num_variants = len(bim)
    
    if use_memmap:
        bed_mem = memmap_bed_file(bed_file)
    else:
        bed_mem = None  # We'll open with normal I/O below.
    
    delayed_chunks = []
    for offset in range(0, num_variants, chunk_variants):
        variants_per_chunk = min(chunk_variants, num_variants - offset)
        if use_memmap:
            # Use the pre-created memmap
            chunk = dask.delayed(read_bed_chunk)(
                bed_mem, num_samples, offset, variants_per_chunk, dtype=dtype
            )
        else:
            # Fallback to reading from file each time.
            def read_chunk(bed_file=bed_file, offset=offset, variants_per_chunk=variants_per_chunk):
                bytes_per_variant = (num_samples + 3) // 4
                off = 3 + offset * bytes_per_variant
                bed_size = variants_per_chunk * bytes_per_variant
                with open(bed_file, 'rb') as f:
                    f.seek(off)
                    buf = f.read(bed_size)
                buffer_np = np.frombuffer(buf, dtype=np.uint8)
                return decode_bed_chunk(buffer_np, num_samples, variants_per_chunk, dtype=dtype)
            chunk = dask.delayed(read_chunk)()
        da_chunk = da.from_delayed(chunk, shape=(num_samples, variants_per_chunk), dtype=dtype)
        delayed_chunks.append(da_chunk)
    genotype_dask = da.concatenate(delayed_chunks, axis=1)
    return bim, fam, genotype_dask

def load_plink_xarray(prefix, chunk_variants=10000, dtype=np.float16, use_memmap=True):
    """
    Loads PLINK genotype data and wraps it in an xarray.DataArray for easier indexing.
    
    The xarray will have dimensions 'iid' and 'snp' with coordinates based on the .fam and .bim files.
    
    Parameters
    ----------
    prefix : str
         The prefix for PLINK files (.bed, .bim, .fam).
    chunk_variants : int, default 10000
         Number of variants per chunk.
    dtype : data-type, default np.float16
         Data type for the genotype matrix.
    use_memmap : bool, default True
         If True, memory-map the entire .bed file ahead of time.
    
    Returns
    -------
    geno_xr : xarray.DataArray
         An xarray.DataArray wrapping the Dask array with dimensions ('iid', 'snp') and associated coordinates.
    """
    import xarray as xr
    bim, fam, genotype_dask = load_plink(prefix, chunk_variants, dtype, use_memmap)
    geno_xr = xr.DataArray(
        genotype_dask,
        dims=("iid", "snp"),
        coords={
            "iid": fam['iid'],
            "snp": bim['snp'],
            "sex": ("iid", fam['gender']),
            "fid": ("iid", fam['fid']),
            "father": ("iid", fam['father']),
            "mother": ("iid", fam['mother']),
            "chrom": ("snp", bim['chrom']),
            "pos": ("snp", bim['pos']),
            "a0": ("snp", bim['a0']),
            "a1": ("snp", bim['a1']),
        },
        name="genotypes"
    )
    return geno_xr


@numba.njit(parallel=True)
def king_robust_kinship_numba(G):
    """
    Compute the KING-robust kinship matrix.
    
    For individuals i and j, let:
      - valid = indices of SNPs where both have non-missing genotypes.
      - IBS0 = count of valid SNPs where one genotype is 0 and the other is 2.
      - het_i = count of valid SNPs where i is heterozygous (==1).
      - het_j = count of valid SNPs where j is heterozygous (==1).
    
    The KING robust estimator is then:
        phi_ij = 0.5 * (1 - IBS0 / (2 * min(het_i, het_j)) )
    If the denominator is zero, phi_ij is set to np.nan.
    
    Parameters
    ----------
    G : 2D numpy.ndarray (n_individuals x m_snps)
        Genotype matrix (0,1,2) with missing values as np.nan.
        
    Returns
    -------
    kin : 2D numpy.ndarray (n_individuals x n_individuals)
        The kinship matrix.
    """
    n, m = G.shape
    kin = np.empty((n, n), dtype=np.float64)
    for i in numba.prange(n): kin[i, i] = 0.5
    for i in numba.prange(n):
        for j in range(i+1, n):
            N_valid = 0
            IBS0 = 0
            het_i = 0
            het_j = 0
            for s in range(m):
                a = G[i, s]
                b = G[j, s]
                if np.isnan(a) or np.isnan(b): continue
                N_valid += 1
                # Count heterozygotes (only among valid SNPs)
                if a == 1:het_i += 1
                if b == 1: het_j += 1
                # Opposite homozygotes: (0,2) or (2,0)
                if ((a == 0 and b == 2) or (a == 2 and b == 0)):IBS0 += 1
            if N_valid == 0: phi = np.nan
            else:
                denom = 2 * min(het_i, het_j)
                if denom == 0: phi = np.nan
                else:phi = 0.5 * (1 - IBS0/denom)
            kin[i, j] = phi
            kin[j, i] = phi
    return kin

    
def GRM(X, scale = True, return_weights= False, nan_policy = 'ignore', correlation_matrix= False):
    ##### z calculation
    x = np.array(X)
    z = x - np.nanmean(x, axis = 0)
    if scale: z /=  np.nanstd(x, axis = 0)
    np.nan_to_num(z, copy = False,  nan=0.0, posinf=None, neginf=None )
    zzt = np.dot(z,z.T)
    if nan_policy == 'mean': zzt_w = x.shape[1]-1
    #### NA adjustment
    elif nan_policy in ['ignore', 'per_iid']: 
        zna = (~np.isnan(x)).astype(np.float64)
        zzt_w = np.dot(zna,zna.T)
        zzt_w = np.clip(zzt_w-1, a_min = 1, a_max = np.inf)
        if nan_policy == 'per_iid':
            zzt_w = zzt_w.max(axis =1)[:, None]
    grm = zzt/zzt_w
    if correlation_matrix:
        sig = np.sqrt(np.diag(grm))
        grm /= np.outer(sig, sig)
        np.fill_diagonal(grm, 1)
    if return_weights: 
        return {'zzt': zzt, 'weights': zzt_w, 'grm': grm }
    else:
        return grm


def king_robust_kinship(G):
    """
    Compute KING-robust kinship matrix using GPU (CuPy).
    
    Parameters
    ----------
    G : cupy.ndarray, shape (n_individuals, m_snps)
        Genotype matrix with values 0, 1, or 2, and missing as np.nan.
        
    Returns
    -------
    kin : cupy.ndarray, shape (n_individuals, n_individuals)
        KING robust kinship estimates.
    """
    n, m = G.shape
    # Create valid mask: 1 where genotype is not nan, 0 otherwise.
    M = (~np.isnan(G))
    
    # Create indicator matrices for genotypes 0, 1, 2
    A = ((G == 0) & (~np.isnan(G)))
    B = ((G == 2) & (~np.isnan(G)))
    H = ((G == 1) & (~np.isnan(G)))  # heterozygotes
    
    # Compute IBS0 counts:
    # For each pair, IBS0 = dot( A[i], B[j] ) + dot( B[i], A[j] )
    IBS0 = np.dot(A, B.T) + np.dot(B, A.T)
    
    # For each pair, compute heterozygote counts for i (restricted to SNPs where j is non-missing)
    het_i = np.dot(H, M.T)
    het_j = np.dot(M, H.T)
    
    # Robust denominator: 2 * min(het_i, het_j)
    denom = 2 * np.minimum(het_i, het_j)
    # Avoid division by zero: set denominator to nan where it is zero.
    denom[denom == 0] = np.nan
    
    kin = 0.5 * (1 - IBS0/denom)
    
    # Set diagonal to 0.5
    np.fill_diagonal(kin, 0.5)
    return kin

def plink2df_old(plinkpath: str, rfids: list = None, c: int = None, 
             pos_start: int = None, pos_end: int = None, 
             snplist: list = None) -> pd.DataFrame:
    if type(plinkpath) == str: 
        snps, iid, gen = load_plink(plinkpath)
    else: snps, iid, gen = plinkpath
    try:
        snps.chrom = snps.chrom.astype(int)
        if c is not None: c = int(c)
    except: 
        c = str(c)
        print('non numeric chromosomes: using str(c) in this case')
        snps = snps.sort_values(['chrom', 'pos'])
    iiid = iid.assign(i = iid.index).set_index('iid')
    if snplist is None:
        snps.pos = snps.pos.astype(int)
        isnps = snps.set_index(['chrom', 'pos'])
        if (pos_start is None) and (pos_end is None):
            if c is None: index = isnps
            else: index = isnps.loc[(slice(c, c)), :]
        else:index = isnps.loc[(slice(c, c),slice(pos_start, pos_end) ), :]
    else: 
        index = snps[snps.snp.isin(snplist)]
    col = iiid if rfids is None else iiid.loc[rfids]
    return pd.DataFrame(gen[col.i ][:,  index.i.values].astype(np.float32), 
                        index = col.index.values.astype(str), columns = index.snp.values )


def plink2df(plinkpath: str, rfids: list = None, c: int = None, 
             pos_start: int = None, pos_end: int = None, 
             snplist: list = None) -> pd.DataFrame:
    if type(plinkpath) == str: 
        snps, iid, gen = load_plink(plinkpath)
    else: snps, iid, gen = plinkpath
    try:
        snps.chrom = snps.chrom.astype(int)
        if c is not None: c = int(c)
    except: 
        c = str(c)
        print('non numeric chromosomes: using str(c) in this case')
    iiid = iid.assign(i = iid.index).set_index('iid')
    query_sentence = []
    if c is not None: query_sentence += ['chrom == @c']
    if (pos_start is not None) and (pos_end is not None): query_sentence += ['pos.between(@pos_start,@pos_end )']
    elif pos_start is not None: query_sentence += ['pos >= @pos_start']
    elif pos_end is not None: query_sentence += ['pos <= @pos_end']
    if snplist is not None: query_sentence += ['snp.isin(@snplist)']
    query_sentence = ' and '.join(query_sentence)
    sset = snps.query(query_sentence) if len(query_sentence) else snps
    col = iiid if rfids is None else iiid.loc[rfids]
    return pd.DataFrame(gen[col.i.values ][:,  sset.i.values].astype(np.float32), 
                        index = col.index.values, columns = sset.snp.values )


def whiten_data(U, D_inv_sqrt, X):
    d = np.diag(D_inv_sqrt)
    temp = np.einsum('ij,jk->ik', U.T, X,optimize=True)
    temp = d[:, None] * temp
    result = np.einsum('ij,jk->ik', U, temp, optimize=True)
    return result

def subblock_svd_from_full(U, s, obs):
    U, s, _ = randomized_svd(U[obs, :] * np.sqrt(s)[None, :]  ,  n_components=len(obs))
    return U, s

def _read_bin_url(path):
    import urllib
    if path.startswith(('http://', 'https://')):
        with urllib.request.urlopen(path) as resp: raw = resp.read()
    else:
        with open(path, 'rb') as f: raw = f.read()
    return raw

def read_grm(path):
    ids = pd.read_csv(f"{path}.id",  sep=r'\s+', header=None, 
                      names=['fid','iid'], dtype={'fid': str, 'iid': str})
    n = len(ids)
    n_elements = (n * (n + 1)) // 2
    grm_values = np.frombuffer(_read_bin_url(f"{path}.bin"), dtype=np.float32, count=n_elements)
    grm = np.zeros((n, n), dtype=np.float64)
    grm[np.tril_indices(n)] = grm_values
    grm += np.tril(grm,-1).T
    grm = xr.DataArray(grm, dims=["sample_0", "sample_1"], 
                 coords={"sample_0":ids.iid, "sample_1": ids.iid, 
                         'iid': ('sample_1', ids.iid), 'fid': ('sample_1', ids.fid)})
    w = np.zeros_like(grm, dtype=np.float64) 
    w[np.tril_indices(n)] = np.frombuffer(_read_bin_url(f"{path}.N.bin"), dtype=np.float32, count=n_elements)
    w += np.tril(w, -1).T
    w = xr.DataArray(w, dims=["sample_0", "sample_1"], coords={"sample_0":grm.sample_0, "sample_1": grm.sample_1} )
    return {'grm':grm,'w': w, 'path': path}

def load_all_grms(paths):
    from glob import glob
    if isinstance(paths, str): paths = glob(paths)
    allgrms = pd.DataFrame.from_records([read_grm(x.replace('.N', '').replace('.grm.bin', '.grm')) for x in paths])
    #allgrms = pd.DataFrame.from_records([read_grm_w(x) for x in glob(r'grm/*.grm.bin')])
    allgrms['path'] = allgrms['path'].str.extract(r'([\d\w]+)chrGRM.')
    allgrms['isnum'] = allgrms['path'].str.isnumeric()
    allgrms = allgrms.set_index('path')
    allgrms_weighted = allgrms.loc["All", 'grm']*allgrms.loc["All", 'w']
    allgrms['subtracted_grm'] = allgrms.progress_apply(lambda x: (allgrms.loc["All", 'grm'] if not x.isnum else 
                                                             (allgrms_weighted - x.grm*x.w)/(allgrms.loc["All", 'w']-x.w)).to_pandas(),
                                                   axis = 1)
    allgrms[['U', 's']] = allgrms.progress_apply(lambda x: randomized_svd(x.subtracted_grm.values, n_components=2000, random_state=0)[:2], axis=1,  result_type="expand")
    allgrms = allgrms.sort_index(key = lambda idx : idx.str.lower().map({str(i): int(i) for i in range(1000)}|\
                              {i: int(i) for i in range(1000)}|\
                              {'all': -1000, 'x':1001, 'y' : 1002, 'mt': 1003, 'm': 1003}))
    return allgrms


def H2SVD(y, grm=None,s=None, U=None, l='REML', n_components = None, return_SVD = False, tol = 1e-8):
    from scipy.optimize import minimize
    from scipy.linalg import solve_triangular
    y = np.array(y).flatten()
    notnan = ~np.isnan(y)
    obs = np.where(notnan)[0]
    sp, m, N = np.nanvar(y, ddof=1), np.nanmean(y), notnan.sum()
    y = y[obs]
    y -= m
    if s is None and U is None and grm is not None:
        grm = np.array(grm)
        grm = grm[np.ix_(obs, obs)]
        if n_components is None: n_components =grm.shape[0]
        U, s, _ = randomized_svd(grm, n_components=n_components, random_state=0)
    elif s is not None and U is not None and grm is None: 
        U = U[obs, :]
        s *= np.sum(U**2, axis=0) #* s
        if n_components is not None:
            U = U[:, :n_components]
            s = s[:n_components] 
        #U, s = subblock_svd_from_full(U, s, obs)
    else: raise ValueError('cannot submit both grm and U s at the same time')
    Ur2 = np.dot(U.T, y)**2
    s = np.maximum(s, tol)        
    def _L(h2_arr):
        h2 = h2_arr[0]
        sg = h2 * sp
        se = sp - sg 
        sgsse = sg * s + se
        log_det = np.sum(np.log(sgsse))
        quad = np.sum(Ur2/sgsse)
        log_likelihood = -0.5 * (quad + log_det + N * np.log(2 * np.pi))
        if l == 'REML':
            Xt_cov_inv_X = np.sum(1.0 / sgsse)
            log_likelihood -= 0.5 * np.log(Xt_cov_inv_X)
        return -log_likelihood
        
    result = minimize(fun=_L, x0=[0.5], bounds=[(0., 1.)])
    h2, likelihood = result.x[0], result.fun
    if return_SVD: 
        sg, se = h2 * sp, sp*(1- h2)
        sgsse = sg * s + se
        log_det = np.sum(np.log(sgsse))
        quad = np.sum(Ur2/sgsse)
        eps = 1e-5
        f0, fph, fmh = _L([h2]),  _L([min(h2 + eps, 1.0)]),  _L([max(h2 - eps, 0.0)])
        second_deriv = (fph - 2*f0 + fmh) / eps**2
        se_hat = np.sqrt(1.0 / second_deriv)
        return {'U':U, 's': s, 'h2': result.x[0], 'L': likelihood,
                'quad':quad, 'log_det': log_det, 'se': se_hat, 
               'pval' : 1 - chi2.cdf(2 * (_L([0.0]) - likelihood), df=1)}
    return h2

def remove_relatedness_transformation(G=None, U=None, s=None, h2=0.5, yvar=1, tol=1e-8, random_state=None, n_components=None, return_eigen = False, **kws):
    if s is None and U is None and G is not None:
        G = yvar * (h2 * G + (1 - h2) * np.eye(G.shape[0]))
        if n_components is None: n_components = G.shape[0]
        U, s, _ = randomized_svd(G, n_components=n_components, random_state=0)
    elif s is not None and U is not None and G is None: pass
    else: raise ValueError('cannot submit both G and U s at the same time')
    eigs_fG = yvar * (h2 * s + (1 - h2))
    eigs_fG[eigs_fG < tol] = tol
    D_inv_sqrt = np.diag(1 / np.sqrt(eigs_fG))
    if return_eigen: return U, D_inv_sqrt
    W = U @ D_inv_sqrt @ U.T
    return W
    
def rm_relatedness(c, trait, df, n_components = None,return_eigen=True, svd_input = True, ):
    grm_c = allgrms.loc[str(c),'subtracted_grm']
    trait_ = df.loc[grm_c.index, trait]
    navals = ~trait_.isna()
    yvar = np.nanvar(trait_)
    if navals.sum()< 10: 
        print('not enough samples')
        return None
    if svd_input:
        h2_res = H2SVD(y = trait_, 
                   s = allgrms.loc[str(c),'s'], 
                   U = allgrms.loc[str(c),'U'], #grm = grm_c, 
                   return_SVD=True, 
                   n_components = n_components) 
    else: 
        h2_res = H2SVD(y = trait_, 
                   grm = grm_c, 
                   return_SVD=True, 
                   n_components = n_components) 
    U, D_inv_sqrt = remove_relatedness_transformation(yvar= yvar,**h2_res, return_eigen=True)
    if U.shape[0] == sum(navals): 
        trait_vec = trait_.loc[navals]#.to_frame()
        idx = navals[navals].index
    else: 
        trait_vec = trait_.copy()
        idx = navals.index
    transformed = U @ (D_inv_sqrt @ (U.T @ trait_vec))
    #transformed = whiten_data(U,D_inv_sqrt,trait_vec)
    if return_eigen:
        return {'transformed':pd.Series(transformed, index = idx, name = f'{trait}__subtractgrm{c}'),
                'U': U,
                'D_inv_sqrt': D_inv_sqrt,
                'c': c,
                'trait': trait,
                'h2': h2_res['h2'],
                'n_components':n_components}
    return pd.Series(transformed, index = idx, name = f'{trait}__subtractgrm{c}')

def scale_with_mask(X):
    X = np.asarray(X, dtype=np.float32)
    M = ~np.isnan(X)
    X_centered = X - np.nanmean(X, axis=0)
    X_centered[~M] = 0.0
    sum_sq = np.einsum("ij,ij->j", X_centered, X_centered, optimize=True)
    std = np.sqrt(sum_sq / M.sum(axis=0))
    std[std == 0] = np.nan
    X_scaled = X_centered / std
    return X_scaled, std, M.astype(np.float32)

def regression_with_einsum(ssnps, straits, snps_mask, traits_mask,dof='correct'):
    # Compute cross-product between SNPs and traits.
    XtY = np.einsum("ij,ik->jk", ssnps, straits, optimize=True)  # shape: (num_snps, num_traits)
    # Compute sum of squares of SNP values (over individuals where the trait is observed).
    diag_XtX = np.einsum("ij,ik->jk", ssnps**2, traits_mask, optimize=True)
    # Compute sum of squares of trait values over overlapping samples.
    term1 = np.einsum("ij,ik->jk", snps_mask, traits_mask * (straits**2), optimize=True)
    # Compute the number of overlapping (non-missing) individuals for each SNP–trait pair.
    if dof!='incorrect': df = np.einsum("ij,ik->jk", snps_mask, traits_mask, optimize=True) - 1
    else:  df = np.broadcast_to(traits_mask.sum(axis=0) - 1 , (ssnps.shape[1], traits_mask.shape[1])).copy()
    df[df <= 0] = np.nan 
    beta = XtY / diag_XtX
    SSR = term1 - 2 * beta * XtY + beta**2 * diag_XtX
    sigma2 = SSR / df
    se_beta = np.sqrt(sigma2 / diag_XtX)
    t_stats = beta / se_beta
    p_values = 2 * (1 - scipyt.cdf(np.abs(t_stats), df=df))
    neg_log10_p_values = -np.log10(p_values)
    return beta, se_beta, t_stats, neg_log10_p_values, df

def GWA(traits, snps, dtype = 'pandas'):
    ssnps, snps_std, snps_mask = scale_with_mask(snps)
    straits, traits_std, traits_mask = scale_with_mask(traits)
    res = xr.DataArray(
             np.stack(regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof = 'correct'), axis=0), 
             dims=["metric", "snp", "trait"],
             coords={"metric": np.array(['beta', 'beta_se', 't_stat', 'neglog_p', 'dof']),
                     "snp":   list(snps.columns),
                     "trait": traits.columns.map(lambda x: x.split('__subtractgrm')[0]).to_list()} )
    if dtype == 'pandas':
        return res.to_dataset(dim="metric").to_dataframe().reset_index()
    else: return res

