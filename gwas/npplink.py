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
from scipy.special import stdtr, erfc
from tqdm import tqdm
import itertools
from typing import Literal
from scipy.linalg import blas
import os
from collections import defaultdict


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

def GRM(X, scale = True, return_weights= False, nan_policy = 'ignore', correlation_matrix= False, savefile = None):
    ##### z calculation
    ids = X.index if isinstance(X, pd.DataFrame) else None
    toxr = lambda x,ids: xr.DataArray(x, dims=["sample_0", "sample_1"], coords={"sample_0":ids, "sample_1":ids})
    x = np.array(X)
    z = x - np.nanmean(x, axis = 0)
    if scale: z /= np.nanstd(x, axis = 0)
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
    if savefile is not None:
        if ids is not None:
            ids.to_frame().set_axis(['idx'], axis = 1).reset_index()\
               .to_csv(f"{savefile}.grm.id", index = False, header = None, sep = '\t')
        idxs = np.tril_indices_from(grm)
        grm[idxs].astype(np.float32).tofile(f"{savefile}.grm.bin") 
        zzt_w[idxs].astype(np.float32).tofile(f"{savefile}.grm.N.bin")
    if correlation_matrix:
        sig = np.sqrt(np.diag(grm))
        grm /= np.outer(sig, sig)
        np.fill_diagonal(grm, 1)
    if ids is not None:
        if not return_weights: return toxr(grm,ids)
        else: return {'zzt':toxr(zzt,ids),'w':toxr(zzt_w,ids),'grm':toxr(grm,ids)}
    if return_weights: return {'zzt': zzt, 'w': zzt_w, 'grm': grm }
    else: return grm


def GRM_lowmem(X,*, scale: bool = True, return_weights: bool = False, nan_policy: str = "ignore", correlation_matrix: bool = False, dtype=np.float32,  savefile = None):
    ids = X.index if isinstance(X, pd.DataFrame) else None
    toxr = lambda x,ids: xr.DataArray(x, dims=["sample_0", "sample_1"], coords={"sample_0":ids, "sample_1":ids})
    syrk = blas.ssyrk if np.dtype(dtype) == np.float32 else blas.dsyrk
    x = np.array(X)# X = np.asarray(X, dtype=dtype, order="C")
    m, n = X.shape
    mask = ~np.isnan(X)
    counts = mask.sum(0)
    sums   = np.where(mask, X, 0).sum(0)
    means  = sums / np.maximum(counts, 1)
    Z = np.where(mask, X - means, 0)
    if scale:
        ssq = (Z**2).sum(0)
        std = np.sqrt(ssq / np.maximum(counts - 1, 1))
        Z /= np.where(std == 0, 1.0, std)
    zzt_low = syrk(1.0, Z, lower=1, trans=0, beta=0.0)
    if nan_policy == "mean":  denom_low = n - 1
    else:
        w_low = syrk(1.0, mask.astype(dtype), lower=1, trans=0, beta=0.0)
        np.maximum(w_low - 1, 1, out=w_low)
        if nan_policy == "per_iid":
            row_max = np.maximum(w_low.max(1), w_low.max(0)) 
            denom_low = row_max[:, None]                   
        else: denom_low = w_low
    grm_low = zzt_low / denom_low
    if correlation_matrix:
        d = np.sqrt(np.diag(grm_low))
        grm_low /= d[:, None] * d[None, :]
        np.fill_diagonal(grm_low, 1.0)
    grm_full = grm_low + grm_low.T - np.diag(np.diag(grm_low))
    if (savefile is not None) or return_weights:
        w_full = denom_low + denom_low.T - np.diag(np.diag(denom_low))
    if savefile is not None:
        if ids is not None:
            ids.to_frame().set_axis(['idx'], axis = 1).reset_index()\
               .to_csv(f"{savefile}.grm.id", index = False, header = None, sep = '\t')
        idxs = np.tril_indices_from(grm_full)
        grm_full[idxs].astype(np.float32).tofile(f"{savefile}.grm.bin")
        w_full[idxs].astype(np.float32).tofile(f"{savefile}.grm.N.bin")
    if return_weights:
        zzt_full = zzt_low+zzt_low.T-np.diag(np.diag(zzt_low))
        if ids is None: return {"zzt":zzt_full,"w":w_full, "grm":grm_full}
        else: return {"zzt":toxr(zzt_full, ids),"w":toxr(w_full,ids), "grm":toxr(grm_full,ids)}
    return grm_full if ids is None else toxr(grm_full, ids)

def plink2GRM(plinkfile:str, n_autosomes:int=None, downsample_snps:float = None, downsample_stategy:str = 'equidistant', rfids = None, chrs_subset = None,
              double_male_x:bool = False, double_y:bool = False, double_mt:bool = False,save_grms_path:bool = None, decompose_grm:bool = False):
    if isinstance(plinkfile, pd.DataFrame):
        load_snps_from_df = True
        if 'sex' in plinkfile.columns: 
            fam = plinkfile.loc[:, ['sex']].rename({'sex':'gender'}, axis = 1).reset_index(names = 'iid')
        elif 'gender' in plinkfile.columns: 
            fam = plinkfile.loc[:, ['gender']].reset_index(names = 'iid')
        else: 
            print('sex|gender data not provided assigning all individuals to male')
            double_male_x = False
            fam = pd.DataFrame(data = {'gender':1, 'iid': plinkfile.index})
        bim = plinkfile.columns[plinkfile.columns.str.contains(':')].to_series()\
                  .str.split(':', expand = True).set_axis(['chrom', 'pos'], axis = 1)\
                  .astype({'pos': int}) 
    else:
        bim, fam, gen = load_plink(plinkfile) if isinstance(plinkfile,str) else plinkfile 
        load_snps_from_df = False
    if save_grms_path is not None:
        save_grms_path = save_grms_path.rstrip('/')
        if not os.path.isdir(save_grms_path): os.makedirs(save_grms_path, exist_ok=True)
    allgrms = pd.DataFrame(columns = ['grm', 'w', 'zzt'])
    loopchr = bim.chrom.unique()
    if chrs_subset is not None:
        chrs_subset = list(map(str, chrs_subset))
        loopchr     = [x for x in loopchr if str(x) in chrs_subset]
    for c in tqdm(loopchr, desc = 'making grm'):
        if load_snps_from_df: 
            snps = plinkfile.filter(regex = f'^{c}:')
            if rfids is not None: snps = snps.loc[list(rfids)]
        else: snps = plink2df((bim, fam, gen), c = c, downsample_snps=downsample_snps, rfids = rfids,
                                downsample_stategy=downsample_stategy)
        if n_autosomes is not None:
            num2xymt = lambda x: x if not (z:=defaultdict(int, {n_autosomes+1:'x',n_autosomes+2:'y',
                                                                n_autosomes+3:'xy', n_autosomes+4:'mt'})[int(x)]) else z
            if (c2 := str(c).replace('chr', '').lower()).isdigit(): 
                c2 = num2xymt(c2)
            if c2 in ['x', 'y']: 
                maleiids = fam[fam.gender.isin(['M', 'm', 1, '1'])].iid
                if   c2 == 'x' and double_male_x: snps.loc[maleiids, :] *= 2
                elif c2 == 'y':  
                    snps = snps.loc[maleiids, :]
                    if double_y: snps *= 2
                elif c2 == 'mt' and double_mt: snps *= 2
        else: c2 = c
        filename = f'{save_grms_path}/{c2}chrGRM' if (save_grms_path is not None) else None
        allgrms.loc[str(c2)] = GRM(snps, return_weights=True, savefile=filename)
        
    allgrms['isnum'] = allgrms.index.str.isnumeric()
    if (save_grms_path is not None) and (chrs_subset is None):
        allzzt = allgrms.loc[allgrms.isnum,'zzt'].sum()
        allw = allgrms.loc[allgrms.isnum,'w'].sum()    
        allgrmf = allzzt/allw
        fam[['iid', 'iid']].to_csv(f"{save_grms_path}/AllchrGRM.grm.id", index = False, header = None, sep = '\t')
        idxs = np.tril_indices_from(allgrmf)
        allgrmf.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.bin") 
        allw.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.N.bin") 
        allgrms['subtracted_grm'] = allgrms.progress_apply(lambda x: (allgrmf.to_pandas() if not x.isnum else 
                                                                     (allzzt - x.zzt)/(allw-x.w).to_pandas()), axis = 1)
        allgrms.loc['All', ['grm', 'w', 'zzt', 'subtracted_grm']] = (allgrmf, allw, allzzt, allgrmf)
        allgrms.loc['All','isnum'] = False
    if decompose_grm and (chrs_subset is None):
        allgrms[['U', 's']] = allgrms.progress_apply(lambda x: grm2Us(x.subtracted_grm.values, n_components=2000)[:2], 
                                                         axis=1, result_type="expand")
    allgrms = allgrms.sort_index(key = lambda idx : idx.str.lower().map({str(i): int(i) for i in range(1000)}|\
                                  {i: int(i) for i in range(1000)}|\
                                  {'all': -1000, 'x':1001, 'y' : 1002, 'mt': 1003, 'm': 1003}))
    return allgrms


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

def recodeSNP(s, a1, a2):
    return np.where(np.isnan(s), 'NA',
                    np.where(s == 0, a1+a1,
                    np.where(s == 1, a1+a2,
                    np.where(s == 2, a2+a2, '??')))).astype('U2')

def recodeSNPs(arr, a0, a1):
    a0, a1 = np.asarray(a0, dtype='U1'), np.asarray(a1, dtype='U1')
    hom_ref, het, hom_alt = np.char.add(a0, a0), np.char.add(a0, a1), np.char.add(a1, a1)
    out = np.full(arr.shape, 'NA', dtype='U2')
    rows, cols = np.indices(arr.shape)
    out[(arr == 0)] = hom_ref[cols[(arr == 0)]]
    out[(arr == 1)] = het[cols[(arr == 1)]]
    out[(arr == 2)] = hom_alt[cols[(arr == 2)]]
    if isinstance(arr, pd.DataFrame):
        out = pd.DataFrame(out, index = arr.index, columns = arr.columns).astype('string[pyarrow]')
    return out

def plink2df(plinkpath: str, rfids: list = None, snplist: list = None, c: int = None, pos_start: int = None, pos_end: int = None, 
             downsample_snps: int = None, downsample_stategy: Literal['random', 'equidistant'] = 'random', recodeACGT: bool = False) -> pd.DataFrame:
    if type(plinkpath) == str:  snps, iid, gen = load_plink(plinkpath)
    else: snps, iid, gen = plinkpath
    try:
        snps.chrom = snps.chrom.astype(int)
        if c is not None: c = int(c)
    except: 
        if c is not None: c = str(c)
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
    if downsample_snps is not None:
        if (0<downsample_snps<=1): downsample_snps= int(downsample_snps*sset.shape[0])
        if downsample_stategy == 'random': sset = sset.sample(min(int(downsample_snps), sset.shape[0])).sort_values('i')
        elif downsample_stategy == 'equidistant': sset = sset[::max(1,sset.shape[0]//int(downsample_snps))]
        else: print('''downsample_stategy not in ['random', 'equidistant'], ignoring downsampling''')
    col = iiid if rfids is None else iiid.loc[rfids]
    if not recodeACGT:
        return pd.DataFrame(gen[col.i.values ][:, sset.i.values].astype(np.float32),
                            index = col.index.values, columns = sset.snp.values )
    return pd.DataFrame(recodeSNPs(gen[col.i.values ][:, sset.i.values].astype(np.float32), 
                                   sset.a0.values, sset.a1.values),
                        index = col.index.values, columns = sset.snp.values )\
             .astype('string[pyarrow]').replace('NA', np.nan)

def whiten_data(U, D_inv_sqrt, X):
    d = np.diag(D_inv_sqrt)
    temp = np.einsum('ij,jk->ik', U.T, X,optimize=True)
    temp = d[:, None] * temp
    result = np.einsum('ij,jk->ik', U, temp, optimize=True)
    return result

def subblock_svd_from_full(U, s, obs, n_components=50):
    Usub, ssub = grm2Us(U[obs, :] * np.sqrt(s)[None, :]  ,  n_components=min(obs.sum(), n_components))
    return Usub, ssub**2

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

def grm2Us(G, n_components=None):
    rsvd = n_components if (n_components is not None) else 2000
    if (G.shape[0]> 4000) or (rsvd<200): 
        U, s, _ = randomized_svd(G, n_components=rsvd, random_state=0)
    else:
        s, U = np.linalg.eigh(G)
        idx = np.argsort(s)[::-1]
        s, U = s[idx], U[:, idx]
        if n_components is not None: s, U = s[:n_components], U[:, :n_components]
    return U, s

def load_all_grms(paths, decompose_grm = True, save_all_grm_if_missing: bool = True):
    from glob import glob
    if isinstance(paths, str): paths = glob(paths)
    allgrms = pd.DataFrame.from_records([read_grm(x.replace('.N', '').replace('.grm.bin', '.grm')) for x in paths])
    #allgrms = pd.DataFrame.from_records([read_grm_w(x) for x in glob(r'grm/*.grm.bin')])
    allgrms['path'] = allgrms['path'].str.extract(r'([\d\w]+)chrGRM.')
    allgrms['isnum'] = allgrms['path'].str.isnumeric()
    allgrms = allgrms.set_index('path')
    if 'All' not in allgrms.index:  
        allw = allgrms.loc[allgrms.isnum,'w'].sum()    
        allgrmf = ( allgrms.loc[allgrms.isnum,'grm'] * allgrms.loc[allgrms.isnum,'w'] ).sum() / allw
        allgrms.loc['All', ['grm', 'w', 'subtracted_grm']] = (allgrmf, allw, allgrmf)
        if save_all_grm_if_missing:
            save_grms_path = paths.rsplit('/', 1)[0]
            allgrms.loc['All','grm'].index.to_series().reset_index(name= '___')\
                   .to_csv(f"{save_grms_path}/AllchrGRM.grm.id", index = False, header = None, sep = '\t')
            idxs = np.tril_indices_from(allgrmf)
            allgrmf.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.bin") 
            allw.values[idxs].astype(np.float32).tofile(f"{save_grms_path}/AllchrGRM.grm.N.bin")
        
    allgrms_weighted = allgrms.loc["All", 'grm']*allgrms.loc["All", 'w']
    allgrms['subtracted_grm'] = allgrms.progress_apply(lambda x: (allgrms.loc["All", 'grm'] if not x.isnum else 
                                                             (allgrms_weighted - x.grm*x.w)/(allgrms.loc["All", 'w']-x.w)).to_pandas(),
                                                   axis = 1)
    if decompose_grm:
        allgrms[['U', 's']] = allgrms.progress_apply(lambda x: grm2Us(x.subtracted_grm.values, n_components=2000, random_state=0)[:2], 
                                                     axis=1,  result_type="expand")
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
        U, s = grm2Us(grm, n_components=n_components)
    elif s is not None and U is not None and grm is None: 
        #Ue, se = subblock_svd_from_full(U, s, obs, n_components = 700)
        U = U[obs, :]
        s *= np.sum(U**2, axis=0) #* s
        if n_components is not None:
            U, s = U[:, :n_components], s[:n_components] 
        # U[:, :len(se)] = Ue
        # s[:len(se)]    = se
        
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
        U, s  = grm2Us(G, n_components=n_components)
    elif s is not None and U is not None and G is None: pass
    else: raise ValueError('cannot submit both G and U s at the same time')
    eigs_fG = yvar * (h2 * s + (1 - h2))
    eigs_fG[eigs_fG < tol] = tol
    D_inv_sqrt = np.diag(1 / np.sqrt(eigs_fG))
    if return_eigen: return U, D_inv_sqrt
    W = U @ D_inv_sqrt @ U.T
    return W
    
def rm_relatedness(c, trait, df, allgrms, n_components = None,return_eigen=True, svd_input = True, ):
    grm_c = allgrms.loc[str(c),'subtracted_grm']
    trait_ = df.loc[grm_c.index, trait]
    navals = ~trait_.isna()
    yvar = np.nanvar(trait_)
    if navals.sum()< 10: 
        print('not enough samples')
        return None
    if svd_input:
        h2_res = H2SVD(y = trait_, s = allgrms.loc[str(c),'s'], U = allgrms.loc[str(c),'U'],
                   return_SVD=True, n_components = n_components) 
    else: 
        h2_res = H2SVD(y = trait_, grm = grm_c, return_SVD=True, n_components = n_components) 
    U, D_inv_sqrt = remove_relatedness_transformation(yvar= yvar,**h2_res, return_eigen=True)
    if U.shape[0] == sum(navals): 
        trait_vec = trait_.loc[navals]#.to_frame()
        idx = navals[navals].index
    else: 
        trait_vec = trait_.copy()
        idx = navals.index
    uttrait = U.T @ trait_vec
    transformed = U @ (D_inv_sqrt @ uttrait)
    #transformed = whiten_data(U,D_inv_sqrt,trait_vec)
    weights = (h2_res['h2'] * h2_res['s']) / (h2_res['h2'] * h2_res['s'] + (1.0 - h2_res['h2']))
    blup_vec = (U @ (weights * uttrait))
    if return_eigen:
        return {'transformed':pd.Series(transformed, index = idx, name = f'{trait}__subtractgrm{c}'),
                'blup': pd.Series(blup_vec, index=idx, name=f'{trait}__blup{c}'),
                'U': U, 'D_inv_sqrt': D_inv_sqrt, 'c': c, 'trait': trait,
                'h2': h2_res['h2'], 'n_components':n_components}
    return pd.Series(transformed, index = idx, name = f'{trait}__subtractgrm{c}')

def scale_with_mask(X, precision = np.float32):
    X = np.asarray(X, dtype=precision)
    M = ~np.isnan(X)
    X_centered = X - np.nanmean(X, axis=0)
    X_centered[~M] = 0.0
    sum_sq = np.einsum("ij,ij->j", X_centered, X_centered, optimize=True)
    std = np.sqrt(sum_sq / M.sum(axis=0))
    std[std == 0] = np.nan
    X_scaled = X_centered / std
    return X_scaled, std, M.astype(precision)

def regression_with_einsum_old(ssnps, straits, snps_mask, traits_mask,dof='correct', stat = 'ttest', sided = 'two-sided'):
    if sided not in ['two-sided','one-sided']: raise ValueError("sided must be 'two-sided' or 'one-sided'")
    if stat  not in ['ttest', 'wald']: raise ValueError("stat must be 'ttest' or 'wald'")
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
    #sigma2 = SSR / df
    se_beta = np.sqrt(SSR / df / diag_XtX)
    stats = beta / se_beta
    if stat == 'ttest':
        p_values = (2 if sided == 'two-sided' else 1) * scipyt.sf(np.abs(stats), df=df)
    else:
        np.square(stats, out = stats) # *= stats
        p_values = chi2.sf(stats, df=1)
        if sided == 'one-sided': p_values = np.where(beta >= 0, 0.5 * p_values, 1 - 0.5 * p_values)
    neg_log10_p_values = -np.log10(p_values)
    return beta, se_beta, stats, neg_log10_p_values, df

def regression_with_einsum(ssnps, straits, snps_mask, traits_mask,dof='correct', stat = 'ttest', sided = 'two-sided'):
    if sided not in ['two-sided','one-sided']: raise ValueError("sided must be 'two-sided' or 'one-sided'")
    if stat  not in ['ttest', 'wald', 'score']: raise ValueError("stat must be 'ttest' or 'wald'")
    XtY      = np.empty((ssnps.shape[1],straits.shape[1]), dtype=float)
    diag     = np.empty_like(XtY)
    ssr      = np.empty_like(XtY)
    df       = np.empty_like(XtY)
    np.einsum("ij,ik->jk", ssnps, straits, out=XtY, optimize=True)
    np.einsum("ij,ij,ik->jk", ssnps, ssnps, traits_mask, out=diag,  optimize=True)
    phen = straits * traits_mask   
    np.einsum("ij,ik,ik->jk", snps_mask, phen, phen, out=ssr, optimize=True)
    if dof!='incorrect': 
        np.einsum("ij,ik->jk", snps_mask, traits_mask, out=df,  optimize=True)
        df -= 1
    else: df = np.broadcast_to(traits_mask.sum(axis=0) - 1 , (ssnps.shape[1], traits_mask.shape[1])).copy()
    df[df <= 0] = np.nan
    beta = XtY
    beta /= diag
    if stat != 'score': ssr -= beta * beta * diag
    np.divide(ssr, df*diag, out=ssr)
    np.sqrt(ssr, out=ssr)
    se_beta = ssr
    stats = beta/se_beta
    if stat == 'ttest':
        p_values = scipyt.sf(np.abs(stats), df=df)
        if sided == 'two-sided': p_values*=2
    elif stat in ['wald', 'score']:
        np.abs(stats, out = stats)
        p_values = erfc(stats/np.sqrt(2))
        if sided == 'one-sided': p_values*=.5
        np.square(stats, out = stats)
    np.log10(p_values, out=p_values)
    np.negative(p_values, out=p_values)
    return beta, se_beta, stats, p_values, df

# def GWA(traits, snps, dtype = 'pandas', precision=np.float32, stat = 'ttest', sided = 'two-sided'):
#     if isinstance(snps, tuple) and len(snps)==3: ssnps, snps_std, snps_mask = snps
#     else: ssnps, snps_std, snps_mask = scale_with_mask(snps, precision = precision)
#     if isinstance(traits, tuple) and len(snps)==3: straits, traits_std, traits_mask = traits
#     straits, traits_std, traits_mask = scale_with_mask(traits, precision = precision)
#     if dtype == 'tuple':  
#         return regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof = 'correct', stat = stat, sided = sided)
#     res = xr.DataArray(
#              np.stack(regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof = 'correct', stat = stat, sided = sided), axis=0), 
#              dims=["metric", "snp", "trait"],
#              coords={"metric": np.array(['beta', 'beta_se', 'stat', 'neglog_p', 'dof']),
#                      "snp":   list(snps.columns),
#                      "trait": traits.columns.map(lambda x: x.split('__subtractgrm')[0]).to_list()} )
#     if dtype == 'pandas':  return res.to_dataset(dim="metric").to_dataframe().reset_index()
#     return res

def GWA(traits, snps, dtype = 'pandas_highmem', precision=np.float32, stat = 'ttest', sided = 'two-sided',dof='correct'):
    if isinstance(snps, tuple) and len(snps)==3: ssnps, snps_std, snps_mask = snps
    else: ssnps, snps_std, snps_mask = scale_with_mask(snps, precision = precision)
    if isinstance(traits, tuple) and len(snps)==3: straits, traits_std, traits_mask = traits
    straits, traits_std, traits_mask = scale_with_mask(traits, precision = precision)
    results = regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof = dof, stat = stat, sided = sided)
    metrics     = ['beta', 'beta_se', 'stat', 'neglog_p', 'dof']
    snp_names, trait_names   = list(snps.columns), [t.split('__subtractgrm')[0] for t in traits.columns]
    if dtype not in ['tuple', 'xarray', 'pandas', 'pandas_highmem']: raise ValueError(" dtype has to be in ['tuple', 'xarray', 'pandas']")
    if dtype == 'tuple':  return results
    elif dtype == 'xarray_dataset':
        return xr.Dataset( { m: (('snp','trait'), arr) for m,arr in zip(metrics, results) },
            coords={'snp': snp_names, 'trait': trait_names})
    elif dtype in ['xarray', 'pandas_highmem']:
        xar =  xr.DataArray( np.stack(results, axis=0),dims=["metric", "snp", "trait"],
             coords={"metric": np.array(['beta', 'beta_se', 'stat', 'neglog_p', 'dof']),
                     "snp":   snp_names, "trait": trait_names} )
        if dtype == 'xarray': return xar
        return xar.to_dataset(dim="metric").to_dataframe().reset_index()
    elif dtype == 'pandas': 
        snp_col   = np.repeat(snp_names, len(traits.columns))
        trait_col = np.tile(trait_names, len(snps.columns))
        return  pd.DataFrame({ 'snp':   snp_col, 'trait': trait_col,
            **{ metric: arr.ravel(order='C') for metric, arr in zip(metrics, results)}})
    return results

def GWAS(traitdf, genotypes = 'genotypes/genotypes', grms_folder = 'grm', save = True , y_correction  = 'ystar',
         save_path = 'results/gwas/', return_table = True, stat = 'ttest', dtype = 'pandas_highmem',dof='correct'):
    res = []
    read_gen = load_plink(genotypes)
    chrunique =[str(x) for x in read_gen[0].chrom.unique()]
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    chrs2run = grms.index[~grms.index.str.contains('^All$')].to_list()
    for C in tqdm(chrs2run,position=0, desc = 'running Chr'):
        if y_correction=='blup_resid':
            traits =  tdf - pd.concat([rm_relatedness(C,_t,tdf,grms, svd_input=False)['blup'] for _t in tdf.columns], axis = 1).rename(lambda x: x.split('__')[0], axis = 1)
        elif y_correction=='ystar': 
            traits = pd.concat([rm_relatedness(C,_t,tdf,grms, svd_input=False)['transformed'] for _t in tdf.columns], axis = 1)
        elif y_correction is None:  traits = tdf
        else: raise ValueError(" y_correction has to be in ['blup_resid', 'ystar', None]")
        if str(C) not in chrunique: chr_alias = read_gen[0].loc[read_gen[0].snp.str.lower().str.startswith(f'{C}:').idxmax(), 'chrom']
        else: chr_alias = C
        snps = plink2df(read_gen, c=chr_alias, rfids=traits.index)
        if not (snps.index == traits.index).all(): 
            print('reordering snps to align with traits')
            snps =snps.loc[traits.index]
        if return_table: 
            res += [GWA(traits, snps, dtype=dtype, stat=stat, dof='correct')]
            if save: res[-1].to_parquet(f'{save_path}gwas{C}.parquet.gz', compression='gzip', engine = 'pyarrow',  compression_level=1, use_dictionary=True)
        else: 
            if save: GWA(traits, snps, dtype=dtype, stat=stat, dof='correct')\
                           .to_parquet(f'{save_path}gwas{C}.parquet.gz', compression='gzip',  engine = 'pyarrow',  compression_level=1, use_dictionary=True) #use_byte_stream_split=True
    if return_table: 
        if 'pandas' not in dtype: return res
        return pd.concat(res)
    return


def describe_trait_chr(traitdf, grms_folder = 'grm', return_allchrs=False, include_cols = ['transformed', 'blup', 'U', 'D_inv_sqrt', 'h2', 'n_components']):
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    include_cols = list(set(include_cols+['c', 'trait'] ))
    chrs2run = ['All'] + (grms.index[~grms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    return pd.DataFrame((pd.Series(rm_relatedness(_c,_t,tdf,grms,  svd_input=False)).loc[include_cols] \
                                for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), 
                                                   leave = True, desc = 'describing: ' + '|'.join(include_cols) )))\
           .set_index(['c', 'trait'])

def BLUP(traitdf, grms_folder = 'grm', return_allchrs=False):
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    chrs2run = ['All'] + (grms.index[~grms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    return pd.concat([rm_relatedness(_c,_t,tdf,grms, svd_input=False)['blup'] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave = True, desc = 'calculating BLUP')], axis = 1)

def ySTAR(traitdf, grms_folder = 'grm', return_allchrs=False):
    grms = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False)
    tdf = traitdf.loc[grms.loc[ 'All', 'subtracted_grm'].index, :]
    chrs2run = ['All'] + (grms.index[~grms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    return pd.concat([rm_relatedness(_c,_t,tdf,grms, svd_input=False)['transformed'] for _c, _t in tqdm(list(itertools.product(chrs2run, tdf.columns)), leave = True, desc = 'whittening y')], axis = 1)

def heritability(traitdf, grms_folder = 'grm', return_allchrs=False, svd_input=False, n_components = None):
    allgrms,res = load_all_grms(f'{grms_folder}/*.grm.bin', decompose_grm=False), []
    chrs2run = ['All'] + (allgrms.index[~allgrms.index.str.contains('^All$')].to_list() if return_allchrs else [])
    for _c, _t in tqdm(list(itertools.product(chrs2run, traitdf.columns)), leave = True, desc = 'calculating heritability'):
        grm_c= allgrms.loc[str(_c),'grm'].to_pandas()
        _y = traitdf.loc[grm_c.index, _t]
        params = dict(y = _y, return_SVD=True, n_components = n_components) 
        params = params | (dict(s = allgrms.loc[str(c),'s'], U = allgrms.loc[str(c),'U']) if svd_input else dict(grm = grm_c))
        res += [pd.Series(H2SVD(**params) | dict(trait = _t, chrom = _c, n = _y.count()), name = f'{_t}_{_c}').drop(['U', 's'])]
    return pd.concat(res, axis=1).T

def shuffle_replicates(df,col ,n=500):
    reps = df[[col]*n]
    for i in range(1, n):  reps.iloc[:, i] = np.random.permutation(reps.iloc[:, i].values)
    reps.columns = [f'{col}_ORIGINAL'] + [f'{col}_SHUFFLE{str(i).zfill(3)}' for i in range(1, n)]  
    return reps

def shuffle_replicates_normal(df,col ,n=500):
    napct = df[col].isna().mean()
    r = np.random.RandomState(42)
    reps =  pd.DataFrame(r.normal(size = (df.shape[0], n)), \
                         columns = [f'{col}_SHUFFLE{str(i).zfill(3)}' for i in range( n)], \
                         index = df.index )
    reps  *=  r.choice([1, np.nan],size =  reps.shape ,   p = [1-napct, napct])
    return reps


def r2toD(r2, snp1maf,snp2maf):
    return np.sqrt(r2*snp1maf*(1-snp1maf)*snp2maf*(1-snp2maf))

def R2(X, Y= None, return_named = True, return_square = True, statistic = 'r2', dtype = np.float64):
    if statistic not in ['r2', 'r', 'cov', 'D', 'D2', 'chord']: raise ValueError("statistic has to be in ['r2', 'r', 'cov', 'D', 'D2', 'chord']")
    x = np.array(X).astype(dtype)
    xna = (~np.isnan(x)).astype(dtype) ##get all nas
    xnaax0 = xna.sum(axis = 0)    
    if statistic in ['D', 'D2']: 
        p_x = (np.nansum(x, axis=0) / xnaax0)
        x -= p_x
        p_x = np.clip(p_x*0.5, 1e-12, 1 - 1e-12)
    else: x -= (np.nansum(x, axis = 0)/xnaax0) #subtract mean
    np.nan_to_num(x, copy = False,  nan=0.0, posinf=None, neginf=None ) ### will not affect sums 
    xstd = np.sqrt(np.sum(x**2, axis = 0)/xnaax0) #estimate std
    xstd[xstd == 0] = np.nan
    if Y is None:  
        y, yna, ystd = x, xna, xstd 
        if statistic in ['D', 'D2']: p_y = p_x
    else:
        y = np.array(Y).astype(dtype)
        yna = (~np.isnan(y)).astype(dtype) ##get all nas
        ynaax0 = yna.sum(axis = 0)
        if statistic in ['D', 'D2']: 
            p_y = (np.nansum(x, axis=0) / xnaax0)
            y -= p_y
            p_y = np.clip(p_y*0.5, 1e-12, 1 - 1e-12)
        else:  y -= (np.nansum(y, axis = 0)/ynaax0) #subtract mean
        np.nan_to_num(y, copy = False,  nan=0.0, posinf=None, neginf=None ) ### will not affect sums 
        ystd = np.sqrt(np.sum(y**2, axis = 0)/ynaax0) #estimate std
        ystd[ystd == 0] = np.nan
    xty_w = np.dot(xna.T,yna)
    xty_w[xty_w == 0] = np.nan
    res = np.dot(x.T,y) / xty_w
    if statistic in ['r2', 'D2']: 
        res = np.clip(np.power(res/np.outer(xstd, ystd), 2), a_min = 0, a_max = 1)
        if statistic == 'D2': res *= np.outer(p_x*(1-p_x), p_y*(1-p_y))
    elif statistic in ['r', 'D', 'chord']: 
        res = np.clip(res/np.outer(xstd, ystd), a_min = -1, a_max = 1)
        if statistic == 'chord': res = np.sqrt(2*(1-res))
        if statistic == 'D': res *= np.sqrt(np.outer(p_x*(1-p_x), p_y*(1-p_y)))
    rindex = X.columns if isinstance(X, pd.DataFrame) else list(range(x.shape[1]))
    if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.columns
    elif isinstance(Y, pd.DataFrame): rcolumns = Y.columns
    else: rcolumns = list(range(y.shape[1]))
    if return_named: 
        res = pd.DataFrame(res, index = rindex, columns = rcolumns)  
        if not return_square:
            res = res.reset_index(names = 'bp1').melt(id_vars = 'bp1', var_name='bp2')
            chrom = res['bp1'].iloc[0].split(':')[0]
            pos = len(chrom)+ 1
            res['c'] = chrom
            res['distance'] = (res.bp1.str.slice(start=pos).astype(int)  - res.bp2.str.slice(start=pos).astype(int)).abs()
    return res

from numba import njit, prange
@njit(parallel=True, fastmath=False)
def _cityblock_distance(X, Y, axis = 'rows', dtype = np.float32, scale = True):
    if axis == 'columns': 
        if Y is None: return _cityblock_distance(X.T, Y = None, axis = 'rows')
        return _cityblock_distance(X.T, Y.T, axis = 'rows')
    nx, n = X.shape
    if Y is not None:
        ny = Y.shape[0]
        out = np.full((nx, ny), np.nan, dtype=dtype)
        for i in prange(nx):
            for j in range(ny):
                s = 0.0
                c = 0
                for k in range(n):
                    a,b = X[i, k], Y[j, k]
                    if not np.isnan(a) and not np.isnan(b):
                        s += abs(a - b)
                        c += 1
                if c > 0: out[i, j] = s if not scale else s/c   
        return out
    out = np.full((nx, nx), np.nan, dtype=dtype)
    for i in prange(nx):
        for j in range(nx):
            if i>j: 
                s = 0.0
                c = 0
                for k in range(n):
                    a,b = X[i, k], X[j, k]
                    if not np.isnan(a) and not np.isnan(b):
                        s += abs(a - b)
                        c += 1
                if c > 0: 
                    out[i, j] = s if not scale else s/c  
                    out[j, i] = out[i, j] 
            if i == j: 
                out[i, j] = 0
    return out

def cityblock(X, Y=None,  axis = 'rows', dtype = np.float32, scale = True):
    if axis == 'columns':
        rindex = X.columns if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.columns
        elif isinstance(Y, pd.DataFrame): rcolumns = Y.columns
        elif Y is None: rcolumns = list(range(X.shape[1]))
        else:  rcolumns = list(range(Y.shape[1]))
    if axis == 'rows':
        rindex = X.index if isinstance(X, pd.DataFrame) else list(range(X.shape[0]))
        if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.index
        elif isinstance(Y, pd.DataFrame): rcolumns = Y.index
        elif Y is None: rcolumns = list(range(X.shape[0]))
        else:  rcolumns = list(range(Y.shape[0]))
    return pd.DataFrame(_cityblock_distance(X.values if isinstance(X, pd.DataFrame) else X, 
                                            Y.values  if isinstance(Y, pd.DataFrame) else Y, 
                                            axis = axis, dtype = dtype, scale = scale), 
                        index = rindex, columns = rcolumns)







    