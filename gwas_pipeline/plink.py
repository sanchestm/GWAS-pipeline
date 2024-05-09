import pandas as pd
import pandas_plink
from wrapbash import bash


def plink2pddf(plinkpath,rfids = 0, c = 0, pos_start = 0, pos_end = 0, snplist = 0):
    if type(plinkpath) == str: 
        snps, iid, gen = pandas_plink.read_plink(plinkpath)
    else: snps, iid, gen = plinkpath
    snps.chrom = snps.chrom.astype(int)
    snps.pos = snps.pos.astype(int)
    isnps = snps.set_index(['chrom', 'pos'])
    iiid = iid.set_index('iid')
    if not snplist:
        if (pos_start == pos_start == 0 ):
            if not c: index = isnps
            else: index = isnps.loc[(slice(c, c)), :]
        index = isnps.loc[(slice(c, c),slice(pos_start, pos_end) ), :]
    else:
        snplist = list(set(snplist) & set(isnps.snp))
        index = isnps.set_index('snp').loc[snplist].reset_index()
    col = iiid  if not rfids else iiid.loc[rfids]
    return pd.DataFrame(gen.astype(np.float16)[index.i.values ][:, col.i].T, index = col.index.values.astype(str), columns = index.snp.values )

def plink(print_call = False, **kwargs):
    call = 'plink ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                for k,v in kwargs.items() if k not in ['print_call'] ])
    call = re.sub(r' +', ' ', call).strip(' ')
    bash(call, print_call=print_call)
    return