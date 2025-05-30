import pandas as pd
import dask.dataframe as dd
import gzip
import matplotlib.pyplot as plt

class vcf:
    def header(vcf_path):
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return line.strip().split('\t')
                if num > 10000: return '-1'
        return '-1'

    def read(filename, method = 'pandas'):
        if method == 'dask':
            return dd.read_csv(filename,  compression='gzip', comment='#',  sep ='\s+', header=None, 
                               names = vcf_manipulation.get_vcf_header(filename),blocksize=None,  dtype=str, ).repartition(npartitions = 100000)
        return pd.read_csv(filename,  compression='gzip', comment='#',  sep ='\s+',
                           header=None, names = vcf_manipulation.get_vcf_header(filename),  dtype=str )
    def metadata(vcf_path):
        out = ''
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return out
                out += line 
        return '-1'

    def df2vcf(df, filename, metadata = ''):
        if  metadata == '':
            header = '\n'.join(["##fileformat=VCFv4.1",
            '##fileDate=20090805',
            '##source=myImputationProgramV3.1',
            '##reference=000GenomesPilot-NCBI36',
            '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">',
            '##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">']) + '\n'
        elif metadata[-4:] == '.vcf': header = self.get_vcf_metadata(metadata)
        with open(filename, 'w') as vcf: 
            vcf.write(header)
        df.to_csv(filename, sep="\t", mode='a', index=False)