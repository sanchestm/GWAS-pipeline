import pandas as pd
import mygene

def annotate_vep(snpdf, species, snpcol = 'SNP', refcol = 'A2', altcol = 'A1', refseq = 1, expand_columns = True, intergenic = False):
    import requests, sys
    server = "https://rest.ensembl.org"
    ext = f"/vep/{species}/hgvs"
    res =  '"' + snpdf[snpcol].str.replace(':', ':g.').str.replace('chr', '') + snpdf[refcol] + '>'+snpdf[altcol] + '"' 
    res = f"[{','.join(res)}]"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    call = '{ "hgvs_notations" : ' + res + f', "refseq":{refseq}, "OpenTargets":1, "AlphaMissense":1,"Phenotypes":1, "Enformer":1,"LoF":1'+' }'
    print(call) # , 
    r = requests.post(server+ext, headers=headers, data=call) #
    if not r.ok:
      print('VEP annotation failed')#r.raise_for_status()
      return snpdf
    decoded = pd.json_normalize( r.json())
    #print(repr( r.json()))
    decoded['SNP'] = decoded.input.map(lambda x: x.replace(':g.', ':')[:-3])
    if (not intergenic) and ('intergenic_consequences' in decoded.columns):
        decoded = decoded.drop('intergenic_consequences', axis = 1)
    if 'colocated_variants' in decoded.columns: 
        decoded['colocated_variants'] = decoded['colocated_variants'].map(lambda x: pd.json_normalize(x) if isinstance(x, list) else pd.DataFrame())
        decoded.colocated_variants =  decoded.colocated_variants.map(lambda x: '|'.join(x['id']) if 'id' in x.columns else x)
    jsoncols = list(set(['transcript_consequences','intergenic_consequences']) & set(decoded.columns))
    if len(jsoncols):
        decoded[jsoncols] = decoded[jsoncols].applymap(lambda x: pd.json_normalize(x) if isinstance(x, list) else pd.DataFrame())
        if expand_columns:
            for i in jsoncols:
                tempdf = pd.concat(decoded.apply(lambda x: x[i].rename(lambda y: f'{i}_{y}', axis = 1).assign(SNP = x.SNP), axis = 1).to_list())
                tempdf = tempdf.applymap(lambda x: x[0] if isinstance(x, list) else x)
                tempdf.loc[:, tempdf.columns.str.contains('phenotypes')] = \
                     tempdf.loc[:, tempdf.columns.str.contains('phenotypes')].applymap(lambda x: x['phenotype'].replace(' ', '_') if isinstance(x, dict) else x)
                tempdf = tempdf.drop_duplicates(subset = tempdf.loc[:, ~tempdf.columns.str.contains('phenotypes')].filter(regex =f'{i}_') \
                                              .drop(['transcript_consequences_transcript_id', 'transcript_consequences_cdna_start', 'transcript_consequences_cdna_end'
                                                     'transcript_consequences_consequence_terms', 'transcript_consequences_strand',
                                                    'transcript_consequences_distance', 'transcript_consequences_variant_allele'], errors='ignore', axis =1).columns.to_list())
                decoded = decoded.merge(tempdf, how = 'left', on = 'SNP')
            decoded = decoded.drop(jsoncols, axis = 1)
            decoded = decoded.rename({'transcript_consequences_gene_symbol':'gene', 'transcript_consequences_gene_id': 'geneid',
                                     'transcript_consequences_biotype': 'transcriptbiotype', 'transcript_consequences_impact':'putative_impact'}, axis = 1)
            if 'putative_impact' in decoded.columns: decoded.putative_impact = decoded.putative_impact.fillna('MODIFIER')
            decoded = decoded.rename(lambda x: x.replace('consequences_',''), axis = 1)

    return snpdf.merge(decoded.drop(['id', 'seq_region_name', 'end', 'strand', 'allele_string', 'start'], axis = 1).rename({'input': ''}), on = 'SNP', how = 'left')

def query_gene(genelis, species):
    mg = mygene.MyGeneInfo()
    species = translate_dict(species, {'rn7': 'rat', 'rn8':'rat', 'm38':'mice', 'rn6': 'rat'})
    a = mg.querymany(genes_in_range.gene_name , scopes='all', fields='all', species=species, verbose = False, silent = True)
    res = pd.concat(pd.DataFrame({k:[v]  for k,v in x.items()}) for x in a)
    res = res.assign(**{k:np.nan for k in (set(['AllianceGenome','symbol', 'ensembl', 'notfound']) - set(res.columns))} )
    return res[res.notfound.isna()].set_index('query')

def get_ncbi_gtf(genome):
    printwithlog('reading gene list from NCBI RefSeq from NCBI GTF...')
    linkdict = {'rn7':'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/015/227/675/GCF_015227675.2_mRatBN7.2/GCF_015227675.2_mRatBN7.2_genomic.gtf.gz' , 
                    'rn6':'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/895/GCF_000001895.5_Rnor_6.0/GCF_000001895.5_Rnor_6.0_genomic.gtf.gz',
                   'm38': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_genomic.gtf.gz',
                  'rn8': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/10116/GCF_036323735.1-RS_2024_02/GCF_036323735.1_GRCr8_genomic.gtf.gz'}
    gtf = pd.read_csv(linkdict[genome], sep = '\t', header = None, comment='#')\
               .set_axis(['Chr', 'source', 'biotype', 'start', 'end', 'score', 'strand', 'phase', 'ID'], axis = 1)
    gtf = gtf[gtf.biotype != 'transcript'].reset_index(drop = True)
    gtf['biotype'] = gtf['biotype'].str.replace('gene','transcript')
    gtfjson = pd.json_normalize(gtf['ID'].map(lambda x: {y.split(' "')[0].strip(' '): y.split(' "')[-1][:-1] for y in x.strip(';').split(';')}).to_list())
    gtf =pd.concat([gtf.drop('ID', axis = 1),gtfjson], axis = 1)
    gtf[['gene', 'gene_id']] = gtf[['gene', 'gene_id']].fillna('').astype(str)
    gtf = gtf[~gtf.gene.str.contains('-ps')]
    if genome == 'm38': gtf['Chr'] = gtf['Chr'].map(lambda x: translatechrmice[x]) 
    elif genome in ['rn6', 'rn7']:  gtf['Chr'] = gtf['Chr'].map(lambda x: translatechr[x])
    elif genome in ['rn8']:   gtf['Chr'] = gtf['Chr'].map(lambda x: translatechr8[x])
    else: raise ValueError('no genome that was able to download')
    gtf = gtf[~gtf.Chr.str.lower().str.contains('un|na| nc')]
    gtf = gtf.dropna(subset = 'gene')
    gtf = gtf[~gtf.gene.str.startswith('LOC')&~gtf.gene.str.startswith('NEW')]
    gtf['Chr'] = gtf['Chr'].map(lambda x: replaceXYMTtonums(x.split('_')[0]))
    gtf = gtf.loc[gtf.gene.fillna('') != '', ~gtf.columns.str.contains(' ')]
    return gtf

translatechr = defaultdict(lambda: 'UNK',{k.split('\t')[1]: k.split('\t')[0] for k in'''1	NC_051336.1
2	NC_051337.1
3	NC_051338.1
4	NC_051339.1
5	NC_051340.1
6	NC_051341.1
7	NC_051342.1
8	NC_051343.1
9	NC_051344.1
10	NC_051345.1
11	NC_051346.1
12	NC_051347.1
13	NC_051348.1
14	NC_051349.1
15	NC_051350.1
16	NC_051351.1
17	NC_051352.1
18	NC_051353.1
19	NC_051354.1
20	NC_051355.1
X	NC_051356.1
Y	NC_051357.1
MT	NC_001665.2'''.split('\n')})

translatechrmice = defaultdict(lambda: 'UNK',{k.split(' ')[1]: k.split(' ')[0] for k in '''1 NC_000067.6
2 NC_000068.7
3 NC_000069.6
4 NC_000070.6
5 NC_000071.6
6 NC_000072.6
7 NC_000073.6
8 NC_000074.6
9 NC_000075.6
10 NC_000076.6
11 NC_000077.6
12 NC_000078.6
13 NC_000079.6
14 NC_000080.6
15 NC_000081.6
16 NC_000082.6
17 NC_000083.6
18 NC_000084.6
19 NC_000085.6
X NC_000086.7
Y NC_000087.7
MT NC_005089.1'''.split('\n')})

translatechr8 = defaultdict(lambda: 'UNK',{k.split(' ')[1]: k.split(' ')[0] for k in'''1 NC_086019.1
2 NC_086020.1
3 NC_086021.1
4 NC_086022.1
5 NC_086023.1
6 NC_086024.1
7 NC_086025.1
8 NC_086026.1
9 NC_086027.1
10 NC_086028.1
11 NC_086029.1
12 NC_086030.1
13 NC_086031.1
14 NC_086032.1
15 NC_086033.1
16 NC_086034.1
17 NC_086035.1
18 NC_086036.1
19 NC_086037.1
20 NC_086038.1
X NC_086039.1
Y NC_086040.1
MT NC_001665.2'''.split('\n')})
