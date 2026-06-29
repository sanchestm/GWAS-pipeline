import ftplib

def genome_links(ftp_link = 'ftp.ncbi.nlm.nih.gov', subpath = '', verbose = False, recursive = True, max_recursion=2000):
    ftp = ftplib.FTP(ftp_link)
    ftp.login()
    recursion_cnt = 0
    def genome_links_sub( subpath = '', verbose = False , recursion_cnt = 0):
        if max_recursion < recursion_cnt: return [f'{ftp_link.strip("/")}/{subpath.strip("/")}/']
        ftp = ftplib.FTP(ftp_link)
        ftp.login()
        if verbose: print(f'{ftp_link.strip("/")}/{subpath.strip("/")}')
        ftp.cwd(subpath)
        res = []
        paths = [i for i in ftp.mlsd()]
        for i in paths:
            if i[1]['type'] == 'file':
                res += [f'{ftp_link.strip("/")}/{subpath.strip("/")}/{i[0]}']
                if verbose: print(f'{ftp_link.strip("/")}/{subpath.strip("/")}/{i[0]}')
        for i in paths:
            if i[1]['type'] == 'dir':
                if recursive: res += genome_links_sub(subpath = subpath.strip('/') + '/' + i[0], verbose = verbose, recursion_cnt = recursion_cnt+1)
                else: res += [f'{ftp_link.strip("/")}/{subpath.strip("/")}/{i[0]}/']
                if verbose: print(f'{ftp_link.strip("/")}/{subpath.strip("/")}/{i[0]}')
        ftp.close()
        return res
    all_paths = genome_links_sub(subpath =subpath, verbose = verbose, recursion_cnt =0)
    ftp.close()
    return all_paths

#genome_links(subpath = 'genomes/refseq', verbose = True, recursive = True, max_recursion=1)
            
def annotate_vep(snpdf, species, snpcol = 'SNP', refcol = 'A2', altcol = 'A1'):
    import requests, sys
    server = "https://rest.ensembl.org"
    ext = f"/vep/{species}/hgvs"
    res =  '"' + snpdf[snpcol].str.replace(':', ':g.') + snpdf[refcol] + '>'+snpdf[altcol] + '"' 
    res = str(list(res))
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    r = requests.post(server+ext, headers=headers, data='{ "hgvs_notations" : ' + res  + ', "refseq":1 , "Enformer":1}')
    if not r.ok:
      #r.raise_for_status()
      return snpdf
    
    decoded = r.json()
    return repr(decoded)


basepaths = ['protozoa','vertebrate_mammalian', 'vertebrate_other', 'viral', 'plant', 'invertebrate', 'fungi', 'archaea']

res = []
for x in basepaths:
    try: res = [pd.read_csv(f'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/{x}/assembly_summary.txt',  comment='##', sep = '\t', engine = 'python')]
    except: print(x)
genome_table = pd.concat(res).reset_index(drop = True)


basepaths = ['protozoa','vertebrate_mammalian', 'vertebrate_other', 'viral', 'plant', 'invertebrate', 'fungi', 'archaea']
genome_table = pd.concat([pd.read_csv(f'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/{x}/assembly_summary.txt', 
                comment='##', sep = '\t', engine = 'python').assign(phylopath=f'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/{x}/') for x in basepaths])
genome_table['ftp_path2'] = genome_table.phylopath + genome_table.organism_name.str.replace(' ', '_') + '/all_assembly_versions/'

d.read_csv(f'https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt',  skiprows=0, sep = '\t', header = None)