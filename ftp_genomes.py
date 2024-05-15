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
            
