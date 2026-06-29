import subprocess
import logging
import re

def printwithlog(string):
    print(string)
    logging.info(string)

def bash(call, verbose = 0, return_stdout = True, print_call = True,silent = False,shell = False):
    call = re.sub(r' +', ' ', call).strip(' ')
    if print_call: printwithlog(call)
    out = subprocess.run(call if shell else call.split(' '), capture_output = True,  shell =shell) 
    if verbose and (not return_stdout) and (not silent): printwithlog(out.stdout)
    if out.stderr and (not silent): 
        try:printwithlog(out.stderr.decode('ascii'))
        except: printwithlog(out.stderr.decode('utf-8'))
    if return_stdout: 
        try: oo =  out.stdout.decode('ascii').strip().split('\n')
        except: oo =  out.stdout.decode('utf-8').strip().split('\n')
        return oo
    return out

def translate_dict(s, d):
    return re.sub(f"({')|('.join(d.keys())})",lambda y: d[str(y.group(0))] ,s )