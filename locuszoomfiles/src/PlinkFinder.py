#!/usr/bin/env python

#===============================================================================
# Copyright (C) 2010 Ryan Welch, Randall Pruim
# 
# LocusZoom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# LocusZoom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#===============================================================================

import gzip
import os
import subprocess
import sys
import hashlib
import re
from LDRegionCache import *
from m2zutils import *
from textwrap import fill

class PlinkSettings:
  def __init__(self,bim_path,plink_path):
    # Check each path. If it doesn't exist, try to find it relative
    # to the m2zfast root directory. 
    for arg,value in locals().items():
      if arg == 'self':
        continue
      
      path = find_systematic(value)
      if path == None or not os.path.exists(path):
        if arg == "plink_path":
          die("Error: cannot find plink - please set the path in the configuration file, or make it available on your PATH.")
        else:
          die("Error: path either does not exist or insufficient permissions to access it: %s" % str(value))
      else:
        exec "%s = \"%s\"" % (arg,path)
    
    self.bim_path = bim_path
    self.plink_path = plink_path

  def createLDCacheKey(self):
    key_string = self.bim_path
    key = hashlib.sha512(key_string).hexdigest()
    return key

class PlinkFinder():
  def __init__(self,plink_settings,cache=None,cleanup=True,verbose=False):
    if not isinstance(plink_settings,PlinkSettings):
      raise ValueError
    
    self.data = {}
    self.snp = None
    self.settings = plink_settings
    self.debug = False
    self.start = None
    self.stop = None
    self.chr = None
    self.cache = cache
    self.cleanup = cleanup
    self.verbose = verbose

  def write(self,filename):
    try:
      f = open(filename,'w')
      print >> f, "snp1 snp2 dprime rsquare"
  
      if len(self.data) == 0:
        return False
  
      for snp in self.data:
        print >> f, "%s %s %s %s" % (
          snp,
          self.snp,
          str(self.data.get(snp)[0]),
          str(self.data.get(snp)[1])
        )
  
      f.close()
    except:
      print >> sys.stderr, "Error: could not write computed LD to disk, permissions?"
      return False
    
    return True

  def _check_geno_paths(self):
    chr = self.chr
    bim_file = os.path.join(self.settings.bim_path,"chr" + chr2chrom(chr) + ".bim")
    bed_file = os.path.join(self.settings.bim_path,"chr" + chr2chrom(chr) + ".bed")
    fam_file = os.path.join(self.settings.bim_path,"chr" + chr2chrom(chr) + ".fam")
    
    for file in (bim_file,bed_file,fam_file):
      if not os.path.exists(file):
        msg = fill("Error: could not find required file to generate LD using "
                  "PLINK: %s. Check your conf file to make sure paths are "
                  "correct. " % file)
        die(msg)

  def _run_sequence(self):
    self._check_geno_paths()
    ld_file = self._run_plink()

    if ld_file is None:
      return

    # Load LD.
    data = self._loadLD(ld_file)

    # Cleanup files.
    if self.cleanup:
      try:
        os.remove(ld_file)
      except:
        pass
      
    return data

  def compute(self,snp,chr,start,stop):
    self.snp = snp
    self.start = start
    self.stop = stop
    self.chr = chr
    self.data = None

    # If the cache has data for this SNP and region, use it.
    # Otherwise, compute it.
    if self.cache:
      if self.cache.hasRegion(snp,start,stop):
        self.data = self.cache.getAllLD(snp)
      else:
        self.data = self._run_sequence()
        self.cache.updateLD(snp,start,stop,self.data)
    else:
      self.data = self._run_sequence()

    # Complete successfully?
    if self.data == None or len(self.data) == 0:
      return False
    else:
      return True

  def _loadLD(self,ld_file):
    # Load LD data into memory.
    f = open(ld_file)
    data = {}

    f.readline()
    for line in f:
      e = line.split()
      snp1 = e[2]
      snp2 = e[5]

      if snp1 == self.snp:
        data[snp2] = ("NA",float(e[6]))
      elif snp2 == self.snp:
        data[snp1] = ("NA",float(e[6]))
      else:
        raise Exception
    f.close()

    return data

  def _run_plink(self):
    # Fix chromosome name for looking up file. 
    chrom = chr2chrom(self.chr)

    bfile_path = os.path.join(self.settings.bim_path,"chr" + chrom)

    com = "%s --bfile %s --chr %s --from-bp %s --to-bp %s --ld-snp %s --r2 --ld-window-r2 0 --ld-window 999999 --ld-window-kb 99999 --noweb" % (
      self.settings.plink_path,
      bfile_path,
      self.chr,
      self.start,
      self.stop,
      self.snp
    )

    if self.verbose:
      print "Executing PLINK: %s" % com
      proc = subprocess.Popen(com,shell=True)
    else:
      proc = subprocess.Popen(com,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    (stdout, stderr) = proc.communicate()

    ld_loc = os.path.join(os.getcwd(),"plink.ld")
    
    if proc.returncode != 0:
      log_file = os.path.join(os.getcwd(),"plink.log")

      bMarkerMissing = _check_log_marker(log_file)
      if not bMarkerMissing:
        print >> sys.stderr, "Error: PLINK did not complete successfully. Please check the log file (run with --no-cleanup to see the directory with the log file.)"

      ld_loc = None

    if self.cleanup:
      delete_files = ['plink.log']
      for file in delete_files:
        try:
          os.remove(file)
        except:
          pass

    return ld_loc

def _check_log_marker(log_file):
  try:
    data = open(log_file).read()
  except:
    return False

  matches = re.findall("ERROR: --ld-snp.*not found",data)
  if len(matches) > 0:
    return True
  else:
    return False

