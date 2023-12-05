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
import pdb
from LDRegionCache import *
from m2zutils import *

class FugueSettings:
  def __init__(self,map_dir,ped_dir,fugue_path):
    # Check each path. If it doesn't exist, try to find it relative
    # to the m2zfast root directory. 
    for arg,value in locals().items():
      if arg == 'self':
        continue
      
      path = find_systematic(value)
      if path == None or not os.path.exists(path):
        if arg == "fugue_path":
          die("Error: cannot find new_fugue - please set the path in the configuration file, or make it available on your PATH.")
        else:
          die("Error: path either does not exist or insufficient permissions to access it: %s" % str(value))
      else:
        exec "%s = \"%s\"" % (arg,path)
    
    self.map_dir = map_dir
    self.ped_dir = ped_dir
    self.fugue_path = fugue_path

  def createLDCacheKey(self):
    key_string = \
      self.map_dir + \
      self.ped_dir

    key = hashlib.sha512(key_string).hexdigest()
    return key

class FugueFinder():
  def __init__(self,fugue_settings,cache=None,cleanup=True,verbose=False):
    if not isinstance(fugue_settings,FugueSettings):
      raise ValueError
    
    self.data = {}
    self.snp = None
    self.settings = fugue_settings
    self.debug = False
    self.start = None
    self.stop = None
    self.chr = None
    self.cache = cache
    self.cleanup = cleanup
    self.verbose = verbose

  def getStart(self):
    return self.start

  def getStop(self):
    return self.stop

  def getLD(self,other_snp):
    return self.data.get(other_snp)

  #snp1       snp2     CHR MIDPOINT DISTANCE  dprime rsquare
  #rs882020   rs217386   7 44356244   421952 0.17240 0.00305
  # Function returns true if data was available to write and was written, 
  # false otherwise. 
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

  def _runsequence(self,snp,chr,start,stop):
    # Use new_fugue to compute LD in a region with this SNP.
    (hacked_dat,hacked_map,found_snp) = self._makeFugueFiles(chr,start,stop)
    
    # If the SNP wasn't found in the LD files, we can't compute LD with it.
    if found_snp:
      ld_file = self._runNewFugue(chr,hacked_dat,hacked_map,snp)
  
      # Load LD.
      data = self._loadLD(ld_file)
  
      # Cleanup files.
      if self.cleanup:
        os.remove(hacked_dat)
        os.remove(hacked_map)
        os.remove(ld_file + ".xt")
        
      return data
    
    else:
      return None

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
        self.data = self._runsequence(snp,chr,start,stop)
        self.cache.updateLD(snp,start,stop,self.data)
    else:
      self.data = self._runsequence(snp,chr,start,stop)

    # Complete successfully?
    if self.data == None or len(self.data) == 0:
      return False
    else:
      return True

  def _loadLD(self,fugue_xt):
    # Load LD data into memory.
    f = open(fugue_xt + ".xt")
    data = {}

    f.readline()
    for line in f:
      e = line.split()
      snp1 = e[0]
      snp2 = e[1]

      if snp1 == self.snp:
        data[snp2] = (float(e[5]),float(e[6]))
      elif snp2 == self.snp:
        data[snp1] = (float(e[5]),float(e[6]))
      else:
        raise Exception
    f.close()

    return data

  def _makeFugueFiles(self,chr,start,end):
    # Open map file. 
    map_file = None
    possible_map_files = [
      os.path.join(self.settings.map_dir,"chr" + chr2chrom(chr) + ".map"),
      os.path.join(self.settings.map_dir,"chr" + chr2chrom(chr) + ".map.gz")
    ]
    for file in possible_map_files:
      if os.path.isfile(file):
        map_file = file
    if map_file == None:
      msg = "Error: could not find map file, tried the following: \n"
      for file in possible_map_files:
        msg += "%s\n" % file
      msg += "\n"
      die(msg)
    
    # Create dat file. 
    temp_dat = "temp_fugue_dat_chr%s_%s-%s" % (str(chr),str(start),str(end))
    temp_map = "temp_fugue_map_chr%s_%s-%s" % (str(chr),str(start),str(end))
    dat = open(temp_dat,"w")
    map_out = open(temp_map,"w")
    
    if os.path.splitext(map_file)[1] == ".gz":
      map = gzip.open(map_file)
    else:
      map = open(map_file)

    found_snp = False
    for line in map:
      e = line.split()
      map_chr = chrom2chr(e[0])
      map_pos = int(e[2])
      map_snp = e[1]
      map_snp_chrpos = "chr%s:%s" % (str(map_chr),str(map_pos));      

      if map_snp_chrpos == self.snp:
        found_snp = True
    
      # Check to see if this row's SNP is within our "compute LD region." 
      if map_pos <= end and map_pos >= start:
        print >> dat, "M %s" % map_snp_chrpos
      else:
        print >> dat, "S2 %s" % map_snp_chrpos

      # Write out map file. 
      print >> map_out, "%i %s %i" % (map_chr,map_snp_chrpos,map_pos)
    
    map.close();  
    dat.close()
    map_out.close()
    
    return (temp_dat,temp_map,found_snp)

  def _runNewFugue(self,chr,fixed_dat,fixed_map,ref_snp):
    # Required files for new_fugue. 
    ped_loc = os.path.join(self.settings.ped_dir,"chr" + chr2chrom(chr) + ".ped")

    if not os.path.isfile(ped_loc):
      ped_loc = os.path.join(self.settings.ped_dir,"chr" + chr2chrom(chr) + ".ped.gz")

    if not os.path.isfile(ped_loc):
      die("Error: could not find map or ped file for chrom %s." % str(chr))

    # Create temporary file for LD generated by new_fugue.
    ld_loc = "templd_newfugue_%s" % ref_snp

    # Command to run new_fugue.
    new_fugue = self.settings.fugue_path
    if chr == 'X' or chr == 23:
      new_fugue += 'X'
    
    com = "%s --quiet --diseq --window 99999999999999999 -m %s -d %s -p %s -o %s --names --minrsq 0 --pairWith %s" % (
      new_fugue,
      fixed_map,
      fixed_dat,
      ped_loc,
      ld_loc,
      ref_snp
    )

    if self.verbose:
      print "Executing new_fugue: %s" % com
      proc = subprocess.Popen(com,shell=True)
    else:
      proc = subprocess.Popen(com,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    #proc.wait()
    #os.waitpid(proc.pid,0)
    proc.communicate()

    if proc.returncode != 0:
      print >> sys.stderr, "Error: new_fugue did not complete successfully. Check logs for more information."
      sys.exit(1)

    if self.cleanup:
      delete_files = [ld_loc + i for i in (".freq",".maf",".xt.log")]
      for file in delete_files:
        try:
          os.remove(file)
        except:
          pass

    return ld_loc
