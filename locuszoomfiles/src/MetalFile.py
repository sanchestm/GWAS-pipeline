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

import sys
import decimal
import re
from m2zutils import *

def safe_long(x):
  try:
    x = long(x)
  except:
    x = None

  return x

def safe_int(x):
  try:
    x = int(x)
  except:
    x = None

  return x

def findCol(header_elements,col_name):
  for i in xrange(len(header_elements)):
    if header_elements[i] == col_name:
      return i

  return None

class MetalFile:
  def __init__(self,file=None):
    self.delim = "\t"
    self.data = None
    self.snp_pos = {}
    self.snp_index = {}
    self.snpcol = None
    self.pvalcol = None
    self.use_decimal = False; # arbitrary precision p-values
    self.logpval = False; # are p-values passed in as -log p-values? 
    if file != None:
      self.load(file)

  def load(self,file,pval_cutoff=None):
    if isinstance(file,str):
      f = open(file)
    else:
      f = file

    header = f.readline().split(self.delim)
    header[-1] = header[-1].rstrip()
    ncols = len(header)
    results = []

    # Find snp column.
    snp_col = None; 
    if self.snpcol != None:
      if type(self.snpcol) == type(str()):
        snp_col = findCol(header,self.snpcol)
      elif type(self.snpcol) == type(int()):
        snp_col = self.snpcol
      else:
        raise ValueError, "Error: marker column specified with something other than a string or integer: %s" % str(self.snpcol)
    else:
      try_names = [ 'MarkerName','markername' ]
      snp_col = None
      for name in try_names:
        if name in header:
          snp_col = header.index(name)
          break

    # After all that, we still couldn't find the snp column. Fail..
    if snp_col == None:
      raise ValueError, "Error: could not locate SNP column in metal file %s" % str(file)

    # Find p-value column.
    pval_col = None
    if self.pvalcol != None:
      if type(self.pvalcol) == type(str()):
        pval_col = findCol(header,self.pvalcol)
      elif type(self.pvalcol) == type(int()):
        pval_col = self.pvalcol
      else:
        raise ValueError, "Error: pval column specified with something other than a string or integer: %s" % str(self.pvalcol); 
    else: 
      try_pvals = [ 'P.value','P-value','Pvalue','pval','pvalue','p-value' ]
      pval_col = None
      for pval in try_pvals:
        if pval in header:
          pval_col = header.index(pval)
          break

    # We still couldn't find the p-value column. FAIL!
    if pval_col == None:
      raise ValueError, "Error: could not locate p-value column in metal file %s" % str(file)

    # Find chr column. 
    try_chrs = [ 'chr','Chr','Chromosome','chrom','Chrom' ]
    chr_col = None
    for chr in try_chrs:
      if chr in header:
        chr_col = header.index(chr)
        break
    if chr_col == None:
      die("Error: could not locate chr column in metal file %s" % str(file))
      
    # Find pos column. 
    try_pos = [ 'pos','position' ]
    pos_col = None
    for pos in try_pos:
      if pos in header:
        pos_col = header.index(pos)
        break
    if pos_col == None:
      die("Error: could not locate pos column in metal file %s" % str(file))

    # Load file.
    row = 0
    for line in f:
      # Skip blank lines. 
      if line.rstrip() == "":
        continue
      
      e = line.split(self.delim)
      e[-1] = e[-1].rstrip()

      if len(e) != ncols:
        print >> sys.stderr, "Error: line had %i columns, should have %i:" % (len(e),ncols)
        sys.stderr.write(line)
        sys.exit(1)

      snp_name = e[snp_col]
      try:
        if self.use_decimal:
          pval = decimal.Decimal(e[pval_col])
        else:
          pval = float(e[pval_col])
      except:
        continue; # skip SNP if p-value is invalid
      chrom = chrom2chr(e[chr_col])
      position = safe_long(e[pos_col])
      if pval_cutoff != None:
        if use_decimal:
          pval_cutoff = decimal.Decimal(pval_cutoff)
        
        if pval < pval_cutoff:
          results.append((
            snp_name,
            pval,
            chrom,
            position
          ))

          self.snp_pos.setdefault(snp_name,(chrom,position))
          self.snp_index.setdefault(snp_name,row)
          row += 1

      else:
        results.append((
          snp_name,
          pval,
          chrom,
          position
        ))

        self.snp_pos.setdefault(snp_name,(chrom,position))
        self.snp_index.setdefault(snp_name,row)
        row += 1

    f.close()
 
    self.data = results
    self._sortByPosition()

  def getPos(self,snp):
    return self.snp_pos.get(snp)

  def getPval(self,snp):
    return self.data[self.snp_index.get(snp)][1]

  def getSNPIndex(self,snp):
    return self.snp_index.get(snp)
  
  def getPosTable(self):
    return self.snp_pos
  
  def hasSNP(self,snp):
    return self.snp_index.has_key(snp)
  
  def iter_snps(self):
    return iter(self.snp_index)

  def write(self,file=sys.stdout,delim="\t"):
    print >> file, delim.join(['snp','pval','chr','pos'])
    for line in self.data:
      print >> file, delim.join([str(i) for i in line])

  def _sortByPosition(self):
    def cmp_genome(x,y):
      if x[2] > y[2]:
        return 1
      elif x[2] < y[2]:
        return -1
      else:
        if x[3] == y[3]:
          return 0
        elif x[3] > y[3]:
          return 1
        else:
          return -1

    self.data = sorted(self.data,cmp=cmp_genome)

    # Fix SNP index now that we've sorted.
    self.snp_index = {}
    for i in xrange(len(self.data)):
      self.snp_index.setdefault(self.data[i][0],i)

  # Returns (best SNP, p-value for best SNP) in a genomic region. 
  def getBestSNPInRegion(self,chr,start,stop):
    chr = int(chr)
    start = int(start)
    stop = int(stop)

    best_snp = None
    if not self.logpval:
      if self.use_decimal:
        best_pval = decimal.Decimal("1")
      else:
        best_pval = 1
      for row in self.data:
        if (row[2] == chr) and (row[3] <= stop) and (row[3] >= start):
          if row[1] < best_pval:
            best_snp = row[0]
            best_pval = row[1]
    else:
      if self.use_decimal:
        best_pval = decimal.Decimal(0)
      else:
        best_pval = 0
      for row in self.data:
        if (row[2] == chr) and (row[3] <= stop) and (row[3] >= start):
          if row[1] > best_pval:
            best_snp = row[0]
            best_pval = row[1]
      
    return (best_snp,best_pval)

  # Should probably filter by p-value before using this..
  # SLOPPY : should be rewritten at some point
  def dist_filter(self,dist):
    data = self.data
    data.sort(key=lambda x: float(x[1]))
    
    keepers = set()
    chr_pos = {}; 
    for line in data:
      chr = line[2]
      chr_pos.setdefault(chr,[]).append(list(line))

    for chr in chr_pos:
      pos_snps = chr_pos[chr]
      pos_snps = sorted(pos_snps,key = lambda x: int(x[3]))

      block = 0
      pos_snps[0].append(block)

      for i in xrange(1,len(pos_snps)):
        if int(pos_snps[i][3]) - int(pos_snps[i-1][3]) > dist:
          block += 1
          pos_snps[i].append(block)
        else:
          pos_snps[i].append(block)

      block_best = {}
      for entry in pos_snps:
        cur_block = entry[4]
        cur_block_best = block_best.get(cur_block)
        if cur_block_best == None:
          block_best[cur_block] = entry
        else:
          if float(cur_block_best[1]) > float(entry[1]):
            block_best[cur_block] = entry

      for b in block_best.itervalues():
        keepers.add(b[0])

    self.data = filter(lambda x: x[0] in keepers,data)

  def snp_filter(self,snps):
    self.data = filter(lambda x: x[0] not in snps,self.data)
