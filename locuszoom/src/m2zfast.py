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

import os
import sys

# Fix path of script to be absolute.
sys.argv[0] = os.path.abspath(sys.argv[0])

# Add the locuszoom bin/ to the PATH. 
LZ_ROOT = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]),".."))
LZ_BIN = os.path.join(LZ_ROOT,"bin")
os.environ['PATH'] = LZ_BIN + os.pathsep + os.environ['PATH']

import time
import re
import tempfile
import platform
import math
import shlex
import json
from m2zutils import *
from FugueFinder import *
from PlinkFinder import *
from LDRegionCache import *
from pquery import *
from glob import glob
from optparse import OptionParser, SUPPRESS_HELP
from subprocess import *
from shutil import move,rmtree
from prettytable import *
from textwrap import fill
from vcf_ld import *
from verboseparser import *
from cStringIO import StringIO
from ordered_set import OrderedSet

# Try importing modules that may not exist on a user's machine. 
try:
  import gzip
except:
  print >> sys.stderr, "Warning: gzip not available on this system"

try:
  import bz2
except:
  print >> sys.stderr, "Warning: bz2 not available on this system"

try:
  import sqlite3
except:
  print >> sys.stderr, "Error: your python interpeter is not compiled against sqlite3 (is it really out of date?)"
  raise

# Program strings.
M2ZFAST_VERSION = "1.4"
M2ZFAST_DATE = "05/01/2017"

def repeat_char(s,n):
  from itertools import repeat
  return "".join(repeat(s,n))

def table_pad(lines):
  maxw = 0
  for l in lines:
    maxw = max(len(l),maxw)
  pad = 2
  print("+" + repeat_char("-",maxw+3) + "+")
  for l in lines:
    print("| {} {} |".format(l,repeat_char(" ",maxw - len(l))))
  print("+" + repeat_char("-",maxw+3) + "+")

M2ZFAST_TITLE = [
  "LocusZoom {} ({})".format(M2ZFAST_VERSION,M2ZFAST_DATE),
  "Plot regional association results",
  "from GWA scans or candidate gene studies"
]

# Program constants. 
M2ZFAST_CONF = "conf/m2zfast.conf"
DEFAULT_SNP_FLANK = "200kb"
DEFAULT_GENE_FLANK = "20kb"
RE_SNP_1000G = re.compile("(chr)?([0-9a-zA-z]+):([0-9]+).*")
RE_SNP_RS = re.compile("rs(\d+)")
M2ZCL_FIRST = False
MULTI_CAP = 8; 

# Database constants. 
SQLITE_SNP_POS = "snp_pos"
SQLITE_REFFLAT = "refFlat"
SQLITE_KNOWNGENE = "knownGene"
SQLITE_GENCODE = "gencode"
SQLITE_SNP_SET = "snp_set"
SQLITE_VAR_ANNOT = "var_annot"
SQLITE_RECOMB_RATE = "recomb_rate"
SQLITE_TRANS = "refsnp_trans"

# Debug flag. Makes all output "more verbose."
_DEBUG = False

class Conf(object):
  def __init__(self,conf_file):
    self._load(conf_file)

  def _load(self,file):
    conf_dict = {}
    execfile(file,conf_dict)

    for k,v in conf_dict.iteritems():
      exec "self.%s = v" % str(k)

def getConf(conf_file=M2ZFAST_CONF):
  conf_file = find_relative(conf_file)
  conf = Conf(conf_file)
  return conf

# Get LD file information given parameters.
# Returns a dictionary with 4 elements:
# ped_dir, map_dir, dat_dir, pos_file
def getLDInfo(pop,source,build,ld_db):
  if source in ld_db:
    node = ld_db[source]
    if build in node:
      node = node[build]
      if pop in node:
        # Success! File paths are known for this DB/build/population trio.
        return node[pop]

  # If we make it here, the supplied combination of source/build/pop is not supported. 
  return None

def getGWASCat(build,code,gwas_cats):
  if build in gwas_cats:
    node = gwas_cats[build]
    if code in node:
      subnode = node[code]
      return subnode['file']

  return None

# Print a table of all available GWAS catalogs.
def printGWACatalogs(cat_db,build):
  print "Available GWAS catalogs for build %s:" % build
  print ""

  tree = cat_db.get(build)
  if tree is None:
    print "-- No catalogs available for this build."
    return

  table = PrettyTable(['Option','Description'])
  table.set_field_align('Option','l')
  table.set_field_align('Description','l')

  exists = False;  
  for cat_code, cat_tree in tree.iteritems():
    table.add_row([
      cat_code,
      cat_tree['desc']
    ])
    exists = True
    
  if exists:
    table.printt()
  else:
    print "-- No valid options for this build."

def getAllGWACatalogs(cat_db):
  table = PrettyTable(['Build','--gwas-cat','Description'])
  table.set_field_align('Build','l')
  table.set_field_align('Description','l')
  table.set_field_align('--gwas-cat','l')
 
  exists = False;  
  for build in cat_db:
    tree = cat_db.get(build)

    for cat_code, cat_tree in tree.iteritems():
      table.add_row([
        build,
        cat_code,
        cat_tree['desc']
      ])
      
      exists = True
      
  if exists:
    return table.get_string()
  else:
    return; 

# Print a list of all supported population/build/database combinations. 
def printSupportedTrios(ld_db):
  print "Genotype files available for: "
  print getSupportedTrios(ld_db)

def getSupportedTrios(ld_db):
  lines = []
  for source in ld_db:
    lines += ["--source " + source]
    for build in ld_db[source]:
      lines += ["  --build %s" % build]
      for pop in ld_db[source][build]:
        lines += ["    --pop %s" % pop]
  
    lines += [""]
  
  lines.pop()
  s = os.linesep.join(lines)
  
  return s

def parse_rargs(arg_list):
  args = map(str.strip," ".join(arg_list).split("="))

  new_args = {}

  try:
    for i in xrange(1,len(args)):
      k = args[i-1].split()[-1]

      vlist = args[i].split()
      if len(vlist) == 1:
        v = vlist[0]
      else:
        v = " ".join(vlist[0:-1])

      new_args[k] = v
  except:
    raise Exception, ("Error: something wrong with your plotting arguments. They "
                      "should all be of the form arg=value, no spaces, value should "
                      "be quoted if it has spaces. ")

  return new_args

#def parse_rargs(args):
#  result = []
#  for i in xrange(len(args)):
#    if args[i] != "=" and args[i].find("=") != -1:
#      result.append(args[i])
#    elif args[i] == "=":
#      result.append(args[i-1] + "=" + args[i + 1])
#      
#  d = dict()
#  for e in result:
#    (arg, val) = e.split("=")
#    d[arg] = val
#  return d

def quoteArgs(m2z_args):
  new_args = []
  for arg in m2z_args:
    test = arg.split("=")
    if len(test) == 2:
      test[1] = "\"%s\"" % test[1]
    new_args.append("=".join(test))

  return new_args

# Tests to see if the supplied string is a SNP.
# The SNP should be of the form: rs##### or chr#:pos.
def isSNP(string):
  string = str(string)
  if RE_SNP_RS.search(string):
    return True
  elif RE_SNP_1000G.search(string):
    return True
  else:
    return False

def isRSID(string):
  p = re.compile(r"^rs(\d+?)$")
  if p.search(string) is not None:
    return True
  else:
    return False

# Parse a 1000G SNP into chromosome/position.
# Example: chr4:172274 --> (4,172274)
def parse1000G(snp):
  if snp is None:
    return None

  c = snp.split(":")
  if len(c) == 2:
    chrom = c[0]
    pos = c[1]

    if "_" in pos:
      pos = pos.split("_")[0]

    chrom = chrom2chr(chrom)
    try:
      pos = long(pos)
    except:
      return None
    
    return (chrom,pos)
  else:
    return None

# Pretty string for chromosome/start/stop interval. 
def regionString(chr,start,end):
  return "chr%s:%s-%s" % (str(chr),str(start),str(end))

# Delete a directory and all files underneath. 
def kill_dir(d):
  def report_error(function,path,excinfo):
    msg = None
    if excinfo is not None:
      msg = str(excinfo[1])
      
    print >> sys.stderr, "Error: could not remove: %s, message was: %s" % (
      path,
      msg
    )
    
  if os.path.isdir(d):
    rmtree(d,onerror=report_error)
  else:
    print >> sys.stderr, "Error: could not remove %s, not a directory." % d

# Pretty string for a given number of seconds. 
def timeString(seconds):
  tuple = time.gmtime(seconds)
  days = tuple[2] - 1
  hours = tuple[3]
  mins = tuple[4]
  secs = tuple[5]
  if sum([days,hours,mins,secs]) == 0:
    return "<1s"
  else:
    string = str(days) + "d"
    string += ":" + str(hours) + "h"
    string += ":" + str(mins) + "m"
    string += ":" + str(secs) + "s"
  return string

def transSNP(snp,db_file):
  # If this isn't a rs# SNP, it has no translation. 
  if not isRSID(snp):
    return snp

  con = sqlite3.connect(db_file)
  cur = con.execute("SELECT * FROM %s where rs_orig='%s'" % (SQLITE_TRANS,snp))

  new_name = None
  while 1:
    d = cur.fetchone()
    if d is not None:
      new_name = d[1]
    else:
      break

  if cur.rowcount > 1:
    print >> sys.stderr, "Warning: SNP %s had multiple names in the latest build.." % snp

  if new_name is None:
    print >> sys.stderr, "Warning: tried to translate SNP %s to latest name in genome build, but it does not exist in the database table.." % str(snp)
  elif new_name == snp:
    return snp
  else:
    print >> sys.stderr, "Warning: %s is not the current name in genome build (should be: %s)" % (snp,new_name)
    return new_name

# Given a gene, return info for it from the database. 
def findGeneInfo(gene,db_file):
  con = sqlite3.connect(db_file)
  cur = con.execute("SELECT chrom,txStart,txEnd,cdsStart,cdsEnd FROM %s WHERE geneName='%s'" % (SQLITE_REFFLAT,gene))

  row = None
  while 1:
    d = cur.fetchone()
    if d is None:
      break
    
    # Turn row into dictionary object.
    d = dict(zip([i[0] for i in cur.description],d))
    
    # This block of code basically picks out the largest isoform
    # as the one we'll use for txstart/txend.
    if row is None:
      row = d
    else:
      if abs(row['txEnd']-row['txStart']) < abs(d['txEnd']-d['txStart']):
        row = d

  # Fix chromosome if possible.
  if row is not None:
    chrom = chrom2chr(row['chrom'][3:])
    if chrom is None:
      raise ValueError, "Error: refgene found on non-supported chromosome: %s" % str(row['chrom'])
    else:
      row['chrom'] = chrom

  # Return row, with fixed chromosome (see chrom2chr function.) 
  return row

class PosLookup:
  def __init__(self,db_file):  
    if not os.path.isfile(db_file):
      sys.exit("Error: could not locate SQLite database file: %s. Check conf file setting SQLITE_DB." % db_file)
      
    self.db = sqlite3.connect(db_file)
    self.execute = self.db.execute
    
    self.execute("""
      CREATE TEMP VIEW snp_pos_trans AS SELECT rs_orig as snp,chr,pos FROM %s p INNER JOIN %s t ON (t.rs_current = p.snp)
    """ % (SQLITE_SNP_POS,SQLITE_TRANS))
    
    self.query = """
      SELECT snp,chr,pos FROM snp_pos_trans WHERE snp='%s'
    """

  def __call__(self,snp):
    snp = str(snp)
    
    # If the SNP is a 1000G SNP, it already knows its chrom/pos by definition,
    # i.e. the SNP will be chr4:91941. 
    gcheck = parse1000G(snp)
    if gcheck:
      return gcheck
    
    cur = self.execute(self.query % snp)
    chr = None
    pos = None

    res = 0
    for row in cur:
      chr = row[1]
      pos = row[2]
      res += 1

    region = "chr%s:%s" % (chr,pos)
    if res > 1:
      print >> sys.stderr, "Warning: SNP %s has more than 1 position in database, using: %s" % (str(snp),region)

    return (chr,pos)

# Given a list of header elements, determine if col_name is among them. 
def findCol(header_elements,col_name):
  for i in xrange(len(header_elements)):
    if header_elements[i] == col_name:
      return i

  return None

def is_gzip(file):
  b = False
  try:
    f = gzip.open(file)
    f.read(1024)
    f.close()
    b = True
  except:
    pass
  finally:
    f.close()
  
  return b
  
def is_bz2(file):
  try:
    f = bz2.BZ2File(file)
  except:
    return False

  try:
    f.read(1024)
    b = True
  except:
    b = False
  finally:
    f.close()

  return b

# Given a metal file, this function extracts the region from the file between
# chr/start/end.
def read_metal(metal_file,snp_column,pval_column,no_transform,chr,start,end,db_file,delim):
  region = "chr%s:%s-%s" % (str(chr),start,end)
  output_file = "temp_metal_%s_%s.txt" % (region.replace(":","_"),tempfile.mktemp(dir=""))
  
  con = sqlite3.connect(db_file)
  query = """
    SELECT rs_orig as snp,chr,pos
    FROM %(snp_table)s p
    INNER JOIN %(trans_table)s t on (t.rs_current = p.snp)
    WHERE chr = %(chr)i AND pos < %(end)i AND pos > %(start)i
  """ % {'snp_table':SQLITE_SNP_POS,'trans_table':SQLITE_TRANS,'chr':chr,'end':end,'start':start}
  cur = con.execute(query)
  
  sptable = {}
  while 1:
    row = cur.fetchone()
    if row is not None:
      sptable.setdefault(row[0],(int(row[1]),int(row[2])))
    else:
      break

  cur.close()

  # Open file for reading. Attempt to determine if file is compressed before opening.
  if metal_file == "-":
    f = sys.stdin
  else:
    if is_gzip(metal_file):
      try:
        f = gzip.open(metal_file); # throws exception if gz not on system
      except:
        die("Error: gzip is not supported on your system, cannot read --metal file.")
    elif is_bz2(metal_file):
      try:
        f = bz2.BZ2File(metal_file,"rU")
      except NameError:
        die("Error: bz2 is not supported on your system, cannot read --metal file.")
    else:
      f = open(metal_file,"rU")

  # Find snp column.
  metal_header = f.next().split(delim)
  metal_header[-1] = metal_header[-1].rstrip()

  snp_col = None; 
  if snp_column is not None:
    if type(snp_column) == type(str()):
      snp_col = findCol(metal_header,snp_column)
    elif type(snp_column) == type(int()):
      snp_col = snp_column
    else:
      die("Error: marker column specified with something other than a string or integer: %s" % str(snp_column))

  # After all that, we still couldn't find the snp column. Fail..
  if snp_col is None:
    msg = "Error: could not locate SNP column in data. You may need to specify "\
          "it using --markercol <snp column name>. Your delimiter should also "\
          "be specified if it is not a tab by using --delim."
    die(msg)

  # Find p-value column.
  pval_col = None
  if pval_column is not None:
    if type(pval_column) == type(str()):
      pval_col = findCol(metal_header,pval_column)
    elif type(pval_column) == type(int()):
      pval_col = pval_column
    else:
      die("Error: pval column specified with something other than a string or integer: %s" % str(pval_column)); 

  # We still couldn't find the p-value column. FAIL!
  if pval_col is None:
    die("Error: could not locate p-value column in data, column name I'm looking for is: %s. Is your delimiter correct?" % (pval_column))
  
  out = open(output_file,"w")
  print >> out, "\t".join(["chr","pos"] + metal_header)
  format_str = "\t".join(["%i","%i"] + ["%s" for i in xrange(len(metal_header))])
  
  # P-value check functions
  pval_checks = [
    lambda x: x.is_finite()
  ]

  if not no_transform:
    # If we're transforming to -log10, p-values coming in should be > 0. 
    pval_checks.append(lambda x: x > 0)
  
  found_in_region = False
  found_chrpos = False
  min_snp = None
  min_pval = decimal.Decimal(1)
  marker_count = 0
  invalid_pval_count = 0
  for line in f:
    # Skip blank lines. 
    if line.rstrip() == "":
      continue
    
    e = line.split(delim)
    e[-1] = e[-1].rstrip()

    snp = e[snp_col]
    snp = snp.lower(); # sometimes people put in RS39393 or CHR9:19191...

    # Is this a 1000G SNP? If so, we can pull the position from it.
    gcheck = parse1000G(snp)
    if gcheck:
      found_chrpos = True
      gchr = gcheck[0]
      gpos = gcheck[1]
      
      if gchr == int(chr) and gpos > int(start) and gpos < int(end):
        sptable.setdefault(snp,(gchr,gpos))

    if snp in sptable:
      found_in_region = True
      marker_count += 1

      # Insert fixed log10 p-value
      pval = e[pval_col]
  
      if is_number(pval):     
        dec_pval = decimal.Decimal(pval)
        
        pval_ok = all((f(dec_pval) for f in pval_checks))
        if not pval_ok:
          print >> sys.stderr, "Warning: marker %s has invalid p-value: %s, skipping.." % (snp,str(pval))
          continue
      
        if dec_pval < min_pval:
          min_snp = snp
          min_pval = dec_pval; 
          
        if not no_transform:
          e[pval_col] = str(-1*dec_pval.log10())
        
        (schr,spos) = sptable.get(snp)
        e[snp_col] = "chr%i:%i" % (schr,spos)
        elements = (schr,spos) + tuple(e)
        print >> out, format_str % elements
      else:
        invalid_pval_count += 1

  f.close()
  out.close()

  if invalid_pval_count > 0:
    print >> sys.stderr, "Warning: of %i markers in the region, %i had invalid or missing p-values" % (marker_count,invalid_pval_count)
  
  if found_chrpos:
    print >> sys.stderr, ""
    print >> sys.stderr, fill("WARNING: your association results file has "
                          "chr:pos SNP names. Please make sure you have selected the "
                          "correct genome build by using the --build parameter, or by "
                          "selecting the appropriate build on the website.")
    print >> sys.stderr, ""

  return found_in_region, output_file, min_snp

# Given an EPACTS file, this function extracts the region from the file between
# chr/start/end.
def read_epacts(epacts_file,chr,start,end,chr_col,beg_col,end_col,pval_col,no_transform):
  conf = getConf()
  region = "%s:%s-%s" % (str(chr),start,end)
  output_file = "temp_epacts_%s_%s.txt" % (region.replace(":","_"),tempfile.mktemp(dir=""))

  chr = int(chr)
  start = int(start)
  end = int(end)

  # Should we just read from STDIN?
  if epacts_file == "-":
    f = sys.stdin
  else:
    # Does the EPACTS file have a tabix index with it? If it does, we can use tabix to pull the region out and make
    # parsing much faster.
    has_index = os.path.isfile(epacts_file + ".tbi")
    tabix_path = find_systematic(conf.TABIX_PATH)

    if not has_index:
      print >> sys.stderr, "Warning: EPACTS file was given, but could not find tabix index for it. " \
                           "EPACTS should have generated a tabix index for this file."

    # Do we have both tabix, and the EPACTS file has a tabix index?
    f = None
    if has_index and tabix_path is not None:
      # Run tabix to pull out our region.
      print "Tabix found, using index to extract region.."
      proc = subprocess.Popen([tabix_path,"-h",epacts_file,region],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      stdout, stderr = proc.communicate()

      # No variants in the region...
      if stdout == '':
        raise IOError, "Error: no variants in region (%s) when using tabix on EPACTS file" % region

      # Unknown error occurred
      if stderr != '':
        raise IOError, "Error: while grabbing region from EPACTS file, tabix generated an error: \n%s" % stderr

      # Setup the input handle for reading.
      f = StringIO(stdout)

    else:
      # Open file for reading. Attempt to determine if file is compressed before opening.
      if is_gzip(epacts_file):
        try:
          f = gzip.open(epacts_file); # throws exception if gz not on system
        except:
          die("Error: gzip is not supported on your system, cannot read --epacts file.")
      elif is_bz2(epacts_file):
        try:
          f = bz2.BZ2File(epacts_file,"rU")
        except NameError:
          die("Error: bz2 is not supported on your system, cannot read --epacts file.")
      else:
        f = open(epacts_file,"rU")

  if f is None:
    raise IOError, "Unknown error while loading EPACTS file, contact developer with this traceback"

  # Find column indices. 
  epacts_header = f.next().split("\t")
  epacts_header[-1] = epacts_header[-1].rstrip()

  # Find chr/begin/end columns. 
  try:
    i_chr_col = epacts_header.index(chr_col)
    i_begin_col = epacts_header.index(beg_col)
    i_end_col = epacts_header.index(end_col)
  except:
    raise IOError, "Error: could not find chrom/begin/end columns in EPACTS file. Try specifying --epacts-chr-col, --epacts-beg-col, --epacts-end-col."

  # Find p-value column.
  try:
    i_pval_col = epacts_header.index(pval_col)
  except:
    raise IOError, "Error: could not find p-value column in EPACTS file. Try specifying with --epacts-pval-col."

  cols_not_needed = [chr_col,beg_col,end_col,pval_col] + "MARKER_ID NS AC CALLRATE GENOCNT BETA SEBETA STAT MAF SCORE N.CASE N.CTRL AF.CASE AF.CTRL".split()
  extra_cols = filter(lambda x: x not in cols_not_needed,epacts_header)
  extra_ind = [epacts_header.index(x) for x in extra_cols]
  out = open(output_file,"w")
  print >> out, "\t".join(['chr','pos','MarkerName','P-value'] + extra_cols)
  format_str = "\t".join(['%s','%i','%s','%s'] + ["%s" for _ in xrange(len(extra_cols))])
  
  # Arbitrary arithmetic precision  
  decimal.getcontext().prec = 8

  # P-value check functions
  pval_checks = [
    lambda x: x.is_finite()
  ]

  if not no_transform:
    # If we're transforming to -log10, p-values coming in should be > 0. 
    pval_checks.append(lambda x: x > 0)

  found_in_region = False
  min_snp = None
  min_pval = decimal.Decimal(1)
  skipped_markers = []
  marker_count = 0
  invalid_pval_count = 0
  for line in f:
    # Skip blank lines. 
    if line.rstrip() == "":
      continue
    
    e = line.split("\t")
    e[-1] = e[-1].rstrip()

    marker_name = "chr%s:%s" % (e[i_chr_col],e[i_begin_col])

    try:
      file_chrom = int(e[i_chr_col])
      file_begin = int(e[i_begin_col])
      file_end = int(e[i_end_col])
    except:
      print >> sys.stderr, "Warning: skipping marker %s - could not convert chr/begin/end to integers.." % marker_name
      continue

#    if file_begin != file_end:
#      skipped_markers.append("chr%s:%s-%s" % (file_chrom,file_begin,file_end))
#      continue

    if file_chrom == chr and file_begin >= start and file_end <= end:
      # Did we find a SNP in this region at all? 
      found_in_region = True
      marker_count += 1

      pval = e[i_pval_col]
  
      if is_number(pval):     
        dec_pval = decimal.Decimal(pval)

        pval_ok = all((f(dec_pval) for f in pval_checks))
        if not pval_ok:
          print >> sys.stderr, "Warning: marker %s has invalid p-value: %s, skipping.." % (marker_name,str(pval))
          continue

        if dec_pval < min_pval:
          min_snp = marker_name
          min_pval = dec_pval; 
          
        if not no_transform:
          pval = str(-1*dec_pval.log10())

        out_values = tuple([file_chrom,file_begin,marker_name,pval] + [e[x] for x in extra_ind])
        print >> out, format_str % out_values
      else:
        invalid_pval_count += 1

  if len(skipped_markers) > 0:
    print >> sys.stderr, "Warning: skipped %i markers that did not appear to be SNPs.." % len(skipped_markers)

  if invalid_pval_count > 0:
    print >> sys.stderr, "Warning: of %i markers in the region, %i had invalid or missing p-values" % (marker_count,invalid_pval_count)

  f.close()
  out.close()

  return found_in_region, output_file, min_snp

# Runs the R script which actually generates the plot. 
def runM2Z(metal,metal2zoom_path,ld_files,refsnp,chr,start,end,no_snp_name,verbose,opts,args=""):
  conf = getConf()

  rscript_path = find_systematic(conf.RSCRIPT_PATH)
  if rscript_path is None:
    die("Error: could not locate Rscript interpreter. It either needs to be located on your PATH, or set on the configuration file.")

  # If no LD file was created, make m2z use the database instead. 
  refsnp_ld = "NULL"
  cond_ld = ""
  if (ld_files is None) or (len(ld_files) == 0):
    pass
  else:
    if isinstance(ld_files,type('str')):
      refsnp_ld = ld_files
    else:
      refsnp_ld = ld_files[0]
      if len(ld_files) > 1:
        cond_ld = "cond_ld=" + ",".join(ld_files[1:])
      else:
        cond_ld = ""
  
  cond_pos = ""
  cond_snps = ""
  if opts.condsnps is not None:
    cond_pos = "cond_pos=" + ",".join(map(lambda x: x.chrpos,opts.condsnps))
    cond_snps = "cond_snps=" + ",".join(map(lambda x: x.snp,opts.condsnps))

  if no_snp_name:
    refsnpName = r'" "'
  else:
    refsnpName = refsnp.snp

  com = "%s %s metal=%s clobber=F clean=F refsnp=%s refsnpName=%s ld=%s %s %s %s chr=%s start=%s end=%s %s" % (
    rscript_path,
    metal2zoom_path,
    metal,
    refsnp.chrpos,
    refsnpName,
    refsnp_ld,
    cond_ld,
    cond_pos,
    cond_snps,
    str(chr),
    str(start),
    str(end),
    args
  )

  if _DEBUG:
    print "DEBUG: locuszoom.R command: %s" % com

  #if verbose:
  if 1:
    proc = Popen(com,shell=True)
  else:
    proc = Popen(com,shell=True,stdout=PIPE,stderr=PIPE)
    
  proc.wait()

  if proc.returncode != 0:
    die("Error: locuszoom.R did not complete successfully. Check logs for more information.")

# Attemps to delete a list of files.
# Prints a warning if a file could not be deleted, but does not throw
# an exception. 
def cleanup(files):
  if _DEBUG:
    print "cleanup() called for files: "
    for f in files:
      print ".. %s" % f
  
  def check_file(f):
    return (f is not None) and (f != '') and (f != "NULL") and (f != '/')

  files = filter(check_file,files)
  for f in files:
    try:
      os.remove(f)
    except:
      print >> sys.stderr, "Warning: failed to remove file %s" % f

# Terminates program with a message, and a list of help options.
def die_help(msg,parser):
  print >> sys.stderr, msg
  parser.print_help()
  sys.exit(1)

def safe_int(x):
  try:
    x = int(x)
  except:
    x = None

  return x

# Reads a "hitspec" or batch run configuration file. 
# The file has 6 columns: 
# 0 - snp or gene
# 1 - chromosome
# 2 - start
# 3 - stop
# 4 - flank 
# 5 - run?
# 6 - m2zargs
# 
# Each row is for a SNP or gene, specified in column 0. 
# Either chr/start/stop or flank must be specified in each row. 
# If neither is specified, a default flank is used (see DEFAULT_*_FLANK variables.) 
# Missing values should be entered as "NA"
# Column 5 is either 'yes' or 'no' denoting whether or not this snp/gene should be plotted. 
# The final column contains a list of arguments to be passed to locuszoom.R, separated by whitespace. 
# The entire file is whitespace delimited. 
def readWhitespaceHitList(file,db_file):
  if not os.path.isfile(file):
    die("Could not open hitspec file for reading - check your path.")

  f = open(file,"rU")
  h = f.readline()

  # This format should have at least 6 columns.
  if len(h.split()) < 6:
    die("Error: hitspec not formatted properly, see documentation. Should be 6 columns, found: %i " % len(h.split()))

  find_pos = PosLookup(db_file)

  snplist = []
  for line in f:
    # Skip blank lines. 
    if line.strip() == "":
      print >> sys.stderr, "Warning: skipping blank line in hitspec file.."
      continue

    e = line.split()
    e[-1] = e[-1].strip()
    
    if len(e) < 6:
      print >> sys.stderr, "Error: hitspec line not formatted properly, missing the proper number of columns on line #%i: %s" (lineno,str(line))
      continue

    if e[5] == 'no'or e[5] == "":
      print >> sys.stderr, "Skipping disabled line '%s' in hitspec file.." % " ".join(e)
      continue

    snp = e[0]
    chr = e[1]
    start = e[2]
    end = e[3]
    flank = e[4]

    m2z_args = None
    if len(e) > 6:
      m2z_args = " ".join(e[6:])

    if flank != "" and flank != "NA":
      flank = convertFlank(flank)
      if flank is None:
        die("Error: could not parse flank \"%s\", format incorrect." % e[4])

    # If they messed up and put "RS" instead of "rs" in the SNP name, fix it. 
    # The only case in which I can't be sure to fix it is if the gene is RS1. 
    if snp != "RS1":
      snp = re.sub("^RS(\d+)","rs\\1",snp)

    if isSNP(snp):
      snp = SNP(snp=snp)
      snp.tsnp = transSNP(snp.snp,db_file)
      (snp.chr,snp.pos) = find_pos(snp.tsnp)
      snp.chrpos = "chr%s:%s" % (snp.chr,snp.pos)
        
    if flank is not None and flank != "NA":
      if isSNP(snp):
        fchr = snp.chr
        fpos = snp.pos
        try:
          chr = int(fchr)
          start = fpos - flank
          end = fpos + flank
        except:
          if fchr is None or fpos is None:
            print >> sys.stderr, "Error: could not find position for SNP %s, skipping.." % e[0]
          else:
            print >> sys.stderr, "Error: bad chromosome/position for SNP %s, chr/pos were: %s,%s, skipping.." % (e[0],str(fchr),str(fpos))
          continue
      else:
        gene_info = findGeneInfo(snp,db_file)
    
        if gene_info is not None:
          if m2z_args is None:
            m2z_args = "requiredGene=%s" % snp
          else:
            m2z_args += " requiredGene=%s" % snp;        
  
          chr = gene_info['chrom']
          start = gene_info['txStart'] - flank
          end = gene_info['txEnd'] + flank
        else:
          try:
            chr = int(chr)
            start = long(start)
            end = long(end)
          except:
            print >> sys.stderr, "Error: no known SNP or gene present in first column, and invalid chr/start/end given in hitspec file."
            continue
          
          snp = regionString(chr,start,end)
    else:
      try:
        chr = int(chr)
      except:
        if isSNP(snp):
          chr = snp.chr
        else:
          print >> sys.stderr, "Error: invalid chr/start/end in hitspec file, no chrom given, line was: '%s'" % " ".join(e)
          continue

      try:
        start = long(start)
        end = long(end)
      except:
        print >> sys.stderr, "Error: invalid start/end in hitspec file, line was: '%s'" % " ".join(e)
        continue
      
      # If something wasn't given in the SNP/gene column, need a placeholder.
      if not isSNP(snp) and findGeneInfo(snp,db_file) is None:
        snp = regionString(chr,start,end)

    snplist.append((
      snp,
      chr,
      start,
      end,
      m2z_args
    ))

  f.close()

  return snplist

# Given a flank string, such as: "500kb" or "500MB", convert to raw position.
# Examples:
# 500kb --> 500,000
# 1.5MB --> 1,500,000
def convertFlank(flank):
  iFlank = None
  p = re.compile("(.+)(kb|KB|Kb|kB|MB|Mb|mb|mB)")
  match = p.search(flank)
  if match:
    digits = match.groups()[0]
    suffix = match.groups()[1]

    if suffix in ('kb','KB','Kb','kB'):
      iFlank = float(digits)*1000
    elif suffix in ('MB','Mb','mb','mB'):
      iFlank = float(digits)*1000000

    iFlank = int(round(iFlank))
  else:
    try:
      iFlank = int(flank)
    except:
      pass

  return iFlank

def printOpts(opts):
  table = PrettyTable(['Option','Value'])
  table.set_field_align('Option','l')
  table.set_field_align('Value','l')
  
  if opts.metal:
    table.add_row(['metal',os.path.split(opts.metal)[1]])
    table.add_row(['markercol',opts.snpcol])
    table.add_row(['pvalcol',opts.pvalcol])
  elif opts.epacts:
    table.add_row(['epacts',os.path.split(opts.epacts)[1]])
    table.add_row(["epacts-chr-col",opts.epacts_chr_col])
    table.add_row(["epacts-beg-colcol",opts.epacts_beg_col])
    table.add_row(["epacts-end-col",opts.epacts_end_col])
    table.add_row(["epacts-pval-col",opts.epacts_pval_col])

  if opts.refsnp:
    table.add_row(['refsnp',opts.refsnp])
  elif opts.refgene:
    table.add_row(['refgene',opts.refgene])
  elif opts.hitspec:
    table.add_row(['hitspec',opts.hitspec])
  elif opts.chr and opts.start and opts.end:
    table.add_row(['chr',opts.chr])
    table.add_row(['start',opts.start])
    table.add_row(['end',opts.end])

  if opts.gwas_cat:
    table.add_row(['gwas-cat',opts.gwas_cat])

  display_opts = OrderedSet([
    'flank','build','ld','ld_measure','pop','source','snpset','db','gene_table','cache',
    'no_clean','no_transform','enable_db_annot','verbose','m2zpath','plotonly'
  ])
 
  if opts.ld_vcf:
    table.add_row(['ld-vcf',os.path.split(opts.ld_vcf)[1]])
    display_opts = display_opts - {"pop","source"}

  if opts.condsnps:
    cond_snps = [str(i) for i in opts.condsnps]
    table.add_row(['condsnps',",".join(cond_snps)])

  for opt in display_opts:
    val = getattr(opts,opt)
    if val is not None and val:
      opt = opt.replace("_","-")
      table.add_row([opt,str(val)])
    
  table.printt()

def printArgs(args):
  table = PrettyTable(['Option','Value'])
  table.set_field_align('Option','l')
  table.set_field_align('Value','l')
  
  d = parse_rargs(args)
  del d["markerCol"]
  del d["pvalCol"]
  for i,j in d.iteritems():
    table.add_row([str(i),str(j)])
  
  table.printt()

class SNP:
  def __init__(self,snp=None,tsnp=None,chrpos=None,chr=None,pos=None):
    self.snp = snp;         # snp name given by user
    self.tsnp = tsnp;       # snp name translated to current genome build
    self.chrpos = chrpos;   # "chrpos" name for SNP, e.g. chr#:###
    self.chr = chr;         # chromosome
    self.pos = pos;         # position
  
  def __str__(self):
    return self.snp

def parse_denote_marker_file(filepath,sqlite_file):
  find_pos = PosLookup(sqlite_file)
  
  out_name = "temp_denote_marker_" + tempfile.mktemp(dir="")
  with open(filepath) as f:
    with open(out_name,'w') as out:
      header = f.readline().rstrip().split("\t")
      for hitem in ('snp','string'):
        try:
          header.index(hitem)
        except:
          die("Error: file given for --denote-markers-file does not have the proper header.")

      header.insert(1,'chrpos')
      header.insert(2,'chr')
      header.insert(3,'pos')
      print >> out, "\t".join(header)

      for line in f:
        e = line.split("\t")
        e[-1] = e[-1].rstrip()

        snp = e[0]
        (chrom,pos) = find_pos(snp)

        if chrom is None or pos is None:
          print >> sys.stderr, "Warning: could not find position for SNP %s in file given by --denote-markers-file, skipping.." % snp
          continue

        snp_chrpos = "chr%s:%s" % (chrom,pos)

        e.insert(1,snp_chrpos)
        e.insert(2,chrom)
        e.insert(3,pos)

        print >> out, "\t".join(map(str,e))

  return out_name

# Parse command line arguments and return a tuple (opts,args) where:
# opts - settings that are specific to the program and have been error-checked
# opts looks like an object, settings are of the form: opt.some_setting
# args - all command line arguments that were positional, these are not checked
# and are immediately passed on to locuszoom.R 
def getSettings():
  conf = getConf()
  
  usage = "usage: locuszoom [options]"
  
  parser = VerboseParser(usage=usage)
  parser.add_option("--metal",dest="metal",help="Metal file.")
  parser.add_option("--epacts",dest="epacts",help="EPACTS results file.")
  parser.add_option("--delim",dest="delim",help="Delimiter for metal file.")
  parser.add_option("--pvalcol",dest="pvalcol",help="Name of p-value column or 0-based integer specifying column number.")
  parser.add_option("--markercol",dest="snpcol",help="Name of SNP column or 0-based integer specifying column number.")
  parser.add_option("--epacts-chr-col",dest="epacts_chr_col",help="Name of chrom column for EPACTS file. Defaults to #CHROM.")
  parser.add_option("--epacts-beg-col",dest="epacts_beg_col",help="Name of begin column for EPACTS file. Defaults to BEGIN.")
  parser.add_option("--epacts-end-col",dest="epacts_end_col",help="Name of end column for EPACTS file. Defaults to END.")
  parser.add_option("--epacts-pval-col",dest="epacts_pval_col",help="Name of pvalue column for EPACTS file. Defaults to PVALUE.")
  parser.add_option("--cache",dest="cache",help="Change the location of the cache file to use for LD. Set to 'None' to disable.")
  parser.add_option("--hitspec",dest="hitspec",help="File containing a list of SNPs, chromosome, start, and stop.")
  parser.add_option("--refsnp",dest="refsnp",help="Create region plot around this reference SNP.")
  parser.add_option("--add-refsnps",dest="condsnps",help="Plot LD with additional SNPs.")
  parser.add_option("--conditional",dest="cond_compat",help=SUPPRESS_HELP); # backward compat, same as above option
  parser.add_option("--denote-markers-file",dest="denote_markers_file",help="Designate additional markers to highlight on the plot with a file.")
  parser.add_option("--bed-tracks",dest="bed_tracks",help="Give a BED file to display tracks (one track per label in name column.) Uses first 4 columns of BED, ignoring the rest.")
  parser.add_option("--refgene",dest="refgene",help="Create region plot flanking a gene.")
  parser.add_option("--flank",dest="flank",help="Distance around refsnp to plot.")
  parser.add_option("--chr",dest="chr",help="Chromosome that refsnp is located on - this is only used for sanity checking, but it is good to include.")
  parser.add_option("--start",dest="start",help="Start position for interval near refsnp. Can be specified along with --end instead of using --flank.")
  parser.add_option("--end",dest="end",help="End position for interval near refsnp.")
  parser.add_option("--ld",dest="ld",help="Specify a user-defined LD file.")
  parser.add_option("--ld-vcf",dest="ld_vcf",help="Specify a VCF file from which to compute LD.")
  parser.add_option("--ld-measure",dest="ld_measure",help="Specify LD measure to use. Can be either 'rsquare' or 'dprime'. Default is rsquare.")
  parser.add_option("--ignore-vcf-filter",action="store_true",default=False,help="Ignore FILTER column when using VCF file to calculate LD.")
  parser.add_option("--no-ld",dest="no_ld",help="Disable calculating and displaying LD information.",action="store_true")
  parser.add_option("--enable-db-annot",action="store_true",default=False,help="Enable pulling variant annotation from SQLite. This is for backwards compatibility only in 1.3, will be completely disabled in 1.4.")

  # Options controlling plot output
  parser.add_option("--svg",action="store_true",default=False,help="Also generate SVG in addition to PDF format.")

  # Options controlling which database tables to use
  parser.add_option("--gene-table",help="Name of gene table to use in database. Defaults to 'refFlat'. GENCODE is also supplied for some builds - use 'gencode'.",default=SQLITE_REFFLAT)

  gwas_help = "Select GWAS catalog to use for plotting GWAS hits track. Available catalogs are: {spacer}{tables}".format(
    spacer = "".join([os.linesep]*2),
    tables = getAllGWACatalogs(conf.GWAS_CATS)
  )
  
  parser.add_option("--gwas-cat",dest="gwas_cat",help=gwas_help)
  
  parser.add_option("--build",dest="build",help="NCBI build to use when looking up SNP positions, and for selecting the correct dataset to use for LD calculations.")
  
  pop_help = "Population to use for LD. This option, and the --source option below, have the following available options: {spacer}{trios}".format(
    spacer = "".join([os.linesep]*2),
    trios = getSupportedTrios(conf.LD_DB)
  )
  
  parser.add_option("--pop",dest="pop",help=pop_help)
  parser.add_option("--source",dest="source",help="Source to use for LD (defaults to hapmap.) See above for more available options.")
  
  parser.add_option("--snpset",dest="snpset",help="Set of SNPs to plot as a rug.")
  parser.add_option("--no-snp-name",dest="no_snp_name",default=False,action="store_true",help="Remove reference SNP name from plots.")
  parser.add_option("--no-cleanup",dest="no_clean",action="store_true",help="Leave temporary files generated by this script. This does not affect clobber or clean in locuszoom.R.")
  parser.add_option("--no-transform",dest="no_trans",action="store_true",help="Disable automatic transformation of p-values.")
  parser.add_option("-v","--verbose",dest="verbose",action="store_true",help="Make the script be more talkative.")
  parser.add_option("--multi",dest="multi",type="int",help="Number of zoomplots to create in parallel. Default is 1 (serial mode). (not yet implemented)")
  parser.add_option("--plotonly",dest="plotonly",help="Generate only the plot figure, delete all other temporary files and do not create a directory.",action="store_true")
  parser.add_option("-p","--prefix",dest="prefix",type="str",help="Prefix to add to zoomplot pdfs or directories.")
  parser.add_option("--no-date",dest="no_date",action="store_true",default=False,help="Remove date from directory and filenames.")
  parser.add_option("--rundir",dest="rundir",help="Directory where locuszoom will attempt to run.")
  parser.add_option("-e","--experimental",dest="exper",action="store_true",help=SUPPRESS_HELP)
  parser.add_option("--override-m2z",dest="m2zpath",help=SUPPRESS_HELP)
  parser.add_option("--db",type="string",help="SQLite database file. This overrides the conf file.")
  parser.add_option("--offline",dest="offline",action="store_true",default=False,help=SUPPRESS_HELP)

  # Defaults.
  parser.set_defaults(
    multi = 1,
    delim = "\t",
    epacts_chrom_col = "#CHROM",
    epacts_beg_col = "BEGIN",
    epacts_end_col = "END",
    epacts_pval_col = "PVALUE",
    no_clean = False,
    no_ld = False,
    no_transform = False,
    build = None,
    ld_measure = 'rsquare',
    gwas_cat = None,
    plotonly = False,
    prefix = None,
    pvalcol = "P-value",
    snpcol = "MarkerName", 
    verbose = False,
    pop = None,
    snpset = "Illu1M",
    rundir = ".",
    source = None,
    experimental = False,
    cache = "../ld_cache.db",
  )

  (opts,args) = parser.parse_args()

  # Absolutely must specify genome build
  if opts.build is None:
    die("Error: there is no longer a default for --build, you must specify it.")

  # Should we override M2Z path?
  if opts.m2zpath is not None:
    if os.path.isfile(opts.m2zpath):
      print "Overriding locuszoom.R path: %s" % opts.m2zpath
      opts.metal2zoom_path = os.path.abspath(os.path.expanduser(opts.m2zpath))
      
      opts.metal2zoom_path = find_systematic(opts.metal2zoom_path)
      if opts.metal2zoom_path is None:
        die("Error: could not find locuszoom.R - check conf file %s" % M2ZFAST_CONF)
      
    else:
      print "locuszoom.R override specified, but path \'%s\' does not exist - using default." % opts.m2zpath
      print "Current directory is: %s" % os.getcwd()
  else:
    opts.metal2zoom_path = find_systematic(conf.METAL2ZOOM_PATH)
    if not os.path.isfile(opts.metal2zoom_path):
      die("Error: could not find locuszoom.R - check conf file %s" % M2ZFAST_CONF)

  # Are we running in experimental mode?
  if opts.exper:
    opts.verbose = True
    globals()['_DEBUG'] = True
  
  # Did they specify a SQLite database to use? 
  if opts.db:
    if os.path.isfile(opts.db):
      opts.sqlite_db_file = os.path.abspath(opts.db)
    else:
      die("Error: --db %s does not exist!" % str(opts.db))
  else:
    opts.sqlite_db_file = find_relative(conf.SQLITE_DB[opts.build]); # read from conf file
    if not os.path.isfile(opts.sqlite_db_file):
      die("Error: could not locate sqlite database, tried: %s, check your conf file" % str(opts.sqlite_db_file))

  # SNP position looker-upper. 
  find_pos = PosLookup(opts.sqlite_db_file)

  # If a temporary directory was specified, it better exist!
  if opts.rundir:
    if not os.path.isdir(opts.rundir):
      die("Error: temporary directory %s does not exist, you must create it first." % str(opts.tempdir))

  # Check to see if --hitspec and --refsnp were specified together. This shouldn't happen.
  mode_count = sum(map(lambda x: x is not None,[opts.hitspec,opts.refsnp]))
  if mode_count > 1:
    die_help("Must specify either --refsnp or --hitspec. These options are mutually exclusive.",parser)

  # Check to see if --hitspec and --refgene were specified together. This shouldn't happen.
  mode_count = sum(map(lambda x: x is not None,[opts.hitspec,opts.refgene]))
  if mode_count > 1:
    die_help("Must specify either --refgene or --hitspec. These options are mutually exclusive.",parser)
  
  # Perform checks on input file. 
  if opts.metal is not None:
    if opts.metal != "-":
      input_file = find_systematic(opts.metal)
      if input_file is None:
        die("Error: could not find file: %s" % opts.metal)

      opts.metal = input_file
    else:
      input_file = "-"

  elif opts.epacts is not None:
    if opts.epacts != "-":
      input_file = find_systematic(opts.epacts)
      if input_file is None:
        die("Error: could not find file: %s" % opts.epacts)

      opts.epacts = input_file
    else:
      input_file = "-"

    opts.delim = "\t"
    opts.pvalcol = "P-value",
    opts.snpcol = "MarkerName",
  else:
    die("Error: must supply either --metal or --epacts file.")

  # Check file size if appropriate.
  if input_file != "-":
    if os.path.getsize(input_file) <= 0:
      die("Error: input file is empty: %s" % str(input_file))

    # Check if we have access rights.
    if not os.access(input_file, os.R_OK):
      die("Error: cannot access input file, insufficient permissions: %s" % str(input_file))

  # Fix delimiter.
  if opts.delim in ("tab","\\t","\t"):
    opts.delim = "\t"
  elif opts.delim in (" ","space"):
    opts.delim = " "
  elif opts.delim in ("","whitespace","None"):
    opts.delim = None
  elif opts.delim in (",","comma"):
    opts.delim = ","
  else:
    opts.delim = "\t"

  # Error checking on main modes.
  if opts.refsnp:
    if not isSNP(opts.refsnp):
      die_help("Error: SNP %s not recognized as SNP" % str(opts.refsnp),parser)

    # Check if refsnp appears to have specified alleles, warn user
    rmatches = re.search("(chr)?([0-9a-zA-z]+):([0-9]+)(_.*)?",opts.refsnp)
    if rmatches is not None:
      alleles = rmatches.groups()[3]
      if alleles is not None:
        alleles = alleles.replace("_","")
        print >> sys.stderr, ("Warning: it appears that you specified alleles (%s) with your --refsnp, but LocusZoom "
                             "does not consider alleles, only the chrom and pos. If your results have multiple "
                             "variants at the same position, LocusZoom will not distinguish between them." % alleles)

  elif opts.hitspec:
    opts.hitspec = find_systematic(opts.hitspec)
    if opts.hitspec is None:
      die("Error: hitspec file does not exist.")

  # Check that multithread count is less than maximum allowed.
  if opts.multi > MULTI_CAP:
    opts.multi = MULTI_CAP
    print >> sys.stderr, "Warning: --multi was higher than maximum allowed value of %i, reducing value to that limit." % MULTI_CAP

  # Compute flank in raw digits instead of with kb or mb as a suffix, i.e. "100kb" --> "100000"
  if opts.flank is not None:
    iFlank = convertFlank(opts.flank)

    if iFlank is None:
      die("Error: flank specification did not match pattern, flank was: %s.\n"
          "Example flanks are: \n"
          "500kb (kilobases)\n"
          "1.25MB (megabases)\n"
          "1323414 (bases)\n"
          % opts.flank)

    opts.flank = iFlank

  # If the user disabled LD, we shouldn't use it..
  if opts.no_ld:
    opts.ld = None
    opts.ld_vcf = None

  # Check to see if user passed in LD file. If they did, it better exist.
  # A user passing in an LD file eliminates the need to check if their
  # build/population/source combination (see directly below) is correct.
  if opts.ld is not None:
    opts.ld = find_systematic(opts.ld)
    if opts.ld is None or not os.path.isfile(opts.ld):
      die("Error: user-specified LD file does not exist.\nFile was: %s " % opts.ld)
  elif opts.ld_vcf is not None:
    opts.ld_vcf = find_systematic(opts.ld_vcf)
    if opts.ld_vcf is None or not os.path.isfile(opts.ld_vcf):
      die("Error: user-specified VCF (or JSON) file does not exist.\nFile was: %s " % opts.ld_vcf)

    if '.json' in opts.ld_vcf:
      import json
      with open(opts.ld_vcf) as jsin:
        ld_vcf_dict = json.load(jsin)

      for chrom, vcf_file in ld_vcf_dict.iteritems():
        vcf_file_tabix = vcf_file + ".tbi"
        if not os.path.isfile(vcf_file_tabix):
          die("Error: expected tabix index for VCF file, but could not find one: %s" % vcf_file_tabix)

      opts.ld_vcf_dict = ld_vcf_dict
    else:
      ld_vcf_tabix = opts.ld_vcf + ".tbi"
      if not os.path.isfile(ld_vcf_tabix):
        die("Error: expected a tabix index for VCF file but could not find it: %s" % ld_vcf_tabix)

  else:
    if not opts.no_ld:
      # Check that pop and source were provided
      if opts.pop is None:
        die("Error: you must specify --pop, there is no longer a default")

      if opts.source is None:
        die("Error: you must specify --source, there is no longer a default")

      # Fix up population/build/source settings before checking.
      opts.pop = opts.pop.upper(); # populations are always upper-case

      # Check to see if the population, LD source, and build supplied are compatible.
      info_geno = getLDInfo(opts.pop,opts.source,opts.build,conf.LD_DB)
      if info_geno is None:
        print >> sys.stderr, "Error: source %s, population %s, and build %s are not jointly supported." % (
          opts.source,opts.pop,opts.build)
        print >> sys.stderr, "See below for supported combinations."
        print >> sys.stderr, ""
        printSupportedTrios(conf.LD_DB)
        sys.exit(1)

  # Check GWAS catalog setting.
  if opts.gwas_cat is not None:
    gwas_cat_file = getGWASCat(opts.build,opts.gwas_cat,conf.GWAS_CATS)
    if gwas_cat_file is None:
      opts.gwas_cat_file = find_systematic(opts.gwas_cat)
      if opts.gwas_cat_file is None:
        print >> sys.stderr, "Error: no gwas catalog '%s' exists for selected build, and was not a file either!" % opts.gwas_cat
        print >> sys.stderr, "If you were trying to specify a catalog by code, they are: "
        printGWACatalogs(conf.GWAS_CATS,opts.build)
        sys.exit(1)
    else:
      opts.gwas_cat_file = find_systematic(gwas_cat_file)
  else:
    opts.gwas_cat_file = None

  # Change refSNP into a chr:pos SNP. 
  if opts.refsnp:
    opts.refsnp = SNP(snp=opts.refsnp)
    opts.refsnp.tsnp = transSNP(opts.refsnp.snp,opts.sqlite_db_file)
    (chr,pos) = find_pos(opts.refsnp.tsnp)
  
    if chr is None or pos is None:
      die("Error: could not find chr/pos information for SNP %s in database." % opts.refsnp)
    
    opts.refsnp.chrpos = "chr%s:%s" % (chr,pos)
    opts.refsnp.chr = chr
    opts.refsnp.pos = pos

  # If the user specified conditional / second signal SNPs, parse these into chr/pos.
  if opts.cond_compat is not None:
    opts.condsnps = opts.cond_compat; # if user gave --conditional (old option) use this

  if opts.condsnps:
    cond_snps = [i.strip() for i in opts.condsnps.split(",")]
    cond_snps_chrpos = list()
    for csnp in cond_snps:
      csnp = SNP(snp=csnp)
      csnp.tsnp = transSNP(csnp.snp,opts.sqlite_db_file)
      (csnp_chr,csnp_pos) = find_pos(csnp.tsnp)

      csnp.chr = csnp_chr
      csnp.pos = csnp_pos
      csnp.chrpos = "chr%s:%s" % (csnp_chr,csnp_pos)

      cond_snps_chrpos.append(csnp)

    opts.condsnps = cond_snps_chrpos

  # If specified, check the denote markers file. 
  if opts.denote_markers_file:
    denote_file = find_systematic(opts.denote_markers_file)
    if denote_file is None:
      die("Error: could not find file specified by --denote-markers-file: %s" % opts.denote_markers_file)
    else:
      opts.denote_markers_file = denote_file
  
  # Check BED tracks. 
  if opts.bed_tracks:
    bed_tracks = find_systematic(opts.bed_tracks)
    if bed_tracks is None:
      die("Error: could not find file specified by --bed-tracks: %s" % opts.bed_tracks)
    else:
      opts.bed_tracks = bed_tracks

  # Compute start/end positions for each SNP, unless already specified and in refsnp mode.
  opts.snplist = []
  if opts.refsnp:
    (chr,pos) = (opts.refsnp.chr,opts.refsnp.pos)

    if opts.start and opts.end and opts.chr:
      opts.start = long(opts.start)
      opts.end = long(opts.end)
      opts.chr = chrom2chr(chr)
      if opts.chr == chr and opts.start < pos and opts.end > pos:
        opts.snplist.append( (opts.refsnp,chr,opts.start,opts.end) )
      else:
        msg = "Warning: skipping SNP %s, genomic interval given does not overlap SNP position according to our database." % opts.refsnp.snp
        msg += "\nGiven interval: %s\t Genomic position: %s" % (
          regionString(opts.chr,opts.start,opts.end),
          "chr" + str(chr) + ":" + str(pos)
        )
        die(msg)
        
    elif opts.refgene:
      refgene_info = findGeneInfo(opts.refgene,opts.sqlite_db_file)
      if refgene_info is None:
        die("Error: gene selected for plotting was not found in refFlat.")
  
      flank = opts.flank
      if opts.flank is None:
        flank = convertFlank(DEFAULT_GENE_FLANK)
        
      gene_rs = refgene_info['txStart'] - flank
      gene_re = refgene_info['txEnd'] + flank
  
      opts.snplist.append((
        opts.refsnp,
        refgene_info['chrom'],
        gene_rs,
        gene_re,
      ))
      
    elif opts.flank:
      opts.snplist.append( (opts.refsnp,chr,pos-opts.flank,pos+opts.flank) )
      
    else:
      print "No flank, chr/start/stop, or reference gene given, using default flank of %s.." % DEFAULT_SNP_FLANK
      def_flank = convertFlank(DEFAULT_SNP_FLANK)
      opts.snplist.append( (opts.refsnp,chr,pos-def_flank,pos+def_flank) )
      
  elif opts.hitspec:
    opts.snplist = readWhitespaceHitList(opts.hitspec,opts.sqlite_db_file)
  
  elif opts.refgene:
    refgene_info = findGeneInfo(opts.refgene,opts.sqlite_db_file)
    if refgene_info is None:
      die("Error: gene selected for plotting was not found in refFlat.")

    flank = opts.flank
    if opts.flank is None:
      flank = convertFlank(DEFAULT_GENE_FLANK)
      
    gene_rs = refgene_info['txStart'] - flank
    gene_re = refgene_info['txEnd'] + flank

    opts.snplist.append((
      opts.refgene,
      refgene_info['chrom'],
      gene_rs,
      gene_re,
    ))

  elif opts.chr and opts.start and opts.end:
    region = "chr%s_%s-%s" % (str(opts.chr),str(opts.start),str(opts.end))
    opts.snplist.append((
      region,
      chrom2chr(opts.chr),
      int(opts.start),
      int(opts.end),
    ))
  
  else:
    die("Error: you must specify one of these options: \n"
        "--refsnp\n"
        "--refgene\n"
        "--hitspec\n"
        "--chr, --start, and --end together\n"
    )

  # Fix cache location.
  if opts.cache not in (None,"None","False","disable"):
    (cache_path,cache_file) = os.path.split(opts.cache)
    if cache_path == '':
      cache_path = ".."
  
    opts.cache = os.path.join(cache_path,cache_file)
  else:
    opts.cache = None

  # Do we need to add snpcol and pvalcol to the m2z args? 
  if opts.snpcol is not None:
    args.append("markerCol=%s" % opts.snpcol)

  if opts.pvalcol is not None:
    args.append("pvalCol=%s" % opts.pvalcol)

  m2zargs = parse_rargs(args)
  if 'snpset' in m2zargs:
    print >> sys.stderr, "Warning: overriding --snpset %s with option given as snpset=%s.." % (opts.snpset,m2zargs['snpset'])
    opts.snpset = m2zargs['snpset']

  # Print warnings about deprecated options. 
  if opts.offline:
    print >> sys.stderr, "Warning: --offline no longer required, option will be ignored.."

  return (opts,args)

# On Unix: use gunzip to decompress a gzipped file
# On Windows: use gzip module to write gzipped file
# If file "out" exists, appends to file. Otherwise creates a new file "out". 
def decompGZFile(file,out):
  if not os.path.isfile(file):
    raise ValueError, "Error: file does not exist: %s" % file

  if platform.system() == "Linux":
    if os.path.isfile(out):
      os.system("gunzip -c %s >> %s" % (file,out))
    else:
      os.system("gunzip -c %s > %s" % (file,out))
  
    if not os.path.isfile(out) or os.path.getsize(out) < 1:
      raise Exception, "Error: could not decompress file %s" % file
  else:
    out = None
    if os.path.isfile(out):
      out = open(out,"a")
    else:
      out = open(out,"w")
    
    f = gzip.open(file)
    out.writelines(f)
    f.close()
    out.close()

def computeLD(snp,chr,start,end,build,pop,source,cache_file,fugue_cleanup,verbose):
  conf = getConf()
 
  conf.NEWFUGUE_PATH = find_systematic(conf.NEWFUGUE_PATH)
  conf.PLINK_PATH = find_systematic(conf.PLINK_PATH)
 
  ld_info = getLDInfo(pop,source,build,conf.LD_DB)
  if 'map_dir' in ld_info:
    settings = FugueSettings(
      ld_info['map_dir'],
      ld_info['ped_dir'],
      conf.NEWFUGUE_PATH
    )
  elif 'bim_dir' in ld_info:
    settings = PlinkSettings(
      ld_info['bim_dir'],
      conf.PLINK_PATH
    )

  else:
    raise Exception, "Error: conf file specification for %s/%s/%s is invalid, please check syntax." % (pop,source,build)

  # Check that LD program exists. 
  if isinstance(settings,FugueSettings):
    if not os.path.exists(conf.NEWFUGUE_PATH):
      raise Exception, "Error: could not find %s for computing LD.." % conf.NEWFUGUE_PATH
    else:
      print "Using %s to compute LD.." % conf.NEWFUGUE_PATH
  elif isinstance(settings,PlinkSettings):
    if not os.path.exists(conf.PLINK_PATH):
      raise Exception, "Error: could not find %s for computing LD.." % conf.PLINK_PATH
    else:
      print "Using %s to compute LD.." % conf.PLINK_PATH

  if cache_file is not None:
    cache = LDRegionCache(settings.createLDCacheKey(),cache_file)
  else:
    cache = None
  
  if isinstance(settings,FugueSettings): 
    ld_finder = FugueFinder(settings,cache,fugue_cleanup,verbose)
  elif isinstance(settings,PlinkSettings):
    ld_finder = PlinkFinder(settings,cache,fugue_cleanup,verbose)

  try:
    ld_success = ld_finder.compute(snp.chrpos,chr,start,end)
    if ld_success:
      ld_filename = "templd_" + snp.snp.replace(":","_").replace("/","") + "_" + time.strftime("%y%m%d",time.localtime()) + "-" + time.strftime("%H%M%S",time.localtime()) + ".txt"
    
      if os.path.isfile(ld_filename):
        print >> sys.stderr, "Warning: LD file already exists for some reason: %s" % ld_filename

      write_success = ld_finder.write(ld_filename)
    
      if not write_success:
        ld_filename = None
    else:
      print >> sys.stderr, "Warning: LD could not be computed for SNP %s. "\
      "This SNP does not exist in the genotype files for computing LD from %s/%s/%s.." % (str(snp),source,build,pop)
      ld_filename = None
  finally:
    if cache is not None:
      cache.close()

  return ld_filename

# Fixes up a user supplied LD file for passing into m2z.
# "Fixing" means:
# -- remove rows that do not contain the refsnp
# -- return None if the header does not contain dprime or rsquare
# -- return None if refsnp is not ever seen
# -- translates SNP names into chr:pos format
# Returns filename of fixed LD file, or None if a failure occurred
def fixUserLD(file,refsnp,db_file):
  # Create temporary file to write LD to.
  out = "temp_user_ld_" + tempfile.mktemp(dir="")
  
  # Open user LD file. 
  if os.path.splitext(file)[1] == ".gz":
    f = gzip.open(file)
  else:
    f = open(file)
  
  found_refsnp = False
  
  # Checks on LD file format. 
  h = f.readline().lower().rstrip()
  h_s = h.split()
  
  for column in ('snp1','snp2','dprime','rsquare'):
    try:
      exec "%s_col = h_s.index(column)" % column
    except:
      print >> sys.stderr, "Error: user-supplied LD file does not have column '%s' in header (or no header row exists.)" % column
      return None
  
  find_pos = PosLookup(db_file)
  
  out_file = open(out,"w")
  print >> out_file, h
  for line in f:
    e = line.rstrip().split()
    
    # If the line contains the refsnp, extract it. 
    if line.find(refsnp.snp) != -1:
      found_refsnp = True
      
      skip = False
      for col in (snp1_col,snp2_col):
        snp = e[col]
        (chr,pos) = find_pos(snp)
        if chr is not None and pos is not None:
          e[col] = "chr%s:%s" % (chr,pos)
        else:
          print >> sys.stderr, "Warning: could not find position for SNP %s in user-supplied --ld file, skipping.." % str(snp)
          skip = True
          
      if not skip:
        print >> out_file, " ".join(e)
  
  f.close()
  out_file.close()
  
  if found_refsnp:
    return out
  else:
    print >> sys.stderr, "Error: user-supplied LD file does not contain the reference SNP %s.." % str(refsnp)
    return None

def windows_file_replace_chars(name):
  bad_chars = "\\ / : * ? \" < > |".split()
  for c in bad_chars:
    name = name.replace(c,"_")
  
  return name

def windows_path_replace_chars(name):
  bad_chars = ": * ? \" < > |".split()
  for c in bad_chars:
    name = name.replace(c,"_")

  return name

def runQuery(query,args):
  hash = str(int(time.time())) + "_" + str(os.getpid())
  file = "%s_%s.txt" % (query.func_name,hash)
 
  try:
    cur = query(*args)
  except:
    error_msg = str(sys.exc_info()[1])
    print >> sys.stderr, "Error: SQL query failed, error was: %s" % error_msg
    cur = None
    file = None
  
  count = 0
  if cur is not None:
    try:
      out = open(file,"w")
      count = print_results(cur,"\t",out)
      out.close()
      
      if count <= 0:
        file = None

    except:
      error_msg = str(sys.exc_info()[1])
      print >> sys.stderr, "Error: could not write SQL query '%s' to file. Exception was: " % error_msg
      file = None

  return file

def listTables(db_file):
  try:
    db = sqlite3.connect(db_file)
    cur = db.execute("select name from sqlite_master where type='table'")

    res = cur.fetchall()
    tables = [str(i[0]) for i in res]
    
    db.close()
  finally:
    db.close()

  return tables

def runQueries(chr,start,stop,snpset,build,db_file,do_annot,gene_table):
  results = {}
  
  db = sqlite3.connect(db_file)
 
  db_tables = listTables(db_file)

  if gene_table in db_tables:
    results['refFlat'] = runQuery(refflat_in_region,[db,gene_table,chr,start,stop,build])
  else:
    print >> sys.stderr, "Warning: gene table '%s' not found in database, skipping gene lookups in region" % gene_table
 
  if do_annot:
    if all([x in db_tables for x in (SQLITE_SNP_POS,SQLITE_VAR_ANNOT)]):
      results['annot'] = runQuery(snp_annot_in_region,[db,SQLITE_SNP_POS,SQLITE_VAR_ANNOT,chr,start,stop,build])
    else:
      print >> sys.stderr, "Warning: either SNP position table '%s' or annotation table '%s' were not found in database" % (SQLITE_SNP_POS,SQLITE_VAR_ANNOT)

  if SQLITE_RECOMB_RATE in db_tables:
    results['recomb'] = runQuery(recomb_in_region,[db,SQLITE_RECOMB_RATE,chr,start,stop,build])
  else:
    print >> sys.stderr, "Warning: recombination rate table '%s' not found in database, skipping recomb rate lookups in region" % SQLITE_RECOMB_RATE
    
  if all([x in db_tables for x in (SQLITE_SNP_POS,SQLITE_SNP_SET)]):
    results['snpsetFile'] = runQuery(snpset_in_region,[db,SQLITE_SNP_POS,SQLITE_SNP_SET,snpset,chr,start,stop,build])
  
  return results

def runAll(input_file,input_type,refsnp,chr,start,end,opts,args):
  conf = getConf()

  build = opts.build
  delim = opts.delim
  no_clean = opts.no_clean
  
  print "Beginning plotting sequence for: %s" % str(refsnp)
  print "Extracting region of interest (%s) from input file.." % regionString(chr,start,end)

  if input_type == 'metal':
    (bPocull,metal_temp,min_snp) = read_metal(input_file,opts.snpcol,opts.pvalcol,opts.no_trans,chr,start,end,opts.sqlite_db_file,delim)
  elif input_type == 'epacts':
    (bPocull,metal_temp,min_snp) = read_epacts(input_file,chr,start,end,opts.epacts_chrom_col,opts.epacts_beg_col,opts.epacts_end_col,opts.epacts_pval_col,opts.no_trans)

  ld_temp = None
  
  # If poculling does not give us any SNPs in the region, we're done. 
  if not bPocull:
    print >> sys.stderr, "Error: region specified contains no SNPs, skipping.."
    if not no_clean:
      print "Deleting temporary files.."
      cleanup([ld_temp,metal_temp])
    return; # hack

  # If a gene was passed in, we need to tell M2Z it is a required gene to plot. 
  if not isSNP(refsnp):
    if findGeneInfo(refsnp,opts.sqlite_db_file) is not None:
      args += " requiredGene=\"%s\"" % str(refsnp)

  find_pos = PosLookup(opts.sqlite_db_file)

  # If something other than a SNP was passed in, we need to find 
  # the best SNP in the region. 
  if not isSNP(refsnp):
    print "Attempting to find best SNP in region.."
    print "Found: %s" % min_snp;  
    
    refsnp = SNP(snp=min_snp)
    refsnp.tsnp = transSNP(min_snp,opts.sqlite_db_file)
    (best_chr,best_pos) = find_pos(refsnp.tsnp)
    refsnp.chr = best_chr
    refsnp.pos = best_pos
    refsnp.chrpos = "chr%s:%s" % (best_chr,best_pos)

  # Get refsnp position. 
  # If a position cannot be found, bail out. 
  if refsnp.pos is None:
    print >> sys.stderr, "Error: could not find position for %s, skipping.." % str(refsnp)
    if not no_clean:
      print "Deleting temporary files.."
      cleanup([ld_temp,metal_temp])
    return; 
  
  # Should we calculate LD? 
  if not opts.no_ld:
    # Did the user supply an LD file? If so, we don't need to calculate it.
    if opts.ld:
      print "Using user-specified LD file.."
      opts.ld = fixUserLD(opts.ld,refsnp,opts.sqlite_db_file)
      if opts.ld is None:
        return
    elif opts.ld_vcf:
      print "Using user-specified VCF file to calculate LD with reference SNP %s.." % str(refsnp)
      
      ld_files = []
     
      if '.json' in opts.ld_vcf:
        ld_vcf = opts.ld_vcf_dict.get(str(chr))

        if ld_vcf is None:
          die("Error: no VCF file available for chromosome %s in JSON file: %s" % opts.ld_vcf)
      else:
        ld_vcf = opts.ld_vcf

      tabix_region = "{0}:{1}-{2}".format(chr,start,end)
      ld_temp = ld_from_vcf(opts.ld_measure,refsnp.pos,ld_vcf,tabix_region,conf.TABIX_PATH,opts.ignore_vcf_filter)
      if ld_temp is not None:
        ld_files.append(ld_temp)

      if opts.condsnps:
        for cond_snp in opts.condsnps:
          print "Using user-specified VCF file to calculate LD with conditional SNP %s.." % str(cond_snp)
          ld_temp = ld_from_vcf(opts.ld_measure,cond_snp.pos,ld_vcf,tabix_region,conf.TABIX_PATH,opts.ignore_vcf_filter)
          if ld_temp is not None:
            ld_files.append(ld_temp)
    else:
      print "Finding pairwise LD with reference SNP %s.." % str(refsnp)
      print "Source: %s | Population: %s | Build: %s" % (opts.source,opts.pop,build)
      
      ld_files = []
      ld_temp = computeLD(refsnp,chr,start,end,build,opts.pop,opts.source,opts.cache,not no_clean,opts.verbose)
      if ld_temp is not None:
        ld_files.append(ld_temp)

      if opts.condsnps:
        for cond_snp in opts.condsnps:
          print "Finding pairwise LD with conditional SNP %s.." % str(cond_snp)
          print "Source: %s | Population: %s | Build: %s" % (opts.source,opts.pop,build)
          ld_temp = computeLD(cond_snp,chr,start,end,build,opts.pop,opts.source,opts.cache,not no_clean,opts.verbose)
          if ld_temp is not None:
            ld_files.append(ld_temp)

  else:
    print "Skipping LD computations, --no-ld was given.."
    ld_temp = "NULL"

  if opts.no_ld:
    ld_final = None
  elif opts.ld:
    ld_final = opts.ld
  elif len(ld_files) > 0:
    ld_final = ld_files
  else:
    ld_final = None

  if ld_final is not None:
    args += " ldCol=%s" % opts.ld_measure

  pqueries = {}
  print "Grabbing annotations from SQLite database.."

  # Starting in 1.3, var_annot is disabled by default (opts.enable_db_annot)
  # Will be completely removed in 1.4
  pqueries = runQueries(chr,start,end,opts.snpset,build,opts.sqlite_db_file,opts.enable_db_annot,opts.gene_table)

  user_rargs = parse_rargs(shlex.split(args))
 
  user_snpset = user_rargs.get('snpset')
  if user_snpset is not None and 'NULL' in user_snpset:
    user_rargs['snpsetFile'] = "NULL"

  if opts.snpset == "NULL":
    user_rargs['snpsetFile'] = "NULL"

  for m2zarg,file in pqueries.iteritems():
    # If the user overrides any of the "pquery files", don't set it. 
    if m2zarg in user_rargs:
      continue

    if file is not None:
      args += " %s=%s" % (m2zarg,file)
    else:
      args += " %s=NULL" % m2zarg

  # Do we want to show annotations? If enable_db_annot is false, and the user hasn't specified
  # their own columns in the metal/epacts file, then we should also pass showAnnot=F. 
  has_user_annot = re.search("showAnnot|annotPch|annotCol|annotOrder",args) is not None
  if not (has_user_annot or opts.enable_db_annot):
    args += " showAnnot=F"

  # GWAS catalog.
  if opts.gwas_cat_file is not None:
    args += " gwasHits=%s" % opts.gwas_cat_file

  # Marker denote file. 
  if opts.denote_markers_file:
    denote_file = parse_denote_marker_file(opts.denote_markers_file,opts.sqlite_db_file)
    args += " denoteMarkersFile=%s" % denote_file
  
  # BED tracks file. 
  if opts.bed_tracks is not None:
    args += " bedTracks=%s" % opts.bed_tracks

  # Should we also generate SVG?
  if opts.svg:
    args += " format=pdf,svg"

  print "Creating plot.."

  runM2Z(metal_temp,opts.metal2zoom_path,ld_final,refsnp,chr,start,end,opts.no_snp_name,opts.verbose,opts,args)

  if not no_clean:
    print "Deleting temporary files.."
    pquery_files = filter(lambda x: x is not None,pqueries.values())
    cleanup([ld_temp,metal_temp] + pquery_files)

def main():
  table_pad(M2ZFAST_TITLE)
  print ""

  conf = getConf()

  # Get command-line arguments and parse them for errors.
  opts,args = getSettings()
  print "Loading settings.."
  
  # Print important options. 
  print "Options in effect are:\n"
  printOpts(opts)
  print ""

  display_args = parse_rargs(args)
  del display_args["markerCol"]
  del display_args["pvalCol"]
  if len(display_args) > 0:
    print "Plotting parameters in effect are:\n"
    printArgs(args)
    print ""
  print "Using %s.." % opts.metal2zoom_path

  # Metal2zoom arguments must be quoted to work properly on the command line.
  # i.e. title="Plot title"
  args = " ".join(['%s="%s"' % (i[0],i[1]) for i in parse_rargs(args).items()])

  # Build parameter.
  args += " build=%s" % opts.build

  # Tell locuszoom.R never to try using pquery. 
  args += " pquery=F"

  # Always disable p-value transformation in R. 
  # It is dangerous to leave it enabled - we don't always know what the R script will do. 
  args += " alreadyTransformed=TRUE"

  # Change into runtime directory.
  if _DEBUG:
    print "DEBUG: runtime directory: %s" % opts.rundir

  os.chdir(opts.rundir)

  # Single process mode - one plot generated at a time. 
  if opts.multi == 1:
    # entry[0] - snp
    # entry[1] - chr
    # entry[2] - start plotting position
    # entry[3] - end plotting position
    # entry[4] - locuszoom.R args specified in hitspec file, if one was used
    # all of these are computed in getSettings()
    for entry in opts.snplist:
      # Time the code.
      st = time.time()

      # M2Z arguments can be passed in either via the command line,
      # or by using a "hitspec" file passed into this script. They need to
      # be merged appropriately.
      iter_args = args
      if len(entry) == 5:
        if entry[4] is not None:
          more_args = entry[4]

          if M2ZCL_FIRST:
            iter_args = args + " " + more_args
          else:
            iter_args = more_args + " " + args

      # Figure out what the temp directory should be called.
      temp_dir = ""
      if opts.prefix is not None:
        temp_dir += opts.prefix + "_"
      if not opts.no_date:
        temp_dir += time.strftime("%y%m%d") + "_"
      temp_dir += windows_file_replace_chars(str(entry[0]))

      # Fix directories to not have characters that are invalid on Windows.
      temp_dir = windows_path_replace_chars(temp_dir)

      # Check that this directory is valid.
      if temp_dir in ("","/"):
        raise IOError, "Error: temporary directory is not valid, was: %s, contact developer" % temp_dir

      if _DEBUG:
        print "DEBUG: plot directory: %s" % temp_dir

      # Setup the temporary directory.
      # If it exists already, it was used on a previous plotting run, and should
      # be killed so as not to use the temporary files left in it on accident. 
      if os.path.isdir(temp_dir):
        kill_dir(temp_dir)

      # Create the directory.
      try:
        os.mkdir(temp_dir)
      except:
        print >> sys.stderr, "Error: Tried to create temporary directory %s but failed" % temp_dir
        print >> sys.stderr, "Current working directory is: %s" % os.getcwd()
        raise

      # Change into our temporary directory. This is where files will be generated
      # while creating the plot. locuszoom.R, new_fugue, and m2zfast will all
      # create files here. 
      os.chdir(temp_dir)

      # Get input file and type (metal, epacts)
      input_file = None
      input_type = None

      if opts.metal:
        input_file = opts.metal
        input_type = "metal"
      elif opts.epacts:
        input_file = opts.epacts
        input_type = "epacts"
      else:
        die("Error: did not get either --metal or --epacts.")

      # Create the plot. This runs through the whole sequence of fetching LD and running M2Z.
      try:
        runAll(input_file,input_type,entry[0],entry[1],entry[2],entry[3],opts,iter_args)
      except sqlite3.OperationalError:
        print >> sys.stderr, ""

        print >> sys.stderr, fill("Error: A sqlite3 disk error was caught - this is *usually* the "
                              "result of /tmp or /var/tmp being filled to capacity on the "
                              "machine, or a permissions error on the database file itself. ",80)
        
        print >> sys.stderr, ""

        print >> sys.stderr, fill("You should check that /tmp or /var/tmp are not filled to "
                              "capacity, and additionally check that all users can read the "
                              "database. If /tmp or /var/tmp are full, you should email the "
                              "sysadmin, and then set TMPDIR to a suitable location that is "
                              "not full. ",80)

        print >> sys.stderr, ""

        print >> sys.stderr, "For reference, your database file is currently located here: %s\n" % opts.sqlite_db_file

        print >> sys.stderr, "The original error was:"
        raise

      # Change back up to the parent directory where we started running from. 
      os.chdir("..")

      # If they only want the plot, move/rename the pdf and delete the directory created.
      if opts.plotonly:
        image_file = glob(os.path.join(temp_dir,"*.pdf"))
        image_file += glob(os.path.join(temp_dir,"*.svg"))
        exts = [os.path.splitext(i)[1] for i in image_file]
        if len(image_file) == 0:
          print >> sys.stderr, "Error: no image file found"
          kill_dir(temp_dir)
        else:
          for i in xrange(len(image_file)):
            image = image_file[i]
            ext = exts[i]

            new_image_name = ""
            if opts.prefix is not None:
              new_image_name += opts.prefix + "_"
            if not opts.no_date:
              new_image_name += time.strftime("%y%m%d") + "_"
            new_image_name += windows_file_replace_chars(str(entry[0]))
            new_image_name += ext

            try:
              move(image,new_image_name)
            except:
              print >> sys.stderr, "Error: extracting image %s failed.." % image

          kill_dir(temp_dir)

      se = time.time()
      print "Time required: %s\n" % (timeString(se - st))

  # Generate plots simultaneously. Not planning on implementing this anytime soon. 
  elif opts.multi > 1:
    die("--multi not yet implemented.")

# Call main if script is executed. 
if __name__ == "__main__":
  main()

