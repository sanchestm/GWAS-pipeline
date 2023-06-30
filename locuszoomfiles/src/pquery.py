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

import sqlite3
import sys

def chr2ucsc(c):
  c = int(c)
  if c < 23 and c > 0:
    return 'chr' + str(c)
  elif c == 23:
    return 'chrX'
  elif c == 24:
    return 'chrY'
  elif c == 25:
    return 'chrmito'
  elif c == 26:
    return 'chrXY'
  else:
    return None

def sql_if(test,if_true,if_false):
  if test:
    return if_true
  return if_false

def print_results(cur,delim="\t",out=sys.stdout):
  def na_none(x):
    if x == None:
      return 'NA'
    else:
      return x
  
  header = [i[0] for i in cur.description]
  print >> out, delim.join(header)
  count = 0
  for row in cur:
    row = map(na_none,row)
    print >> out, delim.join([str(i) for i in row])
    count += 1
    
  return count

#def snp_annot_in_region(db,snp_pos_table,var_annot_table,chr,start,stop,build):
#  db.create_function("if",3,sql_if)
#  
#  query = """
#    SELECT
#      map.snp,
#      if(tx_framestop,        1,
#        if(tx_splice,           2,
#        if(tx_nonsyn,           3,
#        if(tx_coding,           4,
#        if(tx_utr3 or tx_utr5,  5,
#        if(tfbsConsSites,       6,
#        if(mcs44placental,      7,
#      0))))))) as annot_rank,
#      v.tx_framestop,
#      v.tx_nonsyn,
#      v.tx_splice,
#      v.tx_coding,
#      v.tx_utr3,
#      v.tx_utr5,
#      v.tfbsConsSites
#    FROM
#      %s map
#      left join %s v on (v.snp = map.snp)
#    WHERE
#      chr = ?
#      and (pos >= ?)
#      and (pos <= ?)
#      and (build = ?)
#  """ % (snp_pos_table,var_annot_table)
#
#  result = db.execute(query,[chr,start,stop,build])
#  return result

#def snp_annot_in_region(db,snp_pos_table,var_annot_table,chr,start,stop,build): 
#  query = """
#    SELECT
#      map.snp,
#      annot_rank
#    FROM
#      %s map
#      left join %s v on (v.snp = map.snp)
#    WHERE
#      chr = ?
#      and (pos >= ?)
#      and (pos <= ?)
#  """ % (snp_pos_table,var_annot_table)
#
#  result = db.execute(query,[chr,start,stop])
#  return result

def snp_annot_in_region(db,snp_pos_table,var_annot_table,chr,start,stop,build):
  query = """
    SELECT
      snp,annot_rank
    FROM
      %s
    WHERE
      chr = ?
      and (pos >= ?)
      and (pos <= ?)
  """ % (var_annot_table)

  result = db.execute(query,[chr,start,stop])
  return result

def recomb_in_region(db,recomb_table,chr,start,stop,build):
  query = """
    SELECT chr, pos, recomb, cm_pos
    FROM %s
    WHERE
      chr = ?
      and pos BETWEEN ? and ?
    ORDER BY chr, pos
  """ % recomb_table

  return db.execute(query,(chr,start,stop))

def snpset_in_region(db,snp_pos_table,snp_set_table,snp_set,chr,start,stop,build):
  snp_set = ",".join(["'" + i.strip() + "'" for i in snp_set.split(",")])
  query = """
    SELECT ss.snp, p.chr, p.pos, ss.snp_set
    FROM %s ss, %s p
    WHERE
      ss.snp = p.snp
      and p.chr = %s
      and ss.snp_set in (%s)
      and p.pos >= %s and p.pos <= %s
  """ % (
    snp_set_table,
    snp_pos_table,
    chr,
    snp_set,
    start,
    stop
  )

  return db.execute(query)

def refflat_in_region(db,refflat,chr,start,stop,build):
  query = """
    SELECT *
    FROM %s
    WHERE
      chrom = ?
      and txEnd >= ?
      and txStart < ?
  """ % refflat

  return db.execute(query,(chr2ucsc(chr),start,stop))

def test():
  # Connect.  
  db = sqlite3.connect("fusion_100423.db")

  # add if function
  db.create_function("if",3,sql_if)

#  cur = snp_annot_in_region(db,1,0,100000,"hg18")
#  print_results(cur,"\t")

#  cur = recomb_in_region(db,"recomb_rate_hg18",1,0,100000)
#  print_results(cur,"\t")

  cur = refflat_in_region(db,"refFlat_hg18",1,0,10000)
  print_results(cur)

def main():
  test()

if __name__ == "__main__":
  main()
