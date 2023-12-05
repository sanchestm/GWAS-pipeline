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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.    If not, see <http://www.gnu.org/licenses/>.
#===============================================================================

import os
import sys
import sqlite3
import imp
import logging
import tempfile
import csv
import time
from subprocess import Popen,PIPE
from optparse import OptionParser,SUPPRESS_HELP

# Fix script location.
sys.argv[0] = os.path.abspath(sys.argv[0]);

# Program constants.
LOG_NAME = "locuszoom.dbmeister";

# Config constants. 
sys.path.insert(0,os.path.join(os.path.dirname(sys.argv[0]),"../src/"));
from m2zfast import *
from m2zutils import which

def die(msg):
    print >> sys.stderr, msg;
    sys.exit(1);

def time_string(seconds):
    time_tuple = time.gmtime(seconds);
    days = time_tuple[2] - 1;
    hours = time_tuple[3];
    mins = time_tuple[4];
    secs = time_tuple[5];
    if sum([days,hours,mins,secs]) == 0:
        return "<1s";
    else:
        string = str(days) + "d";
        string += ":" + str(hours) + "h";
        string += ":" + str(mins) + "m";
        string += ":" + str(secs) + "s";
    return string;

def create_log(filepath=None):
    logger = logging.getLogger(LOG_NAME);
    
    formatter = logging.Formatter("%(message)s");
    
    if filepath is None:
        handler = logging.StreamHandler();
        handler.setFormatter(formatter);
        logger.addHandler(handler);
    else:
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=2000000,
            backupCount=3
        );
        handler.setFormatter(formatter);
        logger.addHandler(handler);
    
    logger.setLevel(logging.INFO);
    
    return logger;

def get_log():
    return logging.getLogger(LOG_NAME);

def get_settings():
    p = OptionParser();
    p.add_option("-d","--db",help="Database file to modify or create.");
    p.add_option("--snp_pos",help="Flat file for SNP positions.");
    p.add_option("--refflat",help="Flat file for RefSeq Genes (refFlat) gene information.");
    p.add_option("--knowngene",help="Flat file for UCSC Genes (knownGene) gene information.");
    p.add_option("--gencode",help="Flat file for GENCODE gene information.");
    p.add_option("--gencode-tag",help="Restrict to this GENCODE tag only.")
    p.add_option("--snp_set",help="Flat file for sets of SNPs.")
    p.add_option("--var_annot",help="Flat file for SNP annotation.");
    p.add_option("--recomb_rate",help="Flat file containing recombination rates.");
    p.add_option("--trans",help="Flat file containing translations from older "
                             "SNP builds.");
#    p.add_option("-v","--verbose",help="Enable verbosity - the script will be "
#                             "much more vocal about its activities.");
    p.add_option("-l","--log",help="Name of log file, if one is desired.");
    p.add_option("--sqlite",help="Path to SQLite3 executable, if installed. "
                             "This is not required to run the program, but can help in "
                             "the speed at which tables are loaded.");
    p.add_option("--no-cleanup",help="Don't delete temporary files after usage.",default=False,action="store_true");
    p.add_option("--no-cmd-sqlite",action="store_true",default=False,help=SUPPRESS_HELP);
    p.add_option("--debug",action="store_true",default=False,help=SUPPRESS_HELP)
    
    (opts,args) = p.parse_args();
    
    # A DB file must be specified.
    if not opts.db:
        die("Error: must provide name of database file to create or update, "
                "use -d or --db.");
    
    # Check input files for existence. 
    for arg in ('snp_pos','refflat','knowngene','snp_set','var_annot','recomb','trans'):
        file = getattr(opts,arg,None);
        if file is not None:
            if not os.path.isfile(file):
                die("Error: could not find file supplied to --%s argument." % arg);
    
    # Fix DB to be absolute path.
    opts.db = os.path.abspath(opts.db);
    
    # Unless specifically disabled, search for the sqlite3 install on host.
    if not opts.no_cmd_sqlite:
        if opts.sqlite:
            # User specified location of sqlite3 shell.. does it exist? 
            if not os.path.exists(opts.sqlite):
                die("Error: could not find file supplied to --sqlite: %s" % opts.sqlite);
        else:
            # Let's see if it's on their path. 
            try:
                sqlite_path = which("sqlite3");
                if sqlite_path != '' and sqlite_path is not None:
                    opts.sqlite = sqlite_path;
            except:
                # Nope! We'll have to fall back to using python sqlite. 
                opts.sqlite = None;

    return opts, args;

# Check a file for proper format.
# Cols is a list denoting each column's name.
# If format is incorrect, returns (False,"explanation"), otherwise (True,None).
def check_file(file,cols,delim="\t"):
    try:
        f = open(file,"U");
    except:
        get_log().error("Error: could not open file %s." % str(file));
        return False;
        
    header = f.readline().strip();
    header_s = header.split(delim);
    
    # Header should have same number of columns as cols. 
    if len(header_s) != len(cols):
        get_log().error("Error: file %s does not have the proper number of "
                                     "columns (or your delimiter is incorrect.)" % file);
        return False;
    
    # Check each header element to see if it matches the corresponding element
    # in cols. 
    for i in xrange(len(cols)):
        h = header_s[i].lower();
        c = cols[i].lower();
        
        if h != c:
            get_log().error("Error: expected column %i's header to be '%s', "
                                         "got '%s' instead." % (i,c,h));
            return False;
    
    # Check the first few lines to see if they're matching up with cols.
    # If not, likely a delimiter or missing data problem.
    i = 0;
    num_lines = 50;
    for line in f:
        e = line.split(delim);
        if len(e) != len(cols):
            get_log().error("Error: line %i did not have the expected number of columns. "
                                         "Your delimiter may be incorrect, or you may have missing data.");
            return False;
            
        if i > num_lines:
            break;
        
        i += 1;
    
    f.close();
    
    return True;

def gencode_to_refflat(gencode_file,out_file,gencode_tag=None):
    """
    Converts a GENCODE GTF into refFlat format for insertion into the database.
    We store both refFlat and GENCODE in refFlat format.
    """

    if gencode_file[-3:] == ".gz":
        f = gzip.open(gencode_file);
    else:
        f = open(gencode_file);

    try:
        tags = set(['gene_id_noversion']);
        cols = [
            'chrom',
            'provider',
            'feature_type',
            'start',
            'end',
            'score',
            'strand',
            'genomic_phase'
        ];

        per_transcript = {};

        for line in f:
            if line[0] == "#":
                continue;

            e = line.rstrip().split("\t");

            info = e[8];
            matches = re.findall("(\w+?) \"(.+?)\";",info)
            values = dict()

            for key, value in matches: 
                if key == "tag":
                    values.setdefault(key,[]).append(value)
                else:
                    values[key] = value
            
            map(tags.add,values)

            if gencode_tag is not None:
                if ("tag" in values) and (gencode_tag not in values["tag"]):
                    continue

            for i in xrange(len(cols)):
                values[cols[i]] = e[i];

            gene_id_noversion = values['gene_id'].split(".")[0];
            values.update({
                'gene_id_noversion' : gene_id_noversion
            });

            if e[2] == "transcript":
                transcript_id = values['transcript_id'];

                per_transcript[transcript_id] = {
                    'txStart' : values['start'],
                    'txEnd' : values['end'],
                    'cdsStart' : "NA",
                    'cdsEnd' : "NA",
                    'geneName' : values['gene_name'],
                    'chrom' : values['chrom'],
                    'strand' : values['strand']
                };

                per_transcript[transcript_id].setdefault('exons',[]);

            elif e[2] == "exon":
                transcript_id = values['transcript_id'];
                start = values['start'];
                end = values['end'];

                per_transcript.setdefault(transcript_id,{}).setdefault('exons',[]).append((start,end));

    finally:
        f.close();

    tags = list(tags);
    header = cols + tags;

    out_file = out_file.replace(".gz","");

    with open(out_file,'w') as out:
        print >> out, "\t".join("geneName name chrom strand txStart txEnd cdsStart cdsEnd exonCount exonStarts exonEnds".split());
        for transcript, tdata in per_transcript.iteritems():
            exon_count = len(tdata['exons']);
            exons_sorted = sorted(tdata['exons']); # sorts on first element in each tuple

            exon_starts = ",".join([i[0] for i in exons_sorted]) + ","; # for some reason refFlat always has a trailing , at the end
            exon_ends = ",".join([i[1] for i in exons_sorted]) + ","; # for some reason refFlat always has a trailing , at the end

            print >> out, "\t".join([
                tdata['geneName'],
                transcript,
                tdata['chrom'],
                tdata['strand'],
                tdata['txStart'],
                tdata['txEnd'],
                tdata['cdsStart'],
                tdata['cdsEnd'],
                str(exon_count),
                exon_starts,
                exon_ends
            ]);

# A very Java-esque way of making sure classes for interacting with a SQLite
# database work roughly the same. 
class SQLiteI():
    def load_table(self,table_name,file):
        pass
    
    def create_table(self,table,columns,types):
        pass
    
    def create_index(self,table,columns):
        pass
    
    def drop_table(self,table):
        pass

    def remove_header(self,table,column,name):
        pass

# Interface to SQLite by using the command line.
# This requires that SQLite is installed on the host machine. 
class SQLiteCommand(SQLiteI):
    def __init__(self,db,sqlite_path):
        self.db = db;
        self.path = sqlite_path;
    
    def _run_cmd(self,cmds):
        if not hasattr(cmds,'__iter__'):
            cmds = [cmds];

        get_log().debug("DEBUG: running commands: \n%s" % "\n".join(cmds))

        # Write commands to temporary file. 
        tmp = tempfile.mkstemp();
        f = os.fdopen(tmp[0],'w');

        # Optimizations courtesy David Hinds (23andme). 
        print >> f, "pragma synchronous=OFF;";
        print >> f, "pragma cache_size=500000;";
        print >> f, "pragma page_size=4096;";
        print >> f, "pragma temp_store=MEMORY;";

        for cmd in cmds:
            print >> f, cmd;
        
        f.close();
        
        # Execute commands.
        sqlite_cmd = "%s %s < %s" % (self.path,self.db,tmp[1]);
        get_log().debug("DEBUG: running %s" % sqlite_cmd);
        os.system(sqlite_cmd);
        
        # Remove temporary file. 
        os.remove(tmp[1]);
    
    def run_query(self,query):
        self._run_cmd(query);
    
    def load_table(self,table_name,file):
        get_log().info("Loading %s into table %s.." % (file,table_name));
        # Load table into database.
        self._run_cmd([
            ".separator \"\\t\"",
            ".import %s %s" % (file,table_name)
        ]);
    
    def create_table(self,table,columns,types):
        get_log().info("Creating table %s.." % table);
        spec = ", ".join(map(lambda x: " ".join(x),zip(columns,types)));
        cmd = "CREATE TABLE %s ( %s );" % (table,spec);
        self._run_cmd(cmd);
        
    # Drops a table from the database. 
    def drop_table(self,table):
        get_log().info("Dropping table %s from database %s.." % (table,self.db));
        cmd = "%s %s \"DROP TABLE IF EXISTS %s\"" % (self.path,self.db,table);
        get_log().debug("DEBUG: running %s" % cmd);
        os.system(cmd);

    def create_index(self,table,columns):
        if not hasattr(columns,'__iter__'):
            columns = [columns];
        
        get_log().info("Creating index for table %s on columns %s.." % (
            table,
            ",".join(columns)
        ))
        
        ind_name = "ind_%s_%s" % (table,"".join(columns));
        ind_spec = ",".join(columns);
        cmd = "CREATE INDEX %s ON %s (%s);" % (ind_name,table,ind_spec);
        
        self._run_cmd(cmd);

    def remove_header(self,table,column,name):
        cmd = "DELETE FROM %s WHERE %s='%s';" % (table,column,name);
        self._run_cmd(cmd);

# Interface to SQLite using the sqlite3 module in python.
# This does not require any special installation on the host machine,
# but bulk loading of data will be slower. 
class SQLitePy(SQLiteI):
    def __init__(self,db):
        try:
            self.db = sqlite3.connect(db);
        except:
            get_log().error("Error: could not connect to %s. You might not have "
                                         "permission to read/write to this location." % db);
            raise;
    
    def create_table(self,table,columns,types):
        get_log().info("Creating table %s.." % table);
        spec = ", ".join(map(lambda x: " ".join(x),zip(columns,types)));
        cmd = "CREATE TABLE %s ( %s );" % (table,spec);
        self.db.execute(cmd);
    
    def drop_table(self,table):
        get_log().info("Dropping table %s from database.." % table);
        cmd = "DROP TABLE IF EXISTS %s" % table;
        self.db.execute(cmd);
    
    def create_index(self,table,columns):
        if not hasattr(columns,'__iter__'):
            columns = [columns];
            
        get_log().info("Creating index for table %s on columns %s.." % (
            table,
            ",".join(columns)
        ))
        
        ind_name = "ind_%s_%s" % (table,"".join(columns));
        ind_spec = ",".join(columns);
        cmd = "CREATE INDEX %s ON %s (%s)" % (ind_name,table,ind_spec);
        
        self.db.execute(cmd);
    
    def remove_header(self,table,column,name):
        cmd = "DELETE FROM %s WHERE %s='%s'" % (table,column,name);
        self.db.execute(cmd);
        
    def run_query(self,query):
        self.db.execute(query);
    
    def load_table(self,table_name,file):
        def quote(x):
            return '"%s"' % x;
        
        get_log().info("Loading %s into table %s.." % (file,table_name));
        
        reader = csv.reader(open(file),delimiter="\t");
        header = reader.next();
        for line in reader:
            self.db.execute("INSERT INTO %s VALUES (%s)" % (
                table_name,
                ",".join([quote(i) for i in line])
            ));

class SQLiteLoader():
    def __init__(self,interface):
        self.dbi = interface;
    
    def load_snp_pos(self,file):
        file_ok = check_file(file,['snp','chr','pos']);
        if file_ok:
            get_log().info("Loading SNP position table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_SNP_POS);
            
            # Drop the refsnp_trans table as well. 
            self.dbi.drop_table(SQLITE_TRANS);
            
            # Create table.
            self.dbi.create_table(SQLITE_SNP_POS,['snp','chr','pos'],['TEXT','INTEGER','INTEGER']);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_SNP_POS,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_SNP_POS,'snp','snp');
            
            # Create indices.
            self.dbi.create_index(SQLITE_SNP_POS,'snp');
            self.dbi.create_index(SQLITE_SNP_POS,['chr','pos']);
            
            # Create translation table. 
            self.dbi.run_query("CREATE TABLE %s AS SELECT snp as rs_orig, "
                                                 "snp as rs_current FROM %s;" 
                                                 % (SQLITE_TRANS,SQLITE_SNP_POS));
    
            # Create indices for translation table. 
            self.dbi.create_index(SQLITE_TRANS,'rs_current');
            self.dbi.create_index(SQLITE_TRANS,'rs_orig');
        
        else:
            get_log().info("Skipping file %s due to errors.." % file);

    def load_snp_set(self,file):
        file_ok = check_file(file,['snp','snp_set']);
        if file_ok:
            get_log().info("Loading SNP set table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_SNP_SET);
            
            # Create table.
            self.dbi.create_table(SQLITE_SNP_SET,['snp','snp_set'],['TEXT','TEXT']);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_SNP_SET,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_SNP_SET,'snp','snp');

            # Create indices.
            self.dbi.create_index(SQLITE_SNP_SET,['snp']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);

    def load_recomb_rate(self,file):
        file_ok = check_file(file,['chr','pos','recomb','cm_pos']);
        if file_ok:
            get_log().info("Loading recombination rates table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_RECOMB_RATE);
            
            # Create table.
            self.dbi.create_table(SQLITE_RECOMB_RATE,
                                                ['chr','pos','recomb','cm_pos'],
                                                ['INTEGER','INTEGER','FLOAT','FLOAT']);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_RECOMB_RATE,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_RECOMB_RATE,'chr','chr');

            # Create indices.
            self.dbi.create_index(SQLITE_RECOMB_RATE,['chr','pos']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);
    
    def load_refflat(self,file):
        columns = "geneName,name,chrom,strand,txStart,txEnd,cdsStart,cdsEnd,"\
                            "exonCount,exonStarts,exonEnds".split(",");
        types = "TEXT,TEXT,TEXT,TEXT,INTEGER,INTEGER,INTEGER,INTEGER,INTEGER,"\
                        "BLOB,BLOB,INTEGER".split(",");
        file_ok = check_file(file,columns);
        if file_ok:
            get_log().info("Loading refFlat table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_REFFLAT);
            
            # Create table.
            self.dbi.create_table(SQLITE_REFFLAT,
                                                columns,
                                                types);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_REFFLAT,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_REFFLAT,'geneName','geneName');
            
            # Create indices.
            self.dbi.create_index(SQLITE_REFFLAT,['chrom','txStart','txEnd']);
            self.dbi.create_index(SQLITE_REFFLAT,['geneName']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);

    def load_gencode(self,file,tag=None):
        columns = "geneName,name,chrom,strand,txStart,txEnd,cdsStart,cdsEnd,"\
                            "exonCount,exonStarts,exonEnds".split(",");
        types = "TEXT,TEXT,TEXT,TEXT,INTEGER,INTEGER,INTEGER,INTEGER,INTEGER,"\
                        "BLOB,BLOB,INTEGER".split(",");
        file_ok = check_file(file,columns);

        table = SQLITE_GENCODE
        if tag is not None:
            table += "_" + tag

        if file_ok:
            get_log().info("Loading refFlat table from file %s.." % file);

            # Drop the original table if one existed.
            self.dbi.drop_table(table);

            # Create table.
            self.dbi.create_table(table,
                                                columns,
                                                types);

            # Load table into database.
            self.dbi.load_table(table,file);

            # Get rid of header row.
            self.dbi.remove_header(table,'geneName','geneName');

            # Create indices.
            self.dbi.create_index(table,['chrom','txStart','txEnd']);
            self.dbi.create_index(table,['geneName']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);

    def load_knowngene(self,file):
        columns = "geneName,name,chrom,strand,txStart,txEnd,cdsStart,cdsEnd,"\
                            "exonCount,exonStarts,exonEnds".split(",");
        types = "TEXT,TEXT,TEXT,TEXT,INTEGER,INTEGER,INTEGER,INTEGER,INTEGER,"\
                        "BLOB,BLOB,INTEGER".split(",");
        file_ok = check_file(file,columns);
        if file_ok:
            get_log().info("Loading knownGene (custom) table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_KNOWNGENE);
            
            # Create table.
            self.dbi.create_table(SQLITE_KNOWNGENE,
                                                columns,
                                                types);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_KNOWNGENE,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_KNOWNGENE,'geneName','geneName');
            
            # Create indices.
            self.dbi.create_index(SQLITE_KNOWNGENE,['chrom','txStart','txEnd']);
            self.dbi.create_index(SQLITE_KNOWNGENE,['geneName']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);

    def load_var_annot(self,file):
        file_ok = check_file(file,['snp','chr','pos','annot_rank']);
        if file_ok:
            get_log().info("Loading SNP annotation table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_VAR_ANNOT);
            
            # Create table.
            self.dbi.create_table(SQLITE_VAR_ANNOT,
                                                ['snp','chr','pos','annot_rank'],
                                                ['TEXT','INTEGER','INTEGER','INTEGER']);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_VAR_ANNOT,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_VAR_ANNOT,'snp','snp');

            # Create indices.
            self.dbi.create_index(SQLITE_VAR_ANNOT,['snp']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);
            
    def load_trans(self,file):
        file_ok = check_file(file,['rs_orig','rs_current']);
        if file_ok:
            get_log().info("Loading SNP annotation table from file %s.." % file);
            
            # Drop the original table if one existed.
            self.dbi.drop_table(SQLITE_TRANS);
            
            # Create table.
            self.dbi.create_table(SQLITE_TRANS,
                                                ['rs_orig','rs_current'],
                                                ['TEXT','TEXT']);
            
            # Load table into database. 
            self.dbi.load_table(SQLITE_TRANS,file);
            
            # Get rid of header row.
            self.dbi.remove_header(SQLITE_TRANS,'rs_orig','rs_orig');

            # Create indices.
            self.dbi.create_index(SQLITE_TRANS,['rs_orig']);
            self.dbi.create_index(SQLITE_TRANS,['rs_current']);
        else:
            get_log().info("Skipping file %s due to errors.." % file);

# Entry point. 
def main():
    # Settings. 
    (opts,args) = get_settings();
    
    # Setup our log.
    # If --log was specified, creates a log file.
    # Otherwise, messages are spit to the console. 
    create_log(opts.log);
    if opts.debug:
        get_log().setLevel(logging.DEBUG);
    
    # Set SQL interface. 
    if opts.sqlite:
        get_log().debug("DEBUG: using command line sqlite3");
        sqlitei = SQLiteCommand(opts.db,opts.sqlite);
        loader = SQLiteLoader(sqlitei);
    else:
        get_log().debug("DEBUG: using python sqlite3");
        get_log().warning("Using python implementation of sqlite - this will be "
                                         "much slower than installing sqlite3 on your system.");
        sqlitei = SQLitePy(opts.db);
        loader = SQLiteLoader(sqlitei);
    
    # Load tables.
    start = time.time();
    if opts.snp_pos:
        loader.load_snp_pos(opts.snp_pos);
        
    if opts.snp_set:
        loader.load_snp_set(opts.snp_set);
        
    if opts.recomb_rate:
        loader.load_recomb_rate(opts.recomb_rate);
        
    if opts.trans:
        loader.load_trans(opts.trans);
        
    if opts.refflat:
        loader.load_refflat(opts.refflat);
        
    if opts.knowngene:
        loader.load_knowngene(opts.knowngene);

    if opts.gencode:
        # We need to convert the GENCODE annotation GTF into refFlat format.
        gencode_refflat_file = opts.gencode.replace(".gtf.gz",".refFlat.tab");
        gencode_to_refflat(opts.gencode,gencode_refflat_file,opts.gencode_tag);

        # Now we can load the refFlat formatted version of GENCODE
        loader.load_gencode(gencode_refflat_file,opts.gencode_tag);

        if not opts.no_cleanup:
            try:
                os.remove(gencode_refflat_file);
            except:
                pass

    if opts.var_annot:
        loader.load_var_annot(opts.var_annot);

    end = time.time();
    get_log().debug("DEBUG: duration %s" % time_string(end - start));

if __name__ == "__main__":
    main();
