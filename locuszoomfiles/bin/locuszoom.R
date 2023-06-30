#! /usr/bin/Rscript --vanilla 
# --default-packages=utils,stats,lattice,grid,getopts
# need to check if the line above works on the web deployment machine.

# Copyright 2010 Randall Pruim, Ryan Welch
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

suppressPackageStartupMessages({
  require(stats);
  require(utils);
  require(grid);
  require(lattice);
  require(methods);
});

omittedGenes <- character(0);     # will be set in gobalenv()
omittedFineMap <- character(0);
omittedGWAS <- character(0);
warningMessages <- character(0);  # will be set in gobalenv()

options(error = function(...) {
 sink(NULL);
 sink(NULL,type="message");
 recover(...);
})

my_browser = function() {
  sink(NULL);
  sink(NULL,type="message");
  sink(NULL);
  sink(NULL,type="message");
  browser();
}

################################################################################################
# function definitions
################################################################################################

stextGrob <- function (label, r=0.1, x = unit(0.5, "npc"), y = unit(0.5, "npc"), 
                       just = "centre", hjust = NULL, vjust = NULL, rot = 0, check.overlap = FALSE, 
                       default.units = "npc", name = NULL, gp = gpar(), vp = NULL){

  let <- textGrob("a")
  
  tg <- textGrob(label=label, x=x, y=y, gp=gpar(col="black"),
                 just = just, hjust = hjust, vjust = vjust, rot = rot,
                 check.overlap = check.overlap, 
                 default.units = default.units)
  
  tgl <- c(lapply(seq(0, 2*pi, length=18), function(theta){

    textGrob(label=label,x=x+cos(theta)*r*grobWidth(let),
             y=y+sin(theta)*r*grobHeight(let), gp=gpar(col="white"),
             just = just, hjust = hjust, vjust = vjust, rot = rot,
             check.overlap = check.overlap, 
             default.units = default.units)
    
    }), list(tg))
  

  g <- gTree(children=do.call(gList, tgl), vp=vp, name=name, gp=gp)

}

parse_expression = function(x) {
  expr = x
  
  try({expr = parse(text=x)},silent=T)
  
  return(expr)
}

################################################################################
#
# takes string and converts '' and 'null' (case insensitive) to NULL, else unchanged.
#
as.filename <- function(x) {
  if (! is.character(x) || toupper(x) == toupper('null') || x == '') {
    return(NULL)
  } else {
    return(x)
  }
}

# Quick function to change column names of data frame
# (by name, not by position.)
change_names = function(dframe,name_list) {
  for (n in names(name_list)) {
    names(dframe)[names(dframe)==n] = name_list[[n]];
  }
  dframe;
}

################################################################################
#
# modify column names the way R does
#
char2Rname <- function(x) {
  x <- gsub('-','.',x);
  x <- gsub('=','.',x);
  x <- gsub('\ ','.',x);
  x <- gsub("^#","X.",x);
  x <- gsub("#",".",x);
  return(x)
}

################################################################################
#
# build a factor
#
MakeFactor <- function(x,levels,na.level=NA) {

  
  f <- factor(x, levels=levels)
  if (any( is.na(f))){
    levels(f) <- c(levels(f),na.level)
    f[ is.na(f) ] <- na.level
  }
  return(f)
}

################################################################################
#
# pretty print scientific notation from log10(x)
#
log2sci <- function(x) {
  # write x as 10^(e) * 10^r where e is an integer and r is in [0,1)
  e <- floor(x)
  r <- x - e
  m <- 10^(r)
    return(paste(format(m,digits=3),"E",e,sep=""));
}

################################################################################
#
# Sniff a vector to see if it smells
#
Sniff <- function(vec,type=c('snp'),n=10) {
  n <- min(n,length(vec))
  type <- match.arg(type)

  if (type == 'snp') {
    yes <- union(union(
        grep('rs',vec[1:n]), 
        grep('chr[[:digit:]]+:[[:digit:]]',vec[1:n])),
        grep('chr[[:alpha:]]+:[[:digit:]]',vec[1:n])
        )
    if ( length(yes) == n ) return (TRUE)
    return(FALSE)
  }
  return(FALSE)
}

################################################################################
#
# Which side should legend go on?
#
AutoLegendSide <- function(pval,pos,posRange = range(pos)) {
  M <- .5 * max(pval);
  left <- min(pos[pval > M]);
  right <- max(pos[pval > M]);
  mid <- mean(posRange);
  if ( (mid - left) > (right - mid) ) { 
    return ('right'); 
  }
  return('left');
}

################################################################################
#
# choose a transformation for p-values
#
SetTransformation <- function(mn,mx,alreadyTransformed=FALSE) {
  if (alreadyTransformed) { return ( function(x){x} ) }
  if (mx > 1)  { return( function(x) {x} ) }  # assume -log10 transform has already been applied
  if (mn < 0)  { return( function(x) {-x} ) } # assume log10 transform has already been applied
  return ( function(x) {-log10(x)} )
}

################################################################################
#
# Set titles correctly for D' and r^2
#
SetLDTitle <- function(col,title) {
  if ( col == 'dprime' & is.null(title) ) { return ("D'"); }
  if ( col == 'rsquare' & is.null(title) ) { return (expression(r^2)); }
  if (is.null(title)) { return (""); }
  return(title);
}

################################################################################
#
# extends default modifyList so that it handles NULL values in list differently.
#
ModifyList <- function (x, val, replaceWithNull = FALSE) 
{
    stopifnot(is.list(x), is.list(val));
    xnames <- names(x);
    for (v in names(val)) {
    if (v %in% xnames && is.list(x[[v]]) && is.list(val[[v]])) {
          x[[v]] <- ModifyList(x[[v]], val[[v]], replaceWithNull=replaceWithNull);
    } else {
      if (!is.null(val[[v]]) || replaceWithNull) {
        x[[v]] <- val[[v]];
      }
    }
    }
    return(x);
}

################################################################################
#
# like lapply but on a subset of a list; rest of list unchanged.
#

sublapply <- function(x,names=names(x),fun) {

    fun <- as.function(fun);

    for (name in names) {
        if (! is.null(x[[name]] ) ) {
            x[[name]] <- fun(x[[name]]);
        }
    }
    return(x);
}

################################################################################
#
# like modifyList, but works when names of val are unique prefixes of names of x
#
ConformList <- function(x,names,case.sensitive=FALSE,message=FALSE) {

  own.ind <- 0;
  for (name in names(x)) {
    own.ind <- own.ind + 1;
    if (case.sensitive) {
      match.ind <- pmatch( name, names );
    } else {
      match.ind <- pmatch( toupper(name), toupper(names) );
    }
    if (! is.na(match.ind)) {
      names(x)[own.ind] <- names[match.ind];
    } else {
      if (! is.null(message) ) { message(paste("No unique match for ",name,"=",x[[own.ind]],sep="")); }
    }
  }
  return(x);
}

################################################################################
#
# like modifyList, but works when names of val are unique prefixes of names of x
#
PModifyList <- function(x,val,...) {
  ModifyList(x,ConformList(val,names(x),...));
}

################################################################################
#
# Modify the list args according to the value of theme
#
ProcessOneTheme <- function(args,theme) {
  if ( is.character(theme) ){
    theme=paste(theme,'.theme',sep='')
    return( do.call(theme,list(args=args)) )
  }
  return(args)
}

################################################################################
#
# process a list of themes in order
#
ProcessThemes <- function(args,themeString) {
  if (!is.character(themeString)) { return(args) }

  for (theme in unlist(strsplit(themeString,",")) ){
    args <- ProcessOneTheme(args,theme)
  }
  return(args)
}

################################################################################
#
# Some themes
#
ryan.theme <- function(args) {

  argUpdates <- list(
    snpset=NULL,
    format="pdf",
    refDot=NULL,
    geneFontSize=1.1,
    refsnpTextSize=1.5,
    axisTextSize=1.45,
    legendSize=1,
    legendFrameAlpha=0,
    legendAlpha=0,
    axisSize=1.45,
    recombPos=3, 
    xlabPos=-2.75,
    height=9,
    rfrows='4'
  )
  return(ModifyList(args,argUpdates));
}

publication.theme <- ryan.theme;
pub.theme <- ryan.theme;

black.theme <- function(args) {
  argUpdates <- list(
    axisTextColor='black',
    rugColor='black',
    frameColor='black'
  )
  return(ModifyList(args,argUpdates));
}

giant.theme <-  function(args) {
  argUpdates <- list(
    rfrows=10,
    recombOver=TRUE,
    recombAxisColor='black',
    recombAxisAlpha=1,
    legend='auto',
    showAnnot=TRUE,
    showRefsnpAnnot=FALSE,
    annotPch='25,21,21,21,21,21,24,24,24',
    recombColor='cyan',
    ldColors='gray50,blue,green,yellow,orange,red,purple3'
  )

  args <- ryan.theme(args);
  args <- black.theme(args);
  args <- ModifyList(args,argUpdates);
  return(args);
}

#############################################################
#
# Remove temporary files (used in final clean-up)
#
RemoveTempFiles <- function (...) {
  l <- list(...);
  removedFiles <- list();

  if (length(l) < 1) { return(removedFiles); }

  method <- unlist(lapply(l,function(x) { attr(x,"method") }));
  file <- unlist(lapply(l,function(x) { attr(x,"file") }));
  for (i in 1:length(method)) {
    if (method[i] == 'pquery') { file.remove( file[i] ); }
    removedFiles <- c(removedFiles, file[i]);
  }

  return(removedFiles); 
}

#############################################################
#
# Cleaning up at the end
#
CleanUp <- function(args,...) {
  if (args[['clean']]) {
      message("\nCleaning up.  The following files are begin removed: ");
      files <- RemoveTempFiles(...);
      message(paste("\t",files,"\n"));
      invisible(files);
  }
}

#############################################################
#
# Obtain data.  Data can be specified using a file or 
# a pquery command.  Pquery is a tool that simplifies
# querying SQL databases.  Most users will simply pass in files
# Much of this is handled by the Python wrapper anyway.
#

GetDataFromFileOrCommand <- function(file, command, default=data.frame(), clobber=FALSE, verbose=TRUE,...) {
  method <- "file";
  if ( !file.exists(file) || clobber ) {
    command <- paste(command,">",file);
    if (verbose) { cat(paste("Getting data with",command,sep="\n")); }
    system(command);
    method <- 'pquery';
    if (! clobber) {
      assign("warningMessages",c(warningMessages,paste("Missing file:",file)), globalenv());
    }
  }
  results <-  read.file(file,...) ;
  attr(results, "file") <- file;
  attr(results, "command") <- command;
  attr(results, "method") <- method;
  return(results);
}

#############################################################
#
# Obtain data.  Data can be specified using a file.  When
# pquery is not available, this version will ignore the command
# and return default data if the file is missing.
#

GetDataFromFileIgnoreCommand <- function(file, command, default=data.frame(), clobber=FALSE, verbose=TRUE,...) {
  method <- "file";
  if (is.null(file)) {
    return(default)
  }
  if ( !file.exists(file) ) { 
    # warning(paste("Missing file:",file)) 
    return(default)
  }

  results <-  read.file(file,...) ;
  attr(results, "file") <- file;
  attr(results, "method") <- method;
  return(results);
}

LoadFineMap = function(file,verbose=TRUE,...) {
  if (!file.exists(file)) {
    # try directory above
    file = file.path("..",file);
    if (!file.exists(file)) {
      warning("could not find fine mapping file..");
      return(NULL);
    }
  }

  ppdata = read.table(file,header=T,sep="\t",stringsAsFactors=F);
  names(ppdata) = tolower(names(ppdata))

  # Do we have the correct columns? 
  req_cols = c("snp","chr","pos","pp","group","color");
  if (!all(req_cols %in% names(ppdata))) {
    errmsg = paste(
      "fine mapping pp file given, but did not have correct columns (or header)",
      "required columns are: ",
      paste(req_cols, collapse = ", ")
      ,collapse = "\n"
    );
    warning(errmsg)
    return(NULL);
  }

  # Coordinates need to be in MB...
  ppdata$pos = ppdata$pos / 1E6;

  ppdata;
}

LoadGWASHits = function(file,verbose=TRUE,...) {
  if (!file.exists(file)) {
    # try directory above
    file = file.path("..",file);
    if (!file.exists(file)) {
      warning("could not find gwas file..");
      return(NULL);
    }
  }

  gwas = read.table(file,header=T,sep="\t",comment.char="",stringsAsFactors=F,quote="");
  names(gwas) = tolower(names(gwas));

  # Do we have the correct columns? 
  if (!all(c("chr","pos","trait") %in% names(gwas))) {
    warning("gwas hits file given, but did not have correct columns (or header)")
    return(NULL);
  }

  gwas$pos = gwas$pos / 1E6;

  gwas;
}

LoadBarplotData = function(file,chrom_col="chr",pos_col="pos",value_col="value",verbose=TRUE,...) {
  if (!file.exists(file)) {
    # try directory above
    file = file.path("..",file);
    if (!file.exists(file)) {
      warning("could not find barplot data file..");
      return(NULL);
    }
  }

  barplotdf = read.table(file,header=T,sep="\t",comment.char="",stringsAsFactors=F,quote="");

  # Do we have the correct columns? 
  if (!all(c(chrom_col,value_col,pos_col) %in% names(barplotdf))) {
    warning("barplot data file given, but did not have correct columns (or header)")
    return(NULL);
  }

  barplotdf[,"chr"] = gsub("chr","",barplotdf$chr)
  barplotdf[,"pos_int"] = barplotdf[,pos_col]
  barplotdf[,"pos"] = barplotdf[,pos_col] / 1E6;

  barplotdf;
}

df_empty = function(dframe) { return(dim(dframe)[1] == 0) }

#############################################################
#
# return an empty data from with some additonal attributes
#
empty.data.frame <- function(
  file="none",command="none", method="empty.data.frame") {

  result <- data.frame();
  attr(result, "file") <- file;
  attr(result, "command") <- command;
  attr(result, "method") <- method;
}


#############################################################
#
# This is used to clone values from user specified arguemnts 
# to other arguments that the user did not set.  
# 
MatchIfNull <- function(args,toupdate,updatewith) {
  if ( is.null(args[[toupdate]]) ) {
    args[[toupdate]] <- args[[updatewith]]
  }
  return(args)
}

#############################################################
#
# All arguments are passed in with mode character (i.e., as strings)
# This functions converts arguments to the correct mode for
# internal use.
#
AdjustModesOfArgs <- function(args) {
  args <- sublapply(args,
    c('legendAlpha', 'width','height',
      'frameAlpha','hiAlpha','rugAlpha',
      'refsnpLineAlpha', 'recombFillAlpha','recombLineAlpha', 'refsnpTextAlpha', 'refsnpLineWidth',
      'ymin','ymax','legendSize','refsnpTextSize','axisSize','axisTextSize','geneFontSize','smallDot',
      'largeDot','refDot','ldThresh','rightMarginLines','barplotMinY','barplotMaxY'),
    as.numeric);

  args <- sublapply(args,
    c('metal','recomb','ld','refSnpPosFile','snpsetFile','annot','refFlat','denoteMarkersFile'),
    as.filename);

  args <- sublapply(args,
    c('chr','unit','xnsmall','fmrows','gwrows','barplotRows','condRefsnpPch'),
    as.integer);

  args <- sublapply(args,
    c('condLdLow'),
    as.character);
  
  args <- sublapply(args,
    c('experimental','clobber','recombOver','recombFill','pquery','drawMarkerNames',
      'showRecomb','showAnnot','showRefsnpAnnot','bigDiamond','showPartialGenes','shiftGeneNames',
      'clean', 'dryRun','legendMissing','hiRequiredGene','refsnpShadow'),
    as.logical);

  args <- sublapply( args,
    c('ldCuts','xat','yat','annotPch','condPch','signifLine','signifLineWidth','barplotAxisTicks'),
    function(x) { as.numeric(unlist(strsplit(x,","))) } );
  
  args <- sublapply( args,
    c('condLdColors','signifLineColor','requiredGene'),
    function(x) { 
      x = gsub("\\s+","",x)
      as.character(unlist(strsplit(x,",")))
    } 
  );

  if (!is.null(args[['weightRange']])) {
    args <- sublapply( args,
      c('weightRange'),
      function(x) { as.numeric(unlist(strsplit(x,","))) } );
  }

  args <- sublapply( args,
    c('rfrows','signifLineType'),
    function(x) { as.integer(unlist(strsplit(x,","))) } );

  args <- sublapply( args,
    c('ldColors', 'format', 'annotOrder','cond_ld'),
    function(x) { unlist(strsplit(x,",")) } );

  return(args);
}

#############################################################
#
# Returns text description of unit along chromosome depending
# on value of unit where unit is a number of base pairs
#
unit2char <- function(unit) {
  if (unit == 1000000) { return ("(Mb)"); }
  if (unit == 1000) { return ("(Kb)"); }
  return("");
}

#############################################################
#
# convert position that might include mb, kb into a base pair position
#
pos2bp <- function(pos) {
  unit<-1;
  posN <- as.character(pos);
  if (regexpr("kb",posN,ignore.case=TRUE) > 0) {
    unit <- 1000;
    posN <- sub("kb","",posN, ignore.case=T);
  }
  if (regexpr("mb",posN,ignore.case=TRUE) > 0) {
    unit <- 1000000;
    posN <- sub("mb","",posN, ignore.case=T);
  }
# message(paste('posN = ',posN, "  unit = ", unit));
  return( as.numeric(posN) * unit);
}

#############################################################
#
# read file, using filename to determine method.
#

read.file <- function(file,header=T,na.strings=c('NA','','.','na'),...) {
  if (! file.exists(file) ) { 
    return(NULL);
    message(paste("Missing file: ", file));
  }

  df = read.table(file,header=header,na.strings=na.strings,...);
  return(df);
}

#############################################################
#
# write file, using filename to determine method.
#

write.file <- function(x, file, append=FALSE, clobber=TRUE, na='NA') {
  if (file.exists(file) && ! clobber && !append ) { 
    return(NULL);
    message(paste("File already exists: ", file));
  }

  # if file ends .csv, then write.csv
  if ( regexpr("\\.csv",file) > 0 ) {
    return(write.csv(x,file,append=append));
  }

  # if file ends .Rdata, then load
  if ( regexpr("\\.Rdata",file) > 0 ) {
    return(save.csv(x,file));
  }

  # default is read.table
  return(write.table(x,file,append=append));
}

#############################################################
#
# Converter from chrom (e.g., chr13) to chr (e.g., 13) format
#
chrom2chr <- function (x) {
        y <- substring(x,first=4);
        y[y=='X'] = '23';
        y[y=='Y'] = '24';
        y[y=='mito'] = '25';
        y[y=='XY'] = '26';
        as.numeric(y);
}

#############################################################
#
# Converter from chr (e.g., 13) to chrom (e.g., chr13) format
#
chr2chrom <- function (x) {
        if (x == 23 ) { return("chrX"); }
        if (x == 24 ) { return("chrY"); }
        if (x == 25 ) { return("mito"); }
        if (x == 26 ) { return("chrXY"); }
        return (paste("chr",as.numeric(x),sep=""));
}


#############################################################
#
# Linearly rescale values to fit in interval 
# If all values are equal, then return a vector the same length as x
# with all values set to constant (by default the larger end of the interval).
#
rescale <- function(x, 
          oldScale=range(x, na.rm=TRUE), 
          newScale = c(0,1), 
          transformation = function(x) x )
{
  if (is.null(oldScale)) oldScale=range(x, na.rm=TRUE)  # treat NULL like missing
  x <- transformation(x)
  oldScale <- transformation(oldScale)
  if ( length(newScale) != 2 || ! is.numeric(newScale) ||
          length(oldScale) != 2 || ! is.numeric(oldScale) ) 
    { return (x); }

  a <- oldScale[1]; b <- oldScale[2];
  u <- newScale[1]; v <- newScale[2];
  
  print( c(a=a,b=b,u=u,v=v) )
  print(summary(x))

  r <- v - (b-x)/(b-a) * (v-u);
  r[r < u] <- u;
  r[r > v] <- v;
  return(r);
}

#############################################################
#
# Flatten information originally in UCSC bed format.
# Tailored to fit nominclature and formatting used in files
# generated by Peter Chines.
#

flatten.bed <- function(x,multiplier=.001) {

    if (prod(dim(flatten.bed)) == 0) {
      df <- data.frame(
            chrom  = c("chr0","chr0","chr0"),
            chr    = c(0,0,0),
            start= c(0,0,0),
            stop = c(2,2,2),
            type = c(0,2,1),
            name = c('none','none','none'),
            nmName = c('none','none','none'),
            strand = c('+','+','+')
            );
      return(df);
    }

    x$geneName <- as.character(x$geneName);
    x$name <- as.character(x$name);
    x$strand <- as.character(x$strand);
    lx <- dim(x)[1];

    blockStart <- unlist(lapply(
                strsplit(as.character(x$exonStarts),split=','),
                as.numeric));
    blockEnd <- unlist(lapply(strsplit(as.character(x$exonEnds),split=',')
                ,
                as.numeric));
    blockSize = blockEnd - blockStart;
    nameDup = rep(x$geneName,times=x$exonCount);
    nmNameDup = rep(x$name,times=x$exonCount);
    startDup = rep(x$txStart,times=x$exonCount);
    stopDup = rep(x$txEnd,times=x$exonCount);
    chromDup = rep(x$chrom,times=x$exonCount);
    strandDup = rep(x$strand,times=x$exonCount);

# types: 
# 0 = txStart to txEnd      (transcription region)
# 1 = exonStart to exonEnd  (exons) 
# 2 = cdsStart to cdsEnd    (coding region)

    df <- data.frame(
            chrom  = c(x$chrom, x$chrom, chromDup),
            chr = chrom2chr(c(x$chrom, x$chrom, chromDup)),
            start= c(x$txStart, x$cdsStart, blockStart),
            stop = c(x$txEnd, x$cdsEnd, blockEnd ),
            type = c(rep(0,lx),rep(2,lx), rep(1,length(startDup))),
            name = c(x$geneName, x$geneName, nameDup),
            nmName = c(x$name, x$name, nmNameDup),
            strand = c(x$strand, x$strand, strandDup)
            );

    df$start <- df$start * multiplier;
    df$stop <- df$stop * multiplier;

    invisible(df);
}

# TODO: add in shadow support for denote marker feature (label if block below) and test
grid.refsnp <- function(name,pos,pval,draw.name=TRUE,label=NULL,color=NULL,shadow=FALSE) {

  if (draw.name) {
    # Figure out text height. 
    text_height = grobHeight(textGrob(as.character(name),gp = gpar(cex = args[['refsnpTextSize']])));
    
    if (!is.null(label) && !is.na(label) && label != "") {
      grid.text(
        as.character(label),
        x=unit(pos,"native"), 
        y=unit(pval,'native') + unit(1.15,'lines'),
        just=c("center","top"),
        gp=gpar(
          cex=args[['refsnpTextSize']],
          col=ifelse(is.null(color) || is.na(color),args[['refsnpTextColor']],color),
          alpha=args[['refsnpTextAlpha']]
        )
      );
      
      grid.text(
        as.character(name),
        x=unit(pos,"native"), 
        y=unit(pval,'native') + unit(1.15,'lines') + text_height + unit(0.25,'lines'),
        just=c("center","top"),
        gp=gpar(
          cex=args[['refsnpTextSize']],
          col=ifelse(is.null(color) || is.na(color),args[['refsnpTextColor']],color),
          alpha=args[['refsnpTextAlpha']]
        )
      );
      
    } else {
      if (TRUE & "gridExtra" %in% installed.packages()[,1] & shadow) { grob_func = stextGrob }
      else { grob_func = textGrob }

      refsnp_text = grob_func(
        as.character(name),
        x=unit(pos,"native"), 
        y=unit(pval,'native') + unit(1.15,'lines'), 
        just=c("center","top"),
        gp=gpar(
          cex=args[['refsnpTextSize']],
          col=ifelse(is.null(color) || is.na(color),args[['refsnpTextColor']],color),
          alpha=args[['refsnpTextAlpha']]
        )
      );

      grid.draw(refsnp_text);
    }
  }
  
  grid.segments(
    x0=unit(pos,"native"),
    x1=unit(pos,"native"),
    y0=unit(0,"npc"),
    y1=unit(pval,'native'),
    gp=gpar(
      col=args[['refsnpLineColor']],
      lwd=args[['refsnpLineWidth']],
      alpha=args[['refsnpLineAlpha']],
      lty=args[['refsnpLineType']]
    )
  );
}

#############################################################
#
# calculte width of text
#
textWidth <- function(text="",gp=gpar()) {
  return ( grobWidth(textGrob(text,gp=gp)) );
}

textHeight <- function(text="",gp=gpar()) {
  return ( grobHeight(textGrob(text,gp=gp)) );
}

#############################################################
#
# generate text with arrow (or just compute width of same)
# this is a bit crude and clunky
#
arrowText <- function(text,x=unit(.5,'npc'), y=unit(.5,'npc'), direction='+',name=NULL,gp=gpar(),
        check.overlap=TRUE, widthOnly=FALSE) {

  tWidth <- textWidth(text,gp)
  aWidth <- textWidth('xx,',gp)

  tHeight <- textHeight(text,gp)

  if (widthOnly) { 
    return( convertWidth(tWidth + aWidth),unitTo='inches',valueOnly=TRUE ) 
  }
  
  cWidth <- .1 * textWidth(',',gp)
  
  if ( direction %in% c('+','forward','->','>','right') ) {
    mult = 1
  } else {
    mult = -1
  }

  tg <- textGrob(text,
    x=x - mult * .5 * aWidth, y=y,
    check.overlap=check.overlap,
    gp=gp,
    name="label"
  )

  ag <- linesGrob(  
    x = unit.c(
      x - mult * .5 * aWidth  +  .5 * mult * tWidth + mult * unit(.005,'npc'), 
      x + mult * .5 * aWidth  +  .5 * mult * tWidth ) ,
    y = unit.c(y,y),
    name = "arrow",
    #gp = gp,
    arrow = arrow(
      type='open',
      angle = 20,
      length = .75 * textWidth('x')
    )
  )

  rect1 <- rectGrob(x=x - .5 * mult * aWidth, y=y, width=tWidth, height=.1);
  rect2 <- rectGrob(x=x + .50 * mult * tWidth, y=y, width=aWidth, height=.1,gp=gpar(col="red"));

  result <- gTree(children=gList(tg,ag),name=name)

  attr(result,'width') <- convertX(tWidth + aWidth,'inches')
  attr(result,'twidth') <- convertX(grobWidth(tg),'inches')
  attr(result,'awidth') <- convertX(grobWidth(ag),'inches')
  attr(result,'theight') <- tHeight
  attr(result,'cWidth') <- cWidth
  attr(result,'tWidth') <- tWidth
  attr(result,'aWidth') <- aWidth
  return(result)
}

#############################################################
#
# hilite a particular region on the plot
#
panel.hilite <- function(range=c(lo,hi),lo,hi,col="transparent",fill="blue",alpha=.1){
  grid.rect( x=unit(range[1],"native"),width=unit(range[2]-range[1],"native"),
        hjust=0,
        gp=gpar(fill=fill,col=col, alpha=alpha)
        );
}

multiRibbonLegend = function(strings,ld_cuts,ld_colors,gp = NULL) { 
  max_str_width = max(sapply(strings,function(x) convertUnit(stringWidth(x),'npc')));
  max_str_width = unit(max_str_width,'npc');
  
  tree = vpTree(
    # Parent viewport
    viewport(
      name = "legend_outer", 
      layout = grid.layout(
        2, 
        2, 
        widths = unit.c(unit(1,"null"),max_str_width + unit(4,'char')),
        heights = unit.c(
          unit(2,'lines'),
          unit(length(strings) + 0.5*length(strings) + 0.75 + 1.75,'lines')
        )
      ),
    ), 
    # Child viewports
    vpList(
      viewport(
        layout.pos.col = 1, 
        layout.pos.row = 1, 
        name = "legend_title"
      ), 
      viewport(
        layout.pos.col = 1, 
        layout.pos.row = 2, 
        name = "legend_ribbons"
      ),
      viewport(
        layout.pos.col=2,
        layout.pos.row=2,
        name = "legend_strings"
      )
    )
  );
    
  ld_to_npc = function(x) { 
    x + abs(x - 0.5) * sign(0.5 - x) * 0.1;
  }
  
  ld_cuts_npc = ld_to_npc(ld_cuts);
  
  rect_x = rep(head(ld_cuts_npc,-1),length(strings));
  rect_y = unlist(sapply(seq(0.5,1.5 * length(strings), 1.5),rep,length(ld_cuts)-1,simplify=F));
  
  #cols = unlist(Map(function(x) { hsv(h = x,s = sat_range)},hues))
  
  cols = unlist(sapply(strings,function(x) ld_colors[[x]],simplify=F));
  
  glist = gList(
    rectGrob(
      x = unit(rect_x,'npc'),
      width = unit(diff(ld_cuts_npc),'npc'),
      height = unit(1,'lines'),
      y = unit(rect_y,'lines'),
      just = c("left","bottom"),
      gp = gpar(
        fill = cols,
        col = "black",
        alpha = 1
      ),
      vp = vpPath("legend_outer","legend_ribbons")
    ),
    segmentsGrob(
      x0 = unit(ld_to_npc(ld_cuts),'npc'),
      x1 = unit(ld_to_npc(ld_cuts),'npc'),
      y0 = unit(1.50 * length(strings),'lines'),
      y1 = unit(1.50 * length(strings) + 0.75,'lines'),
      vp = vpPath("legend_outer","legend_ribbons")
    ),
    textGrob(
      ld_cuts,
      x = ld_cuts_npc,
      y = unit(1.50 * length(strings) + 0.75 + 0.75,'lines'),
      vp = vpPath("legend_outer","legend_ribbons")
    ),
    textGrob(
      strings,
      x = unit(0.5,'char'),
      y = unit(seq(1,1.60 * length(strings),1.5),'lines'),
      just = c("left","center"),
      vp = vpPath("legend_outer","legend_strings")
    ),
    textGrob(
      expression(paste(r^2,"with reference SNP")),
      just = c("center","center"),
      vp = vpPath("legend_outer","legend_title")
    )
  );
  
  gTree(
    children = glist,
    childrenvp = tree,
    name = "multiRibbonLegend",
    cl = "gMultiRibbonLegend",
    gp = gp
  );
}

#############################################################
#
# ribbonLegend from RGraphics example
#
ribbonLegend <- function (nlevels = NULL, breaks = NULL, cols, scale = range(breaks), margin = unit(0.5, "lines"), gp = NULL, vp = NULL, name = NULL) 
{
  gTree(
    nlevels = nlevels, 
    breaks = breaks, 
    cols = cols, 
    scale = scale, 
    children = ribbonKids(nlevels, breaks, cols, scale), 
    childrenvp = ribbonVps(nlevels, breaks, margin, scale), 
    gp = gp, 
    vp = vp, 
    name = name, 
    cl = "ribbonLegend"
  );
}

widthDetails.ribbonLegend <- function (x) 
{
    sum(layout.widths(viewport.layout(x$childrenvp[[1]])))
}

calcBreaks <- function (nlevels, breaks, scale) 
{
    if (is.null(breaks)) {
        seq(min(scale), max(scale), diff(scale)/nlevels)
    }
    else {
        breaks
    }
}

ribbonVps <- function (nlevels, breaks, margin, scale) {
  breaks <- format(signif(calcBreaks(nlevels, breaks, scale), 3))
  vpTree(
    # Parent viewport
    viewport(
      name = "layout", 
      layout = grid.layout(
        3, 
        4, 
        widths = unit.c(margin, unit(1, "lines"), max(unit(0.8,"lines") + stringWidth(breaks)), margin),
        heights = unit.c(margin, unit(1, "null"), margin)
      )
    ), 
    # Child viewports
    vpList(
      viewport(
        layout.pos.col = 2, 
        layout.pos.row = 2, 
        yscale = scale, 
        name = "ribbon"
      ), 
      viewport(
        layout.pos.col = 3, 
        layout.pos.row = 2, 
        yscale = scale, 
        name = "labels"
      )
    )
  )   
}

ribbonKids <- function (nlevels, breaks, cols, scale) {
  breaks <- calcBreaks(nlevels, breaks, scale);
  nb <- length(breaks);
  tickloc <- breaks[-c(1, nb)];
  
  gList(
    rectGrob(
      y = unit(breaks[-1], "native"), 
      height = unit(diff(breaks), "native"), 
      just = "top", 
      gp = gpar(fill = cols), 
      vp = vpPath(
        "layout", 
        "ribbon"
      )  
    ), 
    segmentsGrob(
      x1 = unit(0.5, "lines"), 
      y0 = unit(tickloc, "native"), 
      y1 = unit(tickloc, "native"), 
      vp = vpPath("layout", "labels")
    ), 
    textGrob(
      x = unit(0.8, "lines"), 
      y = unit(tickloc, "native"), 
      just = "left", 
      label = format(signif(tickloc, 3)), 
      vp = vpPath("layout", "labels")
    )
  );
}


#############################################################
#
# make a "list" of genes in flat.  returns a data frame
#
make.gene.list <-  function (flat, showIso=TRUE, subset, unit, ...) 

{       
    if ( prod(dim(flat)) <= 0 ) { return(NULL); }

    df <- flat;

    if (!missing(subset)) { df <- df[subset, ] }

    if ( prod(dim(flat)) <= 0 ) { return(NULL); }

    if (args[['showIso']]) {
      df$idnum <- match(df$nmName,unique(df$nmName));
    } else {
      df$idnum <- match(df$name,unique(df$name));
    }

    df0 <- df[df$type == 0, ];
    df1 <- df[df$type == 1, ];
    df2 <- df[df$type == 2, ];

    if ( "col" %in% names(df0) ) {
        col = df0$col
            fill = df0$col
    }

    return( data.frame(id=df0$idnum, gene=df0$name, chrom=df0$chrom, start=df0$start, stop=df0$stop,
        startbp=df0$start * unit, stopbp=df0$stop*unit ) );
}

#############################################################
#
# display genes taking data from flattened bed format
#
panel.flatbed <- function (
  x=NULL, 
  y=NULL, 
  flat, 
  fill = "navy", 
  col = "navy", 
  alpha = 1, 
  textcol='black',
  multiplier = 0.001, 
  height = 2/14, 
  buffer=0.003, 
  subset, 
  cex=.9, 
  rows=2,
  showPartialGenes=FALSE,
  shiftGeneNames=TRUE,
  computeOptimalRows=FALSE, ...) 
{      
  if ( prod(dim(flat)) <= 0 ) { return(1); }

  df <- flat;

  if (!missing(subset)) { df <- df[subset, ] }

  df$width <- df$stop - df$start;

  if (args[['showIso']]) {
    df$idnum <- match(df$nmName,unique(df$nmName));
  } else {
    df$idnum <- match(df$name,unique(df$name));
  }

  df0 <- df[df$type == 0, ];
  df1 <- df[df$type == 1, ];
  df2 <- df[df$type == 2, ];  # unused?

  if ( "col" %in% names(df0) ) {
    col = df0$col
    fill = df0$col
  }

  # removed duplicate idnums from df0
  df0 <- df0[order(df0$idnum),]     # sort to make sure repeated ids are adjacent
  df0$new <- c(1,diff(df0$idnum))  # identify new (1) vs. repeated (0)
  df0 <- df0[order(df0$idnum),]    # put back into original order
  df0uniq <- df0[df0$new == 1,]
  
  # Helper function. For each item, return the indexes for each 
  # match in the target set. Returns only those indexes that match overall, 
  # not an exact set of matches per item. 
  match_each = function(items,target) { 
    unique(unlist(sapply(items,function(x) { which(target == x) })))
  } 
  
  # If we have a required gene, or isoform, let's make sure
  # they are laid out first.
  reqGene = args[['requiredGene']]
  reqIDList = NULL
  if (!is.null(reqGene)) {
    req_rows = NULL
    
    # Some normalization of ENSEMBL IDs
    reqGene = gsub("(ENS.*)(\\.\\d+$)","\\1",reqGene)
    df0uniq$nmName = gsub("(ENS.*)(\\.\\d+$)","\\1",df0uniq$nmName)
    
    # Try matching in the nmName column first, it could be an isoform. 
    iso_matches = match_each(reqGene,df0uniq$nmName);
    if (length(iso_matches) > 0) {
      reqIDList = df0uniq$idnum[iso_matches]
      iso_gene = unique(df0uniq[iso_matches,]$name)
      gene_matches = seq(dim(df0uniq)[1])[df0uniq$name %in% iso_gene]

      all_matches = union(iso_matches,gene_matches)
      req_rows = c(req_rows,all_matches)
    } else {    
      # See if any of them were also matching genes. 
      # Remember that name is still just gene names at this point. Isoform names are added later. 
      gene_matches = match_each(reqGene,df0uniq$name)
      if (length(gene_matches) > 0) {
        reqIDList = df0uniq$idnum[gene_matches]
        req_rows = gene_matches
      }
    }

    req_rows = unique(req_rows)
    
    if (!is.null(req_rows)) {
      other_rows = setdiff(seq(dim(df0uniq)[1]),req_rows)
      
      # Reorder to have the required gene/isoform rows first, and then the rest. 
      df0uniq = df0uniq[c(req_rows,other_rows),]
    }

  }
    
  # determine the row to use
  maxIdnum <- max(c(0,df$idnum))
  id2row <- rep(0,1+maxIdnum)      # keep track of locations for each gene
  
  # if we're showing isoforms, change the names
  if (args[['showIso']]) {
    df0uniq$name = sprintf("%s (%s)",df0uniq$name,df0uniq$nmName)
  }

  # conversion to 'native' isn't working, so we convert evertyhing via inches to npc below. 
  # conversion utility

  native2npc <- function(x) {
    w <- diff(current.viewport()$xscale)
    a <- current.viewport()$xscale[1]
    return( (x-a) / w )
  }
  
  df0uniq$rowToUse = -Inf
  rowIntervals = lapply(seq(1+maxIdnum),function(x) { list(left=NULL,right=NULL) } )
  for (i in 1:dim(df0uniq)[1]) {
    leftGraphic <- native2npc(min(df$start[df$idnum == df0uniq$idnum[i]]))
    rightGraphic <- native2npc(max(df$stop[df$idnum == df0uniq$idnum[i]]))
    centerGraphic<- mean(c(leftGraphic,rightGraphic))
    at <- arrowText(df0uniq$name[i],
      x = unit((df0uniq$start[i] + df0uniq$stop[i])/2, 'native'),
      y = unit(0,'npc'), 
      direction = df0uniq$strand[i],
      check.overlap = TRUE, 
      gp = gpar(cex = cex, fontface='italic',col=textcol,lwd=1.5),
      widthOnly=FALSE
    );
    w <- 1.1 * convertX(attr(at,'width'),'inches',valueOnly=TRUE)
    viewportWidth <- convertX(unit(1,'npc'),'inches',valueOnly=TRUE);
    w <- w / viewportWidth
    leftName <- centerGraphic - .5 * w
    rightName <- centerGraphic + .5 * w

    if (shiftGeneNames) {
      if (leftName < 0) {
        leftName <- 0; rightName <- w
      }
      if (rightName > 1) {
        rightName <- 1; leftName <- 1-w
      }
    }

    left <- min(c(leftGraphic,leftName)) - buffer 
    right <- max(c(rightGraphic,rightName)) + buffer 

    df0uniq$start[i] <- leftName
    df0uniq$stop[i] <- rightName
    df0uniq$left[i] <- left
    df0uniq$right[i] <- right

    if (!showPartialGenes & (left < 0 | right > 1)) {
      df0uniq[i,"rowToUse"] = -2;
      next;
    }
    
    # Check each row to see if this gene can fit. 
    # This would be *much* faster with IRanges, but we can't require the user
    # to install it, unfortunately. So this slow for loop will have to do. 
    # If there's ever a LZ R package, then we could replace this easily. 
    for (irow in seq(rowIntervals)) {
      intervals = rowIntervals[[irow]]
      
      done = FALSE
      if (is.null(intervals$left)) {
        # This row is completely empty. We can use it. 
        intervals$left = left
        intervals$right = right
        df0uniq[i,"rowToUse"] = irow
        
        done = TRUE
      } else {
        if (!any((intervals$left < right) & (intervals$right > left))) {
          # This interval doesn't overlap anything in the current row. 
          intervals$left = c(intervals$left,left)
          intervals$right = c(intervals$right,right)
          df0uniq[i,"rowToUse"] = irow
          
          done = TRUE
        }
      }
      
      rowIntervals[[irow]] = intervals
      
      if (done) { break; }
    }
  }
      
  id2row = df0uniq[order(df0uniq$idnum),]$rowToUse

  requestedRows = rows;
  optRows = max(df0uniq$rowToUse)
  if (computeOptimalRows) { return (optRows) }

  save(df,flat,df0,df1,df2,df0uniq,id2row,rowIntervals,file="debug.Rdata");

  rows <- min(requestedRows,optRows);

  if (optRows > requestedRows && as.logical(args[['warnMissingGenes']])) {
    omitIdx <- which(id2row > rows)
    assign("omittedGenes",as.character(df0uniq$name[omitIdx]),globalenv())
    numberOfMissingGenes <- length(omittedGenes);
    message <- paste(numberOfMissingGenes,if(args[['showIso']]) " iso" else " gene",if(numberOfMissingGenes > 1) "s" else "", "\nomitted",sep="")
    pushViewport(viewport(clip='off'));
    grid.text(message ,x=unit(1,'npc') + unit(1,'lines'), y=.5, just=c('left','center'),
      gp=gpar(cex=args[['axisTextSize']], col=args[['axisTextColor']], alpha=args[['frameAlpha']])
    );
    upViewport(1);
  }

  increment <- 1.0/rows;

  yPos <- function(id,text=FALSE) {
    if (text) {
      return( unit((rows-id2row[id]) * increment + 4.4/7*increment, "npc") )
    } else {
      return( unit( (rows-id2row[id]) * increment + 2/7*increment, "npc") )
    }
  }

  grid.segments(
    x0 = multiplier + df0$start, 
    x1 = df0$stop, 
    y0 = yPos(df0$idnum),
    y1 = yPos(df0$idnum),
    default.units = "native", 
    gp = gpar(col = col, alpha = alpha)
  );

  if ( "col" %in% names(df1) ) {
    col = df1$col
    fill = df1$col
  }

  if (dim(df1)[1] > 0) {
    grid.rect(
      x = multiplier + df1$start, 
      width = df1$width, 
      just = "left", 
      y = yPos(df1$idnum),
      height = unit(height * increment, "npc"), 
      default.units = "native", 
      gp = gpar(fill = fill, col = col, alpha = alpha)
    );
  }

  if ( "textcol" %in% names(df0uniq) ) {
    textcol = df0uniq$textcol
    fill = df0uniq$textcol
  }

  for (i in 1:dim(df0uniq)[1]) {
    at <- arrowText(
      df0uniq$name[i],
      x = unit((df0uniq$start[i] + df0uniq$stop[i])/2, 'npc'),
      y = yPos(df0uniq$idnum[i], text=TRUE),
      direction = df0uniq$strand[i],
      check.overlap = TRUE, 
      gp = gpar(cex = cex, fontface='italic',col=textcol,lwd=1.5)
    );
    grid.draw(at);
  }

  add_npc = function(...) {
    v = sapply(list(...),function(x) as.numeric(convertUnit(x,"npc")))
    return(unit(sum(v),"npc"))
  }

  midpoint_npc = function(x,y) {
    x = as.numeric(x)
    y = as.numeric(y)
    return(unit((x + y)/2,"npc"))
  }
  
  # This code highlights the required gene with a rectangle around it
  if (args[['hiRequiredGene']] & !is.null(args[['requiredGene']]) & length(reqIDList) > 0) {
    for (requiredGeneIdnum in reqIDList) {
      requiredGeneIdx = which(df0uniq$idnum == requiredGeneIdnum)
      
      # Need the arrow grob to know how wide it is. 
      at <- arrowText(
        df0uniq$name[requiredGeneIdx],
        x = unit((df0uniq$start[requiredGeneIdx] + df0uniq$stop[requiredGeneIdx])/2, 'npc'),
        y = yPos(df0uniq$idnum[requiredGeneIdx], text=TRUE),
        direction = df0uniq$strand[requiredGeneIdx],
        check.overlap = TRUE, 
        gp = gpar(cex = cex, fontface='italic',col=textcol,lwd=1.5)
      );

      npc_width_gene_text = convertUnit(attr(at,"width"),"npc")
      npc_width_gene_body = unit(diff(as.numeric(df0uniq[requiredGeneIdx,c("left","right")])),"npc")
      
      total_height = unit(3.5 * height * increment,"npc") + attr(at,"theight")

      y_center = midpoint_npc(
        yPos(df0uniq$idnum[requiredGeneIdx],text=TRUE),
        yPos(df0uniq$idnum[requiredGeneIdx],text=FALSE)
      )

      grid.rect(
        x = unit((df0uniq$start[requiredGeneIdx] + df0uniq$stop[requiredGeneIdx])/2, 'npc'),
        y = y_center,
        width = 1.025 * max(npc_width_gene_text,npc_width_gene_body),
        height = total_height,
        gp = gpar(col=args[['hiRequiredGeneColor']])
      )

    }
  }
  
  #sink(NULL);
  #sink(NULL,type="message");
  #sink(NULL);
  #sink(NULL,type="message");
  #browser();
}

panel.finemap <- function (
  fmdata, 
  fill = "navy", 
  col = "navy", 
  alpha = 1, 
  textcol='black',
  multiplier = 0.001, 
  height = 3/14, 
  buffer=0.003, 
  cex=.8, 
  rows=2,
  showPartial=FALSE,
  shiftNames=TRUE,
  computeOptimalRows=FALSE, ...) 
{ 
  if (is.null(fmdata)) {
    return(0);
  }     

  if ( prod(dim(fmdata)) <= 0 ) { 
    return(0); 
  }

  # Convert into regions for computing width of intervals 
  # and for name placement. 
  regions = NULL;
  for (fmgroup in unique(fmdata$group)) {
    chunk = subset(fmdata,group == fmgroup);
    chunk_start = min(chunk$pos);
    chunk_stop = max(chunk$pos);
    regions = rbind(regions,data.frame(
      name = fmgroup,
      start = chunk_start,
      stop = chunk_stop
    ));
  }
  
  df <- regions;

  df$width <- df$stop - df$start;

  df$idnum <- match(df$name,unique(df$name));
  
  df0 = df;

  # removed duplicate idnums from df0
  df0 <- df0[order(df0$idnum),]     # sort to make sure repeated ids are adjacent
  df0$new <- c(1,diff(df0$idnum))  # identify new (1) vs. repeated (0)
  df0 <- df0[order(df0$idnum),]    # put back into original order
  df0uniq <- df0[df0$new == 1,]

  # determine the row to use
  maxIdnum <- max(c(0,df$idnum))
  rowUse <- rep(-Inf,1+maxIdnum)
  id2row <- rep(0,1+maxIdnum)      # keep track of locations for each gene
  
  # conversion to 'native' isn't working, so we convert evertyhing via inches to npc below. 
  # conversion utility

  native2npc <- function(x) {
    w <- diff(current.viewport()$xscale)
    a <- current.viewport()$xscale[1]
    return( (x-a) / w )
  }

  for (i in 1:dim(df0uniq)[1]) {
    cat(paste(i,": ",df0uniq$name[i],"\n"));

    leftGraphic <- native2npc(min(df$start[df$idnum == df0uniq$idnum[i]]))
    rightGraphic <- native2npc(max(df$stop[df$idnum == df0uniq$idnum[i]]))
    centerGraphic<- mean(c(leftGraphic,rightGraphic))

    at <- textGrob(
      df0uniq$name[i],
      x = unit((df0uniq$start[i] + df0uniq$stop[i])/2, 'native'),
      y = unit(0,'npc'), 
      check.overlap = TRUE, 
      gp = gpar(cex = cex, fontface='italic',col=textcol,lwd=1.5),
    );

    w <- 1.1 * convertX(widthDetails(at),'inches',valueOnly=TRUE)
    viewportWidth <- convertX(unit(1,'npc'),'inches',valueOnly=TRUE);
    w <- w / viewportWidth
    leftName <- centerGraphic - .5 * w
    rightName <- centerGraphic + .5 * w

    if (shiftNames) {
      if (leftName < 0) {
        leftName <- 0; rightName <- w
      }
      if (rightName > 1) {
        rightName <- 1; leftName <- 1-w
      }
    }

    left <- min(c(leftGraphic,leftName)) - buffer 
    right <- max(c(rightGraphic,rightName)) + buffer 

    df0uniq$start[i] <- leftName
    df0uniq$stop[i] <- rightName
    df0uniq$left[i] <- left
    df0uniq$right[i] <- right

    rowToUse <- min(which(rowUse < left))
    if ( showPartial || (left >= 0 && right <= 1) ) {
      id2row[df0uniq$idnum[i]] <- rowToUse
      rowUse[rowToUse] <- right
    } else {
      id2row[df0uniq$idnum[i]] <- -2   # clipping will hide this
    }
  }

  requestedRows <- rows;
  optRows <- max(c(0,which(rowUse > 0)));
  if (computeOptimalRows) { return (optRows) }

  save(df,regions,df0,df0uniq,id2row,rowUse,file="debug_finemap.Rdata");

  rows <- min(requestedRows,optRows);

  if (optRows > requestedRows && as.logical(args[['warnMissingFineMap']])) {
    omitIdx <- which(id2row > rows)
    assign("omittedFineMap",as.character(df0uniq$name[omitIdx]),globalenv())
    numberOfMissingGenes <- length(omittedFineMap);
    message <- paste(numberOfMissingGenes," region",if(numberOfMissingGenes > 1) "s" else "", "\nomitted",sep="")
    pushViewport(viewport(clip='off'));
    grid.text(message ,x=unit(1,'npc') + unit(1,'lines'), y=.5, just=c('left','center'),
      gp=gpar(cex=args[['axisTextSize']], col=args[['axisTextColor']], alpha=args[['frameAlpha']])
    );
    upViewport(1);
  }

  increment <- 1.0/rows;

  yPos <- function(id,text=FALSE) {
    if (text) {
      return( unit((rows-id2row[id]) * increment + 4.4/7*increment, "npc") )
    } else {
      return( unit( (rows-id2row[id]) * increment + 2/7*increment, "npc") )
    }
  }
  
  # Figure out which row each group/points are supposed to be on..
  fmdata$idnum = df0$idnum[match(fmdata$group,df0$name)];
    
  grid.polyline(
    x = fmdata$pos,
    y = yPos(fmdata$idnum,text=FALSE),
    default.units = "native",
    gp = gpar(col = unique(fmdata$color)),
    id = fmdata$idnum
  );
  
  grid.points(
    x = fmdata$pos,
    y = yPos(fmdata$idnum,text=FALSE),
    default.units = "native",
    pch = 20,
    gp = gpar(col = fmdata$color, cex = 0.8)
  );
  
  if ( "textcol" %in% names(df0uniq) ) {
    textcol = df0uniq$textcol
    fill = df0uniq$textcol
  }

  for (i in 1:dim(df0uniq)[1]) {
    at <- textGrob(
      df0uniq$name[i],
      x = unit((df0uniq$start[i] + df0uniq$stop[i])/2, 'npc'),
      y = yPos(df0uniq$idnum[i], text=TRUE),
      check.overlap = TRUE, 
      gp = gpar(cex = cex,col=textcol,lwd=1.5)
    );
    grid.draw(at);
  }

  # sink(NULL);
  # sink(NULL,type="message");
  # sink(NULL);
  # sink(NULL,type="message");
  # browser();
  
}

# bed_tracks is a data frame with: chr, start, end, name, score, strand, thickStart, thickEnd, itemRgb
panel.bed <- function(bed_data,track_height,startbp,endbp) {
  # compute colors
  if ("itemRgb" %in% names(bed_data)) {
    bed_data$color = sapply(bed_data$itemRgb,function(x) { do.call(rgb,c(as.list(unlist(strsplit(x,","))),maxColorValue=255)) });
    
    is_white = bed_data$color == "#FFFFFF";
    if (any(is_white)) {
      bed_data[is_white,]$color = "#E5E5E5";
    }
  } else {
    bed_data$color = "black";
  }  
  
  startbp = startbp / 1E6;
  endbp = endbp / 1E6;
    
  # Chop off edges that go off the plot. 
  bed_data$start = sapply(bed_data$start,function(x) max(x,startbp));
  bed_data$end = sapply(bed_data$end,function(x) min(x,endbp));
  
  # plot each sub-track separately
  sub_tracks = rev(unique(bed_data$name))
  i = 0;
  for (sub_name in sub_tracks) {
    cat("Plotting track ",sub_name,"\n")
    bed_sub = subset(bed_data,name == sub_name);
  
    # Combine regions that overlap to avoid overplotting
    if (dim(bed_sub)[1] > 1) {
      # Only need to do this check when there's more than 1 row :) 
      
      bed_sub = bed_sub[order(bed_sub$start),];
      for (j in (2:(dim(bed_sub)[1]))) {
        if (bed_sub$start[j] <= bed_sub$end[j-1]) {
          bed_sub$start[j] = bed_sub$start[j-1];
          bed_sub$start[j-1] = NA;
        }
      }
      bed_sub = subset(bed_sub,!is.na(bed_sub$start));
    }
    
    bed_sub$width = abs(bed_sub$end - bed_sub$start);
  
    y_mid = (0.5 * track_height) + (i * track_height);
    
    #bed_sub$y0 = y_mid - (0.25 * track_height);
    #bed_sub$y1 = y_mid + (0.25 * track_height);
        
    grid.rect(
      x = unit(bed_sub$start,'native'),
      y = unit(y_mid,'lines'),
      width = unit(bed_sub$width,'native'),
      height = unit(0.75 * track_height,'lines'),
      just = c("left","center"),
      gp = gpar(
        fill = bed_sub$color,
        col = bed_sub$color
      ),
      default.units = 'native'
    );

    # grid.lines(
      # x = unit(c(startbp,endbp),'native'),
      # y = unit(rep(y_mid,2),'lines')
    # );
    
    grid.text(
      label = sub_name,
      x = unit(1,'npc') + unit(0.5,'char'),
      y = unit(y_mid,'lines'),
      just = c("left","center"),
      gp = gpar(
        cex = 1
      )
    );

    i = i + 1;
  }

}

panel.gwas <- function (
  gwas_hits, 
  fill = "navy", 
  col = "navy", 
  alpha = 1, 
  textcol='black',
  multiplier = 0.001, 
  height = 3/14, 
  buffer=0.003, 
  subset, 
  cex=.9, 
  rows=2,
  showPartial=FALSE,
  shiftNames=TRUE,
  computeOptimalRows=FALSE, ...) 
{ 
  if (is.null(gwas_hits)) {
    return(0);
  }     

  if ( prod(dim(gwas_hits)) <= 0 ) { 
    return(0); 
  }

  df <- gwas_hits;
  df$start = df$pos;
  df$stop = df$pos; 
  df$name = sprintf("chr%s:%s-%s",df$chr,df$pos,df$trait);

  if (!missing(subset)) { df <- df[subset, ] }

  df$width <- df$stop - df$start;

  df$idnum <- match(df$name,unique(df$name));
  df0 = df;

  # removed duplicate idnums from df0
  df0 <- df0[order(df0$idnum),]     # sort to make sure repeated ids are adjacent
  df0$new <- c(1,diff(df0$idnum))  # identify new (1) vs. repeated (0)
  df0 <- df0[order(df0$idnum),]    # put back into original order
  df0uniq <- df0[df0$new == 1,]

  # determine the row to use
  maxIdnum <- max(c(0,df$idnum))
  rowUse <- rep(-Inf,1+maxIdnum)
  id2row <- rep(0,1+maxIdnum)      # keep track of locations for each gene
  
  # conversion to 'native' isn't working, so we convert evertyhing via inches to npc below. 
  # conversion utility

  native2npc <- function(x) {
    w <- diff(current.viewport()$xscale)
    a <- current.viewport()$xscale[1]
    return( (x-a) / w )
  }

  for (i in 1:dim(df0uniq)[1]) {
    cat(paste(i,": ",df0uniq$name[i],"\n"));

    at <- textGrob(
      df0uniq$trait[i],
      x = unit(df0uniq$pos[i], 'native'),
      y = unit(0,'npc'), 
      check.overlap = TRUE, 
      gp = gpar(cex = cex,col=textcol,lwd=1.5),
    );

    w <- 1.1 * convertX(widthDetails(at),'inches',valueOnly=TRUE)
    viewportWidth <- convertX(unit(1,'npc'),'inches',valueOnly=TRUE);
    w <- w / viewportWidth

    leftName <- native2npc(df0uniq$pos[i]) - .5 * w
    rightName <- native2npc(df0uniq$pos[i]) + .5 * w

    if (shiftNames) {
      if (leftName < 0) {
        leftName <- 0; 
        rightName <- w
      }
      if (rightName > 1) {
        rightName <- 1; 
        leftName <- 1-w
      }
    }

    left <- leftName - buffer 
    right <- rightName + buffer 

    df0uniq$start[i] <- leftName
    df0uniq$stop[i] <- rightName
    df0uniq$left[i] <- left
    df0uniq$right[i] <- right

    rowToUse <- min(which(rowUse < left))
    if ( showPartial || (left >= 0 && right <= 1) ) {
      id2row[df0uniq$idnum[i]] <- rowToUse
      rowUse[rowToUse] <- right
    } else {
      id2row[df0uniq$idnum[i]] <- -2   # clipping will hide this
    }
  }

  requestedRows <- rows;
  optRows <- max(c(0,which(rowUse > 0)));
  if (computeOptimalRows) { return (optRows) }

  save(df,gwas_hits,df0,df0uniq,id2row,rowUse,file="debug_gwas.Rdata");

  rows <- min(requestedRows,optRows);

  if (optRows > requestedRows && as.logical(args[['warnMissingGWAS']])) {
    # Get rid of rows that we can't fit on the plot..
    omitIdx <- which(id2row > rows)
    df0uniq = df0uniq[!df0uniq$idnum %in% omitIdx,]
    
    assign("omittedGWAS",as.character(df0uniq$name[omitIdx]),globalenv())
    numberOfMissingGenes <- length(omittedGWAS);
    message <- paste(numberOfMissingGenes," GWAS hit",if(numberOfMissingGenes > 1) "s" else "", "\nomitted",sep="")
    pushViewport(viewport(clip='off'));
    grid.text(message ,x=unit(1,'npc') + unit(1,'lines'), y=.5, just=c('left','center'),
      gp=gpar(cex=0.75*args[['axisTextSize']], col=args[['axisTextColor']], alpha=args[['frameAlpha']])
    );
    upViewport(1);
  }
  
  increment <- 1.0/rows;
  
  yPos <- function(id,text=FALSE) {
    if (text) {
      return( unit( (rows-id2row[id]) * increment + 2/7*increment, "npc") )
    } else {
      return( unit((rows-id2row[id]) * increment + 4.4/7*increment, "npc") )
    }
  }

  df0uniq_rows = dim(df0uniq)[1];
  poly_y = sapply(df0uniq$idnum,yPos,text=FALSE);
  poly_y = as.vector(rbind(
    poly_y,
    rep(0.98,df0uniq_rows))
  ); # interlaces two vectors together

  grid.polyline(
    x = unit(unlist(Map(rep,df0uniq$pos,2)),'native'),
    y = poly_y,
    id = ceiling(1:(2*df0uniq_rows)/2), # 1 1 2 2 3 3 ... to represent which lines the x/y belong to
    default.units = 'npc',
    arrow = arrow(
      angle = 45,
      length = unit(0.05,'inches'),
      ends = 'last',
      type = 'open'
    ),
    gp = gpar(
      lwd = 1
    )
  )

  if ( "textcol" %in% names(df0uniq) ) {
    textcol = df0uniq$textcol
    fill = df0uniq$textcol
  }

  for (i in 1:dim(df0uniq)[1]) {
    at <- textGrob(
      df0uniq$trait[i],
      x = unit(df0uniq$pos[i], 'native'),
      y = yPos(df0uniq$idnum[i], text=TRUE),
      check.overlap = TRUE, 
      gp = gpar(cex = cex,col=textcol,lwd=1.5)
    );
    grid.draw(at);
  }

  # sink(NULL);
  # sink(NULL,type="message");
  # sink(NULL);
  # sink(NULL,type="message");
  # browser();
}

panel.barplot = function(dframe,xcol="pos",ycol="value") {
  if (!is.null(args[['barplotMinY']])) {
    min_val = args[['barplotMinY']]
  } else {
    min_val = min(dframe[,ycol])
  }
  
  if (!is.null(args[['barplotMaxY']])) {
    max_val = args[['barplotMaxY']]
  } else {
    max_val = max(dframe[,ycol])
  }

  #fudge = 0.001 * (max_val - min_val)

  grid.segments(
    x0 = unit(dframe[,xcol],"native"),
    y0 = unit(min_val,"native"),
    x1 = unit(dframe[,xcol],"native"),
    y1 = unit(dframe[,ycol],"native"),
  )

}

#############################################################
#
# Assemble a plot zooming in on a region from various pieces
# including metal output (with positions added), ld (ala newfugue), recombination rate data,
# genes data (refFlat), etc.
# 
# NB: *** passing in entire args list *** 
#
zplot <- function(metal,ld=NULL,recrate=NULL,refidx=NULL,nrugs=0,postlude=NULL,args=NULL,...) {

  refSnp <- metal$MarkerName[refidx];

  metal$P.value <- as.numeric(metal$P.value);
  
  dotSizes <- rep(args[['largeDot']],dim(metal)[1]);
  
  if (!is.null(args[['refDot']]) ) {
    dotSizes[refidx] <- args[['refDot']];
  } 

  if (char2Rname(args[['weightCol']]) %in% names(metal)){
    metal$Weight <- metal[,char2Rname(args[['weightCol']])];
   
    # Weights must vary in order to scale point sizes by them. 
    if (diff(range(metal$Weight)) > 0) {
      dotSizes <- rescale(
        metal$Weight,
        oldScale = args[['weightRange']], 
        newScale = c(args[['smallDot']],args[['largeDot']]), 
        transformation=sqrt
      ); 
    }
  }

  metal$dotSizes = dotSizes; 
  
  if ( is.null(args[['refDot']]) ) {
     # this avoids problems downstream, but dotSize[refidx] has already been set in most cases.
     args[['refDot']] <- args[['largeDot']];
  }

  titlev = args[['title']];
  extitle = args[['expr_title']];

  b_title = (titlev != '') & (!is.null(titlev));
  b_expr_title = (extitle != '') & (!is.null(extitle));
  title_lines = ifelse(b_title | b_expr_title,3,0);

  grid.newpage();

  # Right column width is either fixed (5 lines) or extended based on the names
  # next to the BED tracks
  # if (!is.null(bed_tracks)) {
    # bed_names = unique(bed_tracks$name);
    # longest_bed_name = bed_names[which.max(nchar(bed_names))];
    # longest_bed_name_width = convertUnit(unit(1,'strwidth',longest_bed_name),'lines');
    
    # right_col_width = max(5,as.numeric(longest_bed_name_width));
  # } else {
    # right_col_width = 5;
  # }

  draw_bed = !is.null(bed_tracks)
  draw_gwas = !is.null(gwas_hits)
  draw_barplot = !is.null(barplot_data)
  draw_finemap = !is.null(fmregions)

  # Number of lines to use for drawing bed tracks 
  if (draw_bed) { 
    bed_lines = length(unique(bed_tracks$name));
  } else {
    bed_lines = 0
  }
  bed_height = 1; # lines

  # Number of lines to use for barplot
  # If no data specified, we don't want it to use any lines
  if (!draw_barplot) {
    args[["barplotRows"]] = 0
  }

  sep_size = 0.5
  
  # push viewports just to calculate optimal number of rows for refFlat
  pushViewport(viewport(
    layout = grid.layout(2+3+4+1+1+1+1+1+1+1+1,1+2, 
      widths = unit(c(5,1,5),c('lines','null','lines')),
      heights = unit(
        c(
          .5,
          title_lines,
          nrugs,
          1,
          1,
          sep_size,
          2*args[['barplotRows']],
          ifelse(draw_barplot,sep_size,0),
          bed_height*bed_lines,
          ifelse(draw_bed,sep_size,0),
          1.8*args[['gwrows']],
          ifelse(draw_gwas,sep_size,0),
          2*args[['fmrows']],
          ifelse(draw_finemap,sep_size,0),
          2*args[['rfrows']],
          4,
          .25
        ),
        c(
          'lines', # 0.5      1
          'lines', # title    2
          'lines', # nrugs    3
          'lines', # 1        4
          'null',  # 1        5
          'lines', # 0.25     6
          'lines', # barplot  7
          'lines',
          'lines',
          'lines',
          'lines',
          'lines',
          'lines',
          'lines',
          'lines',
          'lines',
          'lines'
        )
      )
    )
  ));

  pvalVp = dataViewport(
    xRange,yRange,
    extension=c(0,.05),
    layout.pos.row=5,
    layout.pos.col=2,
    name="pvals",
    clip="off"
  );

  pushViewport(viewport(
    xscale=pvalVp$xscale,
    layout.pos.row=15,
    layout.pos.col=2,
    name="refFlatOuter"
  ));
    
  optRows <- panel.flatbed(
    flat=refFlat,
    rows=NULL,
    computeOptimalRows=TRUE,
    showPartialGenes = args[['showPartialGenes']],
    shiftGeneNames = args[['shiftGeneNames']],
    cex=args[['geneFontSize']],
    col=args[['geneColor']],
    fill=args[['geneColor']],
    multiplier=1/args[['unit']]
  );

  if ( length( args[['rfrows']] < 2 ) ) {       # use value as upper bound
    args[['rfrows']] <- min(args[['rfrows']], optRows)
  } else {  # use smallest two values as lower and upper bounds
    args[['rfrows']] <- sort(args[['rfrows']])
    rows <- min( args[['rfrows']][2], optRows )
    args[['rfrows']] <- max( args[['rfrows']][1], rows )
  }

  pushViewport(viewport(
    xscale=pvalVp$xscale,
    layout.pos.row=13,
    layout.pos.col=2,
    name="finemapOuter"
  ));

  optRowsFinemap = panel.finemap(
    fmregions,
    rows=NULL,
    computeOptimalRows=TRUE,
    showPartial = args[['showPartialGenes']],
    shiftNames = args[['shiftGeneNames']],
    cex=args[['geneFontSize']],
    col=args[['geneColor']],
    fill=args[['geneColor']],
    multiplier=1/args[['unit']]
  );

  if ( length( args[['fmrows']] < 2 ) ) {       # use value as upper bound
    args[['fmrows']] <- min(args[['fmrows']], optRowsFinemap)
  } else {  # use smallest two values as lower and upper bounds
    args[['fmrows']] <- sort(args[['fmrows']])
    rows <- min( args[['fmrows']][2], optRowsFinemap )
    args[['fmrows']] <- max( args[['fmrows']][1], rows )
  }

  pushViewport(viewport(
    xscale=pvalVp$xscale,
    layout.pos.row=11,
    layout.pos.col=2,
    name="gwasOuter"
  ));

  optRowsGWAS = panel.gwas(
    gwas_hits,
    rows=NULL,
    computeOptimalRows=TRUE,
    showPartial = args[['showPartialGenes']],
    shiftNames = args[['shiftGeneNames']],
    cex=args[['geneFontSize']],
    col=args[['geneColor']],
    fill=args[['geneColor']],
    multiplier=1/args[['unit']]
  );

  if ( length( args[['gwrows']] < 2 ) ) {       # use value as upper bound
    args[['gwrows']] <- min(args[['gwrows']], optRowsGWAS)
  } else {  # use smallest two values as lower and upper bounds
    args[['gwrows']] <- sort(args[['gwrows']])
    rows <- min( args[['gwrows']][2], optRowsGWAS )
    args[['gwrows']] <- max( args[['gwrows']][1], rows )
  }

  pushViewport(viewport(
    xscale=pvalVp$xscale,
    layout.pos.row=7,
    layout.pos.col=2,
    name="barplot"
  ));

  #optRowsBarplot = ifelse(draw_barplot,99,0)

  #if ( length( args[['barplotRows']] < 2 ) ) {       # use value as upper bound
    #args[['barplotRows']] <- min(args[['barplotRows']], optRowsBarplot)
  #} else {  # use smallest two values as lower and upper bounds
    #args[['barplotRows']] <- sort(args[['barplotRows']])
    #rows <- min( args[['barplotRows']][2], optRowsBarplot )
    #args[['barplotRows']] <- max( args[['barplotRows']][1], rows )
  #}
    
  popViewport(5);

  # OK.  Now we know how many rows to use and we can set up the layout we will actually use.

  pushViewport(viewport(
    layout = grid.layout(
      2+3+4+1+1+1+1+1+1+1+1,1+2, 
      widths = unit(
        c(args[['axisTextSize']]*args[['leftMarginLines']],1,args[['axisTextSize']]*args[['rightMarginLines']]),
        c('lines','null','lines')
      ),
      heights = unit(
        c(
          .5,
          title_lines,
          nrugs,
          1,
          1,
          sep_size,
          2*args[['geneFontSize']]*args[['barplotRows']],
          ifelse(draw_barplot,sep_size,0),
          bed_height*bed_lines,
          ifelse(draw_bed,sep_size,0),
          1.5*args[['geneFontSize']]*args[['gwrows']],
          ifelse(draw_gwas,sep_size,0),
          2*args[['geneFontSize']]*args[['fmrows']],
          ifelse(draw_finemap,sep_size,0),
          2*args[['geneFontSize']]*args[['rfrows']],
          4,
          .25
        ),
        c(
          'lines', # 0.5
          'lines', # title
          'lines', # nrugs
          'lines', # 1
          'null',  # 1 
          'lines', # 0.25
          'lines', # barplot
          'lines', # sep
          'lines', # bed plot
          'lines', # sep
          'lines', # gwas 
          'lines',
          'lines',
          'lines',
          'lines',
          'lines',
          'lines'
        )
        )
      )
    )
  );
  
  ##
  # layout (top to bottom)
  # ----------------------
  #    1  spacer
  #    2  title text
  #    3  rugs
  #    4  separation
  #    5  pvals
  #    6  separation
  #    7  barplot
  #    8  separation
  #    9  bedTracks
  #    10 separation 
  #    11 gwas hits
  #    12 separation
  #    13 fine mapping regions
  #    14 separation
  #    15 genes
  #    16 subtitle text 
  #    17 spacer
  #
  # layout (left to right)
  # ----------------------
  #    vertical axes and labeling
  #    main data panels, titles, horizontal axes, etc.
  #    vertical axes and labeling
  #

  ########## title text
  titleVp = viewport(
    layout.pos.row=2,
    layout.pos.col=2,
    name="title",
    clip="off"
  );
  
  pushViewport(titleVp);
  
  if (b_title) {
    title_esc = gsub("\\\\n","\n",args[['title']])
    grid.text(
      title_esc,
      gp = gpar(
        cex=args[['titleCex']],
        col=args[['titleColor']],
        fontface=args[['titleFontFace']]
      )
    );
  } else if (b_expr_title) {
    grid.text(
      parse_expression(args[['expr_title']]),
      gp = gpar(
        cex=args[['titleCex']],
        col=args[['titleColor']],
        fontface=args[['titleFontFace']]
      )
    );
  }

  upViewport(1);

  ########## pvals
  # this viewport is defined above
  #            pvalVp=dataViewport(
  #         xRange,yRange,
  #                extension=c(0,.05),
  #                layout.pos.row=5,layout.pos.col=2,
  #         name="pvals",
  #                clip="off");
  pushViewport(pvalVp);
  grid.yaxis(at=args[['yat']],gp=gpar(cex=args[['axisSize']],col=args[['frameColor']],alpha=args[['frameAlpha']]));
  if (!is.null(args[['ylab']]) && nchar(args[['ylab']]) > 0) {
    grid.text(x=unit(args[['ylabPos']],'lines'),label=args[['ylab']],rot=90, 
      gp=gpar(cex=args[['axisTextSize']], col=args[['axisTextColor']], alpha=args[['frameAlpha']]) 
    );
  } else {
    grid.text(x=unit(args[['ylabPos']],'lines'),label=expression(paste(-log[10] ,"(p-value)")),rot=90, 
      gp=gpar(cex=args[['axisTextSize']], col=args[['axisTextColor']], alpha=args[['frameAlpha']]) 
    );
  }

  pushViewport(dataViewport(extension=c(0,.05),xRange,recrateRange,name='recrate',clip="off"));
    
  if ( args[['showRecomb']] ) {
    grid.yaxis(main=F,gp=gpar(cex=args[['axisSize']],col=args[['recombAxisColor']],alpha=args[['recombAxisAlpha']]));
    grid.text(x=unit(1,'npc')+unit(args[['recombPos']],'lines'),
      label="Recombination rate (cM/Mb)",rot=270,
      gp=gpar(cex=args[['axisTextSize']],col=args[['recombAxisColor']],alpha=args[['recombAxisAlpha']])
    );
  }

  if ( args[['showRecomb']] && !args[['recombOver']]) {
    pushViewport(dataViewport(extension=c(0,.05),xRange,recrateRange,name='recrateClipped',clip="on"));
              
    if (args[['recombFill']]) {
      grid.polygon(x=recrate$pos,y=recrate$recomb,
        gp=gpar(alpha=args[['recombFillAlpha']],col=args[['recombColor']],fill=args[['recombColor']]),
              default.units='native'
      );
    } else {
      panel.xyplot(recrate$pos,recrate$recomb,type='l',lwd=2,alpha=args[['recombLineAlpha']],col=args[['recombColor']]);
    }
    
    upViewport(1);
  }

  pushViewport(viewport(clip="on",xscale=pvalVp$xscale,yscale=pvalVp$yscale,name='pvalsClipped'));

  grid.rect(gp=gpar(lwd=args[['frameLwd']],col=args[['frameColor']],fill=NA,alpha=args[['frameAlpha']]));
  
  groupIds <-  sort(unique(metal$group))
  print(table(metal$group));

  # Big diamond refsnp?
  if (args[['bigDiamond']] && args[['showRefsnpAnnot']]) {
    grid.points(x=metal$pos[refidx],y=transformation(metal$P.value[refidx]), 
      gp=gpar(col=args[['refsnpColor']],fill=args[['refsnpColor']],cex=1.6*args[['refDot']],alpha=.2),
      pch=23,
      default.units='native'
    );
  }

  barplotThreshold = args[['barplotThreshold']]
  if ((!is.null(barplot_data)) & (!is.null(barplotThreshold))) {
    # Find x-axis points with values that exceed the threshold. 
    barplot_hilites = subset(barplot_data,value > barplotThreshold)
    if (dim(barplot_hilites)[1] > 0) {
      # Figure out which SNPs (points) these positions match up with. 
      metal_match = match(barplot_hilites$pos_int,metal$pos_int)

      # Get the p-values (y axis values)
      barplot_hilites$y = transformation(metal[metal_match,"P.value"])

      # Make a dashed line up for each position still remaining. 
      if (args[['barplotDrawMatchingLine']]) { 
        grid.segments(
          x0 = barplot_hilites$pos,
          y0 = unit(0,'npc'),
          x1 = barplot_hilites$pos,
          y1 = barplot_hilites$y,
          default.units = "native",
          gp = gpar(
            lty = args[['barplotMatchingLineStyle']],
            col = args[['barplotMatchingLineColor']]
          )
        )
      }

      # Wherever the position does not line up perfectly, draw a line all the way to the top
      if (args[['barplotDrawOffLine']]) {
        barplot_missing = barplot_hilites[is.na(barplot_hilites$y),]
        grid.segments(
          x0 = barplot_missing$pos,
          y0 = unit(0,'npc'),
          x1 = barplot_missing$pos,
          y1 = unit(1,'npc'),
          default.units = "native",
          gp = gpar(
            lty = args[['barplotOffLineStyle']],
            col = args[['barplotOffLineColor']]
          )
        )
      }
    }
  }

  if (is.null(args[['cond_ld']])) {
    # Draw SNPs with LD colors based on reference SNP LD. 
    for (i in groupIds) { 
      idx <- which(metal$group == i);
      gmetal <- metal[idx,];

      color_col = char2Rname(args[['colorCol']]);
      if (color_col %in% names(gmetal)) {
        plot_col = gmetal[,color_col];
        plot_fill = gmetal[,color_col];
      } else {
        plot_col <- args[['ldColors']][gmetal$group]; 
        plot_col[which(gmetal$pch %in% 21:25)] <- 'gray20';
        plot_fill = args[['ldColors']][gmetal$group];
      }

      grid.points(
        x=gmetal$pos,
        y=transformation(gmetal$P.value),
        pch=gmetal$pch,
        gp=gpar(
          cex=dotSizes[idx], 
          col=plot_col,
          fill=plot_fill
        )
      ); 
    }
  } else {
    # Draw SNPS with LD colors based on their best (reference, conditional) SNP. 
    by(metal,metal$best_ld_cut,function(submetal) {  
      # All SNPs that have LD. 
      grid.points(
        x = submetal$pos,
        y = transformation(submetal$P.value),
        pch = submetal$pch,
        gp=gpar(
          cex = submetal$dotSizes, 
          col = submetal$ld_color,
          fill = submetal$ld_color
        )
      );
    });

    # Plot reference/conditional SNPs last so they are on top. 
    metal_ref = metal[metal$MarkerName %in% ref_cond_snps,];
    grid.points(
      x = metal_ref$pos,
      y = transformation(metal_ref$P.value),
      pch = metal_ref$pch,
      gp=gpar(
        cex = metal_ref$dotSizes, 
        col = metal_ref$ld_color,
        fill = metal_ref$ld_color
      )
    );
    
  }
  
  # Now we need to label the lead SNPs. 
  # If a "denoteMarkersFile" was specified, this has all the information we need
  # to do the labeling, so we'll use it directly. 
  #
  # Otherwise, we'll label each reference/conditional SNP accordingly. 
  if (!is.null(denote_markers)) {
    for (i in 1:dim(denote_markers)[1]) {
      denote_row = denote_markers[i,];
      
      metal_data = as.list(metal[metal$MarkerName == denote_row$chrpos,]);
      metal_pval = transformation(metal_data$P.value);
      
      grid.refsnp(denote_row$snp,denote_row$pos,metal_pval,args[['drawMarkerNames']],denote_row$string,denote_row$color);
    }
  } else {
    # Draw label for reference SNP. 
    # This is either the label a user provides, or the actual reference SNP rs#. 
    if (! is.null(refidx)) {
      if (!is.null(args[['refsnpName']])) {
        # User-provided refsnp label. 
        grid.refsnp(args[['refsnpName']],metal$pos[refidx],metal$P.value[refidx],args[['drawMarkerNames']],shadow=args[['refsnpShadow']]);
      } else{
        # Use the actual SNP name. 
        grid.refsnp(refSnp,metal$pos[refidx],metal$P.value[refidx],args[['drawMarkerNames']],shadow=args[['refsnpShadow']]);
      }
    }
    
    # If conditional SNPs were given, we should also label those too. 
    if (!is.null(cond_ld)) {
      for (i in 1:length(ref_cond_snps)) {
        csnp = ref_cond_snps[i];
        csnp_name = ref_cond_snps_names[i];
        
        if (csnp == refSnp) {
          next;
        }
        
        csnp_data = as.list(metal[metal$MarkerName == csnp,]);
        csnp_pval = transformation(csnp_data$P.value);
        # grid.text(
          # as.character(csnp),
          # x = unit(csnp_data$pos,"native"), 
          # y = unit(csnp_pval,'native') + unit(0.1,'npc'),
          # just = c("center","top"),
          # gp = gpar(
            # cex=args[['refsnpTextSize']],
            # col=args[['refsnpTextColor']],
            # alpha=args[['refsnpTextAlpha']]
          # )
        # );
        grid.refsnp(csnp_name,csnp_data$pos,csnp_pval,args[['drawMarkerNames']]);
      }
    }
  }
    
  # Draw a dashed line at y = <significance level> if requested. 
  if (!is.null(args[['signifLine']])) {
    for (i in 1:length(args[['signifLine']])) {   
      untrans_y = args[['signifLine']][i];
      signif_line_y = transformation(untrans_y);

      signif_line_col = "black";
      signif_line_lty = 2;
      signif_line_lwd = 2;
      
      try({ signif_line_col = args[['signifLineColor']][i] },silent=T);
      try({ signif_line_lty = args[['signifLineType']][i] },silent=T);
      try({ signif_line_lwd = args[['signifLineWidth']][i] },silent=T);
      
      message("trans signif line at: ",signif_line_y);
      grid.lines(
        x = unit(c(0,1),"npc"),
        y = unit(c(signif_line_y,signif_line_y),"native"),
        gp = gpar(
          lwd=signif_line_lwd,
          lty=signif_line_lty,
          col=signif_line_col
        )
      );
    }
  }

  if (FALSE) {
    grid.points(x=metal$pos[refidx],y=transformation(metal$P.value[refidx]), 
      gp=gpar(col=args[['refsnpColor']],fill=args[['refsnpColor']],
      cex= if (args[['bigDiamond']] & args[['showRefsnpAnnot']]) 1.6*args[['refDot']] else args[['refDot']]),
      pch= if (args[['bigDiamond']] & args[['showRefsnpAnnot']]) 5 else metal$pch[refidx],
      default.units='native'
    );
  }

  if ( args[['showRecomb']] && args[['recombOver']]) {
    pushViewport(dataViewport(extension=c(0,.05),xRange,recrateRange,name='recrateClipped',clip="on"));
    if (args[['recombFill']]) {
      grid.polygon(x=recrate$pos,y=recrate$recomb,
        gp=gpar(alpha=args[['recombFillAlpha']],col=args[['recombColor']],fill=args[['recombColor']]),
              default.units='native'
      );
    } else {
      panel.xyplot(recrate$pos,recrate$recomb,type='l',lwd=2,alpha=args[['recombLineAlpha']],col=args[['recombColor']]);
    }
    upViewport(1); 
  }
  
  grid.rect(gp=gpar(lwd=args[['frameLwd']],col=args[['frameColor']],fill=NA,alpha=args[['frameAlpha']]));

  if (is.null(cond_ld)) {
    pushViewport(viewport(clip="on",name='legend'));
    
    breaks <- union(args[['ldCuts']],c(0,1));
    breaks <- sort(unique(breaks));
    nb <- length(breaks);
    cols <- args[['ldColors']]
    cols <- rep(cols, length=nb+2);
    rl <- ribbonLegend(
      breaks=breaks,
      cols=cols[2:(1+nb)],
      gp=gpar(cex=args[['legendSize']],col=args[['frameColor']],alpha=args[['frameAlapha']])
    );

    if (args[['legend']] == 'auto') { 
      args[['legend']] = AutoLegendSide(transformation(metal$P.value),metal$pos,xRange); 
    }

    if (tolower(args[['legend']]) %in% c('left','right')) {
      pushViewport(viewport(name='legendVp',
        x=if (args[['legend']] == 'left') unit(2.5,"char") else unit(1,'npc') - unit(2.5,'char'),
        y=unit(1,'npc') - unit(.5,'char'),
        just=c('center','top'),
        width=unit(4,'char'),
        height=unit(8,'lines')
      ));
      
      grid.rect(gp=gpar(col='transparent',fill='white',alpha=args[['legendAlpha']]));
      grid.rect(gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));

      pushViewport(viewport(
        name='ribbonLegend',
        y=0,
        just=c('center','bottom'),
        width=unit(4,'char'),
        height=unit(7,'lines')
      ));
      
      grid.draw(rl);
      
      upViewport(1);

      pushViewport(viewport(name='LDTitle',
        clip="off", 
        #x=unit(2.5,"char"),
        width=unit(4,"char"),
        y=unit(1,'npc') - unit(.25,'char'),
        just=c('center','top'),
        height=unit(1,'lines')
      ));
      
      grid.text(args[['LDTitle']], gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));
      
      upViewport(1);

      upViewport(1);
    } # end if show legend on left or right
    
    upViewport(1);
  } 
  # else {
    # pushViewport(viewport(clip="on",name='legend'));
    
    # rl <- multiRibbonLegend(
      # ref_cond_snps,
      # args[['ldCuts']],
      # hues,
      # sat_range,
      # gp = gpar(cex = 0.8)
    # );

    # if (args[['legend']] == 'auto') { 
      # args[['legend']] = AutoLegendSide(transformation(metal$P.value),metal$pos,xRange); 
    # }

    # if (tolower(args[['legend']]) %in% c('left','right')) {
      # legend_side = args[['legend']];
      # if (legend_side == 'left') {
        # legend_just = c('left','top');
      # } else if (legend_side == 'right') {
        # legend_just = c('right','top');
      # }
    
      # pushViewport(viewport(
        # name='legendVp',
        # x = if (args[['legend']] == 'left') unit(2.5,"char") else unit(1,'npc') - unit(2.5,'char'),
        # y = unit(1,'npc') - unit(.5,'char'),
        # just = legend_just,
        # width = unit(0.25,'npc'),
        # height = unit(0.25,'npc')
      # ));
      
      # grid.rect(gp=gpar(col='transparent',fill='white',alpha=args[['legendAlpha']]));
      # grid.rect(gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));

      # # pushViewport(viewport(
        # # name='ribbonLegend',
        # # y=0,
        # # just=c('center','bottom')
      # # ));
      
      # grid.draw(rl);
      
      # # upViewport(1);

      # upViewport(1);
    # } # end if show legend on left or right
    
    # upViewport(1);
  # }

  upViewport(3);   

  ######### subtitle space; place holder for now
  if(FALSE) {
    pushViewport(viewport(layout.pos.row=8,layout.pos.col=2,name="subtitle"));
    grid.rect(gp=gpar(col='red'));
    grid.xaxis(gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));
    grid.text(paste('Position on',chr2chrom(args[['chr']]),"(Mb)"),
      gp=gpar(col="red"));
    upViewport(1);
  }
  
  ########## annotation (genes)
  if(args[['rfrows']] > 0) {
    pushViewport(
      viewport(xscale=pvalVp$xscale,
        layout.pos.row=15,
        layout.pos.col=2,
        name="refFlatOuter")
    );
    
    pushViewport(
      viewport(xscale=pvalVp$xscale,
        name="refFlatInner",
        clip="on")
    );
    
    grid.rect(gp=gpar(lwd=args[['frameLwd']],col=args[['frameColor']],alpha=args[['frameAlpha']]));
    
    panel.flatbed(
      flat=refFlat,
      showPartialGenes = args[['showPartialGenes']],
      shiftGeneNames = args[['shiftGeneNames']],
      rows=args[['rfrows']], 
      cex=args[['geneFontSize']],
      col=args[['geneColor']],
      fill=args[['geneColor']],
      multiplier=1/args[['unit']]
    );
    
    upViewport(1);
    
    if ( !is.null(args[['xnsmall']]) && !is.null(args[['xat']]) ) {
      grid.xaxis(at=args[['xat']], label=format(args[['xat']], nsmall=args[['xnsmall']]),
        gp=gpar(cex=args[['axisSize']],col=args[['frameColor']],alpha=args[['frameAlpha']]));
    } else {
      grid.xaxis(at=args[['xat']], 
        gp=gpar(cex=args[['axisSize']],col=args[['frameColor']],alpha=args[['frameAlpha']]));
    }
    
    grid.text(paste('Position on',chr2chrom(args[['chr']]),unit2char(args[['unit']])), 
      y=unit(args[['xlabPos']],'lines'),just=c('center',"bottom"),
      gp=gpar(cex=args[['axisTextSize']], col=args[['axisTextColor']], alpha=args[['frameAlpha']]) 
    );

    panel.hilite(
      range=c(args[['hiStartBP']]/args[['unit']],args[['hiEndBP']]/args[['unit']]),
      fill=args[['hiColor']],
      alpha=args[['hiAlpha']]
    );

    if (!is.null(args[['hiStarts']]) & !is.null(args[['hiEnds']]) & !is.null(args[['hiColors']])) {
      down = seekViewport("refFlatOuter");
      startV = as.numeric(strsplit(args[['hiStarts']],",")[[1]]);
      endV = as.numeric(strsplit(args[['hiEnds']],",")[[1]]);
      hiColors = strsplit(args[['hiColors']],",")[[1]];

      for (i in seq(1,length(startV))) {
        x1 = startV[[i]];
        x2 = endV[[i]];
        hiCol = hiColors[[i]];

        panel.hilite(
          range=c(x1/args[['unit']],x2/args[['unit']]),
          fill=hiCol,
          alpha=args[['hiAlpha']]
        );
      }
      
    }

    upViewport(1);
  }

  ########## rugs for snpsets
  pushViewport(viewport(
    xscale=pvalVp$xscale,
    layout.pos.row=3,
    layout.pos.col=2,
    name="rugs",
    clip="off")
  );
      
  i <- nrugs;
  for (snpset in levels(rug$snp_set)) {
    grid.text(as.character(snpset),x=unit(-.25,"lines"),
        y=(i-.5)/nrugs, just="right",
        gp=gpar(col=args[['rugColor']], alpha=args[['rugAlpha']],cex=.90*args[['axisTextSize']])
        );
    i <- i-1;
  }

  pushViewport(viewport(
    xscale=pvalVp$xscale,
    layout.pos.row=3,
    layout.pos.col=2,
    name="rugsClipped",
    clip="on")
  );
  
  i <- nrugs;
  for (snpset in levels(rug$snp_set)) {
    panel.rug( rug[ which(rug$snp_set==snpset), "pos" ] , 
      start = (i-1)/(nrugs) + (.15/nrugs),
      end = (i)/(nrugs) - (.15/nrugs),
      y.units=rep("native",2),
      col=args[['rugColor']],
      alpha=args[['rugAlpha']]
      );
    i <- i-1;
  }

  upViewport(2);

  if(args[['fmrows']] > 0) {
    pushViewport(viewport(
      xscale=pvalVp$xscale,
      layout.pos.row=13,
      layout.pos.col=2,
      name="finemapOuter"
    ));

    pushViewport(
      viewport(xscale=pvalVp$xscale,
        name="finemapInner",
        clip="on")
    );

    panel.finemap(
      fmregions,
      showPartial = args[['showPartialGenes']],
      shiftNames = args[['shiftGeneNames']],
      rows=args[['fmrows']], 
      cex=args[['geneFontSize']],
      col=args[['geneColor']],
      fill=args[['geneColor']],
      multiplier=1/args[['unit']]
    );
    
    upViewport(1);

    grid.rect(gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));
    
    upViewport(1);
  }

  if(args[['gwrows']] > 0) {
    pushViewport(viewport(
      xscale=pvalVp$xscale,
      layout.pos.row=11,
      layout.pos.col=2,
      name="gwasOuter"
    ));

    pushViewport(
      viewport(xscale=pvalVp$xscale,
        name="gwasInner",
        clip="on"
      )
    );

    panel.gwas(
      gwas_hits,
      showPartial = args[['showPartialGenes']],
      shiftNames = args[['shiftGeneNames']],
      rows=args[['gwrows']], 
      cex=args[['geneFontSize']],
      col=args[['geneColor']],
      fill=args[['geneColor']],
      multiplier=1/args[['unit']]
    );
    
    upViewport(1);

    grid.rect(gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));
    
    upViewport(1);
  }
  
  if (!is.null(bed_tracks)) {
    pushViewport(viewport(
      xscale = pvalVp$xscale,
      layout.pos.row = 9,
      layout.pos.col = 2,
      name = "bedTracks"
    ));
    
    panel.bed(bed_tracks,bed_height,args[['startBP']],args[['endBP']]);
    grid.rect(gp=gpar(col=args[['frameColor']],fill=NA,alpha=args[['frameAlpha']]));
    
    upViewport(1);
  }

  if (draw_barplot) {
    bar_min_y = ifelse(is.null(args[['barplotMinY']]),min(barplot_data$value),args[['barplotMinY']])
    bar_max_y = ifelse(is.null(args[['barplotMaxY']]),max(barplot_data$value),args[['barplotMaxY']])

    pushViewport(viewport(
      xscale = pvalVp$xscale,
      yscale = c(bar_min_y,bar_max_y),
      layout.pos.row = 7,
      layout.pos.col = 2,
      name = "barplot"
    ));
    
    panel.barplot(barplot_data)
    
    #upViewport(1);

    #pushViewport(viewport(
      #yscale = c(0,1),
      #layout.pos.row = 7,
      #layout.pos.col = 1,
      #name = "barplot_axis"
    #))

    if (!is.null(args[['barplotAxisTicks']])) {
      bar_yat = args[['barplotAxisTicks']]
    } else {
      bar_yat = lattice::yscale.components.default(range(barplot_data$value))$left$ticks$at
    }

    grid.yaxis(
      #at=args[['yat']],
      at = bar_yat,
      gp=gpar(
        cex=args[['axisSize']],
        col=args[['frameColor']],
        alpha=args[['frameAlpha']]
      )
    );

    if (!is.null(args[['barplotAxisLabel']]) && nchar(args[['barplotAxisLabel']]) > 0) {
      grid.text(
        x=unit(args[['barplotLabelPos']],'lines'),
        label=args[['barplotAxisLabel']],
        rot=90, 
        gp=gpar(
          cex=args[['axisTextSize']],
          col=args[['axisTextColor']],
          alpha=args[['frameAlpha']]
        ) 
      );
    }

    grid.rect(gp=gpar(lwd=args[['frameLwd']],col=args[['frameColor']],alpha=args[['frameAlpha']]));

    upViewport(1)
  }

  if (is.character(postlude) && file.exists(postlude)) {
    source(postlude);
  }

}  ## end zplot

# Helper function to draw table grobs on the log page. Prints the table without rownames. 
# This handles the two different gridExtra package versions with different interfaces. 
draw_table = function(...) {
  ge_version = as.character(packageVersion("gridExtra"))
  vcomp = compareVersion(ge_version,"2.0.0")
  if (vcomp == -1) {
    grid.draw(gridExtra::tableGrob(
      ...,
      show.rownames = F
    ))
  } else {
    grid.draw(gridExtra::tableGrob(
      ...,
      rows = NULL
    ))
  }
}

grid.extralog = function(dframe,fudge = 0.63,main = NULL) {
  check <- "gridExtra" %in% installed.packages()[,1]
  
  if (!check) {
    warning("gridExtra not installed - skipping extra PDF pages for GWAS hits and fine-mapping regions");
    return();
  }
  
  # Number of rows we can fit per page using the default tableGrob settings. 
  rows_per_page = floor(as.numeric(convertUnit(unit(1,'npc'),"lines","y")) * 0.6)
  
  # Chunk up the data frame into sections depending on how many rows can fit on a page. 
  dframe$chunk = ceiling(1:dim(dframe)[1] / rows_per_page);
  for (dc in unique(dframe$chunk)) {
    # Chop off the chunk column. 
    dsub = dframe[dframe$chunk == dc,(1:(dim(dframe)[2] - 1))];
    
    # Draw the table. 
    grid.newpage();
    draw_table(dsub);

    # Draw title if provided. 
    if (!is.null(main)) {
      grid.text(main,0.5,unit(1,'npc') - unit(1,'lines'));
    }
  }
}

grid.log <- function(args,metal,linespacing=1.5,ascii=FALSE,debug=FALSE){
  labels=c("date");
  values=c(date());
#    labels=c(labels,"working directory");
#    values=c(values,getwd());

#    labels=c(labels,"unit");
#    values=c(values,args[['unit']]);
  labels=c(labels,"build");
  values=c(values,args[['build']]);
  labels=c(labels,"display range");
  values=c(values,paste( 'chr',args[['chr']],":",args[['start']], "-", args[['end']], " [",args[['startBP']],"-",args[['endBP']], "]",sep=""));
  labels=c(labels,"hilite range");
  values=c(values,paste( args[['hiStart']], "-", args[['hiEnd']], " [",args[['hiStartBP']],"-",args[['hiEndBP']], "]"));
  labels=c(labels,"reference SNP");
  values=c(values,args[['refsnp']]);

#    labels=c(labels,"prefix");
#    values=c(values,args[['prefix']]);
#    labels=c(labels,"log");
#    values=c(values,args[['log']]);
  if (!is.null(args[['reload']])) {
    labels=c(labels,"reload");
    values=c(values,args[['reload']]);
  }

  if(! is.null(args[['reload']]) || debug){
      labels=c(labels,"data reloaded from");
      values=c(values,args[['rdata']]);
  }

  labels=c(labels,"number of SNPs plotted");
  values=c(values,as.character(dim(metal)[1]));

  labels=c(labels,paste("min",args[['pvalCol']]));
  maxIdx <- which.max(transformation(metal$P.value));
  maxName <- as.character(metal$MarkerName[maxIdx]);
  maxNegLogP <- transformation(metal$P.value[maxIdx]);
  maxPSci <- log2sci(-maxNegLogP)
  values=c(values,paste(maxPSci," [", maxName ,"]",sep=""));

  labels=c(labels,paste("max",args[['pvalCol']]));
  minIdx <- which.min(transformation(metal$P.value));
  minName <- as.character(metal$MarkerName[minIdx]);
  minNegLogP <- transformation(metal$P.value[minIdx]);
  minPSci <- log2sci(-minNegLogP)
  values=c(values,paste(minPSci," [", minName ,"]",sep=""));

  if (TRUE) { 
    oG <- omittedGenes;
    while (length(oG) > 0) {
      labels=c(labels,"omitted Genes");
      values=c(values,paste(oG[1:min(length(oG),3)],collapse=", "));
      oG <- oG[-(1:3)]
    }
    
    oG <- omittedGWAS;
    while (length(oG) > 0) {
      labels=c(labels,"omitted GWAS Hits");
      values=c(values,paste(oG[1:min(length(oG),2)],collapse=", "));
      oG <- oG[-(1:3)]
    }
    
    oG <- omittedFineMap;
    while (length(oG) > 0) {
      labels=c(labels,"omitted fine mapping");
      values=c(values,paste(oG[1:min(length(oG),3)],collapse=", "));
      oG <- oG[-(1:3)]
    }
  }
  
  if (TRUE) { 
    w <- warningMessages;
    while (length(w) > 0) {
      labels=c(labels,"Warning");
      values=c(values,w[1]);
      w <- w[-1]
    }
  }

  labels=paste(labels, ":  ",sep='');

  if (ascii) {
    cat(paste(format(labels,width=20,justify="right"),values,sep=" ",collapse="\n"));
    cat('\n');
    cat('\nMake more plots at http://csg.sph.umich.edu/locuszoom/');
    cat('\n');
  } else {
    grid.text(labels,x=.3,y=unit(1,'npc') - unit(linespacing *(1:length(labels)),'lines'), just='right');
    grid.text(values,x=.3,y=unit(1,'npc') - unit(linespacing *(1:length(values)),'lines'), just='left');

    if (FALSE && args[['showAnnot']]) {
      annotlabels <- c('no annotation','framestop','splice','nonsyn','coding','utr','tfbscons','mcs44placental');
      pch <- args[['annotPch']];
      annotlabels <- c(annotlabels[-1],annotlabels[1])
      pch <- c(pch[-1],pch[1]):
      key <- simpleKey(text=annotlabels);
      key$points$pch=pch;
      key$points$col="navy";
      key$points$fill="lightskyblue";
      keyGrob <- draw.key(key,draw=FALSE);
      annotationBoxTop <- unit(0.95,'npc');
      annotationBoxHeight <- unit(3,"lines") + grobHeight(keyGrob);
      pushViewport(viewport(x=.90,y=annotationBoxTop,width=grobWidth(keyGrob),
        height=annotationBoxHeight,just=c('right','top')));
      grid.rect();
      pushViewport(viewport(y=unit(.75,'lines'),height = grobHeight(keyGrob),just=c('center','bottom')));
        draw.key(key,draw=TRUE);
      popViewport();
      grid.text('Annotation key',x=.5,y=unit(1,'npc') - unit(1,'lines'),just=c('center','top'))
      popViewport();
    } 
    if ( 'annot' %in% names(metal) && args[['showAnnot']] && is.null(cond_ld) ) {
      annotlabels <- levels(as.factor(metal$annot))
      pch <- rep(args[['annotPch']],length=length(annotlabels));
      key <- simpleKey(text=annotlabels);
      key$points$pch=pch;
      key$points$col="navy";
      key$points$fill="lightskyblue";
      keyGrob <- draw.key(key,draw=FALSE);
      annotationBoxTop <- unit(0.95,'npc');
      annotationBoxHeight <- unit(3,"lines") + grobHeight(keyGrob);
      pushViewport(viewport(x=.90,y=annotationBoxTop,width=grobWidth(keyGrob),
        height=annotationBoxHeight,just=c('right','top')));
      pushViewport(viewport(y=unit(.75,'lines'),height = grobHeight(keyGrob),just=c('center','bottom')));
        draw.key(key,draw=TRUE);
        grid.rect();
      popViewport();
      grid.text('annotation key',x=.5,y=unit(1,'npc') - unit(1,'lines'),just=c('center','top'))
      popViewport();
    } else { if (args[['showAnnot']]) { 
      annotlabels <- c('no annotation','framestop','splice','nonsyn','coding','utr','tfbscons','mcs44placental');
      pch <- args[['annotPch']];
      annotlabels <- c(annotlabels[-1],annotlabels[1])
      pch <- c(pch[-1],pch[1])
      key <- simpleKey(text=annotlabels);
      key$points$pch=pch;
      key$points$col="navy";
      key$points$fill="lightskyblue";
      keyGrob <- draw.key(key,draw=FALSE);
      annotationBoxTop <- unit(0.95,'npc');
      annotationBoxHeight <- unit(3,"lines") + grobHeight(keyGrob);
      pushViewport(viewport(x=.90,y=annotationBoxTop,width=grobWidth(keyGrob),
        height=annotationBoxHeight,just=c('right','top')));
      popViewport();
    } }

    if ( args[['legend']] %in% c('left','right') ) {
      annotlabels <- c('no annotation','framestop','splice','nonsyn','coding','utr','tfbscons','mcs44placental');
      pch <- args[['annotPch']];
      annotlabels <- c(annotlabels[-1],annotlabels[1])
      pch <- c(pch[-1],pch[1])
      key <- simpleKey(text=annotlabels);
      key$points$pch=pch;
      key$points$col="navy";
      key$points$fill="lightskyblue";
      keyGrob <- draw.key(key,draw=FALSE);
      annotationBoxTop <- unit(0.95,'npc');
      annotationBoxHeight <- unit(3,"lines") + grobHeight(keyGrob);
      
      pushViewport(viewport(name='legendVpPage2',
        x=unit(.9,'npc'),
        y=annotationBoxTop - annotationBoxHeight - unit(2,'lines'),
        just=c('right','top'),
        width=unit(4,'char'),
        height=unit(8,'lines')
      ));
      grid.rect(gp=gpar(col='transparent',fill='white',alpha=args[['legendAlpha']]));
      grid.rect(gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));

      if (is.null(args[['cond_ld']])) {
        breaks <- union(args[['ldCuts']],c(0,1));
        breaks <- sort(unique(breaks));
        nb <- length(breaks);
        cols <- args[['ldColors']]
        cols <- rep(cols, length=nb+2);
        rl <- ribbonLegend(
          breaks=breaks,
          cols=cols[2:(1+nb)],
          gp=gpar(cex=args[['legendSize']],col=args[['frameColor']],alpha=args[['frameAlapha']])
        );
      
        pushViewport(viewport(name='ribbonLegendPage2',
          y=0,
          just=c('center','bottom'),
          width=unit(4,'char'),
          height=unit(7,'lines')
        ))
        grid.draw(rl);
        upViewport(1);

        pushViewport(viewport(name='LDTitlePage2',
          clip="off", 
          width=unit(4,"char"),
          y=unit(1,'npc') - unit(.25,'char'),
          just=c('center','top'),
          height=unit(1,'lines')
        ))
        grid.text(args[['LDTitle']], gp=gpar(col=args[['frameColor']],alpha=args[['frameAlpha']]));
        upViewport(1);
      }

      upViewport(1);
    }

  grid.text('Make more plots at http://csg.sph.umich.edu/locuszoom/', y=unit(1,'lines'), just=c('center','bottom'));
  }
}

#############################################################
#
# process argument list, splitting the key=value pairs
#
argv <- function(){
  args <- commandArgs(TRUE);
  newl <- list()

  for ( i in 1:length(args) ) {
    keyval <- strsplit(args[[i]],"=")[[1]];
    key <- keyval[1]; val <- keyval[2];
    newl[[ key ]] <- val; 
  }
  return(newl)
}

#################################################################################
#                                                                               #
#                         MAIN PROGRAM BEGINS HERE                              #
#                                                                               #
#################################################################################


flags <- list(flank=FALSE,reloaded=FALSE);
createdFiles <- list();
refSnpPos <- empty.data.frame();

#
# set program defaults -- may be overridden with command line arguments
#
default.args <- list(
  theme = NULL,                         # select a theme (collection of settings) for plot
  experimental = FALSE,                 # try some experimental features?
  pquery = FALSE,                       # is pquery available?
  format = "pdf",                       # file format (pdf or png or both)
  recombTable = "results.recomb_rate",  # Recomb Rate Table (for SQL)
  clean=TRUE,                           # remove temp files?
  build = "hg18",                       # build to use for position information
  metal = "metal.tbl",                  # metal output file
  alreadyTransformed=FALSE,             # are metal p-values already -log10() -transformed?
  pvalCol="P.value",                    # name for p-value column in metal file
  posCol="pos",                         # name for positions column in metal file
  markerCol="MarkerName",               # name for MarkerName column in metal file
  weightCol="Weight",                   # name for weights column in metal file
  weightRange=NULL,                     # use this instead of actual range of weights
  ymin=0,                               # min for p-value range (expanded to fit all p-vals if needed)
  ymax=10,                              # max for p-value range (expanded to fit all p-vals if needed)
  yat=NULL,                             # values for y-axis ticks
  xat=NULL,                             # values for x-axis ticks
  xnsmall=NULL,                         # number of digits after decimal point on x-axis labels
  chr = NULL,                           # chromosome
  start = NULL,                         # start of region (string, may include Mb, kb, etc.)
  end = NULL,                           # end of region (string, may include Mb, kb, etc.)
  flank = "300kb",                      # surround refsnp by this much
  xlabPos = -3.0,                       # position of xaxis label (in lines relative to bottom panel)
  ylabPos = -3.0,                       # position of yaxis label (in lines relative to left edge of panel)
  ylab = NULL,                            # override default label for y-axis
  recombPos = 3.0,                      # position of recomb label (in lines relative to right edge of panel)
  axisSize = 1,                         # sclaing factor for axes
  axisTextSize = 1,                     # sclaing factor for axis labels
  axisTextColor = "gray30",             # color of axis labels
  requiredGene = NULL,                  # required gene for gene track; specify as list of genes "GENE1,GENE2,GENE3"
  hiRequiredGene = FALSE,               # should we highlight the required gene?
  hiRequiredGeneColor = "red",          # color for required gene highlight box 
  refsnp = NULL,                        # snp name (string)
  refsnpName = NULL,                    # name given to refsnp on plot (usually same as refsnp)
  refsnpShadow = FALSE,                 # put a shadow around the refsnp?
  drawMarkerNames = TRUE,               # draw the rs# SNP names above them?
  denoteMarkersFile = NULL,              # file specifying marker names to highlight (along with brief label and/or color)
  refsnpTextColor = "black",            # color for ref snp label
  refsnpTextSize = 1,                   # sclaing factor for text size
  refsnpTextAlpha = 1,                  # alpha for ref snp label
  refsnpLineColor = "transparent",      # color for ref snp line (invisible by default)
  refsnpLineAlpha = 1,                 # alpha for ref snp line
  refsnpLineWidth = 1,                  # width of ref snp line
  refsnpLineType = 2,                   # type of ref snp line (2 = dashed, 1 = solid, etc.. R defaults)
  signifLine = NULL,                    # draw a horizontal line at significance threshold (specify in p-value scale), can provide vector of values too for multiple lines
  signifLineType = NULL,                # specify line types, can be a vector 1 per signif line
  signifLineColor = NULL,               # specify line colors, can be a vector 1 per signif line
  signifLineWidth = NULL,               # specify line widths, can be a vector 1 per signif line
  cond_snps = NULL,                     # list of SNPs that remain significant after conditional analysis
  cond_pos = NULL,                      # list of conditional SNPs in chr:pos form
  title = "",                           # title for plot
  expr_title = "",                      # give title as expression instead of plain text, see plotmath for syntax
  titleColor = "black",                 # color for title 
  titleFontFace = "plain",              # font face for title, use "italic" for genes
  titleCex = 2,                         # size change for title
  thresh = 1,                           # only get pvalues <= thresh   # this is now ignored.
  width = 10,                           # width of pdf (inches)
  height = 7,                           # height of pdf (inches)
  leftMarginLines = 5,                  # margin (in lines) on left
  rightMarginLines = 5,                 # margin (in lines) on right
  unit=1000000,                         # bp per unit displayed in plot
  ldTable = "results.ld_point6",        # LD Table (for SQL)
  annot=NULL,                           # file for annotation 
  showAnnot=TRUE,                       # show annotation for each snp?
  showGenes=TRUE,                       # show genes?
  annotCol='annotation',                # column to use for annotation, if it exists
  colorCol='color',                     # column to use to override SNP colors, if it exists
  annotPch='24,24,25,22,22,8,7,21,1',        # plot symbols for annotation
  condPch='4,16,17,15,25,8,7,13,12,9,10',    # plot symbols for groups of LD blocks of SNPs
  condRefsnpPch=23,                     # symbol to use for refsnps in conditional plot, NULL means use same symbol as other SNPs in group
  annotOrder=NULL,                      # ordering of annotation classes
  showRefsnpAnnot=TRUE,                 # show annotation for reference snp too?
  bigDiamond=FALSE,                     # put big diamond around refsnp?
  bedTracks=NULL,
  ld=NULL,                              # file for LD information for reference SNP
  cond_ld=NULL,                         # files for LD information for conditional SNPs
  ldCuts = "0,.2,.4,.6,.8,1",           # cut points for LD coloring
  ldThresh=NULL,                        # LD values below this threshold will be colored identically
  ldColors = "gray60,navy,lightskyblue,green,orange,red,purple3",                   # colors for LD on original LZ plots
  condLdColors = "gray60,#E41A1C,#377EB8,#4DAF4A,#984EA3,#FF7F00,#A65628,#F781BF",  # colors for conditional LD plots (red, blue, green, purple, orange)
  condLdLow = NULL,                     # color all SNPs with conditional LD in the lowest bin the same color (specify that color here)
  ldCol='rsquare',                      # name for LD column
  LDTitle=NULL,                         # title for LD legend
  smallDot = .4,                        # smallest p-value cex 
  largeDot = .8,                        # largest p-value cex 
  refDot = NULL,                        # largest p-value cex 
  rfrows = '10',                        # max number of rows for reflat genes
  fmrows = 3,                           # max number of rows for fine mapping regions
  gwrows = 3,                           # max number of rows for gwas hits
  warnMissingGenes = TRUE,             # should we warn about missing genese on the plot?
  warnMissingFineMap = TRUE,            # should we warn about missing fine mapping regions on the plot? 
  warnMissingGWAS = TRUE,               # should we wearn about missing gwas hits on the plots? 
  showPartialGenes = TRUE,              # should genes that don't fit completely be displayed?
  shiftGeneNames = TRUE,                # should genes that don't fit completely be displayed?
  geneFontSize = .8,                    # size for gene names
  geneColor = "navy",                   # color for genes
  snpset = "Affy500,Illu318,HapMap",    # SNP sets to show
  snpsetFile = NULL,                    # use this file for SNPset data (instead of pquery)
  rugColor = "gray30",                  # color for snpset rugs
  rugAlpha = 1,                         # alpha for snpset rugs
  metalRug = NULL,                      # if not null, use as label for rug of metal positions
  refFlat = NULL,                       # use this file with refFlat info (instead of pquery)
  fineMap = NULL,                       # give a file with fine mapping posterior probabilities
  gwasHits = NULL,                      # give a file with GWAS catalog hits (chr, pos, trait)
  barplotRows = 3,                      # max number of rows for barplot 
  barplotData = NULL,                   # give a file with barplot data (position, value)
  barplotAxisLabel = NULL,              # what should axis be called for barplot data?
  barplotAxisTicks = NULL,              # where should the ticks be drawn for the barplot data?
  barplotMinY = NULL,                   # minimum value to show for barplot (automatically calc if NULL)
  barplotMaxY = NULL,                   # maximum value to show for barplot (automatically calc if NULL)
  barplotLabelPos = -3,                 # distance of barplot axis label from left side of plot (in grid lines)
  barplotThreshold = 0.95,              # show a barplot threshold 
  barplotDrawMatchingLine = TRUE,       # draw dashed lines for where barplot peaks match a SNP position
  barplotDrawOffLine = TRUE,             # draw dashed lines for where barplot peaks do not match a SNP position
  barplotMatchingLineColor = "black",   # color of dashed SNP-matched barplot peak lines
  barplotMatchingLineStyle = 2,
  barplotOffLineColor = "black",         # color of dashed non-SNP-matched barplot peak lines
  barplotOffLineStyle = 3,
  showIso=FALSE,                        # show each isoform of gene separately
  showRecomb = TRUE,                    # show recombination rate?
  recomb=NULL,                          # rcombination rate file
  recombAxisColor=NULL,                 # color for reccomb rate axis labeing
  recombAxisAlpha=NULL,                 # color for reccomb rate axis labeing
  recombColor='blue',                   # color for reccomb rate on plot
  recombOver = FALSE,                   # overlay recombination rate? (else underlay it)
  recombFill = FALSE,                   # fill recombination rate? (else line only)
  recombFillAlpha=0.2,                  # recomb fill alpha
  recombLineAlpha=0.8,                  # recomb line/text alpha
  frameColor='gray30',                  # frame color for plots
  frameAlpha=1,                         # frame alpha for plots
  frameLwd=1,                         # frame line width
  legendSize=.8,                        # scaling factor of legend
  legendAlpha=1,                        # transparency of legend background
  legendMissing=TRUE,                   # show 'missing' as category in legend?
  legend='auto',                        # legend? (auto, left, right, or none)
  hiStart=0,                            # start of hilite region
  hiEnd=0,                              # end of hilite region
  hiColor="blue",                       # hilite color
  hiAlpha=0.25,                          # hilite alpha
  hiStarts=NULL,
  hiEnds=NULL,
  hiColors=NULL,
  clobber=TRUE,                         # overwrite files?
  reload=NULL,                          # .Rdata file to reload data from
  prelude=NULL,                         # code to execute after data is read but before plot is made (allows data modification)
  postlude=NULL,                        # code to execute after plot is made (allows annotation)
  prefix=NULL,                          # prefix for output files
  dryRun=FALSE                          # show a list of the arguments and then halt
  )

### default data

refSnpPos <- data.frame()
recrate.default <- data.frame(chr=NA, pos=NA, recomb=NA, chr=NA, pos=NA)[c(),,drop=FALSE]
rug.default <- data.frame(snp=NA, chr=NA, pos=NA, snp_set=NA)[c(),,drop=FALSE]
annot.default <- data.frame(snp=NA,annot_rank=NA) # [c(),,drop=FALSE]
ld.default <- data.frame(snp1='rs0000', snp2='rs0001', build=NA, 
        chr=0, pos1=0, pos2=2, midpoint=1, distance=2, 
        rsquare=0, dprime=0, r2dp=0) # [c(),,drop=FALSE]

refFlatRaw.default <- data.frame(geneName=NA, name=NA, chrom=NA, strand=NA, txStart=NA, txEnd=NA, 
    cdsStart=NA, cdsEnd=NA, exonCount=NA, exonStarts=NA, exonEnds=NA, status=NA)[c(),,drop=FALSE]

#
# read and process command line arguments
#

user.args <- ConformList(argv(),names(default.args),message=TRUE)

default.args <- ProcessThemes(default.args,user.args[['theme']])

args <- ModifyList(default.args,user.args);

userFile <- list(
      recomb = !is.null(args[['recomb']]),
  snpsetFile = !is.null(args[['snpsetFile']]),
     refFlat = !is.null(args[['refFlat']]),
          ld = !is.null(args[['ld']]),
       annot = !is.null(args[['annot']])
  );

args <- MatchIfNull(args,'recombAxisAlpha','recombLineAlpha')
args <- MatchIfNull(args,'recombAxisColor','recombColor')
args <- AdjustModesOfArgs(args);

if ( args[['pquery']] ){
  GetData <- GetDataFromFileOrCommand
} else {
  GetData <- GetDataFromFileIgnoreCommand
}

args[['showRefsnpAnnot']] <- args[['showAnnot']] & args[['showRefsnpAnnot']];

args[['refsnpColor']] <- args[['ldColors']][length(args[['ldColors']])];

if ( args[['dryRun']] )  {
  message("Argument list:");
  message(paste("\t",names(args),'=', args, "\n"));
  q();
}

#
# read metal data or reload all.
#

if ( is.null(args[['reload']]) ) {
    if ( file.exists( args[['metal']]) ) {
      metal_header = scan(args[['metal']],what=character(),nlines=1);
      
      col_classes = list();
      pval_col = char2Rname(args[['pvalCol']]);
      col_classes[[pval_col]] = "numeric";
      
      t_color_col = char2Rname(args[['colorCol']])
      if (t_color_col %in% metal_header) { 
        col_classes[[t_color_col]] = "character"
      }

      metal <- read.file(args[['metal']],sep="\t",colClasses=col_classes,comment.char="");
    } else {
      stop(paste('No such file: ', args[['metal']]));
    }
} else {
  if ( file.exists(args[['reload']]) ) {
     load( args[['reload']] );
     flags[['reloaded']] <- TRUE;
  } else {
     stop(paste("Stopping: Can't reload from", args[['reload']]));
  }
}
#
# column renaming in metal data.frame
#
if ( char2Rname(args[['pvalCol']]) %in% names(metal) ) {
  metal$P.value <- metal[ ,char2Rname(args[['pvalCol']]) ];
} else {
  stop(paste('No column named',args[['pvalCol']]));
}

transformation <- SetTransformation( min(metal$P.value,na.rm=TRUE), max(metal$P.value,na.rm=TRUE), 
          args[['alreadyTransformed']] );

args[['LDTitle']] <- SetLDTitle( args[['ldCol']],args[['LDTitle']] )

if ( args[['posCol']] %in% names(metal) ) {
  # This one later gets modified to be in Mb, which ends up as float
  metal$pos <- metal[ ,args[['posCol']] ];

  # This one will stay as an integer (or at least, it better be)
  metal$pos_int <- as.integer(metal[,args[['posCol']]])
} else {
  stop(paste('No column named',args[['posCol']]));
}

if ( char2Rname(args[['markerCol']]) %in% names(metal) ) {
  metal$MarkerName <- metal[ ,char2Rname(args[['markerCol']]) ];
} else {
  stop(paste('No column named',args[['markerCol']]));
}

#
# if no region and no refsnp specified, choose best snp and range of data set:
#
if ( (is.null(args[['start']]) || is.null(args[['end']]) || is.null(args[['chr']]) ) && ( is.null(args[['refsnp']]) ) ) 
{
  args[['start']] <- min(metal$pos);
  args[['end']] <- max(metal$pos);
  args[['chr']] <- min(metal$chr);
  args[['refsnp']] <- as.character( metal$MarkerName[ order(metal$P.value)[1] ] );

  args <- ModifyList(list(prefix=paste('chr',
      args[['chr']],"_",args[['start']],"-",args[['end']],sep='')),
      args);

  args <- ModifyList(list(prefix='foo'),args);
  flags[['flank']] <- FALSE;
  

# if region but not refsnp, choose best snp as refsnp
} else if ( !is.null(args[['start']]) && !is.null(args[['end']]) && !is.null(args[['chr']]) && is.null(args[['refsnp']] ) ) 
{
  args <- ModifyList(
    list( refsnp = as.character( metal$MarkerName[ order(metal$P.value)[1] ] ) ),
    args
    );
  flags[['flank']] <- FALSE;

# if refsnp specifed but no region, select region flanking refsnp
} else if ( ( is.null(args[['start']]) || is.null(args[['end']]) || is.null(args[['chr']]) ) && (!is.null(args[['refsnp']]) ) ) 
{
  args <- ModifyList( args, list( flankBP=pos2bp(args[['flank']]) ) );

  refSnpPosFile <- paste(args[['refsnp']],"_pos.tbl",sep="");

  command <- paste("pquery snp_pos",
            " -defaults",
            " -sql",
            " Snp=", args[["refsnp"]],
            " Build=",args[["build"]],
            sep="");
  if ( is.null(refSnpPos) ) { args[['showRug']] = FALSE }
  refSnpPos <- GetData( refSnpPosFile, default=refSnpPos.default, command=command, clobber=TRUE);

  args[['refSnpPos']] <- as.character(refSnpPos$chrpos[1]);
  args[['refSnpBP']] <- pos2bp(refSnpPos$chrpos[1]);

  args <- ModifyList( args, list( start=args[['refSnpBP']] - args[['flankBP']] ) ) ;
  args <- ModifyList( args, list( end=args[['refSnpBP']] + args[['flankBP']] ) );
  args <- ModifyList( args, list( chr=refSnpPos$chr[1] ) );

  flags[['flank']] <- TRUE;

# else refsnp and region specified
} else {  
  flags[['flank']] <- FALSE;
}

# change refsnp to "none" if it was null, else leave as is
args <- ModifyList( list( refsnp = "none"), args);

args <- ModifyList( args, list( start=as.character(args[['start']]) ) );
args <- ModifyList( args, list( end=as.character(args[['end']]) ) );

# prefix
if (flags[['flank']]) {
  args <- ModifyList(
    list( prefix = paste(                   # #1
        args[['refsnp']],
        "_",   args[['flank']],
        sep="")
        ),
    args
    );
} else {
  args <- ModifyList(
    list( prefix = paste(                   # #2
        "chr", args[['chr']],
        "_",   args[['start']],
        "-",   args[['end']],
        sep="")
        ),
    args
    );
}

#log
args <- ModifyList(
  list( log = paste(args[['prefix']], ".log", sep="") ),
  args 
    );

#recomb
args <- ModifyList(
  list( recomb = paste(args[['prefix']], "_recomb", ".tbl", sep="") ),
  args 
    );

# annot
args <- ModifyList(
  list( annot = paste(args[['prefix']], "_annot", ".tbl", sep="") ),
  args 
    );

# ld
args <- ModifyList(
  list( ld = paste(args[['prefix']], "_ld", ".tbl", sep="") ),
  args 
    );

# snpsets
args <- ModifyList(
  list( snpsetFile = paste(args[['prefix']], "_snpsets", ".tbl", sep="") ),
  args 
    );

# pdf
args <- ModifyList(
  list( pdf = paste(args[['prefix']], ".pdf", sep="") ),
  args
  );
  
# pdf
args <- ModifyList(
  list( svg = paste(args[['prefix']], ".svg", sep="") ),
  args
  );

args <- ModifyList(
  list( png = paste(args[['prefix']], ".png", sep="") ),
  args
  );

args <- ModifyList(
  list( tiff = paste(args[['prefix']], ".tiff", sep="") ),
  args
  );

# rdata
args <- ModifyList(
  list( rdata = paste(args[['prefix']], ".Rdata", sep="") ),
  args
  );

# refFlat
args <- ModifyList(
  list( refFlat = paste(args[['prefix']], "_refFlat.txt", sep="") ),
  args
  );

args <- ModifyList(args, list( startBP=pos2bp(args[['start']]), endBP=pos2bp(args[['end']]) ));
args <- ModifyList(args, list( hiStartBP=pos2bp(args[['hiStart']]), hiEndBP=pos2bp(args[['hiEnd']]) ));

#######################################################
#
# now read other (non-metal) data
#
sink(args[['log']]);

if ( is.null(args[['reload']]) ) {

  # recombination rate

  command <- paste("pquery recomb_in_region",
      " -defaults",
      " -sql",
      " RecombTable=", args[["recombTable"]],
      " Chr=",args[["chr"]],
      " Start=",args[["start"]],
      " End=",args[["end"]],
      sep="");
  if ( is.null(args[['recomb']]) && ! args[['pquery']] ) { args[['showRecomb']] <- FALSE }
  tryCatch({
    col_classes = list(
      "recomb" = "numeric",
      "cm_pos" = "numeric"
    );
    recrate <- GetData( args[['recomb']], default=recrate.default, command=command, clobber=!userFile[['recomb']] || args[['clobber']], colClasses=col_classes)
  }, error = function(e) { warning(e) }
  )

  if ( prod(dim(recrate)) == 0 ) { args[['showRecomb']] <- FALSE }
  cat("\n\n");
 

  # snpset positions

  command <- paste("pquery snpset_in_region",
      " -defaults",
      " -sql",
      ' "SnpSet=',args[["snpset"]],'"',
      " Chr=",args[["chr"]],
      " ChrStart=",args[["start"]],
      " ChrEnd=",args[["end"]],
      sep="");
  rug <- GetData( args[['snpsetFile']], default=rug.default, command=command, 
    clobber=!userFile[['snpsetFile']] || args[['clobber']] );

  cat("\n\nsnpset summary:\n");
  print(summary(rug));
  cat("\n\n");

  # annotation
  if ( char2Rname(args[['annotCol']]) %in% names(metal) ) {  
    if (is.null(args[['annotOrder']])) {
      args[['annotOrder']] <- 
        sort( unique( metal[,char2Rname(args[['annotCol']])] ) );
    }

    metal$annot <- MakeFactor(metal[,char2Rname(args[['annotCol']]) ], levels=args[['annotOrder']],
            na.level='none')
    pchVals <- rep(args[['annotPch']], length=length(levels(metal$annot)));
    metal$pch <- pchVals[ as.numeric(metal$annot) ]
    annot <- metal$annot
  } 

  cat("\nR-DEBUG: Loading annotation data...\n");
  if( args[['showAnnot']] && ! 'pch'  %in% names(metal) ) { 
    command <- paste("pquery snp_annot_in_region",
          " -defaults",
          " -sql",
          " Chr=",args[["chr"]],
          " Start=",args[["startBP"]],
          " End=",args[["endBP"]],
          sep="");
    if ( is.null(args[['annot']]) && !args[['pquery']] ) { args[['showAnnot']] <- FALSE }
    annot <- GetData( args[['annot']], annot.default, command=command, 
      clobber=!userFile[['annot']] || args[['clobber']] )
    if (prod(dim(annot)) == 0) { args[['showAnnot']] <- FALSE }
    cat("\nR-DEBUG: Merging in annotation data...");
    metal <- merge(metal, annot,  
      by.x='MarkerName', by.y="snp",
      all.x=TRUE, all.y=FALSE);
    cat(" Done.\n");
    print(head(metal));

    metal$annot <- c('no annotation','framestop','splice','nonsyn','coding','utr','tfbscons','mcs44placental')[1+metal$annot_rank];
    if ( is.null(args[['annotOrder']]) ) {
      args[['annotOrder']] <- c('framestop','splice','nonsyn','coding','utr','tfbscons','mcs44placental','no annotation')
    } 
    metal$annot <- MakeFactor(metal$annot, levels=args[['annotOrder']],na.level='none') 
    pchVals <- rep(args[['annotPch']], length=length(levels(metal$annot)));
    metal$pch <- pchVals[ as.numeric(metal$annot) ]

  }  else {

    if (! 'pch' %in% names(metal)) {
      metal$pch <- 21;
    }

    if (! 'annot' %in% names(metal) ) {
        metal$annot <- "none"
        metal$annot <- factor(metal$annot)
    }
    annot <- data.frame();
  }

  if (FALSE) {  # scraps from above
    cat('else: ');
      pchVals <- rep(args[['annotPch']], length=length(levels(metal$annot)));
      metal$pch <- pchVals[ as.numeric(metal$annot) ]
      annot <- metal$annot
      print(xtabs(~annot+pch,metal));
      print(metal[1:4,])
  }


  sink('annotationTally.txt')
  print( args[['annotOrder']] )
  print(args[['annotPch']])
  print(args[['annotOrder']])
  print(table(metal$annot))
  print(table(metal$pch))
  print(xtabs(~annot+pch,metal))
  sink()
  # ld

  command <- paste("pquery ld_in_region",
      " -defaults",
      " -sql",
      " LDTable=", args[["ldTable"]],
      " Chr=",args[["chr"]],
      " Start=",args[["startBP"]],
      " End=",args[["endBP"]],
      sep="");
      
  if ( is.null(args[['ld']]) && ! args[['pquery']] ) { 
    args[['legend']] = 'none' 
  }

  if (char2Rname(args[['colorCol']]) %in% names(metal)) {
    args[['legend']] = 'none';
  }

  # Load LD for reference SNP. 
  ld_col_classes = list(
    "dprime" = "numeric",
    "rsquare" = "numeric"
  );
  ld <- GetData( args[['ld']], ld.default, command=command, clobber=!userFile[['ld']] || args[['clobber']], colClasses=ld_col_classes )
  
  cond_ld = NULL;
  for (ld_file in args[['cond_ld']]) {
    cond_ld = rbind(cond_ld,read.table(ld_file,header=T,sep="",comment.char="",stringsAsFactors=F,colClasses=ld_col_classes));
  }
  
  cat("\n\n");

  if (! is.null(args[['metalRug']]) ) {
    metalRug <- data.frame(pos=metal$pos, snp_set=args[['metalRug']]);
    origRug <- data.frame(pos=rug$pos,snp_set=rug$snp_set)
    rug <- rbind(origRug,metalRug)
    print(levels(rug))
  }
    
  save(metal,annot,recrate,ld,args,rug,file='loaded.Rdata');

  if ( prod(dim(metal) ) < 1) { stop("No data read.\n"); }

  # Subset the data to the plotting region. 
  s <- metal$pos >= args[['startBP']] & metal$pos <= args[['endBP']] & metal$chr == args[['chr']] ;
  metal <- subset(metal, s);

  # merge LD info into metal data frame
  refSnp <- as.character(args[['refsnp']]);

  metal$group <- 1;
  metal$LD <- NA;
  metal$ldcut <- NA;
  metal$group[metal$MarkerName == refSnp] <- length(args[['ldColors']]);
  
  if (!is.null(ld)) {
    # subset ld for reference SNP
    snpCols <- which(apply(ld,2,Sniff,type="snp"))
    if (length(snpCols) != 2) {
      warning(paste("LD file doesn't smell right. (",length(snpCols)," SNP cols)",sep=""))
      assign("warningMessages",c(warningMessages,"LD file doesn't smell right."), globalenv());
      break;
    }
    
    w1 <- which ( ld[,snpCols[1]] == refSnp );
    w2 <- which ( ld[,snpCols[2]] == refSnp );
    c1 <- c(names(ld)[snpCols[1]],names(ld)[snpCols[2]],args[['ldCol']]); # "rsquare","dprime");
    c2 <- c(names(ld)[snpCols[2]],names(ld)[snpCols[1]],args[['ldCol']]); # "rsquare","dprime");
    ld1 <- ld[ w1, c1, drop=FALSE ]
    ld2 <- ld[ w2, c2, drop=FALSE ]
    names(ld1)[1:2] <- c("refSNP","otherSNP")
    names(ld2)[1:2] <- c("refSNP","otherSNP")
    lld <- rbind( ld1, ld2);
    
    if (prod(dim(lld)) > 0) { 
      metal <- merge(metal, lld,  
        by.x='MarkerName', by.y="otherSNP",
        all.x=TRUE, all.y=FALSE
      );
      
      if ( args[['ldCol']] %in% names(metal) ) {
        metal$LD <- metal[ ,args[['ldCol']] ];
      } else {
        stop(paste('No column named',args[['ldCol']]));
      }
      
      metal$ldcut <- cut(metal$LD,breaks=args[['ldCuts']],include.lowest=TRUE);
      metal$group <- 1 + as.numeric(metal$ldcut);
      metal$group[is.na(metal$group)] <- 1;
      metal$group[metal$MarkerName == refSnp] <- length(args[['ldColors']]) 
    } else {
      assign("warningMessages",c(warningMessages,'No usable LD information for reference SNP.'), globalenv());
      warning("No usable LD information.");
      args[['legend']] <- 'none';
    }
  }
  
  if (!is.null(cond_ld)) {
    # Pool all LD information together. 
    all_ld = rbind(ld,cond_ld);
    
    # Get the reference + conditional SNPs from the LD files. 
    ref_cond_snps_names = c(
      args[['refsnpName']],
      unlist(strsplit(args[['cond_snps']],","))
    );
    
    #ref_cond_snps = as.character(unique(all_ld$snp2));
    ref_cond_snps = c(
      args[['refsnp']],
      unlist(strsplit(args[['cond_pos']],","))
    );
    
    # For each SNP, find the (reference SNP, conditional SNP) with the highest LD value. 
    by_best_ld = by(all_ld,all_ld$snp1,function(x) { ind = which(x$rsquare == max(x$rsquare)); x[ind,]; })
    best_ld = Reduce(rbind,by_best_ld);
    best_ld = best_ld[,c("snp1","snp2",args[['ldCol']])];
    names(best_ld) = c("snp","best_ld_snp","best_ld");
    
    # Merge into metal. 
    metal = merge(metal,best_ld,by.x="MarkerName",by.y="snp",all.x=TRUE,all.y=FALSE);
    
    # Ref/conditional SNPs are in "best LD" with themselves. 
    metal[match(ref_cond_snps,metal$MarkerName),]$best_ld_snp = ref_cond_snps;
    metal[match(ref_cond_snps,metal$MarkerName),]$best_ld = 1;
    
    # Did the user choose to threshold LD? 
    threshold_cuts = function(ld_cuts,thresh) { 
      ld_cuts = as.numeric(ld_cuts);
      ld_cuts = ld_cuts[!ld_cuts <= thresh];
      c(0,thresh,ld_cuts);
    }
    
    if (!is.null(args[['ldThresh']])) {
      args[['ldCuts']] = threshold_cuts(args[['ldCuts']],args[['ldThresh']]);
    }
    
    # Break up best LD into bins. 
    num_ld_bins = length(args[['ldCuts']]) - 1;
    metal$best_ld_cut = cut(metal$best_ld,breaks=args[['ldCuts']],include.lowest=TRUE);
    metal$best_ld_cut = 1 + as.numeric(metal$best_ld_cut);
    metal$best_ld_cut[is.na(metal$best_ld_cut)] = 1;
   
    # Each SNP belongs to a group, which is the SNP it has highest LD with. 
    metal$best_ld_group = as.numeric(factor(metal$best_ld_snp,levels=ref_cond_snps)) + 1;
    metal$best_ld_group[is.na(metal$best_ld_group)] = 1;
        
    # Compute each SNP's color. 
    metal$ld_color_base = args[['condLdColors']][metal$best_ld_group];
    
    col_pick = function(x,num_bins,low_color=NULL) { 
      cols = c(
        args[['condLdColors']][1],
        tail(colorRampPalette(c("white",x))(num_bins + 1),-1) 
      );

      if (!is.null(low_color)) {
        cols[2] = low_color; # cols[1] is missing LD, cols[2] is lowest LD bin
      }

      cols;
    };
    
    # Assign colors to SNPs based on their best LD.  
    metal$ld_color = apply(metal,1,function(x) { 
      this_cut = as.numeric(x['best_ld_cut']);
      col_pick(x['ld_color_base'],num_ld_bins,args[['condLdLow']])[this_cut];
    });
    
    # Collect colors together for LD ribbon legend. 
    base_colors = args[['condLdColors']][seq(2,length(ref_cond_snps)+1)]
    cond_ld_colors = sapply(base_colors,function(x) tail(col_pick(x,num_ld_bins,args[['condLdLow']]),-1),simplify=F);
    names(cond_ld_colors) = ref_cond_snps_names;

    # Color the ref/cond SNPs a different color so they stand out. 
    #metal$ld_color[metal$MarkerName %in% ref_cond_snps] = tail(args[['ldColors']],1);

    # Plotting symbols are based on the group, not the annotation.
    metal$pch = args[['condPch']][metal$best_ld_group];
    
    # Ref/cond SNPs have their own symbol if set by user. 
    if (!is.null(args[['condRefsnpPch']])) {
      metal$pch[metal$MarkerName %in% ref_cond_snps] = args[['condRefsnpPch']];
    }
  }
  
  save_objs = c('metal','refSnp','args');
  if (!is.null(cond_ld)) {
    save_objs = c(save_objs,'ref_cond_snps','ref_cond_snps_names','all_ld','num_ld_bins','col_pick','base_colors','cond_ld_colors');
  }
  save(list=save_objs,file='temp.Rdata');

  command <- paste("pquery refFlat_in_region",
      " -defaults",
      " -sql",
      " Chrom=", chr2chrom(args[["chr"]]),  
      " Start=",args[["start"]],
      " End=",args[["end"]],
      " Build=",args[["build"]],
      sep="");
  if (is.null(args[['refFlat']]) && ! args[['pquery']]) { args[['showGenes']] <- FALSE }
  refFlatRaw <- GetData( args[['refFlat']], refFlatRaw.default, command=command, 
    clobber = !userFile[['refFlat']] || args[['clobber']] );

  summary(refFlatRaw);

  # subset the refFlatdata
  s <- refFlatRaw$txEnd >= args[['startBP']] & 
     refFlatRaw$txStart <= args[['endBP']] & 
     refFlatRaw$chrom == chr2chrom(args[['chr']]
  );
  refFlatRaw <- subset(refFlatRaw, s);
  save(refFlatRaw,args,file="refFlatRaw.Rdata");

  refFlat <- flatten.bed(refFlatRaw,multiplier=1/args[['unit']]);
  summary(refFlat);

  # load fine mapping data
  fmregions = NULL;
  if (!is.null(args[['fineMap']])) {
    fmregions = LoadFineMap(args[['fineMap']]);
  }

  # subset fine mapping regions to plotting region
  if (!is.null(fmregions)) {
    fmregions = subset(fmregions,chr == args[['chr']]);
  }
  summary(fmregions);

  # load gwas hits
  gwas_hits = NULL;
  if (!is.null(args[['gwasHits']])) {
    gwas_hits = LoadGWASHits(args[['gwasHits']]);
  }

  barplot_data = NULL;
  if (!is.null(args[['barplotData']])) {
    barplot_data = LoadBarplotData(args[['barplotData']])
  }
  
  # subset to chrom
  if (!is.null(barplot_data)) {
    barplot_data = subset(barplot_data,chr == args[['chr']]);
  }

  # subset gwas hits to plotting region
  if (!is.null(gwas_hits)) {
    b_gwas = (gwas_hits$chr == args[['chr']]) & (gwas_hits$pos <= args[['endBP']] / 1E6) & (gwas_hits$pos >= args[['startBP']] / 1E6);
    gwas_hits = subset(gwas_hits,b_gwas);
  }
  summary(gwas_hits);
  
  # load marker denote, if available
  denote_markers = NULL;
  try({
    if (!is.null(args[['denoteMarkersFile']])) {
      denote_markers_file = args[['denoteMarkersFile']];
      if (!file.exists(denote_markers_file)) {
        # try directory above
        denote_markers_file = file.path("..",denote_markers_file);
        if (!file.exists(denote_markers_file)) {
          warning("could not find denote_markers file..");
        }
      }
        
      denote_markers = read.table(denote_markers_file,header=T,sep="\t",comment.char="",stringsAsFactors=F);
      
      b_denote = (denote_markers$chr == args[['chr']]) & (denote_markers$pos <= args[['endBP']]) & (denote_markers$pos >= args[['startBP']]);
      denote_markers = denote_markers[b_denote,];
      denote_markers$pos = denote_markers$pos / 1E6;
      
      if (!"color" %in% names(denote_markers)) {
        denote_markers$color = "black";
      }
    }
  },silent=T);

  # load BED file, if available
  bed_tracks = NULL;
  try({
    if (!is.null(args[['bedTracks']])) {

      bed_file = args[['bedTracks']];
      if (!file.exists(bed_file)) {
        # try directory above
        bed_file = file.path("..",bed_file);
        if (!file.exists(bed_file)) {
          warning("could not find bedTracks file..");
          stop();
        }
      }

      bed_tracks = read.table(bed_file,header=F,sep="\t",comment.char="",stringsAsFactors=F);
      header = c("chr","start","end","name","score","strand","thickStart","thickEnd","itemRgb");
      names(bed_tracks) = header[1:dim(bed_tracks)[2]];

      bed_tracks$start = bed_tracks$start / 1E6;
      bed_tracks$end = bed_tracks$end / 1E6;
      
      # subset to region (look for overlap of regions)
      sub_chrom = args[['chr']];
      if (any(grepl('chr',bed_tracks$chr))) {
        sub_chrom = sprintf("chr%s",sub_chrom);
      }
      
      bed_tracks = subset(bed_tracks,(chr == sub_chrom));
      # Removed to preserve ordering of tracks in file
      #bed_tracks = bed_tracks[order(bed_tracks$start),];
      bed_tracks = subset(bed_tracks,
        (start <= args[['endBP']] / 1E6) & 
        (end >= args[['startBP']] / 1E6)
      );
      
    }
  },silent=T);

  # adjust for position units
  metal$pos <- metal$pos / args[['unit']];
  recrate$pos <- recrate$pos / args[['unit']];
  rug$pos <- rug$pos / args[['unit']];

  cat("recrate summary:\n");
  print(summary(recrate));
  cat("\n\n");
  cat("LD summary:\n");
  print(summary(ld));
  cat("\n\n");
  cat("metal summary:\n");
  print(summary(metal));
  cat("\n\n");
  save(metal,annot,recrate,refFlatRaw,refFlat,rug,fmregions,gwas_hits,file=args[['rdata']]);
} else {
  load(args[['rdata']]);
}

if (is.character(args[['prelude']]) && file.exists(args[['prelude']])) {
  source(args[['prelude']]);
}

if ( prod(dim(rug)) == 0 || !("snp_set" %in% names(rug)) ) {
  nrugs <- 0;
} else {
  nrugs <- length(levels(rug$snp_set));
}

xRange <- range(metal$pos,na.rm=T);
xRange <- as.numeric(c(args[['start']],args[['end']])) / args[['unit']];
refFlat <- refFlat[ which( (refFlat$start <= xRange[2]) & (refFlat$stop >= xRange[1]) ), ]
yRange <- c(min(c(args[['ymin']],transformation(metal$P.value),na.rm=T)),
            max(c(args[['ymax']],transformation(metal$P.value)*1.1),na.rm=T));

recrateRange <- c(0,max(c(100,recrate$recomb),na.rm=T));
if (args[['experimental']]) { 
  recrate$recomb <- max(c(100,recrate$recomb),na.rm=T) - recrate$recomb;
  recrateRange <- c(0,max(c(100,recrate$recomb),na.rm=T));
}
recrateRange <- rev(recrateRange);
print("recrateRange: ");
print(recrateRange);

refSnp <- as.character(args[['refsnp']]);
refidx <- match(refSnp, metal$MarkerName);
if (!args[['showRefsnpAnnot']]) {
  metal$pch[refidx] <- 23;  # use a diamond for ref snp
}

if ('pdf' %in% args[['format']]) {
  pdf(file=args[['pdf']],width=args[['width']],height=args[['height']],version='1.4');
  
  if ( prod(dim(metal)) == 0 ) { 
    message ('No data to plot.'); 
  } else {
    zplot(metal,ld,recrate,refidx,nrugs=nrugs,args=args,postlude=args[['postlude']]);
    grid.newpage();
  }
  
  # Arguments, annotation key, website link. Even if a plot wasn't created. 
  grid.log(args,metal);
  
  # Write a page for the legend if conditional SNPs were specified. 
  if (!is.null(cond_ld)) {
    grid.newpage();
    rl <- multiRibbonLegend(
      ref_cond_snps_names,
      args[['ldCuts']],
      cond_ld_colors,
      gp = gpar(cex = 0.8)
    );
    pushViewport(viewport(
      w = 0.35,
      h = 0.7,
      layout = grid.layout(2,1)
    ));

    pushViewport(viewport(layout.pos.row=2,layout.pos.col=1));
    grid.draw(rl);
    popViewport();

    pushViewport(viewport(layout.pos.row=1,layout.pos.col=1));

    cond_key = simpleKey(rev(ref_cond_snps_names));
    cond_key$cex = 1;
    cond_key$points$cex = 1.25;
    cond_key$points$col = rev(base_colors);
    cond_key$points$fill = rev(base_colors);
    cond_key$points$pch = rev(args[['condPch']][1:length(ref_cond_snps_names) + 1]);
    grid.draw(draw.key(cond_key));

    popViewport();

    popViewport();
  }

  # Write log pages for GWAS catalog hits within the plotting region (if provided.) 
  if (!is.null(gwas_hits) & all(dim(gwas_hits) > 0)) {
    gwas_hits = gwas_hits[order(gwas_hits$pos),];
    gwas_hits = change_names(gwas_hits, list('pos' = "pos (Mb)"));
    grid.extralog(gwas_hits,main = "GWAS Catalog SNPs in Region");
  }
  
  # Write log pages for fine-mapping data. 
  if (!is.null(fmregions) & all(dim(fmregions) > 0)) {
    fmregions = fmregions[order(fmregions$pos),];
    fmregions = change_names(fmregions, list('pos' = "pos (Mb)"));
    grid.extralog(fmregions,main = "Fine-mapping Regions");
  }
  
  # Write log pages for data regarding the reference and conditional SNPs. 
  # if (exists("ref_cond_snps")) {
    # metal_isnps = subset(metal,MarkerName %in% ref_cond_snps);
    
    # metal_isnps$ref_or_cond = NA;
    # metal_isnps[metal_isnps$MarkerName %in% ref_cond_snps,'ref_or_cond'] = "Conditional SNP";
    # metal_isnps[metal_isnps$MarkerName == refSnp,'ref_or_cond'] = "Reference SNP";
    
    # metal_isnps = metal_isnps[,c("MarkerName","chr","pos","P.value","ref_or_cond")];
    # grid.extralog(metal_isnps,main = "Reference and Conditional SNPs");
  # }
  
  dev.off();
} 

if ('svg' %in% args[['format']]) {
  svg(file=args[['svg']],width=args[['width']],height=args[['height']]);
  
  if ( prod(dim(metal)) == 0 ) { 
    message ('No data to plot.'); 
  } else {
    zplot(metal,ld,recrate,refidx,nrugs=nrugs,args=args,postlude=args[['postlude']]);
  }
  
  dev.off();
} 

#
# N.B. *** old png and tiff code no longer being maintained.  No guarantees that this works anymore. ***
#
if ('png' %in% args[['format']]) {
    args[['recombLineAlpha']] = 1;
    args[['recombFillAlpha']] = 1;
    args[['hiliteAlpha']] = 1;
    args[['frameAlpha']]=1;
    args[['hiAlpha']]=1;
    args[['rugAlpha']] = 1;
    args[['refsnpLineAlpha']] = 1; 
      args[['refsnpTextAlpha']]=1;
    png(file=args[['png']],
        width=args[['width']]*100,
        height=args[['height']]*100);
    if ( prod(dim(metal)) == 0 ) { 
        message ('No data to plot.'); 
    } else {
        assign("args",args,globalenv());
        zplot(metal,ld,recrate,refidx,nrugs=nrugs,args=args,postlude=args[['postlude']]);
    }
    dev.off();
}

#
# N.B. *** old png and tiff code no longer being maintained.  No guarantees that this works anymore. ***
#
if ('tiff' %in% args[['format']]) {
    args[['recombLineAlpha']] = 1;
    args[['recombFillAlpha']] = 1;
    args[['hiliteAlpha']] = 1;
    args[['frameAlpha']]=1;
    args[['hiAlpha']]=1;
    args[['rugAlpha']] = 1;
    args[['refsnpLineAlpha']] = 1; 
      args[['refsnpTextAlpha']]=1;
    tiff(file=args[['tiff']],
        width=args[['width']]*100,
        height=args[['height']]*100);
    if ( prod(dim(metal)) == 0 ) { 
        message ('No data to plot.'); 
    } else {
        assign("args",args,globalenv());
        zplot(metal,ld,recrate,refidx,nrugs=nrugs,args=args,postlude=args[['postlude']]);
    }
    dev.off();
}

sink(args[['log']], append=TRUE);
  grid.log(args,metal,ascii=TRUE);
  cat('\n\n\n');
  cat("List of genes in region\n");
    cat("#######################\n");
  geneList <- make.gene.list(refFlat,unit=args[['unit']]);
  if (! is.null(geneList)) {
    digits <- 7 + ceiling(log10(max(geneList$stop)));
    print(geneList,digits=digits);
  }
  cat('\n\n\n');
sink();

save(metal,refFlat,ld,recrate,refSnpPos,barplot_data,fmregions,gwas_hits,bed_tracks,args,file='end.Rdata')
CleanUp(args,refSnpPos,recrate,rug,ld,refFlatRaw);

date();
