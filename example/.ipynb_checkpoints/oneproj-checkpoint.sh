#!/bin/bash 
#PBS -S /bin/bash
#PBS -l walltime=15:00:00
#PBS -l nodes=1:ppn=30
#PBS -N GWAS_example
#PBS -j oe
#PBS -q hotel
#PBS -o /projects/ps-palmer/tsanches/gwaspipeline/gwas/example/$PBS_JOBNAME.out       
#PBS -e /projects/ps-palmer/tsanches/gwaspipeline/gwas/example/$PBS_JOBNAME.err    

source activate gwaspipe
cd *code_path*
python gwas_cli.py project=example threads=30 regressout=example/raw_data.csv phewas_path=/projects/ps-palmer/tsanches/gwaspipeline/gwas/phewasdb.parquet.gz subset h2 db gwas qtl phewas store BLUP eqtl gcorr locuszoom sqtl manhattanplot porcupineplot effect report