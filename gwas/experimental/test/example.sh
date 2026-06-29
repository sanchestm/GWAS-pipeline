#!/bin/bash 
#SBATCH -J GWAS_r01_doug_adams
#SBATCH -p hotel
#SBATCH -q hotel
#SBATCH -t 80:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 30
#SBATCH --mem-per-cpu 32
#SBATCH --export ALL
#SBATCH -o /tscc/projects/ps-palmer/gwas/projects/r01_doug_adams/oneproj_complete-%j.o
#SBATCH -e /tscc/projects/ps-palmer/gwas/projects/r01_doug_adams/oneproj_complete-%j.e  
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user tsanches@health.ucsd.edu
#SBATCH -A csd795

source activate gwas
cd /tscc/projects/ps-palmer/gwas/GWAS-pipeline
gwasVersion=$(git describe --tags)
echo $gwasVersion
python gwas_cli.py path=/tscc/projects/ps-palmer/gwas/projects/ genotypes=/tscc/projects/ps-palmer/gwas/databases/rounds/r10.1.1 n_autosome=20 phewas_path=/tscc/projects/ps-palmer/gwas/projects/phewasdb_rn7.parquet.gz round=10.1.1 genome=rn7 founder_genotypes=/tscc/projects/ps-palmer/hs_rats/Ref_panel_mRatBN7.2/Ref_panel_mRatBN7_2_chr_GT project=r01_doug_adams threads=30 threshold=5.58 regressout timeseries clear_directories subset grm h2 db gwas qtl store publish eqtl gcorr h2fig locuszoom sqtl manhattanplot porcupineplot effect report gwas_version=$gwasVersion 