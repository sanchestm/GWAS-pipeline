# GWAS-pipeline

This package is a *potpourri* of genetic analysis tools to:
1) perform ***GWAS*** species and uses the *NCBI datasets* to get genetic information of that species
2) perform downstream analysis to investigate further the identified ***QTLs***
3) wrap all the results into a *HTML* report that contains all the performed analyses.

## Input Files 

| File    | is_essential? | format | defaultname |
| --------| ------------ | ------ | --------- |
| Genotypes  | ✅    | (.bim,.bed,.fam) | None |
| Phenotypes | ✅    | .csv    | raw_data.csv |
| Data dictionary | ✅    | .csv | data_dict_{projectname}.csv| 
| Phewas db  | ⬜️    | .parquet.gz | phewasdb.parquet.gz |
| Founder genotypes| ⬜️ |(.bim,.bed,.fam) | None |

Currently, the pipeline is centered around PLINK for SNP data in the `.bim|.bam|.fam` format (from PLINK 1.96). The Genotype data ***has*** to have the SNPs encoded in the format `{CHR}:{POS}` and thus is currently limited to biallelic SNPs (this information is in the `.bim` file). The code also requires individual `iid` and `fid` to be the same, please check if the `.fam` file to make sure that  

Meanwhile, the phenotype data is written in the `CSV` format and requires 2 essential columns: `rfid` and `sex`. The `sex` column is encoded as 'M' for males and 'F' for females. If the species does not have sex, encode all individuals as females 'F'. Individuals with missing sex ***will be dropped***. All columns have to be in lower case.

Last, the main essential file is the data dictionary. It tells the pipeline which columns are metadata, covariates or traits and for each trait which covariates are used. There are 4 essential columns `measure`, `trait_covariate`, `covariates` and `description`. `measure` will be the columns of the phenotype data file. `trait_covariate` can be {`metadata`,`covariate_categorical`,`covariate_continuous`,`trait`}. `covariates` can be `nan`, `passthrough` or a comma-separated list covariates that are present in the phenotype file and are also a column in the data dictionary file. `description` is a longer description for traits and covariates. We build a helper function `generate_datadic` to facilitate building this file from the phenotype file. Please see the example in `example/example_chitre_obs.ipynb`
| measure        | trait_covariate       | covariates   | description                             |
|:---------------|:----------------------|:-------------|:----------------------------------------|
|rfid|metadata|nan|id of individual|
| sex            | covariate_categorical | nan          | sex                                     |
| age            | covariate_continuous  | nan          | age of individual                       |
| trait1         | trait                 | passthrough  | this trait will not have any covariate |
| trait2         | trait                 | sex,age  | this trait will be adjusted for sex and age |

## Parameters

| parameter    | what |is_essential? | defaultvalue | note |
| --------|--- |------------ | --------- |  ---- |
| threshold| significance threshold  | ✅    |  'auto' | if auto, it will calculate the threshold with 1000 normal traits, this will increase the wall time by a lot |
| threshold05| secondary higher significance threshold | ✅    | 5.643286 |
| genome_accession| NCBI genome accession | ✅    | 'GCF_015227675.2' | if run in the notebook errors will trigger a helper function to find the genome_accession, requires user input |
| threads|  number of threads | ✅ | os.cpu_count() | in a HPC please set this value, because the os.cpu_count() can differ to the resources requested |

## Installation

We use `conda` to manage the installation of all necessary packages. Given the generalist nature of this project, there are a lot of dependencies and the necessity of 2 conda environments. Please consider updating conda and making sure that the solver is `libmamba`.

```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

for downloading the GWAS-pipeline use `git clone` and then create the necessary conda environments

```
git clone https://github.com/sanchestm/GWAS-pipeline.git
cd GWAS-pipeline
conda env create -f environment.yml
conda env create -f environment_lz.yml
```

## directory management

We suggest following one of these standards for data management to make sure we 

```plaintext
path2projects/
├── projectname/
│   ├── raw_data.csv
│   ├── data_dict_projectname.csv
├── genotypes/
│   ├── geno.bim
│   ├── geno.fam
│   ├── geno.bam
└── phewasdb.parquet.gz
```
```plaintext
path2projects/
└── projectname/
   ├── raw_data.csv
   ├── data_dict_projectname.csv
   ├── geno.bim
   ├── geno.fam
   ├── geno.bam
   └── phewasdb.parquet.gz
```

## CLI

for the `CLI` version of the code we can use the `run` flag or we can list the operations that have to be performed. 

```
cd GWAS-pipeline
python gwas_cli.py\
       path=path2projects/ \
       genome_accession= \
       round=versionofgenotypes\
       founder_genotypes= \
       project=projectname\
       threads=8\
       threshold=5.4\
       phewas_path=phewasdb.parquet.gz\
       runall
```
```
cd GWAS-pipeline
python gwas_cli.py\
       path=path2projects/ \
       genome_accession= \
       round=versionofgenotypes\
       founder_genotypes= \
       project=projectname\
       threads=8\
       phewas_path=phewasdb.parquet.gz\
       clear_directories subset h2 db gwas threshold=5.4 qtl store phewas goea eqtl gcorr locuszoom sqtl report gwas_version=0.3 
```






