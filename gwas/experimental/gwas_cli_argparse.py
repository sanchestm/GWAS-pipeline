#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import re
from collections import defaultdict

try:
    from gwas.core import *
except ImportError as e:
    print('Failed to import gwas.gwas; trying local import...')
    try:
        from . import core as gg
    except ImportError:
        raise ImportError("Could not import `gwas.core`. Make sure your package is installed or the structure is correct.") from e
import sys
import pandas as pd


 def typeconverter(s):
    s= str(s)
    if s.lower() in ['1', 'true']: return 1
    if s.lower() in ['0', 'false']: return 0
    try: return int(s)
    except: pass
    if s[-2:] == '()':
        try: return eval(np.random.choice( re.split(r'(\s|\,|\]|\[)', s)))
        except: pass
    try: return float(s)
    except: return s

def parse_extra_args(arg_list):
    return {
        k: typeconverter(v)
        for kv in arg_list
        if '=' in kv
        for k, v in [kv.split('=', 1)]
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run the GWAS pipeline with subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(p):
        p.add_argument("--path", default="", help="Base path to project")
        p.add_argument("--project", default="test", help="Project name")
        p.add_argument("--genotypes", default="/tscc/projects/ps-palmer/gwas/databases/rounds/r10.2.1")
        p.add_argument("--traits")
        p.add_argument("--regressout")
        p.add_argument("--founder_genotypes", default="none")
        p.add_argument("--phewas_path", default="phewasdb.parquet.gz")
        p.add_argument("--genome_accession", default="GCF_015227675.2")
        p.add_argument("--researcher", default="tsanches")
        p.add_argument("--round", default="10.1.0")
        p.add_argument("--gwas_version", default="0.1.2")
        p.add_argument("--threshold", type=float, default=5.39)
        p.add_argument("--threshold05", type=float, default=5.64)
        p.add_argument("--threads", type=int, default=4)

    runall = subparsers.add_parser("runall", help="Run the full GWAS pipeline")
    add_shared_arguments(runall)
    runall.add_argument("--clear_directories", type=typeconverter, default=0)
    runall.add_argument("--impute", type=typeconverter, default=0)
    runall.add_argument("--gwas", type=typeconverter, default=1)
    runall.add_argument("--h2", type=typeconverter, default=1)
    runall.add_argument("--db", type=typeconverter, default=1)
    runall.add_argument("--qtl", type=typeconverter, default=1)
    runall.add_argument("--gcorr", type=typeconverter, default=1)
    runall.add_argument("--eqtl", type=typeconverter, default=1)
    runall.add_argument("--sqtl", type=typeconverter, default=1)
    runall.add_argument("--goea", type=typeconverter, default=1)
    runall.add_argument("--locuszoom", type=typeconverter, default=1)
    runall.add_argument("--report", type=typeconverter, default=1)
    runall.add_argument("--store", type=typeconverter, default=1)
    runall.add_argument("--publish", type=typeconverter, default=1)
    runall.add_argument("--phewas", type=typeconverter, default=1)
    runall.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional keyword arguments")

    for step in ["regressout", "gwas", "report", "store", "phewas", "locuszoom"]:
        p = subparsers.add_parser(step, help=f"Run only the '{step}' step")
        add_shared_arguments(p)
        p.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional keyword arguments")
        p.set_defaults(step=step)

    return parser.parse_args()


def main():
    args = parse_args()

    path = args.path.rstrip("/") + "/" if args.path else ""
    pj = args.project.rstrip("/")
    printwithlog(f"Running '{args.command}' for project '{pj}' at path '{path}'")

    if args.regressout:
        rawdata = args.regressout if len(args.regressout) > 1 else f"{path}{pj}/raw_data.csv"
        df = pd.read_csv(rawdata, dtype={"rfid": str}).drop_duplicates(subset="rfid")
        traits_, traits_d = [], []
    else:
        df = pd.read_csv(f"{path}{pj}/processed_data_ready.csv", dtype={"rfid": str}).drop_duplicates(subset="rfid")
        if not args.traits:
            traits_ = df.columns[df.columns.str.startswith("regressedlr_")]
        elif 'prefix_' in args.traits:
            pref = args.traits.replace("prefix_", "")
            traits_ = df.columns[df.columns.str.startswith(f"regressedlr_{pref}")]
        else:
            traits_ = args.traits.split(',')
        try:
            traits_d = get_trait_descriptions_f(pd.read_csv(f"{path}{pj}/data_dict_{pj}.csv"), traits_)
        except:
            traits_d = ['UNK' for _ in traits_]

    gwas = gwas_pipe(
        path=f"{path}{pj}/",
        all_genotypes=args.genotypes,
        data=df,
        project_name=pj.split('/')[-1],
        traits=traits_,
        genome_accession=args.genome_accession,
        founderfile=args.founder_genotypes,
        phewas_db=args.phewas_path,
        trait_descriptions=traits_d,
        threshold=args.threshold,
        threshold05=args.threshold05,
        threads=args.threads,
    )

    extra_kwargs = parse_extra_args(getattr(args, "extra_args", []))

    if args.command == "runall":
        if args.clear_directories:
            gwas.clear_directories()
        if args.impute:
            gwas.impute_traits()
        if args.gwas:
            gwas.fastGWAS()
        if args.h2:
            gwas.snpHeritability()
        if args.db:
            gwas.addGWASresultsToDb(researcher=args.researcher, round_version=args.round, gwas_version=args.gwas_version)
        if args.qtl:
            try:
                qtls = gwas.callQTLs(NonStrictSearchDir=False, add_founder_genotypes=args.founder_genotypes not in ['none', 'None', 0])
            except:
                qtls = gwas.callQTLs(NonStrictSearchDir=True)
            gwas.effectsize()
        if args.gcorr:
            gwas.genetic_correlation_matrix()
            gwas.make_heritability_figure()
        if args.eqtl:
            gwas.eQTL(annotate=True)
        if args.sqtl:
            gwas.sQTL(annotate=True)
        if args.goea:
            gwas.GeneEnrichment()
        if args.locuszoom:
            gwas.locuszoom(**extra_kwargs)
        if args.report:
            gwas.report(round_version=args.round, gwas_version=args.gwas_version)
            gwas.copy_results()
        if args.store:
            gwas.store(researcher=args.researcher, round_version=args.round, gwas_version=args.gwas_version, remove_folders=False)
        if args.publish:
            try:
                gwas.copy_results()
            except:
                print("[WARN] Could not publish; ensure MinIO or export setup is complete.")
        if args.phewas:
            gwas.phewas(annotate=True, pval_threshold=1e-4, nreturn=1, r2_thresh=0.65)

    elif args.command == "regressout":
        gwas.regressout(**extra_kwargs)
    elif args.command == "gwas":
        gwas.fastGWAS(**extra_kwargs)
    elif args.command == "report":
        gwas.report(round_version=args.round, gwas_version=args.gwas_version, **extra_kwargs)
        gwas.copy_results()
    elif args.command == "store":
        gwas.store(researcher=args.researcher, round_version=args.round, gwas_version=args.gwas_version, remove_folders=False, **extra_kwargs)
    elif args.command == "phewas":
        gwas.phewas(annotate=True, pval_threshold=1e-4, nreturn=1, r2_thresh=0.65, **extra_kwargs)
    elif args.command == "locuszoom":
        gwas.locuszoom(**extra_kwargs)

    gwas.print_watermark()


if __name__ == "__main__":
    main()
