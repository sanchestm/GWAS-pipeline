#!/usr/bin/env python3
"""
pyvep - a small, pure-Python variant effect predictor for GFF3-annotated genomes.

It re-implements the subset of Ensembl VEP that a GFF-driven run actually
produces (``vep --gff genomic.gff.gz --fasta genome.fna --hgvs --per_gene ...``):

* transcript consequences (Sequence Ontology terms, VEP severity ranks and
  predicate logic, including splice donor/acceptor/region, 5th-base, donor
  region and polypyrimidine-tract terms),
* IMPACT, DISTANCE, STRAND, BIOTYPE, SYMBOL, Gene, Feature,
* cDNA/CDS/protein positions, Amino_acids and Codons (VEP capitalisation),
* HGVSc / HGVSp (3'-shifted indels, dup/ins/del/delins/inv, fs/ext/Ter),
* VEP ``--per_gene`` picking (MANE > canonical > biotype > rank > length).

Only numpy, pandas and pyarrow are required.  The GFF3 (NCBI RefSeq or
Ensembl, plain or gzipped) is parsed once and cached as Parquet next to the
GFF; the genome is read through the FASTA ``.fai`` index with ``mmap``.

Python
------
    import pyvep
    out = pyvep.annotate(df, "ncbi_dataset/data/GCF_036323735.1/genomic.gff")

``df`` needs a ``SNP`` column ("chr:pos") or ``Chr``/``bp`` columns, plus
reference/alternate allele columns (defaults ``A2``/``A1`` like GWAS-pipeline).
The FASTA is found automatically next to the GFF (or pass ``fasta=``).  The
result is the input table with one annotation row per variant, using the same
column names GWAS-pipeline's ``annotate`` produced from VEP's tab output.

``pyvep.annotate_vep(...)`` returns the raw VEP-like long table (one row per
variant x transcript) if you want everything.

Command line
------------
    python pyvep.py annotate -i variants.vcf --gff genomic.gff [--fasta genome.fna] -o out.tsv
    python pyvep.py annotate -i qtls.tsv --gff genomic.gff --legacy --distance 30000 -o out.tsv
    python pyvep.py build --gff genomic.gff          # pre-build the parquet cache

Validation
----------
Checked against Ensembl VEP 116.2 (``--gff --fasta --hgvs``) on GRCh38 chr21 with RefSeq and
Ensembl GFF3s: ~133k variants (SNVs, MNVs, indels; GWAS-like random sets and sets concentrated
on splice sites, start/stop codons and repeats), ~890k variant x transcript rows, including
held-out sets never used during development.  Consequence, IMPACT, positions, Amino_acids,
Codons, HGVSc, HGVS_OFFSET, DISTANCE, Gene, SYMBOL and BIOTYPE agree on every row; VEP
``--per_gene`` picks agree except where VEP itself breaks an exact tie at random.  Remaining
differences are deliberate (``validate_pyvep.py`` re-runs this check on any genome):

* VEP (GFF mode) reads the UTR sequence of minus-strand transcripts from the
  wrong coordinates, so its extTer/fsTer counts (and a handful of stop/start
  consequences of indels at the stop codon) are wrong there; pyvep uses the
  real UTR.  ``_EMULATE_VEP_BUGS = True`` reproduces VEP for testing.
* start-loss indels: pyvep writes ``p.Met1?``; VEP sometimes writes malformed
  strings such as ``p.MetLeu1_?2`` or ``p.Lys1_?0``.
* HGVSp synonymous changes are written ``p.Leu45=`` (VEP's tab output escapes
  ``=`` as ``%3D``).
* Mitochondrial transcripts use the vertebrate mitochondrial codon table.
* Alleles that are not plain nucleotides ('.', '*', plink's '0', missing, symbolic, REF==ALT)
  are left unannotated (``allele_check='invalid_allele'``) instead of being guessed at.
* Not implemented (VEP produces nothing for them without a cache either):
  SIFT/PolyPhen, protein domains, APPRIS/TSL from the cache, regulatory build,
  co-located variants; structural variants.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import mmap
import os
import re
import sys
import time
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pyarrow is only needed for GFF parsing and the cache
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
except ImportError:  # pragma: no cover
    pa = None

__version__ = "0.1.1"
CACHE_VERSION = 1

# Testing aid only: reproduce two Ensembl VEP 116 quirks in GFF+FASTA mode so outputs can be
# compared 1:1.  (a) UTR sequences of minus-strand transcripts are read from mirrored
# coordinates of the transcript span (affects extTer/fsTer counts and start/stop predicates);
# (b) 3'-shifted alleles on the minus strand are not rotated when the shift exceeds the allele
# length (affects HGVSp first changed residue).  Both are wrong biologically; keep False.
_EMULATE_VEP_BUGS = False

__all__ = ["annotate", "annotate_vep", "load_annotation", "build_cache", "Fasta",
           "Annotation", "bp2str", "CONSEQUENCES", "IMPACT"]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
#: Sequence Ontology consequence terms in VEP severity order with their IMPACT
CONSEQUENCES: List[Tuple[str, str]] = [
    ("transcript_ablation", "HIGH"), ("splice_acceptor_variant", "HIGH"),
    ("splice_donor_variant", "HIGH"), ("stop_gained", "HIGH"), ("frameshift_variant", "HIGH"),
    ("stop_lost", "HIGH"), ("start_lost", "HIGH"), ("transcript_amplification", "HIGH"),
    ("feature_elongation", "HIGH"), ("feature_truncation", "HIGH"),
    ("inframe_insertion", "MODERATE"), ("inframe_deletion", "MODERATE"),
    ("missense_variant", "MODERATE"), ("protein_altering_variant", "MODERATE"),
    ("splice_donor_5th_base_variant", "LOW"), ("splice_region_variant", "LOW"),
    ("splice_donor_region_variant", "LOW"), ("splice_polypyrimidine_tract_variant", "LOW"),
    ("incomplete_terminal_codon_variant", "LOW"), ("start_retained_variant", "LOW"),
    ("stop_retained_variant", "LOW"), ("synonymous_variant", "LOW"),
    ("coding_sequence_variant", "MODIFIER"), ("mature_miRNA_variant", "MODIFIER"),
    ("5_prime_UTR_variant", "MODIFIER"), ("3_prime_UTR_variant", "MODIFIER"),
    ("non_coding_transcript_exon_variant", "MODIFIER"), ("intron_variant", "MODIFIER"),
    ("NMD_transcript_variant", "MODIFIER"), ("non_coding_transcript_variant", "MODIFIER"),
    ("coding_transcript_variant", "MODIFIER"), ("upstream_gene_variant", "MODIFIER"),
    ("downstream_gene_variant", "MODIFIER"), ("TFBS_ablation", "MODIFIER"),
    ("TFBS_amplification", "MODIFIER"), ("TF_binding_site_variant", "MODIFIER"),
    ("regulatory_region_ablation", "MODIFIER"), ("regulatory_region_amplification", "MODIFIER"),
    ("regulatory_region_variant", "MODIFIER"), ("intergenic_variant", "MODIFIER"),
    ("sequence_variant", "MODIFIER"),
]
RANK = {t: i + 1 for i, (t, _) in enumerate(CONSEQUENCES)}
IMPACT = dict(CONSEQUENCES)
IMPACT_ORDER = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MODIFIER": 3}

_BASES = "TCAG"
_AA_STD = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLES = {1: {a + b + c: _AA_STD[16 * i + 4 * j + k]
                    for i, a in enumerate(_BASES) for j, b in enumerate(_BASES)
                    for k, c in enumerate(_BASES)}}
CODON_TABLES[2] = dict(CODON_TABLES[1], AGA="*", AGG="*", ATA="M", TGA="W")  # vertebrate mito
AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
       "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
       "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter",
       "X": "Xaa", "U": "Sec", "O": "Pyl", "B": "Asx", "Z": "Glx", "J": "Xle"}
_COMP = str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv", "TGCANtgcanYRMKSWVHDByrmkswvhdb")

# GFF feature types VEP reads from a GFF (anything else is ignored, as in VEP)
GFF_INCLUDE_TYPES = {
    "aberrant_processed_transcript", "CDS", "C_gene_segment", "D_gene_segment", "exon", "gene",
    "J_gene_segment", "lnc_RNA", "lincRNA", "lincRNA_gene", "miRNA", "miRNA_gene", "mt_gene",
    "ncRNA_gene", "nc_primary_transcript", "NMD_transcript_variant", "processed_pseudogene",
    "processed_transcript", "protein_coding_gene", "pseudogene", "pseudogenic_transcript",
    "RNA", "rRNA", "rRNA_gene", "snoRNA", "snoRNA_gene", "snRNA", "snRNA_gene", "supercontig",
    "transcript", "VD_gene_segment", "V_gene_segment", "mRNA", "ncRNA", "primary_transcript",
    "tRNA", "stop_codon",
}
#: GFF types reported in the extra ``regulatory_feature`` column (not used by VEP --gff)
REGULATORY_TYPES = {
    "enhancer", "silencer", "promoter", "insulator", "locus_control_region", "TATA_box",
    "CAAT_signal", "GC_rich_promoter_region", "protein_binding_site", "response_element",
    "transcriptional_cis_regulatory_region", "regulatory_region", "DNaseI_hypersensitive_site",
    "enhancer_blocking_element", "epigenetically_modified_region", "matrix_attachment_site",
    "open_chromatin_region", "TF_binding_site", "CTCF_binding_site", "imprinting_control_region",
    "replication_regulatory_region", "sequence_alteration_artifact",
}


def bp2str(num) -> str:
    """Human readable distance (GWAS-pipeline's ``bp2str`` format: 9bp, 392bp, 14.66Kb, 1.23Mb).

    Same output as the original lambda except that it no longer strips zeros from whole
    numbers (the original turned 900 into '9bp' and 10 into '1bp').
    """
    num = int(num)
    if num > 9:
        s = f"{abs(num):,}"
        m = re.findall(r"^\d*\,?\d{2}", s)[0].replace(",", ".")
        if "." in m:
            m = m.rstrip("0").rstrip(".")
        return m + ["bp", "Kb", "Mb", "Gb"][s.count(",")]
    return f"{num}bp"


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _overlap(s1, e1, s2, e2) -> bool:
    return e1 >= s2 and s1 <= e2


def _seq3(pep: str) -> str:
    return "".join(AA3.get(a, "Xaa") for a in pep)


_UNAMBIG = re.compile(r"^[ACGT\-]+$", re.I)


# --------------------------------------------------------------------------------------
# FASTA (mmap + .fai, no external dependency)
# --------------------------------------------------------------------------------------
class Fasta:
    """Random access to an uncompressed, ``.fai``-indexed FASTA via ``mmap``.

    The index is created (pure Python) if it does not exist.  bgzipped FASTA is
    supported only if ``pysam`` is installed.
    """

    def __init__(self, path: str):
        self.path = path
        self._pysam = None
        if path.endswith((".gz", ".bgz")):
            try:
                import pysam  # noqa
                self._pysam = pysam.FastaFile(path)
                self.lengths = dict(zip(self._pysam.references, self._pysam.lengths))
                return
            except ImportError as e:  # pragma: no cover
                raise ImportError("compressed FASTA needs pysam; or decompress the .fna.gz") from e
        fai = path + ".fai"
        if not os.path.exists(fai):
            self.build_fai(path)
        self.index = {}
        with open(fai) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                self.index[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.lengths = {k: v[0] for k, v in self.index.items()}
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    @staticmethod
    def build_fai(path: str) -> str:
        """Write a samtools-compatible ``.fai`` (pure Python)."""
        out = []
        with open(path, "rb") as fh:
            name = None
            offset = length = lb = lw = 0
            pos = 0
            for line in fh:
                n = len(line)
                if line.startswith(b">"):
                    if name is not None:
                        out.append(f"{name}\t{length}\t{offset}\t{lb}\t{lw}")
                    name = line[1:].split()[0].decode()
                    offset = pos + n
                    length = lb = lw = 0
                else:
                    s = line.rstrip(b"\r\n")
                    if lb == 0:
                        lb, lw = len(s), n
                    length += len(s)
                pos += n
            if name is not None:
                out.append(f"{name}\t{length}\t{offset}\t{lb}\t{lw}")
        with open(path + ".fai", "w") as fh:
            fh.write("\n".join(out) + "\n")
        return path + ".fai"

    def __contains__(self, name):
        return name in self.lengths

    def fetch(self, name: str, start: int, end: int) -> str:
        """1-based inclusive; positions outside the sequence are returned as ``N``."""
        if end < start:
            return ""
        L = self.lengths[name]
        s, e = max(start, 1), min(end, L)
        left, right = "N" * max(0, min(1, end + 1) - start), "N" * max(0, end - max(L, start - 1))
        if s > e:
            return "N" * (end - start + 1)
        if self._pysam is not None:
            return left + self._pysam.fetch(name, s - 1, e).upper() + right
        _, off, lb, lw = self.index[name]
        s0, e0 = s - 1, e - 1
        a = off + (s0 // lb) * lw + s0 % lb
        b = off + (e0 // lb) * lw + e0 % lb + 1
        seq = self._mm[a:b].replace(b"\n", b"").replace(b"\r", b"").decode().upper()
        return left + seq + right


# --------------------------------------------------------------------------------------
# GFF -> parquet cache
# --------------------------------------------------------------------------------------
def _parse_attrs(s: str) -> dict:
    d = {}
    for kv in s.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k.strip()] = v
    return d


def _unescape(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return v
    return re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), v)


def _scan_gff(path: str, block_size: int = 1 << 25):
    """Stream a GFF3 (plain or .gz) with pyarrow and keep only what pyvep needs.

    Returns ``(features, parts)``: ``features`` holds gene/transcript/region/regulatory rows
    with their attributes; ``parts`` holds exon/CDS rows reduced to parent, coordinates,
    phase, protein_id and version (the bulky attribute strings are dropped batch by batch).
    """
    cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    ropt = pacsv.ReadOptions(column_names=cols, block_size=block_size)
    popt = pacsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False,
                              invalid_row_handler=lambda row: "skip")
    copt = pacsv.ConvertOptions(column_types={c: pa.string() for c in cols}, strings_can_be_null=False)
    keep_types = pa.array(sorted((GFF_INCLUDE_TYPES - {"exon", "CDS", "stop_codon"}) | REGULATORY_TYPES | {"region"}))
    part_types = pa.array(["exon", "CDS", "stop_codon"])
    feats, parts = [], []
    reader = pacsv.open_csv(pa.input_stream(path, compression="detect"), read_options=ropt,
                            parse_options=popt, convert_options=copt)
    for batch in reader:
        t = pa.Table.from_batches([batch])
        t = t.filter(pc.invert(pc.starts_with(t["seqid"], "#")))
        if t.num_rows == 0:
            continue
        typ = t["type"]
        f = t.filter(pc.is_in(typ, keep_types))
        if f.num_rows:
            feats.append(f.set_column(3, "start", pc.cast(f["start"], pa.int64()))
                         .set_column(4, "end", pc.cast(f["end"], pa.int64())))
        p = t.filter(pc.is_in(typ, part_types))
        if p.num_rows:
            at = p["attributes"]
            ext = lambda k: pc.struct_field(pc.extract_regex(at, r"(?:^|;)%s=(?P<v>[^;]*)" % k), [0])
            parts.append(pa.table({"type": p["type"], "parent": pc.fill_null(ext("Parent"), ""),
                                   "start": pc.cast(p["start"], pa.int64()), "end": pc.cast(p["end"], pa.int64()),
                                   "phase": p["phase"], "protein_id": ext("protein_id"), "pversion": ext("version")}))
    empty_f = pa.table({c: pa.array([], pa.int64() if c in ("start", "end") else pa.string()) for c in cols})
    empty_p = pa.table({"type": pa.array([], pa.string()), "parent": pa.array([], pa.string()),
                        "start": pa.array([], pa.int64()), "end": pa.array([], pa.int64()),
                        "phase": pa.array([], pa.string()), "protein_id": pa.array([], pa.string()),
                        "pversion": pa.array([], pa.string())})
    feats = pa.concat_tables(feats) if feats else empty_f
    parts = pa.concat_tables(parts) if parts else empty_p
    return feats, parts


def _gff_header(path: str) -> dict:
    meta = {}
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"#!genome-build-accession\s+(?:NCBI_Assembly:)?(\S+)", line)
            if m:
                meta["accession"] = m.group(1)
            m = re.match(r"#!genome-build\s+(\S+)", line)
            if m:
                meta["assembly"] = m.group(1)
            m = re.match(r"##species\s+\S+id=(\d+)", line)
            if m:
                meta["taxid"] = m.group(1)
    return meta


def _record_biotype(t_attr: dict, ttype: str, g_attr: Optional[dict]) -> Optional[str]:
    """VEP's biotype guess for GFF transcripts (AnnotationSource::File::GFF)."""
    b = t_attr.get("biotype") or t_attr.get("transcript_type") or t_attr.get("transcript_biotype")
    if b:
        return b
    lt = ttype.lower()
    if lt == "mrna":
        return "protein_coding"
    if lt == "ncrna":
        return t_attr.get("ncrna_class")
    m = re.match(r"^([a-z]+)_gene_segment$", lt)
    if m:
        return "IG_" + m.group(1).upper() + "_gene"
    if g_attr is not None and _unescape(g_attr.get("description") or "").startswith("microRNA"):
        return "miRNA"
    return t_attr.get("gbkey")


def _is_gene_type(t: str) -> bool:
    return "gene" in t or t in ("processed_transcript", "RNA")


def build_cache(gff: str, cache_dir: Optional[str] = None, verbose: bool = True) -> str:
    """Parse ``gff`` and write the pyvep Parquet cache. Returns the cache directory."""
    if pa is None:
        raise ImportError("pyarrow is required to build the pyvep cache")
    t0 = time.time()
    cache_dir = cache_dir or _default_cache_dir(gff)
    os.makedirs(cache_dir, exist_ok=True)
    feats, parts = _scan_gff(gff)
    types = feats["type"]

    # ---------------- chromosome synonyms from 'region' records ----------------
    reg = feats.filter(pc.equal(types, "region")).select(["seqid", "start", "end", "attributes"]).to_pandas()
    chroms = []
    for r in reg.itertuples(index=False):
        a = _parse_attrs(r.attributes)
        if r.start != 1:
            continue
        chroms.append(dict(seqid=r.seqid, length=int(r.end), name=a.get("chromosome") or a.get("Name"),
                           genome=a.get("genome", ""), is_circular=a.get("Is_circular", "")))
    chroms = pd.DataFrame(chroms, columns=["seqid", "length", "name", "genome", "is_circular"]) \
        .drop_duplicates("seqid")

    # ---------------- exon / CDS ----------------
    exdf = parts.to_pandas()
    del parts
    multi = exdf["parent"].str.contains(",", regex=False)
    if multi.any():
        exdf = pd.concat([exdf[~multi], exdf[multi].assign(parent=exdf.loc[multi, "parent"].str.split(","))
                         .explode("parent")], ignore_index=True)
    is_cds = exdf["type"].isin(["CDS", "stop_codon"]).to_numpy()
    cds = exdf[is_cds][["parent", "start", "end", "phase", "protein_id", "pversion"]] \
        .drop_duplicates(["parent", "start"], keep="last")  # VEP keys CDS by start
    ex = exdf[~is_cds][["parent", "start", "end"]].sort_values(["parent", "start"], kind="stable")
    del exdf

    # ---------------- genes / transcripts ----------------
    other = feats.filter(pc.is_in(types, pa.array(sorted(GFF_INCLUDE_TYPES - {"exon", "CDS", "stop_codon"})))) \
        .to_pandas()
    regdf = feats.filter(pc.is_in(types, pa.array(sorted(REGULATORY_TYPES)))) \
        .select(["seqid", "type", "start", "end", "attributes"]).to_pandas()
    del feats
    attrs = [_parse_attrs(a) for a in other["attributes"].to_numpy()]
    other["ID"] = [a.get("ID") or a.get("Name") or a.get("id") or a.get("name") for a in attrs]
    other["Parent"] = [a.get("Parent") or a.get("parent") for a in attrs]
    by_id = {}
    for i, rid in enumerate(other["ID"].to_numpy()):
        if isinstance(rid, str):
            by_id.setdefault(rid, i)
    ex_parents = set(ex["parent"].unique())

    # VEP (BaseGXF::_create_transcripts): top-level records without a Parent are genes
    # (type =~ /gene/, 'processed_transcript', 'RNA') whose children are transcripts, or
    # are themselves transcripts.  Transcripts need exon children.
    parents = [p if isinstance(p, str) and p else None for p in other["Parent"].to_numpy()]
    is_top = np.array([p is None for p in parents], dtype=bool)
    otypes = other["type"].to_numpy()
    rows = []
    for i, r in enumerate(other.itertuples(index=False)):
        if is_top[i]:
            if _is_gene_type(r.type):
                continue
            gene_i = None
        else:
            gi = by_id.get(parents[i].split(",")[0])
            if gi is None or not is_top[gi] or not _is_gene_type(otypes[gi]):
                continue
            gene_i = gi
        if r.ID not in ex_parents:
            continue
        a = attrs[i]
        ga = attrs[gene_i] if gene_i is not None else None
        biotype = _record_biotype(a, r.type, ga)
        if not biotype:
            continue
        tid = a.get("transcript_id") or r.ID
        tid = re.sub(r"^(gene|transcript):", "", tid, flags=re.I)
        version = a.get("version") or a.get("transcript_version")
        gene_id = a.get("gene_id")
        if not gene_id:
            for pair in (a.get("dbxref") or a.get("Dbxref") or "").split(","):
                k, _, v = pair.partition(":")
                if k == "GeneID":
                    gene_id = v
                    break
        if not gene_id and ga is not None:
            gene_id = ga.get("gene_id") or re.sub(r"^(gene|transcript):", "",
                                                    other["ID"].iat[gene_i], flags=re.I)
        symbol = a.get("gene_name") or a.get("gene")
        if not symbol and ga is not None:
            symbol = ga.get("name") or ga.get("Name")
        tags = [t.strip() for t in _unescape(a.get("tag") or "").split(",") if t.strip()]
        rows.append(dict(
            gff_id=r.ID, transcript_id=tid, version=version, chrom=r.seqid, start=int(r.start),
            end=int(r.end), strand=1 if r.strand == "+" else -1, gff_type=r.type, gff_source=r.source,
            biotype=biotype, gene_id=gene_id, symbol=_unescape(symbol),
            gene_biotype=(ga or {}).get("gene_biotype") or (ga or {}).get("biotype"),
            gene_description=_unescape((ga or {}).get("description")),
            tsl=a.get("transcript_support_level") or a.get("tsl"),
            mane_select=int("MANE Select" in tags or "MANE_Select" in tags),
            mane_plus_clinical=int("MANE Plus Clinical" in tags or "MANE_Plus_Clinical" in tags),
            canonical=int("Ensembl_canonical" in tags),
            refseq_select=int("RefSeq Select" in tags),
            tags=",".join(tags) or None,
            partial=a.get("partial"),
            order=i,
        ))
    tx = pd.DataFrame(rows)
    if tx.empty:
        raise ValueError(f"no transcripts found in {gff}")
    tx = tx.drop_duplicates("gff_id", keep="first")

    # exons -> list columns
    ex = ex[ex["parent"].isin(set(tx["gff_id"]))]
    exg = ex.groupby("parent", sort=False)
    ex_s = exg["start"].agg(list)
    ex_e = exg["end"].agg(list)
    tx["exon_starts"] = tx["gff_id"].map(ex_s)
    tx["exon_ends"] = tx["gff_id"].map(ex_e)
    tx = tx[tx["exon_starts"].notna()].reset_index(drop=True)

    # coding region: CDS records whose start lies in an exon (VEP); first CDS in
    # transcript order gives the protein id / start phase
    cds = cds[cds["parent"].isin(set(tx["gff_id"]))]
    strand_of = dict(zip(tx["gff_id"], tx["strand"]))
    cds = cds.assign(_str=cds["parent"].map(strand_of))
    cds = cds.assign(_key=np.where(cds["_str"] > 0, cds["start"], -cds["start"])) \
        .sort_values(["parent", "_key"], kind="stable")
    exmap = dict(zip(tx["gff_id"], zip(tx["exon_starts"], tx["exon_ends"])))
    keep = []
    for p, s in zip(cds["parent"].to_numpy(), cds["start"].to_numpy()):
        es, ee = exmap[p]
        j = bisect.bisect_right(es, s) - 1
        keep.append(j >= 0 and s <= ee[j])
    cds = cds[np.array(keep, dtype=bool)]
    cg = cds.groupby("parent", sort=False)
    first = cg.head(1).set_index("parent")
    tx["cds_start"] = tx["gff_id"].map(cg["start"].min()).astype("Int64")
    tx["cds_end"] = tx["gff_id"].map(cg["end"].max()).astype("Int64")
    tx["cds_phase"] = tx["gff_id"].map(first["phase"]).map(
        lambda p: {"0": 0, "1": 2, "2": 1}.get(p, 0) if isinstance(p, str) else 0).astype("int8")
    prot = {}
    for p, pid, pv in zip(first.index, first["protein_id"].to_numpy(dtype=object),
                          first["pversion"].to_numpy(dtype=object)):
        if isinstance(pid, str) and pid:
            prot[p] = pid if re.search(r"\.\d+$", pid) else f"{pid}.{pv if isinstance(pv, str) and pv else 1}"
    tx["protein_id"] = tx["gff_id"].map(prot)
    tx = tx.sort_values(["chrom", "start", "end", "order"], kind="stable").reset_index(drop=True)
    tx = tx.drop(columns=["order"])

    # regulatory features (extra column, not part of VEP --gff output)
    if len(regdf):
        ra = [_parse_attrs(a) for a in regdf["attributes"]]
        regdf = pd.DataFrame(dict(chrom=regdf["seqid"], type=regdf["type"], start=regdf["start"],
                                  end=regdf["end"],
                                  id=[a.get("ID") or a.get("Name") for a in ra],
                                  note=[_unescape(a.get("regulatory_class") or a.get("Note") or a.get("function"))
                                        for a in ra])).sort_values(["chrom", "start"]).reset_index(drop=True)
    else:
        regdf = pd.DataFrame(columns=["chrom", "type", "start", "end", "id", "note"])

    meta = _gff_header(gff)
    meta.update(cache_version=CACHE_VERSION, pyvep_version=__version__, gff=os.path.abspath(gff),
                gff_size=os.path.getsize(gff), gff_mtime=os.path.getmtime(gff),
                n_transcripts=int(len(tx)), built=time.strftime("%Y-%m-%d %H:%M:%S"))
    tx.to_parquet(os.path.join(cache_dir, "transcripts.parquet"), index=False)
    regdf.to_parquet(os.path.join(cache_dir, "regulatory.parquet"), index=False)
    chroms.to_parquet(os.path.join(cache_dir, "chromosomes.parquet"), index=False)
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if verbose:
        print(f"pyvep: cached {len(tx):,} transcripts from {os.path.basename(gff)} "
              f"in {time.time() - t0:.1f}s -> {cache_dir}", file=sys.stderr)
    return cache_dir


def _default_cache_dir(gff: str) -> str:
    return re.sub(r"(\.gz|\.bgz)$", "", gff) + ".pyvep"


# --------------------------------------------------------------------------------------
# Annotation object (transcripts + genome + chromosome names)
# --------------------------------------------------------------------------------------
class Annotation:
    """Transcript models + genome sequence + chromosome synonyms."""

    def __init__(self, cache_dir: str, fasta: Optional[str] = None, synonyms=None,
                 sequence_report: Optional[str] = None):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.tx = pd.read_parquet(os.path.join(cache_dir, "transcripts.parquet"))
        self.reg = pd.read_parquet(os.path.join(cache_dir, "regulatory.parquet"))
        self.chroms = pd.read_parquet(os.path.join(cache_dir, "chromosomes.parquet"))
        self.fasta = Fasta(fasta) if isinstance(fasta, str) else fasta
        self.accession = self.meta.get("accession")
        self._models: Dict[int, Transcript] = {}
        self._rows = None
        self._by_chrom = {}
        for c, g in self.tx.groupby("chrom", sort=False):
            idx = g.index.to_numpy()
            st, en = g["start"].to_numpy(), g["end"].to_numpy()
            o = np.argsort(st, kind="stable")
            idx, st, en = idx[o], st[o], en[o]
            self._by_chrom[c] = (idx, st, en, np.maximum.accumulate(en))
        self._reg_by_chrom = {c: (g["start"].to_numpy(), g["end"].to_numpy(), g.index.to_numpy(),
                                  np.maximum.accumulate(g["end"].to_numpy()))
                              for c, g in self.reg.groupby("chrom", sort=False)} if len(self.reg) else {}
        self.mito = set(self.chroms.loc[self.chroms["genome"].isin(["mitochondrion"]), "seqid"])
        self._build_synonyms(synonyms, sequence_report)

    # ---- chromosome names ----
    def _build_synonyms(self, synonyms, sequence_report):
        seqids = set(self.tx["chrom"]) | set(self.chroms["seqid"])
        if self.fasta is not None:
            seqids |= set(self.fasta.lengths)
        syn: Dict[str, str] = {}

        def add(name, seqid):
            if name is None or (isinstance(name, float) and np.isnan(name)):
                return
            name = str(name)
            for n in {name, name.lower()}:
                syn.setdefault(n, seqid)

        for s in seqids:
            add(s, s)
            add(s.split(".")[0], s)
        names = {}  # assembled molecules: seqid -> chromosome name
        for r in self.chroms.itertuples(index=False):
            nm = "MT" if r.genome == "mitochondrion" else r.name
            if nm and nm not in ("Unknown", "Un") and (r.genome in ("chromosome", "mitochondrion")
                                                      or str(r.seqid).startswith("NC_")):
                names[r.seqid] = nm
        extra = []
        if sequence_report and os.path.exists(sequence_report):
            rep = pd.read_json(sequence_report, lines=True)
            for r in rep.to_dict("records"):
                sid = r.get("refseqAccession")
                if not isinstance(sid, str) or not sid:
                    continue
                if r.get("role") == "assembled-molecule":
                    loc = r.get("assignedMoleculeLocationType")
                    names[sid] = "MT" if loc == "Mitochondrion" else r.get("chrName")
                for k in ("genbankAccession", "ucscStyleName", "sequenceName"):
                    if isinstance(r.get(k), str) and r.get(k) not in ("", "na"):
                        extra.append((r[k], sid))
        # chromosome names (assembled NC_ molecules win over unlocalized scaffolds)
        order = sorted(names.items(), key=lambda kv: (not str(kv[0]).startswith("NC_"), str(kv[0])))
        n_auto = 0
        for sid, nm in order:
            add(nm, sid)
            add("chr" + str(nm), sid)
            if str(nm).isdigit():
                n_auto = max(n_auto, int(nm))
        for nm, sid in extra:
            add(nm, sid)
        names = dict(order)
        # GWAS-pipeline numeric codes: X=n+1, Y=n+2, MT=n+4
        for sid, nm in names.items():
            u = str(nm).upper()
            code = {"X": n_auto + 1, "Y": n_auto + 2, "MT": n_auto + 4, "M": n_auto + 4}.get(u)
            if code and n_auto:
                add(str(code), sid)
            if u in ("MT", "M"):
                add("M", sid); add("MT", sid); add("chrM", sid); add("chrMT", sid)
        if isinstance(synonyms, str) and os.path.exists(synonyms):
            for line in open(synonyms):
                f = line.rstrip("\n").split("\t")
                if len(f) >= 2:
                    a, b = f[0], f[1]
                    if a in seqids:
                        add(b, a)
                    elif b in seqids:
                        add(a, b)
        elif isinstance(synonyms, dict):
            for k, v in synonyms.items():
                add(k, v)
        self.synonyms = syn
        self.n_autosome = n_auto

    def resolve(self, chrom) -> Optional[str]:
        c = str(chrom)
        if c.endswith(".0") and c[:-2].isdigit():
            c = c[:-2]
        return self.synonyms.get(c) or self.synonyms.get(c.lower())

    def seq_name_for_fasta(self, seqid: str) -> Optional[str]:
        if self.fasta is None:
            return None
        if seqid in self.fasta:
            return seqid
        for k, v in self.synonyms.items():
            if v == seqid and k in self.fasta:
                return k
        return None

    def transcript(self, i: int) -> "Transcript":
        m = self._models.get(i)
        if m is None:
            if self._rows is None:
                self._rows = list(self.tx.itertuples(index=False))
            m = self._models[i] = Transcript(self._rows[i], self)
        return m

    def candidates(self, seqid, vs: np.ndarray, ve: np.ndarray, dist: int):
        """Vectorised: for each variant, transcripts within ``dist`` bp."""
        if seqid not in self._by_chrom:
            return np.zeros(0, int), np.zeros(0, int)
        idx, st, en, cmax = self._by_chrom[seqid]
        lo = np.searchsorted(cmax, vs - dist, side="left")
        hi = np.searchsorted(st, ve + dist, side="right")
        n = np.maximum(hi - lo, 0)
        vi = np.repeat(np.arange(len(vs)), n)
        if len(vi) == 0:
            return vi, vi
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        ti = np.repeat(lo, n) + off
        ok = (en[ti] >= vs[vi] - dist) & (st[ti] <= ve[vi] + dist)
        return vi[ok], idx[ti[ok]]

    def regulatory(self, seqid, vs, ve):
        out = [None] * len(vs)
        if seqid not in self._reg_by_chrom:
            return out
        st, en, idx, cmax = self._reg_by_chrom[seqid]
        lo = np.searchsorted(cmax, vs, side="left")
        hi = np.searchsorted(st, np.maximum(ve, vs), side="right")
        for k in range(len(vs)):
            hits = [j for j in range(lo[k], hi[k]) if en[j] >= vs[k] and st[j] <= max(ve[k], vs[k])]
            if hits:
                out[k] = ";".join(f"{self.reg['type'].iat[idx[j]]}:{self.reg['id'].iat[idx[j]]}" for j in hits)
        return out


def load_annotation(gff: str, fasta: Optional[str] = None, cache_dir: Optional[str] = None,
                    rebuild: bool = False, synonyms=None, sequence_report: Optional[str] = None,
                    verbose: bool = True) -> Annotation:
    """Load (building if needed) the parquet cache for ``gff`` and attach the genome."""
    cache_dir = cache_dir or _default_cache_dir(gff)
    meta_f = os.path.join(cache_dir, "meta.json")
    stale = rebuild or not os.path.exists(meta_f)
    if not stale:
        with open(meta_f) as fh:
            m = json.load(fh)
        stale = (m.get("cache_version") != CACHE_VERSION or m.get("gff_size") != os.path.getsize(gff)
                 or abs(m.get("gff_mtime", 0) - os.path.getmtime(gff)) > 1)
    if stale:
        build_cache(gff, cache_dir, verbose=verbose)
    gdir = os.path.dirname(os.path.abspath(gff))
    if fasta is None:
        fasta = _find_fasta(gdir)
        if fasta is None:
            raise FileNotFoundError(f"no FASTA found next to {gff}; pass fasta=")
    if sequence_report is None and os.path.exists(os.path.join(gdir, "sequence_report.jsonl")):
        sequence_report = os.path.join(gdir, "sequence_report.jsonl")
    if synonyms is None and os.path.exists(os.path.join(gdir, "chrsynon")):
        synonyms = os.path.join(gdir, "chrsynon")
    return Annotation(cache_dir, fasta, synonyms=synonyms, sequence_report=sequence_report)


def _find_fasta(d: str) -> Optional[str]:
    pats = ["*_genomic.fna", "*.fna", "*.fa", "*.fasta", "*_genomic.fna.gz", "*.fa.gz", "*.fna.gz"]
    for p in pats:
        hits = sorted(h for h in glob(os.path.join(d, p)) if "_adjusted" not in h
                      and not re.search(r"(rna|cds_from_genomic|protein)\.f", os.path.basename(h)))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------------------
# Transcript model
# --------------------------------------------------------------------------------------
_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT",
          "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT"}


def _translate_codon(c: str, table: int) -> str:
    t = CODON_TABLES[table]
    aa = t.get(c)
    if aa is not None:
        return aa
    try:
        opts = {t[a + b + d] for a in _IUPAC[c[0]] for b in _IUPAC[c[1]] for d in _IUPAC[c[2]]}
    except KeyError:
        return "X"
    return opts.pop() if len(opts) == 1 else "X"


def _bio_translate(seq: str, table: int = 1) -> str:
    """Bio::Seq->translate: whole codons only, ambiguous codons resolved when possible."""
    t = CODON_TABLES[table]
    out = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i + 3]
        aa = t.get(c)
        out.append(aa if aa is not None else _translate_codon(c, table))
    return "".join(out)


def _scan_translate(seq: str, start_codon: int, table: int = 1, stop: bool = True) -> str:
    """Translate whole codons from codon index ``start_codon`` up to (and including) the first stop."""
    t = CODON_TABLES[table]
    out = []
    for i in range(max(start_codon, 0) * 3, len(seq) - 2, 3):
        c = seq[i:i + 3]
        aa = t.get(c) or _translate_codon(c, table)
        out.append(aa)
        if stop and aa == "*":
            break
    return "".join(out)


class Transcript:
    """Coordinates and (lazily) sequences of one transcript, VEP/Ensembl conventions."""

    def __init__(self, row, ann: Annotation):
        self.ann = ann
        self.stable_id = row.transcript_id
        v = row.version
        self.feature = self.stable_id + (f".{v}" if v and isinstance(v, str) and not re.search(r"\.\d+$", self.stable_id) else "")
        self.chrom, self.start, self.end, self.strand = row.chrom, int(row.start), int(row.end), int(row.strand)
        self.biotype, self.gene_id, self.symbol = row.biotype, row.gene_id, row.symbol
        self.protein_id = row.protein_id if isinstance(row.protein_id, str) else None
        self.mane_select = bool(row.mane_select)
        self.mane_plus_clinical = bool(row.mane_plus_clinical)
        self.canonical = bool(row.canonical)
        self.tsl = row.tsl if isinstance(row.tsl, str) else None
        tsl = 100
        if self.tsl:  # VEP: attribute 'tsl'.$value matched against /tsl(\d+)/
            m = re.match(r"(\d+)", self.tsl)
            if m and int(m.group(1)):
                tsl = int(m.group(1))
        self._tsl_rank = tsl
        es = [int(x) for x in row.exon_starts]
        ee = [int(x) for x in row.exon_ends]
        o = sorted(range(len(es)), key=lambda k: es[k])
        self.es, self.ee = [es[k] for k in o], [ee[k] for k in o]
        n = len(self.es)
        self.exon_cdna_start = [0] * n
        cum = 0
        for k in (range(n) if self.strand > 0 else range(n - 1, -1, -1)):
            self.exon_cdna_start[k] = cum + 1
            cum += self.ee[k] - self.es[k] + 1
        self.length = cum  # spliced length
        self.introns = []  # (start, end, frameshift) genomic ascending
        for k in range(n - 1):
            s, e = self.ee[k] + 1, self.es[k + 1] - 1
            if s <= e:
                self.introns.append((s, e, abs(e - s) <= 12))
        self.has_fs_intron = any(f for _, _, f in self.introns)
        cs, ce = row.cds_start, row.cds_end
        self.coding = cs is not None and not pd.isna(cs)
        self.translation = self.coding
        if self.coding:
            self.crs, self.cre = int(cs), int(ce)  # genomic coding region
            self.pad = int(row.cds_phase) if int(row.cds_phase) > 0 else 0
            self.ccs = self.g2c(self.crs if self.strand > 0 else self.cre)  # cdna_coding_start
            self.cce = self.g2c(self.cre if self.strand > 0 else self.crs)  # cdna_coding_end
            if self.ccs is None or self.cce is None:
                self.coding = self.translation = False
        self.table = 2 if self.chrom in ann.mito else 1
        self._mrna = None
        self._pep = None
        self._full = None
        self._T = self._u5 = self._u3 = None

    # ---- coordinate mapping ----
    def g2c(self, g: int) -> Optional[int]:
        j = bisect.bisect_right(self.es, g) - 1
        if j < 0 or g > self.ee[j]:
            return None
        return self.exon_cdna_start[j] + (g - self.es[j] if self.strand > 0 else self.ee[j] - g)

    # ---- sequences ----
    @property
    def mrna(self) -> str:
        if self._mrna is None:
            fa = self.ann.fasta
            name = self.ann.seq_name_for_fasta(self.chrom)
            if name is None:
                raise KeyError(f"{self.chrom} not found in FASTA {fa.path}")
            parts = [fa.fetch(name, s, e) for s, e in zip(self.es, self.ee)]
            seq = "".join(parts)
            self._mrna = seq if self.strand > 0 else _revcomp(seq)
        return self._mrna

    @property
    def translateable(self) -> str:
        if self._T is None:
            self._T = "N" * self.pad + self.mrna[self.ccs - 1:self.cce]
        return self._T

    @property
    def utr5(self) -> str:
        if self._u5 is None:
            self._u5 = self._utr_mrna[:self.ccs - 1]
        return self._u5

    @property
    def utr3(self) -> str:
        if self._u3 is None:
            self._u3 = self._utr_mrna[self.cce:]
        return self._u3

    @property
    def _utr_mrna(self) -> str:
        if not (_EMULATE_VEP_BUGS and self.strand < 0):
            return self.mrna
        S, E = self.start, self.end
        return "".join(_revcomp(self.genomic_seq(S + E - e, S + E - s))
                       for s, e in sorted(zip(self.es, self.ee), reverse=True))

    @property
    def peptide(self) -> str:
        """Translation->seq (terminal stop removed)."""
        if self._pep is None:
            t = self.translateable
            if len(t) % 3 == 0 and t[-3:] in ("TAG", "TGA", "TAA"):
                t = t[:-3]
            self._pep = _bio_translate(t, self.table)
        return self._pep

    @property
    def full_translation(self) -> str:
        """Bio::Seq translate of the translateable sequence (stop codon kept)."""
        if self._full is None:
            self._full = _bio_translate(self.translateable, self.table)
        return self._full

    def genomic_seq(self, s: int, e: int) -> str:
        return self.ann.fasta.fetch(self.ann.seq_name_for_fasta(self.chrom), s, e)


# --------------------------------------------------------------------------------------
# Variant x transcript consequence calculation (port of Ensembl VariationEffect logic)
# --------------------------------------------------------------------------------------
class _Var:
    __slots__ = ("start", "end", "ref", "alt", "cls", "chrom", "shift")

    def __init__(self, chrom, start, end, ref, alt):
        self.chrom, self.start, self.end, self.ref, self.alt = chrom, start, end, ref, alt
        self.shift = {}  # strand -> genomic 3' shift result (shared by all transcripts)
        if len(ref) == len(alt):
            self.cls = "SNV" if len(ref) == 1 else "substitution"
        elif ref == "":
            self.cls = "insertion"
        elif alt == "":
            self.cls = "deletion"
        else:
            self.cls = "indel"


class _TVA:
    """One variant allele vs one transcript (the bits of TranscriptVariationAllele VEP uses)."""

    def __init__(self, v: _Var, tx: Transcript):
        self.v, self.tx = v, tx
        self.c = {}  # predicate cache (mirrors VEP's _predicate_cache semantics)
        s, e = v.start, v.end
        self.ins = s == e + 1
        self.min, self.max = (e, s) if s > e else (s, e)
        st = tx.strand
        self.A = v.alt if st > 0 else _revcomp(v.alt)  # alt feature_seq
        self.R = v.ref if st > 0 else _revcomp(v.ref)  # ref feature_seq
        self.ref_len = e - s + 1 if not self.ins else 0
        self.alt_len = len(v.alt)
        self.within = _overlap(s, e, tx.start, tx.end)
        self.complete_overlap = s <= tx.start and e >= tx.end
        p = {"within_feature": int(self.within), tx.biotype: 1}
        if self.complete_overlap:
            p["complete_overlap"] = 1
        if self.ref_len == self.alt_len:
            p["snp"] = 1
        elif self.ref_len > self.alt_len:
            p["deletion"] = p["decrease_length"] = 1
        else:
            p["insertion"] = p["increase_length"] = 1
        self.pre = p
        self.cdna_s = self.cdna_e = self.cds_s = self.cds_e = self.tl_s = self.tl_e = None
        self.cdna_any = self.cds_any = False
        if not self.within:
            return
        mn, mx = self.min, self.max
        p["intron"] = int(any(_overlap(mn, mx, a - 3, b + 3) for a, b, _ in tx.introns))
        p["intron_boundary"] = int(any(_overlap(mn, mx, a - 3, a + 7) or _overlap(mn, mx, b - 7, b + 3)
                                       for a, b, _ in tx.introns))
        stretch = 12 if tx.has_fs_intron else 0
        p["exon"] = int(any(_overlap(mn - stretch, mx + stretch, a, b) for a, b in zip(tx.es, tx.ee)))
        if tx.translation:
            if not _overlap(mn, mx, tx.crs, tx.cre):
                p["utr"] = 1
            else:
                if mn < tx.crs or mx > tx.cre:
                    p["utr"] = 1
                if not p["exon"]:
                    p["non_coding"] = 1
                else:
                    p["coding"] = 1
        else:
            p["non_coding"] = 1
        self._map()

    # ------------------------------------------------------------------ mapping
    def _map(self):
        tx, v = self.tx, self.v
        if self.ins:
            L, R = v.end, v.start
            c5, c3 = (tx.g2c(L), tx.g2c(R)) if tx.strand > 0 else (tx.g2c(R), tx.g2c(L))
            x = c5 if c5 is not None else (c3 - 1 if c3 is not None else None)
            self.cdna_s, self.cdna_e = (x + 1, x) if x is not None else (None, None)
            self.cdna_any = x is not None and x > 0 and x + 1 <= tx.length
            if tx.coding and x is not None and tx.ccs <= x and x + 1 <= tx.cce:
                k = x - tx.ccs + 1
                self.cds_s, self.cds_e = k + 1 + tx.pad, k + tx.pad
                self.cds_any = True
            else:
                self.cds_s = self.cds_e = None
                self.cds_any = False
        else:
            b5, b3 = (v.start, v.end) if tx.strand > 0 else (v.end, v.start)
            self.cdna_s, self.cdna_e = tx.g2c(b5), tx.g2c(b3)
            mapped = [c for c in (tx.g2c(g) for g in range(v.start, v.end + 1)) if c is not None] \
                if (v.end - v.start) < 5000 else [c for c in (self.cdna_s, self.cdna_e) if c is not None]
            self.cdna_any = bool(mapped)
            if tx.coding:
                f = lambda c: (c - tx.ccs + 1 + tx.pad) if (c is not None and tx.ccs <= c <= tx.cce) else None
                self.cds_s, self.cds_e = f(self.cdna_s), f(self.cdna_e)
                self.cds_any = any(tx.ccs <= c <= tx.cce for c in mapped)
            else:
                self.cds_s = self.cds_e = None
                self.cds_any = False
        pep = lambda k: (k + 2) // 3 if k is not None else None
        if self.cds_s is not None and self.cds_e is not None:
            self.tl_s, self.tl_e = pep(self.cds_s), pep(self.cds_e)
        else:
            self.tl_s = pep(self.cds_s)
            self.tl_e = pep(self.cds_e)

    # ------------------------------------------------------------------ intron effects
    def intron_effects(self):
        ie = self.c.get("_ie")
        if ie is not None:
            return ie
        ie = {}
        # VEP fetches the overlapped introns / intron boundaries once, using the whole
        # variant span, and then tests every differing region against that list
        mn, mx = self.min, self.max
        introns = [x for x in self.tx.introns if _overlap(mn, mx, x[0] - 3, x[1] + 3)]
        bounds = [(x, int(_overlap(mn, mx, x[0] - 3, x[0] + 7)) + int(_overlap(mn, mx, x[1] - 7, x[1] + 3)))
                  for x in self.tx.introns]
        bounds = [b for b in bounds if b[1]]
        for (rs0, re0) in self._differing_regions():
            r_s, r_e = rs0, re0
            insertion = r_s == r_e + 1
            for (a, b, fs) in introns:  # intron tree
                if fs and _overlap(r_s, r_e, a, b):
                    ie["within_frameshift_intron"] = 1
                    continue
                if _overlap(r_s, r_e, a + 2, b - 2) or (insertion and (r_s == a + 2 or r_e == b - 2)):
                    ie["intronic"] = 1
                s, e = (r_e, r_s) if r_s > r_e else (r_s, r_e)
                if _overlap(s, e, b - 16, b - 2):
                    ie["ppt"] = 1
                if _overlap(s, e, a + 2, a + 16):
                    ie["ppt_rev"] = 1
            for (a, b, fs), n_hits in bounds:
                for _ in range(n_hits):  # boundary tree (an intron can be returned twice)
                    if fs and _overlap(r_s, r_e, a, b):
                        ie["within_frameshift_intron"] = 1
                        continue
                    if _overlap(r_s, r_e, a, a + 1):
                        ie["start_ss"] = 1
                    if _overlap(r_s, r_e, b - 1, b):
                        ie["end_ss"] = 1
                    if _overlap(r_s, r_e, a + 4, a + 4):
                        ie["fifth"] = 1
                    if _overlap(r_s, r_e, a + 2, a + 5):
                        ie["donor_region"] = 1
                    if _overlap(r_s, r_e, b - 16, b - 2):
                        ie["ppt"] = 1
                    if _overlap(r_s, r_e, b - 4, b - 4):
                        ie["fifth_rev"] = 1
                    if _overlap(r_s, r_e, b - 5, b - 2):
                        ie["donor_region_rev"] = 1
                    if _overlap(r_s, r_e, a + 2, a + 16):
                        ie["ppt_rev"] = 1
                    if not (ie.get("start_ss") or ie.get("end_ss")):
                        ie["splice_region"] = int(
                            _overlap(r_s, r_e, a + 2, a + 7) or _overlap(r_s, r_e, b - 7, b - 2) or
                            _overlap(r_s, r_e, a - 3, a - 1) or _overlap(r_s, r_e, b + 1, b + 3) or
                            (insertion and (r_s == a or r_e == b or r_s == a + 2 or r_e == b - 2)))
        self.c["_ie"] = ie
        return ie

    def _differing_regions(self):
        v = self.v
        if self.ref_len > 1 and self.alt_len > 1:
            a1, a2 = v.ref, v.alt
            n = max(len(a1), len(a2))
            pos = [i for i in range(n) if (a1[i] if i < len(a1) else "\0") != (a2[i] if i < len(a2) else "\0")]
            regs = []
            for q in pos:
                if regs and q == regs[-1][1] + 1:
                    regs[-1][1] = q
                else:
                    regs.append([q, q])
            return [(v.start + a, v.start + b) for a, b in regs]
        return [(v.start, v.end)]

    # ------------------------------------------------------------------ codons/peptides
    def codon(self, ref=False):
        key = "_rcodon" if ref else "_acodon"
        if key in self.c:
            return self.c[key]
        out = None
        tl_s, tl_e = self.tl_s, self.tl_e
        if tl_s and tl_e and self.cds_s is not None and self.cds_e is not None:
            T = self.tx.translateable
            ccs, cce = tl_s * 3 - 2, tl_e * 3
            clen = cce - ccs + 1
            if ref:
                out = T[ccs - 1:ccs - 1 + max(clen, 0)] if ccs - 1 <= len(T) else ""
            else:
                vf_nt_len = self.cds_e - self.cds_s + 1
                alt_cds = T[:self.cds_s - 1] + self.A + T[self.cds_e:]
                if len(alt_cds) < 3:
                    alt_cds = ""
                alt_cds += self.tx.utr3
                ln = clen + (len(self.A) - vf_nt_len)
                out = alt_cds[ccs - 1:ccs - 1 + ln] if ln > 0 else (alt_cds[ccs - 1:ln] if ln < 0 else "")
            if out == "":
                out = "-"
        self.c[key] = out
        return out

    def peptide(self, ref=False):
        key = "_rpep" if ref else "_apep"
        if key in self.c:
            return self.c[key]
        out = None
        seq = self.v.ref if ref else self.v.alt
        if _UNAMBIG.match(seq or "-"):
            codon = self.codon(ref)
            if codon:
                if codon == "-":
                    out = "-"
                else:
                    whole = codon[:len(codon) // 3 * 3]
                    part = codon[len(codon) // 3 * 3:]
                    pep = _bio_translate(whole, self.tx.table) if whole else ""
                    if part and pep != "*":
                        pep += "X"
                    out = pep or "-"
        self.c[key] = out
        return out

    def pep_alleles(self):
        """_get_peptide_alleles: () or (ref, alt) with '-' -> ''."""
        a = self.peptide()
        r = self.peptide(ref=True) if a else None
        if a and r:
            return ("" if r == "-" else r, "" if a == "-" else a)
        return ()

    def codon_alleles(self):
        if self.frameshift():
            return ()
        a = self.codon()
        if a is None:
            return ()
        r = self.codon(ref=True)
        if r is None:
            return ()
        return ("" if r == "-" else r, "" if a == "-" else a)

    # ------------------------------------------------------------------ predicates
    def partial_codon(self):
        if "partial_codon" not in self.c:
            r = False
            if self.tl_s is not None:
                L = len(self.tx.translateable)
                last = L - (self.tl_s * 3 - 3)
                r = 0 < last < 3
            self.c["partial_codon"] = r
        return self.c["partial_codon"]

    def overlaps_start_codon(self):
        if "osc" not in self.c:
            a, b = self.cdna_s, self.cdna_e
            self.c["osc"] = bool(a and b and _overlap(a, b, self.tx.ccs, self.tx.ccs + 2))
        return self.c["osc"]

    def _sub(self, s, start1, length, rep):
        return s[:start1 - 1] + rep + s[start1 - 1 + max(length, 0):]

    def _five_prime_subst(self):
        T, utr = self.tx.translateable, self.tx.utr5
        s = utr + T
        a, b = self.cdna_s, self.cdna_e
        return T, utr, self._sub(s, a, b - a + 1, self.A)

    def snp_start_altered(self):
        if "ssa" not in self.c:
            r = False
            if _UNAMBIG.match(self.v.alt or "-") and self.cdna_s and self.cdna_e:
                T, utr, s = self._five_prime_subst()
                r = s[len(s) - len(T):len(s) - len(T) + 3] != "ATG"
            self.c["ssa"] = r
        return self.c["ssa"]

    def ins_del_start_altered(self):
        if "idsa" not in self.c:
            r = False
            p = self.pre
            if (_UNAMBIG.match(self.v.alt or "-") and self.overlaps_start_codon()
                    and (p.get("increase_length") or p.get("decrease_length")) and self.cdna_s and self.cdna_e):
                T, utr, s = self._five_prime_subst()
                done = False
                if utr:
                    if s[:len(utr)] == utr and s[len(utr):len(utr) + 3] == "ATG":
                        r, done = False, True
                if not done:
                    if len(s) < len(T):
                        r = True
                    else:
                        r = T != s[len(s) - len(T):]
            self.c["idsa"] = r
        return self.c["idsa"]

    def inv_start_altered(self):
        if "isa" not in self.c:
            r = False
            if _UNAMBIG.match(self.v.alt or "-") and self.overlaps_start_codon() and self.cdna_s and self.cdna_e:
                utr = self.tx.utr5
                if utr:
                    T = self.tx.translateable
                    s = utr + T
                    if not self.cdna_e > len(s):
                        s = self._sub(s, self.cdna_s, self.cdna_e - self.cdna_s + 1, self.A)
                        r = s[len(utr):len(utr) + 3] != "ATG"
            self.c["isa"] = r
        return self.c["isa"]

    def start_lost(self):
        if "start_lost" in self.c:
            return self.c["start_lost"]
        self.c["start_lost"] = False
        if not self.overlaps_start_codon():
            return False
        if self.ins_del_start_altered() and not (self.inframe_insertion() or self.inframe_deletion()):
            self.c["start_lost"] = True
            return True
        if self.inv_start_altered():
            self.c["start_lost"] = True
            return True
        pa_ = self.pep_alleles()
        if not pa_:
            return False
        rp, ap = pa_
        if not rp or not ap or ap == "X":
            return False
        self.c["start_lost"] = bool(self.tl_s == 1 and not ap.endswith(rp) and not ap.startswith(rp))
        return self.c["start_lost"]

    def start_retained(self):
        if not self.overlaps_start_codon():
            return False
        if self.pre.get("snp"):
            return not self.snp_start_altered()
        return not self.ins_del_start_altered()

    def synonymous(self):
        pa_ = self.pep_alleles()
        if not pa_ or not pa_[0]:
            return False
        rp, ap = pa_
        return ap == rp and not self.stop_retained() and not self.start_retained() and "X" not in ap and "X" not in rp

    def missense(self):
        pa_ = self.pep_alleles()
        if not pa_:
            return False
        if self.start_lost() or self.stop_lost() or self.stop_gained() or self.partial_codon() or self.stop_retained():
            return False
        rp, ap = pa_
        return rp != ap and len(rp) == len(ap)

    def inframe_insertion(self):
        ca = self.codon_alleles()
        if self.start_lost():
            return False
        if not ca:
            return False
        rc, ac = ca
        if not len(ac) > len(rc):
            return False
        pa_ = self.pep_alleles()
        if not pa_:
            return False
        rp, ap = pa_
        if self.start_retained() and ap.endswith(rp):
            return False
        if rp == "*" and ap == "*":
            return False
        ap = re.sub(r"\*.+", "*", ap, count=1)
        return ap.startswith(rp) or ap.endswith(rp)

    def inframe_deletion(self):
        if self.partial_codon():
            return False
        ca = self.codon_alleles()
        pa_ = self.pep_alleles()
        if not ca:
            return False
        rc, ac = ca
        if not len(ac) < len(rc):
            return False
        if pa_ and pa_[0] == "*":
            return False
        if rc.startswith(ac) or rc.endswith(ac):
            return True
        while rc and ac and rc[0] == ac[0]:
            rc, ac = rc[1:], ac[1:]
        while rc and ac and rc[-1] == ac[-1]:
            rc, ac = rc[:-1], ac[:-1]
        return len(ac) == 0 and len(rc) % 3 == 0

    def stop_gained(self):
        if "stop_gained" not in self.c:
            self.c["stop_gained"] = False
            if self.stop_retained() or self.stop_lost():
                return False
            pa_ = self.pep_alleles()
            if not pa_:
                return False
            rp, ap = pa_
            self.c["stop_gained"] = "*" in ap and "*" not in rp
        return self.c["stop_gained"]

    def stop_lost(self):
        if self.partial_codon():
            return False
        if "stop_lost" not in self.c:
            self.c["stop_lost"] = False
            pa_ = self.pep_alleles()
            if pa_ and "X" not in pa_[1]:
                rp, ap = pa_
                self.c["stop_lost"] = "*" not in ap and "*" in rp
            else:
                self.c["stop_lost"] = self.ins_del_stop_altered(cil=False)
        return self.c["stop_lost"]

    def stop_retained(self):
        if self.partial_codon():
            return False
        if self.stop_lost():
            return False
        if "stop_retained" not in self.c:
            self.c["stop_retained"] = False
            pa_ = self.pep_alleles()
            ap = pa_[1] if pa_ else None
            if ap is not None and ap != "" and "X" not in ap:
                self.c["stop_retained"] = self.ref_eq_alt_sequence()
            else:
                p = self.pre
                self.c["stop_retained"] = bool((p.get("increase_length") or p.get("decrease_length"))
                                               and self.overlaps_stop_codon_cil()
                                               and not self.ins_del_stop_altered(cil=True))
        return self.c["stop_retained"]

    def ref_eq_alt_sequence(self):
        ref_seq = self.tx.peptide
        tl_s, tl_e = self.tl_s, self.tl_e
        rp, ap = self.pep_alleles()
        if rp == "X" and ap == "X":
            return False
        if tl_s > len(ref_seq) and ap.startswith("*"):
            return True
        mut = ref_seq[:tl_s - 1] + ap + ref_seq[tl_s - 1 + max(tl_e - tl_s + 1, 0):]
        sub = mut[:len(ref_seq)]
        final = mut[len(ref_seq):] if len(ref_seq) < len(mut) else None
        if (ref_seq == sub and final is not None and final.startswith("*")) or \
                ("*" in rp and rp.find("*") == ap.find("*")):
            return True
        return False

    def overlaps_stop_codon(self):
        a, b = self.cdna_s, self.cdna_e
        return bool(a and b and _overlap(a, b, self.tx.cce - 2, self.tx.cce))

    def overlaps_stop_codon_cil(self):
        if "osc_cil" not in self.c:
            v, tx = self.v, self.tx
            vs, ve = v.start, v.end
            fs = self.A
            if ve < vs and re.match(r"^[ACTGN]+$", fs or ""):
                vs = vs + len(fs) if tx.strand == 1 else vs - len(fs)
            if tx.strand == 1:
                te_s, te_e = tx.cre - 2, tx.cre
            else:
                te_s, te_e = tx.crs, tx.crs + 2
            self.c["osc_cil"] = bool(vs and ve and _overlap(vs, ve, te_s, te_e))
        return self.c["osc_cil"]

    def ins_del_stop_altered(self, cil=False):
        key = "idsta_cil" if cil else "idsta"
        if key not in self.c:
            r = False
            p = self.pre
            ok = _UNAMBIG.match(self.v.alt or "-") and \
                (self.overlaps_stop_codon_cil() if cil else self.overlaps_stop_codon()) and \
                (p.get("increase_length") or p.get("decrease_length")) and \
                self.cdna_s and self.cdna_e and self.cds_s
            if ok:
                T = self.tx.translateable
                s = T + self.tx.utr3
                s = self._sub(s, self.cds_s, self.cdna_e - self.cdna_s + 1, self.A)
                if len(s) < len(T):
                    r = True
                else:
                    pep = _bio_translate(s[len(T) - 3:len(T)], self.tx.table)
                    r = not (pep == "*")
            self.c[key] = r
        return self.c[key]

    def frameshift(self):
        if self.partial_codon() or self.stop_retained():
            return False
        if self.cds_s is None or self.cds_e is None:
            return False
        pa_ = self.pep_alleles()
        if pa_ and pa_[0].startswith("*"):
            return False
        var_len = self.cds_e - self.cds_s + 1
        return abs(self.alt_len - var_len) % 3 != 0

    def protein_altering(self):
        pa_ = self.pep_alleles()
        if not pa_:
            return False
        rp, ap = pa_
        if len(ap) == len(rp):
            return False
        if rp.startswith("*") or ap.startswith("*"):
            return False
        if ap.startswith(rp) or ap.endswith(rp):
            return False
        if self.inframe_deletion() or self.start_lost() or self.frameshift():
            return False
        return True

    def within_cds(self):
        if self.cds_any:
            return True
        if self.tx.translation and self.intron_effects().get("within_frameshift_intron"):
            return _overlap(self.v.start, self.v.end, self.tx.crs, self.tx.cre)
        return False

    def within_cdna(self):
        if self.cdna_any:
            return True
        return bool(self.intron_effects().get("within_frameshift_intron")) and self.within

    def coding_unknown(self):
        if self.complete_overlap:
            return False
        if not self.within_cds():
            return False
        a = self.peptide()
        pa_ = self.pep_alleles()
        if not ((not a) or (not pa_) or ("X" in a) or ("X" in pa_[0])):
            return False
        return not (self.frameshift() or self.inframe_deletion() or self.protein_altering() or
                    self.start_retained() or self.start_lost() or self.stop_retained() or self.stop_lost())

    def _before_coding(self):
        v, tx = self.v, self.tx
        if not tx.translation:
            return False
        if v.start == v.end + 1 and v.start == tx.crs:
            return True
        return _overlap(v.start, v.end, tx.start, tx.crs - 1)

    def _after_coding(self):
        v, tx = self.v, self.tx
        if not tx.translation:
            return False
        if v.start == v.end + 1 and v.end == tx.cre:
            return True
        return _overlap(v.start, v.end, tx.cre + 1, tx.end)

    def non_coding_exon(self):
        if self.complete_overlap or self.tx.translation:
            return False
        stretch = 12 if self.tx.has_fs_intron else 0
        mn, mx = self.min, self.max
        return any(_overlap(mn - stretch, mx + stretch, a, b) and _overlap(self.v.start, self.v.end, a, b)
                   for a, b in zip(self.tx.es, self.tx.ee))

    # ------------------------------------------------------------------ consequences
    _INCLUDE = {
        "transcript_ablation": {"complete_overlap": 1, "deletion": 1},
        "splice_acceptor_variant": {"intron_boundary": 1}, "splice_donor_variant": {"intron_boundary": 1},
        "stop_gained": {"coding": 1}, "frameshift_variant": {"coding": 1, "snp": 0},
        "stop_lost": {"coding": 1}, "start_lost": {"coding": 1},
        "inframe_insertion": {"coding": 1, "insertion": 1}, "inframe_deletion": {"coding": 1, "deletion": 1},
        "missense_variant": {"coding": 1, "decrease_length": 0, "increase_length": 0},
        "protein_altering_variant": {"coding": 1},
        "splice_donor_5th_base_variant": {"intron_boundary": 1}, "splice_region_variant": {"intron_boundary": 1},
        "splice_donor_region_variant": {"intron_boundary": 1},
        "splice_polypyrimidine_tract_variant": {"exon": 0, "intron": 1},
        "incomplete_terminal_codon_variant": {"coding": 1}, "start_retained_variant": {"coding": 1},
        "stop_retained_variant": {"coding": 1}, "synonymous_variant": {"coding": 1},
        "coding_sequence_variant": {"coding": 1},
        "5_prime_UTR_variant": {"exon": 1, "utr": 1}, "3_prime_UTR_variant": {"exon": 1, "utr": 1},
        "non_coding_transcript_exon_variant": {"exon": 1, "protein_coding": 0, "within_feature": 1},
        "intron_variant": {"intron": 1},
        "NMD_transcript_variant": {"nonsense_mediated_decay": 1, "within_feature": 1},
        "non_coding_transcript_variant": {"protein_coding": 0, "within_feature": 1},
        "coding_transcript_variant": {"protein_coding": 1, "within_feature": 1},
        "upstream_gene_variant": {"within_feature": 0}, "downstream_gene_variant": {"within_feature": 0},
    }

    def _skip(self, term):
        p = self.pre
        for k, val in self._INCLUDE.get(term, {}).items():
            if val == 0:
                if k in p and p[k] != 0:
                    return True
            elif not (k in p and p[k] == val):
                return True
        return False

    def _predicate(self, term):
        tx, ie = self.tx, None
        st = tx.strand
        if term == "transcript_ablation":
            return self.complete_overlap and self.ref_len > 0 and (self.alt_len < self.ref_len)
        if term in ("splice_acceptor_variant", "splice_donor_variant", "splice_donor_5th_base_variant",
                    "splice_region_variant", "splice_donor_region_variant",
                    "splice_polypyrimidine_tract_variant", "intron_variant"):
            ie = self.intron_effects()
            donor = ie.get("start_ss") if st == 1 else ie.get("end_ss")
            acceptor = ie.get("end_ss") if st == 1 else ie.get("start_ss")
            fifth = ie.get("fifth") if st == 1 else ie.get("fifth_rev")
            dregion = (ie.get("donor_region") if st == 1 else ie.get("donor_region_rev")) and not fifth
            if term == "splice_acceptor_variant":
                return bool(acceptor)
            if term == "splice_donor_variant":
                return bool(donor)
            if term == "splice_donor_5th_base_variant":
                return bool(fifth)
            if term == "splice_donor_region_variant":
                return bool(dregion)
            if term == "splice_polypyrimidine_tract_variant":
                return bool(ie.get("ppt") if st == 1 else ie.get("ppt_rev"))
            if term == "splice_region_variant":
                if donor or acceptor or dregion or fifth:
                    return False
                return bool(ie.get("splice_region"))
            return bool(ie.get("intronic"))
        if term == "stop_gained":
            return self.stop_gained()
        if term == "frameshift_variant":
            return self.frameshift()
        if term == "stop_lost":
            return self.stop_lost()
        if term == "start_lost":
            return self.start_lost()
        if term == "inframe_insertion":
            return self.inframe_insertion()
        if term == "inframe_deletion":
            return self.inframe_deletion()
        if term == "missense_variant":
            return self.missense()
        if term == "protein_altering_variant":
            return self.protein_altering()
        if term == "incomplete_terminal_codon_variant":
            return self.partial_codon()
        if term == "start_retained_variant":
            return self.start_retained()
        if term == "stop_retained_variant":
            return self.stop_retained()
        if term == "synonymous_variant":
            return self.synonymous()
        if term == "coding_sequence_variant":
            return self.coding_unknown()
        if term == "5_prime_UTR_variant":
            return (self._before_coding() if st == 1 else self._after_coding()) and self.within_cdna()
        if term == "3_prime_UTR_variant":
            return (self._after_coding() if st == 1 else self._before_coding()) and self.within_cdna()
        if term == "non_coding_transcript_exon_variant":
            return self.non_coding_exon()
        if term == "NMD_transcript_variant":
            return self.within and tx.biotype == "nonsense_mediated_decay"
        if term == "non_coding_transcript_variant":
            return self.within and not tx.translation and not self.non_coding_exon()
        if term == "coding_transcript_variant":
            return (not self.coding_unknown()) and self.complete_overlap and tx.translation
        return False

    _TERMS = [t for t, _ in CONSEQUENCES if t in {
        "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant", "stop_gained",
        "frameshift_variant", "stop_lost", "start_lost", "inframe_insertion", "inframe_deletion",
        "missense_variant", "protein_altering_variant", "splice_donor_5th_base_variant",
        "splice_region_variant", "splice_donor_region_variant", "splice_polypyrimidine_tract_variant",
        "incomplete_terminal_codon_variant", "start_retained_variant", "stop_retained_variant",
        "synonymous_variant", "coding_sequence_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
        "non_coding_transcript_exon_variant", "intron_variant", "NMD_transcript_variant",
        "non_coding_transcript_variant", "coding_transcript_variant"}]

    _TERM_CACHE: Dict[tuple, List[str]] = {}

    def consequences(self) -> List[str]:
        key = tuple(sorted(self.pre.items()))
        terms = self._TERM_CACHE.get(key)
        if terms is None:
            terms = self._TERM_CACHE[key] = [t for t in self._TERMS if not self._skip(t)]
        out = []
        for term in terms:
            if self._predicate(term):
                out.append(term)
                if term == "transcript_ablation":  # tier 1
                    break
        return out or ["intergenic_variant"]  # VEP DEFAULT_OVERLAP_CONSEQUENCE

    # ------------------------------------------------------------------ output fields
    def codon_position(self):
        return ((self.cds_s - 1) % 3) + 1 if self.cds_s is not None else None

    def display_codon(self, ref=False):
        codon = self.codon(ref)
        if not codon:
            return None
        d = codon.lower()
        pos = self.codon_position()
        fs = self.R if ref else self.A
        if pos and fs:
            p = pos - 1
            d = d[:p] + d[p:p + len(fs)].upper() + d[p + len(fs):]
        return d

    def amino_acids(self):
        a = self.peptide()
        if not a:
            return None
        r = self.peptide(ref=True)
        if not r:
            return None
        return f"{r}/{a}" if r != a else a

    def codons(self):
        a = self.display_codon()
        if not a:
            return None
        r = self.display_codon(ref=True)
        if not r:
            return None
        return f"{r}/{a}"

    # ------------------------------------------------------------------ HGVS
    def shift_info(self):
        """VEP _genomic_shift: (shift_length, shifted_seq, hgvs_allele) towards the transcript 3' end."""
        if "_shift" in self.c:
            return self.c["_shift"]
        v, tx = self.v, self.tx
        if tx.strand in v.shift:
            self.c["_shift"] = v.shift[tx.strand]
            return self.c["_shift"]
        res = (0, None, None)
        if v.cls in ("insertion", "deletion"):
            seq = v.ref if v.cls == "deletion" else v.alt
            hs = v.alt if v.cls == "insertion" else "-"
            pre = tx.genomic_seq(v.start - 1000, v.start - 1)
            post = tx.genomic_seq(v.end + 1, v.end + 1000)
            rev = tx.strand == -1
            n_len = len(seq)
            lim = (len(pre) - n_len + 1) if rev else (len(post) - n_len)
            if lim < 0:
                lim = len(post)
            k = 0
            n = 1 if rev else 0
            while n <= lim:
                if rev:
                    d, q = seq[-1], pre[len(pre) - n]
                else:
                    d, q = seq[0], post[n]
                if d != q or d == "N":
                    break
                k += 1
                if rev:
                    seq = d + seq[:-1]
                    hs = hs[-1] + hs[:-1] if hs else hs
                else:
                    seq = seq[1:] + d
                    hs = hs[1:] + hs[0] if hs else hs
                n += 1
            res = (k, seq, hs)
        self.c["_shift"] = v.shift[tx.strand] = res
        return res

    def _slice(self, a: int, b: int) -> str:
        """Transcript-oriented genomic slice of the transcript span, 1-based [a, b]."""
        tx = self.tx
        if b < a:
            return ""
        if tx.strand > 0:
            return tx.genomic_seq(tx.start + a - 1, tx.start + b - 1)
        return _revcomp(tx.genomic_seq(tx.end - b + 1, tx.end - a + 1))

    def _cdna_position(self, p: int) -> Optional[str]:
        tx = self.tx
        g = tx.start + p - 1 if tx.strand > 0 else tx.end - p + 1
        pos = None
        n = len(tx.es)
        for i in range(n):
            es, ee = tx.es[i], tx.ee[i]
            if g > ee:
                continue
            ecs = lambda k: tx.exon_cdna_start[k]
            ece = lambda k: tx.exon_cdna_start[k] + tx.ee[k] - tx.es[k]
            if g >= es:
                pos = str(ecs(i) + (g - es if tx.strand > 0 else ee - g))
            else:
                j = i - 1
                up, down = abs(g - tx.ee[j]), abs(es - g)
                if up < down or (up == down and tx.strand >= 0):
                    pos = (f"{ece(j)}+" if tx.strand >= 0 else f"{ecs(j)}-") + str(up)
                else:
                    pos = (f"{ecs(i)}-" if tx.strand >= 0 else f"{ece(i)}+") + str(down)
            break
        if not pos:
            return None
        m = re.match(r"([0-9]+)([+\-][0-9]+)?", pos)
        coord, off = int(m.group(1)), m.group(2)
        if tx.coding:
            if coord > tx.cce:
                return f"*{coord - tx.cce}" + (off or "")
            if coord == tx.cce and off is not None:
                return "*" + off.replace("+", "")
            coord = coord + (1 if coord >= tx.ccs else 0) - tx.ccs
        return f"{coord}{off or ''}"

    @staticmethod
    def _clip(n: dict) -> dict:
        ref, alt, s, e = n["ref"], n["alt"], n["start"], n["end"]
        n["original_ref"] = ref
        pre = ""
        isp = n.get("numbering") == "p"
        for _ in range(len(n["ref"])):
            r0, a0 = ref[:1], alt[:1]
            if isp and r0 == "*" and a0 == "*":
                n["type"] = "="
                return n
            if r0 == a0:
                s += 1
                ref, alt = ref[1:], alt[1:]
                pre += r0
            else:
                break
        for _ in range(len(ref)):
            if ref[-1:] == alt[-1:]:
                ref, alt = ref[:-1], alt[:-1]
                e -= 1
            else:
                break
        n.update(alt=alt, ref=ref, start=s, end=e, preseq=pre)
        if isp and ref == alt:
            n["type"] = "="
        elif ref != "-" and len(ref) == 1 and len(alt) == 1 and alt != ref:
            n["type"] = ">"
        elif len(ref) == 0 and len(alt) >= 1:
            prev = pre[len(pre) - len(alt):] if len(alt) <= len(pre) else pre
            if n.get("numbering") not in (None, "p") and alt == prev:
                n["type"] = "dup"
                n["start"] -= len(alt)
            else:
                n["type"] = "ins"
        elif len(ref) >= 1 and len(alt) == 0:
            n["type"] = "del"
        return n

    def hgvsc(self) -> Tuple[Optional[str], int]:
        """HGVSc and the HGVS_OFFSET shift length."""
        v, tx = self.v, self.tx
        if re.search(r"[^ACGT\-]", v.alt or "-", re.I):
            return None, 0
        k, _, hs = self.shift_info()
        alt = hs if (v.cls == "insertion" and k) else v.alt
        trlen = tx.end - tx.start + 1
        if tx.strand > 0:
            s, e = v.start - tx.start + 1, v.end - tx.start + 1
        else:
            s, e = tx.end - v.end + 1, tx.end - v.start + 1
        if (s < 1 and e < 1) or (s > trlen and e > trlen):
            return None, k
        s, e = min(max(s, 1), trlen), min(max(e, 1), trlen)
        if not s:
            return None, k
        if trlen < e + k:
            return None, k
        A = alt if tx.strand > 0 else _revcomp(alt)
        s, e = s + k, e + k
        # hgvs_variant_notation
        ref_len = max(e - s + 1, 0)
        refa = self._slice(s, s + ref_len - 1) if ref_len else ""
        if refa == A:
            return None, k
        n = dict(start=s, end=e, ref=refa, alt=A)
        alen = len(A)
        if not alen:
            n["type"] = "del"
        elif ref_len == alen:
            n["type"] = ">" if ref_len == 1 else ("inv" if A == _revcomp(refa) else "delins")
        elif not ref_len:
            nxt = self._slice(e + 1, e + alen)
            prv = self._slice(e - alen + 1, e) if e - alen >= 0 else None
            if nxt == A:
                n.update(end=s + alen - 1, type="dup")
            elif prv == A:
                n.update(start=e - alen + 1, type="dup")
            else:
                n.update(start=e, end=s, type="ins")
        elif alen % ref_len == 0 and A == refa * (alen // ref_len):
            m = alen // ref_len
            n["type"] = "dup" if m == 2 else f"[{m}]"
        else:
            n["type"] = "delins"
        if n["type"] != "dup":
            n = self._clip(n)
        same = n["start"] == n["end"]
        if v.cls == "SNV" and self.pre.get("exon") and self.cds_s is not None and self.cds_e is not None:
            n["start"] = n["end"] = str(self.cds_s)
        else:
            n["start"] = self._cdna_position(n["start"])
            n["end"] = n["start"] if same else self._cdna_position(n["end"])
        if n["start"] is None or n["end"] is None:
            return None, k
        pr = lambda x: re.search(r"(-?[0-9]+)\+?(-?[0-9]+)?", x)
        ms, me = pr(n["start"]), pr(n["end"])
        sc, so = int(ms.group(1)), int(ms.group(2) or 0)
        ec, eo = int(me.group(1)), int(me.group(2) or 0)
        if (sc > ec or (sc == ec and so > eo)) and "*" not in n["end"]:
            n["start"], n["end"] = n["end"], n["start"]
        num = "c" if tx.coding else "n"
        coords = n["start"] if n["start"] == n["end"] else f"{n['start']}_{n['end']}"
        t = n["type"]
        if t == ">" or (t == "inv" and len(n["ref"]) == 1):
            body = f"{n['start']}{n['ref']}>{n['alt']}"
        elif t in ("del", "inv", "dup"):
            body = coords + t
        elif t in ("delins", "ins"):
            body = coords + t + n["alt"]
        else:
            body = coords + t
        return f"{tx.feature}:{num}.{body}", k

    def _shifted_tva(self) -> "_TVA":
        """The same allele moved to its 3'-shifted position (what VEP uses for HGVSp)."""
        k, seq, _ = self.shift_info()
        if not k:
            return self
        v, st = self.v, self.tx.strand
        off = k * st
        if _EMULATE_VEP_BUGS and st < 0 and v.cls == "insertion" and k > len(v.alt):
            seq = v.alt
        if v.cls == "deletion":
            nv = _Var(v.chrom, v.start + off, v.end + off, seq, "")
        else:
            nv = _Var(v.chrom, v.start + off, v.end + off, "", seq)
        t = _TVA(nv, self.tx)
        return t

    def _alt_cds(self):
        if self.cds_s is None or self.cds_e is None:
            return None
        T = self.tx.translateable
        s = T[:self.cds_s - 1] + self.A + T[self.cds_e:]
        if len(s) < 3:
            s = ""
        return s + self.tx.utr3

    def _stop_loss_extra_aa(self, ref_var_pos, test=None, sh=None):
        if not ref_var_pos:
            return None
        base = sh or self
        alt_cds = base._alt_cds()
        if alt_cds is None:
            return None
        # codons upstream of the variant are unchanged: reuse the reference translation
        c0 = max((base.cds_s - 1) // 3, 0)
        i = self.tx.full_translation[:c0].find("*")
        if i < 0:
            k = _scan_translate(alt_cds, c0, self.tx.table).find("*")
            i = c0 + k if k >= 0 else -1
        extra = None
        if i >= 0:
            extra = (i + 1 - ref_var_pos) if test == "fs" else (i + 1 - 1 - len(self.tx.peptide))
        return extra if extra is not None and extra > 0 else None

    def hgvsp(self) -> Optional[str]:
        v, tx = self.v, self.tx
        if re.search(r"[^ACGT\-]", v.alt or "-", re.I):
            return None
        if not self.pre.get("coding"):
            return None
        sh = self._shifted_tva()
        if not (sh.tl_s and sh.tl_e):
            return None
        if not tx.protein_id:
            return None
        n = dict(numbering="p", start=sh.tl_s, end=sh.tl_e, alt=sh.peptide(), ref=sh.peptide(ref=True))
        if not n["ref"]:
            return None
        if n["alt"] is not None and n["alt"] != n["ref"]:
            n = self._clip(n)
        # type: VEP evaluates frameshift() with the cached (unshifted) predicates but with the
        # CDS coordinates of the 3'-shifted allele
        if self._hgvsp_frameshift(sh):
            n["type"] = "fs"
        elif n["alt"] is not None:
            n["ref"] = n["ref"].replace("*", "X", 1)
            n["alt"] = n["alt"].replace("*", "X", 1)
            r, a = n["ref"], n["alt"]
            if r in ("-", ""):
                n["type"] = "ins"
            elif a in ("", "-"):
                n["type"] = "del"
            elif len(r) == 1 and len(a) == 1:
                n["type"] = ">"
            elif (len(a) > 0 and len(r) > 0 and len(a) != len(r)) or (len(a) > 1 and len(r) > 1):
                n["type"] = "delins"
            else:
                n["type"] = ">"
        else:
            rl, al = len(v.ref), len(v.alt)
            if al > 1:
                n["type"] = "ins" if n["start"] == n["end"] + 1 else ("delins" if n["start"] != n["end"] else ">")
            elif rl > 1:
                n["type"] = "del"
            else:
                return None
        n = self._hgvs_peptides(n, sh)
        if n is None:
            return None
        return f"{tx.protein_id}:p." + self._hgvsp_format(n, sh)

    def _hgvsp_frameshift(self, sh) -> bool:
        if sh is self:
            return self.frameshift()
        if self.partial_codon() or self.stop_retained():
            return False
        if sh.cds_s is None or sh.cds_e is None:
            return False
        pa_ = self.pep_alleles()
        if pa_ and pa_[0].startswith("*"):
            return False
        return abs(self.alt_len - (sh.cds_e - sh.cds_s + 1)) % 3 != 0

    def _surrounding_peptides(self, pos, original_ref, length=None):
        ref_trans = self.tx.peptide
        if original_ref and original_ref.startswith("*"):
            ref_trans += original_ref
        if len(ref_trans) <= pos:
            return None
        return ref_trans[pos - 1:pos - 1 + length] if length is not None else ref_trans[pos - 1:]

    @staticmethod
    def _shift3(n, post):
        if n["type"] == "ins":
            seq = n["alt"]
        elif n["type"] == "del":
            seq = n["ref"]
        else:
            return n
        if post is None or seq is None:
            return n
        for k in range(0, len(post) - len(seq) + 1):
            if seq[:1] == post[k:k + 1]:
                n["start"] += 1
                n["end"] += 1
                seq = seq[1:] + seq[:1]
            else:
                break
        if n["type"] == "ins":
            n["alt"] = seq
        else:
            n["ref"] = seq
        return n

    def _hgvs_peptides(self, n, sh):
        tx = self.tx
        if n["type"] == "fs":
            alt_cds = sh._alt_cds()
            if alt_cds is None:
                return None
            n_alt = len(alt_cds) // 3
            ref_trans = tx.peptide + "*"
            n["start"] = sh.tl_s
            if n["start"] > n_alt:
                n["alt"], n["type"] = "del", "del"
            else:
                while n["start"] <= n_alt:
                    n["ref"] = ref_trans[n["start"] - 1:n["start"]]
                    c = alt_cds[(n["start"] - 1) * 3:n["start"] * 3]
                    n["alt"] = CODON_TABLES[tx.table].get(c) or _translate_codon(c, tx.table)
                    if n["ref"] == "*" and n["alt"] == "*":
                        n["type"] = "="
                        break
                    if n["ref"] != n["alt"]:
                        break
                    n["start"] += 1
            if n.get("type") is None:
                return None
        elif n["type"] == "ins":
            post = self._surrounding_peptides(n["end"] + 1, n.get("original_ref"))
            if post is not None:
                n = self._shift3(n, post)
            if "*" not in n["alt"]:
                rt = tx.full_translation
                up = rt[:n["start"] - 1] + (n.get("preseq") or "")
                tns = n["start"] - len(n["alt"]) - 1
                if len(up) >= tns + len(n["alt"]) and tns >= 0:
                    if up[tns:tns + len(n["alt"])] == n["alt"]:
                        n["type"] = "dup"
                        n["end"] = n["start"] - 1
                        n["start"] -= len(n["alt"])
                        n["alt"] = _seq3(n["alt"])
            if n["type"] == "dup":
                return n
            mn = min(n["start"], n["end"])
            r = self._surrounding_peptides(mn, n.get("original_ref"), 2)
            if not r:
                return None
            n["ref"] = r
        elif n["type"] == "del":
            post = self._surrounding_peptides(n["end"] + 1, n.get("original_ref"))
            if post is not None:
                n = self._shift3(n, post)
        if n["ref"] != "-":
            n["ref"] = _seq3(n["ref"] or "")
        if n["alt"] != "-":
            n["alt"] = _seq3(n["alt"] or "")
        if n["alt"] == "-":
            n["alt"] = "del"
        if self.start_lost():
            # HGVS start-loss: p.Met1? (VEP sometimes emits malformed ranges such as p.Lys1_?0)
            first = tx.peptide[:1]
            n.update(alt="?", type="", start=1, end=1,
                     ref=_seq3(first if first and first not in "*X" else "M"))
        elif n["type"] == "del":
            if re.search(r"\w+", n["ref"] or ""):
                n["alt"] = "del"
            else:
                alt_cds = sh._alt_cds()
                if alt_cds is None:
                    return None
                st = sh.tl_s - 1
                a = _scan_translate(alt_cds, st, tx.table).split("*")[0]
                r = tx.peptide[st:]
                m = self._clip(dict(numbering="p", ref=r, alt=a, start=sh.tl_s, end=n["end"]))
                n.update(m)
                n["alt"], n["ref"] = _seq3(n["alt"]), _seq3(n["ref"])
        elif n["type"] == "fs":
            n["ref"] = n["ref"][:3]
        n["ref"] = (n["ref"] or "").replace("Xaa", "Ter")
        n["alt"] = (n["alt"] or "").replace("Xaa", "Ter")
        return n

    def _hgvsp_format(self, n, sh=None) -> str:
        t, r, a, s, e = n["type"], n["ref"], n["alt"], n["start"], n["end"]
        if r == a and t not in ("fs", "ins"):
            return f"{r}{s}="
        if self.stop_lost() and t in ("del", ">"):
            a = a[:3]
            extra = self._stop_loss_extra_aa(s - 1, sh=sh)
            a += "extTer" + (str(extra) if extra is not None else "?")
            if len(r) > 3 and t == "del":
                return f"{r[:3]}{s}_{r[-3:]}{e}{a}"
            return f"{r}{s}{a}"
        if t == "dup":
            if s < e:
                return f"{a[:3]}{s}_{a[-3:]}{e}dup"
            return f"{a}{s}dup"
        if t == ">":
            return f"{r}{s}{a}"
        if t in ("delins", "ins"):
            a = re.sub(r"Ter\w+", "Ter", a)
            first = r[:3]
            last = "Ter" if r[-1:] == "X" else r[-3:]
            if r.endswith("X"):
                extra = self._stop_loss_extra_aa(s - 1, "loss", sh=sh)
                if extra is not None:
                    a += f"extTer{extra}"
            if s == e and t == "delins":
                return f"{first}{s}{t}{a}"
            if s > e:
                s, e = e, s
            return f"{first}{s}_{last}{e}{t}{a}"
        if t == "fs":
            if a == "Ter":
                return f"{r}{s}{a}"
            extra = self._stop_loss_extra_aa(s - 1, "fs", sh=sh)
            extra = "?" if extra is None else extra
            if r == "Ter":
                extra = extra if extra == "?" else extra - 1
                return f"{r}{s}{a}extTer{extra}"
            return f"{r}{s}{a}fsTer{extra}"
        if t == "del":
            if len(r) > 3:
                return f"{r[:3]}{s}_{r[-3:]}{e}del"
            return f"{r}{s}del"
        if s != e:
            return f"{r}{s}_{a}{e}"
        return f"{r}{s}{a}"


def _format_coords(s, e) -> Optional[str]:
    if s is not None:
        if e is not None:
            if s > e:
                return f"{e}-{s}"
            if s == e:
                return str(s)
            return f"{s}-{e}"
        return f"{s}-?"
    if e is not None:
        return f"?-{e}"
    return None


# --------------------------------------------------------------------------------------
# Batch annotation
# --------------------------------------------------------------------------------------
VEP_COLUMNS = ["Uploaded_variation", "Location", "Allele", "Gene", "Feature", "Feature_type", "Consequence",
               "cDNA_position", "CDS_position", "Protein_position", "Amino_acids", "Codons", "REF_ALLELE",
               "UPLOADED_ALLELE", "IMPACT", "DISTANCE", "STRAND", "SYMBOL", "BIOTYPE", "MANE_SELECT", "TSL",
               "SOURCE", "HGVSc", "HGVSp", "HGVS_OFFSET"]


def _normalise_alleles(pos: np.ndarray, ref: Sequence[str], alt: Sequence[str]):
    """VEP VCF-parser behaviour: for indels sharing the first base, drop it and shift start."""
    starts, ends, refs, alts, valid = [], [], [], [], []
    for p, r, a in zip(pos, ref, alt):
        r = "" if r in ("-", None) else str(r).upper()
        a = "" if a in ("-", None) else str(a).upper()
        # only plain nucleotide alleles can be annotated ('.', '*', '0', '<DEL>', NaN ... cannot)
        valid.append(bool(_ALLELE_OK.match(r) and _ALLELE_OK.match(a) and (r or a) and r != a))
        p = int(p)
        e = p + len(r) - 1
        if len(r) != len(a) and r and a:
            if r[0] == a[0]:  # VCF anchor base
                r, a, p = r[1:], a[1:], p + 1
            # VEP >= 108 minimises remaining non-minimal indels (trim prefix, then suffix)
            while r and a and r[0] == a[0]:
                r, a, p = r[1:], a[1:], p + 1
            while r and a and r[-1] == a[-1]:
                r, a, e = r[:-1], a[:-1], e - 1
        starts.append(p)
        ends.append(e)
        refs.append(r)
        alts.append(a)
    return (np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64), refs, alts,
            np.array(valid, dtype=bool))


_ALLELE_OK = re.compile(r"^[ACGTN]*$")


def annotate_vep(variants: pd.DataFrame, gff: Optional[str] = None, fasta: Optional[str] = None,
                 ann: Optional[Annotation] = None, distance: int = 5000, pick: Optional[str] = None,
                 hgvs: bool = True, allele_check: str = "warn", source: Optional[str] = None,
                 regulatory: bool = True, verbose: bool = True, threads: int = 1,
                 **load_kw) -> pd.DataFrame:
    """VEP-like long table: one row per variant x transcript (or intergenic).

    ``variants`` columns: ``chrom``, ``pos``, ``ref``, ``alt`` and optionally ``id``.
    ``pick``: None (all transcripts, like plain VEP), ``"per_gene"`` (VEP --per_gene) or
    ``"worst"`` (one row per original variant, including multiallelic records).  ``allele_check``:
    ``"warn"`` (VEP behaviour, ref
    is not validated, mismatches are reported in ``ALLELE_CHECK``), ``"swap"`` (if ALT
    matches the genome and REF does not, swap them) or ``"ignore"``.  ``threads`` > 1 forks
    worker processes (Linux/macOS) for large inputs.
    """
    if ann is None:
        ann = load_annotation(gff, fasta, verbose=verbose, **load_kw)
    src = source or os.path.basename(ann.meta.get("gff", gff or "gff"))
    v = variants.reset_index(drop=True)
    v = v.assign(_src=np.arange(len(v), dtype=np.int64))
    has_multiallelic = any(isinstance(x, str) and "," in x for x in v["alt"])
    if has_multiallelic:  # multi-allelic -> one row per ALT; retain original variant id in _src
        v = v.assign(alt=[x.split(",") if isinstance(x, str) else x for x in v["alt"]]) \
            .explode("alt").reset_index(drop=True)
    ids = v["id"].astype(str).to_numpy() if "id" in v else (
        v["chrom"].astype(str) + ":" + v["pos"].astype(str)).to_numpy()
    raw_ref = np.array([str(x) if isinstance(x, str) else "?" for x in v["ref"]], dtype=object)
    raw_alt = np.array([str(x) if isinstance(x, str) else "?" for x in v["alt"]], dtype=object)
    chroms = v["chrom"].astype(str).to_numpy()
    pos = v["pos"].to_numpy(dtype=np.int64)
    src_idx = v["_src"].to_numpy(dtype=np.int64)
    _res = {c: ann.resolve(c) for c in pd.unique(chroms)}
    seqids = np.array([_res[c] for c in chroms], dtype=object)

    # Normalise once up front, but validate the uploaded REF against the FASTA at its *original*
    # VCF coordinates below.  This preserves the fast path while preventing a bad indel anchor
    # from being trimmed away before validation.  Equal-length REF/ALT swaps do not change
    # coordinates, so their already-normalised alleles can be swapped in place.
    vs, ve, refs, alts, valid = _normalise_alleles(pos, raw_ref, raw_alt)
    check = np.where(valid, "", "invalid_allele").astype(object)
    if verbose and (~valid).any():
        print(f"pyvep: {(~valid).sum()} variant(s) with alleles that cannot be annotated "
              f"('.', '*', '0', symbolic, missing or REF==ALT); left unannotated", file=sys.stderr)

    if allele_check != "ignore" and ann.fasta is not None:
        seq_name = {sid: ann.seq_name_for_fasta(sid) for sid in pd.unique(seqids) if sid is not None}
        fetch = ann.fasta.fetch
        for i in np.flatnonzero(valid):
            name = seq_name.get(seqids[i])
            r = "" if raw_ref[i] == "-" else raw_ref[i]
            if name is None or not r:
                continue
            p = int(pos[i])
            g = fetch(name, p, p + len(r) - 1)
            if g == r:
                continue
            a = "" if raw_alt[i] == "-" else raw_alt[i]
            # REF/ALT swapping is unambiguous only for equal-length alleles.  Indels with a
            # bad uploaded REF are reported rather than rewritten into a different VCF event.
            if len(a) == len(r) and g == a:
                check[i] = "swapped" if allele_check == "swap" else "ref_is_alt"
                if allele_check == "swap":
                    raw_ref[i], raw_alt[i] = a, r
                    refs[i], alts[i] = alts[i], refs[i]
            else:
                check[i] = "ref_mismatch"
        nbad = np.isin(check, ["ref_mismatch", "ref_is_alt", "swapped"]).sum()
        if verbose and nbad:
            print(f"pyvep: {nbad} variant(s) whose REF does not match the genome "
                  f"(see ALLELE_CHECK; allele_check='{allele_check}')", file=sys.stderr)
    arrays = dict(ids=ids, chroms=chroms, seqids=seqids, vs=vs, ve=ve, refs=refs, alts=alts,
                  raw_ref=raw_ref, raw_alt=raw_alt, src=src, valid=valid, src_idx=src_idx)
    n = len(v)
    # For ordinary biallelic data retain the original fast path: pick within each chunk and only
    # calculate HGVS for survivors.  For multiallelic input, --per_gene/--pick are defined per
    # original variant (not per exploded ALT), so defer the tiny final reduction until chunks are
    # concatenated.  This avoids both semantic errors and cross-chunk ALT-group splitting.
    chunk_pick = None if (has_multiallelic and pick is not None) else pick
    chunk_hgvs = False if (has_multiallelic and pick is not None) else hgvs
    if threads and threads > 1 and n >= 2000 and "fork" in _mp_methods():
        import multiprocessing as mp
        global _WORKER_STATE
        _WORKER_STATE = (ann, arrays, distance, chunk_hgvs, chunk_pick)
        bounds = np.linspace(0, n, max(threads * 4, n // _CHUNK + 1) + 1).astype(int)
        chunks = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with mp.get_context("fork").Pool(threads) as pool:
            parts = pool.map(_worker, chunks)
        _WORKER_STATE = None
    else:
        parts = [_process_chunk(ann, arrays, np.arange(a, min(a + _CHUNK, n)), distance, chunk_hgvs, chunk_pick)
                 for a in range(0, max(n, 1), _CHUNK)]
    unknown = sum(p[1] for p in parts)
    if verbose and unknown:
        print(f"pyvep: {unknown} variant(s) on chromosomes not in the annotation", file=sys.stderr)
    res = pd.concat([p[0] for p in parts], ignore_index=True) if parts else _empty_frame()
    if has_multiallelic and pick is not None and len(res):
        res["_src"] = arrays["src_idx"][res["_idx"].to_numpy(dtype=int)]
        if pick == "per_gene":
            res = _pick_per_gene(res, group_col="_src")
        elif pick in ("worst", "pick", "variant"):
            res = _pick_per_gene(res, group_col="_src")
            res = res.sort_values(["_src", "_impact", "_dist", "_rank"], kind="stable") \
                .drop_duplicates("_src", keep="first")
        if hgvs:
            _fill_hgvs(res, ann, arrays)
    res["ALLELE_CHECK"] = check[res["_idx"].to_numpy(dtype=int)]
    res["ALLELE_CHECK"] = res["ALLELE_CHECK"].replace("", None)
    if regulatory and len(ann.reg):
        regs = {}
        for sid in pd.unique(seqids):
            if sid is None:
                continue
            sel = np.where(seqids == sid)[0]
            for i, r in zip(sel, ann.regulatory(sid, vs[sel], ve[sel])):
                regs[i] = r
        res["REGULATORY"] = res["_idx"].map(regs)
    res["_src"] = arrays["src_idx"][res["_idx"].to_numpy(dtype=int)]
    for c in ("DISTANCE", "STRAND", "HGVS_OFFSET"):
        res[c] = pd.to_numeric(res[c], errors="coerce").astype("Int64")
    return res.reset_index(drop=True)


_WORKER_STATE = None
_CHUNK = 20000


def _empty_frame():
    return pd.DataFrame(columns=VEP_COLUMNS + ["_rank", "_pick", "_idx", "_tx", "_impact", "_dist"])


def _process_chunk(ann, arrays, idx, distance, hgvs, pick):
    out, unknown = _annotate_rows(ann, arrays, idx, distance, hgvs and pick is None)
    res = pd.DataFrame(out) if out else _empty_frame()
    for c in VEP_COLUMNS + ["_rank", "_pick", "_idx", "_tx"]:
        if c not in res:
            res[c] = None
    res = res.sort_values(["_idx", "_rank"], kind="stable").reset_index(drop=True)
    if pick == "per_gene":
        res = _pick_per_gene(res)
    elif pick in ("worst", "pick", "variant"):
        res = _pick_per_gene(res)
        res = res.sort_values(["_idx", "_impact", "_dist", "_rank"], kind="stable") \
            .drop_duplicates("_idx", keep="first")
    if hgvs and pick is not None:
        _fill_hgvs(res, ann, arrays)
    return res.reset_index(drop=True), unknown


def _fill_hgvs(res, ann, a):
    """HGVSc/HGVSp/HGVS_OFFSET for the (already picked) transcript rows."""
    hc, hp, ho = [None] * len(res), [None] * len(res), [None] * len(res)
    for k, (i, t) in enumerate(zip(res["_idx"].to_numpy(), res["_tx"].to_numpy())):
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        i, tx = int(i), ann.transcript(int(t))
        var = _Var(a["seqids"][i], int(a["vs"][i]), int(a["ve"][i]), a["refs"][i], a["alts"][i])
        if not _overlap(var.start, var.end, tx.start, tx.end):
            continue
        tva = _TVA(var, tx)
        c, off = tva.hgvsc()
        p = tva.hgvsp()
        hc[k], hp[k] = c, p
        if off and (c or p):
            ho[k] = off * tx.strand
    res["HGVSc"], res["HGVSp"], res["HGVS_OFFSET"] = hc, hp, ho


def _mp_methods():
    import multiprocessing as mp
    return mp.get_all_start_methods()


def _worker(idx):
    ann, arrays, distance, hgvs, pick = _WORKER_STATE
    return _process_chunk(ann, arrays, idx, distance, hgvs, pick)


def _annotate_rows(ann: Annotation, a: dict, idx: np.ndarray, distance: int, hgvs: bool):
    """Annotate the variants ``idx`` (indices into the arrays in ``a``); returns (rows, n_unknown)."""
    ids, chroms, seqids, vs, ve = a["ids"], a["chroms"], a["seqids"], a["vs"], a["ve"]
    refs, alts, raw_ref, raw_alt, src = a["refs"], a["alts"], a["raw_ref"], a["raw_alt"], a["src"]
    out, unknown = [], 0

    def inter(i):
        return dict(_idx=i, Uploaded_variation=ids[i], Location=_location(chroms[i], vs[i], ve[i]),
                    Allele=alts[i] or "-", Feature_type=None, Consequence="intergenic_variant",
                    REF_ALLELE=refs[i] or "-", UPLOADED_ALLELE=f"{raw_ref[i]}/{raw_alt[i]}", IMPACT="MODIFIER",
                    _rank=RANK["intergenic_variant"],
                    _pick=(1, 1, 1, 100, 100, 1, 1, RANK["intergenic_variant"], 0))

    bad = idx[~a["valid"][idx]]
    out.extend(dict(_idx=i, Uploaded_variation=ids[i], Location=_location(chroms[i], vs[i], ve[i]),
                    Allele=alts[i] or "-", REF_ALLELE=refs[i] or "-", UPLOADED_ALLELE=f"{raw_ref[i]}/{raw_alt[i]}",
                    _rank=1000, _pick=(1, 1, 1, 100, 100, 1, 1, 1000, 0)) for i in bad)
    idx = idx[a["valid"][idx]]
    sids = seqids[idx]
    for sid in pd.unique(sids):
        sel = idx[sids == sid]
        if sid is None:
            unknown += len(sel)
            out.extend(inter(i) for i in sel)
            continue
        vi, ti = ann.candidates(sid, vs[sel], ve[sel], distance)
        hit = set()
        cache_var = {}
        for k, t in zip(vi, ti):
            i = sel[k]
            var = cache_var.get(i)
            if var is None:
                var = cache_var[i] = _Var(sid, int(vs[i]), int(ve[i]), refs[i], alts[i])
            row = _annotate_pair(var, ann.transcript(int(t)), distance, hgvs)
            if row is None:
                continue
            hit.add(i)
            row.update(_idx=i, _tx=int(t), Uploaded_variation=ids[i], Location=_location(chroms[i], vs[i], ve[i]),
                       Allele=alts[i] or "-", REF_ALLELE=refs[i] or "-",
                       UPLOADED_ALLELE=f"{raw_ref[i]}/{raw_alt[i]}", SOURCE=src)
            out.append(row)
        out.extend(inter(i) for i in sel if i not in hit)
    return out, unknown


def _location(chrom, s, e):
    return f"{chrom}:{s}" if s == e else (f"{chrom}:{e}-{s}" if s > e else f"{chrom}:{s}-{e}")


def _annotate_pair(var: _Var, tx: Transcript, distance: int, hgvs: bool) -> Optional[dict]:
    within = _overlap(var.start, var.end, tx.start, tx.end)
    tva = _TVA(var, tx) if within else None
    row = dict(Gene=tx.gene_id, Feature=tx.feature, Feature_type="Transcript", STRAND=tx.strand,
               SYMBOL=tx.symbol, BIOTYPE=tx.biotype, MANE_SELECT=tx.feature if tx.mane_select else None,
               TSL=tx.tsl, PROTEIN_ID=tx.protein_id)
    if not within:
        up = (var.end >= tx.start - distance and var.end < tx.start) if tx.strand == 1 else \
            (var.start <= tx.end + distance and var.start > tx.end)
        down = (var.start <= tx.end + distance and var.start > tx.end) if tx.strand == 1 else \
            (var.end >= tx.start - distance and var.end < tx.start)
        if not (up or down):
            return None
        cons = ["upstream_gene_variant"] if up else ["downstream_gene_variant"]
        row["DISTANCE"] = min(abs(var.start - tx.start), abs(var.start - tx.end),
                              abs(var.end - tx.start), abs(var.end - tx.end))
    else:
        cons = tva.consequences()
        p = tva.pre
        if p.get("exon"):
            row["cDNA_position"] = _format_coords(tva.cdna_s, tva.cdna_e)
            if p.get("coding"):
                row["Amino_acids"] = tva.amino_acids()
                row["Codons"] = tva.codons()
                row["CDS_position"] = _format_coords(tva.cds_s, tva.cds_e)
                row["Protein_position"] = _format_coords(tva.tl_s, tva.tl_e)
        if hgvs:
            hc, off = tva.hgvsc()
            hp = tva.hgvsp()
            row["HGVSc"], row["HGVSp"] = hc, hp
            if off and (hc or hp):
                row["HGVS_OFFSET"] = off * tx.strand
    row["Consequence"] = ",".join(cons)
    row["IMPACT"] = IMPACT[cons[0]]
    rank = min(RANK[c] for c in cons)
    row["_rank"] = rank
    length = -((tx.cce - tx.ccs + 1 + tx.pad) if tx.translation else tx.length)
    row["_pick"] = (0 if tx.mane_select else 1, 0 if tx.mane_plus_clinical else 1,
                    0 if tx.canonical else 1, 100, tx._tsl_rank, 0 if tx.biotype == "protein_coding" else 1, 1,
                    rank, length)
    return row


def _pick_per_gene(res: pd.DataFrame, group_col: str = "_idx") -> pd.DataFrame:
    res = res.copy()
    res["_impact"] = res["IMPACT"].map(IMPACT_ORDER).fillna(100)
    res["_dist"] = pd.to_numeric(res["DISTANCE"], errors="coerce").fillna(0)
    is_tx = res["Feature_type"].eq("Transcript")
    tx = res[is_tx]
    if len(tx):
        keys = pd.DataFrame(tx["_pick"].tolist(), index=tx.index)
        tx = pd.concat([tx, keys.add_prefix("_k")], axis=1)
        # VEP's pick order; VEP breaks remaining exact ties arbitrarily (hash order), pyvep
        # deterministically prefers the closest transcript, then GFF order
        order = [group_col, "Gene"] + [f"_k{i}" for i in range(keys.shape[1])] + ["_dist"]
        tx = tx.sort_values(order, kind="stable").drop_duplicates([group_col, "Gene"], keep="first")
        tx = tx.drop(columns=[c for c in tx if c.startswith("_k")])
    return pd.concat([tx, res[~is_tx]]).sort_values(["_idx", "_rank"], kind="stable")


# --------------------------------------------------------------------------------------
# GWAS-pipeline compatible wrapper
# --------------------------------------------------------------------------------------
LEGACY_RENAME = {"SYMBOL": "gene", "Feature": "featureid", "Feature_type": "featuretype",
                 "IMPACT": "putative_impact", "DISTANCE": "distancetofeature", "HGVSc": "HGVS.c",
                 "HGVSp": "HGVS.p", "Consequence": "consequence", "Amino_acids": "aminoacids",
                 "Codons": "codons", "REF_ALLELE": "refallele", "STRAND": "strand", "BIOTYPE": "biotype",
                 "SOURCE": "source", "HGVS_OFFSET": "hgvsoffset", "Gene": "geneid",
                 "REGULATORY": "regulatory_feature", "ALLELE_CHECK": "allele_check",
                 "PROTEIN_ID": "proteinid"}
LEGACY_COLUMNS = ["featureid", "featuretype", "consequence", "aminoacids", "codons", "refallele",
                  "putative_impact", "distancetofeature", "strand", "gene", "biotype", "source", "HGVS.c",
                  "HGVS.p", "hgvsoffset", "position_cdna|cds|protein"]
EXTRA_COLUMNS = ["geneid", "proteinid", "regulatory_feature", "allele_check"]

_ANN_CACHE: Dict[tuple, Annotation] = {}


def annotate(df: pd.DataFrame, gff: str, fasta: Optional[str] = None, snpcol: str = "SNP",
             refcol: str = "A2", altcol: str = "A1", chrcol: str = "Chr", poscol: str = "bp",
             distance: int = 30000, allele_check: str = "swap", extra_columns: bool = False,
             accession: Optional[str] = None, save: Optional[str] = None, hgvs: bool = True,
             threads: int = 1, silent: bool = False, rebuild: bool = False, **load_kw) -> pd.DataFrame:
    """Annotate a variant table; drop-in for GWAS-pipeline's VEP-based ``annotate``.

    Parameters
    ----------
    df : DataFrame with ``snpcol`` ("chr:pos") and/or ``chrcol``/``poscol`` plus allele columns.
    gff : path to the GFF3 used for annotation (NCBI ``genomic.gff`` or Ensembl GFF3, may be .gz).
    fasta : genome FASTA; default: the ``*_genomic.fna`` next to the GFF.
    refcol, altcol : reference / alternate allele columns (GWAS-pipeline: A2 / A1).
    distance : up/downstream window in bp (VEP ``--distance``).
    allele_check : "swap" (default) swaps REF/ALT when ALT is the genome base, "warn" keeps VEP
        behaviour (REF taken on faith), "ignore" skips the check.
    extra_columns : also return geneid, proteinid, regulatory_feature, allele_check.
    threads : worker processes for large tables (fork; 1 = no multiprocessing).
    accession : used for the ``source`` column ("<accession>:refseq:gff"); read from the GFF header.
    save : optional CSV path to also write the result to.
    rebuild : force re-parsing the GFF into the parquet cache (``<gff>.pyvep/``).
    **load_kw : passed to :func:`load_annotation`, e.g. ``synonyms=`` (2-column chromosome
        synonym file, like VEP ``--synonyms``), ``sequence_report=``, ``cache_dir=``.
        ``sequence_report.jsonl`` / ``chrsynon`` next to the GFF are picked up automatically.

    Returns
    -------
    ``df`` (index reset, sorted by Chr/bp) with one annotation per row: the most severe
    consequence across genes (VEP --per_gene, then highest impact / nearest feature).
    """
    if len(df) == 0:
        return df
    key = (os.path.abspath(gff), fasta, repr(sorted(load_kw.items())))
    ann = _ANN_CACHE.get(key)
    if ann is None or rebuild:
        ann = _ANN_CACHE[key] = load_annotation(gff, fasta, rebuild=rebuild, verbose=not silent, **load_kw)
    acc = accession or ann.accession
    src = f"{acc}:refseq:gff" if acc else os.path.basename(gff)

    q = df.reset_index()
    if not {chrcol, poscol}.issubset(q.columns):
        sp = q[snpcol].astype(str).str.split(":", n=1, expand=True)
        q[chrcol] = sp[0]
        q[poscol] = sp[1].str.extract(r"^(\d+)", expand=False).astype(int)
    q[poscol] = q[poscol].astype(int)
    q[chrcol] = q[chrcol].map(lambda c: _chr_to_num(c, ann))
    q = q.sort_values([chrcol, poscol], kind="stable").reset_index(drop=True)
    ids = q[snpcol].astype(str) if snpcol in q else q[chrcol].astype(str) + ":" + q[poscol].astype(str)
    vin = pd.DataFrame({"chrom": q[chrcol].astype(str), "pos": q[poscol], "ref": q[refcol].to_numpy(dtype=object),
                        "alt": q[altcol].to_numpy(dtype=object), "id": ids})
    long = annotate_vep(vin, ann=ann, distance=distance, pick="worst", hgvs=hgvs, threads=threads,
                        allele_check=allele_check, source=src, verbose=not silent)
    long = long.sort_values(["_src", "_impact", "_dist", "_rank"], kind="stable") \
        .drop_duplicates("_src", keep="first").set_index("_src")
    leg = long.rename(columns=LEGACY_RENAME)
    leg["distancetofeature"] = pd.to_numeric(leg["distancetofeature"], errors="coerce").fillna(0) \
        .astype(int).map(bp2str).where(leg["consequence"].notna())
    na = lambda c: [x if isinstance(x, str) and x else "NA" for x in long[c]]
    leg["position_cdna|cds|protein"] = ["|".join(t) for t in zip(na("cDNA_position"), na("CDS_position"),
                                                                    na("Protein_position"))]
    cols = LEGACY_COLUMNS + (EXTRA_COLUMNS if extra_columns else [])
    leg = leg.reindex(columns=cols)
    for c in ("strand", "hgvsoffset"):
        leg[c] = [int(x) if pd.notna(x) else np.nan for x in leg[c]]
    # plain object columns (like the old VEP-parsing code) so that e.g. .fillna('UNK') keeps working
    leg = leg.astype(object).where(leg.notna(), np.nan)
    q = q.drop(columns=[c for c in cols if c in q.columns], errors="ignore")
    out = pd.concat([q, leg.reindex(range(len(q)))], axis=1)
    out = out.drop(columns=["index"] + [f"level_{x}" for x in range(100)], errors="ignore")
    if save:
        out.to_csv(save, index=False)
    return out


def _chr_to_num(c, ann: Annotation):
    """GWAS-pipeline chromosome codes: autosomes as ints, X=n+1, Y=n+2, MT=n+4."""
    s = str(c)
    s2 = re.sub(r"^chr", "", s, flags=re.I)
    if re.fullmatch(r"\d+(\.0)?", s2):
        return int(float(s2))
    n = ann.n_autosome
    u = s2.upper()
    if n and u in ("X", "Y", "MT", "M"):
        return {"X": n + 1, "Y": n + 2, "MT": n + 4, "M": n + 4}[u]
    sid = ann.resolve(s)
    if sid is not None:
        names = ann.chroms.set_index("seqid")["name"].to_dict()
        nm = names.get(sid)
        if nm is not None and str(nm) != s:
            return _chr_to_num(nm, ann)
    return c


# --------------------------------------------------------------------------------------
# I/O helpers + CLI
# --------------------------------------------------------------------------------------
def read_vcf(path: str) -> pd.DataFrame:
    """Minimal VCF reader (CHROM POS ID REF ALT)."""
    op = gzip.open if path.endswith((".gz", ".bgz")) else open
    rows = []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t", 5)
            rows.append((f[0], int(f[1]), f[2], f[3], f[4]))
    v = pd.DataFrame(rows, columns=["chrom", "pos", "id", "ref", "alt"])
    v["id"] = np.where(v["id"].isin([".", ""]), v["chrom"] + ":" + v["pos"].astype(str), v["id"])
    return v


def _read_any(path: str, args) -> pd.DataFrame:
    if re.search(r"\.vcf(\.b?gz)?$", path):
        return read_vcf(path)
    if path.endswith(".parquet"):
        t = pd.read_parquet(path)
    else:
        t = pd.read_csv(path, sep=None, engine="python")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pyvep", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="parse a GFF3 into the parquet cache")
    b.add_argument("--gff", required=True)
    b.add_argument("--cache-dir")
    a = sub.add_parser("annotate", help="annotate a VCF / TSV / parquet of variants")
    a.add_argument("-i", "--input", required=True, help="VCF, or table with chrom/pos/ref/alt (or SNP,A1,A2)")
    a.add_argument("-o", "--output", default="-", help="output .tsv/.csv/.parquet (default stdout)")
    a.add_argument("--gff", required=True)
    a.add_argument("--fasta")
    a.add_argument("--synonyms", help="2-column chromosome synonym file (VEP --synonyms)")
    a.add_argument("--distance", type=int, default=5000)
    a.add_argument("--pick", choices=["all", "per_gene", "worst"], default="all")
    a.add_argument("--allele-check", choices=["warn", "swap", "ignore"], default="warn")
    a.add_argument("--legacy", action="store_true", help="GWAS-pipeline column layout (one row per variant)")
    a.add_argument("--no-hgvs", action="store_true")
    a.add_argument("--threads", type=int, default=1)
    a.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build_cache(args.gff, args.cache_dir)
        return 0
    t = _read_any(args.input, args)
    if args.legacy:
        if {"chrom", "pos", "ref", "alt"}.issubset(t.columns) and "SNP" not in t.columns:
            t = t.assign(SNP=t["chrom"].astype(str) + ":" + t["pos"].astype(str), A2=t["ref"], A1=t["alt"]) \
                .drop(columns=["chrom", "pos", "ref", "alt"])
        res = annotate(t, args.gff, fasta=args.fasta, distance=args.distance, allele_check=args.allele_check,
                       silent=args.quiet, hgvs=not args.no_hgvs, synonyms=args.synonyms, threads=args.threads)
    else:
        cols = {c.lower(): c for c in t.columns}
        if not {"chrom", "pos", "ref", "alt"}.issubset(cols):
            if "#chrom" in cols:
                t = t.rename(columns={cols["#chrom"]: "chrom"})
            elif "snp" in cols:
                sp = t[cols["snp"]].astype(str).str.split(":", expand=True)
                t = t.assign(chrom=sp[0], pos=sp[1].astype(int), id=t[cols["snp"]],
                             ref=t[cols.get("a2", "A2")], alt=t[cols.get("a1", "A1")])
        t = t.rename(columns={cols[k]: k for k in ("chrom", "pos", "ref", "alt", "id") if k in cols})
        res = annotate_vep(t, args.gff, fasta=args.fasta, distance=args.distance,
                           pick=None if args.pick == "all" else args.pick, hgvs=not args.no_hgvs,
                           allele_check=args.allele_check, verbose=not args.quiet, synonyms=args.synonyms,
                           threads=args.threads)
        res = res[[c for c in VEP_COLUMNS + ["PROTEIN_ID", "REGULATORY", "ALLELE_CHECK"] if c in res]]
    out = args.output
    try:
        if out.endswith(".parquet"):
            res.to_parquet(out, index=False)
        else:
            res.to_csv(sys.stdout if out == "-" else out, sep="," if out.endswith(".csv") else "\t",
                       index=False, na_rep="-" if not args.legacy else "")
    except BrokenPipeError:  # e.g. piped into `head`
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
