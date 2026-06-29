# df = pd.DataFrame(glob('/tscc/projects/ps-palmer/hs_rats/20251223_SH00252_0250_ASC2146269_SC3/GRCr8/03_sample_processing/picard_markdup/*.bam'), columns= ['bamfile'])
# df['rfid'] = df.bamfile.str.extract(r'_(933\d+)_', expand = False)
# df['barcode'] = df.bamfile.str.extract(r'_933\d+_([ATCG]+)', expand = False)
# df['Sample_ID'] = df.bamfile.str.extract(r'(Aneuploidy_BC8_933\d+_[ATCG]+)', expand = False)
# df = df.set_index('Sample_ID').combine_first(pd.read_csv('impute_riptide_svm_output_outcome.csv', index_col = 'Sample_ID'))
# df['ploidy'] = df.index.map({'Aneuploidy_BC8_933000321283650_CTCTCACA':'XXY',
# 'Aneuploidy_BC8_933000321075840_CTTCCAAC':'XXY',
# 'Aneuploidy_BC8_933000321284183_TGACCACA':'XYY',
# 'Aneuploidy_BC8_933000321283646_CGTAGCTA':'XYY'})
# df['ploidy'] = df.ploidy.fillna(df.sex.map({'M':'XY', 'F':'XX'}))

import os
import re
import heapq
import pysam
import pandas as pd
from typing import List, Tuple, Dict, Optional

_RE_HAS_LETTER = re.compile(r"[A-Za-z]")

def _md_mismatches_fast(md: str) -> List[Tuple[int, str]]:
    # Parse MD -> (ref_offset, ref_base) for SNVs; skip deletions as variants
    out: List[Tuple[int, str]] = []
    ref_off = 0
    i = 0
    n = len(md)
    while i < n:
        c = md[i]
        oc = ord(c)
        # digits
        if 48 <= oc <= 57:
            num = 0
            while i < n:
                oc2 = ord(md[i])
                if 48 <= oc2 <= 57:
                    num = num * 10 + (oc2 - 48)
                    i += 1
                else:  break
            ref_off += num
            continue

        # deletion: ^ACGT...
        if c == "^":
            i += 1
            start = i
            while i < n:
                oc2 = ord(md[i])
                if (65 <= oc2 <= 90) or (97 <= oc2 <= 122):
                    i += 1
                else: break
            ref_off += (i - start)
            continue
        # mismatch base
        if (65 <= oc <= 90) or (97 <= oc <= 122):
            out.append((ref_off, c))
            ref_off += 1
            i += 1
            continue
        i += 1
    return out


def _map_ref_offsets_to_qpos(cigartuples, ref_offsets):
    # Map each ref_offset to query pos; -1 if not in aligned M/=/X
    qpos = 0
    roff = 0
    j = 0
    m = len(ref_offsets)
    out = [-1] * m
    for op, length in cigartuples:
        if j >= m: break
        if op in (0, 7, 8):  # M, =, X
            block_end = roff + length
            while j < m and ref_offsets[j] < block_end:
                out[j] = qpos + (ref_offsets[j] - roff)
                j += 1
            qpos += length
            roff = block_end
        elif op in (2, 3): roff += length
        elif op in (1, 4): qpos += length
        else:  pass

    return out


def chrY_extract_fast(
    bam_path: str,
    contig_candidates=("chrY", "Y", "ChrY", "NC_086040.1"),
    baseq_threshold: int = 30,          # keep if baseQ > 30
    min_mapq: int = 10,
    include_secondary: bool = False,
    include_supplementary: bool = False,
    include_duplicates: bool = False,
    include_qcfail: bool = False,
    max_reads: Optional[int] = None,
) -> pd.DataFrame:
    # Ensure index exists
    bai1 = bam_path + ".bai"
    bai2 = os.path.splitext(bam_path)[0] + ".bai"
    if not (os.path.exists(bai1) or os.path.exists(bai2)): pysam.index(bam_path)

    pos0_list = []
    end0_list = []
    mate_pos0_list = []
    mate_end0_list = []
    variants_list = []
    y_list = []
    pending = {}  # qname -> row_idx for mate_end0 fill (chrY-only)
    lane_heap = []  # (end0, lane_id)
    qual_list = []
    c_list = []
    next_lane_id = 0
    thr = baseq_threshold
    has_letter = _RE_HAS_LETTER.search  # local bind (speed)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        refs = set(bam.references)
        y_contig = next((c for c in contig_candidates if 'chrY' in c), None)
        if y_contig is None:
            raise ValueError(f"Could not find chrY contig in BAM. First 30 refs: {list(bam.references)[:30]}")

        for i, r in enumerate(bam.fetch(y_contig)):
            if max_reads is not None and i >= max_reads: break
            if r.is_unmapped: continue
            if r.is_qcfail and not include_qcfail: continue
            if r.is_duplicate and not include_duplicates: continue
            if r.is_secondary and not include_secondary: continue
            if r.is_supplementary and not include_supplementary: continue
            if r.mapping_quality < min_mapq: continue
            pos0 = r.reference_start
            end0 = r.reference_end
            if pos0 is None or end0 is None or end0 <= pos0: continue
            if lane_heap and lane_heap[0][0] <= pos0:  _, lane_id = heapq.heappop(lane_heap)
            else:
                lane_id = next_lane_id
                next_lane_id += 1
            heapq.heappush(lane_heap, (end0, lane_id))
            mate_pos0 = r.next_reference_start if (r.is_paired and not r.mate_is_unmapped) else None
            mate_end0 = None
            # ---- Variant extraction: gate hard, do minimal work on most reads ----
            variants = None  # allocate dict only if we actually record something
            # Fastest skip: NM==0 => no mismatches/indels, so no SNV variants
            if r.has_tag("NM") and r.get_tag("NM") == 0: pass
            else:
                # Need MD for mismatch positions
                if r.has_tag("MD") and r.cigartuples is not None:
                    md = r.get_tag("MD")
                    # If MD has no letters, there are no mismatches (only matches and maybe deletions)
                    if has_letter(md) is not None:
                        mm = _md_mismatches_fast(md)
                        if mm:
                            ref_offsets = [x[0] for x in mm]
                            qposes = _map_ref_offsets_to_qpos(r.cigartuples, ref_offsets)
                            qseq = r.query_sequence
                            quals = r.query_qualities
                            if qseq is not None and quals is not None:
                                ref_start = pos0
                                for (ref_off, ref_base), qpos in zip(mm, qposes):
                                    if qpos < 0:
                                        continue
                                    if quals[qpos] <= thr:
                                        continue
                                    refb = ref_base.upper()
                                    alt = qseq[qpos].upper()
                                    if refb in "ACGT" and alt in "ACGT" and alt != refb:
                                        if variants is None:
                                            variants = {}
                                        variants[ref_start + ref_off] = alt

            if variants is None: variants = {}
            row_idx = len(pos0_list)
            c_list.append(r.reference_name)
            pos0_list.append(pos0)
            end0_list.append(end0)
            mate_pos0_list.append(mate_pos0)
            mate_end0_list.append(mate_end0)
            variants_list.append(variants)
            y_list.append(lane_id)
            qual_list.append(r.mapping_quality)
            # mate_end0 only if mate also appears
            if r.is_paired and not r.mate_is_unmapped:
                qn = r.query_name
                prev = pending.get(qn)
                if prev is None: pending[qn] = row_idx
                else:
                    mate_end0_list[row_idx] = end0_list[prev]
                    mate_end0_list[prev] = end0
                    del pending[qn]

    res = pd.DataFrame({'chr': c_list,"pos0": pos0_list, "end0": end0_list, "mate_pos0": mate_pos0_list,
                         "mate_end0": mate_end0_list,"variants": variants_list,"y": y_list, 'mapping_qual': qual_list})
    res.loc[res.mate_end0.isna(), 'mate_pos0'] = np.nan
    res[['read_start', 'read_end']] = res.filter(regex='0').agg(['min', 'max'], axis = 1).set_axis(['read_start', 'read_end'], axis = 1)
    res[['fw_start', 'fw_end']] = res[['pos0', 'end0']].agg(['min', 'max'], axis = 1).set_axis(['fw_start', 'fw_end'], axis = 1)
    res[['mate_start', 'mate_end']] = res[['mate_pos0', 'mate_end0']].agg(['min', 'max'], axis = 1).set_axis(['mate_start', 'mate_end'], axis = 1)
    return res

def add_cols(bamfile, rfid, min_mapq = 15):
    t = chrY_extract_fast(bamfile,  max_reads=100000000,min_mapq=min_mapq).assign(rfid = rfid)
    mut = pd.concat(t[t.variants.map(len).gt(0)].apply(lambda x: pd.DataFrame.from_dict([x.variants]).assign(y = x.y, mapping_qual = x.mapping_qual)\
                                                        .melt(id_vars=['mapping_qual','y'], var_name='bp', value_name='mutation' ), axis = 1).to_list())
    mut['color'] = mut.mutation.map({'A': 'red','C':'blue', 'G':'orange', 'T':'green'})
    return t, mut.assign(rfid = rfid)
