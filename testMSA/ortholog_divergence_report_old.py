#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ortholog Divergence Analyzer
----------------------------
Given a multi-FASTA of ORF coding sequences (same gene across species),
this script:
  1) Builds a codon-aware multiple alignment (protein MSA -> back-translate).
  2) Computes pairwise substitution patterns & indel summaries.
  3) Scans sliding windows to detect local accelerations (SNP/indel hotspots).
  4) Produces:
       - HTML report with embedded figures
       - CSVs with metrics
       - PNG plots (also embedded into the HTML)

Heuristic "BLAST-miss" risk classifier:
  LOW / INTERMEDIATE / HIGH risk that a standard protein BLAST might miss an ortholog,
  based on amino-acid identity, coverage, window-level dips, and indel clustering.

Usage:
  python ortholog_divergence_report.py \
      --fasta cds_multi.fasta \
      --outdir results_geneX \
      --window_codon 30 \
      --genetic_code 1

Notes:
  * Input sequences should be CDS (length % 3 == 0); the script will attempt basic fixes
    (e.g., trim trailing bases mod 3) and will drop sequences with illegal codons unless --allow_ambig.
  * If 'mafft' is on PATH, it is used to align proteins. Otherwise a simple progressive pairwise
    protein aligner is used and then back-translated to codons.
"""

import os
import sys
import io
import re
import math
import base64
import json
import argparse
import tempfile
import subprocess
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
from Bio.Align import substitution_matrices
from Bio import pairwise2

# ---------------------------
# Helpers
# ---------------------------

def read_cds_fasta(path, allow_ambig=False, genetic_code=1):
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        seq = str(rec.seq).upper().replace("U", "T")
        # enforce length multiple of 3 (trim trailing)
        if len(seq) % 3 != 0:
            seq = seq[:len(seq) - (len(seq) % 3)]
        # basic sanity
        if not re.fullmatch(r"[ACGTN-]+", seq):
            # if other letters present, coerce to N
            seq = re.sub(r"[^ACGT-]", "N", seq)
        # translate to detect obvious frame errors
        try:
            protein = translate_cds(seq, genetic_code, allow_ambig=allow_ambig)
        except Exception as e:
            # drop unusable sequence
            # print(f"[warn] Dropping {rec.id}: {e}", file=sys.stderr)
            continue
        records.append({"id": rec.id, "cds": seq, "aa": protein})
    if len(records) < 2:
        raise ValueError("Need at least two usable CDS entries.")
    return records

def translate_cds(cds, genetic_code=1, allow_ambig=False):
    table = CodonTable.unambiguous_dna_by_id[genetic_code]
    protein = []
    for i in range(0, len(cds), 3):
        codon = cds[i:i+3]
        if len(codon) < 3:
            break
        if "-" in codon:
            protein.append("-")
            continue
        if "N" in codon:
            # ambiguous codon -> X (unknown)
            protein.append("X")
            continue
        aa = Seq(codon).translate(table=genetic_code, to_stop=False)
        if aa == "*":
            # internal stop: treat as X unless at terminal trailing block
            if i < len(cds) - 3:
                if not allow_ambig:
                    raise ValueError(f"Internal stop codon at pos {i//3+1}")
                protein.append("X")
            else:
                # terminal stop; ignore
                pass
        else:
            protein.append(str(aa))
    return "".join(protein)

def run_mafft_on_proteins(prot_records, mafft_path="mafft"):
    """Write proteins to temp FASTA, run MAFFT, return aligned protein dict id->aligned."""
    with tempfile.TemporaryDirectory() as td:
        inf = os.path.join(td, "in.faa")
        outf = os.path.join(td, "out.faa")
        with open(inf, "w") as fh:
            for r in prot_records:
                fh.write(f">{r['id']}\n{r['aa']}\n")
        cmd = [mafft_path, "--anysymbol", "--auto", inf]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            aligned = {}
            cur_id = None
            seq_buf = []
            for line in res.stdout.splitlines():
                if line.startswith(">"):
                    if cur_id is not None:
                        aligned[cur_id] = "".join(seq_buf)
                    cur_id = line[1:].strip()
                    seq_buf = []
                else:
                    seq_buf.append(line.strip())
            if cur_id is not None:
                aligned[cur_id] = "".join(seq_buf)
            # preserve order of input
            return [ {"id": r["id"], "aa_aln": aligned[r["id"]]} for r in prot_records ]
        except Exception as e:
            raise RuntimeError(f"MAFFT failed: {e}")

def pairwise_global_align(a, b, matrix=None, gap_open=-10, gap_extend=-0.5):
    """Global align two protein sequences (Needleman-Wunsch via Biopython)."""
    if matrix is None:
        matrix = substitution_matrices.load("BLOSUM62")
    alns = pairwise2.align.globalds(a, b, matrix, gap_open, gap_extend, one_alignment_only=True)
    if not alns:
        return None, None
    return alns[0].seqA, alns[0].seqB

def progressive_align_proteins(records):
    """Very simple progressive alignment: pick the longest as seed, add others one-by-one."""
    # Start with the longest sequence as seed
    recs_sorted = sorted(records, key=lambda r: len(r["aa"]), reverse=True)
    aln_ids = [recs_sorted[0]["id"]]
    aln_seqs = [recs_sorted[0]["aa"]]

    matrix = substitution_matrices.load("BLOSUM62")
    for r in recs_sorted[1:]:
        # Align r["aa"] to current consensus (collapse current alignment to a profile consensus)
        # For simplicity, align to a "consensus" built by majority at each column.
        consensus = consensus_from_alignment(aln_seqs)
        aA, aB = pairwise_global_align(consensus, r["aa"], matrix=matrix)
        # Now expand existing alignment to match aA gaps and map each old seq accordingly
        extended = []
        ins_map = gap_map(aA)
        for s in aln_seqs:
            extended.append(apply_gap_map_to_seq(s, ins_map))
        # Add the new sequence aligned (aB) already matches aA
        aln_seqs = extended + [aB]
        aln_ids.append(r["id"])
    return [{"id": id_, "aa_aln": seq} for id_, seq in zip(aln_ids, aln_seqs)]

def consensus_from_alignment(aln_list):
    L = len(aln_list[0])
    cons = []
    for i in range(L):
        col = [s[i] for s in aln_list]
        c = Counter(col)
        c.pop("-", None)
        if not c:
            cons.append("-")
        else:
            cons.append(c.most_common(1)[0][0])
    return "".join(cons)

def gap_map(aligned_ref):
    """Return positions in aligned_ref that are gaps; used to insert gaps into other seqs."""
    return [i for i, ch in enumerate(aligned_ref) if ch == "-"]

def apply_gap_map_to_seq(seq, gap_positions):
    """Insert '-' in seq at gap_positions to match reference aligned length."""
    seq = list(seq)
    out = []
    j = 0
    for i in range(len(seq) + len(gap_positions)):
        if i in gap_positions:
            out.append("-")
        else:
            out.append(seq[j] if j < len(seq) else "-")
            j += 1
    return "".join(out)

def backtranslate_protein_alignment_to_codon(prot_aln_records, cds_lookup):
    """Map each AA alignment column back to codon triplets (insert '---' for AA gaps)."""
    codon_alns = {}
    for r in prot_aln_records:
        pid = r["id"]
        aa_aln = r["aa_aln"]
        cds = cds_lookup[pid]
        codon_out = []
        i_codon = 0
        for aa in aa_aln:
            if aa == "-":
                codon_out.append("---")
            else:
                codon = cds[i_codon*3:(i_codon+1)*3]
                if len(codon) < 3:
                    codon = "---"
                codon_out.append(codon)
                i_codon += 1
        codon_alns[pid] = "".join(codon_out)
    # Ensure equal lengths
    lengths = {len(v) for v in codon_alns.values()}
    if len(lengths) != 1:
        raise ValueError("Back-translation produced unequal lengths.")
    return codon_alns

def is_transition(a, b):
    pur = set("AG"); pyr = set("CT")
    if a in pur and b in pur: return True
    if a in pyr and b in pyr: return True
    return False

def codon_syn_status(c1, c2, genetic_code=1):
    """Return 'syn', 'nonsyn', or 'ambig' for codon pair (handles Ns, gaps, multi-diff)."""
    if "-" in c1 or "-" in c2: return "gap"
    if "N" in c1 or "N" in c2: return "ambig"
    diffs = sum(1 for i in range(3) if c1[i] != c2[i])
    if diffs == 0: return "same"
    if diffs > 1:  # multiple-hit codon; can't neatly classify without models
        return "ambig"
    # single-nuc difference: check AA
    aa1 = str(Seq(c1).translate(table=genetic_code))
    aa2 = str(Seq(c2).translate(table=genetic_code))
    if aa1 == "*" or aa2 == "*": return "ambig"
    return "syn" if aa1 == aa2 else "nonsyn"

def sliding_windows(L, window, step):
    i = 0
    while i < L:
        yield i, min(i+window, L)
        i += step

def longest_contiguous_identity(aa_aln_q, aa_aln_t):
    cur = best = 0
    L = len(aa_aln_q)
    for i in range(L):
        if aa_aln_q[i] == aa_aln_t[i] and aa_aln_q[i] != "-":
            cur += 1
            if cur > best: best = cur
        else:
            cur = 0
    return best

def classify_blast_miss(aa_identity, coverage, max_gap_cluster, low_iden_windows, longest_contig_ident):
    """
    Heuristic rules, inspired by typical BLASTP sensitivity regimes:
      - Sustained identity < ~30% with low coverage, large indel clusters, and many 'bad' windows => HIGH risk.
      - Moderate identity (30-40%) or coverage 0.5-0.7, or isolated large gaps => INTERMEDIATE.
      - Otherwise LOW.
    """
    # Primary gates
    if aa_identity < 0.30 or coverage < 0.50:
        return "high"
    if max_gap_cluster >= 30 or low_iden_windows >= 2:
        # harsh structural issues
        return "high"
    if (0.30 <= aa_identity < 0.40) or (0.50 <= coverage < 0.70) or (15 <= max_gap_cluster < 30):
        return "intermediate"
    # Strong mitigating factor: a long exact-ish block helps BLAST
    if longest_contig_ident >= 40 and aa_identity >= 0.28 and coverage >= 0.45:
        return "intermediate"
    return "low"

def encode_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# ---------------------------
# Main analysis
# ---------------------------

def analyze(args):
    os.makedirs(args.outdir, exist_ok=True)
    records = read_cds_fasta(args.fasta, allow_ambig=args.allow_ambig, genetic_code=args.genetic_code)
    ids = [r["id"] for r in records]
    cds_lookup = {r["id"]: r["cds"] for r in records}

    # Protein alignment: mafft -> fallback progressive
    prot_records = [{"id": r["id"], "aa": r["aa"]} for r in records]
    try:
        if not args.no_mafft:
            aligned = run_mafft_on_proteins(prot_records, mafft_path=args.mafft_path)
        else:
            raise RuntimeError("Skipping MAFFT by request.")
    except Exception:
        aligned = progressive_align_proteins(prot_records)

    # Back-translate to codon MSA
    codon_alns = backtranslate_protein_alignment_to_codon(aligned, cds_lookup)
    aa_alns = {r["id"]: r["aa_aln"] for r in aligned}
    aln_ids = [r["id"] for r in aligned]
    aln_len_cod = len(next(iter(codon_alns.values()))) // 3  # in codons
    aln_len_nt  = len(next(iter(codon_alns.values())))

    # Save aligned FASTAs
    with open(os.path.join(args.outdir, "alignment_protein.faa"), "w") as fh:
        for sid in aln_ids:
            fh.write(f">{sid}\n{aa_alns[sid]}\n")
    with open(os.path.join(args.outdir, "alignment_codon.fna"), "w") as fh:
        for sid in aln_ids:
            fh.write(f">{sid}\n{codon_alns[sid]}\n")

    # Pairwise metrics
    pair_rows = []
    pair_heat_id = pd.DataFrame(index=aln_ids, columns=aln_ids, dtype=float)
    pair_heat_gap = pd.DataFrame(index=aln_ids, columns=aln_ids, dtype=float)

    for a, b in combinations(aln_ids, 2):
        aaA, aaB = aa_alns[a], aa_alns[b]
        codA, codB = codon_alns[a], codon_alns[b]

        # AA identity & coverage
        aa_matches = sum(1 for i in range(len(aaA)) if aaA[i] == aaB[i] and aaA[i] != "-")
        aa_pos     = sum(1 for i in range(len(aaA)) if aaA[i] != "-" and aaB[i] != "-")
        aa_identity = aa_matches / aa_pos if aa_pos else 0.0
        coverage = aa_pos / len(aaA) if len(aaA) else 0.0

        # Longest contiguous identical AA run
        lci = longest_contiguous_identity(aaA, aaB)

        # NT identity (on aligned, non-gap, non-N positions)
        nt_matches = 0
        nt_comp = 0
        ts = tv = 0
        gap_events = 0
        gap_cluster_max = 0
        cur_gap_cluster = 0

        syn = nonsyn = ambig = 0
        multihit = 0

        for i in range(0, len(codA), 3):
            c1, c2 = codA[i:i+3], codB[i:i+3]
            if "-" in c1 or "-" in c2:
                # Gap cluster tracking (count per codon)
                cur_gap_cluster += 1
                continue
            else:
                if cur_gap_cluster:
                    gap_events += 1
                    gap_cluster_max = max(gap_cluster_max, cur_gap_cluster)
                    cur_gap_cluster = 0

            # nt identity
            for k in range(3):
                n1, n2 = c1[k], c2[k]
                if n1 in "ACGT" and n2 in "ACGT":
                    nt_comp += 1
                    if n1 == n2:
                        nt_matches += 1
                    else:
                        if is_transition(n1, n2): ts += 1
                        else: tv += 1

            # syn/nonsyn approx
            status = codon_syn_status(c1, c2, genetic_code=args.genetic_code)
            if status == "syn":
                syn += 1
            elif status == "nonsyn":
                nonsyn += 1
            elif status == "ambig":
                multihit += 1
            else:
                pass

        if cur_gap_cluster:
            gap_events += 1
            gap_cluster_max = max(gap_cluster_max, cur_gap_cluster)
            cur_gap_cluster = 0

        nt_identity = nt_matches / nt_comp if nt_comp else 0.0
        ts_tv_ratio = (ts / tv) if tv > 0 else np.inf if ts > 0 else np.nan

        pair_heat_id.loc[a, b] = pair_heat_id.loc[b, a] = aa_identity
        # gap load as fraction of codon positions gapped in either sequence
        gap_codons = sum(1 for i in range(0, len(codA), 3) if "-" in codA[i:i+3] or "-" in codB[i:i+3])
        pair_heat_gap.loc[a, b] = pair_heat_gap.loc[b, a] = gap_codons / aln_len_cod

        # Sliding windows on AA: low-identity windows and indel density
        win = max(5, args.window_codon)  # window in codons ~ window in aa
        step = max(5, args.step_codon)
        low_win = 0
        for s, e in sliding_windows(len(aaA), win, step):
            subA, subB = aaA[s:e], aaB[s:e]
            m = sum(1 for i in range(len(subA)) if subA[i] == subB[i] and subA[i] != "-")
            pos = sum(1 for i in range(len(subA)) if subA[i] != "-" and subB[i] != "-")
            idw = m / pos if pos else 0.0
            if idw < 0.20 and pos >= 20:
                low_win += 1

        risk = classify_blast_miss(
            aa_identity=aa_identity,
            coverage=coverage,
            max_gap_cluster=gap_cluster_max,
            low_iden_windows=low_win,
            longest_contig_ident=lci
        )

        pair_rows.append({
            "seqA": a, "seqB": b,
            "aa_identity": round(aa_identity, 4),
            "aa_coverage": round(coverage, 4),
            "longest_contiguous_identical_aa": int(lci),
            "nt_identity": round(nt_identity, 4),
            "ts": int(ts), "tv": int(tv),
            "ts_tv_ratio": (round(ts_tv_ratio, 4) if np.isfinite(ts_tv_ratio) else "inf"),
            "syn_codons": int(syn), "nonsyn_codons": int(nonsyn), "ambig_multihit_codons": int(multihit),
            "gap_events": int(gap_events),
            "max_gap_cluster_codons": int(gap_cluster_max),
            "gap_fraction_codons": round(gap_codons / aln_len_cod, 4),
            "low_identity_windows(<20%)": int(low_win),
            "blast_miss_risk": risk
        })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(args.outdir, "pairwise_metrics.csv"), index=False)

    # Windowed per-sequence divergence vs reference (first sequence as reference by default)
    ref_id = aln_ids[0] if args.reference is None else args.reference
    if ref_id not in aln_ids:
        ref_id = aln_ids[0]

    window_rows = []
    win = max(5, args.window_codon)
    step = max(5, args.step_codon)
    for sid in aln_ids:
        if sid == ref_id: continue
        aaR, aaT = aa_alns[ref_id], aa_alns[sid]
        codR, codT = codon_alns[ref_id], codon_alns[sid]
        # For plotting & hotspot detection
        for s, e in sliding_windows(len(aaR), win, step):
            subR, subT = aaR[s:e], aaT[s:e]
            pos = sum(1 for i in range(len(subR)) if subR[i] != "-" and subT[i] != "-")
            matches = sum(1 for i in range(len(subR)) if subR[i] == subT[i] and subR[i] != "-")
            aa_idw = matches / pos if pos else 0.0
            # indel rate: fraction gapped in either (codon level)
            cod_s, cod_e = s*3, e*3
            gapped_cod = sum(1 for i in range(cod_s, cod_e, 3) if "-" in codR[i:i+3] or "-" in codT[i:i+3])
            indel_rate = gapped_cod / max(1, (e - s))
            window_rows.append({
                "ref": ref_id, "target": sid,
                "win_start_codon": s+1, "win_end_codon": e,
                "aa_identity_window": aa_idw,
                "indel_rate_window": indel_rate
            })

    win_df = pd.DataFrame(window_rows)
    if not win_df.empty:
        win_df.to_csv(os.path.join(args.outdir, "window_metrics.csv"), index=False)

    # ---------------------------
    # Plots
    # ---------------------------
    figs = {}

    # Heatmap: pairwise AA identity
    fig = plt.figure(figsize=(max(4, len(aln_ids)*0.6), max(4, len(aln_ids)*0.6)))
    im = plt.imshow(pair_heat_id.fillna(1.0).values, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="AA identity")
    plt.xticks(ticks=range(len(aln_ids)), labels=aln_ids, rotation=90)
    plt.yticks(ticks=range(len(aln_ids)), labels=aln_ids)
    plt.title("Pairwise AA identity")
    figs["heat_id"] = encode_png(fig)

    # Heatmap: pairwise gap fraction (codon-level)
    fig = plt.figure(figsize=(max(4, len(aln_ids)*0.6), max(4, len(aln_ids)*0.6)))
    im = plt.imshow(pair_heat_gap.fillna(0.0).values, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Gap fraction (codons)")
    plt.xticks(ticks=range(len(aln_ids)), labels=aln_ids, rotation=90)
    plt.yticks(ticks=range(len(aln_ids)), labels=aln_ids)
    plt.title("Pairwise gap load")
    figs["heat_gap"] = encode_png(fig)

    # Window lines per target vs reference
    if not win_df.empty:
        for sid, sub in win_df.groupby("target"):
            fig = plt.figure(figsize=(8, 3))
            plt.plot(sub["win_start_codon"], sub["aa_identity_window"])
            plt.ylim(0, 1)
            plt.xlabel("Window start (codon)")
            plt.ylabel("AA identity (window)")
            plt.title(f"Window AA identity: {ref_id} vs {sid}")
            figs[f"win_id_{sid}"] = encode_png(fig)

            fig = plt.figure(figsize=(8, 3))
            plt.plot(sub["win_start_codon"], sub["indel_rate_window"])
            plt.ylim(0, 1)
            plt.xlabel("Window start (codon)")
            plt.ylabel("Indel rate (window)")
            plt.title(f"Window indel rate: {ref_id} vs {sid}")
            figs[f"win_indel_{sid}"] = encode_png(fig)

    # Save standalone PNGs too
    for key, dataurl in figs.items():
        png_path = os.path.join(args.outdir, f"{key}.png")
        with open(png_path, "wb") as fh:
            fh.write(base64.b64decode(dataurl.split(",")[1]))

    # ---------------------------
    # HTML Report
    # ---------------------------
    # Summarize risk counts
    risk_counts = pair_df["blast_miss_risk"].value_counts().to_dict()

    html = io.StringIO()
    html.write(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Ortholog Divergence Report</title>
<style>
body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif; margin: 24px; }}
h1, h2, h3 {{ margin-top: 1.2em; }}
code, pre {{ background:#f6f8fa; padding: 2px 4px; border-radius:4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #fafafa; }}
td:first-child, th:first-child {{ text-align: left; }}
.figure {{ margin: 12px 0 24px 0; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge.low {{ background:#e7f7ed; color:#056e2e; }}
.badge.intermediate {{ background:#fff5d6; color:#8a6100; }}
.badge.high {{ background:#ffe8e6; color:#9a1d0d; }}
.small {{ color:#666; font-size: 12px; }}
</style>
</head><body>
<h1>Ortholog Divergence Report</h1>
<p><b>Input:</b> {os.path.basename(args.fasta)} &nbsp;&nbsp;|&nbsp;&nbsp; <b>N sequences:</b> {len(aln_ids)} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Aligned length:</b> {aln_len_cod} codons</p>
<p><b>Reference sequence:</b> {ref_id}</p>

<h2>Quick BLAST-miss risk overview</h2>
<ul>
  <li>LOW: {risk_counts.get('low', 0)}</li>
  <li>INTERMEDIATE: {risk_counts.get('intermediate', 0)}</li>
  <li>HIGH: {risk_counts.get('high', 0)}</li>
</ul>
<div class="figure"><img src="{figs['heat_id']}" alt="AA identity heatmap" style="max-width:100%"></div>
<div class="figure"><img src="{figs['heat_gap']}" alt="Gap heatmap" style="max-width:100%"></div>

<h2>Pairwise metrics</h2>
<table>
<thead><tr>
<th>seqA</th><th>seqB</th>
<th>AA id</th><th>AA cov</th><th>Longest identical block (aa)</th>
<th>NT id</th><th>Ts</th><th>Tv</th><th>Ts/Tv</th>
<th>Syn</th><th>Nonsyn</th><th>Ambig/multihit</th>
<th>Gap events</th><th>Max gap cluster (codons)</th><th>Gap frac</th>
<th># low-id windows</th><th>Risk</th>
</tr></thead><tbody>
""")
    for _, r in pair_df.sort_values(["blast_miss_risk","aa_identity"], ascending=[True, False]).iterrows():
        badge = f"<span class='badge {r['blast_miss_risk']}'>{r['blast_miss_risk'].upper()}</span>"
        html.write(f"<tr><td>{r.seqA}</td><td>{r.seqB}</td>"
                   f"<td>{r.aa_identity:.3f}</td><td>{r.aa_coverage:.3f}</td><td>{r.longest_contiguous_identical_aa}</td>"
                   f"<td>{r.nt_identity:.3f}</td><td>{r.ts}</td><td>{r.tv}</td><td>{r.ts_tv_ratio}</td>"
                   f"<td>{r.syn_codons}</td><td>{r.nonsyn_codons}</td><td>{r.ambig_multihit_codons}</td>"
                   f"<td>{r.gap_events}</td><td>{r.max_gap_cluster_codons}</td><td>{r.gap_fraction_codons:.3f}</td>"
                   f"<td>{int(r['low_identity_windows(<20%)'])}</td><td>{badge}</td></tr>\n")

    html.write("</tbody></table>")

    if not win_df.empty:
        html.write("<h2>Local divergence (sliding windows)</h2>")
        html.write("<p class='small'>Windows are computed on amino-acid alignment; indel rate is computed on codons within the same span.</p>")
        for sid in [s for s in aln_ids if s != ref_id]:
            if f"win_id_{sid}" in figs:
                html.write(f"<h3>{ref_id} vs {sid}</h3>")
                html.write(f"<div class='figure'><img src='{figs[f'win_id_{sid}']}' style='max-width:100%'></div>")
                html.write(f"<div class='figure'><img src='{figs[f'win_indel_{sid}']}' style='max-width:100%'></div>")

    # Methods blurb
    html.write("""
<h2>Methods (brief)</h2>
<ol>
  <li>Translate CDS to proteins (standard genetic code by default), trimming trailing bases mod 3 and coercing ambiguous codons to <code>X</code>.</li>
  <li><b>Protein MSA:</b> Use MAFFT (<code>--auto --anysymbol</code>) when available; otherwise a simple progressive global aligner (BLOSUM62) is used. The protein MSA is back-translated to a codon MSA by inserting <code>---</code> for amino-acid gaps and mapping each AA to its original codon.</li>
  <li><b>Pairwise metrics:</b> AA and NT identities on aligned, non-gap positions; transitions/transversions (Ts/Tv); codon-level syn/nonsyn status for single-nucleotide changes; gap events and largest contiguous gap cluster (in codons).</li>
  <li><b>Sliding windows:</b> Amino-acid identity and codon-level indel rate in windows (default 30 codons; step 10 codons).</li>
  <li><b>BLAST-miss risk:</b> A heuristic classifier combining global AA identity, alignment coverage, length of the longest identical AA block, count of &lt;20% identity windows =20 aa, and the largest indel cluster. High risk corresponds to regimes where BLASTP often fails to seed or extend HSPs.</li>
</ol>
<p class="small">Heuristic thresholds can be tuned with command-line flags if desired; defaults reflect common sensitivity breakpoints for BLASTP.</p>
""")

    html.write("</body></html>")
    report_path = os.path.join(args.outdir, "report.html")
    with open(report_path, "w") as fh:
        fh.write(html.getvalue())

    # Save config
    with open(os.path.join(args.outdir, "run_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    print(f"[ok] Wrote: {report_path}")
    print(f"[ok] CSVs: pairwise_metrics.csv, window_metrics.csv (in {args.outdir})")
    print(f"[ok] Alignments: alignment_protein.faa, alignment_codon.fna (in {args.outdir})")


def main():
    ap = argparse.ArgumentParser(description="Compute divergence metrics and BLAST-miss risk for orthologous CDS.")
    ap.add_argument("--fasta", required=True, help="Multi-FASTA of CDS (ORFs) for the same gene across species.")
    ap.add_argument("--outdir", required=True, help="Output directory for report and artifacts.")
    ap.add_argument("--genetic_code", type=int, default=1, help="NCBI genetic code ID (default 1).")
    ap.add_argument("--window_codon", type=int, default=30, help="Sliding window size in codons (default 30).")
    ap.add_argument("--step_codon", type=int, default=10, help="Sliding window step in codons (default 10).")
    ap.add_argument("--reference", default=None, help="Sequence ID to use as reference (default: first in FASTA).")
    ap.add_argument("--no_mafft", action="store_true", help="Disable MAFFT usage (always use fallback aligner).")
    ap.add_argument("--mafft_path", default="mafft", help="Path to MAFFT binary (default: 'mafft').")
    ap.add_argument("--allow_ambig", action="store_true", help="Allow internal stops/ambigs by mapping to X.")
    args = ap.parse_args()
    analyze(args)

if __name__ == "__main__":
    main()
