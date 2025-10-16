#!/usr/bin/env python3
# -*- coding: cp1252 -*-
# -*- coding: utf-8 -*-

"""
Ortholog Divergence Analyzer — lightweight HTML + complexity diagnostics
------------------------------------------------------------------------
What’s new vs. previous version:
  • Much lighter HTML:
      - images are written as PNGs and *referenced* (no base64 inlined)
      - lazy-loading for all images
      - large tables truncated in-page with links to full CSVs
      - optional cap on per-target window plots (defaults: top risky pairs only)
  • Extra diagnostics per sequence:
      - GC content, GC skew, CpG density
      - AA low-complexity fraction (SEG-like entropy window)
      - nucleotide repeat content (homo-/di-/tri-nucleotide tandem runs)
  • Correlation checks (report Pearson & Spearman):
      - AA identity vs |GC difference|
      - AA identity vs mean AA low-complexity fraction
      - AA identity vs mean repeat fraction
  • Risk model updated:
      - Up-rank risk when low-complexity or repeat content is high
      - Note when much of a sequence would be masked by LCR filters

Usage (same as before):
  python ortholog_divergence_report.py --fasta cds_multi.fasta --outdir results_geneX
"""

import os, sys, io, re, math, json, base64, argparse, tempfile, subprocess
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
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# Utility: file/plot helpers
# ---------------------------

def save_fig(fig, path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def write_png_plot(x, y, xlabel, ylabel, title, out_png):
    fig = plt.figure(figsize=(6, 3.6))
    plt.plot(x, y, marker=".", linestyle="None", alpha=0.6, ms=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(fig, out_png)

# ---------------------------
# Sequence I/O and alignment
# ---------------------------

def read_cds_fasta(path, allow_ambig=False, genetic_code=1):
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        seq = str(rec.seq).upper().replace("U", "T")
        if len(seq) % 3 != 0:
            seq = seq[:len(seq) - (len(seq) % 3)]
        if not re.fullmatch(r"[ACGTN-]+", seq):
            seq = re.sub(r"[^ACGT-]", "N", seq)
        try:
            protein = translate_cds(seq, genetic_code, allow_ambig=allow_ambig)
        except Exception:
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
        if len(codon) < 3: break
        if "-" in codon:
            protein.append("-"); continue
        if "N" in codon:
            protein.append("X"); continue
        aa = Seq(codon).translate(table=genetic_code, to_stop=False)
        if aa == "*":
            if i < len(cds) - 3:
                if not allow_ambig:
                    raise ValueError(f"Internal stop codon at codon {i//3+1}")
                protein.append("X")
        else:
            protein.append(str(aa))
    return "".join(protein)

def run_mafft_on_proteins(prot_records, mafft_path="mafft"):
    with tempfile.TemporaryDirectory() as td:
        inf = os.path.join(td, "in.faa")
        with open(inf, "w") as fh:
            for r in prot_records:
                fh.write(f">{r['id']}\n{r['aa']}\n")
        cmd = [mafft_path, "--anysymbol", "--auto", inf]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        aligned = {}
        cur_id, buf = None, []
        for line in res.stdout.splitlines():
            if line.startswith(">"):
                if cur_id is not None: aligned[cur_id] = "".join(buf)
                cur_id, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
        if cur_id is not None: aligned[cur_id] = "".join(buf)
        return [{"id": r["id"], "aa_aln": aligned[r["id"]]} for r in prot_records]

def pairwise_global_align(a, b, matrix=None, gap_open=-10, gap_extend=-0.5):
    if matrix is None:
        matrix = substitution_matrices.load("BLOSUM62")
    alns = pairwise2.align.globalds(a, b, matrix, gap_open, gap_extend, one_alignment_only=True)
    if not alns: return None, None
    return alns[0].seqA, alns[0].seqB

def consensus_from_alignment(aln_list):
    L = len(aln_list[0]); cons = []
    for i in range(L):
        col = [s[i] for s in aln_list]
        c = Counter(col); c.pop("-", None)
        cons.append(c.most_common(1)[0][0] if c else "-")
    return "".join(cons)

def gap_map(aligned_ref):
    return [i for i, ch in enumerate(aligned_ref) if ch == "-"]

def apply_gap_map_to_seq(seq, gap_positions):
    seq = list(seq); out = []; j = 0
    for i in range(len(seq) + len(gap_positions)):
        if i in gap_positions: out.append("-")
        else:
            out.append(seq[j] if j < len(seq) else "-"); j += 1
    return "".join(out)

def progressive_align_proteins(records):
    recs_sorted = sorted(records, key=lambda r: len(r["aa"]), reverse=True)
    aln_ids = [recs_sorted[0]["id"]]
    aln_seqs = [recs_sorted[0]["aa"]]
    matrix = substitution_matrices.load("BLOSUM62")
    for r in recs_sorted[1:]:
        consensus = consensus_from_alignment(aln_seqs)
        aA, aB = pairwise_global_align(consensus, r["aa"], matrix=matrix)
        ins_map = gap_map(aA)
        extended = [apply_gap_map_to_seq(s, ins_map) for s in aln_seqs]
        aln_seqs = extended + [aB]; aln_ids.append(r["id"])
    return [{"id": id_, "aa_aln": seq} for id_, seq in zip(aln_ids, aln_seqs)]

def backtranslate_protein_alignment_to_codon(prot_aln_records, cds_lookup):
    codon_alns = {}
    for r in prot_aln_records:
        pid, aa_aln = r["id"], r["aa_aln"]
        cds = cds_lookup[pid]
        codon_out = []; i_codon = 0
        for aa in aa_aln:
            if aa == "-": codon_out.append("---")
            else:
                codon = cds[i_codon*3:(i_codon+1)*3]
                codon_out.append(codon if len(codon) == 3 else "---")
                i_codon += 1
        codon_alns[pid] = "".join(codon_out)
    lengths = {len(v) for v in codon_alns.values()}
    if len(lengths) != 1:
        raise ValueError("Back-translation produced unequal lengths.")
    return codon_alns

# ---------------------------
# Complexity / composition
# ---------------------------

def gc_content(seq):
    g = seq.count("G"); c = seq.count("C")
    a = seq.count("A"); t = seq.count("T")
    denom = g + c + a + t
    return (g + c) / denom if denom else 0.0

def gc_skew(seq):
    g = seq.count("G"); c = seq.count("C")
    return (g - c) / (g + c) if (g + c) else 0.0

def cpg_density(seq):
    return seq.count("CG") / max(1, len(seq) - 1)

def shannon_entropy(window):
    if not window: return 0.0
    counts = Counter(window); n = len(window)
    probs = [v / n for v in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def lcr_mask_aa(aa_seq, win=12, thr=2.2):
    """
    SEG-like: mark positions within windows whose Shannon entropy < thr.
    Returns a boolean mask array and fraction of LCR positions.
    """
    L = len(aa_seq)
    mask = np.zeros(L, dtype=bool)
    if L < win: return mask, 0.0
    for i in range(0, L - win + 1):
        w = aa_seq[i:i+win].replace("-", "")
        if len(w) < win * 0.8:  # skip heavy-gap windows
            continue
        if shannon_entropy(w) < thr:
            mask[i:i+win] = True
    frac = mask.sum() / L if L else 0.0
    return mask, frac

def tandem_repeat_mask_nt(nt_seq):
    """
    Mark homopolymer runs >=5, dinucleotide repeats (unit len 2) with >=4 copies,
    and trinucleotide repeats (unit len 3) with >=3 copies. Return mask + fraction.
    """
    L = len(nt_seq)
    mask = np.zeros(L, dtype=bool)
    # homopolymers
    for m in re.finditer(r"(A{5,}|C{5,}|G{5,}|T{5,})", nt_seq):
        mask[m.start():m.end()] = True
    # di-nt repeats
    for m in re.finditer(r"((?:AT|TA|AC|CA|AG|GA|CT|TC|CG|GC|GT|TG)){4,}", nt_seq):
        mask[m.start():m.end()] = True
    # tri-nt repeats
    for m in re.finditer(r"([ACGT]{3})(?:\1){2,}", nt_seq):
        mask[m.start():m.end()] = True
    frac = mask.sum() / L if L else 0.0
    return mask, frac

# ---------------------------
# Core analyses
# ---------------------------

def is_transition(a, b):
    pur, pyr = set("AG"), set("CT")
    if a in pur and b in pur: return True
    if a in pyr and b in pyr: return True
    return False

def codon_syn_status(c1, c2, genetic_code=1):
    if "-" in c1 or "-" in c2: return "gap"
    if "N" in c1 or "N" in c2: return "ambig"
    diffs = sum(1 for i in range(3) if c1[i] != c2[i])
    if diffs == 0: return "same"
    if diffs > 1:  return "ambig"
    aa1 = str(Seq(c1).translate(table=genetic_code))
    aa2 = str(Seq(c2).translate(table=genetic_code))
    if aa1 == "*" or aa2 == "*": return "ambig"
    return "syn" if aa1 == aa2 else "nonsyn"

def sliding_windows(L, window, step):
    i = 0
    while i < L:
        yield i, min(i + window, L)
        i += step

def longest_contiguous_identity(aa_aln_q, aa_aln_t):
    cur = best = 0
    for i in range(len(aa_aln_q)):
        if aa_aln_q[i] == aa_aln_t[i] and aa_aln_q[i] != "-":
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best

def classify_blast_miss(aa_identity, coverage, max_gap_cluster, low_iden_windows,
                        longest_contig_ident, lcr_mean=None, repeat_mean=None):
    """
    Heuristic classifier (now considers complexity/repeats).
    """
    # Base gates
    if aa_identity < 0.30 or coverage < 0.50:
        base = "high"
    elif max_gap_cluster >= 30 or low_iden_windows >= 2:
        base = "high"
    elif (0.30 <= aa_identity < 0.40) or (0.50 <= coverage < 0.70) or (15 <= max_gap_cluster < 30):
        base = "intermediate"
    elif longest_contig_ident >= 40 and aa_identity >= 0.28 and coverage >= 0.45:
        base = "intermediate"
    else:
        base = "low"

    # Complexity adjustments: if much would be soft-masked, BLAST may miss seeds
    bump = 0
    if lcr_mean is not None and lcr_mean >= 0.25:
        bump += 1
    if repeat_mean is not None and repeat_mean >= 0.10:
        bump += 1

    levels = ["low", "intermediate", "high"]
    idx = min(2, levels.index(base) + bump)
    return levels[idx]

# ---------------------------
# Main pipeline
# ---------------------------

def analyze(args):
    os.makedirs(args.outdir, exist_ok=True)
    records = read_cds_fasta(args.fasta, allow_ambig=args.allow_ambig, genetic_code=args.genetic_code)
    ids = [r["id"] for r in records]
    cds_lookup = {r["id"]: r["cds"] for r in records}

    # Per-sequence composition/complexity
    seq_stats = []
    aa_lcr_masks = {}
    nt_repeat_masks = {}
    for r in records:
        gc = gc_content(r["cds"])
        skew = gc_skew(r["cds"])
        cpg = cpg_density(r["cds"])
        lcr_mask, lcr_frac = lcr_mask_aa(r["aa"])
        rep_mask, rep_frac = tandem_repeat_mask_nt(r["cds"])
        seq_stats.append({
            "id": r["id"],
            "gc": round(gc, 5),
            "gc_skew": round(skew, 5),
            "cpg_density": round(cpg, 5),
            "aa_lcr_fraction": round(lcr_frac, 5),
            "nt_repeat_fraction": round(rep_frac, 5),
            "aa_len": len(r["aa"].replace("-", "")),
            "cds_len": len(r["cds"].replace("-", "")),
        })
        aa_lcr_masks[r["id"]] = lcr_mask
        nt_repeat_masks[r["id"]] = rep_mask
    seq_df = pd.DataFrame(seq_stats).set_index("id")
    seq_df.to_csv(os.path.join(args.outdir, "per_sequence_composition.csv"))

    # Protein alignment: MAFFT if available, else fallback
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
    aln_len_cod = len(next(iter(codon_alns.values()))) // 3

    # Save alignments
    with open(os.path.join(args.outdir, "alignment_protein.faa"), "w") as fh:
        for sid in aln_ids: fh.write(f">{sid}\n{aa_alns[sid]}\n")
    with open(os.path.join(args.outdir, "alignment_codon.fna"), "w") as fh:
        for sid in aln_ids: fh.write(f">{sid}\n{codon_alns[sid]}\n")

    # Pairwise metrics (now also include complexity averages and diffs)
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

        # Longest identical AA stretch
        lci = longest_contiguous_identity(aaA, aaB)

        # NT identity, Ts/Tv, syn/nonsyn; gap clusters
        nt_matches = nt_comp = ts = tv = 0
        syn = nonsyn = ambig = 0
        gap_events = gap_cluster_max = cur_gap_cluster = 0

        for i in range(0, len(codA), 3):
            c1, c2 = codA[i:i+3], codB[i:i+3]
            if "-" in c1 or "-" in c2:
                cur_gap_cluster += 1; continue
            else:
                if cur_gap_cluster:
                    gap_events += 1
                    gap_cluster_max = max(gap_cluster_max, cur_gap_cluster)
                    cur_gap_cluster = 0

            for k in range(3):
                n1, n2 = c1[k], c2[k]
                if n1 in "ACGT" and n2 in "ACGT":
                    nt_comp += 1
                    if n1 == n2: nt_matches += 1
                    else:
                        if is_transition(n1, n2): ts += 1
                        else: tv += 1

            status = codon_syn_status(c1, c2, genetic_code=args.genetic_code)
            if status == "syn": syn += 1
            elif status == "nonsyn": nonsyn += 1
            elif status == "ambig": ambig += 1

        if cur_gap_cluster:
            gap_events += 1
            gap_cluster_max = max(gap_cluster_max, cur_gap_cluster)

        nt_identity = nt_matches / nt_comp if nt_comp else 0.0
        ts_tv_ratio = (ts / tv) if tv > 0 else (np.inf if ts > 0 else np.nan)

        gap_codons = sum(1 for i in range(0, len(codA), 3) if "-" in codA[i:i+3] or "-" in codB[i:i+3])

        pair_heat_id.loc[a, b] = pair_heat_id.loc[b, a] = aa_identity
        pair_heat_gap.loc[a, b] = pair_heat_gap.loc[b, a] = gap_codons / aln_len_cod

        # Sliding windows for low-ID regions & indel density
        win = max(5, args.window_codon); step = max(5, args.step_codon)
        low_win = 0
        for s in range(0, len(aaA), step):
            e = min(s + win, len(aaA))
            subA, subB = aaA[s:e], aaB[s:e]
            pos = sum(1 for i in range(len(subA)) if subA[i] != "-" and subB[i] != "-")
            if pos == 0: continue
            m = sum(1 for i in range(len(subA)) if subA[i] == subB[i] and subA[i] != "-")
            idw = m / pos
            if idw < 0.20 and pos >= 20: low_win += 1

        # Complexity / composition pair features
        gc_a, gc_b = seq_df.loc[a, "gc"], seq_df.loc[b, "gc"]
        lcr_a, lcr_b = seq_df.loc[a, "aa_lcr_fraction"], seq_df.loc[b, "aa_lcr_fraction"]
        rep_a, rep_b = seq_df.loc[a, "nt_repeat_fraction"], seq_df.loc[b, "nt_repeat_fraction"]

        gc_diff = abs(gc_a - gc_b)
        lcr_mean = (lcr_a + lcr_b) / 2
        rep_mean = (rep_a + rep_b) / 2
        gc_mean  = (gc_a + gc_b) / 2

        risk = classify_blast_miss(
            aa_identity=aa_identity, coverage=coverage,
            max_gap_cluster=gap_cluster_max, low_iden_windows=low_win,
            longest_contig_ident=lci, lcr_mean=lcr_mean, repeat_mean=rep_mean
        )

        pair_rows.append({
            "seqA": a, "seqB": b,
            "aa_identity": round(aa_identity, 4),
            "aa_coverage": round(coverage, 4),
            "longest_contiguous_identical_aa": int(lci),
            "nt_identity": round(nt_identity, 4),
            "ts": int(ts), "tv": int(tv),
            "ts_tv_ratio": (round(ts_tv_ratio, 4) if np.isfinite(ts_tv_ratio) else "inf"),
            "syn_codons": int(syn), "nonsyn_codons": int(nonsyn), "ambig_multihit_codons": int(ambig),
            "gap_events": int(gap_events),
            "max_gap_cluster_codons": int(gap_cluster_max),
            "gap_fraction_codons": round(gap_codons / aln_len_cod, 4),
            "low_identity_windows(<20%)": int(low_win),
            "gc_mean": round(gc_mean, 5),
            "gc_abs_diff": round(gc_diff, 5),
            "lcr_fraction_mean": round(lcr_mean, 5),
            "repeat_fraction_mean": round(rep_mean, 5),
            "blast_miss_risk": risk
        })

    pair_df = pd.DataFrame(pair_rows)
    pair_csv = os.path.join(args.outdir, "pairwise_metrics.csv")
    pair_df.to_csv(pair_csv, index=False)

    # --- Correlations ---
    corr_rows = []
    def corr_pair(x, y, label_x, label_y):
        if len(x) >= 3 and len(y) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
        else:
            pr = pp = sr = sp = np.nan
        return {
            "x": label_x, "y": label_y,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp
        }

    # 1) AA id vs |GC diff|
    corr_rows.append(corr_pair(pair_df["aa_identity"], -pair_df["gc_abs_diff"], "AA identity", "- |GC diff|"))
    # 2) AA id vs mean LCR fraction
    corr_rows.append(corr_pair(pair_df["aa_identity"], -pair_df["lcr_fraction_mean"], "AA identity", "- mean LCR fraction"))
    # 3) AA id vs mean repeat fraction
    corr_rows.append(corr_pair(pair_df["aa_identity"], -pair_df["repeat_fraction_mean"], "AA identity", "- mean repeat fraction"))

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(args.outdir, "correlations.csv"), index=False)

    # --- Reference-based windows (top risky pairs only, to keep HTML light) ---
    ref_id = aln_ids[0] if args.reference is None else (args.reference if args.reference in aln_ids else aln_ids[0])
    win = max(5, args.window_codon); step = max(5, args.step_codon)
    risk_order = {"high": 2, "intermediate": 1, "low": 0}
    top_targets = (pair_df.assign(score=lambda d: d["blast_miss_risk"].map(risk_order))
                   .sort_values(["score", "aa_identity"], ascending=[False, True]))
    # pick up to max_targets unique partners involving ref_id first, then others
    chosen = []
    for _, r in top_targets.iterrows():
        if r["seqA"] == ref_id:
            other = r["seqB"]
        elif r["seqB"] == ref_id:
            other = r["seqA"]
        else:
            continue
        if other not in chosen:
            chosen.append(other)
        if len(chosen) >= args.max_window_targets:
            break

    window_pngs = []
    for sid in chosen:
        aaR, aaT = aa_alns[ref_id], aa_alns[sid]
        # identity windows
        xs, ys_id, ys_indel = [], [], []
        codR, codT = codon_alns[ref_id], codon_alns[sid]
        for s in range(0, len(aaR), step):
            e = min(s + win, len(aaR))
            subR, subT = aaR[s:e], aaT[s:e]
            pos = sum(1 for i in range(len(subR)) if subR[i] != "-" and subT[i] != "-")
            m = sum(1 for i in range(len(subR)) if subR[i] == subT[i] and subR[i] != "-")
            idw = (m / pos) if pos else 0.0
            cod_s, cod_e = s*3, e*3
            gapped_cod = sum(1 for i in range(cod_s, cod_e, 3) if "-" in codR[i:i+3] or "-" in codT[i:i+3])
            indel_rate = gapped_cod / max(1, (e - s))
            xs.append(s+1); ys_id.append(idw); ys_indel.append(indel_rate)

        # plot identity
        fig = plt.figure(figsize=(7.2, 2.8))
        plt.plot(xs, ys_id)
        plt.ylim(0, 1); plt.xlabel("Window start (codon)"); plt.ylabel("AA identity")
        plt.title(f"{ref_id} vs {sid}: sliding AA identity")
        p1 = os.path.join(args.outdir, f"win_id_{ref_id}_vs_{sid}.png"); save_fig(fig, p1)

        # plot indel rate
        fig = plt.figure(figsize=(7.2, 2.8))
        plt.plot(xs, ys_indel); plt.ylim(0, 1)
        plt.xlabel("Window start (codon)"); plt.ylabel("Indel rate (codon-level)")
        plt.title(f"{ref_id} vs {sid}: sliding indel rate")
        p2 = os.path.join(args.outdir, f"win_indel_{ref_id}_vs_{sid}.png"); save_fig(fig, p2)
        window_pngs.append((sid, p1, p2))

    # --- Heatmaps ---
    heat_id_png  = os.path.join(args.outdir, "heat_identity.png")
    heat_gap_png = os.path.join(args.outdir, "heat_gap.png")

    fig = plt.figure(figsize=(max(4, len(aln_ids)*0.55), max(4, len(aln_ids)*0.55)))
    im = plt.imshow(pair_heat_id.fillna(1.0).values, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="AA identity")
    plt.xticks(ticks=range(len(aln_ids)), labels=aln_ids, rotation=90)
    plt.yticks(ticks=range(len(aln_ids)), labels=aln_ids)
    plt.title("Pairwise AA identity")
    save_fig(fig, heat_id_png)

    fig = plt.figure(figsize=(max(4, len(aln_ids)*0.55), max(4, len(aln_ids)*0.55)))
    im = plt.imshow(pair_heat_gap.fillna(0.0).values, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Gap fraction (codons)")
    plt.xticks(ticks=range(len(aln_ids)), labels=aln_ids, rotation=90)
    plt.yticks(ticks=range(len(aln_ids)), labels=aln_ids)
    plt.title("Pairwise gap load")
    save_fig(fig, heat_gap_png)

    # --- Scatter plots for correlations (lightweight) ---
    scatters = []
    s1 = os.path.join(args.outdir, "scatter_id_vs_gcDiff.png")
    write_png_plot(pair_df["gc_abs_diff"], pair_df["aa_identity"],
                   "GC difference (abs)", "AA identity", "AA id vs |GC diff|", s1)
    scatters.append(("AA id vs |GC diff|", s1))

    s2 = os.path.join(args.outdir, "scatter_id_vs_LCR.png")
    write_png_plot(pair_df["lcr_fraction_mean"], pair_df["aa_identity"],
                   "Mean AA LCR fraction", "AA identity", "AA id vs mean LCR", s2)
    scatters.append(("AA id vs mean LCR", s2))

    s3 = os.path.join(args.outdir, "scatter_id_vs_repeats.png")
    write_png_plot(pair_df["repeat_fraction_mean"], pair_df["aa_identity"],
                   "Mean NT repeat fraction", "AA identity", "AA id vs repeats", s3)
    scatters.append(("AA id vs repeats", s3))

    # --- HTML (lightweight) ---
    # Show only the N worst rows in-page
    worst = pair_df.sort_values(["blast_miss_risk", "aa_identity"],
                                ascending=[True, True])  # high < inter < low (sort lexicographically)
    # Map to ordinal for sort: enforce high > intermediate > low
    order = {"high": 2, "intermediate": 1, "low": 0}
    worst = pair_df.assign(o=pair_df["blast_miss_risk"].map(order)).sort_values(["o", "aa_identity"])
    top_rows = worst.head(args.max_table_rows).drop(columns=["o"])

    report_path = os.path.join(args.outdir, "report.html")
    with open(report_path, "w") as H:
        H.write(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Ortholog Divergence Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif; margin: 18px; }}
h1,h2,h3 {{ margin-top: 1em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
th {{ background:#fafafa; position: sticky; top: 0; }}
td:first-child, th:first-child {{ text-align: left; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge.low {{ background:#e7f7ed; color:#056e2e; }}
.badge.intermediate {{ background:#fff5d6; color:#8a6100; }}
.badge.high {{ background:#ffe8e6; color:#9a1d0d; }}
.small {{ color:#666; font-size: 12px; }}
.figure {{ margin: 10px 0 16px 0; }}
code,pre {{ background:#f6f8fa; padding:2px 4px; border-radius:4px; }}
.container {{ max-width: 1240px; margin: 0 auto; }}
</style>
</head><body><div class="container">
<h1>Ortholog Divergence Report</h1>
<p><b>Input:</b> {os.path.basename(args.fasta)} &nbsp;&middot;&nbsp; <b>N sequences:</b> {len(aln_ids)} &nbsp;&middot;&nbsp; <b>Aligned length:</b> {aln_len_cod} codons</p>
<p><b>Reference:</b> {ref_id}</p>

<h2>Quick risk overview</h2>
<ul>
  <li>LOW: {int((pair_df["blast_miss_risk"]=="low").sum())}</li>
  <li>INTERMEDIATE: {int((pair_df["blast_miss_risk"]=="intermediate").sum())}</li>
  <li>HIGH: {int((pair_df["blast_miss_risk"]=="high").sum())}</li>
</ul>

<div class="figure"><img loading="lazy" src="{os.path.basename(heat_id_png)}" alt="AA identity heatmap" style="max-width:100%"></div>
<div class="figure"><img loading="lazy" src="{os.path.basename(heat_gap_png)}" alt="Gap heatmap" style="max-width:100%"></div>

<h2>Composition & complexity per sequence</h2>
<p class="small">Saved as <code>per_sequence_composition.csv</code>. High low-complexity or repeat content can reduce BLAST sensitivity due to soft masking.</p>
<table>
<thead><tr>
<th>id</th><th>GC</th><th>GC skew</th><th>CpG dens.</th>
<th>AA LCR frac</th><th>NT repeat frac</th><th>AA len</th><th>CDS len</th>
</tr></thead><tbody>
""")
        for sid, r in seq_df.reset_index().iterrows():
            H.write(f"<tr><td>{r['id']}</td><td>{r['gc']}</td><td>{r['gc_skew']}</td>"
                    f"<td>{r['cpg_density']}</td><td>{r['aa_lcr_fraction']}</td>"
                    f"<td>{r['nt_repeat_fraction']}</td><td>{int(r['aa_len'])}</td>"
                    f"<td>{int(r['cds_len'])}</td></tr>\n")
        H.write(f"""</tbody></table>

<h2>Pairwise metrics (top {len(top_rows)} shown)</h2>
<p class="small">Full table: <a href="{os.path.basename(pair_csv)}">pairwise_metrics.csv</a></p>
<table>
<thead><tr>
<th>seqA</th><th>seqB</th><th>AA id</th><th>AA cov</th><th>Longest block</th>
<th>NT id</th><th>Ts</th><th>Tv</th><th>Ts/Tv</th>
<th>Syn</th><th>NonSyn</th><th>Ambig</th>
<th>Gap ev</th><th>Max gap cluster</th><th>Gap frac</th>
<th>&lt;20% win</th><th>GC mean</th><th>|GC diff|</th>
<th>mean LCR</th><th>mean repeats</th><th>Risk</th>
</tr></thead><tbody>
""")
        # Show worst rows in compact form
        for _, r in top_rows.iterrows():
            badge = f"<span class='badge {r['blast_miss_risk']}'>{r['blast_miss_risk'].upper()}</span>"
            H.write(
                f"<tr><td>{r.seqA}</td><td>{r.seqB}</td>"
                f"<td>{r.aa_identity:.3f}</td><td>{r.aa_coverage:.3f}</td><td>{r.longest_contiguous_identical_aa}</td>"
                f"<td>{r.nt_identity:.3f}</td><td>{r.ts}</td><td>{r.tv}</td><td>{r.ts_tv_ratio}</td>"
                f"<td>{r.syn_codons}</td><td>{r.nonsyn_codons}</td><td>{r.ambig_multihit_codons}</td>"
                f"<td>{r.gap_events}</td><td>{r.max_gap_cluster_codons}</td><td>{r.gap_fraction_codons:.3f}</td>"
                f"<td>{int(r['low_identity_windows(<20%)'])}</td><td>{r.gc_mean:.3f}</td><td>{r.gc_abs_diff:.3f}</td>"
                f"<td>{r.lcr_fraction_mean:.3f}</td><td>{r.repeat_fraction_mean:.3f}</td><td>{badge}</td></tr>\n"
            )
        H.write("</tbody></table>")

        # Correlations section
        H.write("""
<h2>Correlation checks</h2>
<p class="small">Pearson and Spearman correlations (negative association means the composition feature increases as identity decreases).</p>
<table>
<thead><tr><th>X</th><th>Y</th><th>Pearson r</th><th>Pearson p</th><th>Spearman ?</th><th>Spearman p</th></tr></thead><tbody>
""")
        for _, r in corr_df.iterrows():
            H.write(f"<tr><td>{r['x']}</td><td>{r['y']}</td>"
                    f"<td>{np.round(r['pearson_r'], 4) if pd.notna(r['pearson_r']) else 'NA'}</td>"
                    f"<td>{np.format_float_scientific(r['pearson_p'], precision=2) if pd.notna(r['pearson_p']) else 'NA'}</td>"
                    f"<td>{np.round(r['spearman_rho'], 4) if pd.notna(r['spearman_rho']) else 'NA'}</td>"
                    f"<td>{np.format_float_scientific(r['spearman_p'], precision=2) if pd.notna(r['spearman_p']) else 'NA'}</td></tr>\n")
        H.write("</tbody></table>")

        H.write("<div class='figure'>")
        for title, path in scatters:
            H.write(f"<figure style='display:inline-block;margin:6px 8px;'><img loading='lazy' src='{os.path.basename(path)}' alt='{title}' style='max-width:380px;width:100%'><figcaption class='small' style='text-align:center'>{title}</figcaption></figure>")
        H.write("</div>")

        # Window plots (only for top risky vs ref)
        if window_pngs:
            H.write(f"<h2>Local divergence vs reference ({ref_id})</h2>")
            for sid, p1, p2 in window_pngs:
                H.write(f"<h3>{ref_id} vs {sid}</h3>")
                H.write(f"<div class='figure'><img loading='lazy' src='{os.path.basename(p1)}' style='max-width:100%'></div>")
                H.write(f"<div class='figure'><img loading='lazy' src='{os.path.basename(p2)}' style='max-width:100%'></div>")

        # Methods
        H.write(f"""
<h2>Methods (brief)</h2>
<ol>
  <li>CDS are translated (genetic code {args.genetic_code}); proteins aligned by MAFFT when available, otherwise a progressive global aligner (BLOSUM62). Protein MSA is back-translated into a codon MSA.</li>
  <li>Pairwise metrics: AA/NT identities on aligned nongap positions; Ts/Tv; single-hit syn/nonsyn counts; gap events and largest contiguous gap cluster (codons); sliding-window AA identity and codon-level indel rate.</li>
  <li>Per-sequence diagnostics: GC content/skew, CpG density, AA low-complexity fraction via SEG-like entropy windows (win=12, H&lt;2.2), and NT tandem repeats (homopolymers =5; di-nt =4 copies; tri-nt =3 copies).</li>
  <li>Correlations: Pearson/Spearman between AA identity and |GC difference|, mean AA LCR fraction, and mean NT repeat fraction across pairs.</li>
  <li>Risk classifier: combines identity, coverage, longest identical block, low-ID windows, max indel cluster, and up-ranks risk when mean LCR =0.25 or mean repeat =0.10 (soft-masking can remove seeds and reduce sensitivity).</li>
</ol>

<p class="small">Artifacts: <code>{os.path.basename(pair_csv)}</code>, <code>per_sequence_composition.csv</code>, <code>correlations.csv</code>, <code>alignment_protein.faa</code>, <code>alignment_codon.fna</code>.</p>
</div></body></html>
""")

    print(f"[ok] Wrote: {report_path}")
    print(f"[ok] CSVs: pairwise_metrics.csv, per_sequence_composition.csv, correlations.csv")
    print(f"[ok] Plots: heat_identity.png, heat_gap.png, scatter_*.png, win_* (limited)")
    return report_path

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute divergence metrics, complexity diagnostics, and BLAST-miss risk.")
    ap.add_argument("--fasta", required=True, help="Multi-FASTA of CDS (ORFs) for the same gene across species.")
    ap.add_argument("--outdir", required=True, help="Output directory for report and artifacts.")
    ap.add_argument("--genetic_code", type=int, default=1, help="NCBI genetic code ID (default 1).")
    ap.add_argument("--window_codon", type=int, default=30, help="Sliding window size in codons (default 30).")
    ap.add_argument("--step_codon", type=int, default=10, help="Sliding window step in codons (default 10).")
    ap.add_argument("--reference", default=None, help="Sequence ID to use as reference (default: first in FASTA).")
    ap.add_argument("--no_mafft", action="store_true", help="Disable MAFFT usage (always use fallback aligner).")
    ap.add_argument("--mafft_path", default="mafft", help="Path to MAFFT binary (default: 'mafft').")
    ap.add_argument("--allow_ambig", action="store_true", help="Allow internal stops/ambigs by mapping to X.")

    # NEW knobs for lighter HTML
    ap.add_argument("--max_window_targets", type=int, default=6,
                    help="Max number of ref-vs-target window plots to include (default 6).")
    ap.add_argument("--max_table_rows", type=int, default=80,
                    help="Max pairwise rows to render inline in HTML (default 80).")

    args = ap.parse_args()
    analyze(args)

if __name__ == "__main__":
    main()
