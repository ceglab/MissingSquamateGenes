#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize multiple gene folders from ortholog_divergence_report outputs.

Input:
  --list <txt>   A text file with one RESULTS FOLDER per line (e.g., 53 lines).
                 Each folder should contain at least 'pairwise_metrics.csv'.
                 If available, 'per_sequence_composition.csv' is used for better
                 LCR/repeat/GC summaries.

Output:
  genes_summary.csv   A ranked table (one row per gene/folder) with summary stats
                      and an overall gene label: low / intermediate / high.

Usage:
  python summarize_genes_from_folders.py --list folders.txt --outdir ALL_SUMMARY

Optional knobs:
  --gene_low_cut 40   Gene score < this => LOW
  --gene_high_cut 65  Gene score >= this => HIGH
  --tail_q 0.10       Lower/upper tail quantile used for robust tails
  --coverage_low_thresh 0.50  Coverage threshold to count as low
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

REQ_PAIR_COLS = [
    "seqA","seqB","aa_identity","aa_coverage","longest_contiguous_identical_aa",
    "nt_identity","ts","tv","ts_tv_ratio","syn_codons","nonsyn_codons",
    "ambig_multihit_codons","gap_events","max_gap_cluster_codons",
    "gap_fraction_codons","low_identity_windows(<20%)",
    "gc_abs_diff","lcr_fraction_mean","repeat_fraction_mean"
]

def read_folder_list(path):
    folders = []
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"): continue
            folders.append(os.path.abspath(s))
    return folders

def safe_quantile(series, q, default=np.nan):
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0: return default
        return float(np.quantile(s, q))
    except Exception:
        return default

def safe_mean(series, default=np.nan):
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0: return default
        return float(s.mean())
    except Exception:
        return default

def safe_sum(series, default=np.nan):
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0: return default
        return float(s.sum())
    except Exception:
        return default

def clip01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def compute_pair_composite_score(df):
    """
    Build a composite pair score (0..100) using robust rank-like scaling.
    Not used directly for labels here, but tails feed the gene score.
    Features (weights sum ~1.0; mitigation by long identical block):
      - lower identity (35%), lower coverage (20%),
      - large gap cluster (12%), many low-ID windows (8%),
      - gap fraction (7%), LCR mean (7%), repeat mean (6%), GC abs diff (5%),
      - mitigation: long identical block reduces risk up to 10%.
    """
    # Helper to convert a numeric column into 0..1 "risk ranks"
    def rank_risk(col, higher_is_worse=True):
        s = pd.to_numeric(df[col], errors="coerce")
        med = s.median(skipna=True)
        s = s.fillna(med)
        if not higher_is_worse:
            s = -s
        r = s.rank(method="average", pct=True)  # 0..1
        return r.fillna(0.5).astype(float)

    parts = []
    # main signals
    parts.append(0.35 * rank_risk("aa_identity", higher_is_worse=False))
    parts.append(0.20 * rank_risk("aa_coverage", higher_is_worse=False))
    parts.append(0.12 * rank_risk("max_gap_cluster_codons", higher_is_worse=True))
    parts.append(0.08 * rank_risk("low_identity_windows(<20%)", higher_is_worse=True))
    parts.append(0.07 * rank_risk("gap_fraction_codons", higher_is_worse=True))
    parts.append(0.07 * rank_risk("lcr_fraction_mean", higher_is_worse=True))
    parts.append(0.06 * rank_risk("repeat_fraction_mean", higher_is_worse=True))
    parts.append(0.05 * rank_risk("gc_abs_diff", higher_is_worse=True))

    score = sum(parts)

    # mitigation by longest identical AA block (higher is better)
    block = pd.to_numeric(df["longest_contiguous_identical_aa"], errors="coerce")
    block = block.fillna(block.median(skipna=True))
    # convert to rank where larger block => lower risk
    block_rank = block.rank(method="average", pct=True)  # 0..1, higher is larger block
    # subtract up to 0.10
    score = score - 0.10 * block_rank

    # normalize to 0..100 across this gene's pairs
    smin = float(score.min())
    smax = float(score.max())
    if smax - smin < 1e-9:
        score100 = pd.Series(0.0, index=score.index)
    else:
        score100 = 100.0 * (score - smin) / (smax - smin)
    return score100.astype(float)

def compute_gene_row(folder, gene_name=None, tail_q=0.10, coverage_low_thresh=0.50):
    """
    Produce a dict of gene-level summary stats for a single folder.
    """
    pair_csv = os.path.join(folder, "pairwise_metrics.csv")
    if not os.path.exists(pair_csv):
        return {"gene": gene_name or os.path.basename(folder),
                "folder": folder, "status": "missing_pairwise_metrics"}

    try:
        df = pd.read_csv(pair_csv)
    except Exception as e:
        return {"gene": gene_name or os.path.basename(folder),
                "folder": folder, "status": "read_error:%s" % str(e)}

    # Verify required columns are present; if not, try to proceed with what we can
    missing = [c for c in REQ_PAIR_COLS if c not in df.columns]
    if missing:
        # We will still compute what is possible
        pass

    # Basic counts
    n_pairs = len(df)

    # Identity / coverage summaries
    aa_id_mean   = safe_mean(df.get("aa_identity", []))
    aa_id_median = safe_mean(df.get("aa_identity", []))
    aa_id_q10    = safe_quantile(df.get("aa_identity", []), tail_q)
    aa_id_q90    = safe_quantile(df.get("aa_identity", []), 1.0 - tail_q)

    cov_mean     = safe_mean(df.get("aa_coverage", []))
    cov_low_share = clip01((pd.to_numeric(df.get("aa_coverage", []), errors="coerce") < coverage_low_thresh).mean()
                           if "aa_coverage" in df else np.nan)

    # Nucleotide identity and Ts/Tv
    nt_id_mean   = safe_mean(df.get("nt_identity", []))
    ts_sum       = safe_sum(df.get("ts", []), 0.0)
    tv_sum       = safe_sum(df.get("tv", []), 0.0)
    ts_tv_ratio  = (ts_sum / tv_sum) if (tv_sum and tv_sum > 0) else (np.inf if ts_sum > 0 else np.nan)

    # Syn / nonsyn / multihit (sum across pairs; you may prefer per-site rates if desired)
    syn_sum      = safe_sum(df.get("syn_codons", []), 0.0)
    nonsyn_sum   = safe_sum(df.get("nonsyn_codons", []), 0.0)
    ambig_sum    = safe_sum(df.get("ambig_multihit_codons", []), 0.0)

    # Indel/gap features
    max_gap_cluster_q90 = safe_quantile(df.get("max_gap_cluster_codons", []), 0.90)
    gap_frac_mean       = safe_mean(df.get("gap_fraction_codons", []))
    lowIDwin_mean       = safe_mean(df.get("low_identity_windows(<20%)", []))

    # Complexity / repeats / GC
    # Prefer per_sequence_composition.csv if present (aggregated over sequences)
    per_seq_csv = os.path.join(folder, "per_sequence_composition.csv")
    if os.path.exists(per_seq_csv):
        try:
            comp = pd.read_csv(per_seq_csv)
            lcr_seq_mean = safe_mean(comp.get("aa_lcr_fraction", []))
            rep_seq_mean = safe_mean(comp.get("nt_repeat_fraction", []))
            gc_seq_mean  = safe_mean(comp.get("gc", []))
        except Exception:
            lcr_seq_mean = np.nan
            rep_seq_mean = np.nan
            gc_seq_mean  = np.nan
    else:
        # Fall back to pairwise means if necessary
        lcr_seq_mean = safe_mean(df.get("lcr_fraction_mean", []))
        rep_seq_mean = safe_mean(df.get("repeat_fraction_mean", []))
        gc_seq_mean  = np.nan

    gc_abs_diff_mean = safe_mean(df.get("gc_abs_diff", []))

    # Long identical block
    long_block_mean   = safe_mean(df.get("longest_contiguous_identical_aa", []))
    long_block_q90    = safe_quantile(df.get("longest_contiguous_identical_aa", []), 0.90)

    # Composite pair score distribution (0..100) and tails
    if n_pairs > 0:
        pair_score100 = compute_pair_composite_score(df)
        pair_score_q90 = safe_quantile(pair_score100, 0.90)
        pair_score_mean = safe_mean(pair_score100)
    else:
        pair_score100 = pd.Series([], dtype=float)
        pair_score_q90 = np.nan
        pair_score_mean = np.nan

    # Build gene score (0..100) similar to reassess_v2 logic
    # Weights: tail of composite (0.40), coverage low share (0.15),
    # lower-tail identity contribution (0.20), gap/complexity burden (0.25 total split)
    # Identity tail contribution: if q10 >= 0.35, contribute ~0; if near 0, contribute ~1.
    pivot_ok = 0.35
    if not np.isnan(aa_id_q10):
        id_tail_contrib = clip01((pivot_ok - max(0.0, min(1.0, aa_id_q10))) / max(1e-9, pivot_ok))
    else:
        id_tail_contrib = 0.0

    # Complexity bump lite
    c_bump = 0.0
    if not np.isnan(lcr_seq_mean) and lcr_seq_mean >= 0.25:
        c_bump += 0.05
    if not np.isnan(rep_seq_mean) and rep_seq_mean >= 0.10:
        c_bump += 0.05
    c_bump = clip01(c_bump)

    # Gap burden proxy: mean gap fraction and q90 of gap cluster, scaled to 0..1
    gap_burden = 0.5 * clip01(gap_frac_mean) + 0.5 * clip01((max(0.0, float(max_gap_cluster_q90 or 0.0)) / 60.0))
    gap_burden = clip01(gap_burden)

    # Combine to gene score
    w_tail = 0.40
    w_covl = 0.15
    w_idtl = 0.20
    w_gapc = 0.25
    gene_score = (
        w_tail * clip01((pair_score_q90 or 0.0) / 100.0) +
        w_covl * clip01(cov_low_share) +
        w_idtl * id_tail_contrib +
        w_gapc * gap_burden
    )
    gene_score = clip01(gene_score + c_bump)
    gene_score_100 = 100.0 * gene_score

    # Collect into a dict
    row = {
        "gene": gene_name or os.path.basename(os.path.normpath(folder)),
        "folder": folder,
        "n_pairs": int(n_pairs),
        # identities
        "aa_identity_mean": aa_id_mean,
        "aa_identity_q10": aa_id_q10,
        "aa_identity_q90": aa_id_q90,
        "aa_coverage_mean": cov_mean,
        "coverage_low_share": cov_low_share,
        # substitutions
        "nt_identity_mean": nt_id_mean,
        "ts_sum": ts_sum, "tv_sum": tv_sum,
        "ts_tv_ratio": (float(ts_tv_ratio) if np.isfinite(ts_tv_ratio) else np.nan),
        "syn_codons_sum": syn_sum,
        "nonsyn_codons_sum": nonsyn_sum,
        "ambig_multihit_codons_sum": ambig_sum,
        # indels/gaps
        "gap_fraction_mean": gap_frac_mean,
        "max_gap_cluster_q90": max_gap_cluster_q90,
        "lowID_windows_mean": lowIDwin_mean,
        # complexity/GC
        "lcr_fraction_seq_mean": lcr_seq_mean,
        "repeat_fraction_seq_mean": rep_seq_mean,
        "gc_seq_mean": gc_seq_mean,
        "gc_abs_diff_mean": gc_abs_diff_mean,
        # long identical block
        "long_block_mean": long_block_mean,
        "long_block_q90": long_block_q90,
        # pair composite score
        "pair_score_mean_0_100": pair_score_mean,
        "pair_score_q90_0_100": pair_score_q90,
        # final gene score
        "gene_score_0_100": gene_score_100,
        "status": "ok"
    }
    return row

def assign_gene_label(score_0_100, cut_low=40.0, cut_high=65.0):
    s = float(score_0_100)
    if s < cut_low: return "low"
    if s < cut_high: return "intermediate"
    return "high"

def main():
    ap = argparse.ArgumentParser(description="Compile a ranked cross-gene summary from many result folders.")
    ap.add_argument("--list", required=True, help="Text file with one results folder per line.")
    ap.add_argument("--outdir", required=True, help="Output directory for genes_summary.csv.")
    ap.add_argument("--tail_q", type=float, default=0.10, help="Tail quantile for robust metrics (default 0.10).")
    ap.add_argument("--coverage_low_thresh", type=float, default=0.50, help="Coverage threshold to count as low (default 0.50).")
    ap.add_argument("--gene_low_cut", type=float, default=40.0, help="Gene score < this => LOW (default 40).")
    ap.add_argument("--gene_high_cut", type=float, default=65.0, help="Gene score >= this => HIGH (default 65).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    folders = read_folder_list(args.list)
    if not folders:
        print("No folders found in --list.", file=sys.stderr)
        sys.exit(2)

    rows = []
    for f in folders:
        gene_name = os.path.basename(os.path.normpath(f))
        row = compute_gene_row(f, gene_name=gene_name, tail_q=args.tail_q, coverage_low_thresh=args.coverage_low_thresh)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Assign labels only for rows with status ok
    labels = []
    for _, r in df.iterrows():
        if r.get("status") == "ok":
            labels.append(assign_gene_label(r.get("gene_score_0_100", np.nan),
                                            cut_low=args.gene_low_cut,
                                            cut_high=args.gene_high_cut))
        else:
            labels.append("na")
    df["gene_label"] = labels

    # Sort by gene_score desc, keeping non-ok at bottom
    df["_ok"] = (df["status"] == "ok").astype(int)
    df = df.sort_values(by=["_ok", "gene_score_0_100"], ascending=[True, False])
    df = df.drop(columns=["_ok"])

    out_csv = os.path.join(args.outdir, "genes_summary.csv")
    df.to_csv(out_csv, index=False)

    # Also write a minimal TXT leaderboard for quick glance
    with open(os.path.join(args.outdir, "genes_leaderboard.txt"), "w") as fh:
        fh.write("Rank\tGene\tScore_0_100\tLabel\tPairs\tStatus\n")
        ok_df = df[df["status"] == "ok"].copy()
        ok_df = ok_df.sort_values("gene_score_0_100", ascending=False)
        for i, r in enumerate(ok_df.itertuples(index=False), start=1):
            fh.write("%d\t%s\t%.2f\t%s\t%d\t%s\n" % (
                i, getattr(r, "gene"),
                float(getattr(r, "gene_score_0_100")),
                getattr(r, "gene_label"),
                int(getattr(r, "n_pairs")),
                getattr(r, "status")
            ))
        # Append any problematic rows
        bad_df = df[df["status"] != "ok"]
        if len(bad_df) > 0:
            fh.write("\n# Folders with issues:\n")
            for r in bad_df.itertuples(index=False):
                fh.write("- %s\t[%s]\n" % (getattr(r, "folder"), getattr(r, "status")))

    print("[ok] Wrote:", out_csv)
    print("[ok] Also wrote:", os.path.join(args.outdir, "genes_leaderboard.txt"))
    print("Columns include: identities, coverage, Ts/Tv, syn/nonsyn, gap stats, LCR/repeats/GC, long-block, composite pair score tails, and final gene_score_0_100 + gene_label.")

if __name__ == "__main__":
    main()
