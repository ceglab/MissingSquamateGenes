#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reassess BLAST-miss risk (pairwise and gene-wide, dataset-adaptive)

Run this on an output folder from the ortholog_divergence_report.py pipeline.
It will:
  1) Load pairwise_metrics.csv (+ optional per_sequence_composition.csv).
  2) Compute a robust composite risk score per pair (rank- and weight-based).
  3) Assign balanced labels (LOW / INTERMEDIATE / HIGH) via quantiles.
  4) Derive a gene-wide classification (LOW / INTERMEDIATE / HIGH)
     estimating likelihood that BLAST would miss orthologs from other distant vertebrates.

Outputs:
  - reassessed_pairs.csv   (pair-level balanced labels and reasons)
  - reassess_report.html   (lightweight summary)
  - gene_summary.json      (machine-readable gene-wide metrics)
  - gene_summary.txt       (human-readable one-liner + details)

Usage:
  python reassess_blast_risk_v2.py --indir results_geneX \
      --high_q 0.20 --intermediate_q 0.30 \
      --outdir results_geneX/_reassess_v2

Notes:
  - All math is ASCII-safe; no fancy unicode to avoid encoding issues.
  - If per_sequence_composition.csv is available, LCR/repeat means are taken
    from there; otherwise the pairwise means are averaged as a proxy.
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------

def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns in pairwise_metrics.csv: %s" % missing)

def rankify(series, higher_is_worse=True):
    """
    Convert a numeric series to [0,1] risk ranks (robust).
    If higher_is_worse=False, invert meaning (so higher value -> lower risk).
    NaN filled with median.
    """
    s = series.astype(float).copy()
    med = np.nanmedian(s)
    s = s.fillna(med)
    if not higher_is_worse:
        s = -s
    ranks = s.rank(method="average", pct=True)  # 0..1
    return ranks

def clip01(x):
    return max(0.0, min(1.0, float(x)))

def save_fig(fig, path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def top_reasons(contribs, k=3):
    # contribs: dict(feature -> contribution value)
    if not contribs:
        return ""
    parts = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ";".join(["%s+%.2f" % (a, b) for a, b in parts if b > 0])

# ---------------------------
# Pairwise composite scoring
# ---------------------------

FEATURE_SPECS = [
    # name, column, higher_is_worse, weight
    ("low_identity",       "aa_identity",               False, 0.35),
    ("low_coverage",       "aa_coverage",               False, 0.20),
    ("gap_cluster",        "max_gap_cluster_codons",     True, 0.12),
    ("lowID_windows",      "low_identity_windows(<20%)", True, 0.08),
    ("gap_fraction",       "gap_fraction_codons",        True, 0.07),
    ("LCR_mean",           "lcr_fraction_mean",          True, 0.07),
    ("repeats_mean",       "repeat_fraction_mean",       True, 0.06),
    ("gc_abs_diff",        "gc_abs_diff",                True, 0.05),
    # mitigation via long identical block handled separately (reduces risk)
]

def compute_pair_scores(df):
    """
    Build composite pairwise risk score S in [0,100] by weighted sum of per-feature ranks,
    mitigated by longest identical block rank.
    """
    contrib = {}
    score = np.zeros(len(df), dtype=float)

    for name, col, higher_worse, w in FEATURE_SPECS:
        r = rankify(df[col], higher_is_worse=higher_worse)
        part = w * r.values
        score += part
        contrib[name] = part

    # Mitigation: longer identical block reduces risk
    block = df["longest_contiguous_identical_aa"].astype(float)
    block_rank = rankify(block, higher_is_worse=False)  # higher block -> lower risk
    block_weight = 0.10
    block_part = block_weight * (1.0 - block_rank.values)  # subtract up to 0.10
    score -= block_part
    contrib["long_block_mitig"] = -block_part

    # Normalize to [0,100]
    # First clamp to 0..1 based on observed min/max of weighted sum
    smin, smax = float(np.min(score)), float(np.max(score))
    if smax - smin < 1e-9:
        score_norm = np.zeros_like(score)
    else:
        score_norm = (score - smin) / (smax - smin)
    score_100 = 100.0 * score_norm

    # Build top reasons string per row
    top_r = []
    for i in range(len(df)):
        row_contribs = {k: float(v[i]) for k, v in contrib.items() if not np.isnan(v[i])}
        top_r.append(top_reasons(row_contribs, k=3))

    return score_100, top_r

def assign_pair_bins(score_100, high_q=0.20, intermediate_q=0.30):
    """
    Quantile-based pair labels from the composite score (0..100 scale).
      - HIGH: top high_q fraction
      - INTERMEDIATE: next intermediate_q fraction
      - LOW: remainder
    Returns (labels, cutoff_info).
    """
    if len(score_100) == 0:
        return [], {"q_high": 100.0, "q_inter": 0.0}
    q_high = float(np.quantile(score_100, 1.0 - high_q))
    q_inter = float(np.quantile(score_100, 1.0 - high_q - intermediate_q))
    labels = []
    for s in score_100:
        if s >= q_high: labels.append("high")
        elif s >= q_inter: labels.append("intermediate")
        else: labels.append("low")
    return labels, {"q_high": q_high, "q_inter": q_inter}

# ---------------------------
# Gene-wide classification
# ---------------------------

def scale_identity_tail(id_tail, pivot_ok=0.35):
    """
    Convert lower-tail identity (e.g., 10th percentile) into a 0..1 risk contribution.
    If id_tail >= pivot_ok -> 0 risk; if id_tail == 0 -> 1 risk.
    """
    x = clip01((pivot_ok - max(0.0, min(1.0, float(id_tail)))) / max(1e-9, pivot_ok))
    return x

def compute_gene_metrics(pairs_df, pair_score_100, pair_labels,
                         coverage_low_thresh=0.50,
                         lcr_flag=0.25, rep_flag=0.10,
                         tail_q=0.10):
    """
    Derive gene-wide risk indicators from the pairwise distribution.
    Returns a dict of metrics and a final classification.
    """

    # Tail of composite risk: how bad are the worst observed pairs in this dataset?
    tail_score = float(np.quantile(pair_score_100, 1.0 - tail_q)) if len(pair_score_100) else 0.0  # 0..100

    # Share of pairs flagged high under the balanced labeling
    labels_array = np.array(pair_labels)
    share_high = float((labels_array == "high").mean()) if len(labels_array) else 0.0

    # Coverage low share (can imply partial ORFs or strong indel issues)
    cov_low_share = float((pairs_df["aa_coverage"].astype(float) < coverage_low_thresh).mean()) if len(pairs_df) else 0.0

    # Lower-tail AA identity (10th percentile by default)
    id_tail = float(np.quantile(pairs_df["aa_identity"].astype(float), tail_q)) if len(pairs_df) else 1.0
    id_tail_contrib = scale_identity_tail(id_tail, pivot_ok=0.35)  # 0..1

    # Mean complexity and repeat content (prefer per-sequence file if available)
    if "lcr_fraction_mean" in pairs_df.columns:
        lcr_mean_over_pairs = float(pairs_df["lcr_fraction_mean"].astype(float).mean())
    else:
        lcr_mean_over_pairs = 0.0
    if "repeat_fraction_mean" in pairs_df.columns:
        rep_mean_over_pairs = float(pairs_df["repeat_fraction_mean"].astype(float).mean())
    else:
        rep_mean_over_pairs = 0.0

    # Complexity bumps
    bump = 0.0
    if lcr_mean_over_pairs >= lcr_flag:
        bump += 0.05
    if rep_mean_over_pairs >= rep_flag:
        bump += 0.05
    bump = clip01(bump)

    # Combine into a gene score in 0..100
    # Weights sum to 1.0 before bump; tail_score already in 0..100, others scaled to 0..1 then expanded.
    # Intuition:
    #  - 0.40: tail of composite risk (how bad are the worst cases seen)
    #  - 0.25: share of high labels (breadth of serious risk)
    #  - 0.20: identity lower tail (proxy for extrapolation to even more distant vertebrates)
    #  - 0.15: share of low coverage pairs (partial sequences / indel issues)
    w_tail = 0.40
    w_high = 0.25
    w_idtl = 0.20
    w_covl = 0.15

    gene_score = (
        w_tail * (tail_score / 100.0)
        + w_high * share_high
        + w_idtl * id_tail_contrib
        + w_covl * cov_low_share
    )
    gene_score = clip01(gene_score + bump)
    gene_score_100 = 100.0 * gene_score

    # Map to gene-wide class with sensible defaults (tunable via CLI)
    return {
        "tail_score_q90": tail_score,                 # 0..100
        "share_high_pairs": share_high,               # 0..1
        "coverage_low_share": cov_low_share,          # 0..1
        "aa_identity_tail_q10": id_tail,              # 0..1
        "lcr_mean_over_pairs": lcr_mean_over_pairs,   # 0..1
        "repeat_mean_over_pairs": rep_mean_over_pairs,# 0..1
        "complexity_bump": bump,                      # 0..1 (0.00, 0.05, 0.10)
        "gene_score_0_100": gene_score_100
    }

def label_gene(gene_score_100, cut_low=40.0, cut_high=65.0):
    """
    Convert gene_score_100 to LOW / INTERMEDIATE / HIGH.
      - LOW: score < cut_low
      - INTERMEDIATE: cut_low <= score < cut_high
      - HIGH: score >= cut_high
    """
    s = float(gene_score_100)
    if s < cut_low: return "low"
    if s < cut_high: return "intermediate"
    return "high"

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Balanced reassessment of BLAST-miss risk with gene-wide classification.")
    ap.add_argument("--indir", required=True, help="Folder containing pairwise_metrics.csv from the previous run.")
    ap.add_argument("--outdir", default=None, help="Output folder (default: <indir>/_reassess_v2)")
    # Pairwise label calibration
    ap.add_argument("--high_q", type=float, default=0.20, help="Fraction labeled HIGH (default 0.20).")
    ap.add_argument("--intermediate_q", type=float, default=0.30, help="Fraction labeled INTERMEDIATE (default 0.30).")
    # Gene-wide knobs
    ap.add_argument("--tail_q", type=float, default=0.10, help="Quantile for tail metrics, e.g., 0.10 for 10th/90th (default 0.10).")
    ap.add_argument("--coverage_low_thresh", type=float, default=0.50, help="Coverage threshold counted as low (default 0.50).")
    ap.add_argument("--lcr_flag", type=float, default=0.25, help="Mean LCR fraction that adds a bump (default 0.25).")
    ap.add_argument("--rep_flag", type=float, default=0.10, help="Mean repeat fraction that adds a bump (default 0.10).")
    ap.add_argument("--gene_low_cut", type=float, default=40.0, help="Gene score < this => LOW (default 40).")
    ap.add_argument("--gene_high_cut", type=float, default=65.0, help="Gene score >= this => HIGH (default 65).")
    ap.add_argument("--max_table_rows", type=int, default=120, help="Rows to show inline in HTML.")
    args = ap.parse_args()

    indir = os.path.abspath(args.indir)
    outdir = os.path.abspath(args.outdir if args.outdir else os.path.join(indir, "_reassess_v2"))
    os.makedirs(outdir, exist_ok=True)

    pair_csv = os.path.join(indir, "pairwise_metrics.csv")
    if not os.path.exists(pair_csv):
        raise FileNotFoundError("Cannot find %s" % pair_csv)
    df = pd.read_csv(pair_csv)

    # Required columns
    req = [
        "seqA","seqB","aa_identity","aa_coverage","longest_contiguous_identical_aa",
        "nt_identity","ts","tv","ts_tv_ratio","syn_codons","nonsyn_codons",
        "ambig_multihit_codons","gap_events","max_gap_cluster_codons",
        "gap_fraction_codons","low_identity_windows(<20%)",
        "gc_abs_diff","lcr_fraction_mean","repeat_fraction_mean"
    ]
    ensure_cols(df, req)

    # Pairwise composite scores and balanced labels
    score_100, reasons = compute_pair_scores(df)
    labels, qinfo = assign_pair_bins(score_100, high_q=args.high_q, intermediate_q=args.intermediate_q)

    out = df.copy()
    out["reassess_score"] = np.round(score_100, 2)
    out["reassess_label"] = labels
    out["reassess_top_reasons"] = reasons

    out_csv = os.path.join(outdir, "reassessed_pairs.csv")
    out.to_csv(out_csv, index=False)

    # Gene-wide metrics and label
    gene_metrics = compute_gene_metrics(
        pairs_df=df,
        pair_score_100=score_100,
        pair_labels=labels,
        coverage_low_thresh=args.coverage_low_thresh,
        lcr_flag=args.lcr_flag,
        rep_flag=args.rep_flag,
        tail_q=args.tail_q
    )
    gene_label = label_gene(
        gene_metrics["gene_score_0_100"],
        cut_low=args.gene_low_cut,
        cut_high=args.gene_high_cut
    )

    # Save gene summary (json + txt)
    gene_json = {
        "gene_label": gene_label,
        "gene_score_0_100": round(gene_metrics["gene_score_0_100"], 2),
        "cutoffs": {
            "pair_high_q": args.high_q,
            "pair_intermediate_q": args.intermediate_q,
            "gene_low_cut": args.gene_low_cut,
            "gene_high_cut": args.gene_high_cut
        },
        "pair_score_cutoffs_0_100": qinfo,
        "drivers": {
            "tail_score_q90": round(gene_metrics["tail_score_q90"], 2),
            "share_high_pairs": round(gene_metrics["share_high_pairs"], 3),
            "coverage_low_share": round(gene_metrics["coverage_low_share"], 3),
            "aa_identity_tail_q10": round(gene_metrics["aa_identity_tail_q10"], 3),
            "lcr_mean_over_pairs": round(gene_metrics["lcr_mean_over_pairs"], 3),
            "repeat_mean_over_pairs": round(gene_metrics["repeat_mean_over_pairs"], 3),
            "complexity_bump": round(gene_metrics["complexity_bump"], 3)
        }
    }
    with open(os.path.join(outdir, "gene_summary.json"), "w") as fh:
        json.dump(gene_json, fh, indent=2)

    with open(os.path.join(outdir, "gene_summary.txt"), "w") as fh:
        fh.write("Gene-wide BLAST-miss likelihood: %s\n" % gene_label.upper())
        fh.write("Gene score (0-100): %.2f\n" % gene_metrics["gene_score_0_100"])
        fh.write("Drivers:\n")
        fh.write("  tail_score_q90 (0-100): %.2f\n" % gene_metrics["tail_score_q90"])
        fh.write("  share_high_pairs: %.3f\n" % gene_metrics["share_high_pairs"])
        fh.write("  coverage_low_share: %.3f\n" % gene_metrics["coverage_low_share"])
        fh.write("  aa_identity_tail_q10: %.3f\n" % gene_metrics["aa_identity_tail_q10"])
        fh.write("  lcr_mean_over_pairs: %.3f\n" % gene_metrics["lcr_mean_over_pairs"])
        fh.write("  repeat_mean_over_pairs: %.3f\n" % gene_metrics["repeat_mean_over_pairs"])
        fh.write("  complexity_bump: %.3f\n" % gene_metrics["complexity_bump"])

    # Quick plots
    hist_png = os.path.join(outdir, "score_hist.png")
    fig = plt.figure(figsize=(6, 3.6))
    plt.hist(score_100, bins=30)
    plt.xlabel("Composite pair score")
    plt.ylabel("Count")
    plt.title("Pairwise score distribution")
    save_fig(fig, hist_png)

    sc_png = os.path.join(outdir, "scatter_score_vs_identity.png")
    fig = plt.figure(figsize=(6, 3.6))
    plt.plot(df["aa_identity"].astype(float), score_100, marker=".", linestyle="None", alpha=0.6)
    plt.xlabel("AA identity")
    plt.ylabel("Composite pair score")
    plt.title("Score vs AA identity")
    save_fig(fig, sc_png)

    # Lightweight HTML
    html = os.path.join(outdir, "reassess_report.html")
    counts = out["reassess_label"].value_counts().to_dict()
    total = len(out)
    with open(html, "w") as H:
        H.write("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reassessed BLAST-miss Risk (with Gene-wide Classification)</title>
<style>
body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif; margin: 18px; }
h1,h2,h3 { margin-top: 1em; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
th { background:#fafafa; position: sticky; top: 0; }
td:first-child, th:first-child { text-align: left; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; }
.badge.low { background:#e7f7ed; color:#056e2e; }
.badge.intermediate { background:#fff5d6; color:#8a6100; }
.badge.high { background:#ffe8e6; color:#9a1d0d; }
.small { color:#666; font-size: 12px; }
.figure { margin: 10px 0 16px 0; }
.container { max-width: 1240px; margin: 0 auto; }
</style>
</head><body><div class="container">
""")
        H.write("<h1>Reassessed BLAST-miss Risk</h1>\n")
        H.write("<p><b>Input:</b> %s &nbsp;&middot;&nbsp; <b>Total pairs:</b> %d</p>\n" % (os.path.basename(indir), total))

        H.write("<h2>Gene-wide classification</h2>\n")
        badge = "<span class='badge %s'>%s</span>" % (gene_label, gene_label.upper())
        H.write("<p><b>Likelihood BLAST would miss orthologs from other distant vertebrates:</b> %s</p>\n" % badge)
        H.write("<p>Gene score (0-100): %.2f &nbsp;&middot;&nbsp; Cutoffs: LOW < %.1f, HIGH >= %.1f</p>\n"
                % (gene_metrics["gene_score_0_100"], args.gene_low_cut, args.gene_high_cut))

        H.write("<h3>Drivers</h3>\n")
        H.write("<ul>\n")
        H.write("<li>Tail of composite risk (q90): %.2f</li>\n" % gene_metrics["tail_score_q90"])
        H.write("<li>Share of HIGH pairs: %.3f</li>\n" % gene_metrics["share_high_pairs"])
        H.write("<li>Coverage low share (< %.2f): %.3f</li>\n" % (args.coverage_low_thresh, gene_metrics["coverage_low_share"]))
        H.write("<li>AA identity lower tail (q10): %.3f</li>\n" % gene_metrics["aa_identity_tail_q10"])
        H.write("<li>Mean LCR fraction: %.3f</li>\n" % gene_metrics["lcr_mean_over_pairs"])
        H.write("<li>Mean repeat fraction: %.3f</li>\n" % gene_metrics["repeat_mean_over_pairs"])
        H.write("<li>Complexity bump: %.3f</li>\n" % gene_metrics["complexity_bump"])
        H.write("</ul>\n")

        H.write("<div class='figure'><img loading='lazy' src='%s' alt='Histogram' style='max-width:100%%'></div>\n" % os.path.basename(hist_png))
        H.write("<div class='figure'><img loading='lazy' src='%s' alt='Score vs identity' style='max-width:100%%'></div>\n" % os.path.basename(sc_png))

        H.write("<h2>Balanced pair labels</h2>\n")
        H.write("<p>Cutoffs (score 0-100): HIGH >= %.2f; INTERMEDIATE >= %.2f; LOW below.</p>\n" % (qinfo["q_high"], qinfo["q_inter"]))
        H.write("<ul>\n")
        H.write("<li>LOW: %d</li>\n" % counts.get("low", 0))
        H.write("<li>INTERMEDIATE: %d</li>\n" % counts.get("intermediate", 0))
        H.write("<li>HIGH: %d</li>\n" % counts.get("high", 0))
        H.write("</ul>\n")

        H.write("<h3>Top rows (showing up to %d)</h3>\n" % args.max_table_rows)
        H.write("<p class='small'>Full CSV: <a href='reassessed_pairs.csv'>reassessed_pairs.csv</a></p>\n")
        H.write("<table>\n<thead><tr>"
                "<th>seqA</th><th>seqB</th><th>AA id</th><th>AA cov</th>"
                "<th>Longest block</th><th>Max gap cluster</th><th>< 20%% win</th>"
                "<th>Gap frac</th><th>LCR mean</th><th>Repeat mean</th><th>|GC diff|</th>"
                "<th>Score</th><th>Label</th><th>Top reasons</th>"
                "</tr></thead><tbody>\n")
        # Sort: high first, then intermediate, then low
        order = {"high": 0, "intermediate": 1, "low": 2}
        show = out.assign(_o=out["reassess_label"].map(order)).sort_values(["_o","reassess_score"], ascending=[True, False]).drop(columns=["_o"])
        for _, r in show.head(args.max_table_rows).iterrows():
            b = "<span class='badge %s'>%s</span>" % (r["reassess_label"], str(r["reassess_label"]).upper())
            H.write(
                "<tr><td>%s</td><td>%s</td>"
                "<td>%.3f</td><td>%.3f</td>"
                "<td>%d</td><td>%d</td><td>%d</td>"
                "<td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td>"
                "<td>%.2f</td><td>%s</td><td>%s</td></tr>\n" % (
                    r["seqA"], r["seqB"],
                    float(r["aa_identity"]), float(r["aa_coverage"]),
                    int(r["longest_contiguous_identical_aa"]), int(r["max_gap_cluster_codons"]), int(r["low_identity_windows(<20%)"]),
                    float(r["gap_fraction_codons"]), float(r["lcr_fraction_mean"]), float(r["repeat_fraction_mean"]), float(r["gc_abs_diff"]),
                    float(r["reassess_score"]), b, r["reassess_top_reasons"]
                )
            )
        H.write("</tbody></table>\n</div></body></html>\n")

    print("[ok] Reassessed CSV:", out_csv)
    print("[ok] Gene summary:", os.path.join(outdir, "gene_summary.txt"))
    print("[ok] HTML:", html)

if __name__ == "__main__":
    main()
