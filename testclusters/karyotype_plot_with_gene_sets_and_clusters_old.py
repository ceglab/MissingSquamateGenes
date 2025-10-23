#!/usr/bin/env python3
"""
Karyotype-like plot of two gene sets on a GTF genome, with cluster detection.

Inputs
------
- GTF file (optionally .gz). Gene-level entries are preferred; if missing, the
  script will infer genes from any feature having a gene identifier.
- Two text files, each with one gene name per line (comments using '#'
  allowed). Names are matched case-insensitively against common GTF attributes
  (gene_name, gene, Name, gene_symbol) and gene_id as a fallback.

Outputs
-------
- <out_prefix>.karyoplot.png / .pdf : karyotype-like plot with two sets colored
  differently and clusters boxed in red.
- <out_prefix>.clusters.tsv        : table of detected clusters.
- <out_prefix>.gene_positions.tsv  : table of all matched genes and positions.
- <out_prefix>.unmatched.txt       : any queried genes not found in the GTF.

Cluster definition
------------------
A cluster is any set of = min_genes genes on the same chromosome whose
positions lie within a sliding window of `cluster_window_bp`.
Overlapping/adjacent windows are merged. Clusters are reported irrespective of
which set the genes belong to (combined density), but per-set counts are also
provided in the output table.

Example
-------
python karyotype_plot_with_gene_sets_and_clusters.py \
    --gtf genome.gtf.gz \
    --set1 setA.txt \
    --set2 setB.txt \
    --out mygenes \
    --cluster-window-bp 2000000 \
    --min-genes 3

"""
from __future__ import annotations
import argparse
import gzip
import io
import sys
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

try:
    import pandas as pd
except ImportError as e:
    sys.stderr.write("This script requires pandas. Install with: pip install pandas\n")
    raise


# ------------------------------ Utilities ---------------------------------- #

ATTR_KEYS = ["gene_name", "gene", "Name", "gene_symbol", "gene_id"]


def smart_open(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_attributes(attr_field: str) -> Dict[str, str]:
    """Parse the 9th GTF column into a dict (robust to minor format quirks)."""
    d: Dict[str, str] = {}
    # Standard key "value"; pairs separated by ';'
    # Also handle key=value forms just in case.
    for part in filter(None, [p.strip() for p in attr_field.strip().strip(";").split(";")]):
        if not part:
            continue
        if "=" in part and '"' not in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
        else:
            m = re.match(r"([A-Za-z0-9_]+)\s+\"(.*)\"", part)
            if m:
                d[m.group(1)] = m.group(2)
            else:
                # Fallback: key value without quotes
                toks = part.split()
                if len(toks) >= 2:
                    d[toks[0]] = " ".join(toks[1:]).strip('"')
    return d


def extract_gene_name(attrs: Dict[str, str]) -> Tuple[str, str]:
    """Return (gene_name_like, gene_id_like)."""
    gene_id = attrs.get("gene_id") or attrs.get("ID") or ""
    gene_name = ""
    for k in ATTR_KEYS:
        if k in attrs and attrs[k]:
            gene_name = attrs[k]
            break
    if not gene_name:
        gene_name = gene_id
    return gene_name, gene_id


# ------------------------------ GTF parsing -------------------------------- #

def load_gene_positions(gtf_path: str) -> pd.DataFrame:
    """Load gene positions from a GTF into a DataFrame.

    Columns: chrom, start, end, strand, gene_id, gene_name
    """
    rows: List[Tuple[str, int, int, str, str, str]] = []
    with smart_open(gtf_path) as fh:
        for ln in fh:
            if not ln or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            # Prefer true gene features, else use any feature carrying a gene_id
            if feature != "gene" and "gene_id" not in attrs:
                continue
            a = parse_attributes(attrs)
            gname, gid = extract_gene_name(a)
            if not gname and not gid:
                continue
            try:
                s = int(start)
                e = int(end)
            except ValueError:
                continue
            rows.append((chrom, min(s, e), max(s, e), strand, gid, gname))
    if not rows:
        raise RuntimeError("No gene-like records found in GTF. Ensure the file has 'gene' features or attributes with gene_id.")
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "gene_id", "gene_name"])\
           .drop_duplicates(subset=["chrom", "start", "end", "gene_id", "gene_name"])

    # Build natural sort order for chromosomes (1..22, X, Y, MT; otherwise lexical)
    def chr_key(c: str):
        x = re.sub(r"^(chr|chromosome)_?", "", c, flags=re.IGNORECASE)
        try:
            return (0, int(x))
        except ValueError:
            # Common special: X,Y,M/MT
            order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
            return (1, order.get(x.upper(), 1000), x)

    df["chrom"] = df["chrom"].astype(str)
    chrom_order = sorted(df["chrom"].unique(), key=chr_key)
    df["chrom"] = pd.Categorical(df["chrom"], categories=chrom_order, ordered=True)
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


# ------------------------------ Gene set IO -------------------------------- #

def load_gene_set(path: str) -> List[str]:
    names: List[str] = []
    with smart_open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            names.append(ln)
    return names


def match_genes(df: pd.DataFrame, query_names: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Match query_names to df gene_name/gene_id case-insensitively.

    Returns matched subset (with a column 'query_name') and a list of unmatched names.
    """
    # Build maps for fast case-insensitive lookup
    by_name = defaultdict(list)
    by_id = defaultdict(list)
    for idx, row in df.iterrows():
        by_name[row.gene_name.lower()].append(idx)
        by_id[row.gene_id.lower()].append(idx)
    matched_rows = []
    unmatched = []
    for q in query_names:
        ql = q.lower()
        idxs = by_name.get(ql) or by_id.get(ql)
        if idxs:
            for i in idxs:
                r = df.loc[i].to_dict()
                r["query_name"] = q  # preserve original spelling
                matched_rows.append(r)
        else:
            # Try relaxed match: strip version suffix after '.' (e.g., ENST.. or NM..)
            qcore = ql.split('.') [0]
            idxs = by_name.get(qcore) or by_id.get(qcore)
            if idxs:
                for i in idxs:
                    r = df.loc[i].to_dict()
                    r["query_name"] = q
                    matched_rows.append(r)
            else:
                unmatched.append(q)
    if matched_rows:
        mdf = pd.DataFrame(matched_rows)
        # Keep original category order of chrom
        mdf["chrom"] = pd.Categorical(mdf["chrom"], categories=df["chrom"].cat.categories, ordered=True)
        mdf = mdf.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    else:
        mdf = pd.DataFrame(columns=list(df.columns) + ["query_name"])
    return mdf, unmatched


# ------------------------------ Clustering --------------------------------- #

@dataclass
class Cluster:
    chrom: str
    start: int
    end: int
    gene_count: int
    set1_count: int
    set2_count: int
    gene_names: List[str]


def find_clusters(df: pd.DataFrame, window_bp: int, min_genes: int) -> List[Cluster]:
    clusters: List[Cluster] = []
    for chrom, sub in df.groupby("chrom", sort=False, observed=True):
        pos = sub[["start", "end"]].mean(axis=1).astype(int).values
        order = np.argsort(pos)
        pos = pos[order]
        subo = sub.iloc[order]
        left = 0
        n = len(pos)
        while left < n:
            right = left
            # expand window
            while right < n and pos[right] - pos[left] <= window_bp:
                right += 1
            count = right - left
            if count >= min_genes:
                cstart = int(pos[left])
                cend = int(pos[right - 1])
                genes = list(subo.iloc[left:right]["gene_name"].astype(str))
                set1c = int((subo.iloc[left:right]["set_label"] == "set1").sum())
                set2c = int((subo.iloc[left:right]["set_label"] == "set2").sum())
                clusters.append(Cluster(str(chrom), cstart, cend, count, set1c, set2c, genes))
            left += 1
        # Merge overlapping/adjacent clusters
        clusters = merge_clusters(clusters)
    return clusters


def merge_clusters(clusters: List[Cluster]) -> List[Cluster]:
    if not clusters:
        return []
    # sort by chrom then start
    clusters.sort(key=lambda c: (c.chrom, c.start, c.end))
    merged: List[Cluster] = []
    cur = clusters[0]
    for c in clusters[1:]:
        if c.chrom == cur.chrom and c.start <= cur.end:
            # merge
            cur.end = max(cur.end, c.end)
            cur.gene_count += c.gene_count
            cur.set1_count += c.set1_count
            cur.set2_count += c.set2_count
            cur.gene_names.extend(c.gene_names)
        else:
            # finalize, but de-duplicate names
            cur.gene_names = sorted(dict.fromkeys(cur.gene_names))
            merged.append(cur)
            cur = c
    cur.gene_names = sorted(dict.fromkeys(cur.gene_names))
    merged.append(cur)
    return merged


# ------------------------------ Plotting ----------------------------------- #

def plot_karyotype(df_all: pd.DataFrame, clusters: List[Cluster], out_prefix: str):
    # Determine chromosome lengths from max end in df
    # Ensure no unused categorical levels and no NaNs before computing lengths
    if hasattr(df_all["chrom"].dtype, "categories"):
        df_all["chrom"] = df_all["chrom"].cat.remove_unused_categories()
    df_all = df_all.dropna(subset=["end"])  # safety in case of any stray NaNs
    chrom_lengths = df_all.groupby("chrom", observed=True)["end"].max().astype(int)
    chroms = list(chrom_lengths.index)
    y_positions = {c: i for i, c in enumerate(chroms)}

    fig, ax = plt.subplots(figsize=(14, max(6, len(chroms) * 0.35)))

    # Draw chromosome baselines
    for chrom in chroms:
        y = y_positions[chrom]
        ax.hlines(y, 0, chrom_lengths[chrom], linewidth=2)

    # Plot genes
    color_map = {"set1": "tab:blue", "set2": "tab:orange"}
    for set_label, sub in df_all.groupby("set_label"):
        if sub.empty:
            continue
        xs = sub[["start", "end"]].mean(axis=1).values
        ys = [y_positions[c] for c in sub["chrom"]]
        ax.scatter(xs, ys, s=18, label=set_label, alpha=0.9, edgecolor="none", zorder=3, c=color_map.get(set_label, None))

    # Draw cluster boxes (red rectangles)
    for cl in clusters:
        y = y_positions[cl.chrom]
        rect = mpatches.Rectangle((cl.start, y - 0.35), cl.end - cl.start if cl.end > cl.start else 1,
                                  0.7, fill=False, linewidth=1.5, edgecolor="red")
        ax.add_patch(rect)

    # Axes formatting
    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels([str(c) for c in chroms])
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Chromosome")
    ax.legend(title="Gene set", loc="upper right", frameon=False)

    # X in Mb
    ax.set_xlim(0, max(chrom_lengths.max(), 1))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}"))

    ax.set_title("Karyotype-like distribution of gene sets with clusters")
    ax.margins(x=0.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.karyoplot.{ext}", dpi=300)
    plt.close(fig)


# ------------------------------ Main --------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Plot two gene sets on a GTF genome and detect clusters")
    ap.add_argument("--gtf", required=True, help="Path to GTF (.gtf or .gtf.gz)")
    ap.add_argument("--set1", required=True, help="Text file: one gene name per line (set 1)")
    ap.add_argument("--set2", required=True, help="Text file: one gene name per line (set 2)")
    ap.add_argument("--out", required=True, help="Output prefix")
    ap.add_argument("--cluster-window-bp", type=int, default=1_000_000, help="Window size (bp) to call clusters [default: 1,000,000]")
    ap.add_argument("--min-genes", type=int, default=3, help="Minimum genes within window to define a cluster [default: 3]")
    args = ap.parse_args()

    print("Loading GTF", file=sys.stderr)
    df = load_gene_positions(args.gtf)

    print("Loading gene sets", file=sys.stderr)
    set1_list = load_gene_set(args.set1)
    set2_list = load_gene_set(args.set2)

    print("Matching genes", file=sys.stderr)
    m1, u1 = match_genes(df, set1_list)
    m2, u2 = match_genes(df, set2_list)

    if m1.empty and m2.empty:
        raise SystemExit("No genes from either set matched the GTF. Check naming conventions and attributes in the GTF.")

    m1 = m1.assign(set_label="set1")
    m2 = m2.assign(set_label="set2")
    allm = pd.concat([m1, m2], ignore_index=True)

    # Prefer the query_name as display name if available
    allm["display_name"] = allm["query_name"].fillna(allm["gene_name"]).astype(str)
    # Midpoint position for plotting and clustering
    allm["pos"] = allm[["start", "end"]].mean(axis=1).astype(int)

    # Save positions
    pos_out = allm[["chrom", "start", "end", "strand", "gene_id", "gene_name", "display_name", "set_label"]]
    pos_out.to_csv(f"{args.out}.gene_positions.tsv", sep="\t", index=False)

    # Unmatched report
    unmatched = sorted(set(u1) | set(u2))
    if unmatched:
        with open(f"{args.out}.unmatched.txt", "w", encoding="utf-8") as fh:
            fh.write("# Genes not matched to the GTF (case-insensitive search over gene_name/gene/gene_id)\n")
            for g in unmatched:
                fh.write(g + "\n")

    print("Finding clusters", file=sys.stderr)
    clusters = find_clusters(allm, window_bp=args.cluster_window_bp, min_genes=args.min_genes)

    # Write clusters table
    with open(f"{args.out}.clusters.tsv", "w", encoding="utf-8") as fh:
        fh.write("chrom\tstart\tend\tspan_bp\tgene_count\tset1_count\tset2_count\tgenes\n")
        for c in clusters:
            span = max(1, int(c.end) - int(c.start))
            fh.write(f"{c.chrom}\t{c.start}\t{c.end}\t{span}\t{c.gene_count}\t{c.set1_count}\t{c.set2_count}\t{','.join(c.gene_names)}\n")

    print("Plotting", file=sys.stderr)
    plot_karyotype(allm, clusters, args.out)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
