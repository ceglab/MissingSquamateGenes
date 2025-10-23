#!/usr/bin/env python3
"""
Karyotype-like plot of two gene sets with cluster detection, flanking genes row,
vertical labels, and per-cluster co-localisation statistics.

Features
- Two main rows per chromosome: set1 (top) and set2 (below).
- Third row: immediate flanking genes (upstream & downstream) for every gene in either set:
   triangle markers; colored by which set they flank (blue=set1, orange=set2, green=both).
- One dot per gene (gene-span midpoint). Vertical labels under dots with simple de-overlap.
- Clusters (dense windows) boxed in red + per-cluster permutation p-values above each box.
- Outputs:
   <out>.gene_positions.tsv  (matched set genes with coordinates)
   <out>.clusters.tsv        (chrom, start, end, span, counts, p_coloc)
   <out>.png / <out>.pdf     (plot)

Example:
python karyotype_plot_with_gene_sets_and_clusters.py \
  --gtf genome.gtf.gz \
  --set1 setA.txt \
  --set2 setB.txt \
  --out mygenes \
  --cluster-window-bp 2000000 \
  --min-genes 3 \
  --fig-width 16 --row-height 0.9 \
  --label-min-dx-bp 750000 \
  --coloc-window-bp 500000 \
  --permutations 2000 \
  --length-bins 5 \
  --random-seed 13
"""

import argparse, gzip, io, re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ------------------------------ IO helpers --------------------------------- #

def smart_open(path: str):
    return io.TextIOWrapper(gzip.open(path, 'rb')) if path.endswith('.gz') else open(path, 'r', encoding='utf-8', errors='replace')


def parse_attributes(attr_field: str) -> Dict[str, str]:
    d = {}
    for part in filter(None, [p.strip() for p in attr_field.strip().strip(';').split(';')]):
        m = re.match(r'([A-Za-z0-9_]+)\s+"(.*)"', part)
        if m:
            d[m.group(1)] = m.group(2)
        else:
            toks = part.split()
            if len(toks) >= 2:
                d[toks[0]] = ' '.join(toks[1:]).strip('"')
    return d


def load_gene_positions(gtf: str) -> pd.DataFrame:
    """
    Load gene spans from GTF. If 'gene' features exist, use them directly.
    Otherwise, aggregate any features to min(start) max(end) per gene_id.
    Adds midpoint 'mid' for plotting / stats.
    """
    rows = []
    with smart_open(gtf) as f:
        for ln in f:
            if not ln or ln.startswith('#'):
                continue
            p = ln.rstrip('\n').split('\t')
            if len(p) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attr = p
            a = parse_attributes(attr)
            gid = a.get('gene_id', '')
            gname = a.get('gene_name') or a.get('gene') or a.get('Name') or a.get('gene_symbol') or gid
            try:
                s = int(start); e = int(end)
            except ValueError:
                continue
            rows.append((chrom, feature, min(s, e), max(s, e), gid, gname))

    df = pd.DataFrame(rows, columns=['chrom','feature','start','end','gene_id','gene_name']).dropna()

    if (df['feature'] == 'gene').any():
        df = df[df['feature'] == 'gene'].copy()
    else:
        df = df.groupby(['chrom','gene_id','gene_name'], observed=True, as_index=False).agg(
            start=('start','min'), end=('end','max')
        )

    df = df[['chrom','start','end','gene_id','gene_name']].sort_values(['chrom','start','end']).reset_index(drop=True)
    df['mid'] = ((df['start'] + df['end']) // 2).astype(int)
    return df


def load_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return [x.strip() for x in fh if x.strip() and not x.startswith('#')]


def match(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """
    Case-insensitive match by gene_name; fallback to gene_id; allow versionless match.
    Returns a DataFrame subset of df.
    """
    by_name, by_id = {}, {}
    for i, row in df.iterrows():
        by_name.setdefault(str(row.gene_name).lower(), []).append(i)
        by_id.setdefault(str(row.gene_id).lower(), []).append(i)

    hits = []
    for n in names:
        q = n.lower()
        idxs = by_name.get(q) or by_id.get(q)
        if not idxs and '.' in q:
            qcore = q.split('.', 1)[0]
            idxs = by_name.get(qcore) or by_id.get(qcore)
        if idxs:
            for i in idxs:
                hits.append(df.iloc[i])

    return pd.DataFrame(hits).reset_index(drop=True) if hits else pd.DataFrame(columns=df.columns)


# ------------------------------ Flanking genes ----------------------------- #

def find_flanking_genes(genome_df: pd.DataFrame, targets_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    For each target gene, find immediate upstream and downstream genes on the same chromosome.
    Returns unique flank genes with column 'flank_of' set to label ('set1'/'set2' or 'both').
    """
    if targets_df.empty:
        return pd.DataFrame(columns=list(genome_df.columns) + ['flank_of'])

    flanks = []
    for chrom, chrom_df in genome_df.groupby('chrom', observed=True):
        sub_targets = targets_df[targets_df['chrom'] == chrom]
        if sub_targets.empty:
            continue
        cds = chrom_df.sort_values('start').reset_index(drop=False)  # keep original index
        starts = cds['start'].to_numpy()
        for _, t in sub_targets.iterrows():
            # locate insertion point by start coordinate
            pos = int(np.searchsorted(starts, int(t['start'])))
            cand_positions = []
            if pos - 1 >= 0:
                cand_positions.append(pos - 1)      # upstream
            if pos < len(starts):
                cand_positions.append(pos)          # downstream (or current)
            for pidx in cand_positions:
                r = cds.iloc[pidx]
                # if same gene, try the next downstream neighbor
                if r['gene_id'] == t['gene_id']:
                    nxt = pidx + 1
                    if nxt < len(cds):
                        r = cds.iloc[nxt]
                    else:
                        continue
                flanks.append({
                    'chrom': r['chrom'],
                    'start': int(r['start']),
                    'end': int(r['end']),
                    'mid': int(r['mid']),
                    'gene_id': r['gene_id'],
                    'gene_name': r['gene_name'],
                    'flank_of': label
                })

    if not flanks:
        return pd.DataFrame(columns=list(genome_df.columns) + ['flank_of'])

    fdf = pd.DataFrame(flanks)
    # consolidate duplicates and mark 'both'
    fdf = fdf.groupby(['chrom','start','end','gene_id','gene_name','mid'], observed=True, as_index=False)['flank_of'] \
             .agg(lambda s: 'both' if len(set(s)) > 1 else list(s)[0])
    return fdf


# ------------------------------ Clustering --------------------------------- #

def find_clusters(df: pd.DataFrame, window_bp: int, min_genes: int):
    """
    Sliding-window cluster finder on midpoints. Merges overlapping windows per chromosome.
    Returns list of (chrom, start_mid, end_mid).
    """
    clusters = []
    if df.empty:
        return clusters

    for chrom, sub in df.groupby('chrom', sort=False, observed=True):
        pos = np.sort(sub['mid'].to_numpy(dtype=np.int64))
        n = len(pos)
        for left in range(n):
            right = left
            while right < n and pos[right] - pos[left] <= window_bp:
                right += 1
            if right - left >= min_genes:
                clusters.append((str(chrom), int(pos[left]), int(pos[right - 1])))

    # Merge overlapping windows per chromosome
    merged = []
    for c in sorted(clusters, key=lambda x: (x[0], x[1], x[2])):
        if not merged or c[0] != merged[-1][0] or c[1] > merged[-1][2]:
            merged.append([c[0], c[1], c[2]])
        else:
            merged[-1][2] = max(merged[-1][2], c[2])
    return [(c, s, e) for c, s, e in merged]


# ------------------------------ Plotting ----------------------------------- #

def _assign_label_levels(xs, min_dx_bp=1_000_000, max_levels=4):
    """
    Greedy vertical tier assignment to reduce label overlap.
    """
    levels = np.zeros(len(xs), dtype=int)
    last_x_at_level = [-np.inf] * max_levels
    for i, x in enumerate(xs):
        placed = False
        for lvl in range(max_levels):
            if x - last_x_at_level[lvl] >= min_dx_bp:
                levels[i] = lvl
                last_x_at_level[lvl] = x
                placed = True
                break
        if not placed:
            levels[i] = max_levels - 1
            last_x_at_level[max_levels - 1] = x
    return levels


def plot(df1: pd.DataFrame, df2: pd.DataFrame, df_flanks: pd.DataFrame, clusters,
         chrom_lengths: Dict[str, int], out: str,
         fig_width: float = 14.0, row_height: float = 0.6,
         label_min_dx_bp: int = 1_000_000,
         cluster_pvals: Optional[Dict[Tuple[str,int,int], float]] = None):
    # which chroms to show
    chroms = [c for c in sorted(chrom_lengths.keys())
              if (c in set(df1.chrom) or c in set(df2.chrom) or c in set(df_flanks.chrom))]
    if not chroms:
        chroms = sorted(chrom_lengths.keys())

    fig, ax = plt.subplots(figsize=(fig_width, max(5, row_height * 3 * len(chroms))))
    ymap = {c: i * 3 for i, c in enumerate(chroms)}  # three rows per chrom: y, y-1, y-2
    xmax = max(int(chrom_lengths[c]) for c in chroms)

    # Baselines
    for c in chroms:
        y = ymap[c]
        L = int(chrom_lengths[c])
        ax.hlines(y,     0, L, linewidth=2, color='gray')
        ax.hlines(y - 1, 0, L, linewidth=2, color='gray')
        ax.hlines(y - 2, 0, L, linewidth=2, color='gray')

    def scatter_and_label(sub, base_y, color, label):
        if sub.empty:
            return
        xs = sub['mid'].to_numpy()
        ys = np.full(len(xs), base_y + 0.12)
        ax.scatter(xs, ys, s=18, color=color, label=label, zorder=3)
        # vertical labels under dots, stacked
        order = np.argsort(xs)
        xs_sorted = xs[order]
        names_sorted = sub['label'].to_numpy()[order]
        lvls = _assign_label_levels(xs_sorted, min_dx_bp=label_min_dx_bp, max_levels=4)
        for x, lvl, name in zip(xs_sorted, lvls, names_sorted):
            y_text = base_y - (0.18 + 0.17 * lvl)
            ax.text(x, y_text, str(name), rotation=90, ha='center', va='top', fontsize=7)

    # ensure label cols exist
    if not df1.empty and 'label' not in df1.columns:
        df1 = df1.copy(); df1['label'] = df1['gene_name']
    if not df2.empty and 'label' not in df2.columns:
        df2 = df2.copy(); df2['label'] = df2['gene_name']

    # per-chrom plotting
    for c in chroms:
        sub1 = df1[df1['chrom'] == c]
        sub2 = df2[df2['chrom'] == c]
        subf = df_flanks[df_flanks['chrom'] == c]
        scatter_and_label(sub1, ymap[c],     'tab:blue',   'set1')
        scatter_and_label(sub2, ymap[c] - 1, 'tab:orange', 'set2')
        if not subf.empty:
            xs = subf['mid'].to_numpy()
            ys = np.full(len(xs), ymap[c] - 2 + 0.12)
            colors = subf['flank_of'].map({'set1':'tab:blue','set2':'tab:orange','both':'tab:green'}).fillna('tab:gray').to_numpy()
            ax.scatter(xs, ys, s=30, marker='^', c=colors, label='flanks', zorder=3, edgecolor='none')

    # Cluster boxes and p-values
    for (c, s, e) in clusters:
        if c not in ymap:
            continue
        y = ymap[c]
        w = max(1, e - s)
        ax.add_patch(mpatches.Rectangle((s, y - 2.6), w, 3.2, fill=False, ec='red', lw=1.2))
        if cluster_pvals is not None:
            p = cluster_pvals.get((c, s, e), None)
            if p is not None and not (np.isnan(p)):
                ax.text(s + w/2, y + 0.6, f"p={p:.3g}", ha='center', va='bottom', fontsize=8, color='red')

    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylim(-3.2, ymap[chroms[-1]] + 2.2)
    ax.set_yticks([ymap[c] - 1 for c in chroms])
    ax.set_yticklabels(chroms)
    ax.set_xlabel('Position (Mb)')
    ax.legend(loc='upper right', frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}"))
    ax.margins(x=0.01)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ------------------------------ Co-localisation (per-cluster) -------------- #

def _count_pairs_within_window(pos1, pos2, window):
    count = 0
    j_left = 0
    j_right = 0
    n2 = len(pos2)
    for x in pos1:
        while j_left < n2 and pos2[j_left] < x - window:
            j_left += 1
        if j_right < j_left:
            j_right = j_left
        while j_right < n2 and pos2[j_right] <= x + window:
            j_right += 1
        count += max(0, j_right - j_left)
    return int(count)


def _nearest_neighbor_dist(pos1, pos2):
    if len(pos1) == 0 or len(pos2) == 0:
        return np.nan, np.nan
    j = 0
    n2 = len(pos2)
    dists = np.empty(len(pos1), dtype=float)
    for i, x in enumerate(pos1):
        while j + 1 < n2 and abs(pos2[j + 1] - x) <= abs(pos2[j] - x):
            j += 1
        dists[i] = abs(pos2[j] - x)
    return float(np.nanmean(dists)), float(np.nanmedian(dists))


def compute_coloc_metrics(df_set1, df_set2, window_bp):
    """
    Compute number of Set1 Set2 pairs within window on shared chromosomes
    and nearest-neighbor distances (mean/median).
    Expects column 'mid' (int) and 'chrom'.
    """
    pair_count = 0
    nn_means = []
    nn_medians = []
    if df_set1.empty or df_set2.empty:
        return {'pairs_within_window': 0, 'mean_nearest_bp': np.nan, 'median_nearest_bp': np.nan}
    for chrom, sub1 in df_set1.groupby('chrom', observed=True):
        sub2 = df_set2[df_set2['chrom'] == chrom]
        p1 = np.sort(sub1['mid'].to_numpy(dtype=np.int64))
        p2 = np.sort(sub2['mid'].to_numpy(dtype=np.int64))
        if len(p1) == 0 or len(p2) == 0:
            continue
        pair_count += _count_pairs_within_window(p1, p2, window_bp)
        mu, med = _nearest_neighbor_dist(p1, p2)
        if not np.isnan(mu):   nn_means.append(mu)
        if not np.isnan(med):  nn_medians.append(med)
    mean_nn = float(np.mean(nn_means)) if nn_means else np.nan
    median_nn = float(np.median(nn_medians)) if nn_medians else np.nan
    return {'pairs_within_window': int(pair_count), 'mean_nearest_bp': mean_nn, 'median_nearest_bp': median_nn}


def stratify_bins(lengths, n_bins):
    if n_bins <= 1 or len(lengths) == 0:
        return np.zeros(len(lengths), dtype=int)
    qs = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    return np.digitize(lengths, qs[1:-1], right=True).astype(int)


def permute_within_cluster(universe: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, window_bp: int,
                           n_perm: int = 1000, length_bins: int = 4, seed: int = 42) -> float:
    """
    Return p-value for enrichment of pairs_within_window by permuting labels within the cluster.
    Preserves counts of set1 and set2; optionally stratifies by gene-length bins.
    Expects 'chrom','start','end','mid' in universe; and 'chrom','start','end','mid' in s1/s2.
    """
    rng = np.random.default_rng(seed)
    if universe.empty or (len(s1) == 0) or (len(s2) == 0):
        return np.nan

    # observed
    obs = compute_coloc_metrics(s1, s2, window_bp)
    obs_pairs = obs['pairs_within_window']

    # Prepare universe with length and bins; reset index so sampled indices are positional
    U = universe.copy().reset_index(drop=True)
    U['len'] = (U['end'] - U['start']).astype(int)
    if length_bins > 1:
        U['len_bin'] = stratify_bins(U['len'].to_numpy(), length_bins)
        qs = np.quantile(U['len'].to_numpy(), np.linspace(0, 1, length_bins + 1))
        qs = np.unique(qs)
    else:
        U['len_bin'] = 0
        qs = None

    # Attach bins to s1/s2 via merge; fallback to binning with U's quantiles if merge misses
    def attach_bins(df):
        if df.empty:
            return df.copy()
        df2 = df.merge(U[['chrom','start','end','gene_name','len_bin']], on=['chrom','start','end','gene_name'], how='left')
        if 'len_bin' not in df2.columns or df2['len_bin'].isna().any():
            df2 = df2.copy()
            if length_bins > 1 and qs is not None:
                lengths = (df2['end'] - df2['start']).to_numpy()
                df2['len_bin'] = np.digitize(lengths, qs[1:-1], right=True).astype(int)
            else:
                df2['len_bin'] = 0
        return df2

    s1b = attach_bins(s1)
    s2b = attach_bins(s2)

    # counts per bin
    def counts_by_bin(df):
        return df.groupby('len_bin', observed=True).size().to_dict() if not df.empty else {}
    n1b = counts_by_bin(s1b)
    n2b = counts_by_bin(s2b)

    # pre-index by len_bin (positional indices)
    groups = {b: np.array(list(idxs), dtype=int)
              for b, idxs in U.groupby('len_bin', observed=True).groups.items()}

    perm_pairs = np.zeros(n_perm, dtype=int)
    for p in range(n_perm):
        sel1 = []
        sel2 = []
        for bin_id, idxs in groups.items():
            k1 = n1b.get(bin_id, 0)
            k2 = n2b.get(bin_id, 0)
            if k1 + k2 == 0:
                continue
            if len(idxs) < k1 + k2:
                # fallback: sample from full U without length constraint
                idxs = np.arange(len(U))
                k1 = len(s1b)
                k2 = len(s2b)
            picks = rng.choice(idxs, size=k1 + k2, replace=False)
            sel1.extend(picks[:k1])
            sel2.extend(picks[k1:])
        ps1 = U.iloc[sel1][['chrom','mid']].copy(); ps1['start'] = ps1['mid']
        ps2 = U.iloc[sel2][['chrom','mid']].copy(); ps2['start'] = ps2['mid']
        m = compute_coloc_metrics(ps1, ps2, window_bp)
        perm_pairs[p] = m['pairs_within_window']

    p_val = (np.sum(perm_pairs >= obs_pairs) + 1) / (n_perm + 1)
    return float(p_val)


# ------------------------------ Main --------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gtf', required=True)
    ap.add_argument('--set1', required=True)
    ap.add_argument('--set2', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cluster-window-bp', type=int, default=1_000_000)
    ap.add_argument('--min-genes', type=int, default=3)
    ap.add_argument('--fig-width', type=float, default=14.0)
    ap.add_argument('--row-height', type=float, default=0.6)
    ap.add_argument('--label-min-dx-bp', type=int, default=1_000_000,
                    help='Minimum x-distance (bp) between vertical labels before stacking')
    # Per-cluster co-localisation
    ap.add_argument('--coloc-window-bp', type=int, default=500_000, help='Distance threshold for co-localisation tests')
    ap.add_argument('--permutations', type=int, default=1000, help='Number of permutations within each cluster')
    ap.add_argument('--length-bins', type=int, default=4, help='Gene-length bins for stratified shuffling (0/1 disables)')
    ap.add_argument('--random-seed', type=int, default=42)

    a = ap.parse_args()

    # Genome and sets
    df = load_gene_positions(a.gtf)
    chrom_lengths = df.groupby('chrom', observed=True)['end'].max().to_dict()

    s1_names = load_list(a.set1)
    s2_names = load_list(a.set2)
    s1 = match(df, s1_names)
    s2 = match(df, s2_names)

    # Ensure one row per gene with midpoints & labels
    if not s1.empty:
        s1 = s1.drop_duplicates(subset=['chrom','gene_id','gene_name']).copy()
        if 'mid' not in s1.columns:
            s1['mid'] = ((s1['start'] + s1['end']) // 2).astype(int)
        s1['label'] = s1['gene_name']
    if not s2.empty:
        s2 = s2.drop_duplicates(subset=['chrom','gene_id','gene_name']).copy()
        if 'mid' not in s2.columns:
            s2['mid'] = ((s2['start'] + s2['end']) // 2).astype(int)
        s2['label'] = s2['gene_name']

    # Flanking genes (third row)
    f1 = find_flanking_genes(df, s1, 'set1')
    f2 = find_flanking_genes(df, s2, 'set2')
    if not f1.empty and not f2.empty:
        df_flanks = pd.concat([f1, f2], ignore_index=True)
        df_flanks = df_flanks.groupby(['chrom','start','end','gene_id','gene_name','mid'], observed=True, as_index=False)['flank_of'] \
                             .agg(lambda s: 'both' if len(set(s)) > 1 else list(s)[0])
    else:
        df_flanks = pd.concat([f1, f2], ignore_index=True)

    # Export positions
    allm = pd.concat([s1.assign(set_label='set1'), s2.assign(set_label='set2')], ignore_index=True)
    allm.to_csv(f"{a.out}.gene_positions.tsv", sep='\t', index=False)

    # Clusters on union of sets
    cl = find_clusters(allm, a.cluster_window_bp, a.min_genes)

    # Per-cluster co-localisation p-values
    cluster_pvals: Dict[Tuple[str,int,int], float] = {}
    cluster_rows = []
    for (c, s, e) in cl:
        U = df[(df['chrom'] == c) & (df['mid'] >= s) & (df['mid'] <= e)][['chrom','start','end','mid','gene_name']].copy()
        S1 = s1[(s1['chrom'] == c) & (s1['mid'] >= s) & (s1['mid'] <= e)]
        S2 = s2[(s2['chrom'] == c) & (s2['mid'] >= s) & (s2['mid'] <= e)]
        p = permute_within_cluster(U, S1, S2, a.coloc_window_bp,
                                   n_perm=a.permutations, length_bins=a.length_bins, seed=a.random_seed)
        cluster_pvals[(c, s, e)] = p
        cluster_rows.append({
            'chrom': c, 'start': s, 'end': e, 'span_bp': int(e - s),
            'n_genes': int(len(U)), 'set1_in_cluster': int(len(S1)), 'set2_in_cluster': int(len(S2)),
            'p_coloc': p
        })

    # Write clusters table
    pd.DataFrame(cluster_rows).to_csv(f"{a.out}.clusters.tsv", sep='\t', index=False)

    # Plot
    plot(s1, s2, df_flanks, cl, chrom_lengths, a.out,
         fig_width=a.fig_width, row_height=a.row_height,
         label_min_dx_bp=a.label_min_dx_bp, cluster_pvals=cluster_pvals)


if __name__ == '__main__':
    main()
