#!/usr/bin/env python3
"""
Karyotype-like plot of two gene sets with cluster detection and co-localisation statistics.

Features
- Two rows per chromosome (set1 on top, set2 below)
- Dots plotted slightly above baselines for visibility
- Vertical gene-name labels placed below dots with simple de-overlap
- Red boxes mark clusters (dense windows)
- Permutation test for co-localisation controlling for chromosome-specific density
  and gene-length distribution
"""

import argparse, gzip, io, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ------------------------------ IO helpers --------------------------------- #

def smart_open(path: str):
    return io.TextIOWrapper(gzip.open(path, 'rb')) if path.endswith('.gz') else open(path, 'r', encoding='utf-8', errors='replace')


def parse_attributes(attr_field: str):
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

    df = pd.DataFrame(rows, columns=['chrom', 'feature', 'start', 'end', 'gene_id', 'gene_name']).dropna()

    # Prefer true gene spans; otherwise aggregate per gene_id.
    if (df['feature'] == 'gene').any():
        df = df[df['feature'] == 'gene'].copy()
    else:
        df = df.groupby(['chrom', 'gene_id', 'gene_name'], observed=True, as_index=False).agg(
            start=('start', 'min'),
            end=('end', 'max')
        )
    df = df[['chrom', 'start', 'end', 'gene_id', 'gene_name']].sort_values(['chrom', 'start', 'end']).reset_index(drop=True)
    df['mid'] = ((df['start'] + df['end']) // 2).astype(int)  # midpoint used for plotting/metrics
    return df


def load_list(path: str):
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return [x.strip() for x in fh if x.strip() and not x.startswith('#')]


def match(df: pd.DataFrame, names):
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


# ------------------------------ Clustering --------------------------------- #

def find_clusters(df: pd.DataFrame, window_bp: int, min_genes: int):
    """
    Sliding-window cluster finder on midpoints. Merges overlapping windows per chromosome.
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

    # Merge overlapping within chromosome
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
    Greedy label-tier assignment: left-to-right stacking when labels are too close.
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


def plot(df1: pd.DataFrame, df2: pd.DataFrame, clusters, chrom_lengths: dict,
         out: str, fig_width: float = 14.0, row_height: float = 0.6, label_min_dx_bp: int = 1_000_000):
    chroms = [c for c in sorted(chrom_lengths.keys()) if (c in set(df1.chrom) or c in set(df2.chrom))]
    if not chroms:
        chroms = sorted(chrom_lengths.keys())

    fig, ax = plt.subplots(figsize=(fig_width, max(4, row_height * 2 * len(chroms))))
    ymap = {c: i * 2 for i, c in enumerate(chroms)}
    xmax = max(int(chrom_lengths[c]) for c in chroms)

    # Baselines
    for c in chroms:
        y = ymap[c]
        L = int(chrom_lengths[c])
        ax.hlines(y, 0, L, linewidth=2, color='gray')
        ax.hlines(y - 1, 0, L, linewidth=2, color='gray')

    def scatter_and_label(sub, base_y, color, label):
        if sub.empty:
            return
        xs = sub['mid'].to_numpy()
        ys = np.full(len(xs), base_y + 0.12)
        ax.scatter(xs, ys, s=18, color=color, label=label, zorder=3)
        # Labels: vertical, below point; stack to avoid collisions
        order = np.argsort(xs)
        xs_sorted = xs[order]
        names_sorted = sub['label'].to_numpy()[order]
        lvls = _assign_label_levels(xs_sorted, min_dx_bp=label_min_dx_bp, max_levels=4)
        for x, lvl, name in zip(xs_sorted, lvls, names_sorted):
            y_text = base_y - (0.18 + 0.17 * lvl)
            ax.text(x, y_text, str(name), rotation=90, ha='center', va='top', fontsize=7)

    # Ensure label col
    if not df1.empty and 'label' not in df1.columns:
        df1 = df1.copy(); df1['label'] = df1['gene_name']
    if not df2.empty and 'label' not in df2.columns:
        df2 = df2.copy(); df2['label'] = df2['gene_name']

    # Per-chromosome rows
    for c in chroms:
        sub1 = df1[df1['chrom'] == c]
        sub2 = df2[df2['chrom'] == c]
        scatter_and_label(sub1, ymap[c], 'tab:blue', 'set1')
        scatter_and_label(sub2, ymap[c] - 1, 'tab:orange', 'set2')

    # Cluster boxes (span both rows)
    for c, s, e in clusters:
        if c not in ymap:
            continue
        y = ymap[c]
        w = max(1, e - s)
        ax.add_patch(mpatches.Rectangle((s, y - 1.6), w, 2.3, fill=False, ec='red', lw=1.2))

    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylim(-2.5, ymap[chroms[-1]] + 2.0)
    ax.set_yticks([ymap[c] - 0.5 for c in chroms])
    ax.set_yticklabels(chroms)
    ax.set_xlabel('Position (Mb)')
    ax.legend(loc='upper right', frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}"))
    ax.margins(x=0.01)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ------------------------------ Co-localisation ---------------------------- #

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
        if not np.isnan(mu):
            nn_means.append(mu)
        if not np.isnan(med):
            nn_medians.append(med)
    mean_nn = float(np.mean(nn_means)) if nn_means else np.nan
    median_nn = float(np.median(nn_medians)) if nn_medians else np.nan
    return {'pairs_within_window': int(pair_count), 'mean_nearest_bp': mean_nn, 'median_nearest_bp': median_nn}


def stratify_bins(lengths, n_bins):
    if n_bins <= 1 or len(lengths) == 0:
        return np.zeros(len(lengths), dtype=int)
    qs = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    return np.digitize(lengths, qs[1:-1], right=True).astype(int)


def run_permutation_test(df_all: pd.DataFrame, s1_names, s2_names, window_bp, n_perm=1000, length_bins=4, seed=42):
    if len(s1_names) == 0 or len(s2_names) == 0:
        return {
            'window_bp': int(window_bp),
            'observed_pairs_within_window': 0,
            'p_value_pairs': np.nan,
            'observed_mean_nearest_bp': np.nan,
            'p_value_mean_nearest': np.nan,
            'observed_median_nearest_bp': np.nan,
            'p_value_median_nearest': np.nan,
            'permutations': int(n_perm),
            'length_bins': int(length_bins)
        }

    universe = df_all[['chrom', 'start', 'end', 'gene_name', 'mid']].copy()
    universe['length'] = (universe['end'] - universe['start']).astype(int)

    # Assign length bins per chromosome so s1/s2 inherit len_bin
    universe['len_bin'] = 0
    if length_bins > 1:
        for chrom, sub in universe.groupby('chrom', observed=True):
            idx = sub.index
            universe.loc[idx, 'len_bin'] = stratify_bins(sub['length'].to_numpy(), length_bins)

    s1 = universe[universe['gene_name'].isin(set(s1_names))].copy()
    s2 = universe[universe['gene_name'].isin(set(s2_names))].copy()

    obs = compute_coloc_metrics(s1, s2, window_bp)

    # Counts per stratum
    def counts(df):
        return df.groupby(['chrom', 'len_bin'], observed=True).size().to_dict() if not df.empty else {}
    target_s1 = counts(s1)
    target_s2 = counts(s2)

    grouped = universe.groupby(['chrom', 'len_bin'], observed=True)
    strata_idx = {k: np.fromiter(v, dtype=int) for k, v in grouped.groups.items()}
    keys = list(strata_idx.keys())

    rng = np.random.default_rng(seed)
    perm_pairs = np.zeros(n_perm, dtype=int)
    perm_mean_nn = np.zeros(n_perm, dtype=float)
    perm_median_nn = np.zeros(n_perm, dtype=float)

    for b in range(n_perm):
        sel1, sel2, ok = [], [], True
        for k in keys:
            idxs = strata_idx.get(k, np.array([], dtype=int))
            n1 = target_s1.get(k, 0)
            n2 = target_s2.get(k, 0)
            if n1 == 0 and n2 == 0:
                continue
            if len(idxs) < n1 + n2:
                ok = False
                break
            pick = rng.choice(idxs, size=n1 + n2, replace=False)
            sel1.extend(pick[:n1])
            sel2.extend(pick[n1:])
        if not ok:
            # Fallback preserving per-chrom counts only
            sel1, sel2 = [], []
            for chrom, idxs_list in universe.groupby('chrom', observed=True).groups.items():
                idxs = np.fromiter(idxs_list, dtype=int)
                n1 = (s1['chrom'] == chrom).sum()
                n2 = (s2['chrom'] == chrom).sum()
                if len(idxs) < n1 + n2:
                    # Ultimate fallback: sample globally
                    idxs = np.arange(len(universe))
                    n1 = len(s1)
                    n2 = len(s2)
                pick = rng.choice(idxs, size=n1 + n2, replace=False)
                sel1.extend(pick[:n1])
                sel2.extend(pick[n1:])
        ps1 = universe.iloc[sel1][['chrom', 'mid']].rename(columns={'mid': 'mid'})
        ps2 = universe.iloc[sel2][['chrom', 'mid']].rename(columns={'mid': 'mid'})
        m = compute_coloc_metrics(ps1.rename(columns={'mid': 'mid'}).assign(start=lambda d: d['mid']),
                                  ps2.rename(columns={'mid': 'mid'}).assign(start=lambda d: d['mid']),
                                  window_bp)
        perm_pairs[b] = m['pairs_within_window']
        perm_mean_nn[b] = m['mean_nearest_bp']
        perm_median_nn[b] = m['median_nearest_bp']

    # P-values
    p_pairs = (np.sum(perm_pairs >= obs['pairs_within_window']) + 1) / (n_perm + 1)
    p_mean = (np.sum(perm_mean_nn <= obs['mean_nearest_bp']) + 1) / (n_perm + 1) if not np.isnan(obs['mean_nearest_bp']) else np.nan
    p_median = (np.sum(perm_median_nn <= obs['median_nearest_bp']) + 1) / (n_perm + 1) if not np.isnan(obs['median_nearest_bp']) else np.nan

    return {
        'window_bp': int(window_bp),
        'observed_pairs_within_window': int(obs['pairs_within_window']),
        'p_value_pairs': float(p_pairs),
        'observed_mean_nearest_bp': float(obs['mean_nearest_bp']) if not np.isnan(obs['mean_nearest_bp']) else np.nan,
        'p_value_mean_nearest': float(p_mean) if not np.isnan(p_mean) else np.nan,
        'observed_median_nearest_bp': float(obs['median_nearest_bp']) if not np.isnan(obs['median_nearest_bp']) else np.nan,
        'p_value_median_nearest': float(p_median) if not np.isnan(p_median) else np.nan,
        'permutations': int(n_perm),
        'length_bins': int(length_bins)
    }


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
    # Co-localisation arguments
    ap.add_argument('--coloc-window-bp', type=int, default=500_000, help='Distance threshold for co-localisation tests')
    ap.add_argument('--permutations', type=int, default=1000, help='Number of permutations for significance testing')
    ap.add_argument('--length-bins', type=int, default=4, help='Number of gene-length bins for stratified shuffling (0/1 disables)')
    ap.add_argument('--random-seed', type=int, default=42)

    a = ap.parse_args()

    df = load_gene_positions(a.gtf)
    chrom_lengths = df.groupby('chrom', observed=True)['end'].max().to_dict()

    s1_names = load_list(a.set1)
    s2_names = load_list(a.set2)
    s1 = match(df, s1_names)
    s2 = match(df, s2_names)

    # One row per gene (span already aggregated). Ensure mid and labels exist.
    if not s1.empty:
        s1 = s1.drop_duplicates(subset=['chrom', 'gene_id', 'gene_name']).copy()
        if 'mid' not in s1.columns:
            s1['mid'] = ((s1['start'] + s1['end']) // 2).astype(int)
        s1['label'] = s1['gene_name']
    if not s2.empty:
        s2 = s2.drop_duplicates(subset=['chrom', 'gene_id', 'gene_name']).copy()
        if 'mid' not in s2.columns:
            s2['mid'] = ((s2['start'] + s2['end']) // 2).astype(int)
        s2['label'] = s2['gene_name']

    allm = pd.concat([s1.assign(set_label='set1'), s2.assign(set_label='set2')], ignore_index=True)
    allm.to_csv(f"{a.out}.gene_positions.tsv", sep='\t', index=False)

    cl = find_clusters(allm, a.cluster_window_bp, a.min_genes)
    with open(f"{a.out}.clusters.tsv", "w", encoding='utf-8') as fh:
        fh.write("chrom\tstart\tend\n")
        for c, s, e in cl:
            fh.write(f"{c}\t{s}\t{e}\n")

    plot(
        s1, s2, cl, chrom_lengths, a.out,
        fig_width=a.fig_width,
        row_height=a.row_height,
        label_min_dx_bp=a.label_min_dx_bp
    )

    # Co-localisation stats
    stats = run_permutation_test(df, s1_names, s2_names, a.coloc_window_bp,
                                 n_perm=a.permutations, length_bins=a.length_bins, seed=a.random_seed)
    pd.DataFrame([stats]).to_csv(f"{a.out}.colocalisation_stats.tsv", sep='\t', index=False)


if __name__ == '__main__':
    main()
