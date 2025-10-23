#!/usr/bin/env python3
"""
Karyotype-like plot of two gene sets on a GTF genome, with cluster detection and improved visualization.
Each chromosome shows two rows (set1 and set2) and the dots are slightly above the lines for clarity.
"""
import argparse, gzip, io, re, sys
from collections import defaultdict
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import matplotlib.ticker as mticker

def smart_open(path):
    return io.TextIOWrapper(gzip.open(path, 'rb')) if path.endswith('.gz') else open(path)

def parse_attributes(attr_field):
    d = {}
    for part in filter(None, [p.strip() for p in attr_field.strip().strip(';').split(';')]):
        m = re.match(r'([A-Za-z0-9_]+)\s+"(.*)"', part)
        if m:
            d[m.group(1)] = m.group(2)
    return d

def load_gene_positions(gtf):
    rows = []
    with smart_open(gtf) as f:
        for ln in f:
            if ln.startswith('#'): continue
            p = ln.rstrip().split('\t')
            if len(p) < 9: continue
            chrom, source, feature, start, end, score, strand, frame, attr = p
            if feature != 'gene' and 'gene_id' not in attr: continue
            a = parse_attributes(attr)
            gene_id = a.get('gene_id','')
            gene_name = a.get('gene_name', gene_id)
            rows.append((chrom,int(start),int(end),gene_id,gene_name))
    df = pd.DataFrame(rows, columns=['chrom','start','end','gene_id','gene_name'])
    df = df.sort_values(['chrom','start'])
    return df

def load_list(f):
    with open(f) as fh:
        return [x.strip() for x in fh if x.strip() and not x.startswith('#')]

def match(df, names):
    by_name = {n.lower():i for i,n in enumerate(df.gene_name)}
    hits = []
    for n in names:
        if n.lower() in by_name:
            hits.append(df.iloc[[by_name[n.lower()]]])
    return pd.concat(hits) if hits else pd.DataFrame(columns=df.columns)

def find_clusters(df, window_bp, min_genes):
    clusters = []
    for chrom, sub in df.groupby('chrom', observed=True):
        pos = sub.start.values
        pos.sort()
        for i in range(len(pos)):
            j=i
            while j<len(pos) and pos[j]-pos[i]<=window_bp: j+=1
            if j-i>=min_genes:
                clusters.append((chrom,pos[i],pos[j-1]))
    merged=[]
    for c in sorted(clusters,key=lambda x:(x[0],x[1])):
        if not merged or c[0]!=merged[-1][0] or c[1]>merged[-1][2]:
            merged.append(list(c))
        else:
            merged[-1][2]=max(merged[-1][2],c[2])
    return merged

def plot(df1, df2, clusters, chrom_lengths, out, fig_width=14, row_height=0.6):
    chroms = [c for c in sorted(chrom_lengths.keys()) if (c in set(df1.chrom) or c in set(df2.chrom))]
    if not chroms:
        chroms = sorted(chrom_lengths.keys())
    fig, ax = plt.subplots(figsize=(fig_width, max(4, row_height * 2 * len(chroms))))
    ymap = {c: i * 2 for i, c in enumerate(chroms)}
    xmax = max(chrom_lengths[c] for c in chroms)

    for c in chroms:
        y = ymap[c]
        L = chrom_lengths[c]
        ax.hlines(y, 0, L, linewidth=2, color='gray')
        ax.hlines(y - 1, 0, L, linewidth=2, color='gray')

    # Slight vertical offsets so dots appear above lines
    if not df1.empty:
        ax.scatter(df1.start.values, [ymap[c] + 0.1 for c in df1.chrom], s=16, label='set1', color='tab:blue', zorder=3)
    if not df2.empty:
        ax.scatter(df2.start.values, [ymap[c] - 1 + 0.1 for c in df2.chrom], s=16, label='set2', color='tab:orange', zorder=3)

    for c, s, e in clusters:
        if c not in ymap:
            continue
        y = ymap[c]
        w = max(1, e - s)
        ax.add_patch(mpatches.Rectangle((s, y - 1.6), w, 2.2, fill=False, ec='red', lw=1.5))

    ax.set_xlim(0, xmax * 1.02)
    top_y = ymap[chroms[-1]]
    ax.set_ylim(-2, top_y + 1.75)
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gtf',required=True)
    ap.add_argument('--set1',required=True)
    ap.add_argument('--set2',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--cluster-window-bp',type=int,default=1_000_000)
    ap.add_argument('--min-genes',type=int,default=3)
    ap.add_argument('--fig-width',type=float,default=14.0)
    ap.add_argument('--row-height',type=float,default=0.6)
    a=ap.parse_args()

    df=load_gene_positions(a.gtf)
    chrom_lengths = df.groupby('chrom', observed=True)['end'].max().to_dict()
    s1=match(df,load_list(a.set1))
    s2=match(df,load_list(a.set2))
    allm=pd.concat([s1.assign(set_label='set1'),s2.assign(set_label='set2')],ignore_index=True)
    allm.to_csv(f'{a.out}.gene_positions.tsv',sep='\t',index=False)
    cl=find_clusters(allm,a.cluster_window_bp,a.min_genes)
    with open(f'{a.out}.clusters.tsv','w') as fh:
        fh.write('chrom\tstart\tend\n')
        for c,s,e in cl: fh.write(f'{c}\t{s}\t{e}\n')
    plot(s1,s2,cl,chrom_lengths,a.out,fig_width=a.fig_width,row_height=a.row_height)

if __name__=='__main__':
    main()
