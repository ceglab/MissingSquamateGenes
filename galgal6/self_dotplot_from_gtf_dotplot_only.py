#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, shutil, subprocess, tempfile, re
from collections import namedtuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Hit = namedtuple("Hit", ["qstart","qend","sstart","send","pident","length","bitscore","evalue"])

# ----------------------- utilities -----------------------

def die(msg): sys.exit(f"ERROR: {msg}")

def check_exec(name):
    p = shutil.which(name)
    if not p: die(f"Required executable '{name}' not found in PATH.")
    return p

def run_cmd(cmd, cwd=None, capture=True):
    try:
        r = subprocess.run(cmd, cwd=cwd, check=True,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.PIPE if capture else None,
                           text=True)
        return r.stdout if capture else ""
    except subprocess.CalledProcessError as e:
        if capture:
            sys.stderr.write((e.stdout or "") + "\n" + (e.stderr or "") + "\n")
        die(f"Command failed: {' '.join(cmd)}")

def read_fasta_dict(fa_path):
    seqs = {}
    hdr = None
    buf = []
    with open(fa_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if hdr is not None:
                    seqs[hdr] = "".join(buf).upper()
                hdr = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if hdr is not None:
        seqs[hdr] = "".join(buf).upper()
    if not seqs:
        die("No sequences found in FASTA.")
    return seqs

def parse_gtf_attributes(attr_field):
    d = {}
    for m in re.finditer(r'(\S+)\s+"([^"]+)"|(\S+)\s+\'([^\']+)\'', attr_field):
        key = m.group(1) or m.group(3)
        val = m.group(2) or m.group(4)
        d[key] = val
    return d

def load_gene_features(gtf_path, gene_id=None, gene_name=None):
    if not (gene_id or gene_name):
        die("Provide --gene_id or --gene_name")

    exons, cdss, gene_records = [], [], []
    chrom = strand = None
    label = None

    with open(gtf_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9: continue
            seqname, source, feature, start, end, score, fstrand, frame, attrs = parts
            try:
                start = int(start); end = int(end)
            except:
                continue
            A = parse_gtf_attributes(attrs)
            gid = A.get("gene_id")
            gname = A.get("gene_name") or A.get("Name") or A.get("gene")

            match = False
            if gene_id and gid == gene_id: match = True
            if gene_name and gname == gene_name: match = True
            if not match: continue

            if label is None:
                label = gene_id or gene_name or gid or gname
            if chrom is None: chrom = seqname
            if strand is None: strand = fstrand

            if feature == "gene":
                gene_records.append((start, end))
            elif feature.lower() == "exon":
                exons.append((start, end))
            elif feature.upper() == "CDS":
                cdss.append((start, end))

    if chrom is None:
        die(f"Gene not found in GTF: gene_id={gene_id} gene_name={gene_name}")

    if gene_records:
        gene_start = min(s for s,e in gene_records)
        gene_end   = max(e for s,e in gene_records)
    elif exons:
        gene_start = min(s for s,e in exons)
        gene_end   = max(e for s,e in exons)
    else:
        die("Found gene entry but no coordinates from 'gene' or 'exon' features.")

    exons = sorted(exons)
    cdss  = sorted(cdss)
    return chrom, strand, gene_start, gene_end, exons, cdss, (label or "gene")

def merge_intervals(ivls):
    if not ivls: return []
    ivls = sorted(ivls)
    merged = []
    for s,e in ivls:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(a,b) for a,b in merged]

def subtract_intervals(a_ivls, b_ivls):
    if not a_ivls: return []
    if not b_ivls: return merge_intervals(a_ivls)
    a = merge_intervals(a_ivls)
    b = merge_intervals(b_ivls)
    out = []
    i = j = 0
    while i < len(a):
        s,e = a[i]
        cur_s = s
        while j < len(b) and b[j][1] < cur_s:
            j += 1
        k = j
        while k < len(b) and b[k][0] <= e:
            bs, be = b[k]
            if bs > cur_s:
                out.append((cur_s, min(e, bs-1)))
            cur_s = max(cur_s, be+1)
            if cur_s > e:
                break
            k += 1
        if cur_s <= e:
            out.append((cur_s, e))
        i += 1
    return [(s,e) for s,e in out if s <= e]

def clamp(a, lo, hi): return max(lo, min(hi, a))

def write_fasta(path, header, seq):
    with open(path, "w") as out:
        out.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            out.write(seq[i:i+80] + "\n")

def axis_scale(n_bp):
    if n_bp < 10_000: return 1.0, "bp"
    if n_bp < 10_000_000: return 1e-3, "kb"
    return 1e-6, "Mb"

# ----------------------- BLAST & parsing -----------------------

def make_blastdb(fasta, dbprefix):
    run_cmd(["makeblastdb", "-in", fasta, "-dbtype", "nucl", "-out", dbprefix])

def run_self_blast(fasta, dbprefix, out_tsv):
    cmd = [
        "blastn",
        "-query", fasta,
        "-db", dbprefix,
        "-word_size", "28",
        "-reward", "1",
        "-penalty", "-2",
        "-gapopen", "0",
        "-gapextend", "2",
        "-dust", "no",
        "-soft_masking", "false",
        "-evalue", "0.05",
        "-outfmt", "6 qstart qend sstart send pident length bitscore evalue qseqid sseqid",
        "-out", out_tsv
    ]
    run_cmd(cmd)

def load_hits(tsv, min_len=50, min_ident=0.0, dedup=True):
    hits = []
    with open(tsv) as fh:
        for line in fh:
            if not line.strip(): continue
            p = line.strip().split("\t")
            qstart,qend,sstart,send = map(int, p[0:4])
            pident = float(p[4]); length = int(p[5])
            bitscore = float(p[6]); evalue = float(p[7])
            if length < min_len or pident < min_ident: continue
            if qstart==sstart and qend==send: continue  # trivial self
            hits.append(Hit(qstart,qend,sstart,send,pident,length,bitscore,evalue))
    if dedup:
        seen=set(); uniq=[]
        for h in hits:
            qm=(h.qstart+h.qend)/2.0; sm=(h.sstart+h.send)/2.0
            key=(int(min(qm,sm)), int(max(qm,sm)), h.length, int(round(h.pident)))
            if key in seen: continue
            seen.add(key); uniq.append(h)
        hits=uniq
    return hits

# ----------------------- plotting -----------------------

def plot_dot_with_annotation_and_exon_outlines(hits, seq_len, out_png, title,
                                              exon_rel, cds_rel, exon_hit_index_sets):
    """
    exon_rel, cds_rel: intervals in extracted-sequence coordinates (1-based inclusive).
    exon_hit_index_sets: list of sets of hit indices (per exon) to outline.
    """
    plt.figure(figsize=(8.8,7.4))
    ax = plt.gca()

    scale, unit = axis_scale(seq_len)

    def rel_to_scaled(ivls):
        return [(max(0,(s-1)*scale), min(seq_len*scale, e*scale)) for s,e in ivls]

    cds_spans = rel_to_scaled(cds_rel)
    utr_spans = rel_to_scaled(subtract_intervals(exon_rel, cds_rel))

    # Background bands (both axes)
    for s,e in utr_spans:
        ax.axvspan(s, e, alpha=0.10, linewidth=0)
        ax.axhspan(s, e, alpha=0.10, linewidth=0)
    for s,e in cds_spans:
        ax.axvspan(s, e, alpha=0.22, linewidth=0)
        ax.axhspan(s, e, alpha=0.22, linewidth=0)

    # Main scatter (all hits)
    if hits:
        x = np.array([(h.qstart+h.qend)/2.0 for h in hits], dtype=float) * scale
        y = np.array([(h.sstart+h.send)/2.0 for h in hits], dtype=float) * scale
        c = np.array([h.pident for h in hits], dtype=float)
        sc = ax.scatter(x, y, c=c, s=6, cmap="viridis", marker="s", linewidths=0)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("% identity")

        # Overlay: for each exon, circle its related hits in red (hollow markers)
        for idxset in exon_hit_index_sets:
            if not idxset: continue
            idxs = np.array(sorted(idxset), dtype=int)
            ax.scatter(x[idxs], y[idxs], s=28, marker="o",
                       facecolors="none", edgecolors="red", linewidths=0.6)
    else:
        ax.text(0.5, 0.5, "No BLAST hits after filtering", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(f"Position in extracted region ({unit})")
    ax.set_ylabel(f"Position in extracted region ({unit})")
    ax.set_xlim(0, seq_len*scale)
    ax.set_ylim(0, seq_len*scale)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    legend_elems = [
        Patch(alpha=0.22, label="CDS (both axes)"),
        Patch(alpha=0.10, label="UTR (both axes)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------------------- reporting -----------------------

def write_annotation_report(path, chrom, strand, gene_start, gene_end, flank, region_start, region_end, exons, cdss):
    with open(path, "w") as out:
        out.write("# Gene region and annotation\n")
        out.write(f"chrom: {chrom}\nstrand: {strand}\n")
        out.write(f"gene_start: {gene_start}\ngene_end: {gene_end}\n")
        out.write(f"flank_bp: {flank}\n")
        out.write(f"extracted_region: {region_start}-{region_end} (length {region_end-region_start+1} bp)\n\n")

        def rel(ivls):
            r=[]
            for s,e in ivls:
                rs = s - region_start + 1
                re = e - region_start + 1
                r.append((rs,re))
            return r

        ex_rel = rel(exons)
        cds_rel = rel(cdss)
        utr_rel = subtract_intervals(ex_rel, cds_rel)

        out.write("Exons (genome) -> (relative):\n")
        for (s,e),(rs,re) in zip(exons, ex_rel):
            out.write(f"  {s}-{e} -> {rs}-{re}\n")
        out.write("\nCDS (genome) -> (relative):\n")
        for (s,e),(rs,re) in zip(cdss, cds_rel):
            out.write(f"  {s}-{e} -> {rs}-{re}\n")
        out.write("\nUTR (relative only):\n")
        for (rs,re) in utr_rel:
            out.write(f"  {rs}-{re}\n")

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Extract gene flank via GTF, run self-BLAST, and make a colored dotplot with exon/CDS overlays and exon-specific red circles.")
    ap.add_argument("--genome", required=True, help="Genome FASTA")
    ap.add_argument("--gtf", required=True, help="GTF annotation")
    ap.add_argument("--gene_id", help="Target gene_id in GTF")
    ap.add_argument("--gene_name", help="Target gene_name in GTF")
    ap.add_argument("--flank", type=int, default=1000, help="Flank (bp) to add both sides of the gene")
    ap.add_argument("--outdir", default="gene_selfdot_out", help="Output directory")

    # filters for BLAST hits
    ap.add_argument("--min_len", type=int, default=50, help="Min HSP length to keep")
    ap.add_argument("--min_ident", type=float, default=0.0, help="Min %identity to keep")

    # housekeeping
    ap.add_argument("--keep_tmpdb", action="store_true", help="Keep temporary BLAST DB folder")
    args = ap.parse_args()

    check_exec("makeblastdb"); check_exec("blastn")
    os.makedirs(args.outdir, exist_ok=True)

    # 1) parse GTF
    chrom, strand, gene_start, gene_end, exons, cdss, gene_label = load_gene_features(
        args.gtf, gene_id=args.gene_id, gene_name=args.gene_name
    )
    if not exons: die("No exons found for the specified gene.")
    exons = merge_intervals(exons)
    cdss  = merge_intervals(cdss)

    # 2) Load genome & extract region
    genome = read_fasta_dict(args.genome)
    if chrom not in genome: die(f"Chromosome/contig '{chrom}' not found in genome FASTA.")
    chrom_seq = genome[chrom]
    n = len(chrom_seq)
    region_start = clamp(gene_start - args.flank, 1, n)
    region_end   = clamp(gene_end + args.flank, 1, n)
    subseq = chrom_seq[region_start-1:region_end]
    seg_len = len(subseq)

    # 3) Save extracted region FASTA
    region_fa = os.path.join(args.outdir, "extracted_region.fa")
    region_header = f"{chrom}:{region_start}-{region_end}|strand={strand}|gene={gene_label}"
    write_fasta(region_fa, region_header, subseq)

    # 4) Build temp DB and self BLAST
    tmpdir = tempfile.mkdtemp(prefix="gene_selfdot_db_")
    dbprefix = os.path.join(tmpdir, "db")
    make_blastdb(region_fa, dbprefix)
    blast_tsv = os.path.join(args.outdir, "self_blast.tsv")
    run_self_blast(region_fa, dbprefix, blast_tsv)

    # 5) Parse hits
    hits = load_hits(blast_tsv, min_len=args.min_len, min_ident=args.min_ident, dedup=True)

    # 6) Convert exon/CDS genome coords to relative coords (1..seg_len)
    def to_rel(ivls):
        out=[]
        for s,e in ivls:
            rs = clamp(s - region_start + 1, 1, seg_len)
            re = clamp(e - region_start + 1, 1, seg_len)
            if re >= 1 and rs <= seg_len:
                out.append((max(1,rs), min(seg_len,re)))
        return merge_intervals([(s,e) for s,e in out if s<=e])

    exon_rel = to_rel(exons)
    cds_rel  = to_rel(cdss)

    # 7) Prepare exon-hit outlines for plotting
    qmid = np.array([(h.qstart+h.qend)/2.0 for h in hits]) if hits else np.array([])
    smid = np.array([(h.sstart+h.send)/2.0 for h in hits]) if hits else np.array([])

    def mid_in_interval(mid, iv):
        return (mid >= iv[0]) & (mid <= iv[1])

    exon_hit_index_sets = []
    has_hits = len(hits) > 0
    for ex in exon_rel:
        if has_hits:
            mask = (mid_in_interval(qmid, ex) | mid_in_interval(smid, ex))
            idxs = set(np.nonzero(mask)[0].tolist())
        else:
            idxs = set()
        exon_hit_index_sets.append(idxs)

    # 8) Plot with annotation & exon outlines; title = gene label
    dotplot_png = os.path.join(args.outdir, "dotplot_annotated.png")
    plot_dot_with_annotation_and_exon_outlines(
        hits, seg_len, dotplot_png, title=str(gene_label),
        exon_rel=exon_rel, cds_rel=cds_rel, exon_hit_index_sets=exon_hit_index_sets
    )

    # 9) Annotation report
    anno_report = os.path.join(args.outdir, "annotation_report.txt")
    write_annotation_report(
        anno_report, chrom, strand, gene_start, gene_end, args.flank,
        region_start, region_end, exons, cdss
    )

    # 10) Cleanup
    if args.keep_tmpdb:
        print(f"Temporary BLAST DB kept at: {tmpdir}")
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 11) Final pointers
    print("Done.")
    print(f"- Extracted region FASTA:  {region_fa}")
    print(f"- BLAST results:           {blast_tsv}")
    print(f"- Dot plot (annotated):    {dotplot_png}")
    print(f"- Annotation report:       {anno_report}")

if __name__ == "__main__":
    main()
