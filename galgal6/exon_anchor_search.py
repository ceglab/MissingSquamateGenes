#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exon-anchored search pass for self-dot analysis with per-intron breakdown.

Usage:
  python3 exon_anchor_search.py --folder PATH/TO/OUTDIR \
    [--word_size 7] [--evalue 100] [--min_len 18] [--min_ident 0.0] \
    [--min_exon_overlap 12] [--min_exon_overlap_frac 0.30] \
    [--target_radius 25] [--repeat_degree 5] [--keep_tmpdb]

Inputs in --folder:
  - extracted_region.fa
  - annotation_report.txt

Outputs in --folder:
  - exon_anchor_exons.fa
  - exon_anchor_blast.tsv
  - exon_anchor_arcs_data.tsv
  - exon_anchor_summary.tsv
  - exon_anchor_per_intron.tsv        <-- NEW
  - exon_anchor_arcs.png
  - exon_anchor_dotplot.png
"""

import os, sys, re, csv, argparse, shutil, subprocess, tempfile
from collections import defaultdict, namedtuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Hit = namedtuple("Hit", ["qstart","qend","sstart","send","pident","length","bitscore","evalue","qseqid","sseqid"])

# ----------------------- utilities -----------------------

def die(msg): sys.exit(f"ERROR: {msg}")

def run_cmd(cmd, capture=True, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, check=True,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.PIPE if capture else None,
                           text=True)
        return (r.stdout or "")
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        sys.stderr.write(out)
        die(f"Command failed: {' '.join(cmd)}")

def read_fasta(path):
    seqs = {}
    name = None
    buf = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf).upper()
    if not seqs:
        die(f"No sequences found in FASTA: {path}")
    return seqs

def write_fasta(path, items):
    with open(path, "w") as out:
        for hdr, seq in items:
            out.write(f">{hdr}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

def merge_intervals(ivls, gap=0):
    if not ivls: return []
    ivls = sorted(ivls)
    out = []
    cs, ce = ivls[0]
    for s,e in ivls[1:]:
        if s <= ce + 1 + gap:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out

def subtract_intervals(a_ivls, b_ivls):
    if not a_ivls: return []
    if not b_ivls: return merge_intervals(a_ivls)
    A = merge_intervals(a_ivls)
    B = merge_intervals(b_ivls)
    out = []
    j = 0
    for s,e in A:
        cur = s
        while j < len(B) and B[j][1] < cur:
            j += 1
        k = j
        while k < len(B) and B[k][0] <= e:
            bs, be = B[k]
            if bs > cur:
                out.append((cur, min(e, bs-1)))
            cur = max(cur, be+1)
            if cur > e:
                break
            k += 1
        if cur <= e:
            out.append((cur, e))
    return out

def load_annotation_report(path):
    meta = {}
    exon_rel, cds_rel, utr_rel = [], [], []
    mode = None
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if not t: continue
            if t.startswith("chrom: "): meta["chrom"] = t.split(": ",1)[1]
            elif t.startswith("strand: "): meta["strand"] = t.split(": ",1)[1]
            elif t.startswith("gene_start: "): meta["gene_start"] = int(t.split(": ",1)[1])
            elif t.startswith("gene_end: "): meta["gene_end"] = int(t.split(": ",1)[1])
            elif t.startswith("flank_bp: "): meta["flank"] = int(t.split(": ",1)[1])
            elif t.startswith("extracted_region: "):
                m = re.search(r"(\d+)-(\d+).*length\s+(\d+)\s*bp", t)
                if m:
                    meta["region_start"] = int(m.group(1))
                    meta["region_end"] = int(m.group(2))
                    meta["length"] = int(m.group(3))
            elif t.startswith("Exons"):
                mode = "exon"
            elif t.startswith("CDS"):
                mode = "cds"
            elif t.startswith("UTR"):
                mode = "utr"
            elif re.match(r"^\d", t):
                m = re.search(r"->\s*(\d+)-(\d+)$", t)
                if m and mode:
                    rs, re_ = int(m.group(1)), int(m.group(2))
                    if mode == "exon": exon_rel.append((rs, re_))
                    elif mode == "cds": cds_rel.append((rs, re_))
                    elif mode == "utr": utr_rel.append((rs, re_))
    if "length" not in meta:
        die("annotation_report.txt missing extracted region length")
    return meta, merge_intervals(exon_rel), merge_intervals(cds_rel), merge_intervals(utr_rel)

def make_blastdb(fasta, dbprefix):
    run_cmd(["makeblastdb", "-in", fasta, "-dbtype", "nucl", "-out", dbprefix])

def run_blast_exon_queries(exon_fa, dbprefix, out_tsv, task="blastn-short",
                           word_size=7, evalue=100.0):
    cmd = [
        "blastn",
        "-task", task,
        "-query", exon_fa,
        "-db", dbprefix,
        "-word_size", str(word_size),
        "-reward", "2",
        "-penalty", "-3",
        "-gapopen", "2",
        "-gapextend", "2",
        "-dust", "no",
        "-soft_masking", "false",
        "-evalue", str(evalue),
        "-outfmt", "6 qstart qend sstart send pident length bitscore evalue qseqid sseqid",
        "-out", out_tsv
    ]
    run_cmd(cmd)

def load_hits(tsv, min_len=18, min_ident=0.0):
    hits = []
    if not os.path.isfile(tsv):
        return hits
    with open(tsv) as fh:
        for line in fh:
            if not line.strip(): continue
            p = line.rstrip("\n").split("\t")
            qstart,qend,sstart,send = map(int, p[0:4])
            pident = float(p[4]); length = int(p[5])
            bitscore = float(p[6]); evalue = float(p[7])
            qseqid = p[8]; sseqid = p[9]
            if length < min_len or pident < min_ident: continue
            hits.append(Hit(qstart,qend,sstart,send,pident,length,bitscore,evalue,qseqid,sseqid))
    return hits

def overlap_len(iv, ivls):
    s,e = iv; tot = 0
    for a,b in ivls:
        lo = max(s,a); hi = min(e,b)
        if lo <= hi: tot += (hi-lo+1)
    return tot

def side_label(interval, exon_rel, cds_rel, utr_rel, min_bp, min_frac):
    s,e = interval
    L = max(1, e - s + 1)
    cds_ol = overlap_len(interval, cds_rel)
    exon_ol = overlap_len(interval, exon_rel)  # includes CDS
    utr_ol  = overlap_len(interval, utr_rel)
    if cds_ol >= min_bp or (cds_ol / L >= min_frac): return "CDS"
    if exon_ol >= min_bp or (exon_ol / L >= min_frac): return "exon"
    if utr_ol  >= min_bp or (utr_ol  / L >= min_frac): return "UTR"
    return "intron"

def cluster_targets(points, radius=25):
    if points is None or len(points) == 0:
        return []
    order = np.argsort(points)
    centers = []
    cur = [order[0]]
    for idx in order[1:]:
        if abs(points[idx] - points[cur[-1]]) <= radius:
            cur.append(idx)
        else:
            centers.append(cur)
            cur = [idx]
    centers.append(cur)
    out = []
    for grp in centers:
        c = float(np.median([points[i] for i in grp]))
        out.append((c, grp))
    return out

# ----------------------- plotting -----------------------

def draw_arcs(length, exon_rel, arcs_by_exon, out_png, title=None):
    plt.figure(figsize=(10.5, 3.6))
    ax = plt.gca()
    ax.plot([1, length], [0, 0], lw=2, color="black")
    for s,e in exon_rel:
        ax.add_patch(plt.Rectangle((s, -0.25), e-s+1, 0.5, color="tab:blue", alpha=0.6, lw=0))
    colors = plt.cm.tab20.colors
    for exi, items in arcs_by_exon.items():
        color = colors[exi % len(colors)]
        emid = 0.5*(exon_rel[exi-1][0] + exon_rel[exi-1][1])  # exi is 1-based below
        for tpos, count in items:
            x0, x1 = sorted([emid, tpos])
            xm = 0.5*(x0+x1)
            w = max(1.0, 0.5*(x1-x0))
            h = 0.6 * (1.0 if count < 3 else 1.2 if count < 6 else 1.6)
            xs = np.linspace(x0, x1, 80)
            ys = h * (1.0 - ((xs - xm)/w)**2)
            ys[ys < 0] = 0
            ax.plot(xs, ys, lw=1.5, color=color, alpha=0.95)
    ax.set_ylim(-0.6, 2.0)
    ax.set_xlim(1, length)
    ax.set_yticks([])
    ax.set_xlabel("Extracted region positions (bp)")
    if title: ax.set_title(title, fontsize=11)
    ex_proxy = plt.Line2D([0],[0], color="tab:blue", lw=6)
    arc_proxy = plt.Line2D([0],[0], color="k", lw=1.5)
    ax.legend([ex_proxy, arc_proxy], ["Exons", "Exon?intron copies"], loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def draw_exon_anchor_dotplot(length, exon_rel, subject_mids, exon_indices, out_png, title=None):
    plt.figure(figsize=(10.5, 3.6))
    ax = plt.gca()
    for i,(s,e) in enumerate(exon_rel, start=1):
        ax.axvspan(s, e, alpha=0.12, lw=0, color="tab:blue")
    if len(subject_mids) > 0:
        ax.scatter(subject_mids, exon_indices, s=10, alpha=0.9)
    ax.set_xlim(1, length)
    ax.set_ylim(0.5, len(exon_rel)+0.5)
    ax.set_yticks(range(1, len(exon_rel)+1))
    ax.set_xlabel("Subject position (bp)")
    ax.set_ylabel("Exon index (query)")
    if title: ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------------------- main logic -----------------------

def main():
    ap = argparse.ArgumentParser(description="Exon-anchored BLAST: query = exons, subject = extracted region (with per-intron breakdown).")
    ap.add_argument("--folder", required=True, help="Folder from previous pipeline (contains extracted_region.fa, annotation_report.txt)")
    # BLAST sensitivity
    ap.add_argument("--word_size", type=int, default=7)
    ap.add_argument("--evalue", type=float, default=100.0)
    ap.add_argument("--min_len", type=int, default=18)
    ap.add_argument("--min_ident", type=float, default=0.0)
    # labeling thresholds
    ap.add_argument("--min_exon_overlap", type=int, default=12)
    ap.add_argument("--min_exon_overlap_frac", type=float, default=0.30)
    # clustering for copies
    ap.add_argument("--target_radius", type=int, default=25)
    ap.add_argument("--repeat_degree", type=int, default=5)
    ap.add_argument("--keep_tmpdb", action="store_true")
    args = ap.parse_args()

    folder = args.folder
    region_fa = os.path.join(folder, "extracted_region.fa")
    report    = os.path.join(folder, "annotation_report.txt")
    if not os.path.isfile(region_fa): die(f"Missing {region_fa}")
    if not os.path.isfile(report):    die(f"Missing {report}")

    # Load region and annotation
    region = read_fasta(region_fa)
    key = next(iter(region))
    region_seq = region[key]
    meta, exon_rel, cds_rel, utr_rel = load_annotation_report(report)
    L = int(meta["length"])

    # Build exon query FASTA
    ex_items = []
    for i,(s,e) in enumerate(exon_rel, start=1):
        s0 = max(1, s); e0 = min(L, e)
        seq = region_seq[s0-1:e0]
        ex_items.append((f"Exon_{i:02d}|rel={s0}-{e0}|len={e0-s0+1}", seq))
    exon_fa = os.path.join(folder, "exon_anchor_exons.fa")
    write_fasta(exon_fa, ex_items)

    # Build intron intervals (relative: 1..L)
    intron_rel = subtract_intervals([(1, L)], exon_rel)  # 1-based inclusive
    # index introns 1..M
    intron_indexed = [(i+1, s, e) for i, (s,e) in enumerate(intron_rel)]

    # BLAST database (temporary)
    tmpdir = tempfile.mkdtemp(prefix="exon_anchor_db_")
    dbprefix = os.path.join(tmpdir, "db")
    make_blastdb(region_fa, dbprefix)

    # Run BLAST: exon queries vs region
    out_tsv = os.path.join(folder, "exon_anchor_blast.tsv")
    run_blast_exon_queries(exon_fa, dbprefix, out_tsv,
                           task="blastn-short",
                           word_size=args.word_size,
                           evalue=args.evalue)

    # Load and label hits
    hits = load_hits(out_tsv, min_len=args.min_len, min_ident=args.min_ident)

    def exon_index_from_q(qid):
        m = re.match(r"Exon_(\d+)", qid)
        return int(m.group(1)) if m else None

    rows = []
    for h in hits:
        ex_idx = exon_index_from_q(h.qseqid)
        qa = "exon"  # query is exon by construction
        qb = side_label((h.sstart, h.send), exon_rel, cds_rel, utr_rel,
                        args.min_exon_overlap, args.min_exon_overlap_frac)
        rows.append({
            "exon_index": ex_idx,
            "qstart": h.qstart, "qend": h.qend,
            "sstart": h.sstart, "send": h.send,
            "pident": h.pident, "length": h.length,
            "qlabel": qa, "slabel": qb
        })

    # ----- summaries and clustering -----

    # Per-exon clustered intronic copies
    arcs_by_exon = defaultdict(list)         # exi -> [(tpos_center, count)]
    per_exon_copy_counts = defaultdict(int)  # exi -> clusters count
    cats = defaultdict(int)

    subj_mids_all = []
    exon_idx_all  = []

    for r in rows:
        cats[f"{r['qlabel']}->{r['slabel']}"] += 1
        if r["exon_index"] is not None:
            subj_mids_all.append(0.5*(r["sstart"] + r["send"]))
            exon_idx_all.append(r["exon_index"])

    # cluster exon->intron midpoints per exon
    ex_to_intr = [r for r in rows if r["slabel"] == "intron" and r["exon_index"] is not None]
    by_exon = defaultdict(list)
    for r in ex_to_intr:
        by_exon[r["exon_index"]].append(0.5*(r["sstart"] + r["send"]))
    for exi, mids in by_exon.items():
        mids = np.array(sorted(mids), dtype=float)
        clusters = cluster_targets(mids, radius=args.target_radius)
        arcs_by_exon[exi] = [(c, len(idxs)) for (c, idxs) in clusters]
        per_exon_copy_counts[exi] = len(clusters)

    # ----- per-intron breakdown -----

    def intron_index_of(mid):
        for ii, s, e in intron_indexed:
            if s <= mid <= e:
                return ii
        return None

    # raw counts and clustered counts per intron
    intron_raw_hits = defaultdict(int)            # ii -> raw exon->intron hits
    intron_contrib_exons = defaultdict(set)       # ii -> set(exon_idx)
    intron_midpoints = defaultdict(list)          # ii -> list of subject mids

    for r in ex_to_intr:
        mid = 0.5*(r["sstart"] + r["send"])
        ii = intron_index_of(mid)
        if ii is None:
            continue
        intron_raw_hits[ii] += 1
        if r["exon_index"] is not None:
            intron_contrib_exons[ii].add(r["exon_index"])
        intron_midpoints[ii].append(mid)

    intron_copy_counts = {}  # clustered distinct loci within each intron
    for ii, mids in intron_midpoints.items():
        mids = np.array(sorted(mids), dtype=float)
        clusters = cluster_targets(mids, radius=args.target_radius)
        intron_copy_counts[ii] = len(clusters)

    # ----- write per-hit arcs data -----

    arcs_data_path = os.path.join(folder, "exon_anchor_arcs_data.tsv")
    with open(arcs_data_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["exon_index","qstart","qend","sstart","send","pident","length","qlabel","slabel"])
        for r in rows:
            w.writerow([r["exon_index"], r["qstart"], r["qend"], r["sstart"], r["send"],
                        r["pident"], r["length"], r["qlabel"], r["slabel"]])

    # ----- write per-intron table (NEW) -----

    per_intron_path = os.path.join(folder, "exon_anchor_per_intron.tsv")
    with open(per_intron_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["intron_index","start","end","length_bp","raw_exon_to_intron_hits",
                    "clustered_copies","contributing_exons"])
        for ii, s, e in intron_indexed:
            raw = intron_raw_hits.get(ii, 0)
            copies = intron_copy_counts.get(ii, 0)
            exset = sorted(intron_contrib_exons.get(ii, set()))
            exstr = ",".join(str(x) for x in exset) if exset else "-"
            w.writerow([ii, s, e, e - s + 1, raw, copies, exstr])

    # ----- write top-level summary -----

    summary_path = os.path.join(folder, "exon_anchor_summary.tsv")
    with open(summary_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["metric","value"])
        # class totals
        for k in sorted(cats.keys()):
            w.writerow([f"hits:{k}", cats[k]])
        # per-exon copies
        total_copies = sum(per_exon_copy_counts.values())
        w.writerow(["copies:exon_to_intron_total", total_copies])
        for i,(s,e) in enumerate(exon_rel, start=1):
            w.writerow([f"copies:exon_{i:02d}_({s}-{e})", per_exon_copy_counts.get(i,0)])
        # per-intron summary lines
        total_intron_raw = sum(intron_raw_hits.values())
        total_intron_copies = sum(intron_copy_counts.values())
        w.writerow(["intron:total_raw_exon_to_intron_hits", total_intron_raw])
        w.writerow(["intron:total_clustered_copies", total_intron_copies])
        for ii, s, e in intron_indexed:
            w.writerow([f"intron_{ii:02d}_({s}-{e}):raw_hits", intron_raw_hits.get(ii,0)])
            w.writerow([f"intron_{ii:02d}_({s}-{e}):clustered_copies", intron_copy_counts.get(ii,0)])

    # ----- plots -----

    arcs_png = os.path.join(folder, "exon_anchor_arcs.png")
    # note: arcs_by_exon uses 1-based exon indices in this script’s loop
    # convert to 1-based keys for draw_arcs:
    arcs_by_exon_1based = {int(exi): v for exi, v in arcs_by_exon.items()}
    draw_arcs(L, exon_rel, arcs_by_exon_1based, arcs_png, title="Exon-anchored copies")

    dot_png = os.path.join(folder, "exon_anchor_dotplot.png")
    if len(subj_mids_all) > 0:
        draw_exon_anchor_dotplot(L, exon_rel,
                                 np.array(subj_mids_all, dtype=float),
                                 np.array(exon_idx_all, dtype=int),
                                 dot_png,
                                 title="Exon-anchored hits (subject midpoints)")
    else:
        plt.figure(figsize=(10.5, 3.6))
        plt.text(0.5, 0.5, "No exon-anchored hits after filtering",
                 ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(dot_png, dpi=300)
        plt.close()

    # Cleanup DB
    if args.keep_tmpdb:
        print(f"[keep] temp BLAST DB at {tmpdir}")
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Final pointers
    print("Done.")
    print(f"- Exon queries FASTA:     {exon_fa}")
    print(f"- BLAST hits (exon->reg): {out_tsv}")
    print(f"- Arcs data:              {arcs_data_path}")
    print(f"- Summary:                {summary_path}")
    print(f"- Per-intron:             {per_intron_path}")
    print(f"- Arcs plot:              {arcs_png}")
    print(f"- Dot plot:               {dot_png}")

if __name__ == "__main__":
    main()
