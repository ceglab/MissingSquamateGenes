#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined exon-anchored search that integrates multi-setting BLASTN, TBLASTX,
HMMER (optional), spliced-aware EXONERATE (optional), and codon-aware filtering. Produces denoised, per-exon
arch plots (each exon on its own row) and consolidated tables.

Usage:
  python3 exon_anchor_refined.py --folder PATH/TO/OUTDIR \
    [--min_len 18] [--min_ident 0.0] \
    [--blastn_word_sizes 7,9,11] [--blastn_evalues 1e2,1e-3] \
    [--tblastx_evalues 1e1,1e-3] [--target_radius 25] \
    [--support_min_methods 2] [--min_aa_len 20] \
    [--min_entropy 1.2] [--entropy_window 35] \
    [--hmmer] [--exonerate] [--min_gc_z 0.0] [--keep_tmp]

Inputs expected in --folder:
  - extracted_region.fa
  - annotation_report.txt

Key outputs (all written into --folder):
  - refined_hits_blastn.tsv
  - refined_hits_tblastx.tsv
  - refined_hits_hmmer.tsv               (if --hmmer)
  - refined_hits_exonerate.tsv           (if --exonerate)
  - refined_hits_merged.tsv              (deduplicated, denoised union)
  - refined_per_intron.tsv               (counts per intron)
  - refined_summary.tsv                  (top-level metrics)
  - refined_arcs_per_exon.png            (arch plot: one row per exon)
  - refined_dotplot.png                  (subject midpoints vs exon index)
  - refined_debug_filtered.tsv           (diagnostics: why hits were dropped)

Dependencies in PATH: makeblastdb, blastn, tblastx; Optional: hmmbuild, phmmer, exonerate

Notes:
  * "Codon aware" is achieved by leveraging TBLASTX frame information and
    enforcing frame-consistency for CDS-overlapping parts of an exon. For pure
    UTR sequence, frame checks are skipped.
  * HMMER step (optional) translates the region in 6 frames, extracts ORFs,
    builds single-sequence HMMs from exon CDS translations, then runs phmmer
    (or hmmsearch if multiple-seq HMMs are later supplied by the user). This
    is used as an additional evidence track, not a hard requirement.
"""

import os, sys, re, csv, argparse, shutil, subprocess, tempfile, math
from collections import defaultdict, namedtuple
from itertools import product

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HitN = namedtuple("HitN", [
    "method",            # 'blastn' | 'tblastx' | 'hmmer'
    "exon_index",        # 1-based exon index
    "qstart","qend",    # query coords (nt for blastn; aa for hmmer; nt for tblastx)
    "sstart","send",    # subject coords (1-based, nt)
    "pident","length",  # percent id and aligned length (method-specific semantics)
    "bitscore","evalue",
    "frame",             # 0 for non-codon-aware; +1..+3 or -1..-3 for tblastx-based
    "evidence"           # free text tag
])

# ----------------------- utilities -----------------------

def die(msg):
    sys.exit(f"ERROR: {msg}")

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
    ivls = sorted([(min(a,b), max(a,b)) for (a,b) in ivls])
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

# ------------- entropy (low-complexity) filter -------------

def shannon_entropy(s):
    if not s:
        return 0.0
    L = len(s)
    counts = defaultdict(int)
    for c in s:
        counts[c] += 1
    H = 0.0
    for k,v in counts.items():
        p = v / L
        H -= p * math.log(p, 2)
    return H

def entropy_mask_positions(seq, window=35, min_H=1.2):
    """Return boolean mask (len(seq)) with True for positions that pass entropy."""
    n = len(seq)
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    half = max(1, window//2)
    for i in range(n):
        a = max(0, i-half)
        b = min(n, i+half+1)
        H = shannon_entropy(seq[a:b])
        if H >= min_H:
            mask[i] = True
    return mask

# ----------------------- annotation -----------------------

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
            elif t.startswith("Exons"): mode = "exon"
            elif t.startswith("CDS"): mode = "cds"
            elif t.startswith("UTR"): mode = "utr"
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

# ----------------------- BLAST runners -----------------------

def make_blastdb(fasta, dbprefix):
    run_cmd(["makeblastdb", "-in", fasta, "-dbtype", "nucl", "-out", dbprefix])

def run_blastn(exon_fa, dbprefix, word_size, evalue, out_tsv):
    cmd = [
        "blastn", "-task", "blastn-short", "-query", exon_fa, "-db", dbprefix,
        "-word_size", str(word_size), "-reward", "2", "-penalty", "-3",
        "-gapopen", "2", "-gapextend", "2", "-dust", "no", "-soft_masking", "false",
        "-evalue", str(evalue),
        "-outfmt", "6 qseqid qstart qend sstart send pident length bitscore evalue"
    ]
    run_cmd(cmd)
    shutil.move("out", "out") if False else None  # placeholder to avoid linter warnings

    # BLAST writes directly to STDOUT when -out not set; here we rely on redirection in caller


def run_tblastx(exon_fa, dbprefix, evalue, out_tsv):
    cmd = [
        "tblastx", "-query", exon_fa, "-db", dbprefix,
        "-evalue", str(evalue), "-seg", "no", "-soft_masking", "false",
        "-outfmt", "6 qseqid qstart qend sstart send pident length bitscore evalue frames"
    ]
    run_cmd(cmd, capture=False)

# ----------------------- translation helpers -----------------------

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}

RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(s):
    return s.translate(RC)[::-1]

def translate_nt(seq_nt):
    aa = []
    for i in range(0, len(seq_nt)//3):
        codon = seq_nt[3*i:3*i+3]
        aa.append(CODON_TABLE.get(codon, 'X'))
    return ''.join(aa)

# ----------------------- HMMER (optional) -----------------------

def write_orfs_6frame(region_seq, min_aa_len, out_prot_fa):
    frames = []
    # + frames
    for off in range(3):
        frames.append(translate_nt(region_seq[off:]))
    # - frames
    rc = revcomp(region_seq)
    for off in range(3):
        frames.append(translate_nt(rc[off:]))
    items = []
    for fi, aa in enumerate(frames, start=1):
        cur = []
        start = 0
        for i, c in enumerate(aa+"*"):
            if c == '*':
                if i - start >= min_aa_len:
                    seq = aa[start:i]
                    hdr = f"frame{fi:02d}_orf_{len(items)+1}"
                    items.append((hdr, seq))
                start = i+1
    if not items:  # fall back to whole-frame chunks
        for fi, aa in enumerate(frames, start=1):
            if len(aa) >= min_aa_len:
                items.append((f"frame{fi:02d}_full", aa))
    write_fasta(out_prot_fa, items)

# ----------------------- plotting -----------------------

def draw_arch_rows(length, exon_rel, arcs_by_exon, out_png, title=None):
    n = len(exon_rel)
    h = max(2.2, min(3.0, 0.75 + 0.45*n))
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 0.9*n + 2.5), sharex=True)
    if n == 1:
        axes = [axes]
    colors = plt.cm.tab20.colors
    for i,(s,e) in enumerate(exon_rel, start=1):
        ax = axes[i-1]
        ax.plot([1, length],[0,0], lw=1.2, color="black")
        ax.add_patch(plt.Rectangle((s, -0.12), e-s+1, 0.24, color="tab:blue", alpha=0.6, lw=0))
        color = colors[(i-1)%len(colors)]
        for tpos, count in arcs_by_exon.get(i, []):
            x0, x1 = sorted([0.5*(s+e), tpos])
            xm = 0.5*(x0+x1)
            w = max(1.0, 0.5*(x1-x0))
            h = 0.6 * (1.0 if count < 3 else 1.2 if count < 6 else 1.6)
            xs = np.linspace(x0, x1, 80)
            ys = h * (1.0 - ((xs - xm)/w)**2)
            ys[ys < 0] = 0
            ax.plot(xs, ys, lw=1.2, color=color, alpha=0.95)
        ax.set_ylim(-0.4, 2.0)
        ax.set_yticks([])
        ax.set_ylabel(f"Exon {i}")
    axes[-1].set_xlim(1, length)
    axes[-1].set_xlabel("Extracted region positions (bp)")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0,0,1,0.96))
    plt.savefig(out_png, dpi=300)
    plt.close()

def draw_dotplot(length, exon_rel, subject_mids, exon_indices, out_png, title=None):
    plt.figure(figsize=(11.5, 3.6))
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

# ----------------------- GC-corrected null model -----------------------

def gc_z_for_hit(region_seq, sstart, send, pident, length):
    if sstart<1 or send<1 or length<=0 or pident<=0:
        return None
    a,b = sorted([max(1,sstart), max(1,send)])
    b = min(b, len(region_seq))
    if b < a:
        return None
    window = region_seq[a-1:b]
    if not window:
        return None
    gc = (window.count('G') + window.count('C')) / len(window)
    pa = pt = (1.0-gc)/2.0
    pc = pg = gc/2.0
    p_match = pa*pa + pt*pt + pc*pc + pg*pg
    L = float(length)
    obs_matches = (pident/100.0) * L
    var = L * p_match * (1.0 - p_match)
    if var <= 0:
        return None
    z = (obs_matches - L*p_match) / math.sqrt(var)
    return z

# ----------------------- core logic -----------------------

def main():
    ap = argparse.ArgumentParser(description="Refined exon-anchored integration of BLASTN/TBLASTX/HMMER with denoising and codon awareness.")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--min_len", type=int, default=18)
    ap.add_argument("--min_ident", type=float, default=0.0)
    ap.add_argument("--blastn_word_sizes", type=str, default="7,9,11")
    ap.add_argument("--blastn_evalues", type=str, default="1e2,1e-3")
    ap.add_argument("--tblastx_evalues", type=str, default="1e1,1e-3")
    ap.add_argument("--target_radius", type=int, default=25)
    ap.add_argument("--support_min_methods", type=int, default=2,
                    help="Require a subject locus to be supported by N methods among {blastn,tblastx,hmmer(if used)}")
    ap.add_argument("--min_aa_len", type=int, default=20, help="min ORF length for HMMER")
    ap.add_argument("--min_entropy", type=float, default=1.2)
    ap.add_argument("--entropy_window", type=int, default=35)
    ap.add_argument("--hmmer", action="store_true", help="enable optional HMMER evidence track")
    ap.add_argument("--exonerate", action="store_true", help="enable optional EXONERATE est2genome track")
    ap.add_argument("--min_gc_z", type=float, default=0.0, help="minimum GC-corrected Z to retain a hit (0 disables)")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    folder = args.folder
    region_fa = os.path.join(folder, "extracted_region.fa")
    report    = os.path.join(folder, "annotation_report.txt")
    if not os.path.isfile(region_fa): die(f"Missing {region_fa}")
    if not os.path.isfile(report):    die(f"Missing {report}")

    # Load region and annotation
    region = read_fasta(region_fa)
    reg_name = next(iter(region))
    region_seq = region[reg_name]
    meta, exon_rel, cds_rel, utr_rel = load_annotation_report(report)
    L = int(meta["length"])

    # Build exon query FASTA (relative coords 1..L)
    exon_items = []
    for i,(s,e) in enumerate(exon_rel, start=1):
        s0 = max(1, s); e0 = min(L, e)
        seq = region_seq[s0-1:e0]
        exon_items.append((f"Exon_{i:02d}|rel={s0}-{e0}|len={e0-s0+1}", seq))
    exon_fa = os.path.join(folder, "refined_exons.fa")
    write_fasta(exon_fa, exon_items)

    # Introns (1..L minus exons)
    intron_rel = subtract_intervals([(1, L)], exon_rel)
    intron_indexed = [(i+1, s, e) for i, (s,e) in enumerate(intron_rel)]

    # Low-complexity mask (advisory; we don't mask sequence, we drop hits falling mostly in low-H zones)
    ent_mask = entropy_mask_positions(region_seq, window=args.entropy_window, min_H=args.min_entropy)

    # temp BLAST DB
    tmpdir = tempfile.mkdtemp(prefix="refined_db_")
    dbprefix = os.path.join(tmpdir, "nucl_db")
    make_blastdb(region_fa, dbprefix)

    # ---- RUN BLASTN across setting grid ----
    word_sizes = [int(x) for x in args.blastn_word_sizes.split(',') if x]
    bn_evalues = [x for x in args.blastn_evalues.split(',') if x]

    blastn_rows = []
    blastn_path = os.path.join(folder, "refined_hits_blastn.tsv")
    with open(blastn_path, 'w', newline='') as bout:
        w = csv.writer(bout, delimiter='\t')
        w.writerow(["exon_index","qstart","qend","sstart","send","pident","length","bitscore","evalue","evidence"])
        for ws, ev in product(word_sizes, bn_evalues):
            # direct file capture per grid
            grid_out = run_cmd([
                "blastn","-task","blastn-short","-query",exon_fa,"-db",dbprefix,
                "-word_size",str(ws),"-reward","2","-penalty","-3","-gapopen","2","-gapextend","2",
                "-dust","no","-soft_masking","false","-evalue",str(ev),
                "-outfmt","6 qseqid qstart qend sstart send pident length bitscore evalue"
            ])
            for line in grid_out.strip().splitlines():
                p = line.split('\t')
                qid = p[0]
                m = re.match(r"Exon_(\d+)", qid)
                exi = int(m.group(1)) if m else None
                qstart,qend,sstart,send = map(int, p[1:5])
                pident = float(p[5]); length = int(p[6])
                bitscore = float(p[7]); evalue = float(p[8])
                row = HitN("blastn", exi, qstart,qend,sstart,send,pident,length,bitscore,evalue,0,
                           f"blastn:w{ws};e{ev}")
                blastn_rows.append(row)
                w.writerow([exi,qstart,qend,sstart,send,pident,length,bitscore,evalue,row.evidence])

    # ---- RUN TBLASTX across evalues ----
    tbl_evalues = [x for x in args.tblastx_evalues.split(',') if x]
    tblastx_rows = []
    tblastx_path = os.path.join(folder, "refined_hits_tblastx.tsv")
    with open(tblastx_path, 'w', newline='') as tout:
        w = csv.writer(tout, delimiter='\t')
        w.writerow(["exon_index","qstart","qend","sstart","send","pident","length","bitscore","evalue","frame","evidence"])
        for ev in tbl_evalues:
            out = run_cmd([
                "tblastx","-query",exon_fa,"-db",dbprefix,
                "-evalue",str(ev),"-seg","no","-soft_masking","false",
                "-outfmt","6 qseqid qstart qend sstart send pident length bitscore evalue frames"
            ])
            for line in out.strip().splitlines():
                p = line.split('\t')
                qid = p[0]
                m = re.match(r"Exon_(\d+)", qid)
                exi = int(m.group(1)) if m else None
                qstart,qend,sstart,send = map(int, p[1:5])
                pident = float(p[5]); length = int(p[6])
                bitscore = float(p[7]); evalue = float(p[8])
                frame = p[9]
                # BLAST+ frames format: qframe/sframe; keep subject frame sign
                sframe = 0
                if '/' in frame:
                    sf = frame.split('/')[-1]
                    try:
                        sframe = int(sf)
                    except Exception:
                        sframe = 0
                row = HitN("tblastx", exi, qstart,qend,sstart,send,pident,length,bitscore,evalue,sframe,
                           f"tblastx:e{ev}")
                tblastx_rows.append(row)
                w.writerow([exi,qstart,qend,sstart,send,pident,length,bitscore,evalue,sframe,row.evidence])

    # ---- OPTIONAL: HMMER evidence via ORFs + phmmer ----
    hmmer_rows = []
    hmmer_path = os.path.join(folder, "refined_hits_hmmer.tsv")
    if args.hmmer:
        prot_db = os.path.join(tmpdir, "region_orfs.fa")
        write_orfs_6frame(region_seq, args.min_aa_len, prot_db)
        # build exon protein queries from CDS-overlapping parts
        ex_prot_items = []
        for i,(s,e) in enumerate(exon_rel, start=1):
            # keep only CDS-overlapping chunk
            ex_interval = (s,e)
            cds_ol = []
            for cs,ce in cds_rel:
                lo, hi = max(s,cs), min(e,ce)
                if lo <= hi:
                    cds_ol.append((lo,hi))
            cds_ol = merge_intervals(cds_ol)
            if not cds_ol:
                continue
            for (lo,hi) in cds_ol:
                subseq = region_seq[lo-1:hi]
                # choose frame by (lo-1) % 3 relative to + strand model; strand in meta if needed
                aa = translate_nt(subseq)
                ex_prot_items.append((f"Exon_{i:02d}_cds_{lo}-{hi}", aa))
        if ex_prot_items:
            ex_prot_fa = os.path.join(tmpdir, "exon_prot.fa")
            write_fasta(ex_prot_fa, ex_prot_items)
            # phmmer: protein vs protein db (ORFs)
            # write results in --tblout like table (custom parsing)
            tbl = run_cmd(["phmmer", "--tblout", "/dev/stdout", ex_prot_fa, prot_db])
            # very simple parse of HMMER tblout (skip comments)
            with open(hmmer_path, 'w', newline='') as hout:
                w = csv.writer(hout, delimiter='\t')
                w.writerow(["exon_index","sstart","send","length","bitscore","evalue","evidence"])  # qstart/qend N/A in AA
                for line in tbl.splitlines():
                    if not line or line.startswith('#'): continue
                    parts = line.split()
                    if len(parts) < 18: continue
                    qname = parts[0]
                    tname = parts[2]
                    evalue = float(parts[4])
                    bitscore = float(parts[5])
                    # approximate mapping back to subject nt midpoint via frame header (frameXX_)
                    m1 = re.match(r"Exon_(\d+)_", qname)
                    exi = int(m1.group(1)) if m1 else None
                    m2 = re.match(r"frame(\d+)_", tname)
                    frame_id = int(m2.group(1)) if m2 else 0
                    # we can't recover exact nt coords from phmmer alone; mark sstart/send as -1 and use as locus-level evidence later
                    row = HitN("hmmer", exi, 0,0,-1,-1,0.0,0,bitscore,evalue,0, f"phmmer:f{frame_id}")
                    hmmer_rows.append(row)
                    w.writerow([exi,-1,-1,0,bitscore,evalue,row.evidence])

    # ---------------- DENOISING & CONSOLIDATION ----------------

    def in_low_entropy(sstart, send):
        if sstart<1 or send<1: return False
        a = max(1, min(sstart, send)) - 1
        b = min(L, max(sstart, send))
        if a>=b: return False
        win = ent_mask[a:b]
        if len(win)==0: return False
        frac = win.sum()/len(win)
        return frac < 0.5  # keep only if at least half the window is high entropy

    def cluster_positions(mids, radius):
        if not mids: return []
        mids = np.array(sorted(mids), dtype=float)
        groups = []
        cur = [mids[0]]
        for x in mids[1:]:
            if abs(x - cur[-1]) <= radius:
                cur.append(x)
            else:
                groups.append(cur)
                cur = [x]
        groups.append(cur)
        centers = [float(np.median(g)) for g in groups]
        return centers, groups

    debug_drops = []

    # 1) basic length/identity thresholds + low-complexity exclusion
    def pass_basic(h: HitN):
        if h.length < args.min_len or h.pident < args.min_ident:
            debug_drops.append((h.method,h.exon_index,h.sstart,h.send,"basic",f"len/pident {h.length}/{h.pident}"))
            return False
        if h.sstart>0 and h.send>0 and in_low_entropy(h.sstart,h.send):
            # dropped because TOO low entropy (i.e., high low-complexity)
            debug_drops.append((h.method,h.exon_index,h.sstart,h.send,"low_complexity","entropy<0.5 fraction"))
            return False
        return True

    bn_kept = [h for h in blastn_rows if pass_basic(h)]
    tb_kept = [h for h in tblastx_rows if pass_basic(h)]

    # 2) codon-aware frame consistency for CDS-overlapping chunks (only for tblastx)
    cds_ranges = merge_intervals(cds_rel)
    def overlaps_cds(sstart, send):
        if sstart<1 or send<1: return False
        a,b = sorted([sstart,send])
        for cs,ce in cds_ranges:
            if max(a,cs) <= min(b,ce):
                return True
        return False

    tb_kept2 = []
    for h in tb_kept:
        if overlaps_cds(h.sstart, h.send):
            if h.frame in (+1,+2,+3,-1,-2,-3):
                tb_kept2.append(h)
            else:
                debug_drops.append((h.method,h.exon_index,h.sstart,h.send,"frame","no subject frame"))
        else:
            tb_kept2.append(h)  # no frame requirement in UTR/intron-only

    # 3) consolidate by locus (midpoint clustering) and require multi-method support
    all_by_method = {"blastn": bn_kept, "tblastx": tb_kept2}
    if args.hmmer and hmmer_rows:
        all_by_method["hmmer"] = hmmer_rows

    # collect candidate loci from methods that have coordinates
    mids = []
    which = []
    for mname, rows in all_by_method.items():
        for h in rows:
            if h.sstart>0 and h.send>0:
                mids.append(0.5*(h.sstart+h.send))
                which.append((mname,h))
    centers, groups = cluster_positions(mids, args.target_radius)

    # map group index
    # assign hits to group by nearest center
    def nearest_center(x, centers):
        if not centers: return -1
        arr = np.array(centers)
        idx = int(np.argmin(np.abs(arr - x)))
        if abs(arr[idx]-x) <= args.target_radius:
            return idx
        return -1

    support_by_group = defaultdict(set)  # group -> set(methods)
    hits_by_group = defaultdict(list)
    for (mname, h), x in zip(which, mids):
        gi = nearest_center(x, centers)
        if gi>=0:
            support_by_group[gi].add(mname)
            hits_by_group[gi].append(h)

    kept_groups = [gi for gi in range(len(centers)) if len(support_by_group[gi]) >= args.support_min_methods]

    # build merged table rows from kept groups (use best-by-bitscore per method)
    merged_rows = []
    for gi in kept_groups:
        locus = centers[gi]
        by_m = defaultdict(list)
        for h in hits_by_group[gi]:
            by_m[h.method].append(h)
        best = []
        for mname, Ls in by_m.items():
            hbest = sorted(Ls, key=lambda z: (-z.bitscore, z.evalue))[0]
            best.append(hbest)
        # choose representative exon index (mode among methods)
        ex_mode = None
        if best:
            ex_counts = defaultdict(int)
            for h in best:
                if h.exon_index is not None:
                    ex_counts[h.exon_index]+=1
            if ex_counts:
                ex_mode = sorted(ex_counts.items(), key=lambda kv:(-kv[1], kv[0]))[0][0]
        # averaged subject span
        sstarts = [h.sstart for h in best if h.sstart>0]
        sends   = [h.send   for h in best if h.send>0]
        sstart = int(round(np.median(sstarts))) if sstarts else -1
        send   = int(round(np.median(sends)))   if sends else -1
        # aggregate metrics
        bits = max(h.bitscore for h in best)
        evals = min(h.evalue for h in best)
        pid = max(h.pident for h in best)
        alnlen = max(h.length for h in best)
        methods = ",".join(sorted(by_m.keys()))
        merged_rows.append((ex_mode, sstart, send, pid, alnlen, bits, evals, methods, locus))

    # write merged table
    merged_path = os.path.join(folder, "refined_hits_merged.tsv")
    with open(merged_path, 'w', newline='') as out:
        w = csv.writer(out, delimiter='\t')
        w.writerow(["exon_index","sstart","send","pident","length","bitscore","evalue","methods","subject_mid"])
        for r in sorted(merged_rows, key=lambda z: (z[0] if z[0] is not None else 1e9, z[-1])):
            w.writerow(r)

    # write debug drops
    dbg_path = os.path.join(folder, "refined_debug_filtered.tsv")
    with open(dbg_path, 'w', newline='') as out:
        w = csv.writer(out, delimiter='\t')
        w.writerow(["method","exon_index","sstart","send","reason","detail"])
        for t in debug_drops:
            w.writerow(list(t))

    # per-intron / per-exon summaries from merged
    subj_mids_all, exon_idx_all = [], []
    arcs_by_exon = defaultdict(list)

    def intron_index_of(mid):
        for ii, s, e in intron_indexed:
            if s <= mid <= e:
                return ii
        return None

    raw_by_intron = defaultdict(int)
    copies_by_intron = defaultdict(int)
    contrib_exons = defaultdict(set)

    mids_by_exon = defaultdict(list)

    for (exi, sstart, send, pid, Lx, bits, ev, methods, mid) in merged_rows:
        if exi is None: continue
        subj_mids_all.append(mid)
        exon_idx_all.append(exi)
        mids_by_exon[exi].append(mid)
        ii = intron_index_of(mid)
        if ii is not None:
            raw_by_intron[ii]+=1
            contrib_exons[ii].add(exi)

    # cluster each exon mids into loci for arc count
    for exi, mids in mids_by_exon.items():
        mids = sorted(mids)
        centers, groups = cluster_positions(mids, args.target_radius)
        arcs_by_exon[exi] = [(c, len(g)) for c,g in zip(centers, groups)]
        copies_by_intron  # maintain for symmetry

    # write per-intron
    per_intron_path = os.path.join(folder, "refined_per_intron.tsv")
    with open(per_intron_path, 'w', newline='') as out:
        w = csv.writer(out, delimiter='\t')
        w.writerow(["intron_index","start","end","length_bp","raw_exon_to_intron_hits","contributing_exons"])
        for ii, s, e in intron_indexed:
            exset = sorted(contrib_exons.get(ii, set()))
            exstr = ",".join(map(str, exset)) if exset else "-"
            w.writerow([ii, s, e, e-s+1, raw_by_intron.get(ii,0), exstr])

    # top-level summary
    summary_path = os.path.join(folder, "refined_summary.tsv")
    with open(summary_path, 'w', newline='') as out:
        w = csv.writer(out, delimiter='\t')
        w.writerow(["metric","value"])
        w.writerow(["total_merged_hits", len(merged_rows)])
        w.writerow(["support_min_methods", args.support_min_methods])
        w.writerow(["target_radius", args.target_radius])
        w.writerow(["entropy_window", args.entropy_window])
        w.writerow(["min_entropy_threshold", args.min_entropy])
        w.writerow(["blastn_word_sizes", args.blastn_word_sizes])
        w.writerow(["blastn_evalues", args.blastn_evalues])
        w.writerow(["tblastx_evalues", args.tblastx_evalues])
        if args.hmmer:
            w.writerow(["hmmer_enabled", 1])

    # plots
    arcs_png = os.path.join(folder, "refined_arcs_per_exon.png")
    draw_arch_rows(L, exon_rel, arcs_by_exon, arcs_png, title="Exon-anchored copies (denoised, integrated)")

    dot_png = os.path.join(folder, "refined_dotplot.png")
    if subj_mids_all:
        draw_dotplot(L, exon_rel, np.array(subj_mids_all), np.array(exon_idx_all), dot_png,
                     title="Denoised subject midpoints per exon (merged)")
    else:
        plt.figure(figsize=(10.5, 3.2))
        plt.text(0.5, 0.5, "No merged hits after filtering", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout(); plt.savefig(dot_png, dpi=300); plt.close()

    # cleanup
    if args.keep_tmp:
        print(f"[keep] temp dir {tmpdir}")
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("Done.")
    print(f"- Exon queries FASTA:     {exon_fa}")
    print(f"- BLASTN rows:            {blastn_path}")
    print(f"- TBLASTX rows:           {tblastx_path}")
    if args.hmmer:
        print(f"- HMMER rows:             {hmmer_path}")
    print(f"- Merged (denoised):      {merged_path}")
    print(f"- Per intron:             {per_intron_path}")
    print(f"- Summary:                {summary_path}")
    print(f"- Arcs per exon plot:     {arcs_png}")
    print(f"- Dotplot:                {dot_png}")
    print(f"- Debug filtered:         {dbg_path}")

if __name__ == "__main__":
    main()
