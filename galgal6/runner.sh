for g in ADAP2 ARL11 CAV3 CYYR1 ELOVL3 GPR78 GPR82 GRK4 HHLA2 HPGDS HSD17B1 C5H11orf74 IL13RA2 IL26 IL34 IL5RA INTS6 KDM3A LAPTM5 MATK MOB1A OLAH HRASLS RBBP7 RIPPLY3 SH2D2A SLC24A1 SLC9A5 SSTR4 STAP1 SUV39H2 TBC1D14 TMEM273 UTS2B WNT2 ZNF438 ADRA1B DIP2A DLEU7 DNM3 KREMEN1 NOTCH2 SH2D1A UNKL FREM2 LCP1 LRCH1 MAN2B2 RUBCNL SIAH3 STOML3 TNIP2 YBX3
do
echo $g
python3 ../self_dotplot_from_gtf_dotplot_only.py \
  --genome GCF_000002315.5_GRCg6a_genomic.fna \
  --gtf GCF_000002315.5_GRCg6a_genomic.gtf \
  --gene_id $g \
  --flank 2000 \
  --outdir RefSeq_"$g"_selfdot \
  --min_len 80 \
  --min_ident 70
done

IFTAP
IL26
PLAAT1

for g in ADAP2 ARL11 CAV3 CYYR1 ELOVL3 GPR78 GPR82 GRK4 HHLA2 HPGDS HSD17B1 C5H11orf74 IL13RA2 IL26 IL34 IL5RA INTS6 KDM3A LAPTM5 MATK MOB1A OLAH HRASLS RBBP7 RIPPLY3 SH2D2A SLC24A1 SLC9A5 SSTR4 STAP1 SUV39H2 TBC1D14 TMEM273 UTS2B WNT2 ZNF438 ADRA1B DIP2A DLEU7 DNM3 KREMEN1 NOTCH2 SH2D1A UNKL FREM2 LCP1 LRCH1 MAN2B2 RUBCNL SIAH3 STOML3 TNIP2 YBX3
do
echo $g
python3 ../exon_anchor_search.py --folder RefSeq_"$g"_selfdot \
  --word_size 7 --evalue 100 --min_len 18 \
  --min_exon_overlap 12 --min_exon_overlap_frac 0.30 \
  --target_radius 25
done


grep "exon_to_intron_total" *_selfdot/exon_anchor_summary.tsv

See attached script. It finds that are too noisy. I want you to make a more refined version of the script that integrates the hits from blastn with different settings, tblastx, hmmer search and be codon aware. The script should provide refined results in the same folder using the folder as input argument. Specifically, i want the arch plot to be shown for each exon in a separate row, ensure the hits are denoised. Suggest other methods to clean up the signal and provide a clean result.


python3 exon_anchor_refined.py \
  --folder RefSeq_ADAP2_selfdot \
  --blastn_word_sizes 7,9,11 \
  --blastn_evalues 1e2,1e-3 \
  --tblastx_evalues 1e1,1e-3 \
  --target_radius 25 \
  --support_min_methods 2 \
  --min_entropy 1.2 --entropy_window 35 \
  --hmmer


for g in ADAP2 ARL11 CAV3 CYYR1 ELOVL3 GPR78 GPR82 GRK4 HHLA2 HPGDS HSD17B1 C5H11orf74 IL13RA2 IL26 IL34 IL5RA INTS6 KDM3A LAPTM5 MATK MOB1A OLAH HRASLS RBBP7 RIPPLY3 SH2D2A SLC24A1 SLC9A5 SSTR4 STAP1 SUV39H2 TBC1D14 TMEM273 UTS2B WNT2 ZNF438 ADRA1B DIP2A DLEU7 DNM3 KREMEN1 NOTCH2 SH2D1A UNKL FREM2 LCP1 LRCH1 MAN2B2 RUBCNL SIAH3 STOML3 TNIP2 YBX3
do
echo $g
python3 exon_anchor_refined.py \
  --folder RefSeq_"$g"_selfdot \
  --blastn_word_sizes 7,9,11 \
  --blastn_evalues 1e2,1e-3 \
  --tblastx_evalues 1e1,1e-3 \
  --target_radius 25 \
  --support_min_methods 2 \
  --min_entropy 1.2 --entropy_window 35 \
  --min_gc_z 2.0 \
  --exonerate \
  --hmmer
done


for g in ADAP2 ARL11 CAV3 CYYR1 ELOVL3 GPR78 GPR82 GRK4 HHLA2 HPGDS HSD17B1 C5H11orf74 IL13RA2 IL26 IL34 IL5RA INTS6 KDM3A LAPTM5 MATK MOB1A OLAH HRASLS RBBP7 RIPPLY3 SH2D2A SLC24A1 SLC9A5 SSTR4 STAP1 SUV39H2 TBC1D14 TMEM273 UTS2B WNT2 ZNF438 ADRA1B DIP2A DLEU7 DNM3 KREMEN1 NOTCH2 SH2D1A UNKL FREM2 LCP1 LRCH1 MAN2B2 RUBCNL SIAH3 STOML3 TNIP2 YBX3
do
echo $g
cp RefSeq_"$g"_selfdot/refined_arcs_per_exon.png arcs_plot/"$g"_refined_arcs_per_exon.png
done
