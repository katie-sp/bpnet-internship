set -beEuo pipefail

indir=/home/katie/bp_repo/research/data/chip-seq
outdir=/home/katie/bp_repo/research/data/chip-seq

chromsizes=/home/katie/bp_repo/research/data/hg38.chrom.sizes

shortcutoff=120
longcutoff=150

tempdir=$outdir/temp
mkdir -p $tempdir

stems=(CTCF FOSL2)

echo "Merging replicates..."
for stem in ${stems[*]}
do
	samtools merge $tempdir/$stem\_merged.bam $indir/$stem/*$stem*.bam
done


echo "Filtering for quality and splitting by fragment length..."

# cutnrun
stems=(CTCF FOSL2)
for stem in ${stems[*]}
do
	samtools view -h -F 780 -q 30 $tempdir/$stem\_merged.bam | awk -F "\t" -v maxi=$shortcutoff 'function abs(x){return (x < 0) ? -x : x} $1~/^@/ || abs($9) <= maxi' | samtools view -b - -o $tempdir/$stem\_merged_maxfl$shortcutoff.bam
	samtools view -h -F 780 -q 30 $tempdir/$stem\_merged.bam | awk -F "\t" -v mini=$longcutoff 'function abs(x){return (x < 0) ? -x : x} $1~/^@/ || abs($9) >= mini' | samtools view -b - -o $tempdir/$stem\_merged_minfl$longcutoff.bam
done

# chip-seq
stems=(CTCF FOSL2)
for stem in ${stems[*]}
do
	samtools view -h -F 780 -q 30 $tempdir/$stem\_merged.bam -b - -o $tempdir/$stem\_merged_filtered.bam
done

# for cutnrun, can save fragment lengths to draw histogram after filtering but before splitting by fragment length
for stem in ${stems[*]}
do
samtools view -h -F 780 -q 30 $tempdir/$stem\_merged.bam -b - -o $tempdir/$stem\_merged_filtered.bam # usual filtering
samtools view -h $tempdir/$stem\_merged_filtered.bam | awk -F "\t" '$1~/^@/{print abs($9)}' > $outdir/$temp\_frag_lengths.txt
done

echo "Converting to BedGraphs..."
bam_to_stranded_bedgraph() {
	local infile="$1"
	local stem=$(basename "$infile" .bam)
	bedtools genomecov -5 -bg -strand - -ibam "$infile" | sort -k1,1 -k2,2n > $tempdir/$stem\_neg.bg
	bedtools genomecov -5 -bg -strand + -ibam "$infile" | sort -k1,1 -k2,2n > $tempdir/$stem\_pos.bg
}

# CUTNRUN
for stem in ${stems[*]}
do
	bam_to_stranded_bedgraph $tempdir/$stem\_merged_maxfl$shortcutoff.bam
	bam_to_stranded_bedgraph $tempdir/$stem\_merged_minfl$longcutoff.bam
done

#CHIP-SEQ
for stem in ${stems[*]}
do
	bam_to_stranded_bedgraph $tempdir/$stem\_merged_filtered.bam
done

# and then after all this, you need to use bedGraphToBigWig!