set -beEuo pipefail

indir=/users/amtseng/gata1/data/raw/cutnrun/
outdir=/users/amtseng/gata1/data/interim/cutnrun/

chromsizes=/users/amtseng/genomes/hg38.chrom.sizes

shortcutoff=120
longcutoff=150

tempdir=$outdir/temp
mkdir -p $tempdir

stems=(gata1_patient-day6 gata1_patient-day12 gata1_wildtype-day6 gata1_wildtype-day12)

# 1) Merge BAMs
echo "Merging replicates..."
for stem in ${stems[*]}
do
	samtools merge $tempdir/$stem\_merged.bam $indir/$stem\_*.bam
done

# 2) Filter BAMs for <= $shortcutoff and >= $longcutoff fragments, and for quality/mappability
echo "Filtering for quality and splitting by fragment length..."
for stem in ${stems[*]}
do
	samtools view -h -F 780 -q 30 $tempdir/$stem\_merged.bam | awk -F "\t" -v maxi=$shortcutoff 'function abs(x){return (x < 0) ? -x : x} $1~/^@/ || abs($9) <= maxi' | samtools view -b - -o $tempdir/$stem\_merged_maxfl$shortcutoff.bam
	samtools view -h -F 780 -q 30 $tempdir/$stem\_merged.bam | awk -F "\t" -v mini=$longcutoff 'function abs(x){return (x < 0) ? -x : x} $1~/^@/ || abs($9) >= mini' | samtools view -b - -o $tempdir/$stem\_merged_minfl$longcutoff.bam
done

# 3) Convert BAMs into stranded BedGraphs, counting 5' ends
echo "Converting to BedGraphs..."
bam_to_stranded_bedgraph() {
	local infile="$1"
	local stem=$(basename "$infile" .bam)
	bedtools genomecov -5 -bg -strand - -ibam "$infile" | sort -k1,1 -k2,2n > $tempdir/$stem\_neg.bg
	bedtools genomecov -5 -bg -strand + -ibam "$infile" | sort -k1,1 -k2,2n > $tempdir/$stem\_pos.bg
}

for stem in ${stems[*]}
do
	bam_to_stranded_bedgraph $tempdir/$stem\_merged_maxfl$shortcutoff.bam
	bam_to_stranded_bedgraph $tempdir/$stem\_merged_minfl$longcutoff.bam
done

# 4) Convert BedGraphs into BigWigs
echo "Converting to BigWigs..."
for item in `find $tempdir -name *.bg`
do
	stem=$(basename "$item" .bg)
	bedGraphToBigWig $item $chromsizes $outdir/$stem.bw
done

rm -rf $tempdir
{"mode":"full","isActive":false}