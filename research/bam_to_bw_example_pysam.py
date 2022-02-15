# need to activate samtools environment or any other env that has pysam first

import datetime
bam_input = '/home/katie/bp_repo/research/data/cutnrun/temp/FOSL2_merged_filtered.bam'
bam_out = '/tmp/bam_out.bam'

import pysam

# open existing bam file for reading
samfile = pysam.AlignmentFile(bam_input, 'rb',threads=30)

# open new bam file to which you will write filtered sequences
sam_out = pysam.AlignmentFile(bam_out,'wb',template=samfile,threads=30)

# iterate through input file
# check template_length of AlignmentSegment, if it is longer than desired length then write it to output file
# AlignmentSegment has information about the segment that can be used for more filtering; for example
# z.is_reverse is a flag specifying if the strand is reversed
print(datetime.datetime.now().time())
for z in samfile:
    if z.template_length > 180:
        sam_out.write(z)
print(datetime.datetime.now().time())
# files are done; close both of them
sam_out.close()
samfile.close()

# create an index for the output file; index is needed by bioconvert to convert to bigwig
pysam.index(bam_out)

# convert bam to bigwig on command line
# conda (or source) activate bioconvert
# bioconvert bam2bigwig /tmp/bam_out.bam /tmp/bam_out.bw
