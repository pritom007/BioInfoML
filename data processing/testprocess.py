NUMBER_OF_LINES = 22284  # number of lines/genes in microarray_original.txt
# GPL96-15653.txt also has 22284 lines (plus 16 more for explanation)
# microarray_original.txt has 5897 columns, the number of observations
file_dir = "D:\\Pritom Lab\Sjtu\\2nd Semester\\Bio Info\\Gene_Chip_Data\\Gene_Chip_Data"


def read_only_lines(f, start, finish):
    for ii, line in enumerate(f):
        if ii >= start and ii < finish:
            yield line
        elif ii >= finish:
            return


# 5,897 columns in each line
# lines start with #ID (Affymetrix Probe Set ID),
# then succeeded by numbers (breakdown of appearance of each probe/RMA signal?)
line_start = 0
for line in read_only_lines(open(file_dir + '\\microarray_original.txt'), line_start, line_start + 5):
    print(line.split()[0])
# for word in line.split():
#	print(word)
