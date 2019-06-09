NUMBER_OF_LINES = 22284  # number of lines/genes in microarray_original.txt
# GPL96-15653.txt also has 22284 lines (plus 16 more for explanation)
# microarray_original.txt has 5897 columns, the number of observations

import csv

file_dir = "D:\\Pritom Lab\Sjtu\\2nd Semester\\Bio Info\\Gene_Chip_Data\\Gene_Chip_Data\\"
output_file = 'C:/Users/Petros Debesay/PycharmProjects/BioInfoML/Data/processed.csv'


def read_only_lines(f, start, finish):
    for ii, line in enumerate(f):
        if ii >= start and ii < finish:
            yield line
        elif ii >= finish:
            return


def mean_around_zero(string_list):
    total = 0
    zero_mean_list = []

    for i in range(1, len(string_list)):
        current = float(string_list[i])
        total += current
        zero_mean_list.append(current)

    mean = total / (len(string_list) - 1)

    for x in range(0, len(zero_mean_list)):
        zero_mean_list[x] -= mean

    return zero_mean_list


def normalize(lst, max_val):
    for i in range(0, len(lst)):
        lst[i] /= max_val

    return lst

# 5,897 columns in each line
# lines start with #ID (Affymetrix Probe Set ID),
# then succeeded by numbers (breakdown of appearance of each probe/RMA signal?)
line_start = 1
line_end = 22284
for line in read_only_lines(open(file_dir + 'microarray.original.txt'), line_start, line_end):
    mean_zero = mean_around_zero(line.split())
    max_val = max(mean_zero, key=abs)

    mean_zero = normalize(mean_zero, abs(max_val))

    print(mean_zero)

    with open(output_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(mean_zero)

# for word in line.split():
#	print(word)
