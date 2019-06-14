file_dir = "C:/Users/Petros Debesay/Downloads/Gene_Chip_Data/"
output_file = 'C:/Users/Petros Debesay/PycharmProjects/BioInfoML/Data/processed.csv'


data = [i.strip('\n').split('\t') for i in open(file_dir + 'E-TABM-185.sdrf.txt')]

disease_dict = dict()
for x in range(1, len(data)):
    if data[x][7] in disease_dict.keys():
        current_list = disease_dict[data[x][7]]
        current_list.append(x)
    else:
        disease_dict[data[x][7]] = [x]
    print(data[x][7])
    print(x)

print(disease_dict)
