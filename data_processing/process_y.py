file_dir = "C:/Users/Petros Debesay/Downloads/Gene_Chip_Data/"
output_file = 'C:/Users/Petros Debesay/PycharmProjects/BioInfoML/Data/processed.csv'


def return_y_dict():
    data = [i.strip('\n').split('\t') for i in open(file_dir + 'E-TABM-185.sdrf.txt')]

    disease_dict = dict()
    for x in range(1, len(data)):
        if data[x][7] in disease_dict.keys():
            current_list = disease_dict[data[x][7]]
            current_list.append(x)
        else:
            disease_dict[data[x][7]] = [x]
    return disease_dict


def return_y(key):
    data = [i.strip('\n').split('\t') for i in open(file_dir + 'E-TABM-185.sdrf.txt')]

    y = []
    for x in range(1, len(data)):
        if key in data[x][7].lower():
            y.append([1])
        else:
            y.append([0])

    return y


def return_dict():
    name_size = dict()
    disease_dict = return_y_dict()

    for key in disease_dict.keys():
        name_size[key] = len(disease_dict[key])

    return name_size
