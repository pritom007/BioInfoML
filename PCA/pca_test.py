import pandas as pd
from sklearn.decomposition import PCA

file_dir = 'C:/Users/Petros Debesay/PycharmProjects/BioInfoML/data processing/processed.csv'

number_components = 7

df = pd.read_csv(file_dir)

pca = PCA(n_components=number_components)

components = []

for x in range(0, number_components):
    components.append("Component" + str(x))

principal_components = pca.fit_transform(df)

principal_df = pd.DataFrame(data=principal_components, columns=components)

print(principal_df)
