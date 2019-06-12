import pandas as pd
from sklearn.decomposition import PCA

file_dir = 'C:/Users/Petros Debesay/PycharmProjects/BioInfoML/data processing/processed.csv'

df = pd.read_csv(file_dir)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(df)

principal_df = pd.DataFrame(data=principal_components, columns=["test1", "test2"])

print(principal_df)
