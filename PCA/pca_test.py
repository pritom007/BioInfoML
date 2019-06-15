import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

file_dir = 'C:/Users/Petros Debesay/PycharmProjects/BioInfoML/data_processing/processed.csv'

df = pd.read_csv(file_dir)

pca = PCA(n_components=0.99, svd_solver='full')

components = []

for x in range(0, len(df.columns)):
    components.append("Component" + str(x))

principal_components = pca.fit_transform(df)

principal_df = pd.DataFrame(data=principal_components)

np.savetxt("pca.csv", principal_df, delimiter=",")
