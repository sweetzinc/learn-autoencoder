import h5py
import numpy as np
from sklearn.decomposition import PCA

# Load the HDF dataset
def load_hdf_dataset(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

# Perform PCA and calculate explained variance
def perform_pca(data, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_
    return pca, explained_variance

if __name__ == "__main__":
    # Example usage
    file_path = 'path_to_your_hdf_file.h5'
    dataset_name = 'your_dataset_name'
    
    data = load_hdf_dataset(file_path, dataset_name)
    pca, explained_variance = perform_pca(data)
    
    print("Explained variance by principal components:")
    print(explained_variance)

