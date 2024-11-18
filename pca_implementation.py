import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

class PCA:

    def __init__(self, n_components):

        # The number of components we want to keep (it can be 2 or 3 etc.)
        self.n_components = n_components
        self.mean = None
        self.components = None


    def fit(self, X):

        # Step 1: Subtract the mean from X
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean

        # Step2: Calculate the Covariance Matrix 
        cov_matrix = np.dot(X.T, X)

        # Step 3: Compute the Eigenvalues and Eigenvectors
        eigenvectors, eigenvalues = np.linalg.eig(cov_matrix)

        # We are getting the Transpose 
        # of the eigenvectors
        # just to make the computation
        # easier in the future
        eigenvectors = eigenvectors.T

        # Sort the eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):

        # Data projection
        X = X - self.mean
        print(X.shape)
        print(self.components.T.shape)
        return np.dot(X, self.components.T)


if __name__ == '__main__':

    # Load the dataset
    dataset = datasets.load_iris()
    
    X = dataset['data'] # shape (150, 4)
    y = dataset['target'] # shape (150,)

    pca = PCA(n_components = 2)
    pca.fit(X)

    X_projected = pca.transform(X)


    pca_1 = X_projected[:, 0] # first principal component
    pca_2 = X_projected[:, 1] # second principal component

    plt.scatter(pca_1, pca_2,
                c = y, edgecolor = "none",
                alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
                )
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.colorbar()
    plt.show()
