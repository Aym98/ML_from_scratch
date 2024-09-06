import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

#Algo steps :
#Substract the mean of dataset
#Calculate the matrix covariance Cov(X, X)
#Calculate the matrix eigenvectors and eigenvalues
#Sort the eigenvectors in descending order of their eigenvalues
#Select first K components         | K is n_components
#Project the dataset on the K eigenvectors




class PCA:
    def __init__(self, n_componnents) -> None:
        self.n_componnents = n_componnents
        self.components = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0) 
        X = X - self.mean
        
        cov_matrix = np.cov(X.T)
        eigenvectors, eigenvalues = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T
        
        idxs = np.argsort(eigenvalues, axis=1)[::-1]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[:self.n_componnents]


    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T) 
    



if __name__ == "__main__" :

    

    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    print(f"The projected data shape is {X_pca.shape}")
    print(f"The original data shape is {X.shape}")

    fig, axs = plt.subplots(2)
    pca1, pca2 = X_pca[:, 0], X_pca[:, 1]
    axs[0].scatter(pca1, pca2, c=y)
    axs[0].set_title("PCA axis")
    
    axs[1].scatter(X[:, 0], X[:, 1], c=y)
    axs[1].set_title("First and second feature axes")
    
    fig.suptitle("Scatter plot of the labels of the iris dataset following two sets of different axis")

    
    plt.show()

