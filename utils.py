import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

def generate_shapes(n_samples = 1500, random_state = 170):
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X = np.dot(X, transformation)
    return X

def get_boundary_grid(X, gmm):
    x = np.linspace(np.min(X[...,0])-1,np.max(X[...,0])+1,100)
    y = np.linspace(np.min(X[...,1])-1,np.max(X[...,1])+1,80)
    X_,Y_ = np.meshgrid(x,y)
    x_ = X_.flatten()
    y_ = Y_.flatten()
    pos = np.array([x_,y_]).T
    predictions_grid = gmm.predict(pos)
    return (predictions_grid, x_, y_)

def plot_data(X, predictions, gmm):
    plt.figure()
    plt.title('GMM Clusters {} iters'.format(gmm.max_iter))
    plt.xlabel("x1")
    plt.ylabel("x2")

    (predictions_grid, x, y) = get_boundary_grid(X, gmm)
    plt.scatter(x, y, c=predictions_grid, marker='.', s=0.5)

    plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap=plt.cm.get_cmap('brg'), marker='x', s=1)
    plt.tight_layout()
    
def plot_data_kmeans(X, predictions, iters, kmeans):
    plt.figure()
    plt.title('Kmeans Clusters {} iters'.format(iters))
    plt.xlabel("x1")
    plt.ylabel("x2")

    (predictions_grid, x, y) = get_boundary_grid(X, kmeans)
    plt.scatter(x, y, c=predictions_grid, marker='.', s=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap=plt.cm.get_cmap('brg'), marker='x')
    plt.tight_layout()

def plot_likelihood(values):
    plt.figure()
    plt.title('Log-Likelihood {} iters'.format(len(values)))
    plt.xlabel("iter")
    plt.ylabel("log likelihood")
    plt.plot(values, marker='x')
    plt.tight_layout()