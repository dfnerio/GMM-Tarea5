import numpy as np
from gmm import GMM
if __name__ == "__main__":
    from utils import generate_shapes, plot_data, plot_likelihood, plot_data_kmeans
    import matplotlib.pyplot as plt
    np.random.seed(0)
    
    X = generate_shapes()
    iters = 100

    gmm = GMM(k=3, max_iter=iters)
    gmm.fit(X)

    predictions = gmm.predict(X)

    plot_data(X, predictions, gmm)
    plot_likelihood(gmm.log_likelihoods)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0, max_iter=iters).fit(X)
    plot_data_kmeans(X, kmeans.labels_, iters, kmeans)
    
    plt.show()