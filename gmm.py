import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        # The shape of the dataset
        n, d = X.shape 

        self.priors = np.full(shape=self.k, fill_value=1/self.k)
        self.posteriors = np.full(shape=X.shape, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=n, size=self.k)
        # Each element corresponds to each cluster, and it is an array with the mean for each attribute
        self.mu = [  X[row_index,:] for row_index in random_row ]
        
        # Each element corresponds to each cluster, and it is an array with the variance/covariance for each attribute
        # debug it, you'll see that the upwards diagonal has the same value, that is because it is relating the same variable.
        # The other diagonal has different values
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]

        self.log_likelihoods = []
        

    def e_step(self, X):
        # E-Step: update probabilities and priors holding mu and sigma constant
        self.posteriors = self.predict_proba(X)
        self.priors = self.posteriors.mean(axis=0)
    
    def m_step(self, X):
        # M-Step: update mu and sigma we already have the prioirs and probabilities here, they remain constant at this step
        for i in range(self.k):
            # UPDATE/CALCULATE the prob for all examples for this k, an n x 1 vector
            prob = self.posteriors[:, [i]]
            # UPDATE/CALCULATE total_weight with the total prob (aka weight) for this k
            total_weight = np.average(prob)
            # UPDATE/CALCULATE this: 
            self.mu[i] = np.sum((prob / np.sum(prob)) * X, axis=0)
            # UPDATE/CALCULATE once you have obtained prob and total weight this calculates the covariance
            self.sigma[i] = np.cov(X.T, 
                aweights=(prob/total_weight).flatten(), 
                bias=True)

    # UPDATE/CALCULATE this function
    # You will use the 
    def log_likelihood(self, X):
        likelihood = np.zeros( (X.shape[0], self.k) )
        accum = 0
        for i in range(self.k):
            p = np.average(self.posteriors[:, [i]])
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
            accum += likelihood[:,i] * p

        ll = np.sum(np.log(np.sum(accum)))
        self.log_likelihoods.append(ll)
        

    def fit(self, X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            self.log_likelihood(X)
            
    def predict_proba(self, X):
        # likelihood is an n x k matrix that holds in each column the information for each cluster.
        # in each row, it calculates the likelihood that an example belongs to any cluster.
        # P((x_i)│c) in our slides.
        likelihood = np.zeros( (X.shape[0], self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        # probabilities then is the calculation of P(c|x_i).
        # See in our slides that similar elements are participating in both numerator and denominator
        numerator = likelihood * self.priors
        # that is why here we sum all the likelihoods (there is a sum in our slides in denominator)
        # the np.newaxis is just to create an (n,1) vector, rather than an (n,) that would fail in further calculations
        denominator = numerator.sum(axis=1)[:, np.newaxis] 
        probabilities = numerator / denominator
        return probabilities
    
    def predict(self, X):
        # Again, probabilities here refers to a matrix with the probabilities for each cluster
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)