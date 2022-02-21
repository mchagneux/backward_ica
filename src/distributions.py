from scipy.stats import multivariate_normal
from scipy.stats import uniform
import numpy as np 

class Distribution:

    def __init__(self):
        pass 

    def pdf(self, x):
        pass 

    def sample(self, num_samples):
        pass
    
class DistributionFromScipy(Distribution):
    def __init__(self, scipy_dist):
        self.scipy_dist = scipy_dist
    
    def sample(self, num_samples):
        return self.scipy_dist.rvs(size=num_samples)

    def pdf(self, x): 
        return self.scipy_dist.pdf(x)
        
    def logpdf(self, x):
        return self.scipy_dist.logpdf(x)

class Gaussian(DistributionFromScipy):

    def __init__(self, mean, cov):
        super().__init__(scipy_dist=multivariate_normal(mean=mean, cov=cov))
        self.mean = self.scipy_dist.mean
        self.cov = self.scipy_dist.cov

class MultivariateUniform(Distribution):
    def __init__(self, dim, loc, scale):
        self.dim = dim
        self.scipy_dist = uniform(loc = loc, scale = scale)

    def sample(self, num_samples):
        return np.array([self.scipy_dist.rvs(self.dim) for _ in range(num_samples)]).squeeze()
        
# if __name__ == '__main__':
#     gaussian = Gaussian(mean=np.zeros((2,)), cov = np.eye(2))

#     gaussian.mean = np.ones((2,))
#     print(gaussian.scipy_dist.mean)

