
import numpy as np


class Uniform(object):
    ''' MUlti-domensional uniform distribution
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        self.optimizer = optimizer
        self.dim       = kwargs.get('dim', 2)
        self._interval = kwargs.get('interval', [-10,10])
        
    def f(self, N):
        
        IC = np.random.uniform(low=self._interval[0], high=self._interval[1], size=(self.dim, N))
        
        return IC
        

class Gaussian(object):
    ''' Multi-dimensional Gaussian with independent components
    '''

    def __init__(self, optimizer=None, **kwargs):
        
        self.optimizer = optimizer 
        self._mean     = kwargs.get('IC', None)       
        self._sigma    = np.array(kwargs.get('sigma', None))
        self._cov      = np.diag(self._sigma**2) 
            
    def f(self, N):
        
        IC = np.random.multivariate_normal(self._mean, self._cov, N).T

        return IC
        
        
class GMM(object): 
    ''' Multi-dimensional Gaussian mixture model
    '''

    def __init__(self, optimizer=None, **kwargs):

        self.optimizer = optimizer 
        self._M        = kwargs.get('M', 2) # Number of Gaussian components in the GMM
        self._mean     = kwargs.get('IC', None) # A list of ICs of each component
        self._sigma    = kwargs.get('sigma', None) # A list of stds of each component
        self._weight   = kwargs.get('weight', [0.5,0.5]) # Weights of each component
        
    def f(self, N):
        
        cov = np.diag(np.array(self._sigma[0])**2)
        X = np.random.multivariate_normal(self._mean[0], cov, int(N*self._weight[0])).T
        
        for i in range(1,len(self._mean)):
            cov = np.diag(np.array(self._sigma[i])**2)
            Y = np.random.multivariate_normal(self._mean[i], cov, int(N*self._weight[i])).T
            X = np.concatenate((X,Y), axis=1)
                    
        return X
        


        
        
 

        
        
        









        
