
'''
Parametric model-based algorithms: CE, MRAS, MEO
'''

import numpy as np
import numpy.linalg as la    
import copy

from algorithm import Algorithm # Import base class
        
    
class Model_based_parametric(Algorithm):
    '''Base class for all model-based (parametric) algorithms (CE, MRAS, MEO), inherited from Algorithm class
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(Model_based_parametric, self).__init__(optimizer, **kwargs)
        
        self.rho          = kwargs.get('quantile', 0.1)
        self.nu           = kwargs.get('smooth_param', 1.0)
        self.num_elites   = min(int(np.ceil(self.N*self.rho)), self.N-1)
        self.IS_type      = kwargs.get('IS_type', 'uniform')   
        self.r            = kwargs.get('r', 1e-4)
        
        self.mean         = np.zeros((self.world.num_times, self.d))
        self.cov          = np.zeros((self.world.num_times, self.d, self.d)) 
        self.mean[0]      = np.array(kwargs['mu_0'])
        self.cov[0]       = np.diag(kwargs['sigma_0'])**2
    
    def select_elite(self, X, level, h):
        ''' Select elite samples according to a rule 
        '''
        
        h_elites = []
        count = 0
        for i in range(self.N):
            if h[i] <= level:
                h_elites.append(h[i])
                if count == 0:
                    elites = X[:,i]
                else:
                    elites = np.vstack((elites, X[:,i]))
                count += 1
                
        if count > 1:
            return elites.T, np.array(h_elites)    
        elif count == 1:
            return np.reshape(elites, (self.d,1)), np.array(h_elites)
        else: # If no elite samples are selected
            return 0, []
            
    def importance_weight(self, h_elites):
        ''' Compute importance weight for the samples
        '''        
        
        weights = self.IS(h_elites)
        weights = weights/np.sum(weights)
        
        return weights
            
    def parameter_update(self, elites, weights):
        ''' Parameter update rule, similar to maximum likelihood
        '''
        
        mean = np.sum(weights*elites, 1)        
        diff = np.sqrt(weights)*(elites - np.reshape(mean, (self.d,1)))
        cov  = np.dot(diff, diff.T)
        
        return mean, cov  
        
    def IS(self, h):
        ''' Importance sampling rule
        '''
        
        if self.IS_type == 'uniform': ## Standard CE
            return np.ones_like(h) 
        elif self.IS_type == 'linear': ## Extended CE
            return h
        elif self.IS_type == 'exponential':
            return np.exp(-self.r*h)
        
        
class CE(Model_based_parametric):
    '''Cross Entropy algorithm, inherited from Model_based_parametric class
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(CE, self).__init__(optimizer, **kwargs)
    
    def run(self, X, dt, TI):
        ''' Run the CE algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''

        if TI % 100 == 0:
            print 'TI=' + str(TI)
            
        C = copy.copy(X)
        h_curr = self.obj_fn.model.h(C) ## Can be time-consuming when particles are concentrated 
        h_curr_sorted = sorted(h_curr.tolist())
        
        self.performance(np.mean(C,1), h_curr, TI)
        
        level = h_curr_sorted[self.num_elites]     
        elites, h_elites = self.select_elite(C, level, h_curr)
        
        ## Update model parameters
        if len(h_elites) == 0: # If no elite sample selected
            self.mean[TI+1], self.cov[TI+1] = self.mean[TI], self.cov[TI]
        else:
            weights = self.importance_weight(h_elites)                    
            mean_update, cov_update = self.parameter_update(elites, weights)
            self.mean[TI+1] = self.nu*mean_update + (1-self.nu)*self.mean[TI] 
            self.cov[TI+1]  = self.nu*cov_update  + (1-self.nu)*self.cov[TI]
                
        Y = np.random.multivariate_normal(self.mean[TI+1], self.cov[TI+1], self.N).T
                    
        return Y
        
        
class MRAS(Model_based_parametric):
    '''Model Reference Adaptive Search algorithm, inherited from Model_based_parametric class
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(MRAS, self).__init__(optimizer, **kwargs)
        
        self._e            = kwargs.get('epsilon', 1e-5)

        self._level        = np.zeros(self.world.num_times,)
        self._quantile     = np.zeros(self.world.num_times,)
        self._quantile[0]  = self.rho
    
    def run(self, X, dt, TI):
        ''' Run the MRAS algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''

        if TI % 100 == 0:
            print 'TI=' + str(TI)        
        
        C = copy.copy(X)
        h_curr = self.obj_fn.model.h(C) ## Can be time-consuming when particles are concentrated 
        h_curr_sorted = sorted(h_curr.tolist())
        
        self.performance(np.mean(C,1), h_curr, TI)
        
        ## Update level parameter
        if TI == 0:
            index = min(int(np.ceil(self.N*self._quantile[TI])), self.N-1)
            self._level[TI]  = h_curr_sorted[index]
        else:            
            index = min(int(np.ceil(self.N*self._quantile[TI-1])), self.N-1)
            level = h_curr_sorted[index]
            self.quantile_update(TI, level, h_curr_sorted)
        
        elites, h_elites = self.select_elite(C, self._level[TI], h_curr)
        
        ## Update model parameters
        if len(h_elites) == 0: # If no elite sample selected
            self.mean[TI+1], self.cov[TI+1] = self.mean[TI], self.cov[TI]
        else:
            mean_update, cov_update = self.parameter_update_MRAS(elites, h_elites, TI)
            self.mean[TI+1] = self.nu*mean_update + (1-self.nu)*self.mean[TI] 
            self.cov[TI+1]  = self.nu*cov_update  + (1-self.nu)*self.cov[TI]
                
        Y = np.random.multivariate_normal(self.mean[TI+1], self.cov[TI+1], self.N).T
                    
        return Y        
        
    def quantile_update(self, TI, level, h_sorted):
        ''' Update quantile level s.t. it is non-increasing w.r.t. iterations
        '''
        
        if level <= self._level[TI-1] - self._e/2.0:
            self._level[TI] = level
            self._quantile[TI] = self._quantile[TI-1]
        else:
            quantile_found = False
            quantile = self._quantile[TI-1]
            while (quantile>0) and (not quantile_found):
                index = min(int(np.ceil(self.N*quantile)), self.N-1)
                level_iter = h_sorted[index]
                if level_iter <= self._level[TI-1] - self._e/2.0:
                    quantile_found = True
                quantile -= 0.0001
            if quantile > 0:
                self._level[TI] = level_iter
                self._quantile[TI] = quantile
            else:
                self._level[TI] = self._level[TI-1]
                self._quantile[TI] = self._quantile[TI-1]
                
        if self._level[TI] < self.obj_fn.model.h_min:
            self._level[TI] = self.obj_fn.model.h_min
        
    def parameter_update_MRAS(self, elites, h_elites, TI):
        ''' Parameter update rule for MRAS
        '''
        
        ## Evaluate current parametrized pdf (proposal pdf) which is a mixture with initial distribution
        p_curr = np.zeros(elites.shape[1])
        for i in range(elites.shape[1]):
            diff      = np.reshape(elites[:,i]-self.mean[TI], (self.d,1))
            p_curr[i] = np.exp( -0.5*np.dot(diff.T, np.dot(la.inv(self.cov[TI]), diff)) ) #/np.sqrt(la.det(self.cov[TI]))
                
        ## Evaluate target (reference) distribution   
        ref = self.IS(h_elites)**(TI)
        
        weights = ref/p_curr
        weights = weights/np.sum(weights)

        return self.parameter_update(elites, weights)   
        
        
class MEO(Model_based_parametric):
    '''Model-based Evolutionary Optimization algorithm, inherited from Model_based_parametric class
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(MEO, self).__init__(optimizer, **kwargs)
        
        self._level        = np.zeros(self.world.num_times,)

    
    def run(self, X, dt, TI):
        ''' Run the MEO algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''

        if TI % 100 == 0:
            print 'TI=' + str(TI)        
        
        C = copy.copy(X)
        h_curr = self.obj_fn.model.h(C) ## Can be time-consuming when particles are concentrated 
        
        self.performance(np.mean(C,1), h_curr, TI)
        
        ## Update level parameter
        level  = sorted(h_curr.tolist())[self.num_elites]
        if TI == 0:
            self._level[TI] = level
        else:            
            if level < self._level[TI-1]:
                self._level[TI] = level
            else:
                self._level[TI] = self._level[TI-1]
                
        if self._level[TI] < self.obj_fn.model.h_min:
            self._level[TI] = self.obj_fn.model.h_min
        
        elites, h_elites = self.select_elite(C, self._level[TI], h_curr)
        
        ## Update model parameters
        if len(h_elites) == 0: # If no elite sample selected
            self.mean[TI+1], self.cov[TI+1] = self.mean[TI], self.cov[TI]
        else:
            mean_update, cov_update = self.parameter_update_MEO(elites, h_elites, TI, dt)
            self.mean[TI+1] = self.nu*mean_update + (1-self.nu)*self.mean[TI] 
            self.cov[TI+1]  = self.nu*cov_update  + (1-self.nu)*self.cov[TI]
                
        Y = np.random.multivariate_normal(self.mean[TI+1], self.cov[TI+1], self.N).T
                    
        return Y        
        
    def parameter_update_MEO(self, elites, h_elites, TI, dt):
        ''' Parameter update rule for MEO
        '''
        
        hs = np.sum(h_elites)
        hm = hs/float(self.N)
        weights = (1-(h_elites-hm)/(hs-hm))/float(self.N)
        
        return self.parameter_update(elites, weights)        
        
        
        
        
        
        
        
        
        
