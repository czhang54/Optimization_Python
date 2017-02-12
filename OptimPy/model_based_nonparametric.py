
'''
Non-parametric model-based algorithms: PFO, SMC-SA, SISR, SA
'''

import numpy as np
import numpy.linalg as la    
import copy

from algorithm import Algorithm # Import base class


class Model_based_nonparametric(Algorithm):
    '''Base class for all model-based (non-parametric) algorithms (PFO, SMC-SA, SISR), inherited from Algorithm class
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(Model_based_nonparametric, self).__init__(optimizer, **kwargs)
        
        self.noise_type   = kwargs.get('noise_type', 'constant') # Type of diffusion noise        
        self.kernel_bw    = kwargs.get('bw', 0.1) # Std. for constant noise
        self.cov          = np.diag([self.kernel_bw**2]*self.d) # Cov matrix for constant noise
        self.alpha        = kwargs.get('alpha', 10) # Initial std value of decaying noise
        self.decay        = kwargs.get('decay', 0.95) # Decaying factor
        
    def resampling(self, X, weights):
        '''Implement resampling for general set od weighted particles.
           c.f., 'Probabilistic Robotics' (2005) by S. Thrun
           Arguments:
               + X       : particle states, (d,N)
               + weights : particle weights, (N,)
        '''
        Y = np.zeros_like(X)
        r = np.random.uniform(0,1/float(self.N))
        c = weights[0]
        i = 0
        for n in range(self.N):
            U = r + n/float(self.N)
            while U > c:
                i += 1
                c += weights[i]
            Y[:,n] = X[:,i]
        
        return Y
        
        
class PFO(Model_based_nonparametric):
    '''Particle Filtering for Optimization framework, inherited from Model_based_nonparametric class
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(PFO, self).__init__(optimizer, **kwargs)
                    
        self._elite_select = kwargs.get('elite_select', 'quantile')
        self._rho          = kwargs.get('quantile', 0.1)
        self._num_elites   = min(int(self.N*self._rho), self.N-1)
        self._IS_type      = kwargs.get('IS_type', 'uniform')
        self._r            = kwargs.get('r', 1e-4)
        
        self.measurement  = np.zeros(self.world.num_times,)
        
    def run(self, X, dt, TI):
        ''' Run the PFO algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''
            
        if TI % 100 == 0:
            print 'TI=' + str(TI)
        
        C = copy.copy(X)            
        h_curr = self.obj_fn.model.h(C) 
        
        self.performance(np.mean(C,1), h_curr, TI)            
        
        ## Generate the fictitious measurement 
        measurement = self.set_measurement(h_curr)
            
        if TI == 0:             
            self.measurement[TI]  = self.h_best[TI]        
        else:
            if measurement < self.measurement[TI-1]:
                self.measurement[TI]  = measurement
            else:
                self.measurement[TI]  = self.measurement[TI-1]
                
        if self.measurement[TI] < self.obj_fn.model.h_min:
            self.measurement[TI] = self.obj_fn.model.h_min             
            
        ## Importance sampling
        weights = self.IS(h_curr, measurement)
        
        ## Resampling
        resampled = self.resampling(C, weights)
        
        ## Diffuse
        if self.noise_type == 'constant':
            noise_cov = self.cov
        elif self.noise_type == 'decay':
            noise_std = self.alpha*self.decay**TI
            noise_cov = np.diag(np.array([noise_std]*self.d)**2) 
        Y = resampled + np.random.multivariate_normal([0]*self.d, noise_cov, self.N).T # (d,N) 
                
        return Y
            
    def set_measurement(self, h):
        ''' Generate the fictitious measurement according to a rule
        '''
        
        if self._elite_select == 'quantile': # Default option
            measurement = sorted(h.tolist())[self._num_elites]
        elif self._elite_select == 'minimum':
            measurement = np.min(h)
        
        return measurement            
            
    def IS(self, h, measurement):
        ''' Importance sampling rule
        '''
        
        weights = np.zeros_like(h)
        
        if self._IS_type == 'uniform':        
            for i in range(self.N):
                if h[i] <= measurement:
                    weights[i] = 1
        elif self._IS_type == 'exponential':            
            for i in range(self.N):
                if h[i] <= measurement:
                    weights[i] = np.exp(-self._r*(h[i]-measurement))
        
        return weights/np.sum(weights)            
            
            
class SMC_SA(Model_based_nonparametric):

    def __init__(self, optimizer=None, **kwargs):

        super(SMC_SA, self).__init__(optimizer, **kwargs)

        self._initial_dist = kwargs.get('initial_dist', 'Gaussian')
        self._mu_0         = np.array(kwargs.get('mu_0', [0]*self.d))
        self._sigma_0      = np.diag(np.array(kwargs.get('sigma_0', [1]*self.d))**2)      
        self._temperature  = np.zeros(self.world.num_times,)              
            
    def run(self, X, dt, TI):
        ''' Run the SMC-SA algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''
        
        if TI % 100 == 0:
            print 'TI=' + str(TI)
        
        C = copy.copy(X)
        h_curr = self.obj_fn.model.h(C)

        self.performance(np.mean(C,1), h_curr, TI)
                
        ## Importance sampling
        if TI == 0:
            self._temperature[TI]  = self.cooling(TI, dt) # Temperature should be set AFTER updating h_best
            ## Compute initial pdf and initial weights            
            if self._initial_dist == 'Gaussian':
                weights = np.zeros(self.N,)
                for i in range(self.N):
                    diff_0 = np.reshape(C[:,i]-self._mu_0, (self.d,1))
                    p_0 = np.exp( -0.5*np.dot(diff_0.T, np.dot(la.inv(self._sigma_0), diff_0)) )
                    if h_curr[i] > 200:
                        weights[i] = 0.0
                    else:
                        weights[i] = np.exp(-h_curr[i]/self._temperature[TI])/p_0
            elif self._initial_dist == 'Uniform':
                weights = np.exp(-h_curr/self._temperature[TI])
                
        else:
            self._temperature[TI]  = self.cooling(TI, dt)
            weights = np.exp(h_curr*(1/self._temperature[TI-1]-1/self._temperature[TI]))
                        
        weights = weights/np.sum(weights)           
                       
        ## Resampling
        resampled = self.resampling(C, weights)
        h_resampled = self.obj_fn.model.h(resampled)
        
        ## SA move
        if self.noise_type == 'constant':
            noise_cov = self.cov
        elif self.noise_type == 'decay':
            noise_std = self.alpha*self.decay**TI
            noise_cov = np.diag(np.array([noise_std]*self.d)**2)
        SA_candidates = resampled + np.random.multivariate_normal([0]*self.d, noise_cov, self.N).T # (d,N)
        h_diffused = self.obj_fn.model.h(SA_candidates)
        Y = np.zeros_like(SA_candidates)
        
        diff = (h_resampled-h_diffused)/self._temperature[TI]

        for i in range(self.N):
            if diff[i] < -200:
                prob = 0
            else:
                MH_ratio = np.exp(diff[i])
                prob = min(MH_ratio, 1)
            v = np.random.rand()
            if v < prob:
                Y[:,i] = SA_candidates[:,i]
            else:
                Y[:,i] = resampled[:,i]
            
        return Y            
            
    def cooling(self, TI, dt):
        ''' Cooling schedule
        '''
        
        temperature = abs(self.h_best[TI])/np.log(TI+2)
        
        return temperature            
            
            
class SISR(Model_based_nonparametric):
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(SISR, self).__init__(optimizer, **kwargs)

        self._beta         = kwargs.get('beta', 1.0)
        self._rho          = kwargs.get('quantile', 1.0)
        self._num_elites   = min(int(self.N*self._rho), self.N-1)
        
    def run(self, X, dt, TI):
        ''' Run the SISR algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''
        
        if TI % 100 == 0:
            print 'TI=' + str(TI)
            
        C = copy.copy(X)
        h_curr = self.obj_fn.model.h(C)   
       
        self.performance(np.mean(C,1), h_curr, TI)

        level = sorted(h_curr.tolist())[self._num_elites] 
                
        ## Importance sampling
        weights = np.zeros_like(h_curr)
        for i in range(self.N):
            if h_curr[i] <= level:
                weights[i] = np.exp(-self._beta*h_curr[i]*dt)  
        weights = weights/np.sum(weights)
                
        ## Resampling
        resampled = self.resampling(C, weights)
        
        ## Diffuse
        if self.noise_type == 'constant':
            noise_cov = self.cov
        elif self.noise_type == 'decay':
            noise_std = self.alpha*self.decay**TI
            noise_cov = np.diag(np.array([noise_std]*self.d)**2) 
        Y = resampled + np.random.multivariate_normal([0]*self.d, noise_cov, self.N).T # (d,N) 
                
        return Y       


class SA(Model_based_nonparametric):
    
    def __init__(self, optimizer=None, **kwargs):
        
        super(SA, self).__init__(optimizer, **kwargs)

        self._temperature  = np.zeros(self.world.num_times,)
        
    def run(self, X, dt, TI):
        ''' Run the SA algorithm
            Arguments:
                + X  : Input state/particles, shape (d,1)
                + dt : Time step
                + TI : Time index
        '''
        
        if TI % 100 == 0:
            print 'TI=' + str(TI)
        
        C = copy.copy(X)
        h_curr = self.obj_fn.model.h(C) # A number
        
        self.performance(C[:,0], np.array(h_curr), TI)
        
        ## Diffusion
        if self.noise_type == 'constant':
            noise_cov = self.cov
        elif self.noise_type == 'decay':
            noise_std = self.alpha*self.decay**TI
            noise_cov = np.diag(np.array([noise_std]*self.d)**2)
            
        candidate = C + np.random.multivariate_normal([0]*self.d, noise_cov, self.N).T # (d,1)
        h_cand = self.obj_fn.model.h(candidate)

        ## Accept or reject      
        self._temperature[TI] = self.cooling(TI, dt)
        MH_ratio = np.exp( (h_curr-h_cand)/self._temperature[TI] )
        prob     = min(MH_ratio, 1)
        v = np.random.rand()
        if v < prob:
            Y = candidate
        else:
            Y = C
            
        return Y
        
        
    def cooling(self, TI, dt):
        ''' Cooling schedule
        '''
        temperature = abs(self.h_best[TI])/np.log(TI+2)
        
        return temperature            
            
            
            
            
            
