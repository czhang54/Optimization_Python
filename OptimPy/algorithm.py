"""
Created on Mon Feb  6 17:06:35 2017

@author: zhangchi314
"""

import numpy as np
import numpy.linalg as la    

class Algorithm(object):
    '''Base class for all optimization algorithms
    '''
    
    def __init__(self, sde=None, **kwargs):
        
        self.optimizer     = sde
        self.item         = self.optimizer.item
        self.world        = self.optimizer.item.world
        self.d            = self.world.dim
        self.N            = self.optimizer.N # Number of particles
        
        self.obj_fn       = self.world.obj_fns()[0]
        
        self.h_best       = np.zeros(self.world.num_times,) # Best function value found
        self.h_hat        = np.zeros(self.world.num_times,) # Averaged function value
        self.error        = np.zeros(self.world.num_times,) # State error
        
    def performance(self, mean, h, TI):
        '''Update h_best, h_hat and estimate error (i.e. distance error between estimated mean and true minimizer)
        '''        
        self.h_hat[TI] = np.mean(h)
        
        h_min_curr = np.min(h)
        if TI == 0:
            self.h_best[TI] = h_min_curr
        elif h_min_curr < self.h_best[TI-1]:
            self.h_best[TI] = h_min_curr
        else:
            self.h_best[TI] = self.h_best[TI-1]
            
        self.error[TI] = la.norm(mean-self.obj_fn.model.x_min)
        
        
        
        
        
        
        
        
        
