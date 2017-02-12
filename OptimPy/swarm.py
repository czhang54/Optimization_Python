
import numpy as np
import numpy.linalg as la    
import copy

''' Swarm intelligence algorithms, e.g. particle swarm optimization (PSO)
'''

        
class PSO(object):
    
    def __init__(self, sde=None, **kwargs):

        self.dynamics     = sde
        self.item         = self.dynamics.item
        self.world        = self.dynamics.item.world
        self.d            = self.world.dim
        self.N            = self.dynamics.N # Number of particles  
        
        self.obj_fn       = self.world.obj_fns()[0]
        
        self._a            = kwargs.get('a', 2)
        self._b            = kwargs.get('b', 2)
        self._w            = kwargs.get('w', 1)
        self._x            = kwargs.get('x', 1)
        self._v_l          = kwargs['v_l'] # Used for sampling initial velocity
        self._v_u          = kwargs['v_u'] # Used for sampling initial velocity
        
        self.error        = np.zeros(self.world.num_times,)
        self.h_hat        = np.zeros(self.world.num_times,) 
        self.h_best       = np.zeros(self.world.num_times,) 
        self.swarm_best   = np.zeros((self.world.num_times, self.d)) # Store swarm best solution thru ALL steps
        self.indiv_best   = np.zeros((self.d, self.N)) # Store individual best solutions at ONE step
        self.velocity     = np.zeros((self.d, self.N)) # Store individual velocity at ONE step
        
        ## Initialize particle velocities
        for i in range(self.d):
            self.velocity[i] = np.random.uniform(self._v_l[i], self._v_u[i], self.N)
        
    def run(self, X, dt, TI, **kwargs):
        ''' Run the PSO algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''
        
        C = copy.copy(X)

        if TI % 100 == 0:
            print 'TI=' + str(TI)
        
        self.error[TI] = la.norm(np.mean(C,1)-self.obj_fn.model.x_min)
                    
        if TI == 0:
            ## Initialize best values
            h = self.obj_fn.model.h(C)
            self.h_best[TI] = np.min(h)
            self.h_hat[TI]  = np.mean(h)            
            self.indiv_best = C # Initial individual best solutions are just the initial particles
            self.swarm_best[TI]  = C[:,h.tolist().index(self.h_best[TI])]
            
        p = np.random.random()
        q = np.random.random()
                        
        if TI > 0:               
            ## Update particle velocity and state
            for i in range(self.N):
                p  = np.random.random((self.d,))
                q  = np.random.random((self.d,))
                dv = self._a*p*(self.swarm_best[TI]-X[:,i]) + self._b*q*(self.indiv_best[:,i]-X[:,i])
                self.velocity[:,i] = self.velocity[:,i]*self._w + dv

        Y = C + self.velocity*self._x
                
        ## Update individual and swarm best solutions
        h_update = self.obj_fn.model.h(Y)
        h_min_update = np.min(h_update)
        self.h_hat[TI+1] = np.mean(h_update)
        
        for i in range(self.N):
            if h_update[i] < self.obj_fn.model.h( np.reshape(self.indiv_best[:,i], (self.d,1)) ):
                self.indiv_best[:,i] = Y[:,i]

        if h_min_update < self.h_best[TI]:
            self.swarm_best[TI+1]  = Y[:,h_update.tolist().index(h_min_update)]
            self.h_best[TI+1] = h_min_update
        else:
            self.swarm_best[TI+1] = self.swarm_best[TI]
            self.h_best[TI+1] = self.h_best[TI]
                
        return Y
    






