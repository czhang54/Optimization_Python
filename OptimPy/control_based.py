
import numpy as np
import numpy.linalg as la      
import math  

from algorithm import Algorithm # Import base class

'''
Control-based algorithms, currently the controlled particle filter (CPF)
'''


class CPF(Algorithm):
    
    def __init__(self, optimizer=None, **kwargs):

        super(CPF, self).__init__(optimizer, **kwargs)

        self._beta        = kwargs.get('beta', 1.0)
        self._noise_std   = kwargs.get('noise', 0.0)
        
        self._poisson      = kwargs.get('poisson','galerkin') # Scheme to solve Poisson equation
        ## Parameters for Fourier Galerkin
        self._basis_type   = kwargs.get('basis_type', 'Coordinate') # Type of basis want to use: Fourier_2D, et al.
        ## Parameters for kernel-based method        
        self._bw_type      = kwargs.get('bw_type', 'fixed') # Type of kernel bandwidth
        self._bw_fixed     = kwargs.get('bw_fixed', 1) # Value of bandwidth if constant
                
        self._phi          = np.zeros((self.world.num_times, self.N)) # Used in  the kernel method
        
        # Three numerical schemes to compute the control function
        if self._poisson == 'affine':
            self._poisson_solver = Affine(self.d, self.N, **kwargs)
        elif self._poisson == 'galerkin':
            self._poisson_solver = Galerkin(self.d, self.N, **kwargs)
        elif self._poisson == 'kernel':
            self._poisson_solver = Kernel(self, self.d, self.N, self._bw_type, self._bw_fixed)
        
    def run(self, X, dt, TI):
        ''' Run the CPF algorithm
            Arguments:
                + X  : Input state/particles
                + dt : Time step
                + TI : Time index
        '''

        if TI % 100 == 0:
            print 'TI=' + str(TI)
  
        h  = self.obj_fn.model.h(X)
        
        self.performance(np.mean(X, 1), h, TI)
            
        ## Compute control function using one of the three methods
        hdiff = h - np.mean(h)
        if self._poisson == 'affine':
            u = self._poisson_solver.solve(X, -hdiff, TI)
        if self._poisson == 'galerkin':
            u = self._poisson_solver.solve(X, -hdiff, TI)  
        elif self._poisson == 'kernel':       
            [phi, u] = self._poisson_solver.solve(X, -hdiff, TI, dt)
            self._phi[TI] = phi

        ## Add diffusion noise
        if self._noise_std > 0:
            noise_cov = np.diag(np.array([self._noise_std]*self.d))
            noise = np.random.multivariate_normal([0]*self.d, dt*noise_cov, self.N).T
        else:
            noise = 0
                
        return X + self._beta*u*dt + noise
        
        
class Affine(object):
    ''' The Affine class implements the affine scheme to solve the Poisson equation in CPF
    '''
    
    def __init__(self, d, N, **kwargs):
        
        self.d  = d
        self.N  = N
        self.kwargs = kwargs
        self.L  = self.d # Number of 2nd-order polynomial basis
        if self.d > 1 and self.d <= 50:
            self.L += math.factorial(self.d)/math.factorial(2)/math.factorial(self.d-2) 
        
    def solve(self, X, hdiff, TI):
        
        ## Compute B as constant control
        B = np.mean(X*hdiff, 1)
        B = np.reshape(B, (self.d,1))
        
        ## Compute K via Galerkin using second-order centered polynomials
        mean = np.reshape(np.mean(X, 1), (self.d,1))
        X_diff = X - mean

        Phi = np.zeros((self.L, self.N))
        gradPhi = np.zeros((self.d, self.L, self.N))        

        count = 0
        ## Basis x_n^2
        for n in range(self.d):
            Phi[count] = (X_diff[n])**2
            gradPhi[n, count] = 2*(X_diff[n])
            count += 1
        ## Basis x_n*x_m (m!=n)
        if self.d > 1 and self.d <= 50: # For high dimensional problem, do not use cross basis for efficiency
            for n in range(self.d):
                if n < self.d-1:
                    for m in range(n+1,self.d):
                        Phi[count] = X_diff[n]*X_diff[m]
                        gradPhi[n,count] = X_diff[m]
                        gradPhi[m,count] = X_diff[n]
                        count += 1            
        
        Psi = Phi
        gradPsi = gradPhi
        
        A       = np.zeros((self.L, self.L))
        b       = np.zeros((self.L, 1))
        K_vec   = np.zeros_like(b) 
        K       = np.zeros_like(X) 
        for i in range(0,self.L):
            b[i] = np.dot(hdiff,Psi[i])/float(self.N)
            for j in range(0,self.L):
                for d in range(0,self.d):
                    summ = np.sum(np.multiply(gradPsi[:,i,:], gradPhi[:,j,:]))
                    A[i,j] = summ/float(self.N)        
        K_vec = np.linalg.solve(A,b)
        
        for k in range(0,self.d):
            K[k] = np.dot(K_vec.T,gradPhi[k])

        if TI < 0:
            print 'B ' + str(B[0,0])           
            print 'K ' + str(np.max(K,1))
                    
        return K + B
        

class Kernel(object):
    ''' The Kernel class implements the kernel scheme to solve the Poisson equation in CPF
    '''
    
    def __init__(self, optimizer, d, N, bw_type, bw_fixed):
        
        self.optimizer = optimizer
        self.d  = d
        self.N  = N     
        self._bw_type = bw_type
        self.bw = bw_fixed # User-defined value, as constant or initial value for bw
            
        self.W  = np.zeros((self.N, self.N)) # Weight matrix
        self.A  = np.zeros_like(self.W) # A matrix, has row sum 1
        self.L  = np.zeros_like(self.W)
        
    def solve(self, X, hdiff, TI, dt):

        bw = self.epsilon(X, TI, dt)

        ## Unnormalized weight matrix
        W_uw = np.zeros((self.N, self.N)) 
        for i in range(self.N):
            W_uw[i,i] = self.Gauss_kernel(X[:,i], X[:,i], bw)
            for j in range(i+1, self.N):
                W_uw[i,j] = self.Gauss_kernel(X[:,i], X[:,j], bw)
                W_uw[j,i] = W_uw[i,j]
        summ = np.sum(W_uw, 1) # Sum over columns
        
        ## Weighted matrix
        for i in range(self.N):
            self.W[i,i] = W_uw[i,i]/(summ[i])
            for j in range(i+1, self.N):
                self.W[i,j] = W_uw[i,j]/(np.sqrt(summ[i])*np.sqrt(summ[j]))
                self.W[j,i] = self.W[i,j]
                
        ## Matrix A
        sum_W = np.sum(self.W, 1)
        for i in range(self.N):
            self.A[i] = self.W[i]/sum_W[i]
        
        phi = self.solve_phi(self.A, hdiff, bw, TI) # shape (N,), same as hdiff
        phi = phi - np.mean(phi) # Zero mean
        
        S3  = self.A*(phi + bw*hdiff) # (N,N)
        
        ## Solve grad_phi for each particle
        K   = np.zeros((self.d,self.N))
        
        for n in range(self.d):
            S1 = np.sum(S3*X[n], 1) # (N,), evaluated at each particle
            S2 = np.sum(self.A*X[n], 1)*np.sum(S3,1)
            K[n] = (S1 - S2)/(2*bw)
                
        return [phi, K]
        
    def epsilon(self, X, TI, dt):
                
        if self._bw_type == 'constant': ## Including piecewise constant with jumps
            epsilon = self.bw
            if TI > 40000:
                epsilon = 0.1
                
        elif self._bw_type == 'variance': ## Used for single mode distribution        
            X_mean  = np.reshape(np.mean(X, 1), (self.d,1))
            diff    = X - X_mean
            cov = np.dot(diff, diff.T)/float(self.N-1)                
            epsilon = 1.0*max([cov[i,i] for i in range(self.d)])
            thresh = 0.5
            if epsilon < thresh:
                epsilon = thresh

        elif self._bw_type == 'exponential': ## May increase long-term convergence speed, but needs fine tuning
            t = dt*TI
            tau = 0.2
            thresh = 0.05
            epsilon = self.bw*np.exp(-tau*t)
            if epsilon < thresh:
                epsilon = thresh

        if TI < 0:
            print epsilon

        return epsilon
                                
    def Gauss_kernel(self, X, Y, bw):
        '''Gaussian RBF Kernel in Euclidean space.
           Arguments:
               + X, Y : two points, (d,)
               + bw   : kernel bandwidth
        '''
        diff = X - Y
        d2   = np.dot(diff,diff)
        k    = np.exp(-d2/(4*bw))
        
        if k < 1e-300:
            k = 0
        
        return k
        
    def solve_phi(self, A, hdiff, bw, TI):
        '''Solve the phi vector from the matrix equation
           equation: (I-A)*\phi = \epsilon*h
           Arguments:
               + A : the A matrix
               + h : RHS of the equation
        '''
        if TI == 0:
            phi = np.zeros(self.N)
            N_terminate = 500 
            for i in range(N_terminate):
                phi = np.dot(A, phi) + bw*hdiff
                phi = phi - np.mean(phi) # Zero mean
        else: # Use previous phi as initial guess
            phi = self.optimizer._phi[TI-1]
            N_terminate = 10
            for i in range(N_terminate):
                phi = np.dot(A, phi) + bw*hdiff
                phi = phi - np.mean(phi) # Zero mean

        return phi
        

class Galerkin(object):
    ''' The Galerkin class implements the Galerkin scheme to solve the Poisson equation in CPF
    '''

    def __init__(self, d, N, **kwargs):
        
        self.d  = d
        self.N  = N
        self._basis_type = kwargs['basis_type']
        self.L  = None # Total number of basis

        self.kwargs = kwargs
                            
    def solve(self, X, hdiff, TI):

        [self.L, Phi, gradPhi] = getattr(self, self._basis_type)(X, self.kwargs)

        Psi = Phi
        gradPsi = gradPhi

        # Compute A and b
        A       = np.zeros((self.L, self.L))
        b       = np.zeros((self.L, 1))
        K_vec   = np.zeros_like(b) 
        K       = np.zeros_like(X) 
        for i in range(0,self.L):
            b[i] = np.dot(hdiff,Psi[i])/float(self.N)
            for j in range(0,self.L):
                for d in range(0,self.d):
                    summ = np.sum(np.multiply(gradPsi[:,i,:], gradPhi[:,j,:]))
                    A[i,j] = summ/float(self.N)

        K_vec = np.linalg.solve(A,b)
                
        for k in range(0,self.d):
            K[k] = np.dot(K_vec.T,gradPhi[k])

        return K 
        
        
    def Coordinate(self, L):
        '''Multidimensional coordinate basis
        '''
    
        basisfn = [lambda X, i=i: X[i] for i in range(L)]
        basisfn_grad = []
    
        for i in range(L):
            basisfn_grad.append([lambda X, k=k: 0 for k in range(L)])

        for m in range(L):
            basisfn_grad[m][m] = lambda X: 1
               
        return [len(basisfn), basisfn, basisfn_grad]
        
        
    def Fourier(self, X, kwargs):
        '''
        Multidimensional Fourier basis. states are DECOUPLED, rendering A matrix sparser. 
        Parameters: 
            + X  : Particles, (d,N)
            + Tf : Period of Fourier basis
            + Lf : Number of Fourier basis
        '''
        Lf = kwargs['Lf']
        Tf = kwargs['Tf']
        
        L = self.d + Lf

        Phi = np.zeros((L, self.N))
        gradPhi = np.zeros((self.d, L, self.N))
        
        count = 0        
        ## Add Coordinate basis
        for i in range(self.d):
            Phi[i] = X[i]
            gradPhi[i,i] = np.ones(self.N,)
        count += self.d
        
        ## Add Fourier basis
        period = [float(Tf) for i in range(self.d)]
        
        mode = 1
        while mode >=1:
            if count == L:
                break
            for i in range(self.d):
                C = np.cos(2*np.pi*mode*X[i]/period[i])
                S = np.sin(2*np.pi*mode*X[i]/period[i])
                Phi[count]   = C
                Phi[count+1] = S
                
                gradPhi[i, count]   = -2*np.pi*mode*S/period[i]
                gradPhi[i, count+1] = 2*np.pi*mode*C/period[i]
                
                count += 2
                
            mode += 1
                    
        return [L, Phi, gradPhi]

        
    def Salomon(self, X, kwargs):
        '''Radial basis, only used for Salomon function
        '''
        T = kwargs['Tf']
        ## Necessary parts (avoid repeating calculation)
        r = np.apply_along_axis(la.norm, 0, X)
        C = np.cos(2*np.pi*r/T)
        S = np.sin(2*np.pi*r/T)
        
        ## Add basis functions
        Phi = r
        Phi = np.vstack((Phi, C))      
        Phi = np.vstack((Phi, S))
        
        L = Phi.shape[0] ## Total number of basis functions
        
        ## Add gradient of basis
        gradPhi = np.zeros((self.d, L, self.N))
        for i in range(self.d):
            q = X[i]/r
            gradPhi[i,0] = q            
            gradPhi[i,1] = -2*np.pi*q*S/T
            gradPhi[i,2] = 2*np.pi*q*C/T
        
        return [L, Phi, gradPhi]
        
        
 










        
        
        
    
        
        
        

