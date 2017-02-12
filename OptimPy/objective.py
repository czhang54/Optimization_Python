#!/usr/bin/python

import numpy as np

''' Define all types of objective functions. 
    All objective functions have the variable:
        + self.h_min: minimum value of the function
        + self.x_min: location of the global minimizer
    All objective functions have the member function:
        + self.h(X): evaluate the function at X, where X can be a state or a set of particles
    Some objective functions have the member functions to als compute its gradient:
        + self.gradient(X): Compute gradient at X
        + self.evaluate(X): Compute both h and its gradient at X
    Multi-dimensional functions also have some parameters to generate 3D-plot:
        + self.x_l: left boundary of plot
        + self.x_r: right boundary of plot
        + self.y_b: bottom boundary of plot
        + self.y_t: top boundary of plot
        + self.h_2D(): Evaluate function on 2D domain 
        
'''
                
        
class Double_well(object):
    '''1-D Double-well function, 1D only
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        self._a      = kwargs.get('a', 1.0)
        self._b      = kwargs.get('b', -8.0)
        self._c      = kwargs.get('c', -0.5)
        self._d      = kwargs.get('d', self.b**2/4.0/self.a)
        
        self.h_min  = -1.00387
        self.x_min  = np.array([2.015])
        
    def evaluate(self, X, solve_grad, **kwargs):
        h = self._a*X[0]**4 + self._b*X[0]**2 + self._c*X[0] + self._d
        
        grad_h = np.zeros_like(X)
        if solve_grad:
            grad_h = 4.0*self._a*X[0]**3 + 2.0*self._b*X[0] + self._c
        
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        if len(X.shape) == 2: # X is of shape (N,1)
            return self._a*X[0]**4 + self._b*X[0]**2 + self._c*X[0] + self._d
        else: # X is of shape (N,)
            return self._a*X**4 + self._b*X**2 + self._c*X + self._d
        
    def gradient(self, X):
        return 4.0*self._a*X**3 + 2.0*self._b*X + self._c
        
        
class Ackley(object):

    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([0]*self.optimizer.dim)
        
        self.x_l    = -10
        self.x_r    = 10
        self.y_b    = -10
        self.y_t    = 10
        self.space  = 0.1
        
    def evaluate(self, X, solve_grad, **kwargs):
        d = float(X.shape[0])
        norm2 = np.apply_along_axis(np.linalg.norm, 0, X)**2
        sqr = np.sqrt(norm2/d)
        C = np.cos(2*np.pi*X)
        
        h = -20*np.exp(-0.02*np.sqrt(norm2/d)) - np.exp(np.sum(C,0)/d) + 20 + np.e
        
        grad_h = np.zeros_like(X)
        if solve_grad:
            for i in range(X.shape[0]):
                grad_1  = 0.4*np.exp(-0.02*sqr)*X[i]/sqr/d
                grad_2  = np.exp(np.sum(C,0)/d)*np.sin(2*np.pi*X[i])/d
                grad_h[i] = grad_1 + grad_2
            
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        d = float(X.shape[0])
        sqr = np.sqrt(np.apply_along_axis(np.linalg.norm, 0, X)**2/d) # (N,)
        return -20*np.exp(-0.02*sqr) - np.exp(np.sum(np.cos(2*np.pi*X),0)/d) + 20 + np.e # (N,)
        
    def gradient(self, X):
        d = X.shape[0]
        gradient = np.zeros_like(X)
        sqr = np.sqrt(np.apply_along_axis(np.linalg.norm, 0, X)**2/d) # (N,)
        for i in range(X.shape[0]):
            gradient_1  = 0.4*np.exp(-0.02*sqr)*X[i]/sqr/d
            gradient_2  = np.exp(np.sum(np.cos(2*np.pi*X),0)/d)*np.sin(2*np.pi*X[i])/d
            gradient[i] = gradient_1 + gradient_2
            
        return gradient  
        
    def h_2D(self, X, Y):
        return -20*np.exp(-0.02*np.sqrt(0.5*(X**2+Y**2)))-np.exp(0.5*(np.cos(2*np.pi*X)+np.cos(2*np.pi*Y)))+20+np.e
        

class Camelback(object): # 2D only
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
    def h(self, X, **kwargs):
        return (4-2.1*X[0]**2+X[0]**4/3.0)*X[0]**2 + X[0]*X[1] + 4*(X[1]**2-1)*X[1]**2
        
    def gradient(self, X):
        gradient = np.zeros_like(X)
        gradient[0] = 8*X[0] - 8.4*X[0]**3 + 2*X[0]**5 + X[1]
        gradient[1] = X[0] + 16*X[1]**3 - 8*X[1]           
        return gradient
        
    def plot_2D(self):
        X = np.arange(-3.0, 3.0, 0.1)
        Y = np.arange(-2.0, 2.0, 0.1)
        h = lambda X, Y: (4-2.1*X**2+1/3.0*X**4)*X**2 + X*Y + 4*(Y**2-1)*Y**2
        return [X, Y, h]
        
        
class Dejong5(object): ## Only 2D
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0.998
        self.x_min  = np.array([-32]*self.optimizer.dim)
        self.a      = np.array([[-32,-16,0,16,32]*5, [-32]*5+[-16]*5+[0]*5+[16]*5+[32]*5])

        self.x_l    = -50
        self.x_r    = 50
        self.y_b    = -50
        self.y_t    = 50
        self.space  = 0.5
        
    def h(self, X, **kwargs):
        den = 0 
        for j in range(25):
            X_diff = X - np.reshape(self.a[:,j], (2,1))
            den   += 1/(j + 1 + X_diff[0]**6 + X_diff[1]**6)
            
        return 1/(den + 0.002)
        
    def h_2D(self, X, Y):
        den = 0
        for j in range(25):
            den   += 1/(j + 1 + (X-self.a[0,j])**6 + (Y-self.a[1,j])**6)
            
        return 1/(den + 0.002)     


class Griewank(object):
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([0]*self.optimizer.dim)
        
        self.x_l    = -10
        self.x_r    = 10
        self.y_b    = -10
        self.y_t    = 10
        self.space  = 0.1 
        
    def evaluate(self, X, solve_grad, **kwargs):
        norm = np.apply_along_axis(np.linalg.norm, 0, X)        
        prod = np.ones(X.shape[1],)
        for i in range(X.shape[0]):
            prod *= np.cos(X[i]/np.sqrt(i+1))
        
        h = norm*2/4000.0 - prod + 1
        
        grad_h = np.zeros_like(X)
        if solve_grad:
            for i in range(X.shape[0]):
                grad_h[i] = X[i]/2000.0 + prod*np.sin(X[i]/np.sqrt(i+1))/prod[i]/np.sqrt(i+1)
            
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        norm = np.apply_along_axis(np.linalg.norm, 0, X)
        prod = 1
        for i in range(X.shape[0]):
            prod = prod*np.cos(X[i]/np.sqrt(i+1))
        return norm**2/4000.0 - prod + 1
        
    def gradient(self, X):
        C = np.zeros_like(X)
        for i in range(X.shape[0]):
            C[i] = np.cos(X[i]/np.sqrt(i+1))
        prod = np.prod(C, 0)
        gradient = np.zeros_like(X)
        for i in range(X.shape[0]):
            gradient[i] = X[i]/2000.0 + prod*np.sin(X[i]/np.sqrt(i+1))/C[i]/np.sqrt(i+1)
        return gradient
        
    def h_2D(self, X, Y):
        return 1 + (X**2+Y**2)/4000.0 - np.cos(X)*np.cos(Y/np.sqrt(2))
        
        
class Levy(object):

    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([1.0]*self.optimizer.dim)
        
        self.x_l    = -10
        self.x_r    = 10
        self.y_b    = -10
        self.y_t    = 10
        self.space  = 0.1 
        
    def h(self, X, **kwargs):
        
        d = X.shape[0]
        W  = 1 + (X-1)/4.0
        S1 = np.sin(np.pi*W[0])**2
        S2 = (W[d-1]-1)**2*(1+np.sin(2*np.pi*W[d-1])**2)
        S3 = 0
        for i in range(d-1):
            S3 += (W[i]-1)**2*(1+10*np.sin(np.pi*W[i]+1)**2)
            
        return S1 + S2 + S3
        
    def h_2D(self, X, Y):
        
        X1 = 1 + (X-1)/4.0
        Y1 = 1 + (Y-1)/4.0
        S1 = np.sin(np.pi*X1)**2 
        S2 = (Y1-1)**2*(1+np.sin(2*np.pi*Y1)**2)
        S3 = (X1-1)**2*(1+10*np.sin(np.pi*X1+1)**2)
        
        return S1 + S2 + S3


class Periodic(object): 
        
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0.9
        self.x_min  = np.array([0]*self.optimizer.dim)
        
        self.x_l    = -10
        self.x_r    = 10
        self.y_b    = -10
        self.y_t    = 10
        self.space  = 0.1 
        
    def evaluate(self, X, solve_grad, **kwargs):
        norm2 = np.apply_along_axis(np.linalg.norm, 0, X)**2
        
        h = 1 + np.sum((np.sin(X)**2),0) - 0.1*np.exp(-norm2)
        
        grad_h = np.zeros_like(X)
        if solve_grad:
            for i in range(X.shape[0]):
                grad_h[i] = np.sin(2*X[i]) + 0.2*X[i]*np.exp(-norm2)
            
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        return 1 + np.sum((np.sin(X)**2),0) - 0.1*np.exp(-np.apply_along_axis(np.linalg.norm, 0, X)**2)
        
    def gradient(self, X):
        gradient = np.zeros_like(X)
        for i in range(X.shape[0]):
            gradient[i] = 2*np.sin(X[i])*np.cos(X[i]) + 0.2*X[i]*np.exp(-np.apply_along_axis(np.linalg.norm, 0, X)**2)
        return gradient
        
    def h_2D(self, X, Y):
        return 1 + (np.sin(X))**2 + (np.sin(Y))**2 - 0.1*np.exp(-(X**2+Y**2))
        
        
class Pinter(object):

    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([0]*self.optimizer.dim)
        
        self.x_l    = -10
        self.x_r    = 10
        self.y_b    = -10
        self.y_t    = 10
        self.space  = 0.1                
        
    def h(self, X, **kwargs):
        d = X.shape[0] ## ASSUME d > 1
        summ = 0
        for i in range(d):
            S1 = (i+1)*X[i]**2
            if i == 0:
                S2 = 20*(i+1)*(np.sin( X[d-1]*np.sin(X[i]) - X[i] + np.sin(X[i+1]) ))**2
                S3 = (i+1)*np.log10( 1 + (i+1)*(X[d-1]**2-2*X[i]+3*X[i+1]-np.cos(X[i])+1)**2 )
            elif i == d-1:
                S2 = 20*(i+1)*(np.sin( X[i-1]*np.sin(X[i]) - X[i] + np.sin(X[0]) ))**2
                S3 = (i+1)*np.log10( 1 + (i+1)*(X[i-1]**2-2*X[i]+3*X[0]-np.cos(X[i])+1)**2 )    
            else:
                S2 = 20*(i+1)*(np.sin( X[i-1]*np.sin(X[i]) - X[i] + np.sin(X[i+1]) ))**2
                S3 = (i+1)*np.log10( 1 + (i+1)*(X[i-1]**2-2*X[i]+3*X[i+1]-np.cos(X[i])+1)**2 )                
            summ = summ + S1 + S2 + S3
        
        return summ
        
    def h_2D(self, X, Y):
        
        return X**2+2*Y**2 + 20*np.sin( Y*np.sin(X) - X + np.sin(Y) )**2 + 20*2*np.sin( X*np.sin(Y) - Y + np.sin(X) )**2
        + np.log10( 1 + (Y**2-2*X+3*Y-np.cos(X)+1)**2 ) + np.log10( 1 + 2*(X**2-2*Y+3*X-np.cos(Y)+1)**2 )
        
        
class Powell(object): 
    ''' Dimensional must be at least 4
    '''

    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([0]*self.optimizer.dim)
        
    def h(self, X, **kwargs):
        summ = 0
        for i in range(1,X.shape[0]-2):
            summ += (X[i-1]+10*X[i])**2 + 5*(X[i+1]-X[i+2])**2 + (X[i]-2*X[i+1])**4 + 10*(X[i-1]-X[i+2])**4
        
        return summ
        
        
class Quadratic(object):
    '''Simplest symmetric quadratic function
    '''
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        self.dim    = self.optimizer.dim
        self._H      = kwargs.get('H', np.eye(self.dim))
        self.h_min  = 0
        self.x_min  = np.array([0]*self.dim)
        
    def evaluate(self, X, solve_grad, **kwargs):
        h = np.zeros(X.shape[1],)
        grad_h = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = np.reshape(X[:,i], (self.dim,1))
            h[i] = 0.5*np.dot(col.T, np.dot(self.H, col))
        if solve_grad:
            for i in range(X.shape[1]):
                grad_h[:,i] = self._H*X[:,i] + self.H.T*X[:,i]
            
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        h = np.zeros(X.shape[1],)
        for i in range(X.shape[1]):
            col = np.reshape(X[:,i], (self.dim,1))
            h[i] = 0.5*np.dot(col.T, np.dot(self.H, col))
            
        return h
        
    def h_2D(self, X, Y):
        return 0.5*(X**2 + Y**2)
        
        
class Rastrigin(object): # n-dimensional
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([0]*self.optimizer.dim)
        
        self.x_l    = -3
        self.x_r    = 3
        self.y_b    = -3
        self.y_t    = 3
        self.space  = 0.02
        
    def evaluate(self, X, solve_grad, **kwargs):
        norm = np.apply_along_axis(np.linalg.norm, 0, X)
        
        h = 10*X.shape[0] + norm**2 - 10*np.sum(np.cos(2*np.pi*X),0)
        
        grad_h = np.zeros_like(X)
        if solve_grad:
            for i in range(X.shape[0]):
                grad_h[i] = 2*X[i] + 20*np.pi*np.sin(2*np.pi*X[i]) 
        
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        return 10*X.shape[0] + np.apply_along_axis(np.linalg.norm, 0, X)**2 - 10*np.sum(np.cos(2*np.pi*X), 0)
        
    def gradient(self, X):
        gradient = np.zeros_like(X)
        for i in range(X.shape[0]):
            gradient[i] = 2*X[i] + 20*np.pi*np.sin(2*np.pi*X[i])            
        return gradient
        
    def h_2D(self, X, Y):
        return 20 + X**2 + Y**2 - 10*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))
        
        
class Rosenbrock(object): # n-dimensional
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([1]*self.optimizer.dim)
        
        self.x_l    = -2
        self.x_r    = 2
        self.y_b    = -1
        self.y_t    = 3
        self.space  = 0.1
        
    def evaluate(self, X, solve_grad, **kwargs):
        h = np.zeros(X.shape[1],)
        for i in range(X.shape[0]-1):
            h += 100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2
        
        grad_h = np.zeros_like(X)
        
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        h = 0
        for i in range(X.shape[0]-1):
            h += 100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2        
            
        return h
        
    def h_2D(self, X, Y):
        return 100*(Y-X**2)**2 + (X-1)**2
        
        
class Salomon(object):
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0
        self.x_min  = np.array([0.0]*self.optimizer.dim)
        
        self.x_l    = -15
        self.x_r    = 15
        self.y_b    = -15
        self.y_t    = 15
        self.space  = 0.1
        
    def evaluate(self, X, solve_grad, **kwargs):
        norm = np.apply_along_axis(np.linalg.norm, 0, X)
        
        h = 1 - np.cos(2*np.pi*norm) + 0.1*norm
        
        grad_h = np.zeros_like(X)
        if solve_grad:
            for i in range(X.shape[0]):
                q = X[i]/norm
                grad_h[i] = 2*np.pi*np.sin(2*np.pi*norm)*q + 0.1*q
            
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        norm = np.apply_along_axis(np.linalg.norm, 0, X)
        return 1 - np.cos(2*np.pi*norm) + 0.1*norm
        
    def gradient(self, X):
        gradient = np.zeros_like(X)
        norm = np.apply_along_axis(np.linalg.norm, 0, X)
        for i in range(X.shape[0]):
            q = X[i]/norm
            gradient[i] = 2*np.pi*np.sin(2*np.pi*norm)*q + 0.1*q            
        return gradient
        
    def h_2D(self, X, Y):
        return 1 - np.cos(2*np.pi*np.sqrt(X**2+Y**2)) + 0.1*np.sqrt(X**2+Y**2)
        
        
class Sinusoidal(object):
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 1.0
        
        self.x_l    = 0
        self.x_r    = 180
        self.y_b    = 0
        self.y_t    = 180
        self.space  = 1
        
    def h(self, X, **kwargs):
        
        prod1 = 1
        prod2 = 1
        for i in range(X.shape[0]):
            prod1 *= np.sin(np.pi*X[i]/180.0)
            prod2 *= np.sin(np.pi*X[i]/36.0)
        
        return -2.5*prod1 - prod2 + 3.5
        
    def h_2D(self, X, Y):
        
        return -2.5*np.sin(np.pi*X/180.0)*np.sin(np.pi*Y/180.0) - np.sin(np.pi*X/36.0)*np.sin(np.pi*Y/36.0) + 3.5
        
        
class Sphere_W(object):
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 1.0
        
        self.x_l    = -10
        self.x_r    = 10
        self.y_b    = -10
        self.y_t    = 10
        self.space  = 0.1  
        
    def h(self, X, **kwargs):
        
        S = 0        
        for i in range(X.shape[0]):
            S += (i+1)*X[i]**2

        return S

    def h_2D(self, X, Y):
        return X**2 + 2*Y**2            


class Trigonometric(object):
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 0.0
        self.x_min  = np.array([0.9]*self.optimizer.dim)
        
        self.x_l    = -5
        self.x_r    = 5
        self.y_b    = -5
        self.y_t    = 5
        self.space  = 0.05
        
    def evaluate(self, X, **kwargs):
        S1 = 8*(np.sin(7*(X-0.9)**2))**2
        S2 = 6*(np.sin(14*(X-0.9)**2))**2
        S3 = (X-0.9)**2
        
        h  = 0 + np.sum(S1+S2+S3,0)

        grad_h = np.zeros_like(X)
        for i in range(X.shape[0]):
            grad_h[i] = 8*np.sin(14*(X[i]-0.9)**2)*14*X[i] + 6*np.sin(28*(X[i]-0.9)**2)*28*X[i] + 2*X[i]  
            
        return [h, grad_h]
        
    def h(self, X, **kwargs):
        S1 = 8*(np.sin(7*(X-0.9)**2))**2
        S2 = 6*(np.sin(14*(X-0.9)**2))**2
        S3 = (X-0.9)**2
        return 0 + np.sum(S1+S2+S3,0)
        
    def gradient(self, X):
        gradient = np.zeros_like(X)
        for i in range(X.shape[0]):
            gradient[i] = 8*np.sin(14*(X[i]-0.9)**2)*14*X[i] + 6*np.sin(28*(X[i]-0.9)**2)*28*X[i] + 2*X[i]            
        return gradient
        
    def h_2D(self, X, Y):
        return 0 + 8*(np.sin(7*(X-0.9)**2))**2 + 6*(np.sin(14*(X-0.9)**2))**2 + (X-0.9)**2 + 8*(np.sin(7*(Y-0.9)**2))**2 + 6*(np.sin(14*(Y-0.9)**2))**2 + (Y-0.9)**2


class Zakharov(object): # No local minima
    
    def __init__(self, optimizer=None, **kwargs):
        self.optimizer    = optimizer
        
        self.h_min  = 1.0
        self.x_min  = np.array([0.0]*self.optimizer.dim)
        
        self.x_l    = -1.5
        self.x_r    = 1.5
        self.y_b    = -1.5
        self.y_t    = 1.5
        self.space  = 0.02

    def h(self, X, **kwargs):
        
        norm = np.apply_along_axis(np.linalg.norm, 0, X)
        S = 0
        for i in range(X.shape[0]):
            S += 0.5*(i+1)*X[i] 

        return norm**2 + S**2 + S**4
        
    def h_2D(self, X, Y):
        
        S = 0.5*X + 0.5*2*Y
        return X**2 + Y**2 + S**2 + S**4


















