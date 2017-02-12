"""
Created on Mon Dec 26 11:41:15 2016

@author: Chi Zhang, UIUC
"""

'''
This program implements and compares a series of global optimization algorithms for user-defined objective functions. 
The program is flexible for the user to add:
    (1) objective functions
    (2) initialization methods for the optimizers
    (3) specific algorithms of the optimizer
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

np.seterr(all='raise')

sys.path.append('../') # Locate the OptimPy package using relative path
import OptimPy
print "You are using source code from:"
print OptimPy.__file__

from OptimPy.container import World, Item, Optimizer, Obj_fn
from OptimPy.prior import *
from OptimPy.objective import *
from OptimPy.control_based import CPF
from OptimPy.model_based_parametric import CE, MRAS, MEO
from OptimPy.model_based_nonparametric import PFO, SMC_SA, SISR
from OptimPy.swarm import PSO


'''
Define the world (simulation settings)
'''
start = 0 # Start time
stop  = 10.2 # Terminate time
step  = 0.1 # Time step, depends on the function to be minimized
time  = np.arange(start, stop, step) # Time instants
dim   = 2 # Dimension of the problem
W = World( dim=dim, time=time )

N = 100 # Number of particles


'''
Objective functions
'''
## (function name, global minimizer, minimum function value)
obj_dict = {}
obj_dict[1]  = ('Ackley', [0]*dim, 0) # CPF need dt=0.01
obj_dict[2]  = ('Dejong5', [-32]*dim, 0.998) # HARD, CPF need dt=0.001
obj_dict[3]  = ('Griewank', [0]*dim, 0) # HARD
obj_dict[4]  = ('Levy', [1]*dim, 0) # CPF need dt=0.001
obj_dict[5]  = ('Periodic', [0]*dim, 0.9) # HARD
obj_dict[6]  = ('Pinter', [0]*dim, 0) # CPF need dt=0.0001
obj_dict[7]  = ('Powell', [0]*dim, 0) # Very steep, CPF need dt=0.0001
obj_dict[8]  = ('Quadratic', [0]*dim, 0) # CPF need dt=0.01
obj_dict[9]  = ('Rastrigin', [0]*dim, 0) # CPF need dt=0.0001
obj_dict[10] = ('Rosenbrock', [1]*dim, 0) # Very steep
obj_dict[11] = ('Salomon', [0]*dim, 0) # CPF need dt=0.1
obj_dict[12] = ('Sinusoidal', [90]*dim, 0)
obj_dict[13] = ('Trigonometric', [0.9]*dim, 0) # CPF need dt=0.0001
obj_dict[14] = ('Zakharov', [0]*dim, 0) # Very steep, CPF need dt=0.0001

(obj_name, x_min, h_min) = obj_dict[1]

## Define objective function
item_OBJ = W.add( Item('Obj_fn') )
obj_fn = item_OBJ.add( Obj_fn(dim=dim) )
obj_fn.setObjective( model=eval(obj_name) )
    
print 'You are optimizing %s function' % (obj_name)


'''
Select algorithms
'''
## Will run the algorithm if it is 'on'
algorithms = {}
algorithms['CE']     = ['on', 'off'][0] # Cross-entropy
algorithms['MRAS']   = ['on', 'off'][0] # Model reference adaptive search
algorithms['MEO']    = ['on', 'off'][0] # Model-based evolutionary optimization
algorithms['PSO']    = ['on', 'off'][0] # Particle swarm optimization
algorithms['PFO']    = ['on', 'off'][0] # Particle filtering for optimization
algorithms['SMC-SA'] = ['on', 'off'][0] # Sequential Monte Carlo simulated annealing
algorithms['SISR']   = ['on', 'off'][0] # Sequential importance sampling and resampling
algorithms['CPF-A']  = ['on', 'off'][0] # Controlled particle filter with affine control law
algorithms['CPF-G']  = ['on', 'off'][0] # Controlled particle filter with Galerkin control law
algorithms['CPF-K']  = ['on', 'off'][0] # Controlled particle filter with kernel-based control law


'''
Initilization methods and parameters
'''

initial_dist = ['Uniform', 'Gaussian'][1]
# Uniform
interval = [-50, 50]
# Gaussian
MU = [50]*dim
sigma = np.sqrt(500)
SIGMA = [sigma]*dim

# Generate initial particles and used by all the optimizers
IC = eval(initial_dist)( dim=dim, interval=interval, IC=MU, sigma=SIGMA )
P = IC.f(N)


'''
Algorithm parameters
'''
rho = 0.1 ## Quantile for selecting elite samples, used in CE, MRAS, MEO, PFO
smooth_param = 0.20 ## Smooting parameter used in CE, MRAS, MEO to prevent premature converegence
noise_type   = ['constant', 'decay'][1] # Noise is injected in PFO, SMC-SA, SISR
const_noise_std = 10 ## Std. of constant noise
init_noise_std  = 5 ## Initial std. of decaying noise

#### CPF parameters
# Galerkin parameters
basis_type = ['Fourier', 'Fourier_poly', 'Salomon'][0]
# Fourier basis parameters
mode = 1 # Highest mode in the Fourier basis; mode = 0 means constant basis
Lf = 2*mode*dim # Number of Fourier basis
Tf = 400 # Period of the basis
# Kernel-bandwidth parameters
bw_type  = ['constant', 'variance', 'exponential'][1]
bw_fixed = 0.5 # Used for constant bandwidth


'''
Add algorithms
'''
optimizer_list = []
optimizer_legend = []


#### Cross-entropy (CE) ####
IS_type = ['uniform', 'exponential', 'linear'][0]

item_CE = W.add( Item('CE') )
optimizer_CE = item_CE.add( Optimizer(N=N) )
optimizer_CE.setAlgorithm( model=CE, mu_0=MU, sigma_0=SIGMA, smooth_param=smooth_param, quantile=rho, IS_type=IS_type ) 
optimizer_CE.initialize()
optimizer_CE.value[0] = P

if algorithms['CE'] == 'on':
    optimizer_list.append(optimizer_CE)
    optimizer_legend.append('CE')


#### Model reference adaptive search (MRAS) ####

item_MRAS = W.add( Item('MRAS') )
optimizer_MRAS = item_MRAS.add( Optimizer(N=N) ) 
optimizer_MRAS.setAlgorithm( model=MRAS, mu_0=MU, sigma_0=SIGMA, smooth_param=smooth_param, quantile=rho, IS_type=IS_type ) 
optimizer_MRAS.initialize()
optimizer_MRAS.value[0] = P

if algorithms['MRAS'] == 'on':
    optimizer_list.append(optimizer_MRAS)
    optimizer_legend.append('MRAS')
    

#### Model-based evolutionary optimization (MEO) ####
item_MEO = W.add( Item('MEO') )
optimizer_MEO = item_MEO.add( Optimizer(N=N) )
optimizer_MEO.setAlgorithm( model=MEO, mu_0=MU, sigma_0=SIGMA, smooth_param=smooth_param, quantile=rho ) 
optimizer_MEO.initialize()
optimizer_MEO.value[0] = P

if algorithms['MEO'] == 'on':
    optimizer_list.append(optimizer_MEO)
    optimizer_legend.append('MEO')
    
    
#### Particle swarm optimization (PSO) ####
a = 0.5
b = a
w = 0.5
x = [100*step, 1][1]
v_l = [-3*SIGMA[i] for i in range(len(SIGMA))]
v_u = [3*SIGMA[i] for i in range(len(SIGMA))]

item_PSO = W.add( Item('PSO') )
optimizer_PSO = item_PSO.add( Optimizer(N=N) )
optimizer_PSO.setAlgorithm( model=PSO, a=a, b=b, w=w, x=x, v_l=v_l, v_u=v_u ) 
optimizer_PSO.initialize()
optimizer_PSO.value[0] = P

if algorithms['PSO'] == 'on':
    optimizer_list.append(optimizer_PSO)
    optimizer_legend.append('PSO') 
    
    
#### Particle filtering for optimization (PFO) ####
IS_type = ['uniform', 'exponential', 'linear'][0] # Used in PFO
elite_select = ['quantile', 'minimum'][0] # Used in PFO

item_PFO = W.add( Item('PFO') )
optimizer_PFO = item_PFO.add( Optimizer(N=N) )
optimizer_PFO.setAlgorithm( model=PFO, noise_type=noise_type, bw=const_noise_std, alpha=init_noise_std, quantile=rho, IS_type=IS_type ) 
optimizer_PFO.initialize()
optimizer_PFO.value[0] = P

if algorithms['PFO'] == 'on':
    optimizer_list.append(optimizer_PFO)
    optimizer_legend.append('PFO')
    

#### Sequential Monte-Carlo simulated annealing (SMC-SA) ####
item_SMCSA = W.add( Item('SMC_SA') )
optimizer_SMCSA = item_SMCSA.add( Optimizer(N=N) )
optimizer_SMCSA.setAlgorithm( model=SMC_SA, initial_dist=initial_dist, mu_0=MU, sigma_0=SIGMA, noise_type=noise_type, bw=const_noise_std, alpha=init_noise_std ) 
optimizer_SMCSA.initialize()    
optimizer_SMCSA.value[0] = P

if algorithms['SMC-SA'] == 'on':
    optimizer_list.append(optimizer_SMCSA)
    optimizer_legend.append('SMC-SA')
    
    
#### Sequential importance sampling resampling (SISR) ####
item_SISR = W.add( Item('SISR') )
optimizer_SISR = item_SISR.add( Optimizer(N=N) )
optimizer_SISR.setAlgorithm( model=SISR, noise_type=noise_type, bw=const_noise_std, alpha=init_noise_std ) 
optimizer_SISR.initialize()    
optimizer_SISR.value[0] = P

if algorithms['SISR'] == 'on':
    optimizer_list.append(optimizer_SISR)
    optimizer_legend.append('SISR') 
    

#### Controlled particle filter (CPF) ####
## CPF with affine control
item_CPF = W.add( Item('CPF') )
optimizer_a = item_CPF.add( Optimizer(N=N) )
optimizer_a.setAlgorithm( model=CPF, poisson='affine' )
optimizer_a.initialize()
optimizer_a.value[0] = P

if algorithms['CPF-A'] == 'on':
    optimizer_list.append(optimizer_a)
    optimizer_legend.append('CPF-A')

## CPF with Galerkin control
optimizer_g = item_CPF.add( Optimizer(N=N) )
optimizer_g.setAlgorithm( model=CPF, poisson='galerkin', basis_type=basis_type, Lf=Lf, Tf=Tf )
optimizer_g.initialize()
optimizer_g.value[0] = P

if algorithms['CPF-G'] == 'on':
    optimizer_list.append(optimizer_g)
    optimizer_legend.append('CPF-G')

## CPF with kernel control
optimizer_k = item_CPF.add( Optimizer(N=N) )
optimizer_k.setAlgorithm( model=CPF, poisson='kernel', bw_type=bw_type, bw_fixed=bw_fixed )
optimizer_k.initialize()
optimizer_k.value[0] = P

if algorithms['CPF-K'] == 'on':
    optimizer_list.append(optimizer_k)
    optimizer_legend.append('CPF-K')

    
print 'The following algorithms will be simulated: \t'
print ', '.join(optimizer_legend)

    
'''
Simulate all the algorithms
'''    
W.simulate(optimizer_list)


'''
Performance assessment
'''
L = len(optimizer_list)
TI_end = W.num_times - 2

print 'Final error: \t'
for i in range(L):
    print '%s: ' % optimizer_legend[i] + str(optimizer_list[i].algorithm.error[TI_end])

print 'Final h_best: \t'
for i in range(L):
    final_h_best = optimizer_list[i].algorithm.h_best[TI_end]
    print '%s: ' % optimizer_legend[i] + '%.6f' % (final_h_best) + '\t'

print 'Final h_hat: \t'
for i in range(L):
    final_h_hat = optimizer_list[i].algorithm.h_hat[TI_end]
    print '%s: ' % optimizer_legend[i] + '%.6f' % (final_h_hat) + '\t'



'''
Plots
'''
PROBLUE  = '#6E8BBF'
PRORANGE = '#EF8A1C'
f_size   = 16
l_size   = 24
lw = 3

x_l = -stop*0.02
x_r = stop*1.02

colors = [PROBLUE, PRORANGE, 'g', 'm', 'b', 'r', 'c', 'k', 'y', PROBLUE, PRORANGE, 'g', 'm', 'b', 'r', 'c', 'k', 'y']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

if W.num_times < 100:
    stride = 1 # Used for ploting particles
else:
    stride = int(len(time)/100)

#### Plot best function value as a function of time
init_max = []
plt.figure()
for i in range(L):
    h_data = optimizer_list[i].algorithm.h_best[0:-1]
    init_max.append(max(h_data))
    plt.plot(time[0:-1], h_data, color=colors[i], linestyle=line_styles[i], lw=lw, label=optimizer_legend[i])
plt.plot([x_l, x_r], [h_min, h_min], '--k', lw=2, alpha=0.5)
plt.xlim(start, time[-2])
init_diff = max(init_max) - h_min
plt.ylim(h_min-init_diff*0.1, init_diff*1.1)
plt.text(x_r, h_min-init_diff*0.1, '$t$', fontsize=l_size)
plt.title('%s Compare: $h_{best}$' % (obj_name), fontsize=f_size)    
plt.legend(fontsize=f_size)

#### Plot average function value as a function of time
init_max = []
plt.figure()
for i in range(L):
    h_data = optimizer_list[i].algorithm.h_hat[0:-1]
    init_max.append(max(h_data))
    plt.plot(time[0:-1], h_data, color=colors[i], linestyle=line_styles[i], lw=lw, label=optimizer_legend[i])
plt.plot([x_l, x_r], [h_min, h_min], '--k', lw=2, alpha=0.5)
plt.xlim(start, time[-2])
init_diff = max(init_max) - h_min
plt.ylim(h_min-init_diff*0.1, init_diff*1.1)
plt.text(x_r, h_min-init_diff*0.1, '$t$', fontsize=l_size)
plt.title('%s Compare: $\hat{h}$' % (obj_name), fontsize=f_size)    
plt.legend(fontsize=f_size)

#### Plot distance to global minimizer as a function of time
init_max = []
plt.figure()
for i in range(L):
    x_data = optimizer_list[i].algorithm.error[0:-1]
    init_max.append(max(x_data))
    plt.plot(time[0:-1], x_data, color=colors[i], linestyle=line_styles[i], lw=lw, label=optimizer_legend[i])
plt.plot([x_l, x_r], [0, 0], '--k', lw=2, alpha=0.5)
plt.xlim(start, time[-2])
init_diff = max(init_max) - h_min
plt.ylim(h_min-init_diff*0.1, init_diff*1.1)
plt.text(x_r, h_min-init_diff*0.1, '$t$', fontsize=l_size)
plt.title('%s Compare: error' % (obj_name), fontsize=f_size)    
plt.legend(fontsize=f_size)

#### Each direction
tt = np.arange(-10, int(stop/step)+10, 1)
for i in range(0, len(optimizer_list)):
    if dim > 1:
        f, axarr = plt.subplots(dim, sharex=True)
        for d in range(dim):
            axarr[d].plot([x_l, x_r], [x_min[d]]*2, c=PRORANGE, alpha=0.7, linewidth=1.5) # pos. of glob. min. line x
            for TI in range(0, int(stop/step), stride):
                t = TI*step
                z  = t*np.ones_like(optimizer_list[i].value[TI,d])
                axarr[d].scatter(z, optimizer_list[i].value[TI,d], marker='+', c=PROBLUE, s=1)           
            axarr[d].set_xlim(x_l, x_r)
            axarr[d].set_ylabel('$x_%s$' % (d+1),fontsize = l_size)        
            if d == 0:
    #            axarr[d].legend(('Global min.','Particles'), loc=1)
                axarr[d].set_title('%s: %s' % (obj_name, optimizer_legend[i]))
            if d == dim-1:
#                axarr[d].set_xlabel('$t$', fontsize = l_size)
                axarr[d].text(x_r*1.02, axarr[d].get_ylim()[0], '$t$', fontsize = l_size)
    #            axarr[d].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    else:
        plt.figure()
        plt.plot([x_l, x_r], [x_min[d]]*2, c='g', alpha=0.7, linewidth=1.5)
        for TI in range(0, int(stop/step), stride):
            t = TI*step
            z  = t*np.ones_like(optimizer_list[i].value[TI,d])
            plt.scatter(z, optimizer_list[i].value[TI,d], marker='+', c=PROBLUE, s=1)           
        plt.xlim(x_l, x_r)
        plt.xlabel('$t$', fontsize = l_size)
        plt.ylabel('$x_%s$' % (d+1),fontsize = l_size)      
#        plt.title('%s: %s' % (obj_name, optimizer_legend[i]),fontsize = l_size)
            
            
            
            

plt.show()





