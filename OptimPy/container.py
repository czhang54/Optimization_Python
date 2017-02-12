
import numpy as np
import copy


class World(object):
    ''' Contains the simulation parameters and all items instantiated.
    '''

    def __init__( self, dim=2, time=None ):

        self.dim       = dim # Dimension of the problem
        # Simulation time
        self.time      = time
        self.num_times = len(self.time) # Number of simulations
        self.dt        = np.diff(self.time) # Time step, uniform by default
        self.dt        = np.append(self.dt, self.dt[0])
        
        # List of items in the world
        self.members    = []
 
    def add(self, obj):
        ''' Adds obj to the list of members, and sets the obj's world to self.
        '''

        obj.world = self
        self.members.append(obj)
        return obj

    def remove(self, obj):
        ''' removes obj from the list of members, and deletes the obj.
        '''

        self.members.remove(obj)
        del obj

    def get(self, mytype):
        ''' Returns a list of members of type mytype.'''

        return filter(lambda obj: isinstance(obj, mytype), self.members)

    def items(self, type=None):
        ''' Returns all items (type=None) or items of type.'''

        if type is not None:
            return [item for item in self.get(Item) if item.type == type]
        else:
            return self.get(Item)
            
    def obj_fns(self, type=None):
        ''' Returns the objective function in item (type=None).
        '''

        return [model for item in self.items(type) for model in item.obj_fns()]

    def optimizers(self, type=None):
        ''' Returns all optimizer in item (type=None), or all
            optimizer belonging to an item of type.
        '''

        return [model for item in self.items(type) for model in item.optimizer()]

    def simulate(self, list):
        ''' This method takes a list of optimizer objects as inputs
            and runs each optimizer object's update() method for every time point.
        '''
        for TI in range(0, self.num_times-1):
            for obj in list:
                obj.update(TI)


class Item(object):
    '''Item(object) - Container class for Optimizer.
    '''

    def __init__( self, mytype=None ):
        self.type       = mytype
        self.world      = None # Set when Item is added to World
        self.members    = [] 

    def get(self, mytype):
        ''' Returns a list of members of type mytype.  '''

        return filter(lambda obj: isinstance(obj, mytype), self.members)

    def add(self, obj):
        ''' Add obj to list of members.  '''

        self.members.append(obj)
        obj.item = self
        obj.dim  = self.world.dim
        return obj

    def remove(self, obj):
        ''' Remove obj from list of members and delete that obj.
        '''

        self.members.remove(obj)
        del obj
        
    def obj_fns(self):
        ''' Returns obj_fn object from the members list.
        '''

        return self.get(Obj_fn)

    def optimizers(self):
        ''' Returns all optimizer object from the members list.
        '''

        return self.get(Optimizer)


class Optimizer(object):
    ''' The Optimizer class provides the basic parameters and methods to
        define and simulate optimizers with different algorithms.
    '''

    def __init__( self, N=1, dim=1 ):

        self.N          = N
        self.item       = None # Set when Optimizer is added to an item

        # Parameters set from models
        self.sigma      = None
        self.IC         = None

        # Models
        self.prior      = None # Initial conditions
        self.algorithm  = None # Algorithm
        
        # Parameters set during initialization call
        self.dim        = None
        self.num_times  = None
        self.d          = None
        self.value      = None
            
    def setPrior(self, model=None, **kwargs):
        ''' Sets prior distribution model
            Arguments:
                + model     : The probability density function modeling the prior distribution
                + kwargs    : Extra inputs for the prior model
        ''' 
           
        self.prior = model(self, **kwargs)

        return self.prior    
    
    def setAlgorithm(self, model=None, **kwargs): ## PROBLEM HERE NOT FIXED !!!???
        ''' Sets the algorithm model 
            Arguments:
                + model     : The algorithm model.
                + kwargs    : Extra inputs for the model (model constants, etc.).
        '''
        self.algorithm = model(self, **kwargs)
        
        return self.algorithm    
    
    def initialize(self):
        ''' Initilialize the optimizer '''

        self.num_times  = self.item.world.num_times           
        self.value      = np.zeros((self.num_times, self.dim, self.N))
        if self.prior is not None:
            self.value[0] = self.prior.f(self.N)   
        
    def update(self, TI):
        ''' Updates the Optimizer to the next time-step by running its algorithm
            Arguments:
                + TI    : Time index.
        '''        
        dt = self.item.world.dt[TI] # Current time step
        
        X = copy.copy(self.value[TI]) # Current state
            
        if self.algorithm is not None:
            self.value[TI+1] = self.algorithm.run(self.value[TI], dt, TI=TI)
            
        self.value[TI] = X

        
class Obj_fn(object):
    ''' The Obj_fn class defines an object containing information of an objective function.
    '''
    
    def __init__(self, dim=None): 
        
        self.item  = None
        self.dim   = None
        self.model = None
        
    def setObjective(self, model=None, **kwargs):
        ''' Sets the objective function
            Arguments:
                + model     : The objective model.
                + kwargs    : Extra inputs for the model (model constants, etc.).
        '''
        self.model = model(self, **kwargs)







   
