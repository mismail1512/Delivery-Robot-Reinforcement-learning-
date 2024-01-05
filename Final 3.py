# Main Classes 

class Environment:
    '''
    Abstract base class for an (interactive) environment formulation.
    It declares the expected methods to be used to solve it.
    All the methods declared are just placeholders that throw errors if not overriden by child "concrete" classes!
    '''
    
    def __init__(self):
        '''Constructor that initializes the problem. Typically used to setup the initial state.'''
        self.state = None
    
    def actions(self):
        '''Returns an iterable with the applicable actions to the current environment state.'''
        raise NotImplementedError
    
    def apply(self, action):
        '''Applies the action to the current state of the environment and returns the new state from applying the given action to the current environment state; not necessarily deterministic.'''
        raise NotImplementedError
    
    @classmethod
    def new_random_instance(cls):
        '''Factory method to a problem instance with a random initial state.'''
        raise NotImplementedError

def action_from_q(env, q, verbose=True):
    '''Get the best action for the current state of the environment from Q-values'''
    return max((action for action in env.actions()), key=lambda action: q.get((env.state, action), 0))

import numpy as np

def q_learning(env, q={}, n={}, f=lambda q, n: (q+1)/(n+1), alpha=lambda n: 60/(n+59), error=1e-6, verbose=False):
    '''Q-learning implementation that trains on an environment till no more actions can be taken'''
    if verbose: visualizer = Visualizer(env)
    c1=0
    A=np.zeros((25*25,6))
    while env.state is not None :
        if c1==3:
            env.state=list(env.state)
            env.state[5]=4
            env.state[6]=4
            env.state[7]=0
            env.state[8]=0
            env.state=tuple(env.state)
            break
        c1+=1
        if verbose: visualizer.visualize([env.state])
        state = env.state
        action = max(env.actions(),
                     key=lambda next_action: f(q.get((state, next_action), 0), n.get((state, next_action), 0)))
        print(action)
        print(env.actions())
        n[(state, action)] = n.get((state, action), 0) + 1
        print(action)
        print(env.state)
        #print(n[(state, action)])
        print(env.actions())
        reward = env.apply(action)
        if reward == None:
            reward = 1000
        q[(state, action)] = q.get((state, action), 0) \
                           + alpha(n[state, action]) \
                           * (reward
                              + env.discount * max((q.get((env.state, next_action), 0) for next_action in env.actions()), default=0)
                              - q.get((state, action), 0))
        if action == 'up':
            z=(state[0][0])+(25*(state[0][1]-1))   
            A[z-1,0] = q.get((state, action), 0)
        elif action == 'down':
            z=(state[0][0])+(25*(state[0][1]-1))   
            A[z-1,1] = q.get((state, action), 0)
        elif action == 'left':
            z=(state[0][0])+(25*(state[0][1]-1))   
            A[z-1,2] = q.get((state, action), 0)
        elif action == 'right':
            z=(state[0][0])+(25*(state[0][1]-1))   
            A[z-1,3] = q.get((state, action), 0)
        elif action == 'pick':
            z=(state[0][0])+(25*(state[0][1]-1))   
            A[z-1,4] = q.get((state, action), 0)
        elif action == 'drop':
            z=(state[0][0])+(25*(state[0][1]-1))   
            A[z-1,5] = q.get((state, action), 0) 
    mat=np.matrix(A)
    with open('q_Matrix.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')    
    return q, n  

from math import inf
from time import time
from itertools import count

def simulate(env_ctor, n_iterations=inf, duration=inf, **q_learning_params):
    '''A helper function to train for a fixed number of iterations or fixed time'''
    for param in ('q', 'n'): q_learning_params[param] = q_learning_params.get(param, {})
    start_time = time()
    i = count()
    while time() < start_time + duration and next(i) <= n_iterations:
        env = env_ctor()
        q, n= q_learning(env, **q_learning_params)
    return q_learning_params['q'], q_learning_params['n']

from shutil import get_terminal_size
terminal_width, _ = get_terminal_size()

_visualizers = {}

def _default_visualizer(_, state):
    '''Generic visualizer for unknown problems.'''
    print(state)

# Visualizer

class Visualizer:
    '''Visualization and printing functionality encapsulation.'''

    def __init__(self, problem):
        '''Constructor with the problem to visualize.'''
        self.problem = problem
        self.counter = 0
    
    def visualize(self, frontier):
        '''Visualizes the frontier at every step.'''
        self.counter += 1
        print(f'Frontier at step {self.counter}')
        for state in frontier:
            print()
            _visualizers.get(type(self.problem), _default_visualizer)(self.problem, state)
        print('-' * terminal_width)

def ZCRobitVis(env, state):
    gate_1=(13,1)
    academic_building= (13,14)             
    gate_2=(25,6)
    gate_3=(25,20)
    gate_4=(13,25)
    gate_5=(1,20)       
    gate_6=(1,6)
    admin_building=(13,11)
    helmy_building=(8,9)
    nano_building=(18,9)
    service_building=(3,22)
    one_stop_shop=(21,8)
    work_shops=(4,23)
    dorms=(22,23)      
    c2=0
    RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1 = env.state
    for j in range(1, env.bounds[1] + 1):
        for i in range(1, env.bounds[0] + 1):

            print('🤖' if (i, j) == RobotLocation else'HB' if (i, j) == helmy_building else 'NB' if (i, j) == nano_building else 'AB' if (i, j) == academic_building else'OS' if (i, j) == one_stop_shop else'G1' if (i, j) == gate_1 else 'G2' if (i, j) == gate_2 else 'G3' if (i, j) == gate_3 else'G4' if (i, j) == gate_4 else 'G5' if (i, j) == gate_5 else'G6' if (i, j) == gate_6 else'Ad' if (i, j) == admin_building else 'Dr' if (i, j) == dorms else 'SB' if (i, j) == service_building else'WS' if (i, j) == work_shops else '⬜', end='')
            
        print()

    print(RobotLocation)
    print((state[5],state[6],state[7],state[8]))
    print (c2)
    print(env.actions)
    if state[0]== state[1] or state[0]== state[2] or state[0]== state[3] or state[0]== state[4]:
        print ('arrrrrrrrrrrrrrrive')
        c2+=1
        print (c2)
        print(env.carry)


from random import choice, randrange

# Our Environment

class ZCRobot(Environment):
    
    def __init__(self):
      #   #state={ HB:NoHB, NB: NoNB, OSS :NoOSS, AB: NoAB}
        NoHB=4
        NoNB=4
        NoAB=0
        NoOSS=0
        gate_1=(13,1)
        academic_building= (13,14)             
        gate_2=(25,6)
        gate_3=(25,20)
        gate_4=(13,25)
        gate_5=(1,20)       
        gate_6=(1,6)
        admin_building=(13,11)
        helmy_building=(8,9)
        nano_building=(18,9)
        service_building=(3,22)
        one_stop_shop=(21,8)
        work_shops=(4,23)
        dorms=(22,23)       
        self.bounds=(25,25)
        self.RobotLocation=gate_1
        self.items = { 'HB':NoHB, 'NB': NoNB, 'OSS' :NoOSS, 'AB': NoAB}
        self.state=(self.RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1)
        self.max_reward = 1000
        self.discount = 0.3
        self.carry = False
        self.Obs = {(14, 5), (12, 5), (16, 13), (20, 16)}
    
    def actions(self):
        if self.state is None: return []
        if (self.state[5] == self.state[6] == 0) and (self.state[7] == self.state[8] == 4) : 
            return ["Done!"]
        
        if self.state[0]== self.state[1] and self.state[5] > 0 and self.carry == False  :
             print("'mmmmmm🤖🤖🤖🤖mmmmmmmmmm🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖mm🤖🤖🤖🤖mmmmmmmmmmmmmmmmmmmmmm")

             return ['pick'] 
        if self.state[0]== self.state[2] and self.state[6] > 0 and self.carry == False  :
             
             return ['pick'] 
        
        if self.state[0]== self.state[4] and self.state[8] < 4 and self.carry == True  :


             return ['drop'] 
        if self.state[0]== self.state[3] and self.state[7] < 4 and self.carry == True  :


            return ['drop'] 
           
        return ['up', 'down', 'right', 'left']
    
    def apply(self, action):
        down = lambda position: (position[0], min(position[1] + 1, self.bounds[1] - 1))
        #if down in self.Obs:       # if action is obstacle stay in place
        #  down = self.state[0]
        up = lambda position: (position[0], max(position[1] - 1, 1))
        #if up in self.Obs:      
        #  up = self.state[0]
        left = lambda position: (max(position[0] - 1, 1), position[1])
        #if left in self.Obs:      
        #  left = self.state[0]
        right = lambda position: (min(position[0] + 1, self.bounds[0]), position[1])
        #if right in self.Obs:      
        #  right = self.state[0]
        RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1 = self.state
        if   action ==     'up': self.state = (up(RobotLocation),helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1 ) ; return -0.1*self.max_reward
        elif action ==   'down': self.state = (down(RobotLocation),helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1) ; return -0.1*self.max_reward
        elif action ==   'left': self.state = (left(RobotLocation),helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1) ; return -0.1*self.max_reward
        elif action ==  'right': self.state = (right(RobotLocation),helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS,gate_1) ; return -0.1*self.max_reward
        elif action == 'drop' and self.state[0]== self.state[3] and self.state[7] < 4 :
            self.carry = False 
            self.state = (RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB+1,NoOSS,gate_1 )   ; return self.max_reward
        elif action == 'drop' and self.state[0]== self.state[4] and self.state[8] < 4 : 
            self.carry = False 
            self.state = (RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB,NoAB,NoOSS+1,gate_1 )     ; return self.max_reward
        elif action == 'pick' and self.state[0]== self.state[1] and self.state[5] > 0 :
            self.carry = True 
            self.state = (RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB-1,NoNB,NoAB,NoOSS,gate_1 )     ; return self.max_reward
        elif action == 'pick' and self.state[0]== self.state[2] and self.state[6] > 0 : 
            self.carry = True 
            self.state = (RobotLocation,helmy_building,nano_building,academic_building,one_stop_shop,NoHB,NoNB-1,NoAB,NoOSS,gate_1 )     ; return self.max_reward
        elif action == 'Done!': self.state = (gate_1,helmy_building,nano_building,academic_building,one_stop_shop,4,4,0,0,gate_1 )          ; return +self.max_reward

    @classmethod
    def new_random_instance(cls):
        return cls()


_visualizers[ZCRobot] = ZCRobitVis

# Testing

from random import random

q, n = {}, {}

simulate(lambda: ZCRobot.new_random_instance(),duration=60, q=q, n=n, verbose=True,f=lambda q, n: 1/(n+1))

#simulate(lambda: ZCRobot.new_random_instance(), duration=30, q=q, n=n,verbose=True, f=lambda q, n: random())

simulate(lambda:  ZCRobot.new_random_instance(), n_iterations=1, q=q, n=n, verbose=True, f=lambda q, n: q)




