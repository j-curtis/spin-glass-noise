### This code will implement Glauber type dynamics of a classical spin model in order to simulate glassy dynamics 
### Jonathan Curtis 
### 11/01/2025

import numpy as np 
from matplotlib import pyplot as plt
import scipy as scp
import time as time 


rng = np.random.default_rng()

### This method will initialize an array of the chosen size to either a random or uniform state 
### We store the spin in a one dimensional flattened array 
def initialize_spins(Lx,Ly,random=False):
	spins = np.ones((Lx,Ly))
	
	if random: spins = rng.random.choose([-1.,1.],(Lx,Ly))

	return spins.flatten()



### This method will generate the nearest-neighbor exchange matrix of couplings 
def nn_coupling(J,Lx,Ly):
	J_matrix = np.zeros((Lx*Ly,Lx*Ly))
	sites = np.arange(Lx*Ly)
	for r in sites:
		x = r%Lx 
		y = r//(Ly-1)

		rpx = (x+1)%Lx + y*Lx
		rmx = (x-1)%Lx + y*Lx
		rpy = x%Lx + ( (y+1)%Ly )*Lx
		rmy = x%Lx + ( (y-1)%Ly )*Lx 

		J_matrix[rpx,r] = J
		J_matrix[rmx,r] = J 
		J_matrix[rpy,r] = J 
		J_matrix[rmy,r] = J 

	return J_matrix 












