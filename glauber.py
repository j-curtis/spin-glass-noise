### This code will implement Glauber type dynamics of a classical spin model in order to simulate glassy dynamics 
### Jonathan Curtis 
### 11/01/2025

import numpy as np 


rng = np.random.default_rng()

### This method will initialize an array of the chosen size to either a random or uniform state 
### We store the spin in a one dimensional flattened array 
def initialize_spins(Lx,Ly,random=False):
	spins = np.ones((Lx,Ly))
	
	if random: spins = rng.choice([-1,1],Lx*Ly)

	return spins.flatten()

### This method computes the magnetization of a flattened spin configuration
def calc_mag(spins):
	return np.mean(spins,axis=0)

### This is the total energy of a sampled spin configuration
### This is catastrophically slow for even modest sized systems. Need to implement a running tally update of energy 
def calc_energy(spins,J_matrix):
	return 0.5*np.tensordot(spins, J_matrix@spins,axes=(0,0))

### This method will generate the nearest-neighbor exchange matrix of couplings 
def nn_coupling(J,Lx,Ly):
	J_matrix = np.zeros((Lx*Ly,Lx*Ly))
	sites = np.arange(Lx*Ly)
	for r in sites:
		x = r%Lx 
		y = r//Lx

		rpx = (x+1)%Lx + y*Lx
		rmx = (x-1)%Lx + y*Lx
		rpy = x%Lx + ( (y+1)%Ly )*Lx
		rmy = x%Lx + ( (y-1)%Ly )*Lx 

		J_matrix[rpx,r] = J
		J_matrix[rmx,r] = J 
		J_matrix[rpy,r] = J 
		J_matrix[rmy,r] = J 

		J_matrix[r,r] = 0.

	return 0.5*( J_matrix + np.transpose(J_matrix))

### This methof performs time evolution according to Glauber MCMC
def dynamics(initial_spins,nsteps,J_matrix,T):
	Nspins = len(initial_spins)

	spin_trajectory = np.zeros((Nspins,nsteps))
	spin_trajectory[:,0] = initial_spins[:]

	for i in range(1,nsteps):
		r = rng.choice(np.arange(Nspins))

		p = rng.uniform()

		curie_field = np.sum(J_matrix[r,:]*spin_trajectory[:,i-1])

		spin_trajectory[:,i] = spin_trajectory[:,i-1]

		delta_E = -2.*curie_field*spin_trajectory[r,i]

		if p < np.exp(-delta_E/T):
		    spin_trajectory[r,i] *= -1 

	return spin_trajectory



















