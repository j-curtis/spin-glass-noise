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

### This method generates a list of the nn indices of each site -- used for faster lookup when couplings are NN only 
def nn_indices(Lx,Ly):
	sites = np.arange(Lx*Ly)
	nns = np.zeros((4,Lx*Ly),dtype=int) ### This is the list of 4 nn indices of each site stored in the order [+x,+y,-x,-y]
	for r in sites:
		x = r%Lx 
		y = r//Lx

		rpx = (x+1)%Lx + y*Lx
		rmx = (x-1)%Lx + y*Lx
		rpy = x%Lx + ( (y+1)%Ly )*Lx
		rmy = x%Lx + ( (y-1)%Ly )*Lx 

		nns[:,r] = np.array([rpx,rpy,rmx,rmy],dtype=int)

	return nns


### This method will generate the nearest neighbor couplings but randomize whether they are + or - (p is probability of +)
def nn_coupling_random(J,p,Lx,Ly):
	J_matrix = np.zeros((Lx*Ly,Lx*Ly))
	sites = np.arange(Lx*Ly)
	for r in sites:
		x = r%Lx 
		y = r//Lx

		rpx = (x+1)%Lx + y*Lx
		rmx = (x-1)%Lx + y*Lx
		rpy = x%Lx + ( (y+1)%Ly )*Lx
		rmy = x%Lx + ( (y-1)%Ly )*Lx 

		J_matrix[rpx,r] = rng.choice([J,-J],p=[p,1.-p])
		J_matrix[rmx,r] = rng.choice([J,-J],p=[p,1.-p]) 
		J_matrix[rpy,r] = rng.choice([J,-J],p=[p,1.-p])
		J_matrix[rmy,r] = rng.choice([J,-J],p=[p,1.-p])

		J_matrix[r,r] = 0.

	return 0.5*( J_matrix + np.transpose(J_matrix))


### This method performs time evolution according to Glauber MCMC
### By default the number of steps will actually be the number of sweeps with Lx x Ly individual steps 
### If nn_indices is passed we will use a local update which only checks nearest neighbor indices which are stored in the array passed as indices_of_r = [nn direction,r]
def dynamics(initial_spins,nsweeps,J_matrix,T,nn_indices=None):
	Nspins = len(initial_spins)

	spin_trajectory = np.zeros((Nspins,nsweeps))
	spin_trajectory[:,0] = initial_spins[:]

	### Implements a single step which there are then Lx x Ly of in a sweep
	def MCstep(spins):
		r = rng.choice(np.arange(Nspins))

		p = rng.uniform()

		if nn_indices is None:
			curie_field = np.sum(J_matrix[r,:]*spins[:])

		else:
			nnpx = nn_indices[0,r]
			nnpy = nn_indices[1,r]
			nnmx = nn_indices[2,r]
			nnmy = nn_indices[3,r]

			curie_field = J_matrix[nnpx,r]*spins[nnpx] + J_matrix[nnpy,r]*spins[nnpy] +J_matrix[nnmx,r]*spins[nnmx] +J_matrix[nnmy,r]*spins[nnmy] 

		delta_E = -2.*curie_field*spins[r]

		if p < np.exp(-delta_E/T):
			spins[r] *= -1 

		return spins

	for i in range(1,nsweeps):
		spins = spin_trajectory[:,i-1]
		for j in range(Nspins):
			spins = MCstep(spins)

		spin_trajectory[:,i] = spins


	return spin_trajectory



















