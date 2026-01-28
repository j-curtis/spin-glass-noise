### This code will implement Glauber type dynamics of a classical spin model in order to simulate glassy dynamics 
### Jonathan Curtis 
### 11/01/2025

import numpy as np 
import time 

from scipy import ndimage as ndi 


rng = np.random.default_rng()

### This method will initialize an array of the chosen size to either a random or uniform state 
### We store the spin in a one dimensional flattened array 
def initialize_spins(Lx,Ly,random=False):
	spins = np.ones((Lx,Ly))
	
	if random: spins = rng.choice([-1,1],Lx*Ly)

	return spins.flatten()

### This method computes the magnetization of a flattened spin configuration
def calc_mag(spins):
	return np.mean(spins,axis=-2)

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
### Allows to pass a seed to generate deterministic disorder configs 
def nn_coupling_random(J,p,Lx,Ly,seed=None):
	rng = np.random.default_rng()
	if seed is not None: rng = np.random.default_rng(seed) 
	
	J_matrix = np.zeros((Lx*Ly,Lx*Ly))
	sites = np.arange(Lx*Ly)
	for r in sites:
		x = r%Lx 
		y = r//Lx

		rpx = (x+1)%Lx + y*Lx
		rmx = (x-1)%Lx + y*Lx
		rpy = x%Lx + ( (y+1)%Ly )*Lx
		rmy = x%Lx + ( (y-1)%Ly )*Lx 

		### We should only count the + bonds otherwise we will double count. This is a problem when the couplings are random and we later symmetrize as it can lead to cancellations that are spurious
		J_matrix[rpx,r] = rng.choice([J,-J],p=[p,1.-p])
		#J_matrix[rmx,r] = rng.choice([J,-J],p=[p,1.-p]) 
		J_matrix[rpy,r] = rng.choice([J,-J],p=[p,1.-p])
		#J_matrix[rmy,r] = rng.choice([J,-J],p=[p,1.-p])

		J_matrix[r,r] = 0.

	return J_matrix + np.transpose(J_matrix)


### This method performs time evolution according to Glauber MCMC
### By default the number of steps will actually be the number of sweeps with Lx x Ly individual steps 
### If nn_indices is passed we will use a local update which only checks nearest neighbor indices which are stored in the array passed as indices_of_r = [nn direction,r]
### If reseed is true this will reseed the rng on every call 
def dynamics(initial_spins,nsweeps,J_matrix,T,nn_indices=None,reseed=True):
	Nspins = len(initial_spins)

	spin_trajectory = np.zeros((Nspins,nsweeps))
	spin_trajectory[:,0] = initial_spins[:]
	
	energy = np.zeros(nsweeps)
	
	### Just first time step we compute the total energy for the entire system
	energy[0] = calc_energy(initial_spins,J_matrix)
	
	if reseed:
		rng = np.random.default_rng() ### We reinstantiate the rng 

	### Implements a single step which there are then Lx x Ly of in a sweep
	def MCstep(spins):
		tmp = spins.copy()
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
		
		betadE = delta_E/T
		
		### Metropolis
		#if p < np.exp(-delta_E/T):
		
		betadE_return = 0.
		
		### Glauber dynamics has probability of flip e^(-beta dE)/(1+ e^(-beta dE))
		if p < np.exp(-betadE)/(1.+np.exp(-betadE)):
			tmp[r] *= -1 
			betadE_return = betadE

		return tmp, betadE_return 

	for i in range(1,nsweeps):
		spins = spin_trajectory[:,i-1]
		
		energy_change = 0. 
		
		for j in range(Nspins):
			spins, betadE = MCstep(spins)
			energy_change += betadE*T

		spin_trajectory[:,i] = spins
		energy[i] = energy[i-1] + energy_change

	return spin_trajectory, energy 
	
	
	


### This method will anneal over a given temperature schedule and generate one replica trajectory for the dynamics over this annealing time 
### Returns the spin trajectories and the times for each temperature run
### Allows also for multiple replicas though default is one since we will paralellize this using demler_tools to instantiate multiple processes 
def anneal_dynamics(J_matrix,nn_matrix,nsweeps,temperature_schedule,nreplicas=1,initial_spins = None,verbose=False):

	if not isinstance(temperature_schedule,np.ndarray): temperature_schedule = np.array([temperature_schedule]) ### Recast single float as an array of length 1 for technical reasons

	nTs = len(temperature_schedule)
	
	nspins = J_matrix.shape[0]

	### We generate an output array 
	spins = np.zeros((nreplicas,nTs,nspins,nsweeps))
	energies = np.zeros((nreplicas,nTs,nsweeps))
	times = np.zeros((nreplicas,nTs))

	for a in range(nreplicas):

		### If we don't pass a fixed initial condition we will use a random one which is different for each replica 
		if initial_spins is None: 
			initial = initialize_spins(nspins,1,random=True)

		### If we are given a particular initial condition this will make a copy which we use for the initial condition of the annealing schedule 
		else:
			initial = initial_spins.copy()

		### The shape is just Lx x Ly = nspins and since it is flattened anyways we only need the product

		for n in range(nTs):

			T = temperature_schedule[n]

			t0 = time.time()
			spins[a,n,:,:], energies[a,n,:] = dynamics(initial,nsweeps,J_matrix,T,nn_matrix)
			t1 = time.time()

			times[a,n] = t1-t0
			if verbose: print(f"Replica {a}, temperature {T:0.2f}, time: {(t1-t0):0.2f}s")

			### the start of the next chunk of the schedule is the end of this one 
			initial = spins[a,n,:,-1]

	if nreplicas == 1:
		return spins[0,...], times[0,...], energies[0,...]

	else:
		return spins, times, energies


### Implements annealing dynamics but memory efficient -- only saves a number of observables rather than the whole spin state 
def anneal_dynamics_low_mem(J_matrix,nn_matrix,nsweeps,temperature_schedule,distances):
	if not isinstance(temperature_schedule,np.ndarray): temperature_schedule = np.array([temperature_schedule]) ### Recast single float as an array of length 1 for technical
	
	rng = np.random.default_rng() ### We reinstantiate the rng 

	nTs = len(temperature_schedule)
	ndists = len(distances)
	
	nspins = J_matrix.shape[0]
	
	
	### We generate output arrays
	energies = np.zeros((nTs,nsweeps))
	mags = np.zeros((nTs,nsweeps))
	q_eas = np.zeros(nTs)
	noises = np.zeros((nTs,ndists,nsweeps))


	### Method for computing local magnetic field noise 

	### Units and prefactors to keep track
	mu0 = 1. ### vacuum permeability
	muB = 1. ### magnetic moment in bohr magnetons 
	a = 1. ### lattice constant 

	### ASSUME SQUARE GRID 
	Lx = int(np.sqrt(nspins))
	Ly = int(np.sqrt(nspins))

	### Computes local magnetic noise
	### Assumes spin qubit is a distance d away in the center of the sample 

	### The magnetization in the z direction a distance d away is given in terms of the spins as 
	### B_z(t) = mu_0 mu_B mu_mat/(4 pi) sum_j S_j(t) (2d^2 - R_j^2)/(R_j^2 + d^2)^(3/2) 

	X,Y = np.meshgrid(np.arange(Lx)-Lx//2, np.arange(Ly)-Ly//2,indexing='ij')

	R = np.sqrt(X**2 + Y**2) 

	R = R.ravel() 

	kernels = (2.*distances[None,:]**2 - R[:,None]**2)/( R[:,None]**2 + distances[None,:]**2)**2.5 ### Has shape [Nspins, Nds] 
	
	### For a single spin configuration this returns an array of the magnetic field noises at each distance at each time point
	def local_noise_field(spins):
		return np.tensordot(spins,kernels,axes=[0,0]) * mu0*muB/(4.*np.pi*a**3)

	### Implements a single step which there are then Lx x Ly of in a sweep
	def MCstep(spins):

		tmp = spins.copy()
		
		### Random site is selected
		r = rng.choice(np.arange(nspins))

		### Build Curie Weiss field
		nnpx = nn_matrix[0,r]
		nnpy = nn_matrix[1,r]
		nnmx = nn_matrix[2,r]
		nnmy = nn_matrix[3,r]

		curie_field = J_matrix[nnpx,r]*spins[nnpx] + J_matrix[nnpy,r]*spins[nnpy] +J_matrix[nnmx,r]*spins[nnmx] +J_matrix[nnmy,r]*spins[nnmy] 

		delta_E = -2.*curie_field*spins[r]
		
		betadE = delta_E/T
		
		betadE_return = 0.
		
		### Glauber dynamics has probability of flip e^(-beta dE)/(1+ e^(-beta dE))
		p = rng.uniform()
		if p < np.exp(-betadE)/(1.+np.exp(-betadE)):
			tmp[r] *= -1 
			betadE_return = betadE

		return tmp, betadE_return
	
	### Random initial state for each replica 
	spins = initialize_spins(nspins,1,random=True)
	
	for n in range(nTs):
		T = temperature_schedule[n]
		
		### Just first time step we compute the observables for the entire system
		energies[n,0] = calc_energy(spins,J_matrix)
		mags[n,0] = np.mean(spins) 
		
		### This will be used to derive the q_ea
		
		### We need to chop off the first few time steps (we take first 20% to be safe) 
		chop_size = int(nsweeps//5)   
		nsweeps_chopped = 0
		frozen_moment = np.zeros_like(spins) 
		
		### This logs the local magnetic noise 
		noises[n,:,0] = local_noise_field(spins)
		
		for i in range(1,nsweeps):
		
			energy_change = 0. 
			
			### Run a sweep over all spins 
			for j in range(nspins):
				spins, betadE = MCstep(spins)
				energy_change += betadE*T

				
			### Update the energy 
			energies[n,i] = energies[n,i-1] + energy_change
			
			### Update the magnetization 
			mags[n,i] = np.mean(spins) 

			
			### Update the frozen moment if we are past the chop window 
			if i >= chop_size:
				nsweeps_chopped += 1 
				frozen_moment += spins
			

			### Log the local magnetic fields 
			noises[n,:,i] = local_noise_field(spins) 
			
			### Spins is now updated for the next loop/annealing epoch 
		
		### We now flatten the frozen moment to compute the q_ea order parameter 
		frozen_moment = frozen_moment/float(nsweeps_chopped) ### Normalize by number of time steps 
		q_eas[n] = np.mean(frozen_moment**2) ### Average over volume 
		

	return energies, mags, q_eas, noises





### Some analysis methods 


### Performs a moving average over a 1d time series 
def moving_avg(time_series,window):
	return ndi.uniform_filter1d(time_series,window,mode='constant')
	
### Extracts the single-replica frozen moment edwards-anderson order parameter 
def calc_frozen_moment(spins,chop=500,step=100):
	### We assume spins is a replica array of signature [annealing schedule, spins, nsweeps]
	### First we sample over time points
	s = np.mean(spins[...,chop:-1:step],axis=-1) ### Chop off first few points and then sample every step point

	qea = np.mean(s*s,axis=-1) ### This will average the frozen-moment squared over the sample volume  

	return qea 
	
	
	
### Extracts edwards anderson replica order parameter across different replicas 
### This method will compute the edwards-anderson replica correlation function averaged over the sample volume 
def calc_ea(spins,samplestep=100,chop=500):
	### We assume spins is a replica array of signature [replica, annealing schedule, spins, nsweeps]
	### First we sample over time points
	s = np.mean(spins[...,chop:-1:samplestep],axis=-1) ### Chop off first few points and then sample every samplestep point
	qea = np.einsum('air,bir->abi',s,s)/float(s.shape[-1]) ### This will sum over the spins spatially but tracks the time and annealing epoch and keeps both intra and interreplica correlations as a matrix 
	### Also normalizes by sample volume 

	return qea 









