### This code will implement Glauber type dynamics of a classical spin model in order to simulate glassy dynamics 
### Jonathan Curtis 
### 06/17/2026

### Streamlined and cleaned-up version of old code which is now built to be more efficient and handle arbitrary lattices 
### !!! Due to current demler_tools restrictions cannot pass arbitrary objects to run method so instead we pass a limited set of parameters and built object on the fly 


import numpy as np 
import lattice_methods as lm
import pickle

### This method will initialize an array of the chosen size to either a random or uniform state 
### We store the spin in a one dimensional flattened array 
def initialize_spins(Lx,Ly,random=False,seed=None):
	spins = np.ones((Lx,Ly))

	rng = np.random.default_rng()
	if seed is not None: rng = np.random.default_rng(seed)

	if random: spins = rng.choice([-1,1],Lx*Ly)

	return spins.flatten()

### This method computes the magnetization of a flattened spin configuration
def calc_mag(spins):
	return np.mean(spins,axis=-2)

### This is the total energy of a sampled spin configuration
### This is catastrophically slow for even modest sized systems. Need to implement a running tally update of energy 
def calc_energy(spins,J_matrix):
	return 0.5*np.tensordot(spins, J_matrix@spins,axes=(0,0))


### Low memory dynamics for generic lattice object passed 
def anneal_dynamics_lattice(lattice,nsweeps,temperature_schedule,distances,initial_seed=None,dynamics_seed=None):
	### Recast single float as an array of length 1 for technical
	if not isinstance(temperature_schedule,np.ndarray): temperature_schedule = np.atleast_1d(np.asarray(temperature_schedule,dtype=float))
	if not isinstance(distances,np.ndarray): distances = np.atleast_1d(np.asarray(distances,dtype=float))
	
	rng = np.random.default_rng() ### We reinstantiate the rng 
	if dynamics_seed is not None: rng = np.random.default_rng(dynamics_seed)

	nTs = len(temperature_schedule)
	ndists = len(distances)
	
	nspins = lattice.N
	
	
	### We generate output arrays
	energies = np.zeros((nTs,nsweeps))
	mags = np.zeros((nTs,nsweeps))
	q_eas = np.zeros(nTs)
	neels = np.zeros((nTs,nsweeps))
	noises = np.zeros((nTs,ndists,nsweeps))


	### Method for computing local magnetic field noise 

	### Units and prefactors to keep track
	mu0 = 1. ### vacuum permeability
	muB = 1. ### magnetic moment in bohr magnetons 
	a = 1. ### lattice constant 

	### ASSUME SQUARE GRID 
	Lx = int(lattice.L)
	Ly = int(lattice.L)

	### Computes local magnetic noise
	### Assumes spin qubit is a distance d away in the center of the sample 

	### The magnetization in the z direction a distance d away is given in terms of the spins as 
	### B_z(t) = mu_0 mu_B mu_mat/(4 pi) sum_j S_j(t) (2d^2 - R_j^2)/(R_j^2 + d^2)^(5/2) 

	X,Y = np.meshgrid(np.arange(Lx)-Lx//2, np.arange(Ly)-Ly//2,indexing='ij')

	R = np.sqrt(X**2 + Y**2) 

	R = R.ravel() 

	kernels = (2.*distances[None,:]**2 - R[:,None]**2)/( R[:,None]**2 + distances[None,:]**2)**2.5 ### Has shape [Nspins, Nds] 

	### Mask to compute Neel order
	neel_mask = (X+Y).astype(int)
	neel_mask = neel_mask.ravel() 
	neel_mask = (-1.*np.ones(nspins,dtype=int))**neel_mask
	
	### For a single spin configuration this returns an array of the magnetic field noises at each distance at each time point
	def local_noise_field(spins):
		return np.tensordot(spins,kernels,axes=[0,0]) * mu0*muB/(4.*np.pi*a**3)

	### Implements a single step which there are then Lx x Ly of in a sweep
	### Modified to only in-place flip
	def MCstep(spins):
		r = rng.choice(np.arange(nspins))

		neighbors = lattice.partners[r]
		curie_field = sum(lattice.J_matrix[i, r] * spins[i] for i in neighbors)

		dE = -2.0 * curie_field * spins[r]
		betadE = dE / T

		if betadE >= 0:
			p_flip = np.exp(-betadE) / (1.0 + np.exp(-betadE))
		else:
			p_flip = 1.0 / (1.0 + np.exp(betadE))

		if rng.uniform() < p_flip:
			spins[r] *= -1
			return dE

		return 0.0
	
	### Random initial state for each replica 
	spins = initialize_spins(nspins,1,random=True,seed=initial_seed)
	
	for n in range(nTs):
		T = temperature_schedule[n]
		
		### Just first time step we compute the observables for the entire system
		energies[n,0] = calc_energy(spins,lattice.J_matrix)
		mags[n,0] = np.mean(spins) 
		neels[n,0] = np.mean(spins*neel_mask)
		
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
				dE = MCstep(spins) 
				energy_change += dE

				
			### Update the energy 
			energies[n,i] = energies[n,i-1] + energy_change
			
			### Update the magnetization 
			mags[n,i] = np.mean(spins) 

			
			neels[n,i] = np.mean( spins*neel_mask )

			### Update the frozen moment if we are past the chop window 
			if i >= chop_size:
				nsweeps_chopped += 1 
				frozen_moment += spins
			

			### Log the local magnetic fields 
			noises[n,:,i] = local_noise_field(spins) 
			
			### Spins is now updated for the next loop/annealing epoch 
		
		### We now flatten the frozen moment to compute the q_ea order parameter 
		if nsweeps_chopped > 0: 
			frozen_moment = frozen_moment/float(nsweeps_chopped) ### Normalize by number of time steps 
			q_eas[n] = np.mean(frozen_moment**2) ### Average over volume 
		

	return energies, mags, neels, q_eas, noises



### Saves a compact output and is low memory usage during operation 
### Due to current demler_tools restrictions cannot pass arbitrary objects to run method so instead we pass a limited set of parameters and built object on the fly 

def run_sims(save_filename,L,Jnnn,p,J_seed,nsweeps,temps,distances,replica,initial_seed=None,dynamics_seed=None):
	L = int(L)
	J_seed = int(J_seed)

	latt = lm.lattice(L)
	latt.set_seed(J_seed)
	latt.set_nn_J(1.,1.)
	latt.set_nnn_J(Jnnn,p)
	Lx = L 
	Ly = L 
	
	nsweeps = int(nsweeps)
	replica = int(replica) 

	energies, magnetization, neel, qea, noise = anneal_dynamics_lattice(latt,nsweeps,temps,distances,initial_seed,dynamics_seed)
		
	### Due to large memory of spin configurations we will now compute derived observables to save 
	### 1) Energy vs time 
	### 2) Magnetization vs time 
	### 3) Neel order vs time 
	### 4) Edwards-Anderson OP vs time 
	### 5) Local noise for different distances vs time 
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((latt,energies,magnetization,neel,qea,noise), out_file) ### We store the output spin trajectory, the annealing schedule, and the lattice
        	
      	




