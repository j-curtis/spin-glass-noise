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
	spins = np.ones((Lx,Ly),dtype=np.int8)

	rng = np.random.default_rng()
	if seed is not None: rng = np.random.default_rng(seed)

	if random: spins = rng.choice(np.array([-1,1],dtype=np.int8),Lx*Ly)

	return spins.flatten()

### This method computes the magnetization of a flattened spin configuration
def calc_mag(spins):
	spin_values = np.asarray(spins,dtype=float)
	axis = -2 if spin_values.ndim >= 2 else -1
	return np.mean(spin_values,axis=axis)

### This is the total energy of a sampled spin configuration
### This is catastrophically slow for even modest sized systems. Need to implement a running tally update of energy 
def calc_energy(spins,J_matrix):
	spin_values = np.asarray(spins,dtype=float)
	return 0.5*np.tensordot(spin_values, J_matrix@spin_values,axes=(0,0))


### Low memory dynamics for generic lattice object passed 
def anneal_dynamics_lattice(lattice,nsweeps,temperature_schedule,distances,initial_seed=None,dynamics_seed=None,snapshot_sweeps=None,use_color_updates=False):
	### Recast single float as an array of length 1 for technical
	if not isinstance(temperature_schedule,np.ndarray): temperature_schedule = np.atleast_1d(np.asarray(temperature_schedule,dtype=float))
	if not isinstance(distances,np.ndarray): distances = np.atleast_1d(np.asarray(distances,dtype=float))
	nsweeps = int(nsweeps)

	rng = np.random.default_rng() ### We reinstantiate the rng 
	if dynamics_seed is not None: rng = np.random.default_rng(dynamics_seed)

	nTs = len(temperature_schedule)
	ndists = len(distances)

	nspins = lattice.N
	
	### Block to determine when (if ever) to take snapshots of spin configurations to return 
	snapshots = None
	snapshot_indices = {}
	if snapshot_sweeps is not None:
		snapshot_sweeps = np.atleast_1d(np.asarray(snapshot_sweeps,dtype=int))
		if len(snapshot_sweeps) > 0:
			if np.any(snapshot_sweeps < 0) or np.any(snapshot_sweeps >= nsweeps):
				raise ValueError("snapshot_sweeps must be between 0 and nsweeps - 1.")
			if len(np.unique(snapshot_sweeps)) != len(snapshot_sweeps):
				raise ValueError("snapshot_sweeps cannot contain duplicate sweep indices.")

			snapshots = np.zeros((nTs,len(snapshot_sweeps),nspins),dtype=np.int8)
		snapshot_indices = {int(sweep): idx for idx, sweep in enumerate(snapshot_sweeps)}
	
	
	### We generate output arrays
	energies = np.zeros((nTs,nsweeps))
	mags = np.zeros((nTs,nsweeps))
	q_eas = np.zeros(nTs)
	neels = np.zeros((nTs,nsweeps))
	stripes = np.zeros((2,nTs,nsweeps))
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

	### Masks to compute stripe order along X and Y, respectively
	stripe_x_mask = np.where((X.astype(int).ravel() % 2) == 0, 1., -1.)
	stripe_y_mask = np.where((Y.astype(int).ravel() % 2) == 0, 1., -1.)
	
	### For a single spin configuration this returns an array of the magnetic field noises at each distance at each time point
	def local_noise_field(spins):
		return np.tensordot(np.asarray(spins,dtype=float),kernels,axes=[0,0]) * mu0*muB/(4.*np.pi*a**3)

	def order_parameter(spins,mask):
		return np.mean(np.asarray(spins,dtype=float)*mask)

	### Block to implement (if turned on) color cluster updates 
	color_classes = None
	neighbor_matrix = None
	coupling_matrix = None
	if use_color_updates:
		lattice.check_interaction_coloring()
		color_classes = lattice.interaction_color_classes()
		max_neighbors = max(len(partners) for partners in lattice.partners)
		neighbor_matrix = np.zeros((nspins,max_neighbors),dtype=int)
		coupling_matrix = np.zeros((nspins,max_neighbors),dtype=float)

		for site in lattice.sites:
			partners = np.asarray(lattice.partners[site],dtype=int)
			neighbor_matrix[site,:len(partners)] = partners
			coupling_matrix[site,:len(partners)] = lattice.J_matrix[partners,site]

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

	### Implements simultaneous updates over an independent color class
	def MCcolorstep(spins,sites):
		if len(sites) == 0:
			return 0.0

		fields = np.sum(coupling_matrix[sites] * spins[neighbor_matrix[sites]],axis=1)
		dE = -2.0 * fields * spins[sites]
		betadE = dE / T

		p_flip = np.empty_like(betadE,dtype=float)
		positive = betadE >= 0
		p_flip[positive] = np.exp(-betadE[positive])/(1.0 + np.exp(-betadE[positive]))
		p_flip[~positive] = 1.0/(1.0 + np.exp(betadE[~positive]))

		flips = rng.uniform(size=len(sites)) < p_flip
		if np.any(flips):
			flip_sites = sites[flips]
			spins[flip_sites] *= -1
			return np.sum(dE[flips])

		return 0.0
	
	### Random initial state for each replica 
	spins = initialize_spins(nspins,1,random=True,seed=initial_seed)
	
	for n in range(nTs):
		T = temperature_schedule[n]
		
		### Just first time step we compute the observables for the entire system
		energies[n,0] = calc_energy(spins,lattice.J_matrix)
		mags[n,0] = calc_mag(spins) 
		neels[n,0] = order_parameter(spins,neel_mask)
		stripes[0,n,0] = order_parameter(spins,stripe_x_mask)
		stripes[1,n,0] = order_parameter(spins,stripe_y_mask)
		if 0 in snapshot_indices:
			snapshots[n,snapshot_indices[0],:] = spins

		### This will be used to derive the q_ea
		
		### We need to chop off the first few time steps (we take first 20% to be safe) 
		chop_size = int(nsweeps//5)   
		nsweeps_chopped = 0
		frozen_moment = np.zeros_like(spins,dtype=float) 
		
		### This logs the local magnetic noise 
		noises[n,:,0] = local_noise_field(spins)
		
		for i in range(1,nsweeps):

			energy_change = 0.

			### Run a sweep over all spins
			if use_color_updates:
				for color in rng.permutation(len(color_classes)): ### !!! Codex check this picks a fixed rng permutation of the color classes per sweep and then iterates through it, not randomizing ever step of the color loop 
					dE = MCcolorstep(spins,color_classes[color])
					energy_change += dE ### !!! To match the implementation used in the single site sweep method 
			else:
				for j in range(nspins):
					dE = MCstep(spins)
					energy_change += dE

			### Update the energy 
			energies[n,i] = energies[n,i-1] + energy_change
			
			### Update the magnetization 
			mags[n,i] = calc_mag(spins) 

			
			neels[n,i] = order_parameter(spins,neel_mask)
			stripes[0,n,i] = order_parameter(spins,stripe_x_mask)
			stripes[1,n,i] = order_parameter(spins,stripe_y_mask)

			### Update the frozen moment if we are past the chop window 
			if i >= chop_size:
				nsweeps_chopped += 1 
				frozen_moment += spins
			

			### Log the local magnetic fields 
			noises[n,:,i] = local_noise_field(spins) 
			
			### If snapshots are desired this loop will capture them 
			if i in snapshot_indices:
				snapshots[n,snapshot_indices[i],:] = spins
			
			### Spins is now updated for the next loop/annealing epoch 
		
		### We now flatten the frozen moment to compute the q_ea order parameter 
		if nsweeps_chopped > 0: 
			frozen_moment = frozen_moment/float(nsweeps_chopped) ### Normalize by number of time steps 
			q_eas[n] = np.mean(frozen_moment**2) ### Average over volume 
		

	return energies, mags, neels, stripes, q_eas, noises, snapshots



### Saves a compact output and is low memory usage during operation 
### Due to current demler_tools restrictions cannot pass arbitrary objects to run method so instead we pass a limited set of parameters and built object on the fly 

def run_sims(save_filename,L,Jnnn,p,J_seed,nsweeps,temps,distances,replica,initial_seed=None,dynamics_seed=None,snapshot_sweeps=None,use_color_updates=False):
	L = int(L)
	J_seed = int(J_seed)

	latt = lm.lattice(L)
	latt.set_seed(J_seed)
	latt.set_nn_J(1.,1.)
	latt.set_nnn_J(Jnnn,p)
	
	nsweeps = int(nsweeps)

	energies, magnetization, neel, stripes, qea, noise, snapshots = anneal_dynamics_lattice(
		latt,
		nsweeps,
		temps,
		distances,
		initial_seed=initial_seed,
		dynamics_seed=dynamics_seed,
		snapshot_sweeps=snapshot_sweeps,
		use_color_updates=use_color_updates,
	)
		
	### Due to large memory of spin configurations we will now compute derived observables to save 
	### 1) Energy vs time 
	### 2) Magnetization vs time 
	### 3) Neel order vs time 
	### 4) Stripe order vs time
	### 5) Edwards-Anderson OP vs time
	### 6) Local noise for different distances vs time
	### 7) Optional sampled spin snapshots
	
	with open(save_filename, 'wb') as out_file:
		pickle.dump((latt,energies,magnetization,neel,stripes,qea,noise,snapshots), out_file) ### We store the compact observables and the lattice
        	
      	
