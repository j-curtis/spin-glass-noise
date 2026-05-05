### Implements efficient Heisenberg dynamics for spins on the specified lattice in presence of stochastic noise 
### Jonathan B. Curtis 

import numpy as np 
import lattice_methods as lattice

class dynamics:

	### Converts an array of shape (2,...) which is the array of rotor angles to a vector representation 
	@staticmethod 
	def angle_to_vector(angles): 
		Sz = np.cos(angles[0,...])
		Sx = np.sin(angles[0,...])*np.cos(angles[1,...]) 
		Sy = np.sin(angles[0,...])*np.sin(angles[1,...])
		
		Svec = np.stack([Sx,Sy,Sz]) 
		
		return Svec 
		
	### Converts an array of shape (3,...) which is a vectir representation to an array of rotor angles
	@staticmethod
	def vector_to_angles(vector):	
		### First we should check it is normalized 
		norm = np.sqrt( vec[0,...]**2 + vec[1,...]**2 + vec[2,...]**2)
		if not (norm == 1.).any():
			raise: Exception("Normalization error has occured.") 
			 
		### We proceed 
		theta = np.arccos(Sz) 
		phi = np.arctan2(Sy,Sx)
		
		angles = np.stack([ theta,phi]) 
		
		return angles
		
	### We need a lattice method to initialize on
	### Also forces a few basic parameters 
	def __init__(self,latt, nburn, sample_every, nsamples, damping, temp_schedule, dt = 1.-e3, anisotropy=0.):
		self.latt = latt ### Add the lattice and its methods to the dynamics instance 
		
		self.angles = np.zeros((2,self.latt.N)) ### This will be the variable where the spins (or rather the angular parameterization thereof) actually resides 
		
		### These will be initialized 
		self.seed = None 
		self.rng = np.random.default_rng() ### It will be set at the outset to the default rng  
			
		self.nburn = nburn 
		self.sample_every = sample_every
		self.nsamples = nsamples
			
		self.damping = damping 
		self.temp_schedule = temp_schedule 
		
		self.dt = dt 
			
		self.anisotropy = anisotropy
		
		
	def set_seed(self,seed):
		self.seed = seed
		self.rng = np.random.default_rng(self.seed) 
		
		
	### This method will implement one step of the time evolution 
	def _single_time_step(self,angles,T):
		### Updates one time step at the prescribed T 
		
		### First we compute the effective curie-weiss field 
		S = angle_to_vector(angles)
		Beff = np.zeros_like(S)  
		Beff[2,...] = 2.*self.anisotropy*S[2,...] ### Add anisotropy term 
		
		### Now we go through the lattice and add each interaction up to range 2
		for i in self.latt.sites:
			for j in self.latt.partners[i]:
				Beff[:,i] += 2.*self.latt.J_matrix[j,i]*S[:,i] 
			
		### Next we add the termal noise 
				
		
