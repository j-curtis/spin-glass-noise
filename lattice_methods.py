### This code will be used to generate various lattices and the relevant interaction constants 
### Jonathan Curtis 

import numpy as np 

### Generates various square lattices 
class lattice:
	
	def __init__(self,L):
		### Generates an LxL square lattice 
		self.symmetry_tol = 1.e-7 ### Default tolerance for checking the coupling matrix is symmetric 

		self.L = L 
		self.Lx = L 
		self.Ly = L 
		self.N = self.Lx*self.Ly
		self.sites = np.arange(self.N) 

		### Useful for computing various masks and kernels
		X, Y = np.meshgrid(np.arange(self.Lx)-self.Lx//2, np.arange(self.Ly)-self.Ly//2,indexing='ij')

		self.X = X.ravel()
		self.Y = Y.ravel() 

		self.R = np.sqrt(self.X**2 + self.Y**2)   
		
		### Generate list of nn and nnn sites 
		self.nns = []
		self.nnns = []
		
		### General list of interaction pairs 
		### !!! In future should make this list built in the setter methods so that it can be made variable size depending on distance of neighbors included 
		self.partners = [] 
		
		self.J_matrix = np.zeros((self.N,self.N))
		
		for i in self.sites:
			r = self.index_to_coordinate(i) 
			
			nn_vectors = [ (1,0), (-1,0), (0,1), (0,-1) ] 
			nnn_vectors = [ (1,1), (1,-1), (-1,1), (-1,-1) ] 
			
			nns_site = []
			nnns_site = [] 
			partners_site = []
			
			for nn in nn_vectors:
				x,y = r 
				rnn = (x+ nn[0],y+nn[1] )
				nns_site.append(  self.coordinate_to_index(rnn) )
				partners_site.append(self.coordinate_to_index(rnn))
				
			for nnn in nnn_vectors:
				x,y = r 
				rnnn = (x+nnn[0], y+nnn[1])
				nnns_site.append( self.coordinate_to_index(rnnn) )
				partners_site.append(self.coordinate_to_index(rnnn))
				
			self.nns.append(nns_site)
			self.nnns.append(nnns_site)
			self.partners.append(partners_site)
			
		self.seed = None 
		self.rng = np.random.default_rng()

	### Flattened index to a coordinate in real space 
	def index_to_coordinate(self,i):
		x = i%self.L 
		y = i//self.L
		
		return (x,y)
	
	### Real space coordinates to a flattened index 
	def coordinate_to_index(self,r):
		x,y = r 
		return x%self.L + (y%self.L)*self.L 
			
	### Sets the seed for the rng 
	def set_seed(self,seed):
		self.seed = seed
		self.rng = np.random.default_rng(self.seed)
		
	### Sets nearest-neighbor couplings with optional random choice between +Jnn and -Jnn
	def set_nn_J(self,Jnn,p=1.):
		self.Jnn = Jnn
		self.pnn = p 
	
		for i in self.sites:
			 
			for j in self.nns[i]:
				if j < i:
					Jij = self.rng.choice([self.Jnn, -self.Jnn],p=[self.pnn,1.-self.pnn]) ### This is the random value
					### To avoid double counting we will only fill if j < i but fill both J[j,i] and J[i,j]
					self.J_matrix[j,i] = Jij
					self.J_matrix[i,j] = Jij

		return self.check_symmetric()
		

	### This will set nearest neighbor coupling to be a Gaussian distribution 
	def set_nn_Gaussian(self,Javg,Jstd):
		self.Jnn_avg = Javg
		self.Jnn_std = Jstd
		
		for i in self.sites:
			for j in self.nns[i]:
				if j < i:
					Jij = self.rng.normal(self.Jnn_avg,self.Jnn_std) 
					self.J_matrix[i,j] = Jij
					self.J_matrix[j,i] = Jij

		return self.check_symmetric()

	### Sets the next-nearest-neighbor couplings with optional choice between value and 0 
	def set_nnn_J(self,Jnnn,p=1.):
		self.Jnnn = Jnnn
		self.pnnn = p 
	
		for i in self.sites:
			for j in self.nnns[i]:
				if j < i:
					Jij = self.rng.choice([self.Jnnn, 0.],p=[self.pnnn,1.-self.pnnn])
					self.J_matrix[j,i] = Jij
					self.J_matrix[i,j] = Jij
							
		return self.check_symmetric()

	### Check that the coupling matrix is symmetric 
	def check_symmetric(self):
		max_asymmetry = np.max( np.abs( self.J_matrix - self.J_matrix.T ) )

		if max_asymmetry > self.symmetry_tol:
			raise ValueError(f"Coupling matrix is not symmetric: max asymmetry = {max_asymmetry}")

		return True
		

	### This method returns a mask for computing the neel order of a set of spins 
	def neel_mask(self):

		neel_mask = (self.X+self.Y).astype(int)

		return (-1.*np.ones(self.N,dtype=int))**neel_mask

	### This method returns a magnetostatic mask for computing spins -> local magnetic field pipeline 
	### Gives zz component of full tensor, relevant for Ising z-spins
	def magnetic_field_mask_zz(self,distances):
		distances = np.atleast_1d(np.asarray(distances,dtype=float))

		kernels = (2.*distances[None,:]**2 - self.R[:,None]**2)/( self.R[:,None]**2 + distances[None,:]**2)**2.5 ### Has shape [Nspins, Nds] 

		return kernels 
		
		
	### This method returns a magnetostatic mask for computing spins -> local magnetic field pipeline 
	### Returns full 3x3 tensor for Heisenberg spins 
	def magnetic_field_mask_tensor(self,distances):
		distances = np.atleast_1d(np.asarray(distances,dtype=float))

		noise_kernel = np.zeros((3,3,self.N,len(distances)))

		Reff = np.sqrt( self.R[:,None]**2 + distances[None,:]**2 )

		### Dipolar tensor T_ab = (3 r_a r_b - delta_ab r^2)/r^5
		rvecs = [self.X[:,None], self.Y[:,None], distances[None,:]]
		for a in range(3):
			for b in range(3):
				noise_kernel[a,b,...] = 3.*rvecs[a]*rvecs[b]/Reff**5
				if a == b:
					noise_kernel[a,b,...] -= 1./Reff**3

		return noise_kernel
		
		
		
	
