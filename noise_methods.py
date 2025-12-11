### Jonathan Curtis
### 12/09/25

import numpy as np
import glauber


### Computes local magnetic noise a distance d away given a stochastic realization of the spin configuration 
def calc_local_noise(spins,ds,Lx,Ly):
	### Computes local magnetic noise
	### Assumes spin qubit is a distance d away in the center of the sample 

	### The magnetization in the z direction a distance d away is given in terms of the spins as 
	### B_z(t) = mu_0 mu_B mu_mat/(4 pi) sum_j S_j(t) (2d^2 - R_j^2)/(R_j^2 + d^2)^(3/2) 

	X,Y = np.meshgrid(np.arange(Lx)-Lx//2, np.arange(Ly)-Ly//2,indexing='ij')

	R2 = np.sqrt(X**2 + Y**2) 

	R2 = R2.ravel() 

	kernels = (2.*ds[None,:]**2 - R2[:,None])/( R2[:,None] + ds[None,:]**2)**1.5 ### Has shape [Nspins, Nds] 

	noise = np.tensordot(spins,kernels,axes=[-2,0])
	
	### Now we swap the last two axes so that the time axis is last as always 
	noise = np.swapaxes(noise,-1,-2) 

	return noise 
    
   
   
   
